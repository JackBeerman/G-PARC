"""
ShockTubeDifferentiator
=======================
G-PARCv2 differentiator for compressible Euler equations (shock tube).

Physics:
  - Advection: v·∇φ for conserved quantities transported by flow
  - Diffusion: ∇²φ for numerical viscosity / shock stabilization
  - Global FiLM conditioning from (pressure, density, delta_t)

The shock tube is physically 1D but set in a 2D domain with (x, y) positions,
so standard 2D MLS operators apply.

Dynamic features (after skipping y_momentum at raw index 2):
  [0] density, [1] x_momentum, [2] total_energy

Call signature: (state, edge_index) → dφ/dt
  where state = [static_feats | dynamic_feats | global_context]

IMPORTANT: The numerical integrator sees:
  static_feats = original_static (2) + global_context (64) = 66
  dynamic_state = 3 (after skip)
  It concatenates them as state = [66 + 3] = [69] and calls derivative_fn(state, edge_index)
  So this differentiator splits: static[:2], dynamic[66:69], global[2:66]
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .hop import (
    SolveGradientsLST,
    SolveWeightLST2d,
    AdvectionMLS,
    DiffusionMLS,
)
from .mappingandrecon import MappingAndRecon


class ShockTubeDifferentiator(nn.Module):
    """
    Shock Tube Differentiator with MLS physics + Global FiLM.
    
    Architecture (per-variable MARs, matching ElastoPlastic pattern):
      1. Feature extraction on static geometry → learned features [n_fe_features]
      2. FiLM modulation with global embedding
      3. AdvectionMLS + DiffusionMLS on each dynamic variable → physics features
      4. MappingAndRecon (SPADE) fuses learned + physics → dφ/dt per variable
    """
    
    def __init__(
        self,
        num_static_feats: int,
        num_dynamic_feats: int,
        feature_extractor: nn.Module,
        gradient_solver: SolveGradientsLST,
        laplacian_solver: SolveWeightLST2d,
        n_fe_features: int = 128,
        global_embed_dim: int = 64,
        list_adv_idx: list = None,
        list_dif_idx: list = None,
        velocity_indices: list = None,
        spade_random_noise: bool = False,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        zero_init: bool = False,
    ):
        super().__init__()
        
        self.num_static_feats = num_static_feats       # 2 (x, y positions)
        self.num_dynamic_feats = num_dynamic_feats     # 3 (after skip)
        self.n_fe_features = n_fe_features
        self.global_embed_dim = global_embed_dim
        
        # Velocity indices within the USED dynamic features
        # After skipping y_momentum, dynamic = [density, x_momentum, total_energy]
        # x_momentum is at index 1 → velocity_indices = [1] for 1D advection in x
        # We pad with zero for y-component in the forward pass
        self.velocity_indices = velocity_indices if velocity_indices is not None else [1]
        
        # Feature extractor
        self.feature_extractor = feature_extractor
        
        # MLS solvers
        self.gradient_solver = gradient_solver
        self.laplacian_solver = laplacian_solver
        
        # Default: physics on all dynamic features
        if list_adv_idx is None:
            list_adv_idx = list(range(num_dynamic_feats))
        if list_dif_idx is None:
            list_dif_idx = list(range(num_dynamic_feats))
            
        self.list_adv_idx = list_adv_idx
        self.list_dif_idx = list_dif_idx
        
        # --- Physics Operators ---
        self.list_adv = nn.ModuleList()
        self.list_dif = nn.ModuleList()
        
        for i in range(num_dynamic_feats):
            if i in list_adv_idx:
                self.list_adv.append(AdvectionMLS(gradient_solver))
            else:
                self.list_adv.append(None)
                
            if i in list_dif_idx:
                self.list_dif.append(DiffusionMLS(laplacian_solver))
            else:
                self.list_dif.append(None)

        # --- MappingAndRecon (SPADE) per variable ---
        # Each dynamic variable gets its own MAR block
        self.list_mar = nn.ModuleList()
        for i in range(num_dynamic_feats):
            n_explicit = 0
            if self.list_adv[i] is not None: n_explicit += 1
            if self.list_dif[i] is not None: n_explicit += 1
            
            if n_explicit > 0:
                self.list_mar.append(MappingAndRecon(
                    n_base_features=n_fe_features,
                    n_mask_channel=n_explicit,
                    output_channel=1,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    add_noise=spade_random_noise,
                    zero_init=zero_init,
                ))
            else:
                self.list_mar.append(None)
        
        # --- Global FiLM ---
        self.global_film_scale = nn.Sequential(
            nn.Linear(global_embed_dim, n_fe_features),
            nn.Tanh()
        )
        self.global_film_bias = nn.Linear(global_embed_dim, n_fe_features)
        
        self._weights_initialized = False
    
    def initialize_weights(self, sample_data):
        """Initialize MLS operator weights. Call ONCE before training."""
        if not self._weights_initialized:
            print("Initializing MLS operator weights...")
            
            pos = sample_data.pos if hasattr(sample_data, 'pos') and sample_data.pos is not None \
                  else sample_data.x[:, :2]
            
            dummy_u = torch.zeros(sample_data.num_nodes, 1, device=pos.device)
            self.gradient_solver.solve_single_variable(pos, sample_data.edge_index, dummy_u)
            self.laplacian_solver(sample_data)
            
            self._weights_initialized = True
            print("  ✅ MLS weights initialized")

    def forward(self, state, edge_index):
        """
        Compute dφ/dt for shock tube dynamics.
        
        The numerical integrator calls this as derivative_fn(state, edge_index) where:
          state = [static_feats_augmented | dynamic_state]
          static_feats_augmented = [original_static (2) | global_context (64)]
          dynamic_state = [density, x_momentum, total_energy] (3)
        
        So state is [N, 2 + 64 + 3] = [N, 69]
        """
        if not self._weights_initialized:
            raise RuntimeError("initialize_weights() must be called before forward()")
        
        sf = self.num_static_feats        # 2
        ge = self.global_embed_dim         # 64
        df = self.num_dynamic_feats        # 3
        
        # The integrator concatenated [static_augmented | dynamic]:
        #   static_augmented = state[:, :sf+ge]  (positions + global)
        #   dynamic          = state[:, sf+ge:]
        static_feats   = state[:, :sf]                # [N, 2]  x, y positions
        global_context = state[:, sf:sf + ge]         # [N, 64] global embed (repeated)
        dynamic_feats  = state[:, sf + ge:sf + ge + df]  # [N, 3]  density, x_mom, energy
        
        # --- 1. Learned Features + Global FiLM ---
        learned_features = self.feature_extractor(
            static_feats, edge_index, pos=static_feats
        )
        
        # FiLM modulation (all nodes share same global params, use first row)
        g = global_context[0:1]  # [1, 64]
        scale = self.global_film_scale(g)  # [1, n_fe_features]
        bias = self.global_film_bias(g)    # [1, n_fe_features]
        learned_features = learned_features * (1.0 + scale) + bias
        
        # --- 2. Build mesh_data for MLS ---
        mesh_data = Data(pos=static_feats, edge_index=edge_index)
        mesh_data.num_nodes = state.shape[0]
        if hasattr(edge_index, 'mesh_id'):
            mesh_data.mesh_id = edge_index.mesh_id
        
        # Velocity for advection: x_momentum → [v_x, 0] in 2D
        v_x = dynamic_feats[:, self.velocity_indices[0]:self.velocity_indices[0] + 1]
        velocity = torch.cat([v_x, torch.zeros_like(v_x)], dim=1)  # [N, 2]
        
        # --- 3. Physics + SPADE per variable ---
        t_dot_parts = []
        
        for i in range(self.num_dynamic_feats):
            mar_block = self.list_mar[i]
            
            if mar_block is not None:
                phys_feats = []
                
                if self.list_adv[i] is not None:
                    adv = self.list_adv[i](dynamic_feats[:, i:i+1], velocity, mesh_data)
                    phys_feats.append(adv)
                
                if self.list_dif[i] is not None:
                    dif = self.list_dif[i](dynamic_feats[:, i:i+1], mesh_data)
                    phys_feats.append(dif)
                
                out = mar_block(
                    learned_features,
                    torch.cat(phys_feats, dim=1),
                    edge_index,
                )
                t_dot_parts.append(out)
            else:
                t_dot_parts.append(torch.zeros(state.shape[0], 1, device=state.device))
        
        return torch.cat(t_dot_parts, dim=1)  # [N, num_dynamic_feats]