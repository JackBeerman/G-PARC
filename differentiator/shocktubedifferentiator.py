"""
ShockTubeDifferentiator (v2 — FiLM on learned features only)
=============================================================
G-PARCv2 differentiator for compressible Euler equations (shock tube).

Physics:
  - Advection: v·∇φ for conserved quantities transported by flow
  - Diffusion: ∇²φ for numerical viscosity / shock stabilization
  - FiLM conditioning on LEARNED FEATURES from global params

CRITICAL DESIGN CHOICE: Physics operators (advection, diffusion) use RAW
dynamic features, NOT FiLM-conditioned features. Computing velocity as
FiLM'd_momentum / FiLM'd_density produces physically meaningless values
because the affine transform destroys the density/momentum relationship.
This matches the river differentiator and v1's approach where FiLM
conditioned the learned representation, not the physics.

Dynamic features (after skipping y_momentum at raw index 2):
  [0] density, [1] x_momentum, [2] total_energy

Call signature: (state, edge_index) → dφ/dt
  where state = [static_feats | global_context | dynamic_feats]
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .hop import (
    SolveGradientsLST,
    SolveWeightLST2d,
    AdvectionMLS,
    DiffusionMLS,
    DiffusionFD,
)
from .mappingandrecon import MappingAndRecon

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utilities.embed import SimulationConditionedLayerNorm


class ShockTubeDifferentiator(nn.Module):
    """
    Shock Tube Differentiator with MLS physics + FiLM.
    
    FiLM conditioning (on learned features only):
      1. SimulationConditionedLayerNorm on learned features  [raw 3-vector]
      2. Global embed (64-dim) concatenated into augmented static for context
    
    IMPORTANT: Physics operators (advection, diffusion) use RAW dynamic
    features to preserve physical meaning. FiLM does NOT touch dynamic
    state — this matches the river differentiator and v1's pattern where
    FiLM conditioned the representation, not the physics.
    
    Architecture (per-variable MARs):
      1. Feature extraction on static geometry → learned features [n_fe_features]
      2. FiLM: condition learned features on [pressure, density, dt]
      3. AdvectionMLS + DiffusionFD on RAW dynamic variables → physics features
      4. MappingAndRecon (SPADE) fuses FiLM'd learned + raw physics → dφ/dt
    """
    
    def __init__(
        self,
        num_static_feats: int,
        num_dynamic_feats: int,
        feature_extractor: nn.Module,
        gradient_solver: SolveGradientsLST,
        laplacian_solver: SolveWeightLST2d = None,
        n_fe_features: int = 128,
        global_embed_dim: int = 64,
        global_param_dim: int = 3,
        list_adv_idx: list = None,
        list_dif_idx: list = None,
        velocity_indices: list = None,
        diffusion_type: str = 'mls',
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
        self.global_param_dim = global_param_dim
        self.diffusion_type = diffusion_type.lower()
        
        # Velocity indices within the USED dynamic features
        self.velocity_indices = velocity_indices if velocity_indices is not None else [1]
        
        # Feature extractor
        self.feature_extractor = feature_extractor
        
        # MLS solvers
        self.gradient_solver = gradient_solver
        self.laplacian_solver = laplacian_solver
        
        # Default: physics on all dynamic features
        if list_adv_idx is None:
            list_adv_idx = list(range(num_dynamic_feats))
        if self.diffusion_type == 'none':
            list_dif_idx = []
        elif list_dif_idx is None:
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
                if self.diffusion_type == 'fd':
                    self.list_dif.append(DiffusionFD())
                elif self.diffusion_type == 'mls':
                    assert laplacian_solver is not None, "laplacian_solver required for diffusion_type='mls'"
                    self.list_dif.append(DiffusionMLS(laplacian_solver))
                else:
                    self.list_dif.append(None)
            else:
                self.list_dif.append(None)

        # --- MappingAndRecon (SPADE) per variable ---
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
        
        # --- FiLM: Condition learned features on raw global params ---
        # SimulationConditionedLayerNorm: LayerNorm + gamma/beta from [pressure, density, dt]
        # NOTE: Only learned features are FiLM'd, NOT the dynamic state.
        # Physics operators (advection, diffusion) need raw dynamic features
        # to preserve physical meaning (e.g., velocity = momentum/density).
        self.feature_norm = SimulationConditionedLayerNorm(
            normalized_shape=n_fe_features,
            global_dim=global_param_dim,
        )
        
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
          static_feats_augmented = [pos (2) | global_embed (64) | raw_global (3)]
          dynamic_state = [density, x_momentum, total_energy] (3)
        
        So state is [N, 2 + 64 + 3 + 3] = [N, 72]
        
        IMPORTANT: Physics operators (advection, diffusion) use RAW dynamic
        features to preserve physical meaning. FiLM only conditions the learned
        feature representation, matching the river differentiator pattern and
        v1's approach where FiLM affected the representation, not the physics.
        """
        if not self._weights_initialized:
            raise RuntimeError("initialize_weights() must be called before forward()")
        
        sf = self.num_static_feats        # 2
        ge = self.global_embed_dim         # 64
        gp = self.global_param_dim         # 3
        
        # Parse the concatenated state tensor:
        #   [0:2]     = positions
        #   [2:66]    = global_embed (64)
        #   [66:69]   = raw_global_attrs (3) [pressure, density, dt]
        #   [69:72]   = dynamic_feats (3)
        static_feats   = state[:, :sf]                          # [N, 2]
        global_embed   = state[:, sf:sf + ge]                   # [N, 64]
        raw_global     = state[:, sf + ge:sf + ge + gp]         # [N, 3]
        dynamic_feats  = state[:, sf + ge + gp:]                # [N, 3]
        
        # Raw global attrs for SimulationConditionedLayerNorm (single vector)
        global_attrs = raw_global[0]  # [3] — same for all nodes
        
        # --- 1. Learned Features + FiLM ---
        # FiLM conditions the learned representation, NOT the physics
        learned_features = self.feature_extractor(
            static_feats, edge_index, pos=static_feats
        )
        learned_features = self.feature_norm(learned_features, global_attrs)
        
        # --- 2. Build mesh_data for MLS ---
        mesh_data = Data(pos=static_feats, edge_index=edge_index)
        mesh_data.num_nodes = state.shape[0]
        
        # --- 3. Velocity from RAW dynamic features (physical meaning preserved) ---
        density_raw = dynamic_feats[:, 0:1]
        x_mom_raw = dynamic_feats[:, self.velocity_indices[0]:self.velocity_indices[0] + 1]
        safe_density = torch.clamp(density_raw.abs(), min=1e-6)
        v_x = x_mom_raw / safe_density
        velocity = torch.cat([v_x, torch.zeros_like(v_x)], dim=1)  # [N, 2]
        
        # --- 4. Physics (on RAW dynamic) + SPADE per variable ---
        t_dot_parts = []
        
        for i in range(self.num_dynamic_feats):
            mar_block = self.list_mar[i]
            
            if mar_block is not None:
                phys_feats = []
                
                # Advection and diffusion operate on RAW dynamic features
                if self.list_adv[i] is not None:
                    adv = self.list_adv[i](dynamic_feats[:, i:i+1], velocity, mesh_data)
                    phys_feats.append(adv)
                
                if self.list_dif[i] is not None:
                    dif = self.list_dif[i](dynamic_feats[:, i:i+1], mesh_data)
                    phys_feats.append(dif)
                
                # SPADE fuses FiLM'd learned features with raw physics features
                out = mar_block(
                    learned_features,
                    torch.cat(phys_feats, dim=1),
                    edge_index,
                )
                t_dot_parts.append(out)
            else:
                t_dot_parts.append(torch.zeros(state.shape[0], 1, device=state.device))
        
        return torch.cat(t_dot_parts, dim=1)  # [N, num_dynamic_feats]