"""
RiverDifferentiator
===================
G-PARCv2 differentiator for river/fluvial dynamics using MLS operators.

Uses AdvectionMLS (v·∇φ) + DiffusionMLS (∇²φ) instead of 
StrainMLS + DiffusionMLS (elastoplastic).

CRITICAL: The forward/call signature is (state, edge_index) where
  state = torch.cat([static_feats, dynamic_feats], dim=-1)
This matches the interface expected by integrator.numerical (Euler, Heun, RK4)
which calls: derivative_fn(state, edge_index)

FIX (Feb 2026): Robust mesh_id propagation to prevent MLS cache collisions
between White River (mesh_id=0) and Iowa River (mesh_id=1).
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


class RiverDifferentiator(nn.Module):
    """
    River Differentiator with Elastoplastic-style Mapping & Recon.
    
    Architecture matches ElastoPlasticDifferentiator:
      1. Scalars (Depth, Volume) -> Individual MAR blocks
      2. Vectors (Velocity X, Y) -> Shared MAR block
      
    Call signature: (state, edge_index) -> dφ/dt
      where state = [static_feats | dynamic_feats] concatenated
    """
    
    def __init__(
        self,
        num_static_feats: int,
        num_dynamic_feats: int,
        feature_extractor: nn.Module,
        gradient_solver: SolveGradientsLST,
        laplacian_solver: SolveWeightLST2d,
        n_fe_features: int = 128,
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
        
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.n_fe_features = n_fe_features
        
        # Velocity X/Y indices (Treat as Vector block, similar to Displacement in Elasto)
        self.velocity_indices = velocity_indices if velocity_indices is not None else [2, 3]
        
        # Identify Scalars (Everything not in velocity_indices)
        all_indices = set(range(num_dynamic_feats))
        self.scalar_indices = sorted(list(all_indices - set(self.velocity_indices)))
        
        # Feature extractor
        self.feature_extractor = feature_extractor
        
        # MLS solvers
        self.gradient_solver = gradient_solver
        self.laplacian_solver = laplacian_solver
        
        # Default: apply physics to all features
        if list_adv_idx is None:
            list_adv_idx = list(range(num_dynamic_feats))
        if list_dif_idx is None:
            list_dif_idx = list(range(num_dynamic_feats))
            
        self.list_adv_idx = list_adv_idx
        self.list_dif_idx = list_dif_idx
        
        # --- Build Physics Operators ---
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

        # --- Build MappingAndRecon (SPADE) List ---
        # Structure: [MAR_scalar_1, MAR_scalar_2, ..., MAR_vector_block]
        self.list_mar = nn.ModuleList()
        
        # 1. Scalar Variables (Individual MARs)
        for i in self.scalar_indices:
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
                    zero_init=zero_init
                ))
            else:
                self.list_mar.append(None)
                
        # 2. Vector Block (Velocity components share one MAR)
        n_explicit_vec = 0
        for i in self.velocity_indices:
            if self.list_adv[i] is not None: n_explicit_vec += 1
            if self.list_dif[i] is not None: n_explicit_vec += 1
            
        if n_explicit_vec > 0:
            self.list_mar.append(MappingAndRecon(
                n_base_features=n_fe_features,
                n_mask_channel=n_explicit_vec,
                output_channel=len(self.velocity_indices),
                heads=heads,
                concat=concat,
                dropout=dropout,
                add_noise=spade_random_noise,
                zero_init=zero_init
            ))
        else:
            self.list_mar.append(None)

        self._weights_initialized = False
        
        # Track which mesh_id was last seen (for debug)
        self._last_mesh_id = None

    def initialize_weights(self, sample_data):
        """Initialize MLS operator weights with a sample graph. Call ONCE before training."""
        if not self._weights_initialized:
            print("Initializing MLS operator weights...")
            
            dummy_u = torch.zeros(sample_data.num_nodes, 1, 
                                  device=sample_data.pos.device if hasattr(sample_data, 'pos') 
                                  else sample_data.x.device)
            
            self.gradient_solver.solve_single_variable(
                sample_data.pos, sample_data.edge_index, dummy_u
            )
            self.laplacian_solver(sample_data)
            
            self._weights_initialized = True
            print("  ✅ MLS weights initialized")

    def _build_mesh_data(self, pos, edge_index):
        """
        Build a Data object for MLS operators with ROBUST mesh_id propagation.
        
        The mesh_id MUST be set correctly to prevent cache collisions between
        White River (19148 edges) and Iowa River (9868 edges).
        """
        mesh_data = Data(
            pos=pos,
            edge_index=edge_index,
        )
        mesh_data.num_nodes = pos.shape[0]
        
        # Propagate mesh_id from edge_index annotation (set by model.step)
        if hasattr(edge_index, 'mesh_id') and edge_index.mesh_id is not None:
            mesh_data.mesh_id = edge_index.mesh_id
        
        return mesh_data

    def forward(self, state, edge_index):
        """
        Compute dφ/dt for river dynamics.
        
        Args:
            state: [N, num_static + num_dynamic]
            edge_index: [2, E] (may have .mesh_id attribute)
            
        Returns:
            t_dot: [N, num_dynamic_feats]
        """
        if not self._weights_initialized:
            raise RuntimeError("initialize_weights() must be called before forward()")

        # --- 1. Split state ---
        static_feats = state[:, :self.num_static_feats]
        dynamic_feats = state[:, self.num_static_feats:
                               self.num_static_feats + self.num_dynamic_feats]
        
        # --- 2. Learned Features (Static) ---
        learned_features = self.feature_extractor(static_feats, edge_index, pos=static_feats[:, :2])

        # --- 3. Build mesh_data for MLS (with robust mesh_id propagation) ---
        mesh_data = self._build_mesh_data(static_feats[:, :2], edge_index)
            
        # Velocity field for advection [N, 2]
        velocity = dynamic_feats[:, self.velocity_indices]

        t_dot_parts = []
        
        # --- A. Process Scalars (Individual MARs) ---
        for idx, i in enumerate(self.scalar_indices):
            mar_block = self.list_mar[idx]
            
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
                    edge_index
                )
                t_dot_parts.append(out)
            else:
                t_dot_parts.append(torch.zeros(
                    state.shape[0], 1, device=state.device
                ))

        # --- B. Process Vector/Velocity (Shared MAR) ---
        vec_mar_block = self.list_mar[-1]
        
        if vec_mar_block is not None:
            phys_feats_vec = []
            
            for i in self.velocity_indices:
                if self.list_adv[i] is not None:
                    phys_feats_vec.append(self.list_adv[i](dynamic_feats[:, i:i+1], velocity, mesh_data))
                if self.list_dif[i] is not None:
                    phys_feats_vec.append(self.list_dif[i](dynamic_feats[:, i:i+1], mesh_data))
            
            out_vec = vec_mar_block(
                learned_features,
                torch.cat(phys_feats_vec, dim=1),
                edge_index
            )
            t_dot_parts.append(out_vec)
        else:
            t_dot_parts.append(torch.zeros(
                state.shape[0], len(self.velocity_indices), device=state.device
            ))
            
        # --- 4. Combine ---
        return torch.cat(t_dot_parts, dim=1)