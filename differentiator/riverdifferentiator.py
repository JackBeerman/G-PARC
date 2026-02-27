"""
RiverDifferentiator (Concat+MLP Fusion)
========================================
G-PARCv2 differentiator for river/fluvial dynamics using MLS operators.

Uses AdvectionMLS (v·∇φ) + DiffusionMLS (∇²φ) with concat+MLP fusion
instead of SPADE (MappingAndRecon).

Fusion Strategy — Concatenation + MLP (not SPADE):
  Physics features (advection, diffusion) are concatenated with learned
  features and processed by per-variable MLPs to predict dφ/dt.
  This lets the network learn how to weight physics vs learned features
  rather than forcing modulation through SPADE.

  Motivation: SPADE modulates 128-dim features with only 2 physics channels.
  Most of the 128 dims pass through unchanged. Concat+MLP gives the network
  full nonlinear access to combine all features. This is the same fix that
  gave 100x rollout stability improvement on shocktube.

Architecture (per-variable):
  1. Feature extraction on [static + dynamic] → learned features [n_fe_features]
  2. AdvectionMLS (v·∇φ) + DiffusionMLS (∇²φ) → physics features
  3. LayerNorm on physics features → O(1) scale
  4. Concat [learned | physics] → MLP → dφ_i/dt

Dynamic features:
  [0] Water Depth, [1] Volume (scalars → individual MLPs)
  [2] Velocity X, [3] Velocity Y (vector → shared MLP, output_dim=2)

Call signature: (state, edge_index) → dφ/dt
  where state = [static_feats | dynamic_feats] concatenated

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


class PhysicsFusionMLP(nn.Module):
    """
    Per-variable MLP that fuses learned features with physics features.
    
    Input:  [learned_features (n_fe) | physics_features (n_phys)]
    Output: dφ_i/dt (output_dim)
    
    Two hidden layers with GELU activation and residual connection.
    Optional zero-initialized final layer for stable training start.
    """
    def __init__(self, n_learned, n_physics, output_dim=1, 
                 hidden_dim=128, zero_init=False):
        super().__init__()
        input_dim = n_learned + n_physics
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden_dim, output_dim)
        
        # Residual projection if dimensions differ
        self.residual = (
            nn.Linear(input_dim, hidden_dim) 
            if input_dim != hidden_dim 
            else nn.Identity()
        )
        
        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
    
    def forward(self, learned_features, physics_features):
        x = torch.cat([learned_features, physics_features], dim=-1)
        h = self.net(x) + self.residual(x)
        return self.out(h)


class RiverDifferentiator(nn.Module):
    """
    River Differentiator with Concat+MLP fusion.
    
    Replaces SPADE (MappingAndRecon) with PhysicsFusionMLP per variable.
    
    Structure:
      1. Scalars (Depth, Volume) → Individual PhysicsFusionMLP blocks
      2. Vectors (Velocity X, Y) → Shared PhysicsFusionMLP block (output_dim=2)
      
    Call signature: (state, edge_index) → dφ/dt
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
        fusion_hidden_dim: int = 128,
        zero_init: bool = False,
        # Legacy SPADE args (ignored, kept for training script compatibility)
        spade_random_noise: bool = False,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.n_fe_features = n_fe_features
        
        # Velocity X/Y indices (vector block)
        self.velocity_indices = velocity_indices if velocity_indices is not None else [2, 3]
        
        # Identify Scalars (everything not in velocity_indices)
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

        # --- Physics feature normalization + Fusion MLPs ---
        # Structure: [scalar_0_norm, scalar_0_fusion, scalar_1_norm, scalar_1_fusion, ..., vec_norm, vec_fusion]
        
        self.scalar_phys_norms = nn.ModuleList()
        self.scalar_fusions = nn.ModuleList()
        
        for i in self.scalar_indices:
            n_physics = 0
            if self.list_adv[i] is not None: n_physics += 1
            if self.list_dif[i] is not None: n_physics += 1
            
            if n_physics > 0:
                self.scalar_phys_norms.append(nn.LayerNorm(n_physics))
                self.scalar_fusions.append(PhysicsFusionMLP(
                    n_learned=n_fe_features,
                    n_physics=n_physics,
                    output_dim=1,
                    hidden_dim=fusion_hidden_dim,
                    zero_init=zero_init,
                ))
            else:
                self.scalar_phys_norms.append(None)
                self.scalar_fusions.append(None)
        
        # Vector block (velocity components share one fusion MLP)
        n_physics_vec = 0
        for i in self.velocity_indices:
            if self.list_adv[i] is not None: n_physics_vec += 1
            if self.list_dif[i] is not None: n_physics_vec += 1
        
        if n_physics_vec > 0:
            self.vec_phys_norm = nn.LayerNorm(n_physics_vec)
            self.vec_fusion = PhysicsFusionMLP(
                n_learned=n_fe_features,
                n_physics=n_physics_vec,
                output_dim=len(self.velocity_indices),
                hidden_dim=fusion_hidden_dim,
                zero_init=zero_init,
            )
        else:
            self.vec_phys_norm = None
            self.vec_fusion = None

        self._weights_initialized = False
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
        """
        mesh_data = Data(
            pos=pos,
            edge_index=edge_index,
        )
        mesh_data.num_nodes = pos.shape[0]
        
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
        
        # --- 2. Learned Features (Static + Dynamic) ---
        fe_input = torch.cat([static_feats, dynamic_feats], dim=-1)
        learned_features = self.feature_extractor(fe_input, edge_index, pos=static_feats[:, :2])

        # --- 3. Build mesh_data for MLS ---
        mesh_data = self._build_mesh_data(static_feats[:, :2], edge_index)
            
        # Velocity field for advection [N, 2]
        velocity = dynamic_feats[:, self.velocity_indices]

        t_dot_parts = []
        
        # --- A. Process Scalars (Individual Fusion MLPs) ---
        for idx, i in enumerate(self.scalar_indices):
            fusion = self.scalar_fusions[idx]
            
            if fusion is not None:
                phys_feats = []
                
                if self.list_adv[i] is not None:
                    adv = self.list_adv[i](dynamic_feats[:, i:i+1], velocity, mesh_data)
                    phys_feats.append(adv)
                    
                if self.list_dif[i] is not None:
                    dif = self.list_dif[i](dynamic_feats[:, i:i+1], mesh_data)
                    phys_feats.append(dif)
                
                phys_cat = torch.cat(phys_feats, dim=1)
                phys_cat = self.scalar_phys_norms[idx](phys_cat)
                
                out = fusion(learned_features, phys_cat)
                t_dot_parts.append(out)
            else:
                t_dot_parts.append(torch.zeros(
                    state.shape[0], 1, device=state.device
                ))

        # --- B. Process Vector/Velocity (Shared Fusion MLP) ---
        if self.vec_fusion is not None:
            phys_feats_vec = []
            
            for i in self.velocity_indices:
                if self.list_adv[i] is not None:
                    phys_feats_vec.append(
                        self.list_adv[i](dynamic_feats[:, i:i+1], velocity, mesh_data)
                    )
                if self.list_dif[i] is not None:
                    phys_feats_vec.append(
                        self.list_dif[i](dynamic_feats[:, i:i+1], mesh_data)
                    )
            
            phys_cat_vec = torch.cat(phys_feats_vec, dim=1)
            phys_cat_vec = self.vec_phys_norm(phys_cat_vec)
            
            out_vec = self.vec_fusion(learned_features, phys_cat_vec)
            t_dot_parts.append(out_vec)
        else:
            t_dot_parts.append(torch.zeros(
                state.shape[0], len(self.velocity_indices), device=state.device
            ))
            
        # --- 4. Combine ---
        return torch.cat(t_dot_parts, dim=1)