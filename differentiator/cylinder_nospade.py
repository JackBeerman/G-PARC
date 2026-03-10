"""
CylinderDifferentiator (v2 — Advection-Diffusion + Concat Fusion, No SPADE)
============================================================================
G-PARCv2 differentiator for incompressible Karman vortex (cylinder flow).

Physics — Incompressible Navier-Stokes (advection-diffusion form):
  ∂φ/∂t = −(v · ∇φ) + ν∇²φ + source terms

  For each transported variable φ_i ∈ {p, vx, vy, vz, ωx, ωy, ωz}:
    - Advection: v · ∇φ_i  computed via MLS gradients + dot with velocity
    - Diffusion: ∇²φ_i     computed via MLS Laplacian (or FD)

  Velocity field for advection uses [velocity_x, velocity_y] from the
  dynamic state (2D flow — z-component ignored for transport).

Fusion Strategy — Concatenation + MLP (not SPADE):
  Physics features (advection, diffusion) are concatenated with
  learned features and processed by an MLP to predict dφ/dt per variable.
  Same approach as ShockTubeDifferentiator.

Architecture (per-variable):
  1. Feature extraction on [static + dynamic] → learned features [n_fe_features]
  2. FiLM: condition learned features on [Reynolds number]
  3. Advection (MLS v·∇φ) + Diffusion (∇²φ) → physics features
  4. LayerNorm on physics features → O(1) scale
  5. Concat [learned | physics] → MLP → dφ_i/dt

Dynamic features (7 total, user may skip some via skip_dynamic_indices):
  [0] pressure (p)
  [1] velocity_x (vx)
  [2] velocity_y (vy)
  [3] velocity_z (vz)       — often ~0 for 2D flow, candidate for skipping
  [4] vorticity_x (ωx)      — often ~0 for 2D flow, candidate for skipping
  [5] vorticity_y (ωy)      — often ~0 for 2D flow, candidate for skipping
  [6] vorticity_z (ωz)

Call signature: (state, edge_index) → dφ/dt
  where state = [static_feats | global_embed | raw_global | dynamic_feats]
  
  With default params:
    static_feats:  [N, 3]   (x, y, z positions)
    global_embed:  [N, 64]  (processed Reynolds number)
    raw_global:    [N, 1]   (raw Reynolds number)
    dynamic_feats: [N, D]   (D = num_dynamic_feats after skipping)
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .hop import (
    SolveGradientsLST,
    SolveWeightLST2d,
    DiffusionMLS,
    DiffusionFD,
)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utilities.embed import SimulationConditionedLayerNorm


class PhysicsFusionMLP(nn.Module):
    """
    Per-variable MLP that fuses learned features with physics features.
    
    Input:  [learned_features (n_fe) | physics_features (n_phys)]
    Output: dφ_i/dt (1)
    
    Two hidden layers with GELU activation and residual connection.
    Zero-initialized final layer for stable training start.
    """
    def __init__(self, n_learned, n_physics, hidden_dim=128, zero_init=True):
        super().__init__()
        input_dim = n_learned + n_physics
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden_dim, 1)
        
        # Residual projection if dimensions differ
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
    
    def forward(self, learned_features, physics_features):
        x = torch.cat([learned_features, physics_features], dim=-1)
        h = self.net(x) + self.residual(x)
        return self.out(h)


class CylinderDifferentiator(nn.Module):
    """
    Cylinder Flow Differentiator with advection-diffusion physics + FiLM.
    
    Physics features (advection, diffusion) are concatenated with
    FiLM-conditioned learned features and processed by per-variable MLPs.
    
    MLS operators work on 2D positions (x, y) — the first 2 columns of
    the 3D position vector.
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
        global_param_dim: int = 1,          # Reynolds number only
        velocity_indices: list = None,       # Which dynamic features are velocity [vx, vy]
        list_adv_idx: list = None,
        list_dif_idx: list = None,
        diffusion_type: str = 'fd',
        fusion_hidden_dim: int = 128,
        zero_init: bool = True,
        pos_dims: int = 2,                   # Use first 2 dims of 3D pos for MLS
    ):
        super().__init__()
        
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.n_fe_features = n_fe_features
        self.global_embed_dim = global_embed_dim
        self.global_param_dim = global_param_dim
        self.diffusion_type = diffusion_type.lower()
        self.pos_dims = pos_dims
        
        # Velocity indices for advection: default [0, 1] in post-skip dynamic space
        # For full 7-feature case: velocity_x=1, velocity_y=2 in raw space
        # But these need to be specified in POST-SKIP index space
        self.velocity_indices = velocity_indices if velocity_indices is not None else [1, 2]
        
        # Feature extractor
        self.feature_extractor = feature_extractor
        
        # MLS gradient solver (for advection gradients)
        self.gradient_solver = gradient_solver
        self.laplacian_solver = laplacian_solver
        
        # Physics operator setup
        # Default: advection and diffusion on all dynamic features
        if list_adv_idx is None:
            list_adv_idx = list(range(num_dynamic_feats))
        if self.diffusion_type == 'none':
            list_dif_idx = []
        elif list_dif_idx is None:
            list_dif_idx = list(range(num_dynamic_feats))
            
        self.list_adv_idx = list_adv_idx
        self.list_dif_idx = list_dif_idx
        
        # Diffusion operators (one per variable)
        self.list_dif = nn.ModuleList()
        for i in range(num_dynamic_feats):
            if i in list_dif_idx:
                if self.diffusion_type == 'fd':
                    self.list_dif.append(DiffusionFD())
                elif self.diffusion_type == 'mls':
                    assert laplacian_solver is not None
                    self.list_dif.append(DiffusionMLS(laplacian_solver))
                else:
                    self.list_dif.append(None)
            else:
                self.list_dif.append(None)
        
        # --- Physics feature normalization ---
        self.list_phys_norm = nn.ModuleList()
        
        # --- Per-variable fusion MLPs ---
        self.list_fusion = nn.ModuleList()
        
        for i in range(num_dynamic_feats):
            n_physics = 0
            if i in list_adv_idx:
                n_physics += 1  # advection term (v · ∇φ_i)
            if self.list_dif[i] is not None:
                n_physics += 1  # diffusion term (∇²φ_i)
            
            # Ensure at least 1 physics feature (fallback: just learned features)
            n_physics = max(n_physics, 1)
            
            self.list_phys_norm.append(nn.LayerNorm(n_physics))
            self.list_fusion.append(PhysicsFusionMLP(
                n_learned=n_fe_features,
                n_physics=n_physics,
                hidden_dim=fusion_hidden_dim,
                zero_init=zero_init,
            ))
        
        # --- FiLM: Condition learned features on raw global params (Reynolds) ---
        self.feature_norm = SimulationConditionedLayerNorm(
            normalized_shape=n_fe_features,
            global_dim=global_param_dim,
        )
        
        # --- FiLM on dynamic state ---
        self.derivative_norm = SimulationConditionedLayerNorm(
            normalized_shape=num_dynamic_feats,
            global_dim=global_param_dim,
        )
        
        self._weights_initialized = False
    
    def initialize_weights(self, sample_data):
        """Initialize MLS operator weights. Call ONCE before training."""
        if not self._weights_initialized:
            print("Initializing MLS operator weights...")
            
            # Use 2D positions for MLS
            if hasattr(sample_data, 'pos') and sample_data.pos is not None:
                pos = sample_data.pos[:, :self.pos_dims]
            else:
                pos = sample_data.x[:, :self.pos_dims]
            
            dummy_u = torch.zeros(sample_data.num_nodes, 1, device=pos.device)
            self.gradient_solver.solve_single_variable(pos, sample_data.edge_index, dummy_u)
            self.laplacian_solver(sample_data)
            
            self._weights_initialized = True
            print("  ✅ MLS weights initialized")

    def forward(self, state, edge_index):
        """
        Compute dφ/dt for cylinder flow dynamics.
        
        state = [pos(3) | global_embed(64) | raw_global(1) | dynamic(D)] = [N, 3+64+1+D]
        """
        if not self._weights_initialized:
            raise RuntimeError("initialize_weights() must be called before forward()")
        
        sf = self.num_static_feats
        ge = self.global_embed_dim
        gp = self.global_param_dim
        
        # Parse state
        static_feats   = state[:, :sf]                          # [N, 3] (x,y,z)
        global_embed   = state[:, sf:sf + ge]                   # [N, 64]
        raw_global     = state[:, sf + ge:sf + ge + gp]         # [N, 1]
        dynamic_feats  = state[:, sf + ge + gp:]                # [N, D]
        
        global_attrs = raw_global[0]  # [1] (Reynolds number)
        
        # 2D positions for MLS operators
        pos_2d = static_feats[:, :self.pos_dims]  # [N, 2]
        
        # --- 1. Learned Features (state-dependent) + FiLM ---
        fe_input = torch.cat([static_feats, dynamic_feats], dim=-1)
        learned_features = self.feature_extractor(
            fe_input, edge_index, pos=pos_2d
        )
        learned_features = self.feature_norm(learned_features, global_attrs)
        
        # --- 2. FiLM on dynamic state ---
        dynamic_conditioned = self.derivative_norm(dynamic_feats, global_attrs)
        
        # --- 3. Build mesh_data for MLS (2D positions, detached for numpy ops) ---
        mesh_data = Data(pos=pos_2d.detach(), edge_index=edge_index)
        mesh_data.num_nodes = state.shape[0]
        
        # Propagate mesh_id for MLS cache invalidation across different meshes
        if hasattr(edge_index, 'mesh_id') and edge_index.mesh_id is not None:
            mesh_data.mesh_id = edge_index.mesh_id
        
        # --- 4. Compute physics features ---
        # Velocity field for advection: [vx, vy] from dynamic features
        velocity = dynamic_feats[:, self.velocity_indices]  # [N, 2]
        
        # Compute gradients for advected variables via MLS
        # We compute gradients for ALL advected variables at once
        advected_vars = []
        advected_idx_map = {}
        for i in self.list_adv_idx:
            advected_idx_map[i] = len(advected_vars)
            advected_vars.append(dynamic_feats[:, i:i+1])
        
        if len(advected_vars) > 0:
            phi_stack = torch.cat(advected_vars, dim=1)  # [N, n_advected]
            phi_grads = self.gradient_solver(mesh_data, phi_stack)  # list of [N, 2]
            
            # Compute advection: v · ∇φ_i for each variable
            advection_terms = {}
            for i in self.list_adv_idx:
                grad_i = phi_grads[advected_idx_map[i]]  # [N, 2]
                # v · ∇φ = vx * ∂φ/∂x + vy * ∂φ/∂y
                adv_i = (velocity * grad_i).sum(dim=1, keepdim=True)  # [N, 1]
                advection_terms[i] = adv_i
        
        # --- 5. Concat fusion per variable ---
        t_dot_parts = []
        
        for i in range(self.num_dynamic_feats):
            phys_feats = []
            
            # Advection term
            if i in self.list_adv_idx and i in advection_terms:
                phys_feats.append(advection_terms[i])
            
            # Diffusion term
            if self.list_dif[i] is not None:
                dif = self.list_dif[i](dynamic_feats[:, i:i+1], mesh_data)
                phys_feats.append(dif)
            
            # Fallback: if no physics features, use zeros
            if len(phys_feats) == 0:
                phys_feats.append(torch.zeros(
                    state.shape[0], 1, device=state.device
                ))
            
            # Normalize physics features to O(1)
            phys_cat = torch.cat(phys_feats, dim=1)
            phys_cat = self.list_phys_norm[i](phys_cat)
            
            # Concat learned + physics → MLP → dφ_i/dt
            out = self.list_fusion[i](learned_features, phys_cat)
            t_dot_parts.append(out)
        
        return torch.cat(t_dot_parts, dim=1)