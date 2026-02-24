"""
ShockTubeDifferentiator (v2 — Conservative Fluxes + Concat Fusion)
===================================================================
G-PARCv2 differentiator for compressible Euler equations (shock tube).

Physics — Conservative Flux Form:
  The compressible Euler equations in conservative form:
    ∂ρ/∂t   = −∂F_ρ/∂x     where F_ρ   = ρu
    ∂(ρu)/∂t = −∂F_ρu/∂x   where F_ρu  = ρu² + p
    ∂E/∂t   = −∂F_E/∂x     where F_E   = (E + p)u

  Pressure from ideal gas EOS: p = (γ−1)(E − ½ρu²)

  MLS computes ∂F/∂x for each equation — the actual physics the solver
  computed, not the advective approximation (v·∇φ).

Fusion Strategy — Concatenation + MLP (not SPADE):
  Physics features (flux divergence, diffusion) are concatenated with
  learned features and processed by an MLP to predict dφ/dt per variable.
  This lets the network learn how to weight physics vs learned features
  rather than forcing modulation through SPADE.

  This follows MeshGraphKAN's philosophy (flat input, learned combination)
  but with explicit physics features the network doesn't have to discover
  through message passing.

Architecture (per-variable):
  1. Feature extraction on static geometry → learned features [n_fe_features]
  2. FiLM: condition learned features on [pressure, density, dt]
  3. Conservative flux divergence (MLS ∂F/∂x) + Diffusion (FD ∇²φ) → physics
  4. LayerNorm on physics features → O(1) scale
  5. Concat [learned | physics] → MLP → dφ_i/dt

Dynamic features (after skipping y_momentum at raw index 2):
  [0] density (ρ), [1] x_momentum (ρu), [2] total_energy (E)

Call signature: (state, edge_index) → dφ/dt
  where state = [static_feats | global_context | dynamic_feats]
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


class ShockTubeDifferentiator(nn.Module):
    """
    Shock Tube Differentiator with conservative Euler flux divergences + FiLM.
    
    Physics features (flux divergence, diffusion) are concatenated with
    FiLM-conditioned learned features and processed by per-variable MLPs.
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
        list_dif_idx: list = None,
        diffusion_type: str = 'fd',
        fusion_hidden_dim: int = 128,
        zero_init: bool = True,
        gamma: float = 1.4,
        # Legacy args (ignored, kept for compatibility with training scripts)
        list_adv_idx: list = None,
        velocity_indices: list = None,
        spade_random_noise: bool = False,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.n_fe_features = n_fe_features
        self.global_embed_dim = global_embed_dim
        self.global_param_dim = global_param_dim
        self.diffusion_type = diffusion_type.lower()
        self.gamma = gamma
        
        # Legacy attrs (kept for checkpoint compat)
        self.velocity_indices = velocity_indices if velocity_indices is not None else [1]
        
        # Feature extractor
        self.feature_extractor = feature_extractor
        
        # MLS gradient solver (for flux divergences)
        self.gradient_solver = gradient_solver
        self.laplacian_solver = laplacian_solver
        
        # Diffusion operators
        if self.diffusion_type == 'none':
            list_dif_idx = []
        elif list_dif_idx is None:
            list_dif_idx = list(range(num_dynamic_feats))
        self.list_dif_idx = list_dif_idx
        
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
            n_physics = 1  # flux divergence always present
            if self.list_dif[i] is not None:
                n_physics += 1
            
            self.list_phys_norm.append(nn.LayerNorm(n_physics))
            self.list_fusion.append(PhysicsFusionMLP(
                n_learned=n_fe_features,
                n_physics=n_physics,
                hidden_dim=fusion_hidden_dim,
                zero_init=zero_init,
            ))
        
        # --- FiLM: Condition learned features on raw global params ---
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
            
            pos = sample_data.pos if hasattr(sample_data, 'pos') and sample_data.pos is not None \
                  else sample_data.x[:, :2]
            
            dummy_u = torch.zeros(sample_data.num_nodes, 1, device=pos.device)
            self.gradient_solver.solve_single_variable(pos, sample_data.edge_index, dummy_u)
            self.laplacian_solver(sample_data)
            
            self._weights_initialized = True
            print("  ✅ MLS weights initialized")

    def _compute_euler_fluxes(self, density, x_momentum, total_energy):
        """
        Compute conservative Euler fluxes from conserved variables.
        
        Args:
            density:      [N, 1] ρ
            x_momentum:   [N, 1] ρu
            total_energy: [N, 1] E
            
        Returns:
            F_rho:  [N, 1] flux for density equation = ρu
            F_mom:  [N, 1] flux for momentum equation = ρu² + p
            F_eng:  [N, 1] flux for energy equation = (E + p)u
        """
        safe_density = torch.clamp(density.abs(), min=1e-6)
        u = x_momentum / safe_density
        
        # Pressure from ideal gas EOS: p = (γ-1)(E - ½ρu²)
        kinetic = 0.5 * x_momentum * u
        pressure = (self.gamma - 1.0) * (total_energy - kinetic)
        
        F_rho = x_momentum                          # ρu
        F_mom = x_momentum * u + pressure           # ρu² + p
        F_eng = (total_energy + pressure) * u       # (E + p)u
        
        return F_rho, F_mom, F_eng

    def forward(self, state, edge_index):
        """
        Compute dφ/dt for shock tube dynamics.
        
        state = [pos(2) | global_embed(64) | raw_global(3) | dynamic(3)] = [N, 72]
        """
        if not self._weights_initialized:
            raise RuntimeError("initialize_weights() must be called before forward()")
        
        sf = self.num_static_feats
        ge = self.global_embed_dim
        gp = self.global_param_dim
        
        # Parse state
        static_feats   = state[:, :sf]
        global_embed   = state[:, sf:sf + ge]
        raw_global     = state[:, sf + ge:sf + ge + gp]
        dynamic_feats  = state[:, sf + ge + gp:]
        
        global_attrs = raw_global[0]  # [3]
        
        # --- 1. Learned Features (state-dependent) + FiLM ---
        # Feature extractor sees [positions | dynamic_state] so the 128-dim
        # learned representation updates every timestep as the flow evolves.
        # This is critical for shocktube: the static mesh embedding alone
        # doesn't provide enough information about the current flow state.
        fe_input = torch.cat([static_feats, dynamic_feats], dim=-1)
        learned_features = self.feature_extractor(
            fe_input, edge_index, pos=static_feats
        )
        learned_features = self.feature_norm(learned_features, global_attrs)
        
        # --- 2. FiLM on dynamic state ---
        dynamic_conditioned = self.derivative_norm(dynamic_feats, global_attrs)
        
        # --- 3. Build mesh_data for MLS ---
        mesh_data = Data(pos=static_feats, edge_index=edge_index)
        mesh_data.num_nodes = state.shape[0]
        
        # --- 4. Conservative Euler Fluxes ---
        density      = dynamic_feats[:, 0:1]
        x_momentum   = dynamic_feats[:, 1:2]
        total_energy = dynamic_feats[:, 2:3]
        
        F_rho, F_mom, F_eng = self._compute_euler_fluxes(
            density, x_momentum, total_energy
        )
        
        # MLS gradient of all fluxes: list of 3 x [N, 2]
        flux_stack = torch.cat([F_rho, F_mom, F_eng], dim=1)
        flux_grads = self.gradient_solver(mesh_data, flux_stack)
        
        # Extract ∂F/∂x only (quasi-1D)
        dFdx = [fg[:, 0:1] for fg in flux_grads]
        
        # --- 5. Concat fusion per variable ---
        t_dot_parts = []
        
        for i in range(self.num_dynamic_feats):
            phys_feats = [dFdx[i]]
            
            if self.list_dif[i] is not None:
                dif = self.list_dif[i](dynamic_feats[:, i:i+1], mesh_data)
                phys_feats.append(dif)
            
            # Normalize physics features to O(1)
            phys_cat = torch.cat(phys_feats, dim=1)
            phys_cat = self.list_phys_norm[i](phys_cat)
            
            # Concat learned + physics → MLP → dφ_i/dt
            out = self.list_fusion[i](learned_features, phys_cat)
            t_dot_parts.append(out)
        
        return torch.cat(t_dot_parts, dim=1)