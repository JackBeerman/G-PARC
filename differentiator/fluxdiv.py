"""
ShockTubeDifferentiator (v2 — Conservative Euler Fluxes)
=========================================================
G-PARCv2 differentiator for compressible Euler equations (shock tube).

Physics — Conservative Flux Form:
  The compressible Euler equations in conservative form:
    ∂ρ/∂t   = −∂F_ρ/∂x     where F_ρ   = ρu
    ∂(ρu)/∂t = −∂F_ρu/∂x   where F_ρu  = ρu² + p
    ∂E/∂t   = −∂F_E/∂x     where F_E   = (E + p)u

  Pressure from ideal gas EOS: p = (γ−1)(E − ½ρu²)

  Previous approach used advective form (v·∇φ) which:
    - Drops the compression term ρ·∂u/∂x from the density equation
    - Misses the pressure gradient ∂p/∂x from the momentum equation
    - Is only valid for smooth flows, not at shock discontinuities

  The conservative flux divergence is what the finite volume solver
  actually computes, so MLS(∂F/∂x) aligns with the ground truth data.

Architecture (per-variable):
  1. Feature extraction on static geometry → learned features [n_fe_features]
  2. FiLM: condition learned features on [pressure, density, dt]
  3. Conservative flux divergence (MLS ∂F/∂x) + Diffusion (FD ∇²φ) → physics
  4. LayerNorm on physics features → O(1) scale
  5. MappingAndRecon (SPADE) fuses FiLM'd learned + physics → dφ/dt

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
from .mappingandrecon import MappingAndRecon

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utilities.embed import SimulationConditionedLayerNorm


class ShockTubeDifferentiator(nn.Module):
    """
    Shock Tube Differentiator with conservative Euler flux divergences + FiLM.
    
    Instead of per-variable advection (v·∇φ), computes the actual conservative
    flux divergences (∂F/∂x) from the compressible Euler equations. MLS
    gradient operator computes ∂F/∂x for each equation's flux.
    
    SPADE receives [flux_divergence, diffusion] per variable — physically
    correct features that match what the ground truth solver computed.
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
        spade_random_noise: bool = False,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        zero_init: bool = False,
        gamma: float = 1.4,
        # Legacy args (ignored, kept for compatibility with training scripts)
        list_adv_idx: list = None,
        velocity_indices: list = None,
    ):
        super().__init__()
        
        self.num_static_feats = num_static_feats       # 2 (x, y positions)
        self.num_dynamic_feats = num_dynamic_feats     # 3 (after skip)
        self.n_fe_features = n_fe_features
        self.global_embed_dim = global_embed_dim
        self.global_param_dim = global_param_dim
        self.diffusion_type = diffusion_type.lower()
        self.gamma = gamma  # Ratio of specific heats (ideal gas)
        
        # Legacy attrs (kept for checkpoint compat, not used in forward)
        self.velocity_indices = velocity_indices if velocity_indices is not None else [1]
        
        # Feature extractor
        self.feature_extractor = feature_extractor
        
        # MLS gradient solver (used for flux divergences)
        self.gradient_solver = gradient_solver
        self.laplacian_solver = laplacian_solver
        
        # Diffusion operators (numerical stabilization)
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

        # --- Physics feature normalization + SPADE per variable ---
        # Each variable gets [flux_divergence, diffusion] = 2 physics features
        # (or 1 if no diffusion)
        self.list_phys_norm = nn.ModuleList()
        self.list_mar = nn.ModuleList()
        
        for i in range(num_dynamic_feats):
            n_explicit = 1  # flux divergence always present
            if self.list_dif[i] is not None:
                n_explicit += 1
            
            self.list_phys_norm.append(nn.LayerNorm(n_explicit))
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
        
        # --- FiLM: Condition learned features on raw global params ---
        self.feature_norm = SimulationConditionedLayerNorm(
            normalized_shape=n_fe_features,
            global_dim=global_param_dim,
        )
        
        # --- FiLM on dynamic state (matching V1's derivative_norm) ---
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
        # Velocity: u = (ρu) / ρ
        safe_density = torch.clamp(density.abs(), min=1e-6)
        u = x_momentum / safe_density
        
        # Pressure from ideal gas EOS: p = (γ-1)(E - ½ρu²)
        kinetic = 0.5 * x_momentum * u  # ½ρu² = ½(ρu)(u)
        pressure = (self.gamma - 1.0) * (total_energy - kinetic)
        
        # Conservative fluxes
        F_rho = x_momentum                          # ρu
        F_mom = x_momentum * u + pressure           # ρu² + p
        F_eng = (total_energy + pressure) * u       # (E + p)u
        
        return F_rho, F_mom, F_eng

    def forward(self, state, edge_index):
        """
        Compute dφ/dt for shock tube dynamics using conservative flux divergences.
        
        The numerical integrator calls this as derivative_fn(state, edge_index) where:
          state = [static_feats_augmented | dynamic_state]
          static_feats_augmented = [pos (2) | global_embed (64) | raw_global (3)]
          dynamic_state = [density, x_momentum, total_energy] (3)
        
        So state is [N, 2 + 64 + 3 + 3] = [N, 72]
        """
        if not self._weights_initialized:
            raise RuntimeError("initialize_weights() must be called before forward()")
        
        sf = self.num_static_feats        # 2
        ge = self.global_embed_dim         # 64
        gp = self.global_param_dim         # 3
        
        # Parse the concatenated state tensor
        static_feats   = state[:, :sf]                          # [N, 2]
        global_embed   = state[:, sf:sf + ge]                   # [N, 64]
        raw_global     = state[:, sf + ge:sf + ge + gp]         # [N, 3]
        dynamic_feats  = state[:, sf + ge + gp:]                # [N, 3]
        
        # Raw global attrs for FiLM (single vector, same for all nodes)
        global_attrs = raw_global[0]  # [3]
        
        # --- 1. Learned Features + FiLM ---
        learned_features = self.feature_extractor(
            static_feats, edge_index, pos=static_feats
        )
        learned_features = self.feature_norm(learned_features, global_attrs)
        
        # --- 2. FiLM on dynamic state ---
        dynamic_conditioned = self.derivative_norm(dynamic_feats, global_attrs)
        
        # --- 3. Build mesh_data for MLS ---
        mesh_data = Data(pos=static_feats, edge_index=edge_index)
        mesh_data.num_nodes = state.shape[0]
        
        # --- 4. Conservative Euler Fluxes ---
        density      = dynamic_feats[:, 0:1]  # ρ
        x_momentum   = dynamic_feats[:, 1:2]  # ρu
        total_energy = dynamic_feats[:, 2:3]  # E
        
        F_rho, F_mom, F_eng = self._compute_euler_fluxes(
            density, x_momentum, total_energy
        )
        
        # Stack fluxes: [N, 3] — one column per variable's flux
        flux_stack = torch.cat([F_rho, F_mom, F_eng], dim=1)  # [N, 3]
        
        # MLS gradient of all fluxes: returns list of 3 x [N, 2] tensors
        # Each [N, 2] = [∂F/∂x, ∂F/∂y]
        flux_grads = self.gradient_solver(mesh_data, flux_stack)
        
        # Extract ∂F/∂x (x-component only — quasi-1D problem)
        dFdx = [fg[:, 0:1] for fg in flux_grads]  # list of 3 x [N, 1]
        
        # --- 5. Physics features + SPADE per variable ---
        t_dot_parts = []
        
        for i in range(self.num_dynamic_feats):
            phys_feats = [dFdx[i]]  # flux divergence always present
            
            # Optional diffusion (numerical stabilization)
            if self.list_dif[i] is not None:
                dif = self.list_dif[i](dynamic_feats[:, i:i+1], mesh_data)
                phys_feats.append(dif)
            
            # Normalize physics features to O(1) before SPADE
            phys_cat = torch.cat(phys_feats, dim=1)
            phys_cat = self.list_phys_norm[i](phys_cat)
            
            # SPADE fuses FiLM'd learned features with normalized physics
            out = self.list_mar[i](
                learned_features,
                phys_cat,
                edge_index,
            )
            t_dot_parts.append(out)
        
        return torch.cat(t_dot_parts, dim=1)  # [N, num_dynamic_feats]