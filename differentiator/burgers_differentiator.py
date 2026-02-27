"""
BurgersDifferentiator (Concat+MLP Fusion)
==========================================
G-PARCv2 differentiator for 2D Burgers' equation.

Physics:
    du/dt = -(u·∇)u + (1/Re)∇²u

Fusion Strategy — Concatenation + MLP (not SPADE):
  Physics features (advection, diffusion) are concatenated with learned
  features and processed by an MLP to predict du/dt.

  Same fix applied to shocktube (100x stability improvement) and river
  (10x faster learning, stable val loss). SPADE modulates 128-dim features
  with only 4 physics channels — most dims pass through unchanged.
  Concat+MLP gives full nonlinear access to combine all features.

Architecture:
  1. FeatureExtractor(pos_x, pos_y, Re) → learned features [N, F]
  2. FiLM: condition learned features on Re
  3. Advection: (u·∇)u, (u·∇)v via MLS gradients on RAW velocity
  4. Diffusion: ∇²u, ∇²v via FD or MLS Laplacian on RAW velocity
  5. LayerNorm on physics features → O(1) scale
  6. Concat [learned | physics] → MLP → du/dt [N, 2]

CRITICAL: Physics operators use RAW velocity. FiLM only conditions
the learned representation.

Call signature: (full_state, edge_index) → dφ/dt [N, 2]
  where full_state = [pos_x, pos_y, Re, u, v] → [N, 5]
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .hop import AdvectionMLS, DiffusionMLS, DiffusionFD


class SimulationConditionedLayerNorm(nn.Module):
    """FiLM-style conditioning: LayerNorm then affine transform from global params."""
    def __init__(self, normalized_shape, cond_dim):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)
        self.gamma_proj = nn.Linear(cond_dim, normalized_shape)
        self.beta_proj = nn.Linear(cond_dim, normalized_shape)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x, cond):
        x = self.ln(x)
        if cond.dim() == 0:
            cond = cond.unsqueeze(0)
        gamma = 1.0 + self.gamma_proj(cond)
        beta = self.beta_proj(cond)
        return x * gamma.unsqueeze(0) + beta.unsqueeze(0)


class PhysicsFusionMLP(nn.Module):
    """
    MLP that fuses learned features with physics features.
    
    Input:  [learned_features (n_fe) | physics_features (n_phys)]
    Output: dφ/dt (output_dim)
    
    Two hidden layers with GELU activation and residual connection.
    """
    def __init__(self, n_learned, n_physics, output_dim=2, 
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


class BurgersDifferentiator(nn.Module):
    """
    Differentiator for 2D Burgers' equation with concat+MLP fusion.

    Call signature: (full_state, edge_index) → dφ/dt [N, 2]
      where full_state = [pos_x, pos_y, Re, u, v]  → [N, 5]
    """

    def __init__(
        self,
        feature_extractor,
        gradient_solver,
        laplacian_solver=None,
        n_fe_features=64,
        fusion_hidden_dim=128,
        zero_init=False,
        diffusion_type='fd',
        use_film=True,
        # Legacy SPADE args (accepted but ignored)
        spade_heads=1,
        spade_dropout=0.0,
    ):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.diffusion_type = diffusion_type
        self.use_film = use_film

        # --- Physics Operators ---
        self.advection_op = AdvectionMLS(gradient_solver)

        if diffusion_type == 'fd':
            self.diffusion_op = DiffusionFD()
        elif diffusion_type == 'mls':
            assert laplacian_solver is not None, \
                "laplacian_solver required for diffusion_type='mls'"
            self.diffusion_op = DiffusionMLS(laplacian_solver)
        elif diffusion_type == 'none':
            self.diffusion_op = None
        else:
            raise ValueError(f"Unknown diffusion_type: {diffusion_type}")

        # --- FiLM on Re (learned features ONLY) ---
        if use_film:
            self.film_features = SimulationConditionedLayerNorm(
                n_fe_features, cond_dim=1
            )

        # --- Physics feature normalization ---
        # 4 physics channels: adv_u, adv_v, diff_u, diff_v
        n_physics = 4 if self.diffusion_op is not None else 2
        self.phys_norm = nn.LayerNorm(n_physics)

        # --- Concat+MLP Fusion ---
        self.fusion = PhysicsFusionMLP(
            n_learned=n_fe_features,
            n_physics=n_physics,
            output_dim=2,  # du/dt, dv/dt
            hidden_dim=fusion_hidden_dim,
            zero_init=zero_init,
        )

        self._weights_initialized = False

    def initialize_weights(self, sample_data):
        """Warm up MLS gradient solver (and MLS laplacian if used)."""
        if not self._weights_initialized:
            print("Initializing Burgers MLS Operators...")
            dummy_u = torch.zeros(
                sample_data.num_nodes, 1, device=sample_data.pos.device
            )
            _ = self.advection_op.gradient_solver(sample_data, dummy_u)

            if self.diffusion_type == 'mls':
                _ = self.diffusion_op.laplacian_solver(sample_data)

            self._weights_initialized = True
            print("  ✅ Burgers MLS weights initialized")

    def forward(self, full_state, edge_index):
        """
        Args:
            full_state: [N, 5] → [pos_x, pos_y, Re, u, v]
            edge_index: Graph connectivity
        Returns:
            time_derivative: [N, 2] → [du/dt, dv/dt]
        """
        # 1. Unpack
        static_feats = full_state[:, :3]  # [pos_x, pos_y, Re]
        velocity = full_state[:, 3:]      # [u, v] — RAW, for physics
        re_value = static_feats[0, 2]     # Re is constant per simulation

        # 2. Feature extraction
        learned_feats = self.feature_extractor(static_feats, edge_index)

        # 3. FiLM conditioning on Re — ONLY for learned features
        if self.use_film:
            learned_feats = self.film_features(learned_feats, re_value)

        # 4. Physics operators — ALL on RAW velocity
        mesh_data = Data(pos=static_feats[:, :2], edge_index=edge_index)

        u = velocity[:, 0:1]
        v = velocity[:, 1:2]

        # Advection: (u·∇)u and (u·∇)v
        adv_u = self.advection_op(u, velocity, mesh_data)
        adv_v = self.advection_op(v, velocity, mesh_data)

        # Diffusion: ∇²u and ∇²v
        if self.diffusion_op is not None:
            diff_u = self.diffusion_op(u, mesh_data)
            diff_v = self.diffusion_op(v, mesh_data)
            physics = torch.cat([adv_u, adv_v, diff_u, diff_v], dim=1)
        else:
            physics = torch.cat([adv_u, adv_v], dim=1)

        # 5. Normalize physics features to O(1)
        physics = self.phys_norm(physics)

        # 6. Concat fusion → du/dt
        time_derivative = self.fusion(learned_feats, physics)

        return time_derivative