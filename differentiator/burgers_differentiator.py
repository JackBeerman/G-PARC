"""
BurgersDifferentiator (Fixed)
================================
G-PARCv2 differentiator for 2D Burgers' equation.

Physics:
    du/dt = -(u·∇)u + (1/Re)∇²u

Architecture:
  1. FeatureExtractor(pos_x, pos_y, Re) → learned features [N, F]
  2. FiLM: condition learned features on Re (NOT dynamic state)
  3. Advection: (u·∇)u, (u·∇)v via MLS gradients on RAW velocity
  4. Diffusion: ∇²u, ∇²v via FD or MLS Laplacian on RAW velocity
  5. SPADE fusion: [adv_u, adv_v, diff_u, diff_v] modulates learned features → du/dt [N, 2]

CRITICAL: Physics operators use RAW velocity. FiLM only conditions
the learned representation. This matches the river differentiator
pattern and prevents rollout divergence on unseen Re values.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .hop import AdvectionMLS, DiffusionMLS, DiffusionFD
from .mappingandrecon import MappingAndRecon


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
        """
        Args:
            x: [N, C] features
            cond: [1] or [cond_dim] conditioning vector (e.g. Reynolds number)
        """
        x = self.ln(x)
        if cond.dim() == 0:
            cond = cond.unsqueeze(0)
        gamma = 1.0 + self.gamma_proj(cond)  # [C]
        beta = self.beta_proj(cond)           # [C]
        return x * gamma.unsqueeze(0) + beta.unsqueeze(0)


class BurgersDifferentiator(nn.Module):
    """
    Differentiator for 2D Burgers' equation.

    Call signature: (full_state, edge_index) → dφ/dt [N, 2]
      where full_state = [pos_x, pos_y, Re, u, v]  → [N, 5]
    """

    def __init__(
        self,
        feature_extractor,
        gradient_solver,
        laplacian_solver=None,
        n_fe_features=64,
        spade_heads=1,
        spade_dropout=0.0,
        zero_init=True,
        diffusion_type='fd',
        use_film=True,
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

        # --- Optional FiLM on Re (learned features ONLY) ---
        # Physics operators use raw velocity — FiLM on dynamic state
        # corrupts the physical meaning and causes rollout divergence.
        if use_film:
            self.film_features = SimulationConditionedLayerNorm(
                n_fe_features, cond_dim=1
            )

        # --- SPADE Fusion ---
        # Mask channels: 2 advection + 2 diffusion = 4
        n_physics_features = 4

        self.spade_block = MappingAndRecon(
            n_base_features=n_fe_features,
            n_mask_channel=n_physics_features,
            output_channel=2,  # du/dt, dv/dt
            heads=spade_heads,
            concat=True,
            dropout=spade_dropout,
            add_noise=False,
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
        
        IMPORTANT: Physics operators use RAW velocity (u, v), NOT FiLM'd.
        FiLM only conditions the learned feature representation. Computing
        advection/diffusion on affine-transformed fields is physically
        meaningless and causes rollout divergence on unseen Re values.
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
        else:
            diff_u = torch.zeros_like(u)
            diff_v = torch.zeros_like(v)

        # Stack physics features [N, 4]
        physics_mask = torch.cat([adv_u, adv_v, diff_u, diff_v], dim=1)

        # 5. SPADE fusion → du/dt
        time_derivative = self.spade_block(
            learned_feats, physics_mask, edge_index
        )

        return time_derivative