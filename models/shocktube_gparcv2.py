"""
shocktubeV2.py (models/shocktubeV2.py)
======================================
G-PARCv2 for compressible Euler equations (shock tube).

Key differences from River/Elastoplastic:
  - Global parameter conditioning (pressure, density, delta_t) via FiLM
  - skip_dynamic_indices: raw dynamic has 4 features, we skip y_momentum → use 3
  - GlobalParameterProcessor embeds [pressure, density, dt] → [64]
  - Global embed is threaded into the differentiator via static_feats augmentation

Architecture:
  GlobalParameterProcessor(pressure, density, dt) → global_embed [64]
  static_augmented = cat(static[2], global_embed[64]) → [66]
  ShockTubeDifferentiator(static_aug + dynamic) → dφ/dt [3]
  Euler: φ_{t+1} = φ_t + dt × dφ/dt
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrator.numerical import Euler, Heun, RK4, ImplicitEuler
from utilities.embed import GlobalParameterProcessor


class GPARC_ShockTube_V2(nn.Module):
    """
    G-PARCv2 for shock tube with global FiLM conditioning.
    
    MLS (advection + diffusion) + Euler numerical integration
    + FiLM from global simulation parameters (pressure, density, delta_t).
    
    The derivative solver expects state = [static(2) | global(64) | dynamic(3)]
    The model's step() builds [static + global] as augmented static, then
    the numerical integrator concatenates [augmented_static | dynamic] → derivative_fn.
    """

    def __init__(
        self,
        derivative_solver_physics,
        integrator_type: str = "euler",
        num_static_feats: int = 2,
        num_dynamic_feats: int = 3,       # AFTER skipping
        skip_dynamic_indices: list = None, # Raw indices to skip (e.g., [2] for y_momentum)
        global_param_dim: int = 3,
        global_embed_dim: int = 64,
        clamp_output: bool = True,
        clamp_max: float = 10.0,
    ):
        super().__init__()

        self.derivative_solver = derivative_solver_physics
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.skip_dynamic_indices = skip_dynamic_indices or []
        self.global_embed_dim = global_embed_dim
        self.clamp_output = clamp_output
        self.clamp_max = clamp_max

        # Global parameter processor (shared across timesteps)
        self.global_processor = GlobalParameterProcessor(
            global_dim=global_param_dim,
            embed_dim=global_embed_dim,
        )

        # Integrator
        it = integrator_type.lower()
        if it == "euler":
            self.integrator = Euler()
        elif it == "heun":
            self.integrator = Heun()
        elif it == "rk4":
            self.integrator = RK4()
        elif it in ("implicit", "impliciteuler", "implicit_euler"):
            self.integrator = ImplicitEuler(max_iters=3, damping=0.9)
        else:
            raise ValueError(f"Unknown integrator type: {integrator_type}")

    # -----------------------------------------------------------------
    # Helpers for global params and dynamic feature skipping
    # -----------------------------------------------------------------

    def _extract_global_attrs(self, data):
        """Extract [pressure, density, delta_t] from a Data object."""
        return torch.stack([
            data.global_pressure.flatten()[0],
            data.global_density.flatten()[0],
            data.global_delta_t.flatten()[0],
        ])

    def _extract_dynamic(self, x):
        """Extract dynamic features from x, applying skip_dynamic_indices."""
        num_raw = self.num_dynamic_feats + len(self.skip_dynamic_indices)
        raw = x[:, self.num_static_feats:self.num_static_feats + num_raw]
        keep = [i for i in range(raw.shape[1]) if i not in self.skip_dynamic_indices]
        return raw[:, keep]

    def process_targets(self, y):
        """Extract target features with same skip logic."""
        num_raw = self.num_dynamic_feats + len(self.skip_dynamic_indices)
        raw = y[:, :num_raw]
        keep = [i for i in range(raw.shape[1]) if i not in self.skip_dynamic_indices]
        return raw[:, keep]

    # -----------------------------------------------------------------
    # Integration step
    # -----------------------------------------------------------------

    def step(
        self,
        static_feats: torch.Tensor,     # [N, 2]
        dynamic_state: torch.Tensor,     # [N, 3]
        edge_index,
        global_embed: torch.Tensor,      # [global_embed_dim]
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Single integration step with global conditioning.
        
        We augment static_feats with the global embedding so the integrator
        passes it through to the derivative solver:
          augmented_static = [static(2) | global(64)]  → [N, 66]
          integrator calls: derivative_fn([augmented_static | dynamic], edge_index)
        """
        N = static_feats.shape[0]
        global_expanded = global_embed.unsqueeze(0).expand(N, -1)  # [N, 64]
        static_augmented = torch.cat([static_feats, global_expanded], dim=1)  # [N, 66]
        
        F_next = self.integrator(
            derivative_fn=self.derivative_solver,
            static_feats=static_augmented,
            dynamic_state=dynamic_state,
            edge_index=edge_index,
            dt=dt,
        )

        if self.clamp_output:
            F_next = torch.clamp(F_next, -self.clamp_max, self.clamp_max)

        return F_next

    # -----------------------------------------------------------------
    # Forward (autoregressive with scheduled sampling)
    # -----------------------------------------------------------------

    def forward(self, data_list, dt=None, teacher_forcing_ratio=0.0):
        """
        Autoregressive rollout with scheduled sampling + global conditioning.
        
        Args:
            data_list: List of PyG Data objects (each has global_pressure,
                       global_density, global_delta_t)
            dt: Integration timestep. If None, uses global_delta_t from data.
            teacher_forcing_ratio: probability of using ground truth
        
        Returns:
            predictions: List of [N, num_dynamic_feats] tensors
        """
        predictions = []
        F_prev = None
        
        # Global params from first timestep (constant across sequence)
        first_data = data_list[0]
        global_attrs = self._extract_global_attrs(first_data)
        global_embed = self.global_processor(global_attrs)  # [64]
        
        # Use explicit delta_t from global params for numerical integration
        if dt is None:
            dt = first_data.global_delta_t.flatten()[0].item()
        
        for i, data in enumerate(data_list):
            x = data.x
            edge_index = data.edge_index
            
            static_feats = x[:, :self.num_static_feats]
            
            # Scheduled sampling
            if i == 0:
                current_dynamic = self._extract_dynamic(x)
            else:
                if self.training and teacher_forcing_ratio > 0:
                    if torch.rand(1).item() < teacher_forcing_ratio:
                        current_dynamic = self._extract_dynamic(x)
                    else:
                        current_dynamic = F_prev.detach()
                else:
                    current_dynamic = F_prev.detach()
            
            F_next = self.step(
                static_feats=static_feats,
                dynamic_state=current_dynamic,
                edge_index=edge_index,
                global_embed=global_embed,
                dt=dt,
            )
            
            predictions.append(F_next)
            F_prev = F_next
        
        return predictions

    # -----------------------------------------------------------------
    # Rollout (inference)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def rollout(self, simulation, num_steps, device=None, dt=None):
        """Autoregressive rollout for inference."""
        if device is None:
            device = next(self.parameters()).device
        
        for data in simulation:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            if hasattr(data, 'pos') and data.pos is not None:
                data.pos = data.pos.to(device)
        
        # Init MLS
        deriv = self.derivative_solver
        if hasattr(deriv, 'initialize_weights'):
            sample = simulation[0]
            if not hasattr(sample, 'pos') or sample.pos is None:
                sample.pos = sample.x[:, :2]
            deriv.initialize_weights(sample)
        
        # Global
        global_attrs = self._extract_global_attrs(simulation[0]).to(device)
        global_embed = self.global_processor(global_attrs)
        
        # Use explicit delta_t from global params
        if dt is None:
            dt = simulation[0].global_delta_t.flatten()[0].item()
        
        static = simulation[0].x[:, :self.num_static_feats]
        current_state = self._extract_dynamic(simulation[0].x)
        edge_index = simulation[0].edge_index
        
        states = [current_state.cpu().numpy()]
        
        for t in range(num_steps):
            current_state = self.step(
                static_feats=static,
                dynamic_state=current_state,
                edge_index=edge_index,
                global_embed=global_embed,
                dt=dt,
            )
            states.append(current_state.cpu().numpy())
        
        return states