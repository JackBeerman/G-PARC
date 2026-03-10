"""
cylinder_gparcv2.py (models/cylinder_gparcv2.py)
=================================================
G-PARCv2 for incompressible Karman vortex street (cylinder flow).

Key design:
  - Global parameter conditioning (Reynolds number) via FiLM
  - skip_dynamic_indices: raw dynamic has 7 features, user may skip some
  - GlobalParameterProcessor embeds [Reynolds] → [64]
  - MLS advection-diffusion physics in the differentiator
  - Numerical integration (Euler/Heun/RK4)

Architecture:
  GlobalParameterProcessor(Reynolds) → global_embed [64]
  static_augmented = cat(static[3], global_embed[64], raw_global[1]) → [68]
  CylinderDifferentiator(static_aug + dynamic) → dφ/dt [D]
  Euler: φ_{t+1} = φ_t + dt × dφ/dt

Dynamic features (7 total, after skipping):
  [0] pressure, [1] vx, [2] vy, [3] vz, [4] ωx, [5] ωy, [6] ωz
  Common skip: [3, 4, 5] (vz, ωx, ωy ~0 for 2D flow) → 4 remaining

VRAM Strategy:
  - GraphConvFeatureExtractorV2 (no attention = O(N) not O(N²))
  - Concat fusion MLPs (no GAT in differentiator)
  - Numerical integrator (no learnable integrator weights)
  - Net result: ~60-100k nodes should fit in 24GB VRAM
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrator.numerical import Euler, Heun, RK4, ImplicitEuler
from utilities.embed import GlobalParameterProcessor


class GPARC_Cylinder_V2(nn.Module):
    """
    G-PARCv2 for cylinder flow with global FiLM conditioning.
    
    MLS (advection + diffusion) + Numerical integration (Euler/Heun/RK4)
    + FiLM from Reynolds number.
    
    The derivative solver expects state = [static(3) | global_embed(64) | raw_global(1) | dynamic(D)]
    The model's step() builds [static + global_embed + raw_global] as augmented static, then
    the numerical integrator concatenates [augmented_static | dynamic] → derivative_fn.
    """

    def __init__(
        self,
        derivative_solver_physics,
        integrator_type: str = "euler",
        num_static_feats: int = 3,           # x, y, z positions
        num_dynamic_feats: int = 7,           # AFTER skipping
        skip_dynamic_indices: list = None,    # Raw indices to skip
        global_param_dim: int = 1,            # Reynolds number
        global_embed_dim: int = 64,
        clamp_output: bool = True,
        clamp_max: float = 10.0,
    ):
        super().__init__()

        self.derivative_solver = derivative_solver_physics
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.skip_dynamic_indices = skip_dynamic_indices or []
        self.global_param_dim = global_param_dim
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
    # Helpers
    # -----------------------------------------------------------------

    def _extract_global_attrs(self, data):
        """Extract Reynolds number from a Data object."""
        if hasattr(data, 'global_reynolds'):
            return data.global_reynolds.flatten()[:self.global_param_dim]
        elif hasattr(data, 'global_params'):
            return data.global_params.flatten()[:self.global_param_dim]
        else:
            return torch.zeros(self.global_param_dim, device=data.x.device)

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
        static_feats: torch.Tensor,     # [N, 3]
        dynamic_state: torch.Tensor,     # [N, D]
        edge_index,
        global_embed: torch.Tensor,      # [global_embed_dim]
        global_attrs: torch.Tensor,      # [global_param_dim] raw params
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Single integration step with global conditioning.
        
        Augments static_feats with both global_embed AND raw_global_attrs:
          augmented_static = [static(3) | global_embed(64) | raw_global(1)] → [N, 68]
          integrator calls: derivative_fn([augmented_static | dynamic], edge_index)
        """
        N = static_feats.shape[0]
        global_embed_expanded = global_embed.unsqueeze(0).expand(N, -1)   # [N, 64]
        raw_global_expanded = global_attrs.unsqueeze(0).expand(N, -1)     # [N, 1]
        
        static_augmented = torch.cat([
            static_feats,           # [N, 3]
            global_embed_expanded,  # [N, 64]
            raw_global_expanded,    # [N, 1]
        ], dim=1)                   # [N, 68]
        
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
            data_list: List of PyG Data objects (each has global_params/global_reynolds)
            dt: Integration timestep. Default 1.0.
            teacher_forcing_ratio: probability of using ground truth
        
        Returns:
            predictions: List of [N, num_dynamic_feats] tensors
        """
        predictions = []
        F_prev = None
        
        # Global params from first timestep (constant across sequence)
        first_data = data_list[0]
        global_attrs = self._extract_global_attrs(first_data).to(first_data.x.device)
        global_embed = self.global_processor(global_attrs)      # [64] processed
        
        if dt is None:
            dt = 1.0
        
        for i, data in enumerate(data_list):
            x = data.x
            edge_index = data.edge_index
            
            # Propagate mesh_id for MLS cache invalidation
            if hasattr(data, 'mesh_id'):
                edge_index.mesh_id = data.mesh_id
            
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
                global_attrs=global_attrs,
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
                sample.pos = sample.x[:, :self.num_static_feats]
            deriv.initialize_weights(sample)
        
        # Global
        global_attrs = self._extract_global_attrs(simulation[0]).to(device)
        global_embed = self.global_processor(global_attrs.to(device))
        
        if dt is None:
            dt = 1.0
        
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
                global_attrs=global_attrs,
                dt=dt,
            )
            states.append(current_state.cpu().numpy())
        
        return states