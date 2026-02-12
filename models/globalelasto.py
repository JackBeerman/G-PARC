# In models/elastoplastic.py
import torch
import torch.nn as nn
import sys
import os
from integrator.numerical import Euler, Heun, RK4, ImplicitEuler
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.featureextractor import FeatureExtractorGNN
from differentiator.differentiator import DerivativeGNN
from integrator.integrator import IntegralGNN


class GPARC_ElastoPlastic_Numerical(nn.Module):
    """
    G-PARC using numerical time integration (Euler / Heun / RK4 / Implicit Euler).
    
    Supports scheduled sampling for robust rollout training.
    UPDATED: Supports both z-score and global_max normalization for boundary detection.
    """

    def __init__(
        self,
        derivative_solver_physics,
        integrator_type: str = "euler",
        num_static_feats: int = 2,
        num_dynamic_feats: int = 2,
        skip_dynamic_indices=None,
        pos_mean=None,
        pos_std=None,
        boundary_threshold: float = 0.5,
        clamp_output: bool = True,
        clamp_max: float = 10.0,
        norm_method: str = "z_score",
        max_position: float = None,
    ):
        super().__init__()

        self.derivative_solver = derivative_solver_physics
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.skip_dynamic_indices = skip_dynamic_indices or []
        self.boundary_threshold = boundary_threshold
        self.clamp_output = clamp_output
        self.clamp_max = clamp_max
        self.norm_method = norm_method
        self.max_position = max_position

        # Register position stats for boundary detection
        if pos_mean is not None and pos_std is not None:
            self.register_buffer("pos_mean", torch.tensor(pos_mean, dtype=torch.float32))
            self.register_buffer("pos_std", torch.tensor(pos_std, dtype=torch.float32))
        else:
            self.pos_mean = None
            self.pos_std = None

        # Choose integrator
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

    def _denormalize_positions(self, static_feats: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized position features to physical coordinates.
        
        Handles both normalization methods:
          z_score:    pos_phys = pos_norm * std + mean
          global_max: pos_phys = pos_norm * max_position
        """
        if self.norm_method == 'global_max' and self.max_position is not None:
            return static_feats * self.max_position
        elif self.pos_mean is not None and self.pos_std is not None:
            return static_feats * self.pos_std.to(static_feats.device) + \
                   self.pos_mean.to(static_feats.device)
        else:
            # No normalization info available — return as-is
            return static_feats

    def process_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Extract relevant dynamic features from targets."""
        num_raw_dynamic = self.num_dynamic_feats + len(self.skip_dynamic_indices)
        all_dynamic_targets = y[:, :num_raw_dynamic]
        keep = [i for i in range(all_dynamic_targets.shape[1]) if i not in self.skip_dynamic_indices]
        return all_dynamic_targets[:, keep]

    def _enforce_dirichlet_bc(
        self, 
        F_next: torch.Tensor, 
        F_current: torch.Tensor,
        static_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce fixed boundary (zero velocity).
        
        Physics: The left wall doesn't move, so:
            U_{t+1} = U_t  (for boundary nodes)
            => dU/dt = 0
        """
        if self.pos_mean is None and self.pos_std is None and self.max_position is None:
            return F_next
    
        # Denormalize using the correct method
        pos_phys = self._denormalize_positions(static_feats)
        is_boundary = (pos_phys[:, 0] < self.boundary_threshold).unsqueeze(1)
        
        # Keep displacement constant at boundary (zero velocity)
        return torch.where(is_boundary, F_current, F_next)

    def _extract_dynamic(self, x: torch.Tensor) -> torch.Tensor:
        """Extract dynamic displacement channels from data.x, respecting skip_dynamic_indices."""
        num_raw = self.num_dynamic_feats + len(self.skip_dynamic_indices)
        raw_dynamic = x[:, self.num_static_feats : self.num_static_feats + num_raw]
        keep = [i for i in range(raw_dynamic.shape[1]) if i not in self.skip_dynamic_indices]
        return raw_dynamic[:, keep]

    def step(
        self,
        static_feats: torch.Tensor,
        dynamic_state: torch.Tensor,
        edge_index,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Single integration step:
          derivative -> integrator -> (optional clamp) -> BC enforcement
        """
        F_next = self.integrator(
            derivative_fn=self.derivative_solver,
            static_feats=static_feats,
            dynamic_state=dynamic_state,
            edge_index=edge_index,
            dt=dt,
        )

        if self.clamp_output:
            F_next = torch.clamp(F_next, -self.clamp_max, self.clamp_max)

        F_next = self._enforce_dirichlet_bc(F_next, dynamic_state, static_feats)
        return F_next

    # =========================================================================
    # SCHEDULED SAMPLING FORWARD PASS
    # =========================================================================
    
    def forward(self, data_list, dt=1.0, teacher_forcing_ratio=0.0):
        """
        Autoregressive rollout with scheduled sampling.
        
        Args:
            data_list: List of PyG Data objects (sequence)
            dt: Time step size
            teacher_forcing_ratio: Probability of using ground truth instead of prediction
                                   1.0 = always ground truth (teacher forcing)
                                   0.0 = always prediction (free running)
                                   
        Returns:
            predictions: List of displacement predictions
        """
        predictions = []
        F_prev = None
        
        for i, data in enumerate(data_list):
            x = data.x
            edge_index = data.edge_index
            
            # Preserve mesh_id for MLS/operator caching
            if hasattr(data, "mesh_id"):
                edge_index.mesh_id = data.mesh_id
            
            static_feats = x[:, :self.num_static_feats]
            
            # ================================================================
            # SCHEDULED SAMPLING DECISION
            # ================================================================
            if i == 0:
                # First step: ALWAYS use ground truth (no previous prediction)
                current_dynamic_input = self._extract_dynamic(x)
            else:
                # Subsequent steps: Mix ground truth and predictions
                if self.training and teacher_forcing_ratio > 0:
                    use_ground_truth = torch.rand(1).item() < teacher_forcing_ratio
                    
                    if use_ground_truth:
                        current_dynamic_input = self._extract_dynamic(x)
                    else:
                        current_dynamic_input = F_prev.detach()
                else:
                    current_dynamic_input = F_prev.detach()
            
            # ================================================================
            # ONE-STEP PREDICTION
            # ================================================================
            F_next = self.step(
                static_feats=static_feats, 
                dynamic_state=current_dynamic_input, 
                edge_index=edge_index, 
                dt=dt
            )
            
            predictions.append(F_next)
            F_prev = F_next
        
        return predictions

        # =========================================================================
    # ROLLOUT HELPER — use this for inference to avoid delta vs state confusion
    # =========================================================================
    
    @torch.no_grad()
    def rollout(self, simulation, num_steps, device=None):
        """
        Clean autoregressive rollout for inference.
        
        Returns a list of cumulative displacement states [N, num_dynamic_feats]
        at each timestep, starting from the initial condition.
        
        IMPORTANT: Each entry is the CUMULATIVE state (not a delta).
        This is the method external scripts should call for evaluation.
        
        Args:
            simulation: List of PyG Data objects with .x, .edge_index, etc.
            num_steps: Number of rollout steps
            device: Target device (defaults to model's device)
            
        Returns:
            states: List of [N, num_dynamic_feats] numpy arrays
                    states[0] = initial condition
                    states[t] = cumulative displacement at timestep t
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Move data to device
        for data in simulation:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            if hasattr(data, 'pos') and data.pos is not None:
                data.pos = data.pos.to(device)
            if hasattr(data, 'y') and data.y is not None:
                data.y = data.y.to(device)
        
        # Initialize MLS weights if needed
        deriv = self.derivative_solver
        if hasattr(deriv, 'initialize_weights'):
            deriv.initialize_weights(simulation[0])
        
        sf = self.num_static_feats
        static = simulation[0].x[:, :sf]
        current_state = self._extract_dynamic(simulation[0].x)
        edge_index = simulation[0].edge_index
        
        # Preserve mesh_id for caching
        if hasattr(simulation[0], 'mesh_id'):
            edge_index.mesh_id = simulation[0].mesh_id
        
        states = [current_state.cpu().numpy()]
        
        for t in range(num_steps):
            # step() returns NEXT STATE (not delta)
            # Euler: F_next = F_current + dt * dF/dt
            current_state = self.step(
                static_feats=static,
                dynamic_state=current_state,
                edge_index=edge_index,
                dt=1.0
            )
            states.append(current_state.cpu().numpy())
        
        return states