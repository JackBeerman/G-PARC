"""
river.py (models/river.py)
==========================
G-PARC models for river dynamics.

G-PARCv2: RiverDifferentiator (MLS) + Numerical integration (Euler/Heun/RK4)
G-PARCv1: DerivativeGNN (no MLS) + IntegralGNN (learned integrator)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrator.numerical import Euler, Heun, RK4, ImplicitEuler


class GPARC_River_V2(nn.Module):
    """
    G-PARCv2 for river dynamics.
    
    MLS operators (advection + diffusion) in the computational graph
    with numerical time integration (Euler/Heun/RK4).
    
    Mirrors GPARC_ElastoPlastic_Numerical exactly:
      - derivative_solver is called via integrator as derivative_fn(state, edge_index)
      - No learnable integrator parameters
      - No boundary condition enforcement (Eulerian mesh, not Lagrangian)
    """

    def __init__(
        self,
        derivative_solver_physics,
        integrator_type: str = "euler",
        num_static_feats: int = 9,
        num_dynamic_feats: int = 4,
        clamp_output: bool = True,
        clamp_max: float = 10.0,
    ):
        super().__init__()

        self.derivative_solver = derivative_solver_physics
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.clamp_output = clamp_output
        self.clamp_max = clamp_max

        # Choose integrator (same as elastoplastic)
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

    def step(
        self,
        static_feats: torch.Tensor,
        dynamic_state: torch.Tensor,
        edge_index,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Single integration step:
          derivative_solver → integrator → (optional clamp)
        
        No BC enforcement needed for river (Eulerian, no fixed walls).
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

        return F_next

    def forward(self, data_list, dt=1.0, teacher_forcing_ratio=0.0):
        """
        Autoregressive rollout with optional scheduled sampling.
        
        Matches GPARC_ElastoPlastic_Numerical.forward() signature.
        
        Args:
            data_list: List of PyG Data objects (sequence)
            dt: Time step size
            teacher_forcing_ratio: Probability of using ground truth
                                   1.0 = always ground truth
                                   0.0 = always prediction (free running)
        
        Returns:
            predictions: List of dynamic state predictions
        """
        predictions = []
        F_prev = None
        
        for i, data in enumerate(data_list):
            x = data.x
            edge_index = data.edge_index
            
            # Preserve mesh_id for MLS caching
            if hasattr(data, "mesh_id"):
                edge_index.mesh_id = data.mesh_id
            
            static_feats = x[:, :self.num_static_feats]
            
            # ---- Scheduled sampling decision ----
            if i == 0:
                # First step: always use ground truth
                current_dynamic = x[:, self.num_static_feats:
                                     self.num_static_feats + self.num_dynamic_feats]
            else:
                if self.training and teacher_forcing_ratio > 0:
                    use_gt = torch.rand(1).item() < teacher_forcing_ratio
                    if use_gt:
                        current_dynamic = x[:, self.num_static_feats:
                                             self.num_static_feats + self.num_dynamic_feats]
                    else:
                        current_dynamic = F_prev.detach()
                else:
                    current_dynamic = F_prev.detach()
            
            # ---- One-step prediction ----
            F_next = self.step(
                static_feats=static_feats,
                dynamic_state=current_dynamic,
                edge_index=edge_index,
                dt=dt,
            )
            
            predictions.append(F_next)
            F_prev = F_next
        
        return predictions

    @torch.no_grad()
    def rollout(self, simulation, num_steps, device=None):
        """
        Clean autoregressive rollout for inference.
        
        Returns list of cumulative states (not deltas).
        """
        if device is None:
            device = next(self.parameters()).device
        
        for data in simulation:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            if hasattr(data, 'pos') and data.pos is not None:
                data.pos = data.pos.to(device)
            if hasattr(data, 'y') and data.y is not None:
                data.y = data.y.to(device)
        
        # Initialize MLS weights
        deriv = self.derivative_solver
        if hasattr(deriv, 'initialize_weights'):
            deriv.initialize_weights(simulation[0])
        
        sf = self.num_static_feats
        static = simulation[0].x[:, :sf]
        current_state = simulation[0].x[:, sf:sf + self.num_dynamic_feats]
        edge_index = simulation[0].edge_index
        
        if hasattr(simulation[0], 'mesh_id'):
            edge_index.mesh_id = simulation[0].mesh_id
        
        states = [current_state.cpu().numpy()]
        
        for t in range(num_steps):
            current_state = self.step(
                static_feats=static,
                dynamic_state=current_state,
                edge_index=edge_index,
                dt=1.0
            )
            states.append(current_state.cpu().numpy())
        
        return states


class GPARC_River_V1(nn.Module):
    """
    G-PARCv1 for river dynamics (NO MLS).
    
    Uses FeatureExtractorGNN → DerivativeGNN → IntegralGNN (learned).
    This is the baseline already trained on river data.
    """
    
    def __init__(
        self,
        feature_extractor,
        derivative_solver,
        integral_solver,
        num_static_feats: int = 9,
        num_dynamic_feats: int = 4,
        feature_out_channels: int = 128,
    ):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.derivative_solver = derivative_solver
        self.integral_solver = integral_solver
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.feature_out_channels = feature_out_channels
    
    def forward(self, data_list, dt=1.0, teacher_forcing_ratio=0.0):
        """Autoregressive rollout with optional scheduled sampling."""
        from torch_geometric.data import Data
        
        predictions = []
        F_prev = None
        
        for i, data in enumerate(data_list):
            x = data.x
            edge_index = data.edge_index
            sf = self.num_static_feats
            df = self.num_dynamic_feats
            static_feats = x[:, :sf]
            N = x.shape[0]
            
            if i == 0:
                current_dynamic = x[:, sf:sf + df]
            else:
                if self.training and teacher_forcing_ratio > 0:
                    if torch.rand(1).item() < teacher_forcing_ratio:
                        current_dynamic = x[:, sf:sf + df]
                    else:
                        current_dynamic = F_prev.detach()
                else:
                    current_dynamic = F_prev.detach()
            
            # Feature extraction
            fe_data = Data(x=static_feats, edge_index=edge_index,
                          pos=static_feats[:, :2])
            fe_data.num_nodes = N
            learned = self.feature_extractor(fe_data)
            
            # Derivative
            deriv_data = Data(x=torch.cat([learned, current_dynamic], dim=1),
                            edge_index=edge_index)
            deriv_data.num_nodes = N
            t_dot = self.derivative_solver(deriv_data)
            
            # Learned integrator
            int_data = Data(x=t_dot, edge_index=edge_index)
            int_data.num_nodes = N
            delta = self.integral_solver(int_data)
            
            F_next = current_dynamic + dt * delta
            predictions.append(F_next)
            F_prev = F_next
        
        return predictions