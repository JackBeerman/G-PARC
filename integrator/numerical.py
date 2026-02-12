import torch
import torch.nn as nn

class NumericalIntegrator(nn.Module):
    """
    Base class for numerical time-stepping schemes.
    """
    def __init__(self):
        super().__init__()

    def forward(self, derivative_fn, static_feats, dynamic_state, edge_index, dt):
        """
        Args:
            derivative_fn: The neural network F(state) -> dU/dt
            static_feats: [N, S] Static features (positions)
            dynamic_state: [N, D] Current dynamic state (U_t)
            edge_index: Graph connectivity
            dt: Time step size
        """
        raise NotImplementedError

class Euler(NumericalIntegrator):
    """
    Euler Method (1st Order).
    U_{t+1} = U_t + dt * F(U_t)
    """
    def forward(self, derivative_fn, static_feats, dynamic_state, edge_index, dt):
        # Concatenate [Pos, U_t] to feed the Differentiator
        state = torch.cat([static_feats, dynamic_state], dim=-1)
        
        # Calculate Derivative: dU/dt
        k1 = derivative_fn(state, edge_index)
        
        # Step
        return dynamic_state + dt * k1

class ImplicitEuler(NumericalIntegrator):
    """
    Implicit/Backward Euler - unconditionally stable.
    Solves: U_{t+1} = U_t + dt * F(U_{t+1}) via fixed-point iteration.
    """
    def __init__(self, max_iters=3, damping=0.9):
        super().__init__()
        self.max_iters = max_iters
        self.damping = damping
    
    def forward(self, derivative_fn, static_feats, dynamic_state, edge_index, dt):
        u_next = dynamic_state
        
        for _ in range(self.max_iters):
            state = torch.cat([static_feats, u_next], dim=-1)
            f_next = derivative_fn(state, edge_index)
            u_new = dynamic_state + dt * f_next
            
            # Damped update for convergence
            u_next = self.damping * u_new + (1 - self.damping) * u_next
        
        return u_next

class Heun(NumericalIntegrator):
    """
    Heun's Method / Improved Euler (2nd Order).
    Predictor-Corrector scheme. 2 calls to derivative_fn per step.
    """
    def forward(self, derivative_fn, static_feats, dynamic_state, edge_index, dt):
        # 1. Predictor (Euler)
        state_1 = torch.cat([static_feats, dynamic_state], dim=-1)
        k1 = derivative_fn(state_1, edge_index)
        
        u_tilde = dynamic_state + dt * k1
        
        # 2. Corrector
        state_2 = torch.cat([static_feats, u_tilde], dim=-1)
        k2 = derivative_fn(state_2, edge_index)
        
        # Average
        return dynamic_state + (dt / 2.0) * (k1 + k2)

class RK4(NumericalIntegrator):
    """
    Runge-Kutta 4 (4th Order).
    High precision, 4 calls to derivative_fn per step.
    """
    def forward(self, derivative_fn, static_feats, dynamic_state, edge_index, dt):
        # k1
        state_1 = torch.cat([static_feats, dynamic_state], dim=-1)
        k1 = derivative_fn(state_1, edge_index)
        
        # k2
        u2 = dynamic_state + 0.5 * dt * k1
        state_2 = torch.cat([static_feats, u2], dim=-1)
        k2 = derivative_fn(state_2, edge_index)
        
        # k3
        u3 = dynamic_state + 0.5 * dt * k2
        state_3 = torch.cat([static_feats, u3], dim=-1)
        k3 = derivative_fn(state_3, edge_index)
        
        # k4
        u4 = dynamic_state + dt * k3
        state_4 = torch.cat([static_feats, u4], dim=-1)
        k4 = derivative_fn(state_4, edge_index)
        
        # Combine
        return dynamic_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)