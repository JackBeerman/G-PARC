import torch.nn as nn
from integrator.numerical import Euler, Heun, RK4
import torch
from differentiator.differential_operators import DiffusionMLS, SolveWeightLST2d

class GPARC_Burgers_Numerical(nn.Module):
    """
    G-PARC for Burgers' Equation (Numerical Integration).
    Fixed: Added missing attribute assignment.
    """
    def __init__(
        self,
        derivative_solver,
        integrator_type='rk4', 
        num_static_feats=3,   # x, y, Re
        num_dynamic_feats=2,  # u, v
    ):
        super().__init__()
        self.derivative_solver = derivative_solver
        
        # --- FIX STARTS HERE ---
        # These variables must be saved to 'self' so forward() can access them
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        # --- FIX ENDS HERE ---
        
        if integrator_type.lower() == 'euler': 
            self.integrator = Euler()
        elif integrator_type.lower() == 'heun': 
            self.integrator = Heun()
        elif integrator_type.lower() == 'rk4': 
            self.integrator = RK4()
        else: 
            raise ValueError(f"Unknown integrator: {integrator_type}")

    def process_targets(self, y):
        return y

    def forward(self, data_list, dt=1.0): # Default dt=1.0 (Matching Pixel Model)
        predictions = []
        F_prev = None
        
        for data in data_list:
            x = data.x
            edge_index = data.edge_index
            
            # 1. Split Inputs using the stored variable
            # This line caused the crash before because self.num_static_feats didn't exist
            static_feats = x[:, :self.num_static_feats]
            
            # Dynamic: [u, v]
            current_dynamic = x[:, self.num_static_feats:]
            
            if F_prev is not None:
                current_dynamic = F_prev
            
            # Integration Step
            F_next = self.integrator(
                derivative_fn=self.derivative_solver,
                static_feats=static_feats,
                dynamic_state=current_dynamic,
                edge_index=edge_index,
                dt=dt
            )
            
            predictions.append(F_next)
            F_prev = F_next
            
        return predictions


import torch
import torch.nn as nn
from integrator.numerical import Euler, Heun, RK4

class GPARC_Burgers_Test(nn.Module):
    """
    G-PARC for Burgers' Equation (Numerical Integration).
    Fixed: Boundary Condition Masking to prevent 'Bands'.
    """
    def __init__(
        self,
        derivative_solver,
        integrator_type='rk4', 
        num_static_feats=3,   # x, y, Re
        num_dynamic_feats=2,  # u, v
    ):
        super().__init__()
        self.derivative_solver = derivative_solver
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        
        if integrator_type.lower() == 'euler': 
            self.integrator = Euler()
        elif integrator_type.lower() == 'heun': 
            self.integrator = Heun()
        elif integrator_type.lower() == 'rk4': 
            self.integrator = RK4()
        else: 
            raise ValueError(f"Unknown integrator: {integrator_type}")

    def process_targets(self, y):
        return y

    def forward(self, data_list, dt=1.0):
        predictions = []
        F_prev = None
        
        # --- 1. PRE-COMPUTE BOUNDARY MASK ---
        # We assume all graphs in the batch have the same geometry structure (usually true)
        # or we calculate per graph. Since 'x' is stacked, we can calc for all.
        
        # Get positions from first step (x, y are indices 0, 1)
        # We assume positions are normalized [0, 1]
        all_pos = data_list[0].x[:, :2] 
        
        # Define Margin (e.g. 2% from edge)
        # Nodes within this margin are "Boundary Nodes"
        margin = 0.02
        
        # Mask is 1.0 for Interior, 0.0 for Boundary
        mask_x = (all_pos[:, 0] > margin) & (all_pos[:, 0] < (1.0 - margin))
        mask_y = (all_pos[:, 1] > margin) & (all_pos[:, 1] < (1.0 - margin))
        
        # Combine: Node must be inside X AND inside Y to be updated
        interior_mask = (mask_x & mask_y).float().unsqueeze(1) # [N, 1]
        
        for data in data_list:
            x = data.x
            edge_index = data.edge_index
            
            static_feats = x[:, :self.num_static_feats]
            current_dynamic = x[:, self.num_static_feats:]
            
            if F_prev is not None:
                current_dynamic = F_prev
            
            # --- STABILIZATION ---
            current_dynamic = torch.clamp(current_dynamic, -10.0, 10.0)

            # Integration Step
            F_next = self.integrator(
                derivative_fn=self.derivative_solver,
                static_feats=static_feats,
                dynamic_state=current_dynamic,
                edge_index=edge_index,
                dt=dt
            )
            
            # --- FIX: APPLY BOUNDARY MASK ---
            # Calculate the change the model WANTS to make
            update = F_next - current_dynamic
            
            # Kill the update at the boundaries (Force dU/dt = 0)
            masked_update = update * interior_mask
            
            # Apply masked update
            F_next = current_dynamic + masked_update
            
            # Final clamp
            F_next = torch.clamp(F_next, -10.0, 10.0)
            
            predictions.append(F_next)
            F_prev = F_next
            
        return predictions


class GPARC_Burgers_Dissipative(nn.Module):
    """
    G-PARC for Burgers' Equation with Entropy-Stable Artificial Viscosity.
    
    This model adds a 'Physics Loss' term directly into the derivative calculation.
    It dampens high-frequency noise (Gibbs oscillations) at shock fronts, preventing
    explosions in high-Reynolds number flows.
    """
    def __init__(
        self,
        derivative_solver,
        integrator_type='euler', 
        num_static_feats=3,   
        num_dynamic_feats=2,
        dissipation_factor=0.01  # Strength of the artificial viscosity (blur)
    ):
        super().__init__()
        self.derivative_solver = derivative_solver
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.dissipation_factor = dissipation_factor
        
        # Select Integrator
        if integrator_type.lower() == 'euler': self.integrator = Euler()
        elif integrator_type.lower() == 'heun': self.integrator = Heun()
        elif integrator_type.lower() == 'rk4': self.integrator = RK4()
        else: raise ValueError(f"Unknown integrator: {integrator_type}")
        
        # Helper operator to compute Laplacian (Curvature) for smoothing
        # We create a fresh solver here to ensure it's independent of the main physics brain
        self.smoothing_op = DiffusionMLS(SolveWeightLST2d())

    def process_targets(self, y):
        return y

    def _combined_derivative(self, full_state, edge_index):
        """
        Calculates dU/dt = NN(State) + Viscosity(State)
        This signature matches what the Integrator expects: f(state, edge_index)
        """
        # 1. Unpack parts needed for Physics
        # full_state is [Static, Dynamic]
        current_dynamic = full_state[:, self.num_static_feats:]
        
        # We need the 'pos' for the smoothing operator. 
        # In this specific architecture, 'pos' is the first 2 columns of static feats.
        # We reconstruct a temporary Data object just for the operator
        pos = full_state[:, :2] 
        temp_data = type('obj', (object,), {'pos': pos, 'edge_index': edge_index})
        
        # 2. Compute Neural Derivative (The Flux)
        # Note: If derivative_solver is wrapped in HardBoundaryConstraint, 
        # this Fdot will already be 0.0 at the boundaries.
        Fdot = self.derivative_solver(full_state, edge_index)
        
        # 3. Compute Artificial Viscosity (The Entropy Fix)
        # Calculate Curvature (Laplacian)
        curvature = self.smoothing_op(current_dynamic, temp_data)
        
        # Adaptive Scaling (only apply viscosity at shocks)
        # If Fdot is 0 (boundary), adaptive_scale is 0, so Viscosity is 0.
        # This preserves the Hard Constraint logic automatically.
        adaptive_scale = torch.tanh(torch.abs(Fdot)) 
        
        artificial_viscosity = self.dissipation_factor * adaptive_scale * curvature
        
        # 4. Combine
        total_derivative = Fdot + artificial_viscosity
        
        # Clamp for numerical safety
        total_derivative = torch.clamp(total_derivative, -10.0, 10.0)
        
        return total_derivative

    def forward(self, data_list, dt=1.0):
        predictions = []
        F_prev = None
        
        for data in data_list:
            x = data.x
            edge_index = data.edge_index
            
            static_feats = x[:, :self.num_static_feats]
            current_dynamic = x[:, self.num_static_feats:]
            
            if F_prev is not None:
                current_dynamic = F_prev
            
            # --- USE THE INTEGRATOR CLASS ---
            # We pass our custom "_combined_derivative" method to the integrator.
            # The integrator will call this function 1x (Euler) or 4x (RK4).
            
            F_next = self.integrator(
                derivative_fn=self._combined_derivative, 
                static_feats=static_feats,
                dynamic_state=current_dynamic,
                edge_index=edge_index,
                dt=dt
            )
            
            # Final Safety Clamp on State
            F_next = torch.clamp(F_next, -10.0, 10.0)
            
            predictions.append(F_next)
            F_prev = F_next
            
        return predictions