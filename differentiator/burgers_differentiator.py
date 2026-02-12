import torch
import torch.nn as nn
from torch_geometric.data import Data

from .differential_operators import AdvectionMLS, DiffusionMLS
from .mappingandrecon import MappingAndRecon

class BurgersDifferentiator(nn.Module):
    """
    Differentiator for 2D Burgers' Equation.
    
    Physics Equation:
    du/dt = - (u·∇)u + (1/Re)∇²u
    
    Architecture:
    1. Feature Extractor (Learns from Pos + Re)
    2. Physics Operators (Compute Advection + Diffusion terms)
    3. SPADE (Injects Physics into Learned Features)
    """
    def __init__(
        self,
        feature_extractor,
        gradient_solver,
        laplacian_solver,
        n_fe_features,
        spade_heads=1,
        spade_dropout=0.0,
        zero_init=True
    ):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        
        # --- Physics Operators ---
        # Note: We share solvers to save memory
        self.advection_op = AdvectionMLS(gradient_solver)
        self.diffusion_op = DiffusionMLS(laplacian_solver)
        
        # --- SPADE Injector ---
        # Mask Channels = 2 (Advection + Diffusion) per velocity component
        # Total Mask = 2 * 2 = 4 channels
        n_physics_features = 4 
        
        self.spade_block = MappingAndRecon(
            n_base_features=n_fe_features,
            n_mask_channel=n_physics_features,
            output_channel=2, # Outputs dU/dt, dV/dt
            heads=spade_heads,
            concat=True,
            dropout=spade_dropout,
            add_noise=False,
            zero_init=zero_init
        )
        
        self._weights_initialized = False

    def initialize_weights(self, sample_data):
        """Warm up MLS solvers."""
        if not self._weights_initialized:
            print("Initializing Burgers MLS Operators...")
            # We just need to run the solvers once to cache connectivity if needed
            # (Though vectorized solvers don't cache, this is good practice)
            dummy_u = torch.zeros(sample_data.num_nodes, 1, device=sample_data.pos.device)
            _ = self.advection_op.gradient_solver(sample_data, dummy_u)
            _ = self.diffusion_op.laplacian_solver(sample_data)
            self._weights_initialized = True

    def forward(self, full_state, edge_index):
        """
        Args:
            full_state: [N, 5] -> [pos_x, pos_y, Re, u, v]
            edge_index: Graph connectivity
        """
        # 1. Unpack State
        # Static: Pos(2) + Re(1) = 3
        static_feats = full_state[:, :3] 
        # Dynamic: Velocity(2)
        velocity = full_state[:, 3:]     
        
        # 2. Feature Extraction (Learns spatial context + viscosity impact)
        learned_feats = self.feature_extractor(static_feats, edge_index)
        
        # 3. Compute Physics (Advection & Diffusion)
        # Create Data object for operators (uses Pos from input)
        # Note: static_feats[:, :2] is exactly (x, y)
        mesh_data = Data(pos=static_feats[:, :2], edge_index=edge_index)
        
        u = velocity[:, 0:1]
        v = velocity[:, 1:2]
        
        # Advection: (u·∇)u and (u·∇)v
        adv_u = self.advection_op(u, velocity, mesh_data)
        adv_v = self.advection_op(v, velocity, mesh_data)
        
        # Diffusion: ∇²u and ∇²v
        diff_u = self.diffusion_op(u, mesh_data)
        diff_v = self.diffusion_op(v, mesh_data)
        
        # Stack physics features [N, 4]
        # Note: We do NOT multiply by 1/Re here. The network learns that relation.
        # This keeps the physics features numerically stable even if Re is small.
        physics_mask = torch.cat([adv_u, adv_v, diff_u, diff_v], dim=1)
        
        # 4. SPADE Fusion -> dU/dt
        time_derivative = self.spade_block(learned_feats, physics_mask, edge_index)
        
        return time_derivative