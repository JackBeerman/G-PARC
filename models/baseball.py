import torch
import torch.nn as nn
import os
import sys

debug_path = os.path.join(os.path.dirname(__file__), '..', '..')
print(f"Script location: {__file__}")
print(f"Adding to path: {os.path.abspath(debug_path)}")
print(f"Files in that directory: {os.listdir(debug_path) if os.path.exists(debug_path) else 'Directory not found'}")
sys.path.insert(0, debug_path)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Assuming a project structure where these modules are accessible
from utilities.featureextractor import FeatureExtractorGNN
from utilities.embed import SimulationConditionedLayerNorm, GlobalParameterProcessor
from differentiator.differentiator import DerivativeGNN
from integrator.integrator import IntegralGNN

class BaseballGPARC(nn.Module):
    """
    Graph Physics-Aware Recurrent surrogate model for baseball pitching.
    
    Adapted for baseball data:
    - 9 features per joint (3 pos + 3 vel + 3 angles)
    - 18 joints in skeleton
    - Global feature: pitch_speed (mph)
    - All features are dynamic (no static features)
    """
    def __init__(self, feature_extractor, derivative_solver, integral_solver,
                 num_static_feats=0, num_dynamic_feats=9, skip_dynamic_indices=None, 
                 feature_out_channels=128, num_global_feats=1):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.derivative_solver = derivative_solver
        self.integral_solver = integral_solver
        self.num_static_feats = num_static_feats  # 0 for baseball (all dynamic)
        self.num_dynamic_feats = num_dynamic_feats  # 9 for baseball (pos + vel + angles)
        self.skip_dynamic_indices = skip_dynamic_indices or []
        self.feature_out_channels = feature_out_channels
        self.num_global_feats = num_global_feats  # 1 for baseball (pitch_speed only)

        # Helper modules for processing global parameters and conditioning layers
        # Baseball has 1 global feature: pitch_speed
        self.global_processor = GlobalParameterProcessor(
            global_dim=num_global_feats, 
            embed_dim=64
        )
        self.feature_norm = SimulationConditionedLayerNorm(
            self.feature_out_channels, 
            global_dim=num_global_feats
        )
        self.derivative_norm = SimulationConditionedLayerNorm(
            self.num_dynamic_feats, 
            global_dim=num_global_feats
        )
        
    def process_targets(self, y: torch.Tensor) -> torch.Tensor:
        """
        Slices and returns the relevant dynamic features from a target tensor.
        
        For baseball:
        - All 9 features are dynamic (positions, velocities, angles)
        - No static features to skip
        """
        num_raw_dynamic = self.num_dynamic_feats + len(self.skip_dynamic_indices)
        
        if self.num_static_feats == 0:
            # All features are dynamic, start from beginning
            all_dynamic_targets = y[:, :num_raw_dynamic]
        else:
            # Skip static features first (not used in baseball)
            all_dynamic_targets = y[:, self.num_static_feats:self.num_static_feats + num_raw_dynamic]
        
        keep_indices = [i for i in range(all_dynamic_targets.shape[1]) 
                       if i not in self.skip_dynamic_indices]
        dynamic_targets = all_dynamic_targets[:, keep_indices]
        
        assert dynamic_targets.shape[1] == self.num_dynamic_feats, \
            f"Expected {self.num_dynamic_feats} target features, but got {dynamic_targets.shape[1]}"
        return dynamic_targets

    def forward(self, data_list):
        """
        Forward pass through the baseball GPARC model.
        
        Args:
            data_list: List of PyG Data objects, one per timestep
                Each Data object has:
                - x: (num_joints, 9) - input features
                - y: (num_joints, 9) - target features  
                - edge_index: (2, num_edges) - skeleton connectivity
                - pitch_speed: (1,) - pitch speed in mph
        
        Returns:
            predictions: List of predicted features for each timestep
        """
        predictions = []
        F_prev = None
        current_features = None  # Track all dynamic features
        
        # Extract global attributes from first timestep
        # Baseball global feature: pitch_speed only
        first_data = data_list[0]
        global_attrs = first_data.pitch_speed.flatten()[0].unsqueeze(0)  # (1,) tensor
        global_embed = self.global_processor(global_attrs)

        # Recurrently process each timestep in the sequence
        for t, data in enumerate(data_list):
            x = data.x  # (num_joints, 9)
            edge_index = data.edge_index  # (2, num_edges)
            
            # DYNAMIC FEATURE EXTRACTION: All 9 features are dynamic
            if t == 0:
                # First timestep: use ground truth features
                if self.num_static_feats == 0:
                    # All features are dynamic (baseball case)
                    num_raw_dynamic = self.num_dynamic_feats + len(self.skip_dynamic_indices)
                    all_dynamic_feats = x[:, :num_raw_dynamic]
                else:
                    # Traditional approach with static features (not used in baseball)
                    num_raw_dynamic = self.num_dynamic_feats + len(self.skip_dynamic_indices)
                    all_dynamic_feats = x[:, self.num_static_feats:self.num_static_feats + num_raw_dynamic]
                
                # Filter out any indices to skip
                keep_indices = [i for i in range(all_dynamic_feats.shape[1]) 
                               if i not in self.skip_dynamic_indices]
                current_features = all_dynamic_feats[:, keep_indices]
            else:
                # Subsequent timesteps: use updated features from previous prediction
                current_features = F_prev
            
            # Extract and process features dynamically at each timestep
            learned_features = self.feature_extractor(current_features, edge_index)
            learned_features = self.feature_norm(learned_features, global_attrs)

            # For the derivative solver input, we use current features as "dynamic" features
            F_prev_used = current_features
            F_prev_used = self.derivative_norm(F_prev_used, global_attrs)

            # Combine learned features, current features, and global features for derivative solver
            global_context = global_embed.unsqueeze(0).repeat(data.num_nodes, 1)
            Fdot_input = torch.cat([learned_features, F_prev_used, global_context], dim=-1)

            # Core physics-aware integration step
            Fdot = self.derivative_solver(Fdot_input, edge_index)
            Fint = self.integral_solver(Fdot, edge_index)
            F_pred = F_prev_used + Fint
            
            predictions.append(F_pred)
            F_prev = F_pred  # Update state for the next step

        return predictions


# ==================== MODEL CONFIGURATION ====================

def create_baseball_gparc_model(
    num_joints=18,
    num_dynamic_feats=9,
    feature_out_channels=128,
    num_layers=3,
    hidden_dim=256,
    dropout=0.1
):
    """
    Factory function to create a Baseball GPARC model with appropriate configurations.
    
    Args:
        num_joints: Number of joints in skeleton (18 for baseball)
        num_dynamic_feats: Number of features per joint (9: pos + vel + angles)
        feature_out_channels: Output dimension of feature extractor
        num_layers: Number of GNN layers
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
    
    Returns:
        BaseballGPARC model instance
    """
    
    # Feature Extractor GNN
    # Input: 9 features per joint
    feature_extractor = FeatureExtractorGNN(
        in_channels=num_dynamic_feats,
        hidden_channels=hidden_dim,
        out_channels=feature_out_channels,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Derivative Solver GNN
    # Input: learned_features (128) + current_features (9) + global_context (64) = 201
    derivative_solver = DerivativeGNN(
        in_channels=feature_out_channels + num_dynamic_feats + 64,
        hidden_channels=hidden_dim,
        out_channels=num_dynamic_feats,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Integral Solver GNN
    # Input: derivatives (9 features)
    integral_solver = IntegralGNN(
        in_channels=num_dynamic_feats,
        hidden_channels=hidden_dim,
        out_channels=num_dynamic_feats,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Create the full model
    model = BaseballGPARC(
        feature_extractor=feature_extractor,
        derivative_solver=derivative_solver,
        integral_solver=integral_solver,
        num_static_feats=0,  # All features are dynamic
        num_dynamic_feats=num_dynamic_feats,
        skip_dynamic_indices=[],  # Don't skip any features
        feature_out_channels=feature_out_channels,
        num_global_feats=1  # pitch_speed only
    )
    
    return model


# ==================== USAGE EXAMPLE ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Baseball GPARC Model Configuration")
    print("=" * 80)
    
    # Create model
    model = create_baseball_gparc_model(
        num_joints=18,
        num_dynamic_feats=9,
        feature_out_channels=128,
        num_layers=3,
        hidden_dim=256,
        dropout=0.1
    )
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"\nModel structure:")
    print(f"  - Input features per joint: 9 (3 pos + 3 vel + 3 angles)")
    print(f"  - Number of joints: 18")
    print(f"  - Global feature: 1 (pitch_speed)")
    print(f"  - Feature extractor output: {model.feature_out_channels}")
    print(f"  - All features are dynamic (no static features)")
    
    print("\n" + "=" * 80)
    print("Key Differences from Tennis Model:")
    print("=" * 80)
    print("  Tennis → Baseball")
    print("  - Features per joint: 6 → 9 (added angles)")
    print("  - Joints: 17 → 18")
    print("  - Global features: 5 → 1")
    print("    * Tennis: server_id, serve_number, set_number, game_number, point_number")
    print("    * Baseball: pitch_speed")
    print("  - Both use all dynamic features (no static)")
    print("\n" + "=" * 80)