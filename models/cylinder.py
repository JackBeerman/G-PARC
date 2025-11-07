import torch
import torch.nn as nn
import os
import sys
from torch.amp import custom_fwd # [# NEW] Import the decorator

debug_path = os.path.join(os.path.dirname(__file__), '..', '..')
print(f"Script location: {__file__}")
print(f"Adding to path: {os.path.abspath(debug_path)}")
print(f"Files in that directory: {os.listdir(debug_path) if os.path.exists(debug_path) else 'Directory not found'}")
sys.path.insert(0, debug_path)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Assuming a project structure where these modules are accessible
from utilities.featureextractor import RobustFeatureExtractorGNN
from utilities.embed import SimulationConditionedLayerNorm, GlobalParameterProcessor
from differentiator.differentiator import DerivativeGNN
from integrator.integrator import IntegralGNN


class GPARC(nn.Module):
    """
    Main GPARC surrogate model with full debugging prints and mixed-precision stability fixes.
    """
    def __init__(self, feature_extractor, derivative_solver, integral_solver,
                 num_static_feats=3, num_dynamic_feats=7, skip_dynamic_indices=None, feature_out_channels=64):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.derivative_solver = derivative_solver
        self.integral_solver = integral_solver
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.skip_dynamic_indices = skip_dynamic_indices or []
        self.feature_out_channels = feature_out_channels

        self.global_processor = GlobalParameterProcessor(global_dim=1, embed_dim=64)
        self.feature_norm = SimulationConditionedLayerNorm(self.feature_out_channels, global_dim=1)
        self.derivative_norm = SimulationConditionedLayerNorm(self.num_dynamic_feats, global_dim=1)
        
    def process_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Slices and returns the relevant dynamic features from a target tensor."""
        num_raw_dynamic = self.num_dynamic_feats + len(self.skip_dynamic_indices)
        all_dynamic_targets = y[:, :num_raw_dynamic]
        
        keep_indices = [i for i in range(all_dynamic_targets.shape[1]) if i not in self.skip_dynamic_indices]
        dynamic_targets = all_dynamic_targets[:, keep_indices]
        
        assert dynamic_targets.shape[1] == self.num_dynamic_feats, \
            f"Expected {self.num_dynamic_feats} target features, but got {dynamic_targets.shape[1]}"
        return dynamic_targets

    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def process_globals(self, global_attrs):
        """Processes global attributes in float32 to prevent overflow."""
        return self.global_processor(global_attrs)

    def forward(self, data_list):
        F_prev = None
        predictions = []
    
        first_data = data_list[0]
        
        # Use the decorated method for the global embedding
        global_attrs = first_data.global_params.flatten()
        global_embed = self.process_globals(global_attrs)
    
        # Static features are processed once per simulation sequence
        static_feats_0 = first_data.x[:, :self.num_static_feats]
        learned_static_features = self.feature_extractor(static_feats_0, first_data.edge_index)
        learned_static_features = self.feature_norm(learned_static_features, global_attrs)
        
        # Recurrently process each timestep in the sequence
        for data in data_list:
            num_raw_dynamic = self.num_dynamic_feats + len(self.skip_dynamic_indices)
            all_dynamic_feats = data.x[:, self.num_static_feats:self.num_static_feats + num_raw_dynamic]
            keep_indices = [j for j in range(all_dynamic_feats.shape[1]) if j not in self.skip_dynamic_indices]
            dynamic_feats_t = all_dynamic_feats[:, keep_indices]
            
            F_prev_used = dynamic_feats_t if F_prev is None else F_prev
            F_prev_used = self.derivative_norm(F_prev_used, global_attrs)
    
            global_context = global_embed.unsqueeze(0).repeat(data.num_nodes, 1)
            Fdot_input = torch.cat([learned_static_features, F_prev_used, global_context], dim=-1)
    
            Fdot = self.derivative_solver(Fdot_input, data.edge_index)
            Fint = self.integral_solver(Fdot, data.edge_index)
            F_pred = F_prev_used + Fint
            
            predictions.append(F_pred)
            F_prev = F_pred
            
        return predictions