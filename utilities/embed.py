import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.amp import custom_fwd # [# NEW] Import the decorator

class SimulationConditionedLayerNorm(nn.Module):
    """
    Applies Layer Normalization where the gain (gamma) and bias (beta)
    are dynamically generated from global simulation parameters.
    Updated to use selective float32 precision for stability.
    """
    def __init__(self, normalized_shape, global_dim=1): # Changed global_dim to 1 as per your model
        super().__init__()
        self.normalized_shape = normalized_shape
        
        # MLPs to generate gamma and beta from global attributes
        self.global_to_gamma = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(),
            nn.Linear(64, normalized_shape)
        )
        self.global_to_beta = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(),
            nn.Linear(64, normalized_shape)
        )
        
        # Base LayerNorm without its own learnable affine parameters
        self.ln = nn.LayerNorm(normalized_shape, elementwise_affine=False, eps=1e-6)

    # [# NEW] Decorated method to run in full float32 precision
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def generate_gamma_beta(self, global_attrs):
        """Generates gamma and beta in float32 to prevent overflow."""
        gamma = self.global_to_gamma(global_attrs)
        beta = self.global_to_beta(global_attrs)
        return gamma, beta

    def forward(self, x, global_attrs):
        """
        Applies the conditioned layer normalization.
        """
        normalized = self.ln(x)
        
        # [# CHANGED] Call the new decorated method
        gamma, beta = self.generate_gamma_beta(global_attrs)
        
        return gamma * normalized + beta
    
class GlobalParameterProcessor(nn.Module):
    """
    An MLP that processes low-dimensional global simulation parameters
    into a higher-dimensional embedding.
    """
    def __init__(self, global_dim=3, embed_dim=64):
        super().__init__()
        self.global_embedding = nn.Sequential(
            nn.Linear(global_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, global_attrs):
        return self.global_embedding(global_attrs)

class GlobalModulatedGNN(nn.Module):
    """
    A GNN layer whose output is modulated (scaled and shifted) by a
    global embedding vector. This is a form of feature-wise linear modulation (FiLM).
    """
    def __init__(self, in_channels, out_channels, global_embed_dim=64, heads=4):
        super().__init__()
        self.gnn = GATConv(in_channels, out_channels, heads=heads, concat=False)
        
        # Modulator for scaling the GNN output
        self.feature_modulator = nn.Sequential(
            nn.Linear(global_embed_dim, out_channels),
            nn.Tanh()  # Tanh bounds the scaling factor to prevent explosions
        )
        
        # Modulator for shifting the GNN output
        self.bias_modulator = nn.Sequential(
            nn.Linear(global_embed_dim, out_channels)
        )
        
    def forward(self, x, edge_index, global_embed):
        gnn_out = self.gnn(x, edge_index)
        
        # Generate scale and bias from the global embedding
        feature_mod = self.feature_modulator(global_embed)
        bias_mod = self.bias_modulator(global_embed)
        
        # Apply modulation: y = gnn(x) * (1 + scale) + bias
        modulated_output = gnn_out * (1 + feature_mod) + bias_mod
        
        return modulated_output