"""
Graph-based SPADE (Spatially-Adaptive Normalization) modules.

Direct analogs to pixel SPADE and SPADEGeneratorUnit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, BatchNorm


class GraphSPADE(nn.Module):
    """
    Graph implementation of Spatially-Adaptive Normalization.
    
    Args:
        in_channels: Number of channels in input feature map
        mask_channels: Number of channels in physics mask
        epsilon: Small constant for numerical stability
        zero_init: If True, initializes gamma/beta to zero (Identity start).
                   Recommended for deep physics models to prevent shock.
    """
    def __init__(
        self,
        in_channels: int,
        mask_channels: int,
        epsilon: float = 1e-5,
        zero_init: bool = True  # <--- NEW ARGUMENT
    ):
        super().__init__()
        self.epsilon = epsilon
        
        # 1. Normalization
        self.norm = BatchNorm(in_channels, affine=False)
        
        # 2. Mask Processing (Contextual part)
        self.initial_conv = GCNConv(mask_channels, 128)
        
        # 3. Gamma/Beta Generation
        self.gamma_conv = GCNConv(128, in_channels)
        self.beta_conv = GCNConv(128, in_channels)

        # --- OPTIONAL ZERO INITIALIZATION ---
        if zero_init:
            nn.init.zeros_(self.gamma_conv.lin.weight)
            nn.init.zeros_(self.beta_conv.lin.weight)
        # ------------------------------------

    def forward(self, x, mask, edge_index):
        # --- 1. Process Mask (Contextual Awareness) ---
        mask_feat = self.initial_conv(mask, edge_index)
        mask_feat = F.relu(mask_feat)
        
        # Generate parameters
        gamma = self.gamma_conv(mask_feat, edge_index)
        beta = self.beta_conv(mask_feat, edge_index)
        
        # --- 2. Normalize Input ---
        x_normalized = self.norm(x)
        
        # --- 3. Modulate ---
        out = x_normalized * (1 + gamma) + beta
        
        return out


class GraphSPADEGeneratorUnit(nn.Module):
    """
    Graph equivalent of SPADEGeneratorUnit.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mask_channels: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0,
        add_noise: bool = False,
        zero_init: bool = True # <--- NEW ARGUMENT (Passed down)
    ):
        super().__init__()
        self.add_noise = add_noise
        self.noise_std = 0.05
        
        self.gat_out_channels = out_channels // heads if concat else out_channels
        
        # --- Main Path ---
        self.spade1 = GraphSPADE(in_channels, mask_channels, zero_init=zero_init)
        self.conv1 = GATConv(in_channels, self.gat_out_channels, 
                             heads=heads, concat=concat, dropout=dropout)
        
        self.spade2 = GraphSPADE(out_channels, mask_channels, zero_init=zero_init)
        self.conv2 = GATConv(out_channels, self.gat_out_channels,
                             heads=heads, concat=concat, dropout=dropout)
        
        # --- Skip Path ---
        self.spade_skip = GraphSPADE(in_channels, mask_channels, zero_init=zero_init)
        self.conv_skip = GATConv(in_channels, self.gat_out_channels,
                                 heads=heads, concat=concat, dropout=dropout)

    def forward(self, x, mask, edge_index):
        # 1. Noise Injection
        if self.add_noise and self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            
        # --- Main Path ---
        x_main = self.spade1(x, mask, edge_index)
        x_main = F.leaky_relu(x_main, negative_slope=0.2)
        x_main = self.conv1(x_main, edge_index)
        
        x_main = self.spade2(x_main, mask, edge_index)
        x_main = F.leaky_relu(x_main, negative_slope=0.2)
        x_main = self.conv2(x_main, edge_index)
        
        # --- Skip Path ---
        x_skip = self.spade_skip(x, mask, edge_index)
        x_skip = F.leaky_relu(x_skip, negative_slope=0.2)
        x_skip = self.conv_skip(x_skip, edge_index)
        
        # Combine
        return x_main + x_skip