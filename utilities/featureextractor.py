# featureextractor.py

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphUNet

class FeatureExtractorGNN(nn.Module):
    """
    GraphUNet-based feature extractor for each node with attention.
    """
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=128, 
                 depth=3, pool_ratios=0.5, heads=4, concat=True, dropout=0.6):
        super(FeatureExtractorGNN, self).__init__()
        self.unet = GraphUNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            depth=depth,
            pool_ratios=pool_ratios,
            act=F.relu
        )
        self.attention1 = GATConv(out_channels, out_channels, heads=heads, 
                                  concat=concat, dropout=dropout)
        self.attention2 = GATConv(out_channels * heads if concat else out_channels, 
                                  out_channels, heads=1, concat=False, dropout=dropout)
        # This linear layer for the residual connection needs to match the output of attention2
        self.residual_proj = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        # The original input is passed to the UNet
        unet_out = self.unet(x, edge_index)
        
        # The output of the UNet is used for the residual connection and the attention layers
        residual = self.residual_proj(unet_out)
        
        x = F.elu(self.attention1(unet_out, edge_index))
        x = self.attention2(x, edge_index)
        
        # Add the projected residual
        x += residual
        return x
    
# --- New and Improved Class (for 3D models and beyond) ---
class StableFeatureExtractorGNN(nn.Module):
    """
    An improved, numerically stable version of the feature extractor that
    normalizes the output of the GraphUNet before attention.
    """
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=128, 
                 depth=3, pool_ratios=0.5, heads=4, concat=True, dropout=0.6):
        super(StableFeatureExtractorGNN, self).__init__()
        self.unet = GraphUNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            depth=depth,
            pool_ratios=pool_ratios,
            act=F.relu
        )
        
        # [ADDED FOR STABILITY] Normalization layer for the UNet's output
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)

        self.attention1 = GATConv(out_channels, out_channels, heads=heads, 
                                  concat=concat, dropout=dropout)
        self.attention2 = GATConv(out_channels * heads if concat else out_channels, 
                                  out_channels, heads=1, concat=False, dropout=dropout)
        self.residual_proj = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.unet(x, edge_index)
        
        # [ADDED FOR STABILITY] Normalize the UNet output before using it
        unet_out = self.norm(x)
        
        residual = self.residual_proj(unet_out)
        
        x = F.elu(self.attention1(unet_out, edge_index))
        x = self.attention2(x, edge_index)
        
        x += residual
        return x
    
    
class RobustFeatureExtractorGNN(nn.Module):
    """
    A deep, numerically stable feature extractor using a pre-norm architecture
    with GATConv layers and residual connections. CORRECTED for shape mismatch.
    """
    def __init__(self, in_channels=3, hidden_channels=128, out_channels=64, 
                 depth=4, heads=4, dropout=0.2):
        super().__init__()
        
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.lin_out = nn.Linear(hidden_channels, out_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(depth):
            # [FIX] Set concat=False to average heads and preserve dimension
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout, concat=False)
            )
            self.norms.append(nn.LayerNorm(hidden_channels, eps=1e-6))
            
        # [REMOVED] The separate aggregate_conv is no longer needed

    def forward(self, x, edge_index):
        # 1. Initial projection
        x = self.lin_in(x)

        # 2. Main processing loop
        for i in range(len(self.convs)):
            residual = x
            
            # Pre-normalization
            x = self.norms[i](x)
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            
            # Residual connection (now works correctly)
            x = x + residual
            
        # 3. Final projection
        x = self.lin_out(x)
        
        return x
    