# featureextractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphUNet, GraphConv, GINEConv



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




class GraphConvFeatureExtractor(nn.Module):
    """GraphConv feature extractor for stiff mechanics."""
    
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels, 
        num_layers=4,
        dropout=0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
        self.convs.append(GraphConv(hidden_channels, out_channels))
        
        self._init_weights()
        
    def _init_weights(self):
        for conv in self.convs:
            if hasattr(conv, 'lin_rel'):
                nn.init.xavier_uniform_(conv.lin_rel.weight, gain=0.5)
            if hasattr(conv, 'lin_root'):
                nn.init.xavier_uniform_(conv.lin_root.weight, gain=0.5)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        nn.init.zeros_(self.input_proj.bias)
        
    def forward(self, x, edge_index):
        x_res = self.input_proj(x)
        
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = x + x_res
        
        for conv in self.convs[1:-1]:
            x_res = x
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0 and self.training:
                x = F.dropout(x, p=self.dropout)
            x = x + x_res
        
        x = self.convs[-1](x, edge_index)
        return x

class GraphConvFeatureExtractorV2(nn.Module):
    """
    Improved Feature Extractor for G-PARC:
    - GINEConv: Anisotropic message passing using relative node positions.
    - LayerNorm: Stabilizes latent features and prevents activation explosion.
    - Deep Residuals: Supports stable training for deeper architectures.
    """
    
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=4,
        dropout=0.0,
        use_layer_norm=True,
        use_relative_pos=True,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_relative_pos = use_relative_pos
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Initial node feature projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Edge feature MLP (for relative position vectors dx, dy)
        if use_relative_pos:
            self.edge_encoder = nn.Sequential(
                nn.Linear(2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
        
        # Message passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if use_relative_pos:
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.convs.append(GINEConv(mlp, edge_dim=hidden_channels))
            else:
                self.convs.append(GraphConv(hidden_channels, hidden_channels))
            
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Final output projection
        self.final_proj = nn.Linear(hidden_channels, out_channels)
        
        # Output LayerNorm
        if use_layer_norm:
            self.output_norm = nn.LayerNorm(out_channels)
        
        # Residual projection for final layer (hidden -> out)
        if hidden_channels != out_channels:
            self.res_proj = nn.Linear(hidden_channels, out_channels)
        else:
            self.res_proj = None
    
    def forward(self, x, edge_index, pos=None):
        """
        Args:
            x: Input node features [N, in_channels] (usually normalized pos)
            edge_index: Graph connectivity [2, E]
            pos: Optional positions for dX calculation (defaults to x[:, :2])
        
        Returns:
            Node embeddings [N, out_channels]
        """
        if pos is None:
            pos = x[:, :2]
        
        # 1. Edge Feature Preparation (relative positions)
        edge_attr = None
        if self.use_relative_pos:
            row, col = edge_index
            rel_pos = pos[col] - pos[row]  # [E, 2]
            edge_attr = self.edge_encoder(rel_pos)  # [E, hidden_channels]
        
        # 2. Initial Projection
        h = self.input_proj(x)
        
        # 3. Iterative Message Passing with Residuals
        for i, conv in enumerate(self.convs):
            identity = h
            
            # Message Passing
            if self.use_relative_pos:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            
            # LayerNorm (Pre-activation style)
            if self.use_layer_norm:
                h = self.norms[i](h)
            
            # Activation (GELU is smoother for stiff mechanics)
            h = F.gelu(h)
            
            # Dropout
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Residual connection
            h = h + identity
        
        # 4. Final Output Projection with Residual
        # Save pre-projection state for residual
        h_pre = h
        
        # Project to output dimension
        out = self.final_proj(h)
        
        # Residual connection (with projection if needed)
        if self.res_proj is not None:
            out = out + self.res_proj(h_pre)
        else:
            out = out + h_pre  # Only valid when hidden == out
        
        # Final LayerNorm
        if self.use_layer_norm:
            out = self.output_norm(out)
        
        return out