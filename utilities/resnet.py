import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

class GraphResNetBlock(nn.Module):
    """
    Graph-based ResNet block (Exact analog to pixel ResNetBlock).
    
    Structure (Post-Activation):
    x -> GAT -> Norm -> ReLU -> GAT -> Norm -> Add(Skip) -> ReLU
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Ensure divisibility
        if concat:
            assert out_channels % heads == 0, "Out channels must be divisible by heads"
            self.gat_out = out_channels // heads
        else:
            self.gat_out = out_channels

        # --- Layer 1 ---
        self.conv1 = GATConv(in_channels, self.gat_out,
                             heads=heads, concat=concat, dropout=dropout)
        # We use BatchNorm to match typical ResNet behavior (Pixel code implied batch stats via lack of instance norm)
        self.norm1 = BatchNorm(out_channels) 
        
        # --- Layer 2 ---
        self.conv2 = GATConv(out_channels, self.gat_out,
                            heads=heads, concat=concat, dropout=dropout)
        self.norm2 = BatchNorm(out_channels)

        # --- Skip Connection ---
        if in_channels != out_channels:
            # Linear is the Graph equivalent of 1x1 Conv
            self.skip_proj = nn.Linear(in_channels, out_channels)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x, edge_index):
        identity = x
        
        # Block body
        out = self.conv1(x, edge_index)
        out = self.norm1(out)
        out = F.relu(out) # Match pixel ReLU
        
        out = self.conv2(out, edge_index)
        out = self.norm2(out)
        
        # Residual Addition
        skip = self.skip_proj(identity)
        
        # Final Activation (Crucial for Post-Activation ResNet)
        return F.relu(out + skip)


class GraphResNet(nn.Module):
    """
    Graph-based ResNet (Matches pixel ResNet initialization).
    """
    def __init__(
        self,
        in_channels: int,
        block_dimensions: list,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Calculate first hidden dim (based on block_dimensions[0])
        first_dim = block_dimensions[0]
        if concat:
            assert first_dim % heads == 0
            first_gat_out = first_dim // heads
        else:
            first_gat_out = first_dim

        # --- Initial Double Conv (Matching Pixel Code) ---
        # Conv 1
        self.conv1 = GATConv(in_channels, first_gat_out,
                             heads=heads, concat=concat, dropout=dropout)
        self.norm1 = BatchNorm(first_dim)
        
        # Conv 2
        self.conv2 = GATConv(first_dim, first_gat_out,
                            heads=heads, concat=concat, dropout=dropout)
        self.norm2 = BatchNorm(first_dim)

        # --- Residual Blocks ---
        self.blocks = nn.ModuleList()
        for i, out_ch in enumerate(block_dimensions):
            # Determine input channels for this block
            # If i==0, input is the result of the initial convs (which is block_dimensions[0])
            # If i>0, input is the output of the previous block
            in_ch = block_dimensions[0] if i == 0 else block_dimensions[i-1]
            
            self.blocks.append(
                GraphResNetBlock(in_ch, out_ch, heads, concat, dropout)
            )

    def forward(self, x, edge_index):
        # Initial double convolution
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        
        # Residual Blocks
        for block in self.blocks:
            x = block(x, edge_index)
            
        return x