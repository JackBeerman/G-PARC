# integrator.py

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class IntegralGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=3, 
                 num_layers=4, heads=8, concat=True, dropout=0.3, use_residual=True):
        super(IntegralGNN, self).__init__()
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.out_channels = out_channels

        if use_residual:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            current_in = in_channels if i == 0 else hidden_channels * (heads if concat else 1)
            current_out = hidden_channels if i < num_layers - 1 else out_channels
            is_last_layer = (i == num_layers - 1)

            self.layers.append(nn.LayerNorm(current_in, eps=1e-6))#, eps=1e-6
            self.layers.append(
                GATConv(current_in, current_out, 
                        heads=1 if is_last_layer else heads,
                        concat=False if is_last_layer else concat, 
                        dropout=dropout)
            )

    def forward(self, x, edge_index):
        residual = self.residual_proj(x) if self.use_residual else None
        
        for i in range(self.num_layers):
            ln = self.layers[2*i]
            gnn = self.layers[2*i + 1]
            
            x_res = x
            x = ln(x)
            x = gnn(x, edge_index)

            if i < self.num_layers - 1:
                x = F.gelu(x)
                if x.shape == x_res.shape: # Add skip connections between layers
                    x = x + x_res

        if self.use_residual and residual is not None:
            x = x + residual
            
        return x
