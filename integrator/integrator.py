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


class ExplicitIntegralGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=3, 
                 num_layers=4, heads=8, concat=True, dropout=0.3, use_residual=True):
        super(ExplicitIntegralGNN, self).__init__()
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
            self.layers.append(nn.LayerNorm(current_in, eps=1e-6))
            self.layers.append(
                GATConv(current_in, current_out, 
                        heads=1 if is_last_layer else heads,
                        concat=False if is_last_layer else concat, 
                        dropout=dropout)
            )
    
    def forward(self, x, edge_index, delta_t=None):
        """
        Forward pass with optional direct delta_t multiplication
        
        Args:
            x: Input derivative features from the derivative solver
            edge_index: Graph connectivity
            delta_t: Time step size (scalar tensor or float) for direct integration
        """
        residual = self.residual_proj(x) if self.use_residual else None
        
        # Process through GNN layers
        for i in range(self.num_layers):
            ln = self.layers[2*i]
            gnn = self.layers[2*i + 1]
            
            x_res = x
            x = ln(x)
            x = gnn(x, edge_index)
            if i < self.num_layers - 1:
                x = F.gelu(x)
                if x.shape == x_res.shape:  # Add skip connections between layers
                    x = x + x_res
        
        if self.use_residual and residual is not None:
            x = x + residual
        
        # Apply delta_t directly for interpretable integration
        if delta_t is not None:
            # Convert delta_t to tensor if it's a scalar
            if isinstance(delta_t, (int, float)):
                delta_t = torch.tensor(delta_t, dtype=x.dtype, device=x.device)
            
            # Ensure delta_t is broadcastable with x
            if delta_t.dim() == 0:  # scalar
                integral = x * delta_t
            else:
                integral = x * delta_t.view(-1, 1)  # broadcast appropriately
        else:
            # Fallback: assume unit time step or let the network learn the scaling
            integral = x
            
        return integral