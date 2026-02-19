"""
MeshGraphNet implementation for physics simulations.

Architecture: Encoder-Processor-Decoder GNN (Pfaff et al. 2021)
Convention: Predicts DELTA (change in dynamic state per timestep)
Normalization: Data is pre-normalized (global-max), no runtime normalization needed

Supports elastoplastic, river, and shock tube datasets via configurable
input/output dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LayerNorm
from torch_geometric.nn import MessagePassing


class ProcessorLayer(MessagePassing):
    """Message passing layer with residual connections."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)

        # Edge MLP: [sender, receiver, edge] -> updated edge
        self.edge_mlp = Sequential(
            Linear(3 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels)
        )

        # Node MLP: [node, aggregated_messages] -> updated node
        self.node_mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels)
        )

    def forward(self, x, edge_index, edge_attr, size=None):
        out, updated_edges = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=size
        )
        updated_nodes = torch.cat([x, out], dim=1)
        updated_nodes = x + self.node_mlp(updated_nodes)
        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1)
        updated_edges = self.edge_mlp(updated_edges) + edge_attr
        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size=None):
        target_index = edge_index[0, :]
        if dim_size is None:
            dim_size = target_index.max().item() + 1 if target_index.numel() > 0 else 0
        out = torch.zeros((dim_size, updated_edges.size(-1)),
                          device=updated_edges.device, dtype=updated_edges.dtype)
        out.index_add_(0, target_index, updated_edges)
        return out, updated_edges


class MeshGraphNet(nn.Module):
    """
    MeshGraphNet encoder-processor-decoder.

    Args:
        input_dim_node: Number of node input features
        input_dim_edge: Number of edge features (typically 3: dx, dy, dist)
        hidden_dim: Latent dimension (default: 128)
        output_dim: Number of output features (predicted delta)
        num_layers: Number of message passing layers (default: 10)
    """

    def __init__(self, input_dim_node, input_dim_edge, hidden_dim=128,
                 output_dim=2, num_layers=10):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Encoder
        self.node_encoder = Sequential(
            Linear(input_dim_node, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim), LayerNorm(hidden_dim)
        )
        self.edge_encoder = Sequential(
            Linear(input_dim_edge, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim), LayerNorm(hidden_dim)
        )

        # Processor
        self.processor = nn.ModuleList([
            ProcessorLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Decoder
        self.decoder = Sequential(
            Linear(hidden_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, output_dim)
        )

    def compute_edge_features(self, pos, edge_index):
        """Compute edge features: [dx, dy, distance]."""
        row, col = edge_index
        rel_pos = pos[col] - pos[row]
        distance = torch.norm(rel_pos, dim=1, keepdim=True)
        return torch.cat([rel_pos, distance], dim=1)

    def forward(self, node_features, edge_features, edge_index):
        """
        Forward pass.

        Args:
            node_features: [N, input_dim_node] -- pre-normalized
            edge_features: [E, input_dim_edge] -- computed from positions
            edge_index: [2, E]

        Returns:
            predictions: [N, output_dim] -- predicted delta
        """
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)

        for layer in self.processor:
            x, edge_attr = layer(x, edge_index, edge_attr)

        return self.decoder(x)