"""
MeshGraphNets implementation adapted for elastoplastic dynamics simulations.

Key adaptations from original MeshGraphNets:
- Computes edge features from node positions (no pre-computed edge_attr)
- Removes node type one-hot encoding and associated loss masking
- Handles 4 node features: [x_pos, y_pos, U_x, U_y]
- Predicts 2 targets: [ΔU_x, ΔU_y]
- Uses mesh_id for operator caching compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LayerNorm
from torch_geometric.nn import MessagePassing


class ProcessorLayer(MessagePassing):
    """
    Message passing layer for MeshGraphNets.
    
    Updates both edge and node embeddings through graph message passing.
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        
        # Edge processor: takes [sender_emb, receiver_emb, edge_emb]
        self.edge_mlp = Sequential(
            Linear(3 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels)
        )
        
        # Node processor: takes [node_emb, aggregated_messages]
        self.node_mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters for stacked MLP layers"""
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()
        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()
    
    def forward(self, x, edge_index, edge_attr, size=None):
        """
        Forward pass with message passing.
        
        Args:
            x: Node embeddings [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge embeddings [num_edges, in_channels]
            size: Optional size for bipartite graphs
            
        Returns:
            updated_nodes: Updated node embeddings
            updated_edges: Updated edge embeddings
        """
        # Propagate messages and update edges
        out, updated_edges = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=size
        )
        
        # Aggregate with self-connection
        updated_nodes = torch.cat([x, out], dim=1)
        
        # Apply MLP with residual connection
        updated_nodes = x + self.node_mlp(updated_nodes)
        
        return updated_nodes, updated_edges
    
    def message(self, x_i, x_j, edge_attr):
        """
        Construct messages from sender to receiver.
        
        Args:
            x_i: Sender node embeddings [num_edges, in_channels]
            x_j: Receiver node embeddings [num_edges, in_channels]
            edge_attr: Edge embeddings [num_edges, in_channels]
            
        Returns:
            updated_edges: Updated edge embeddings with residual
        """
        # Concatenate sender, receiver, and edge embeddings
        updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        # Apply edge MLP with residual connection
        updated_edges = self.edge_mlp(updated_edges) + edge_attr
        
        return updated_edges
    
    #def aggregate(self, updated_edges, edge_index, dim_size=None):
    #    """
    #    Aggregate messages from neighboring nodes.
    #    
    #    Args:
    #        updated_edges: Updated edge embeddings
    #        edge_index: Edge connectivity
    #        dim_size: Number of nodes
    #        
    #    Returns:
    #        out: Aggregated messages per node
    #        updated_edges: Updated edge embeddings (passed through)
    #    """
    #    node_dim = 0  # Dimension along which to aggregate
    #    
    #    # Sum aggregation over incoming edges
    #    out = torch_scatter.scatter(
    #        updated_edges, 
    #        edge_index[0, :], 
    #        dim=node_dim, 
    #        reduce='sum'
    #    )
    #    
    #    return out, updated_edges
    def aggregate(self, updated_edges, edge_index, dim_size=None):
        """
        Aggregate messages from neighboring nodes using native PyTorch.
        """
        # edge_index[0, :] are the destination/target indices
        target_index = edge_index[0, :]
        
        # Get the number of nodes (dim_size) if not provided
        if dim_size is None:
            dim_size = target_index.max().item() + 1 if target_index.numel() > 0 else 0
    
        # Initialize the output tensor on the same device as the edges
        out = torch.zeros((dim_size, updated_edges.size(-1)), 
                          device=updated_edges.device, 
                          dtype=updated_edges.dtype)
    
        # Use index_add_ to perform the sum aggregation
        # This is the native version of scatter(reduce='sum')
        out.index_add_(0, target_index, updated_edges)
        
        return out, updated_edges


class MeshGraphNet(torch.nn.Module):
    """
    MeshGraphNets model for elastoplastic dynamics.
    
    Architecture:
    1. Encoder: Separate MLPs for nodes and edges
    2. Processor: Stack of message passing layers
    3. Decoder: MLP for node predictions
    
    Args:
        input_dim_node: Number of node features (4: x_pos, y_pos, U_x, U_y)
        input_dim_edge: Number of edge features (computed from positions)
        hidden_dim: Hidden dimension for all MLPs (default: 128)
        output_dim: Number of output features (2: ΔU_x, ΔU_y)
        num_layers: Number of message passing layers (default: 10)
    """
    
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim=128, 
                 output_dim=2, num_layers=10):
        super(MeshGraphNet, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Encoder: Convert raw features to latent embeddings
        self.node_encoder = Sequential(
            Linear(input_dim_node, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim)
        )
        
        self.edge_encoder = Sequential(
            Linear(input_dim_edge, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim)
        )
        
        # Processor: Stack of message passing layers
        self.processor = nn.ModuleList([
            ProcessorLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Decoder: Convert node embeddings to predictions
        self.decoder = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim)
        )
    
    def _compute_edge_features(self, pos, edge_index):
        """
        Compute edge features from node positions.
        
        For elastoplastic simulations, edge features are:
        - Relative position: (pos_j - pos_i)
        - Distance: ||pos_j - pos_i||
        
        Args:
            pos: Node positions [num_nodes, 2]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            edge_attr: Edge features [num_edges, 3] (dx, dy, distance)
        """
        # Get sender and receiver positions
        row, col = edge_index
        pos_i = pos[row]  # Sender positions
        pos_j = pos[col]  # Receiver positions
        
        # Compute relative positions
        rel_pos = pos_j - pos_i  # [num_edges, 2]
        
        # Compute distances
        distance = torch.norm(rel_pos, dim=1, keepdim=True)  # [num_edges, 1]
        
        # Concatenate: [dx, dy, distance]
        edge_attr = torch.cat([rel_pos, distance], dim=1)
        
        return edge_attr
    
    def forward(self, data, mean_vec_x=None, std_vec_x=None, 
                mean_vec_edge=None, std_vec_edge=None):
        """
        Forward pass through MeshGraphNets.
        
        Args:
            data: PyG Data object with attributes:
                - x: Node features [num_nodes, 4]
                - pos: Node positions [num_nodes, 2]
                - edge_index: Edge connectivity [2, num_edges]
            mean_vec_x: Mean for node feature normalization (optional)
            std_vec_x: Std for node feature normalization (optional)
            mean_vec_edge: Mean for edge feature normalization (optional)
            std_vec_edge: Std for edge feature normalization (optional)
            
        Returns:
            predictions: Predicted displacement increments [num_nodes, 2]
        """
        x = data.x
        pos = data.pos
        edge_index = data.edge_index
        
        # Compute edge features from positions
        edge_attr = self._compute_edge_features(pos, edge_index)
        
        # Normalize features if statistics provided
        if mean_vec_x is not None and std_vec_x is not None:
            x = (x - mean_vec_x) / std_vec_x
        
        if mean_vec_edge is not None and std_vec_edge is not None:
            edge_attr = (edge_attr - mean_vec_edge) / std_vec_edge
        
        # Step 1: Encode node and edge features into latent embeddings
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Step 2: Message passing through processor layers
        for processor_layer in self.processor:
            x, edge_attr = processor_layer(x, edge_index, edge_attr)
        
        # Step 3: Decode node embeddings to predictions
        predictions = self.decoder(x)
        
        return predictions
    
    def loss(self, pred, targets, mean_vec_y=None, std_vec_y=None):
        """
        Compute L2 loss between predictions and targets.
        
        Args:
            pred: Predicted displacement increments [num_nodes, 2]
            targets: Target displacement increments [num_nodes, 2]
            mean_vec_y: Mean for target normalization (optional)
            std_vec_y: Std for target normalization (optional)
            
        Returns:
            loss: Root mean squared error
        """
        # Normalize targets if statistics provided
        if mean_vec_y is not None and std_vec_y is not None:
            targets = (targets - mean_vec_y) / std_vec_y
        
        # Compute squared error
        error = torch.sum((targets - pred) ** 2, dim=1)
        
        # Return RMSE
        loss = torch.sqrt(torch.mean(error))
        
        return loss


def normalize(to_normalize, mean_vec, std_vec):
    """Normalize tensor using mean and standard deviation."""
    return (to_normalize - mean_vec) / std_vec


def unnormalize(to_unnormalize, mean_vec, std_vec):
    """Unnormalize tensor using mean and standard deviation."""
    return to_unnormalize * std_vec + mean_vec


def compute_stats(dataset, max_samples=None):
    """
    Compute normalization statistics from dataset.
    
    Args:
        dataset: ElastoPlasticDataset or list of sequences
        max_samples: Maximum number of samples to use for statistics
        
    Returns:
        stats_dict: Dictionary with mean and std for x, edge_attr, and y
    """
    all_x = []
    all_edge_attr = []
    all_y = []
    
    print("Computing normalization statistics...")
    
    sample_count = 0
    for sequence in dataset:
        for data in sequence:
            all_x.append(data.x)
            all_y.append(data.y)
            
            # Compute edge attributes
            pos = data.pos
            edge_index = data.edge_index
            row, col = edge_index
            rel_pos = pos[col] - pos[row]
            distance = torch.norm(rel_pos, dim=1, keepdim=True)
            edge_attr = torch.cat([rel_pos, distance], dim=1)
            all_edge_attr.append(edge_attr)
            
            sample_count += 1
            if max_samples is not None and sample_count >= max_samples:
                break
        
        if max_samples is not None and sample_count >= max_samples:
            break
    
    # Concatenate all features
    all_x = torch.cat(all_x, dim=0)
    all_edge_attr = torch.cat(all_edge_attr, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    # Compute statistics
    mean_vec_x = all_x.mean(dim=0)
    std_vec_x = torch.maximum(all_x.std(dim=0), torch.tensor(1e-8))
    
    mean_vec_edge = all_edge_attr.mean(dim=0)
    std_vec_edge = torch.maximum(all_edge_attr.std(dim=0), torch.tensor(1e-8))
    
    mean_vec_y = all_y.mean(dim=0)
    std_vec_y = torch.maximum(all_y.std(dim=0), torch.tensor(1e-8))
    
    stats_dict = {
        'mean_vec_x': mean_vec_x,
        'std_vec_x': std_vec_x,
        'mean_vec_edge': mean_vec_edge,
        'std_vec_edge': std_vec_edge,
        'mean_vec_y': mean_vec_y,
        'std_vec_y': std_vec_y
    }
    
    print(f"Statistics computed from {sample_count} samples")
    print(f"  Node features: mean={mean_vec_x}, std={std_vec_x}")
    print(f"  Edge features: mean={mean_vec_edge}, std={std_vec_edge}")
    print(f"  Targets: mean={mean_vec_y}, std={std_vec_y}")
    
    return stats_dict


if __name__ == "__main__":
    # Example usage
    print("MeshGraphNets for Elastoplastic Dynamics")
    print("=" * 60)
    
    # Create dummy data for testing
    num_nodes = 100
    num_edges = 500
    
    from torch_geometric.data import Data
    
    data = Data(
        x=torch.randn(num_nodes, 4),  # [x_pos, y_pos, U_x, U_y]
        pos=torch.randn(num_nodes, 2),  # [x, y]
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        y=torch.randn(num_nodes, 2)  # [ΔU_x, ΔU_y]
    )
    
    # Create model
    model = MeshGraphNet(
        input_dim_node=4,
        input_dim_edge=3,  # [dx, dy, distance]
        hidden_dim=128,
        output_dim=2,
        num_layers=10
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Forward pass
    predictions = model(data)
    print(f"Predictions shape: {predictions.shape}")
    
    # Compute loss
    loss = model.loss(predictions, data.y)
    print(f"Loss: {loss.item():.4f}")