#!/usr/bin/env python3
"""
GraphSAGE Baseline Models
=========================
Three GraphSAGE-based models for comparison against G-PARC variants:
  - ShocktubeGNN:     SAGEConv + residual + LayerNorm (no edge gating)
  - RiverGNN:         SAGEConv + edge gating + residual + LayerNorm
  - ElastoPlasticGNN: SAGEConv + edge gating + residual + LayerNorm

All models predict per-node deltas and use autoregressive rollout.

Usage:
    from models.graphsage_models import ShocktubeGNN, RiverGNN, ElastoPlasticGNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ==============================================================================
# SHARED UTILITIES
# ==============================================================================

def compute_edge_attr(pos, edge_index):
    """Compute edge features [dx, dy, distance] from node positions."""
    src, dst = edge_index
    dx = pos[dst, 0] - pos[src, 0]
    dy = pos[dst, 1] - pos[src, 1]
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
    return torch.stack([dx, dy, dist], dim=-1)


# ==============================================================================
# SHOCKTUBE
# ==============================================================================

class ShocktubeGNN(nn.Module):
    """
    GraphSAGE baseline for 1D shock tube simulations.
    
    Architecture: encoder → N×(SAGEConv + residual + LayerNorm) → decoder
    No edge gating (1D structured mesh doesn't benefit from it).
    
    Default config: in=6 (2 static + 4 dynamic), out=4, hidden=177, layers=8
    """

    def __init__(self, in_channels=6, out_channels=4, hidden_channels=177,
                 num_layers=8, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # Static/dynamic split for rollout
        self.num_static_feats = in_channels - out_channels
        self.num_dynamic_feats = out_channels

        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass: predicts per-node delta.
        
        Args:
            x: [N, in_channels] node features (static + dynamic concatenated)
            edge_index: [2, E] edge connectivity
            edge_attr: unused (kept for API compatibility)
        
        Returns:
            delta: [N, out_channels] predicted change
        """
        h = self.node_encoder(x)
        for i in range(self.num_layers):
            h_in = h
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h + h_in)
            if i < self.num_layers - 1:
                h = F.gelu(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
        return self.decoder(h)


# ==============================================================================
# RIVER
# ==============================================================================

class RiverGNN(nn.Module):
    """
    GraphSAGE + Edge Gating baseline for river (HEC-RAS) simulations.
    
    Architecture: encoder → N×(SAGEConv + edge gate + residual + LayerNorm) → decoder
    Edge gating modulates node messages using learned edge importance.
    
    Default config: in=13 (9 static + 4 dynamic), out=4, hidden=96, layers=6, edge_dim=3
    """

    def __init__(self, in_channels=13, out_channels=4, hidden_channels=96,
                 num_layers=6, edge_dim=3, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # Static/dynamic split for rollout
        self.num_static_feats = in_channels - out_channels
        self.num_dynamic_feats = out_channels

        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.edge_encoder = nn.Linear(edge_dim, hidden_channels)

        self.convs = nn.ModuleList()
        self.edge_gates = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.edge_gates.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Sigmoid(),
            ))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with optional edge gating.
        
        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge connectivity
            edge_attr: [E, edge_dim] edge features (dx, dy, dist). Optional.
        
        Returns:
            delta: [N, out_channels] predicted change
        """
        h = self.node_encoder(x)

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_h = self.edge_encoder(edge_attr)
        else:
            edge_h = None

        for i in range(self.num_layers):
            h_in = h
            h = self.convs[i](h, edge_index)

            if edge_h is not None:
                src, dst = edge_index
                gate = self.edge_gates[i](edge_h)
                gate_agg = torch.zeros_like(h)
                counts = torch.zeros(h.size(0), 1, device=h.device)
                gate_agg.index_add_(0, dst, gate)
                counts.index_add_(0, dst, torch.ones(dst.size(0), 1, device=h.device))
                counts = counts.clamp(min=1)
                gate_agg = gate_agg / counts
                h = h * gate_agg

            h = self.norms[i](h + h_in)
            if i < self.num_layers - 1:
                h = F.gelu(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)

        return self.decoder(h)

    def compute_edge_features(self, data):
        """Compute edge_attr from a PyG Data object."""
        pos = data.pos if hasattr(data, 'pos') and data.pos is not None else data.x[:, :2]
        return compute_edge_attr(pos, data.edge_index)


# ==============================================================================
# ELASTOPLASTIC
# ==============================================================================

class ElastoPlasticGNN(nn.Module):
    """
    GraphSAGE + Edge Gating baseline for elastoplastic simulations.
    
    Same architecture as RiverGNN with different default hyperparameters.
    
    Default config: in=4 (2 static + 2 dynamic), out=2, hidden=135, layers=8, edge_dim=3
    """

    def __init__(self, in_channels=4, out_channels=2, hidden_channels=135,
                 num_layers=8, edge_dim=3, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # Static/dynamic split for rollout
        self.num_static_feats = in_channels - out_channels
        self.num_dynamic_feats = out_channels

        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.edge_encoder = nn.Linear(edge_dim, hidden_channels)

        self.convs = nn.ModuleList()
        self.edge_gates = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.edge_gates.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Sigmoid(),
            ))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with optional edge gating.
        
        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge connectivity
            edge_attr: [E, edge_dim] edge features (dx, dy, dist). Optional.
        
        Returns:
            delta: [N, out_channels] predicted change
        """
        h = self.node_encoder(x)

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_h = self.edge_encoder(edge_attr)
        else:
            edge_h = None

        for i in range(self.num_layers):
            h_in = h
            h = self.convs[i](h, edge_index)

            if edge_h is not None:
                src, dst = edge_index
                gate = self.edge_gates[i](edge_h)
                gate_agg = torch.zeros_like(h)
                counts = torch.zeros(h.size(0), 1, device=h.device)
                gate_agg.index_add_(0, dst, gate)
                counts.index_add_(0, dst, torch.ones(dst.size(0), 1, device=h.device))
                counts = counts.clamp(min=1)
                gate_agg = gate_agg / counts
                h = h * gate_agg

            h = self.norms[i](h + h_in)
            if i < self.num_layers - 1:
                h = F.gelu(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)

        return self.decoder(h)

    def compute_edge_features(self, data):
        """Compute edge_attr from a PyG Data object."""
        pos = data.pos if hasattr(data, 'pos') and data.pos is not None else data.x[:, :2]
        return compute_edge_attr(pos, data.edge_index)


# ==============================================================================
# FACTORY + LOADING HELPERS
# ==============================================================================

# Registry of default configurations (matching trained checkpoints)
DEFAULT_CONFIGS = {
    'shocktube': dict(in_channels=6, out_channels=4, hidden_channels=177, num_layers=8),
    'river':     dict(in_channels=13, out_channels=4, hidden_channels=96, num_layers=6, edge_dim=3),
    'elasto':    dict(in_channels=4, out_channels=2, hidden_channels=135, num_layers=8, edge_dim=3),
}

MODEL_CLASSES = {
    'shocktube': ShocktubeGNN,
    'river':     RiverGNN,
    'elasto':    ElastoPlasticGNN,
}


def create_model(domain, **overrides):
    """
    Create a GraphSAGE model for a given domain.
    
    Args:
        domain: 'shocktube', 'river', or 'elasto'
        **overrides: any constructor args to override defaults
    
    Returns:
        model instance
    """
    if domain not in MODEL_CLASSES:
        raise ValueError(f"Unknown domain '{domain}'. Choose from: {list(MODEL_CLASSES.keys())}")
    
    config = {**DEFAULT_CONFIGS[domain], **overrides}
    return MODEL_CLASSES[domain](**config)


def load_model(domain, checkpoint_path, device='cpu', **overrides):
    """
    Create and load a GraphSAGE model from checkpoint.
    
    Args:
        domain: 'shocktube', 'river', or 'elasto'
        checkpoint_path: path to .pth checkpoint
        device: target device
        **overrides: constructor arg overrides
    
    Returns:
        loaded model in eval mode
    """
    model = create_model(domain, **overrides)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    
    model.to(device).eval()
    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    print(f"  ✓ GraphSAGE-{domain} loaded (epoch {epoch})")
    return model