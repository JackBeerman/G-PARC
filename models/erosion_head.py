"""
Erosion Head for G-PARCv2
=========================
Lightweight element-level erosion classifier that operates on
internal node features from the differentiator.

Architecture:
  node features [N, D] + prev erosion [N, 1] → pool to elements [M, D+1] → MLP → P(erosion) [M, 1]

The previous erosion state is fed back autoregressively,
consistent with TF=0.0 displacement training strategy.

Pool strategy: mean of 3 corner nodes (simple, works for triangles)

Usage:
  erosion_head = ErosionHead(in_features=136, hidden_dim=64)
  logits = erosion_head(node_features, elements, prev_erosion_nodes)  # [M, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for extreme class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    With gamma=2, well-classified examples (p_t > 0.5) are
    down-weighted, focusing learning on hard examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [M, 1] raw logits (before sigmoid)
            targets: [M] or [M, 1] binary targets (1 = eroded)
        """
        targets = targets.float().view_as(logits)
        
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)

        # Alpha weighting: alpha for positive class, (1-alpha) for negative
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = (focal_weight * bce).mean()
        
        return loss


def element_erosion_to_node(erosion_mask_elem, elements, num_nodes):
    """
    Convert element-level erosion mask to node-level.
    A node is marked eroded if ANY of its elements are eroded.
    
    Args:
        erosion_mask_elem: [M] boolean or float — True/1.0 = eroded
        elements: [M, 3] element connectivity (long tensor)
        num_nodes: Total number of nodes
        
    Returns:
        erosion_node: [N, 1] float — 1.0 if node touches eroded element
    """
    erosion_node = torch.zeros(num_nodes, 1, device=elements.device)
    
    eroded_elems = erosion_mask_elem.bool() if erosion_mask_elem.dtype != torch.bool else erosion_mask_elem
    
    if eroded_elems.any():
        eroded_node_indices = elements[eroded_elems].reshape(-1)
        erosion_node[eroded_node_indices] = 1.0
    
    return erosion_node


def get_gt_erosion_targets(data, num_elements, threshold=0.5):
    """
    Extract ground truth erosion targets from data.x_element.
    
    Args:
        data: PyG Data object with .x_element attribute
        num_elements: Expected number of elements
        threshold: Values below this are considered eroded
        
    Returns:
        targets: [M] float tensor — 1.0 = eroded, 0.0 = valid
    """
    if hasattr(data, 'x_element') and data.x_element is not None:
        x_elem = data.x_element.flatten()
        # x_element: 1.0 = valid, 0.0 = eroded
        # We want: 1.0 = eroded (positive class for focal loss)
        return (x_elem < threshold).float()
    return torch.zeros(num_elements, device=data.x.device)


class ErosionHead(nn.Module):
    """
    Element-level erosion classifier.
    
    Takes per-node features (from differentiator cache) plus
    previous erosion state, pools to per-element, classifies.
    """
    def __init__(self, in_features, hidden_dim=64, num_layers=2, dropout=0.1):
        """
        Args:
            in_features: Dimension of per-node input features
                         (e.g., 128 resnet + 7 physics + 1 prev_erosion = 136)
            hidden_dim: Hidden dimension of MLP
            num_layers: Number of MLP layers (2 or 3)
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer (1 logit for binary classification)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, node_features, elements, prev_erosion_nodes=None):
        """
        Args:
            node_features: [N, D] per-node features from differentiator cache
            elements: [M, 3] element connectivity (triangle node indices, long tensor)
            prev_erosion_nodes: [N, 1] node-level erosion state from previous timestep
                                (None → zeros, i.e., no prior erosion)
            
        Returns:
            logits: [M, 1] erosion logits (apply sigmoid for probabilities)
        """
        # Append previous erosion state
        if prev_erosion_nodes is not None:
            node_features = torch.cat([node_features, prev_erosion_nodes], dim=1)
        
        # Pool: mean of 3 corner node features → [M, D+1]
        elem_features = node_features[elements].mean(dim=1)
        
        # Classify
        logits = self.mlp(elem_features)
        
        return logits