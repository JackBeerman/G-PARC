"""
models.meshgraphkan
====================
PyG reimplementation of NVIDIA PhysicsNeMo MeshGraphKAN architecture.

Core building blocks shared across elastoplastic, shock tube, and river domains:
  - KolmogorovArnoldNetwork (KAN) — Fourier-coefficient node encoder
  - MeshGraphMLP — standard MLP with optional LayerNorm
  - MeshEdgeBlock — edge update: concat(src, dst, edge) -> MLP -> residual
  - MeshNodeBlock — node update: aggregate edges -> concat(node, agg) -> MLP -> residual
  - MeshGraphNetProcessor — interleaved [EdgeBlock, NodeBlock] × processor_size
  - MeshGraphKAN — full encode-process-decode model

Domain-specific rollout wrappers (which define how static/dynamic features are
split and how edge features are computed) remain in each eval/train script.

Usage:
    from models.meshgraphkan import MeshGraphKAN

    model = MeshGraphKAN(
        input_dim_nodes=4,   # static + dynamic (+ global for shock tube)
        input_dim_edges=3,   # [rel_x, rel_y, distance]
        output_dim=2,        # predicted delta or next-state
        processor_size=15,
        hidden_dim_processor=128,
    )
    out = model(node_features, edge_features, edge_index)
"""

import torch
import torch.nn as nn
import numpy as np

__all__ = [
    'KolmogorovArnoldNetwork',
    'MeshGraphMLP',
    'MeshEdgeBlock',
    'MeshNodeBlock',
    'MeshGraphNetProcessor',
    'MeshGraphKAN',
]


class KolmogorovArnoldNetwork(nn.Module):
    """
    Learnable Fourier-coefficient network (cos + sin) per output × input × harmonic.
    Exact reimplementation of physicsnemo.nn.KolmogorovArnoldNetwork.
    """

    def __init__(self, input_dim, output_dim, num_harmonics=5, add_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_harmonics = num_harmonics
        self.add_bias = add_bias

        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, output_dim, input_dim, num_harmonics)
            / (np.sqrt(input_dim) * np.sqrt(num_harmonics))
        )
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x_expanded = x.view(batch_size, self.input_dim, 1)
        k = torch.arange(1, self.num_harmonics + 1, device=x.device).view(1, 1, self.num_harmonics)
        cos_terms = torch.cos(k * x_expanded)
        sin_terms = torch.sin(k * x_expanded)
        y_cos = torch.einsum("bij,oij->bo", cos_terms, self.fourier_coeffs[0])
        y_sin = torch.einsum("bij,oij->bo", sin_terms, self.fourier_coeffs[1])
        y = y_cos + y_sin
        if self.add_bias:
            y = y + self.bias
        return y


class MeshGraphMLP(nn.Module):
    """
    Standard MLP with optional LayerNorm.
    Reimplements physicsnemo MeshGraphMLP.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=128, hidden_layers=2,
                 activation_fn=None, norm_type="LayerNorm"):
        super().__init__()
        if activation_fn is None:
            activation_fn = nn.ReLU()
        if hidden_layers is not None and hidden_layers > 0:
            layers = [nn.Linear(input_dim, hidden_dim), activation_fn]
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
            layers.append(nn.Linear(hidden_dim, output_dim))
            if norm_type is not None:
                layers.append(nn.LayerNorm(output_dim))
            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Identity()

    def forward(self, x):
        return self.model(x)


class MeshEdgeBlock(nn.Module):
    """Edge update: concat(src, dst, edge) -> MLP -> residual add."""

    def __init__(self, input_dim_node, input_dim_edge, output_dim_edge,
                 hidden_dim_edge, hidden_layers=2, activation_fn=None,
                 norm_type="LayerNorm"):
        super().__init__()
        if activation_fn is None:
            activation_fn = nn.ReLU()
        mlp_input_dim = 2 * input_dim_node + input_dim_edge
        self.edge_mlp = MeshGraphMLP(
            input_dim=mlp_input_dim, output_dim=output_dim_edge,
            hidden_dim=hidden_dim_edge, hidden_layers=hidden_layers,
            activation_fn=activation_fn, norm_type=norm_type,
        )

    def forward(self, edge_features, node_features, edge_index):
        src_feats = node_features[edge_index[0]]
        dst_feats = node_features[edge_index[1]]
        edge_input = torch.cat([src_feats, dst_feats, edge_features], dim=-1)
        return edge_features + self.edge_mlp(edge_input)


class MeshNodeBlock(nn.Module):
    """Node update: aggregate incoming edges -> concat(node, agg) -> MLP -> residual add."""

    def __init__(self, aggregation, input_dim_node, input_dim_edge, output_dim_node,
                 hidden_dim_node, hidden_layers=2, activation_fn=None,
                 norm_type="LayerNorm"):
        super().__init__()
        self.aggregation = aggregation
        if activation_fn is None:
            activation_fn = nn.ReLU()
        mlp_input_dim = input_dim_node + input_dim_edge
        self.node_mlp = MeshGraphMLP(
            input_dim=mlp_input_dim, output_dim=output_dim_node,
            hidden_dim=hidden_dim_node, hidden_layers=hidden_layers,
            activation_fn=activation_fn, norm_type=norm_type,
        )

    def forward(self, edge_features, node_features, edge_index):
        dst_nodes = edge_index[1]
        num_nodes = node_features.shape[0]

        agg = torch.zeros(num_nodes, edge_features.shape[1],
                          device=node_features.device, dtype=node_features.dtype)
        agg.index_add_(0, dst_nodes, edge_features)

        if self.aggregation == 'mean':
            cnt = torch.zeros(num_nodes, 1,
                              device=node_features.device, dtype=node_features.dtype)
            cnt.index_add_(0, dst_nodes, torch.ones(dst_nodes.shape[0], 1,
                                                     device=node_features.device,
                                                     dtype=node_features.dtype))
            agg = agg / (cnt + 1e-8)
        elif self.aggregation != 'sum':
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        node_input = torch.cat([node_features, agg], dim=-1)
        return node_features + self.node_mlp(node_input)


class MeshGraphNetProcessor(nn.Module):
    """Interleaved [EdgeBlock, NodeBlock] × processor_size."""

    def __init__(self, processor_size=15, input_dim_node=128, input_dim_edge=128,
                 hidden_layers_node=2, hidden_layers_edge=2, aggregation='sum',
                 activation_fn=None, norm_type="LayerNorm"):
        super().__init__()
        if activation_fn is None:
            activation_fn = nn.ReLU()
        layers = []
        for _ in range(processor_size):
            layers.append(MeshEdgeBlock(
                input_dim_node=input_dim_node, input_dim_edge=input_dim_edge,
                output_dim_edge=input_dim_edge, hidden_dim_edge=input_dim_edge,
                hidden_layers=hidden_layers_edge, activation_fn=activation_fn,
                norm_type=norm_type,
            ))
            layers.append(MeshNodeBlock(
                aggregation=aggregation, input_dim_node=input_dim_node,
                input_dim_edge=input_dim_edge, output_dim_node=input_dim_node,
                hidden_dim_node=input_dim_node, hidden_layers=hidden_layers_node,
                activation_fn=activation_fn, norm_type=norm_type,
            ))
        self.processor_layers = nn.ModuleList(layers)

    def forward(self, node_features, edge_features, edge_index):
        for i in range(0, len(self.processor_layers), 2):
            edge_features = self.processor_layers[i](edge_features, node_features, edge_index)
            node_features = self.processor_layers[i + 1](edge_features, node_features, edge_index)
        return node_features


class MeshGraphKAN(nn.Module):
    """
    Full MeshGraphKAN: KAN node encoder, MLP edge encoder, processor, MLP decoder.
    PyG reimplementation of NVIDIA PhysicsNeMo MeshGraphKAN.

    Args:
        input_dim_nodes: Total node feature dim (static + dynamic [+ global]).
        input_dim_edges: Edge feature dim (typically 3: rel_x, rel_y, dist).
        output_dim: Prediction dim (delta or next-state).
        processor_size: Number of message-passing iterations.
        hidden_dim_processor: Latent dimension throughout encoder/processor/decoder.
    """

    def __init__(self, input_dim_nodes=4, input_dim_edges=3, output_dim=2,
                 processor_size=15, mlp_activation_fn='relu',
                 num_layers_node_processor=2, num_layers_edge_processor=2,
                 hidden_dim_processor=128, hidden_dim_node_encoder=128,
                 hidden_dim_edge_encoder=128, num_layers_edge_encoder=2,
                 hidden_dim_node_decoder=128, num_layers_node_decoder=2,
                 aggregation='sum', num_harmonics=5):
        super().__init__()
        self.input_dim_nodes = input_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.output_dim = output_dim

        activation_fn = nn.SiLU() if mlp_activation_fn == 'silu' else nn.ReLU()

        self.edge_encoder = MeshGraphMLP(
            input_dim=input_dim_edges, output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder, hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn, norm_type="LayerNorm",
        )
        self.node_encoder = KolmogorovArnoldNetwork(
            input_dim=input_dim_nodes, output_dim=hidden_dim_processor,
            num_harmonics=num_harmonics, add_bias=True,
        )
        self.processor = MeshGraphNetProcessor(
            processor_size=processor_size,
            input_dim_node=hidden_dim_processor, input_dim_edge=hidden_dim_processor,
            hidden_layers_node=num_layers_node_processor,
            hidden_layers_edge=num_layers_edge_processor,
            aggregation=aggregation, activation_fn=activation_fn,
            norm_type="LayerNorm",
        )
        self.node_decoder = MeshGraphMLP(
            input_dim=hidden_dim_processor, output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder, hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn, norm_type=None,
        )

    def forward(self, node_features, edge_features, edge_index):
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(node_features, edge_features, edge_index)
        return self.node_decoder(x)


# =========================================================================
# DOMAIN-SPECIFIC ROLLOUT WRAPPERS
# =========================================================================

class MeshGraphKANElastoRollout(nn.Module):
    """
    Rollout wrapper for elastoplastic domain.
    Convention: model predicts delta, accumulate current + delta.
    Data: x = [static(2), dynamic(2)], y = next-step full state.
    """

    def __init__(self, model, num_static_feats=2, num_dynamic_feats=2):
        super().__init__()
        self.model = model
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats

    def compute_edge_features(self, pos, edge_index):
        """Compute edge features: [rel_pos, distance]."""
        rel_pos = pos[edge_index[1]] - pos[edge_index[0]]
        distance = torch.norm(rel_pos, dim=1, keepdim=True)
        return torch.cat([rel_pos, distance], dim=1)


class MeshGraphKANShocktubeRollout(nn.Module):
    """
    Rollout wrapper for shock tube domain.
    Handles skip_dynamic_indices and global parameter conditioning.
    Convention: model predicts next-state directly, rollout feeds back.
    Data: x = [static(2), raw_dynamic(4), ...], global_params = [P, rho, dt].
    """

    def __init__(self, model, num_static_feats=2, num_dynamic_feats=3,
                 skip_dynamic_indices=None, global_param_dim=3):
        super().__init__()
        self.model = model
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.skip_dynamic_indices = skip_dynamic_indices or [2]
        self.global_param_dim = global_param_dim
        self.raw_dynamic_feats = num_dynamic_feats + len(self.skip_dynamic_indices)
        self.keep_indices = [i for i in range(self.raw_dynamic_feats)
                             if i not in self.skip_dynamic_indices]

    def _extract_dynamic(self, x):
        """Extract dynamic features, skipping specified indices."""
        sf = self.num_static_feats
        raw = x[:, sf:sf + self.raw_dynamic_feats]
        return raw[:, self.keep_indices]

    def _extract_global_params(self, data):
        """Extract global parameters [pressure, density, delta_t] and broadcast to all nodes."""
        parts = []
        for attr_pairs in [('global_pressure', 'pressure'),
                           ('global_density', 'density_param'),
                           ('global_delta_t', 'delta_t')]:
            val = None
            for a in attr_pairs:
                if hasattr(data, a):
                    val = getattr(data, a)
                    break
            if val is None:
                val = torch.zeros(1, device=data.x.device)
            parts.append(val.view(1))
        gp = torch.cat(parts)
        return gp.unsqueeze(0).expand(data.x.size(0), -1)

    def _apply_skip_to_target(self, y):
        """Apply same skip indices to target tensor."""
        return y[:, self.keep_indices]

    def compute_edge_features(self, pos, edge_index):
        """Compute edge features: [rel_pos, distance]."""
        rel_pos = pos[edge_index[1]] - pos[edge_index[0]]
        distance = torch.norm(rel_pos, dim=1, keepdim=True)
        return torch.cat([rel_pos, distance], dim=1)


class MeshGraphKANRiverRollout(nn.Module):
    """
    Rollout wrapper for river (flood) domain.
    Convention: model predicts delta, accumulate current + delta.
    Data: x = [static(9), dynamic(4)], y = next-step dynamic [4].
    """

    def __init__(self, model, num_static_feats=9, num_dynamic_feats=4):
        super().__init__()
        self.model = model
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats

    def compute_edge_features(self, pos, edge_index):
        """Compute edge features: [rel_pos, distance]."""
        rel_pos = pos[edge_index[1]] - pos[edge_index[0]]
        distance = torch.norm(rel_pos, dim=1, keepdim=True)
        return torch.cat([rel_pos, distance], dim=1)


# =========================================================================
# MODEL BUILDER
# =========================================================================

def build_meshgraphkan(config, device='cpu', domain='elasto'):
    """
    Build MeshGraphKAN with domain-appropriate rollout wrapper from config dict.

    Args:
        config: dict from config.json or CLI args.
        device: torch device.
        domain: 'elasto', 'shocktube', or 'river'.

    Returns:
        Wrapped model on device.
    """
    num_static = config.get('num_static_feats', 2)
    num_dynamic = config.get('num_dynamic_feats', 2)
    skip_indices = config.get('skip_dynamic_indices', [2])
    global_dim = config.get('global_param_dim', 3)

    if domain == 'shocktube':
        input_dim_nodes = config.get('input_dim_nodes', num_static + num_dynamic + global_dim)
    else:
        input_dim_nodes = config.get('input_dim_nodes', num_static + num_dynamic)

    kan_model = MeshGraphKAN(
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=config.get('input_dim_edges', 3),
        output_dim=num_dynamic,
        processor_size=config.get('processor_size', 15),
        mlp_activation_fn=config.get('mlp_activation', 'relu'),
        num_layers_node_processor=config.get('num_layers_node_processor', 2),
        num_layers_edge_processor=config.get('num_layers_edge_processor', 2),
        hidden_dim_processor=config.get('hidden_dim', 128),
        hidden_dim_node_encoder=config.get('hidden_dim', 128),
        hidden_dim_edge_encoder=config.get('hidden_dim', 128),
        num_layers_edge_encoder=config.get('num_layers_edge_encoder', 2),
        hidden_dim_node_decoder=config.get('hidden_dim', 128),
        num_layers_node_decoder=config.get('num_layers_node_decoder', 2),
        aggregation=config.get('aggregation', 'sum'),
        num_harmonics=config.get('num_harmonics', 5),
    )

    if domain == 'elasto':
        model = MeshGraphKANElastoRollout(kan_model, num_static, num_dynamic)
    elif domain == 'shocktube':
        model = MeshGraphKANShocktubeRollout(
            kan_model, num_static, num_dynamic, skip_indices, global_dim)
    elif domain == 'river':
        model = MeshGraphKANRiverRollout(kan_model, num_static, num_dynamic)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return model.to(device)