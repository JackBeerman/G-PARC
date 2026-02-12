#!/usr/bin/env python3
"""
Training Script for MeshGraphKAN Elastoplastic Model - DISPLACEMENT ONLY
=========================================================================
PyG reimplementation of NVIDIA PhysicsNeMo MeshGraphKAN architecture.

Architecture faithfully follows NVIDIA's source (verified against repo):
  - meshgraphkan.py: Inherits MeshGraphNet, replaces node_encoder with KAN
  - meshgraphnet.py: Encoder-Processor-Decoder with MeshGraphMLP
  - mesh_graph_mlp.py: Linear->Act->...->Linear->LayerNorm
  - kan.py: KolmogorovArnoldNetwork with learnable Fourier coefficients
  - mesh_edge_block.py / mesh_node_block.py: Message passing with residuals

Reimplemented in PyG due to DGL incompatibility with PyTorch >=2.5.
See: https://github.com/NVIDIA/physicsnemo/tree/main/physicsnemo/models/meshgraphnet/

Training pipeline mirrors G-PARC:
  - Scheduled Sampling for robust rollout training
  - Global max normalization support
  - Erosion masking for element deletion
  - Same checkpoint/resume logic

References:
  - Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks" (2021)
  - Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024)
  - Peng et al., "Interpretable physics-informed GNNs for flood forecasting" (2024)
  - NVIDIA PhysicsNeMo: https://github.com/NVIDIA/physicsnemo (Apache 2.0)
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.ElastoPlasticDataset import ElastoPlasticDataset, get_simulation_ids


# =========================================================================
# FAITHFUL PyG REIMPLEMENTATION OF NVIDIA MeshGraphKAN
# =========================================================================
# Verified against NVIDIA source files:
#   physicsnemo/nn/module/kan.py
#   physicsnemo/nn/module/gnn_layers/mesh_graph_mlp.py
#   physicsnemo/nn/module/gnn_layers/mesh_edge_block.py
#   physicsnemo/nn/module/gnn_layers/mesh_node_block.py
#   physicsnemo/models/meshgraphnet/meshgraphnet.py
#   physicsnemo/models/meshgraphnet/meshgraphkan.py
# =========================================================================


class KolmogorovArnoldNetwork(nn.Module):
    """
    Exact reimplementation of physicsnemo.nn.KolmogorovArnoldNetwork

    Uses learnable Fourier coefficients (cos + sin) per output × input × harmonic.
    NOT a simple linear projection over concatenated features.

    fourier_coeffs shape: [2, output_dim, input_dim, num_harmonics]
        [0] = cosine coefficients
        [1] = sine coefficients

    Forward: einsum contraction over input_dim and num_harmonics dimensions.
    Initialization: scaled by 1 / (sqrt(input_dim) * sqrt(num_harmonics))
    """

    def __init__(self, input_dim, output_dim, num_harmonics=5, add_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_harmonics = num_harmonics
        self.add_bias = add_bias

        # Learnable Fourier coefficients: [2, output_dim, input_dim, num_harmonics]
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, output_dim, input_dim, num_harmonics)
            / (np.sqrt(input_dim) * np.sqrt(num_harmonics))
        )

        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            [batch_size, output_dim]
        """
        batch_size = x.size(0)

        # Reshape for harmonic multiplication: [batch, input_dim, 1]
        x_expanded = x.view(batch_size, self.input_dim, 1)

        # Harmonic multipliers k = 1..num_harmonics: [1, 1, num_harmonics]
        k = torch.arange(1, self.num_harmonics + 1, device=x.device).view(
            1, 1, self.num_harmonics
        )

        # Compute cos/sin: [batch, input_dim, num_harmonics]
        cos_terms = torch.cos(k * x_expanded)
        sin_terms = torch.sin(k * x_expanded)

        # Fourier expansion via einsum: contract over input_dim (i) and harmonics (j)
        # fourier_coeffs[0]: [output_dim, input_dim, num_harmonics]
        # cos_terms: [batch, input_dim, num_harmonics]
        # result: [batch, output_dim]
        y_cos = torch.einsum("bij,oij->bo", cos_terms, self.fourier_coeffs[0])
        y_sin = torch.einsum("bij,oij->bo", sin_terms, self.fourier_coeffs[1])

        y = y_cos + y_sin

        if self.add_bias:
            y = y + self.bias

        return y


class MeshGraphMLP(nn.Module):
    """
    Reimplements physicsnemo.nn.module.gnn_layers.mesh_graph_mlp.MeshGraphMLP

    Structure (hidden_layers=2, NVIDIA default):
        Linear(in, hidden) -> Act -> Linear(hidden, hidden) -> Act -> Linear(hidden, out) -> [LayerNorm]

    norm_type=None for decoder (no LayerNorm).
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=128,
        hidden_layers=2,
        activation_fn=None,
        norm_type="LayerNorm",
    ):
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
    """
    Reimplements physicsnemo MeshEdgeBlock.
    Edge update: concat(src_node, dst_node, edge) -> MLP -> residual add.
    """

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
    """
    Reimplements physicsnemo MeshNodeBlock.
    Node update: aggregate edges -> concat(node, agg) -> MLP -> residual add.
    """

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

        if self.aggregation == 'sum':
            agg = torch.zeros(num_nodes, edge_features.shape[1],
                              device=node_features.device, dtype=node_features.dtype)
            agg.index_add_(0, dst_nodes, edge_features)
        elif self.aggregation == 'mean':
            agg = torch.zeros(num_nodes, edge_features.shape[1],
                              device=node_features.device, dtype=node_features.dtype)
            cnt = torch.zeros(num_nodes, 1,
                              device=node_features.device, dtype=node_features.dtype)
            agg.index_add_(0, dst_nodes, edge_features)
            cnt.index_add_(0, dst_nodes, torch.ones(dst_nodes.shape[0], 1,
                                                     device=node_features.device,
                                                     dtype=node_features.dtype))
            agg = agg / (cnt + 1e-8)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        node_input = torch.cat([node_features, agg], dim=-1)
        return node_features + self.node_mlp(node_input)


class MeshGraphNetProcessor(nn.Module):
    """
    Reimplements physicsnemo MeshGraphNetProcessor.
    Interleaved [EdgeBlock, NodeBlock] × processor_size.
    """

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
    PyG reimplementation of NVIDIA PhysicsNeMo MeshGraphKAN.

    Architecture:
        edge_encoder: MeshGraphMLP (with LayerNorm)
        node_encoder: KolmogorovArnoldNetwork (learnable Fourier coefficients)
        processor: MeshGraphNetProcessor (interleaved edge/node blocks)
        node_decoder: MeshGraphMLP (NO LayerNorm)
    """

    def __init__(
        self,
        input_dim_nodes: int = 4,
        input_dim_edges: int = 3,
        output_dim: int = 2,
        processor_size: int = 15,
        mlp_activation_fn: str = 'relu',
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: int = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int = 2,
        aggregation: str = 'sum',
        num_harmonics: int = 5,
    ):
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

        # KAN node encoder (replaces MLP in standard MeshGraphNet)
        self.node_encoder = KolmogorovArnoldNetwork(
            input_dim=input_dim_nodes, output_dim=hidden_dim_processor,
            num_harmonics=num_harmonics, add_bias=True,
        )

        self.processor = MeshGraphNetProcessor(
            processor_size=processor_size,
            input_dim_node=hidden_dim_processor, input_dim_edge=hidden_dim_processor,
            hidden_layers_node=num_layers_node_processor,
            hidden_layers_edge=num_layers_edge_processor,
            aggregation=aggregation, activation_fn=activation_fn, norm_type="LayerNorm",
        )

        # Decoder: no LayerNorm (norm_type=None)
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
# ROLLOUT WRAPPER (matches G-PARC sequence interface)
# =========================================================================

class MeshGraphKANRollout(nn.Module):
    """Autoregressive rollout wrapper matching G-PARC's forward() interface."""

    def __init__(self, model, num_static_feats=2, num_dynamic_feats=2):
        super().__init__()
        self.model = model
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats

    def compute_edge_features(self, data):
        """Edge features: (dx, dy, distance) — standard MGN edge encoding."""
        edge_index = data.edge_index
        pos = data.pos if hasattr(data, 'pos') and data.pos is not None else data.x[:, :2]
        src_pos = pos[edge_index[0]]
        dst_pos = pos[edge_index[1]]
        rel_pos = dst_pos - src_pos
        distance = torch.norm(rel_pos, dim=1, keepdim=True)
        return torch.cat([rel_pos, distance], dim=1)

    def forward(self, sequence, dt=1.0, teacher_forcing_ratio=0.0):
        predictions = []
        sf = self.num_static_feats
        df = self.num_dynamic_feats

        current_dynamic = sequence[0].x[:, sf:sf + df].clone()

        for t, data in enumerate(sequence):
            static_feats = data.x[:, :sf]
            node_features = torch.cat([static_feats, current_dynamic], dim=-1)
            edge_features = self.compute_edge_features(data)

            delta = self.model(node_features, edge_features, data.edge_index)
            predictions.append(delta)

            if t < len(sequence) - 1:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    current_dynamic = data.y.clone()
                else:
                    current_dynamic = current_dynamic + delta.detach()

        return predictions


# =========================================================================
# UTILITIES (mirrored from G-PARC)
# =========================================================================

def load_normalization_stats(data_dir):
    stats_file = Path(data_dir).parent / "normalization_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"\n✓ Loaded normalization stats from: {stats_file}")
        print(f"  Method: {stats.get('normalization_method', 'unknown')}")
        if 'position' in stats and 'displacement' in stats:
            print(f"  max_position: {stats['position']['max_position']:.2f} mm")
            print(f"  max_displacement: {stats['displacement']['max_displacement']:.2f} mm")
        return stats
    else:
        print(f"\n⚠️  No normalization_stats.json found at {stats_file}")
        return {
            'normalization_method': 'z_score',
            'position': {
                'x_pos': {'mean': 97.2165, 'std': 59.3803},
                'y_pos': {'mean': 50.2759, 'std': 28.4965}
            }
        }


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{' MODEL PARAMETERS ':~^50}")
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"{'~' * 50}\n")
    return trainable


def get_teacher_forcing_ratio(epoch, total_epochs, schedule='linear',
                               initial_ratio=1.0, final_ratio=0.0):
    if schedule == 'linear':
        ratio = initial_ratio - (initial_ratio - final_ratio) * (epoch / total_epochs)
    elif schedule == 'exponential':
        if initial_ratio > 0:
            decay = (final_ratio / initial_ratio) ** (1 / total_epochs)
            ratio = initial_ratio * (decay ** epoch)
        else:
            ratio = 0.0
    elif schedule == 'sigmoid':
        x = (epoch - total_epochs / 2) / (total_epochs / 10)
        sigmoid = 1 / (1 + torch.exp(torch.tensor(x)).item())
        ratio = final_ratio + (initial_ratio - final_ratio) * sigmoid
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return max(final_ratio, min(initial_ratio, ratio))


def get_valid_node_mask(elements, current_erosion, next_erosion=None, device='cpu'):
    num_nodes = elements.max().item() + 1
    valid_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    if current_erosion is not None:
        eroded = current_erosion.squeeze() < 0.5
        if eroded.any():
            valid_mask[elements[eroded].flatten().unique()] = False
    if next_erosion is not None:
        will_erode = next_erosion.squeeze() < 0.5
        if will_erode.any():
            valid_mask[elements[will_erode].flatten().unique()] = False
    return valid_mask


def compute_masked_loss(pred, target, elements, current_erosion, next_erosion=None):
    device = pred.device
    valid_mask = get_valid_node_mask(elements, current_erosion, next_erosion, device)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device), 0, pred.shape[0]
    loss = F.mse_loss(pred[valid_mask], target[valid_mask])
    return loss, valid_mask.sum().item(), (~valid_mask).sum().item()


# =========================================================================
# TRAINING / VALIDATION
# =========================================================================

def train_epoch(model, train_loader, optimizer, device, epoch, total_epochs, args):
    model.train()
    tf_ratio = get_teacher_forcing_ratio(
        epoch, total_epochs, args.ss_schedule, args.ss_initial_ratio, args.ss_final_ratio
    )
    total_loss, n_batches = 0.0, 0
    total_valid, total_eroded = 0, 0

    pbar = tqdm(train_loader, desc=f"Training (TF={tf_ratio:.3f})")
    for seq in pbar:
        for data in seq:
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :args.num_static_feats]

        optimizer.zero_grad()
        predictions = model(seq, dt=1.0, teacher_forcing_ratio=tf_ratio)

        loss, total_weight = 0.0, 0.0
        for t, (pred, data) in enumerate(zip(predictions, seq)):
            if args.mask_eroding and hasattr(data, 'elements') and hasattr(data, 'x_element'):
                step_loss, nv, nm = compute_masked_loss(
                    pred, data.y, data.elements, data.x_element,
                    data.y_element if hasattr(data, 'y_element') else None
                )
                total_valid += nv
                total_eroded += nm
            else:
                step_loss = F.mse_loss(pred, data.y)

            w = args.loss_decay_gamma ** t if args.use_loss_decay else 1.0
            loss += w * step_loss
            total_weight += w

        loss = loss / (total_weight if args.use_loss_decay else len(predictions))
        loss.backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({
            'loss': f"{loss.item():.6f}", 'TF': f"{tf_ratio:.3f}",
            'masked': f"{100 * total_eroded / (total_valid + total_eroded + 1e-6):.1f}%"
        })

    return {
        'loss': total_loss / n_batches, 'teacher_forcing_ratio': tf_ratio,
        'valid_nodes': total_valid, 'eroded_nodes': total_eroded,
    }


@torch.no_grad()
def validate_epoch(model, val_loader, device, args):
    model.eval()
    total_loss, n_batches = 0.0, 0

    for seq in tqdm(val_loader, desc="Validating"):
        for data in seq:
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :args.num_static_feats]

        predictions = model(seq, dt=1.0, teacher_forcing_ratio=0.0)

        loss = 0.0
        for t, (pred, data) in enumerate(zip(predictions, seq)):
            if args.mask_eroding and hasattr(data, 'elements') and hasattr(data, 'x_element'):
                step_loss, _, _ = compute_masked_loss(
                    pred, data.y, data.elements, data.x_element,
                    data.y_element if hasattr(data, 'y_element') else None
                )
            else:
                step_loss = F.mse_loss(pred, data.y)
            loss += step_loss

        total_loss += (loss / len(predictions)).item()
        n_batches += 1

    return {'loss': total_loss / n_batches}


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }, filepath)


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train MeshGraphKAN (NVIDIA arch) with Scheduled Sampling"
    )

    # Dataset
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, default="*.pt")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=2)

    # Architecture (NVIDIA defaults)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--processor_size", type=int, default=15)
    parser.add_argument("--num_harmonics", type=int, default=5)
    parser.add_argument("--num_layers_node_processor", type=int, default=2)
    parser.add_argument("--num_layers_edge_processor", type=int, default=2)
    parser.add_argument("--num_layers_edge_encoder", type=int, default=2)
    parser.add_argument("--num_layers_node_decoder", type=int, default=2)
    parser.add_argument("--aggregation", type=str, default="sum", choices=['sum', 'mean'])
    parser.add_argument("--mlp_activation", type=str, default="relu", choices=['relu', 'silu'])

    # Loss
    parser.add_argument("--mask_eroding", action="store_true", default=True)
    parser.add_argument("--use_loss_decay", action="store_true", default=False)
    parser.add_argument("--loss_decay_gamma", type=float, default=0.9)

    # Scheduled Sampling
    parser.add_argument("--ss_schedule", type=str, default="linear",
                        choices=['linear', 'exponential', 'sigmoid'])
    parser.add_argument("--ss_initial_ratio", type=float, default=1.0)
    parser.add_argument("--ss_final_ratio", type=float, default=0.0)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./outputs_meshgraphkan")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset_best", action="store_true", default=False)
    parser.add_argument("--fresh_scheduler", action="store_true", default=False)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    norm_stats = load_normalization_stats(args.train_dir)

    print("\n" + "=" * 70)
    print("MeshGraphKAN TRAINING - NVIDIA ARCHITECTURE (PyG Reimplementation)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Normalization: {norm_stats.get('normalization_method', 'unknown')}")
    print(f"")
    print(f"Architecture: MeshGraphKAN (NVIDIA PhysicsNeMo)")
    print(f"  Node encoder: KolmogorovArnoldNetwork ({args.num_harmonics} harmonics)")
    print(f"    fourier_coeffs: [2, {args.hidden_dim}, {args.num_static_feats + args.num_dynamic_feats}, {args.num_harmonics}]")
    print(f"  Edge encoder: MeshGraphMLP ({args.num_layers_edge_encoder} hidden layers, LayerNorm)")
    print(f"  Processor: {args.processor_size} message passing blocks")
    print(f"    Edge block: {args.num_layers_edge_processor} hidden layers")
    print(f"    Node block: {args.num_layers_node_processor} hidden layers")
    print(f"  Decoder: MeshGraphMLP ({args.num_layers_node_decoder} hidden layers, no LayerNorm)")
    print(f"  Hidden dim: {args.hidden_dim}, Aggregation: {args.aggregation}")
    print(f"")
    print(f"Scheduled Sampling: {args.ss_schedule} ({args.ss_initial_ratio} → {args.ss_final_ratio})")
    print(f"Loss: mask_eroding={args.mask_eroding}, decay={args.use_loss_decay}")
    print("=" * 70)

    # Dataset
    train_ids = get_simulation_ids(Path(args.train_dir), pattern=args.file_pattern)
    val_ids = get_simulation_ids(Path(args.val_dir), pattern=args.file_pattern)
    print(f"\nFound {len(train_ids)} train, {len(val_ids)} val simulations")

    train_dataset = ElastoPlasticDataset(
        directory=Path(args.train_dir), simulation_ids=train_ids,
        seq_len=args.seq_len, stride=args.stride,
        num_static_feats=args.num_static_feats, num_dynamic_feats=args.num_dynamic_feats
    )
    val_dataset = ElastoPlasticDataset(
        directory=Path(args.val_dir), simulation_ids=val_ids,
        seq_len=args.seq_len, stride=args.stride,
        num_static_feats=args.num_static_feats, num_dynamic_feats=args.num_dynamic_feats
    )

    loader_kw = {'batch_size': None, 'num_workers': args.num_workers, 'pin_memory': True}
    train_loader = DataLoader(train_dataset, **loader_kw)
    val_loader = DataLoader(val_dataset, **loader_kw)

    # Sample
    print("\nGetting sample...")
    sample = next(iter(train_loader))[0]
    print(f"  Nodes: {sample.num_nodes}, Edges: {sample.edge_index.shape[1]}")
    print(f"  x: {sample.x.shape}, y: {sample.y.shape}")

    # Model
    input_dim_nodes = args.num_static_feats + args.num_dynamic_feats

    kan_model = MeshGraphKAN(
        input_dim_nodes=input_dim_nodes, input_dim_edges=3,
        output_dim=args.num_dynamic_feats, processor_size=args.processor_size,
        mlp_activation_fn=args.mlp_activation,
        num_layers_node_processor=args.num_layers_node_processor,
        num_layers_edge_processor=args.num_layers_edge_processor,
        hidden_dim_processor=args.hidden_dim,
        hidden_dim_node_encoder=args.hidden_dim,
        hidden_dim_edge_encoder=args.hidden_dim,
        num_layers_edge_encoder=args.num_layers_edge_encoder,
        hidden_dim_node_decoder=args.hidden_dim,
        num_layers_node_decoder=args.num_layers_node_decoder,
        aggregation=args.aggregation, num_harmonics=args.num_harmonics,
    )

    model = MeshGraphKANRollout(
        kan_model, num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
    ).to(device)

    count_parameters(model)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        if args.fresh_scheduler:
            optimizer = AdamW(model.parameters(), lr=args.lr)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs - ckpt['epoch'] - 1, eta_min=args.lr * 0.01
            )
        else:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if ckpt['scheduler_state_dict']:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        start_epoch = ckpt['epoch'] + 1
        best_val_loss = float('inf') if args.reset_best else ckpt['metrics'].get('val_loss', float('inf'))
        print(f"  Epoch {start_epoch}, best_val_loss={best_val_loss:.8f}")

    # Save config
    config = vars(args)
    config['architecture'] = 'MeshGraphKAN (NVIDIA PhysicsNeMo, PyG reimplementation)'
    config['kan_implementation'] = 'Learnable Fourier coefficients [2, out, in, harmonics] with einsum'
    config['normalization'] = norm_stats.get('normalization_method', 'unknown')
    config['input_dim_nodes'] = input_dim_nodes
    config['input_dim_edges'] = 3
    config['references'] = [
        'Pfaff et al., Learning Mesh-Based Simulation with Graph Networks, 2021',
        'Liu et al., KAN: Kolmogorov-Arnold Networks, 2024',
        'Peng et al., Interpretable physics-informed GNNs for flood forecasting, 2024',
        'NVIDIA PhysicsNeMo: github.com/NVIDIA/physicsnemo',
    ]
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    norm_src = Path(args.train_dir).parent / "normalization_stats.json"
    if norm_src.exists():
        import shutil
        shutil.copy2(norm_src, output_dir / "normalization_stats.json")
        print(f"✓ Copied normalization_stats.json to output directory")

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Seq: {args.seq_len}")
    print(f"  TF: {args.ss_schedule} ({args.ss_initial_ratio} → {args.ss_final_ratio})")
    print("=" * 70)

    history = {'train_loss': [], 'val_loss': [], 'teacher_forcing_ratio': []}

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 70}\nEPOCH {epoch + 1}/{args.epochs}\n{'=' * 70}")

        train_m = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs, args)
        val_m = validate_epoch(model, val_loader, device, args)
        scheduler.step()

        print(f"\nTrain: {train_m['loss']:.6f} (TF={train_m['teacher_forcing_ratio']:.3f})")
        print(f"Val:   {val_m['loss']:.6f} (free rollout)")
        print(f"Nodes: {train_m['valid_nodes']:,} valid, {train_m['eroded_nodes']:,} masked")

        history['train_loss'].append(train_m['loss'])
        history['val_loss'].append(val_m['loss'])
        history['teacher_forcing_ratio'].append(train_m['teacher_forcing_ratio'])

        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            save_checkpoint(model, optimizer, scheduler, epoch,
                            {'val_loss': best_val_loss, 'tf': train_m['teacher_forcing_ratio']},
                            output_dir / "best_model.pth")
            print(f"✓ Saved best model (val_loss: {best_val_loss:.6f})")

        save_checkpoint(model, optimizer, scheduler, epoch,
                        {'val_loss': val_m['loss'], 'tf': train_m['teacher_forcing_ratio']},
                        output_dir / "latest_model.pth")

        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\n{'=' * 70}\nTRAINING COMPLETE\n{'=' * 70}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Models: {output_dir / 'best_model.pth'}, {output_dir / 'latest_model.pth'}")


if __name__ == "__main__":
    main()