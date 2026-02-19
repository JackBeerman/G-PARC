#!/usr/bin/env python3
"""
Training Script for MeshGraphKAN River Model
==============================================
PyG reimplementation of NVIDIA PhysicsNeMo MeshGraphKAN architecture,
adapted for HEC-RAS river simulation data (White River / Iowa River).

Architecture (identical to elasto version):
  - Node encoder: KolmogorovArnoldNetwork (learnable Fourier coefficients)
  - Edge encoder: MeshGraphMLP (with LayerNorm)
  - Processor: interleaved MeshEdgeBlock + MeshNodeBlock with residuals
  - Decoder: MeshGraphMLP (NO LayerNorm)

Key differences from elastoplastic:
  - 9 static features (bathymetry, Manning's n, etc.), 4 dynamic features
    (depth, volume, velocity_x, velocity_y)
  - Model predicts FULL next-step state (not delta)
  - No erosion masking
  - RiverDataset with location-based mesh_id (0=White, 1=Iowa)

Usage:
    python train_river_mgkan.py \\
        --train_dir /path/to/train \\
        --val_dir /path/to/val \\
        --output_dir ./river_meshgraphkan \\
        --epochs 500 --seq_len 20 --stride 10

References:
  - Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks" (2021)
  - Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024)
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

from data.Riverdataset import RiverDataset, get_simulation_ids


# =========================================================================
# ARCHITECTURE — identical to elasto MeshGraphKAN (imported from shared code)
# =========================================================================

class KolmogorovArnoldNetwork(nn.Module):
    """Learnable Fourier coefficients (cos + sin) per output × input × harmonic."""

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
    """PyG reimplementation of NVIDIA PhysicsNeMo MeshGraphKAN."""

    def __init__(self, input_dim_nodes=13, input_dim_edges=3, output_dim=4,
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
            aggregation=aggregation, activation_fn=activation_fn, norm_type="LayerNorm",
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
# ROLLOUT WRAPPER — adapted for river (predicts delta)
# =========================================================================

class MeshGraphKANRollout(nn.Module):
    """Autoregressive rollout wrapper for river: predicts full next-step state."""

    def __init__(self, model, num_static_feats=9, num_dynamic_feats=4):
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
        """Delta-prediction forward pass (faithful to NVIDIA MeshGraphNet convention).

        The network predicts the *change* (delta) from current state to next state.
        Rollout accumulates: next_state = current_state + predicted_delta.

        Returns:
            predictions: list of predicted deltas [N, df] per timestep
            input_states: list of current_dynamic used as input at each timestep
        """
        predictions = []
        input_states = []
        sf = self.num_static_feats
        df = self.num_dynamic_feats

        current_dynamic = sequence[0].x[:, sf:sf + df].clone()

        for t, data in enumerate(sequence):
            static_feats = data.x[:, :sf]
            node_features = torch.cat([static_feats, current_dynamic], dim=-1)
            edge_features = self.compute_edge_features(data)

            input_states.append(current_dynamic.clone())

            pred_delta = self.model(node_features, edge_features, data.edge_index)
            predictions.append(pred_delta)

            if t < len(sequence) - 1:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    current_dynamic = data.y[:, :df].clone()
                else:
                    current_dynamic = (current_dynamic + pred_delta).detach()

        return predictions, input_states


# =========================================================================
# UTILITIES
# =========================================================================

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
            decay = (final_ratio / max(initial_ratio, 1e-10)) ** (1 / total_epochs)
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


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }, filepath)


# =========================================================================
# TRAINING / VALIDATION
# =========================================================================

def train_epoch(model, train_loader, optimizer, device, epoch, total_epochs, args):
    model.train()
    tf_ratio = get_teacher_forcing_ratio(
        epoch, total_epochs, args.ss_schedule, args.ss_initial_ratio, args.ss_final_ratio
    )
    total_loss, n_batches = 0.0, 0

    pbar = tqdm(train_loader, desc=f"Training (TF={tf_ratio:.3f})")
    for seq in pbar:
        for data in seq:
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :2]

        optimizer.zero_grad()
        predictions, input_states = model(seq, dt=1.0, teacher_forcing_ratio=tf_ratio)

        loss = 0.0
        for t, (pred_delta, input_state, data) in enumerate(zip(predictions, input_states, seq)):
            target_full = data.y[:, :args.num_dynamic_feats]
            target_delta = target_full - input_state
            step_loss = F.mse_loss(pred_delta, target_delta)

            w = args.loss_decay_gamma ** t if args.use_loss_decay else 1.0
            loss += w * step_loss

        loss = loss / len(predictions)
        loss.backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f"{loss.item():.6f}", 'TF': f"{tf_ratio:.3f}"})

    return {
        'loss': total_loss / max(n_batches, 1),
        'teacher_forcing_ratio': tf_ratio,
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
                data.pos = data.x[:, :2]

        predictions, input_states = model(seq, dt=1.0, teacher_forcing_ratio=0.0)

        loss = 0.0
        for t, (pred_delta, input_state, data) in enumerate(zip(predictions, input_states, seq)):
            target_full = data.y[:, :args.num_dynamic_feats]
            target_delta = target_full - input_state
            loss += F.mse_loss(pred_delta, target_delta)

        total_loss += (loss / len(predictions)).item()
        n_batches += 1

    return {'loss': total_loss / max(n_batches, 1)}


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train MeshGraphKAN for River Simulations"
    )

    # Dataset
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, default="*.pt")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--num_static_feats", type=int, default=9)
    parser.add_argument("--num_dynamic_feats", type=int, default=4)
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle training data")

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--processor_size", type=int, default=4)
    parser.add_argument("--num_harmonics", type=int, default=5)
    parser.add_argument("--num_layers_node_processor", type=int, default=2)
    parser.add_argument("--num_layers_edge_processor", type=int, default=2)
    parser.add_argument("--num_layers_edge_encoder", type=int, default=2)
    parser.add_argument("--num_layers_node_decoder", type=int, default=2)
    parser.add_argument("--aggregation", type=str, default="sum", choices=['sum', 'mean'])
    parser.add_argument("--mlp_activation", type=str, default="relu", choices=['relu', 'silu'])

    # Loss
    parser.add_argument("--use_loss_decay", action="store_true", default=False)
    parser.add_argument("--loss_decay_gamma", type=float, default=0.9)

    # Scheduled Sampling
    parser.add_argument("--ss_schedule", type=str, default="linear",
                        choices=['linear', 'exponential', 'sigmoid'])
    parser.add_argument("--ss_initial_ratio", type=float, default=1.0)
    parser.add_argument("--ss_final_ratio", type=float, default=0.0)

    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./outputs_river_meshgraphkan")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset_best", action="store_true", default=False)
    parser.add_argument("--fresh_scheduler", action="store_true", default=False)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 70)
    print("MeshGraphKAN TRAINING — RIVER SIMULATIONS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"")
    print(f"Architecture: MeshGraphKAN (NVIDIA PhysicsNeMo, PyG)")
    print(f"  Node encoder: KAN ({args.num_harmonics} harmonics)")
    input_dim_nodes = args.num_static_feats + args.num_dynamic_feats
    print(f"  Input: {input_dim_nodes} node features ({args.num_static_feats} static + {args.num_dynamic_feats} dynamic)")
    print(f"  Processor: {args.processor_size} message passing blocks")
    print(f"  Hidden dim: {args.hidden_dim}, Aggregation: {args.aggregation}")
    print(f"  Output: {args.num_dynamic_feats} ( delta)")
    print(f"")
    print(f"Scheduled Sampling: {args.ss_schedule} ({args.ss_initial_ratio} -> {args.ss_final_ratio})")
    print("=" * 70)

    # Dataset
    train_ids = get_simulation_ids(Path(args.train_dir), file_pattern=args.file_pattern)
    val_ids = get_simulation_ids(Path(args.val_dir), file_pattern=args.file_pattern)
    print(f"\nFound {len(train_ids)} train, {len(val_ids)} val simulations")

    train_dataset = RiverDataset(
        directory=Path(args.train_dir), simulation_ids=train_ids,
        seq_len=args.seq_len, stride=args.stride,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        shuffle=args.shuffle,
    )
    val_dataset = RiverDataset(
        directory=Path(args.val_dir), simulation_ids=val_ids,
        seq_len=args.seq_len, stride=args.stride,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        shuffle=False,
    )

    loader_kw = {'batch_size': None, 'num_workers': args.num_workers, 'pin_memory': True}
    train_loader = DataLoader(train_dataset, **loader_kw)
    val_loader = DataLoader(val_dataset, **loader_kw)

    # Sample
    print("\nGetting sample...")
    sample = next(iter(train_loader))[0]
    print(f"  Nodes: {sample.num_nodes}, Edges: {sample.edge_index.shape[1]}")
    print(f"  x: {sample.x.shape}, y: {sample.y.shape}")
    print(f"  pos: {sample.pos.shape if hasattr(sample, 'pos') and sample.pos is not None else 'None'}")

    # Model
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
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        if args.fresh_scheduler:
            optimizer = AdamW(model.parameters(), lr=args.lr)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs - ckpt['epoch'] - 1, eta_min=args.lr * 0.01
            )
        else:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if ckpt.get('scheduler_state_dict'):
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        start_epoch = ckpt['epoch'] + 1
        best_val_loss = float('inf') if args.reset_best else ckpt['metrics'].get('val_loss', float('inf'))
        print(f"  Epoch {start_epoch}, best_val_loss={best_val_loss:.8f}")

    # Save config
    config = vars(args)
    config['architecture'] = 'MeshGraphKAN (NVIDIA PhysicsNeMo, PyG reimplementation)'
    config['prediction_mode'] = 'delta'
    config['input_dim_nodes'] = input_dim_nodes
    config['input_dim_edges'] = 3
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Seq: {args.seq_len}")
    print(f"  TF: {args.ss_schedule} ({args.ss_initial_ratio} -> {args.ss_final_ratio})")
    print("=" * 70)

    history = {'train_loss': [], 'val_loss': [], 'teacher_forcing_ratio': []}

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 70}\nEPOCH {epoch + 1}/{args.epochs}\n{'=' * 70}")

        train_m = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs, args)
        val_m = validate_epoch(model, val_loader, device, args)
        scheduler.step()

        print(f"\nTrain: {train_m['loss']:.6f} (TF={train_m['teacher_forcing_ratio']:.3f})")
        print(f"Val:   {val_m['loss']:.6f} (free rollout)")

        history['train_loss'].append(train_m['loss'])
        history['val_loss'].append(val_m['loss'])
        history['teacher_forcing_ratio'].append(train_m['teacher_forcing_ratio'])

        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            save_checkpoint(model, optimizer, scheduler, epoch,
                            {'val_loss': best_val_loss, 'tf': train_m['teacher_forcing_ratio']},
                            output_dir / "best_model.pth")
            print(f"  Saved best model (val_loss: {best_val_loss:.6f})")

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