#!/usr/bin/env python3
"""
MeshGraphKAN Evaluation Script
==============================
Evaluates MeshGraphKAN models trained with scheduled sampling.
Supports both ROLLOUT and SNAPSHOT evaluation modes.
Mirrors G-PARC evaluation script for direct comparison.

Architecture: PyG reimplementation of NVIDIA PhysicsNeMo MeshGraphKAN
"""

import argparse
import os
import sys
from pathlib import Path
import json
import warnings
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================================
# FAITHFUL PyG REIMPLEMENTATION OF NVIDIA MeshGraphKAN
# (Must match training script exactly)
# =========================================================================

import torch.nn as nn
import torch.nn.functional as F


class KolmogorovArnoldNetwork(nn.Module):
    """Exact reimplementation of physicsnemo.nn.KolmogorovArnoldNetwork."""

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
    """Reimplements physicsnemo MeshGraphMLP."""

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
    """Edge update: concat(src, dst, edge) -> MLP -> residual."""

    def __init__(self, input_dim_node, input_dim_edge, output_dim_edge,
                 hidden_dim_edge, hidden_layers=2, activation_fn=None, norm_type="LayerNorm"):
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
    """Node update: aggregate edges -> concat(node, agg) -> MLP -> residual."""

    def __init__(self, aggregation, input_dim_node, input_dim_edge, output_dim_node,
                 hidden_dim_node, hidden_layers=2, activation_fn=None, norm_type="LayerNorm"):
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
                hidden_layers=hidden_layers_edge, activation_fn=activation_fn, norm_type=norm_type,
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


class MeshGraphKANRollout(nn.Module):
    """Autoregressive rollout wrapper."""

    def __init__(self, model, num_static_feats=2, num_dynamic_feats=2):
        super().__init__()
        self.model = model
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats

    def compute_edge_features(self, data):
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

    def step(self, static_feats, dynamic_state, edge_index, pos, dt=1.0):
        """Single-step prediction for evaluation (matches G-PARC interface)."""
        node_features = torch.cat([static_feats, dynamic_state], dim=-1)
        # Compute edge features from positions
        src_pos = pos[edge_index[0]]
        dst_pos = pos[edge_index[1]]
        rel_pos = dst_pos - src_pos
        distance = torch.norm(rel_pos, dim=1, keepdim=True)
        edge_features = torch.cat([rel_pos, distance], dim=1)
        delta = self.model(node_features, edge_features, edge_index)
        return dynamic_state + delta


# =========================================================================
# NORMALIZATION UTILITIES
# =========================================================================

def load_normalization_stats(data_dir):
    """Load normalization statistics from the data directory."""
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
        return None


def load_normalization_stats_from_checkpoint_dir(checkpoint_dir):
    """Try loading normalization_stats.json from model checkpoint directory."""
    stats_file = Path(checkpoint_dir) / "normalization_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"\n✓ Loaded normalization stats from checkpoint dir: {stats_file}")
        return stats
    return None


# =========================================================================
# EROSION HANDLING
# =========================================================================

EROSION_THRESHOLD = 0.5


def get_erosion_mask(data, num_elements):
    if hasattr(data, 'x_element') and data.x_element is not None:
        erosion_status = data.x_element.cpu().numpy().flatten()
        return erosion_status < EROSION_THRESHOLD
    return np.zeros(num_elements, dtype=bool)


def get_valid_node_mask(elements, eroded_mask):
    valid_elements = elements[~eroded_mask]
    if len(valid_elements) == 0:
        return np.zeros(elements.max() + 1, dtype=bool)
    valid_nodes = np.unique(valid_elements.flatten())
    valid_node_mask = np.zeros(elements.max() + 1, dtype=bool)
    valid_node_mask[valid_nodes] = True
    return valid_node_mask


# =========================================================================
# FAST RENDERING
# =========================================================================

def precompute_element_polygons(pos, elements):
    return pos[elements]


def render_mesh_fast(ax, poly_verts, node_values, elements, eroded_mask,
                     vmin, vmax, cmap_obj, norm, show_eroded=False):
    valid_mask = ~eroded_mask
    if valid_mask.sum() == 0:
        return None
    valid_verts = poly_verts[valid_mask]
    valid_elements = elements[valid_mask]
    elem_node_vals = node_values[valid_elements]
    elem_vals = np.clip(elem_node_vals.mean(axis=1), vmin, vmax)
    colors = cmap_obj(norm(elem_vals))
    pc = PolyCollection(valid_verts, facecolors=colors, edgecolors='k',
                        linewidths=0.1, alpha=1.0)
    ax.add_collection(pc)
    if show_eroded and eroded_mask.sum() > 0:
        eroded_verts = poly_verts[eroded_mask]
        pc_eroded = PolyCollection(eroded_verts, facecolors='lightgray',
                                   edgecolors='gray', linewidths=0.1, alpha=0.3)
        ax.add_collection(pc_eroded)
    return pc


# =========================================================================
# PRECOMPUTATION
# =========================================================================

def precompute_visualization_data(simulation, seq_pred, seq_targ, elements):
    max_steps = min(len(seq_pred), len(seq_targ), len(simulation))
    num_elements = len(elements)
    pos_ref = simulation[0].pos.cpu().numpy() if hasattr(simulation[0], 'pos') and simulation[0].pos is not None else simulation[0].x[:, :2].cpu().numpy()
    poly_verts_ref = precompute_element_polygons(pos_ref, elements)

    erosion_masks, erosion_counts, valid_node_masks = [], [], []
    for t in range(max_steps):
        eroded_mask = get_erosion_mask(simulation[t], num_elements)
        erosion_masks.append(eroded_mask)
        erosion_counts.append(eroded_mask.sum())
        valid_node_masks.append(get_valid_node_mask(elements, eroded_mask))

    disp_max, error_max = 0, 0
    Ux_min, Ux_max, Uy_min, Uy_max = 0, 0, 0, 0
    x_ref, y_ref = pos_ref[:, 0], pos_ref[:, 1]
    def_x_min, def_x_max = x_ref.min(), x_ref.max()
    def_y_min, def_y_max = y_ref.min(), y_ref.max()

    for t in range(max_steps):
        valid_nodes = valid_node_masks[t]
        if valid_nodes.sum() == 0:
            continue
        U_targ, U_pred = seq_targ[t], seq_pred[t]
        for U in [U_targ, U_pred]:
            u_mag = np.sqrt(U[valid_nodes, 0]**2 + U[valid_nodes, 1]**2)
            disp_max = max(disp_max, u_mag.max())
        error_mag = np.sqrt((U_targ[valid_nodes, 0] - U_pred[valid_nodes, 0])**2 +
                           (U_targ[valid_nodes, 1] - U_pred[valid_nodes, 1])**2)
        error_max = max(error_max, error_mag.max())
        for U in [U_targ, U_pred]:
            Ux_min = min(Ux_min, U[valid_nodes, 0].min())
            Ux_max = max(Ux_max, U[valid_nodes, 0].max())
            Uy_min = min(Uy_min, U[valid_nodes, 1].min())
            Uy_max = max(Uy_max, U[valid_nodes, 1].max())
        eroded_mask = erosion_masks[t]
        valid_elements_t = elements[~eroded_mask]
        if len(valid_elements_t) > 0:
            valid_node_indices = np.unique(valid_elements_t.flatten())
            for U in [U_targ, U_pred]:
                x_def = x_ref[valid_node_indices] + U[valid_node_indices, 0]
                y_def = y_ref[valid_node_indices] + U[valid_node_indices, 1]
                def_x_min = min(def_x_min, x_def.min())
                def_x_max = max(def_x_max, x_def.max())
                def_y_min = min(def_y_min, y_def.min())
                def_y_max = max(def_y_max, y_def.max())

    pad = 0.1
    pad_x = (x_ref.max() - x_ref.min()) * pad
    pad_y = (y_ref.max() - y_ref.min()) * pad
    camera_ref = (x_ref.min() - pad_x, x_ref.max() + pad_x,
                  y_ref.min() - pad_y, y_ref.max() + pad_y)
    pad_x_def = (def_x_max - def_x_min) * pad
    pad_y_def = (def_y_max - def_y_min) * pad
    camera_def = (def_x_min - pad_x_def, def_x_max + pad_x_def,
                  def_y_min - pad_y_def, def_y_max + pad_y_def)

    return {
        'max_steps': max_steps, 'pos_ref': pos_ref, 'x_ref': x_ref, 'y_ref': y_ref,
        'poly_verts_ref': poly_verts_ref, 'erosion_masks': erosion_masks,
        'erosion_counts': erosion_counts, 'valid_node_masks': valid_node_masks,
        'disp_max': disp_max, 'error_max': error_max,
        'Ux_range': (Ux_min, Ux_max), 'Uy_range': (Uy_min, Uy_max),
        'camera_ref': camera_ref, 'camera_def': camera_def,
    }


# =========================================================================
# GIF CREATION FUNCTIONS
# =========================================================================

def _create_reference_gif(frames, precomputed, seq_pred, seq_targ, elements,
                          case_name, output_dir, fps, eval_mode='rollout'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    disp_max = precomputed['disp_max']
    norm = Normalize(vmin=0, vmax=disp_max)
    cmap = plt.colormaps.get_cmap('viridis')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label('Displacement Magnitude', fontsize=11)
    camera = precomputed['camera_ref']
    poly_verts = precomputed['poly_verts_ref']
    erosion_masks = precomputed['erosion_masks']
    mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'

    def animate(frame_idx):
        frame = frames[frame_idx]
        for ax in axes:
            ax.clear(); ax.set_xlim(camera[0], camera[1]); ax.set_ylim(camera[2], camera[3])
            ax.set_aspect('equal'); ax.axis('off')
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        d_targ = np.sqrt(U_targ[:, 0]**2 + U_targ[:, 1]**2)
        d_pred = np.sqrt(U_pred[:, 0]**2 + U_pred[:, 1]**2)
        eroded_mask = erosion_masks[frame]
        n_eroded = eroded_mask.sum()
        render_mesh_fast(axes[0], poly_verts, d_targ, elements, eroded_mask, 0, disp_max, cmap, norm, show_eroded=True)
        title = f'Target (t={frame})'
        if n_eroded > 0: title += f' [{n_eroded} eroded]'
        axes[0].set_title(title, fontsize=12)
        render_mesh_fast(axes[1], poly_verts, d_pred, elements, eroded_mask, 0, disp_max, cmap, norm, show_eroded=True)
        axes[1].set_title(f'MeshGraphKAN (t={frame})', fontsize=12)
        fig.suptitle(f'Reference Config ({mode_label}): {case_name}', fontsize=14)
        return axes.tolist()

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(Path(output_dir) / f'reference_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def _create_deformed_gif(frames, precomputed, seq_pred, seq_targ, elements,
                         case_name, output_dir, fps, eval_mode='rollout'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    disp_max = precomputed['disp_max']
    norm = Normalize(vmin=0, vmax=disp_max)
    cmap = plt.colormaps.get_cmap('viridis')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label('Displacement Magnitude', fontsize=11)
    camera = precomputed['camera_def']
    x_ref, y_ref = precomputed['x_ref'], precomputed['y_ref']
    erosion_masks = precomputed['erosion_masks']
    mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'

    def animate(frame_idx):
        frame = frames[frame_idx]
        for ax in axes:
            ax.clear(); ax.set_xlim(camera[0], camera[1]); ax.set_ylim(camera[2], camera[3])
            ax.set_aspect('equal'); ax.axis('off')
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        pos_targ_def = np.column_stack([x_ref + U_targ[:, 0], y_ref + U_targ[:, 1]])
        pos_pred_def = np.column_stack([x_ref + U_pred[:, 0], y_ref + U_pred[:, 1]])
        poly_verts_targ = precompute_element_polygons(pos_targ_def, elements)
        poly_verts_pred = precompute_element_polygons(pos_pred_def, elements)
        d_targ = np.sqrt(U_targ[:, 0]**2 + U_targ[:, 1]**2)
        d_pred = np.sqrt(U_pred[:, 0]**2 + U_pred[:, 1]**2)
        eroded_mask = erosion_masks[frame]
        n_eroded = eroded_mask.sum()
        render_mesh_fast(axes[0], poly_verts_targ, d_targ, elements, eroded_mask, 0, disp_max, cmap, norm, show_eroded=False)
        title = f'Target (t={frame})'
        if n_eroded > 0: title += f' [{n_eroded} eroded]'
        axes[0].set_title(title, fontsize=12)
        render_mesh_fast(axes[1], poly_verts_pred, d_pred, elements, eroded_mask, 0, disp_max, cmap, norm, show_eroded=False)
        axes[1].set_title(f'MeshGraphKAN (t={frame})', fontsize=12)
        fig.suptitle(f'Deformed Config ({mode_label}): {case_name}', fontsize=14)
        return axes.tolist()

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(Path(output_dir) / f'deformed_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def _create_error_gif(frames, precomputed, seq_pred, seq_targ, elements,
                      case_name, output_dir, fps, eval_mode='rollout'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    error_max = precomputed['error_max']
    norm = Normalize(vmin=0, vmax=error_max)
    cmap = plt.colormaps.get_cmap('hot')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label('Error Magnitude', fontsize=11)
    camera = precomputed['camera_ref']
    poly_verts = precomputed['poly_verts_ref']
    erosion_masks = precomputed['erosion_masks']
    mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'

    def animate(frame_idx):
        frame = frames[frame_idx]
        ax.clear(); ax.set_xlim(camera[0], camera[1]); ax.set_ylim(camera[2], camera[3])
        ax.set_aspect('equal'); ax.axis('off')
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        error_mag = np.sqrt((U_targ[:, 0] - U_pred[:, 0])**2 + (U_targ[:, 1] - U_pred[:, 1])**2)
        eroded_mask = erosion_masks[frame]
        render_mesh_fast(ax, poly_verts, error_mag, elements, eroded_mask, 0, error_max, cmap, norm, show_eroded=True)
        n_eroded = eroded_mask.sum()
        title = f'Prediction Error - t={frame}'
        if n_eroded > 0: title += f' [{n_eroded} eroded]'
        ax.set_title(title, fontsize=14)
        fig.suptitle(f'Error ({mode_label}): {case_name}', fontsize=14)
        return [ax]

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(Path(output_dir) / f'error_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def _create_erosion_plot(precomputed, case_name, output_dir):
    erosion_counts = precomputed['erosion_counts']
    max_steps = precomputed['max_steps']
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(max_steps), erosion_counts, 'r-', linewidth=2, marker='o', markersize=4)
    ax.fill_between(range(max_steps), erosion_counts, alpha=0.3, color='red')
    ax.set_xlabel('Timestep', fontsize=12); ax.set_ylabel('Eroded Elements', fontsize=12)
    ax.set_title(f'Element Erosion Progression: {case_name}', fontsize=14)
    ax.grid(alpha=0.3); ax.set_xlim(0, max_steps - 1); ax.set_ylim(0, max(erosion_counts) * 1.1 + 1)
    fig.savefig(Path(output_dir) / f'erosion_{case_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_single_gif(gif_type, precomputed, seq_pred, seq_targ, elements,
                      case_name, output_dir, fps, frame_skip, eval_mode='rollout'):
    max_steps = precomputed['max_steps']
    frames = list(range(0, max_steps, frame_skip))
    if gif_type == 'reference':
        _create_reference_gif(frames, precomputed, seq_pred, seq_targ, elements, case_name, output_dir, fps, eval_mode)
    elif gif_type == 'deformed':
        _create_deformed_gif(frames, precomputed, seq_pred, seq_targ, elements, case_name, output_dir, fps, eval_mode)
    elif gif_type == 'error':
        _create_error_gif(frames, precomputed, seq_pred, seq_targ, elements, case_name, output_dir, fps, eval_mode)


# =========================================================================
# METRICS
# =========================================================================

def compute_plaid_rrmse(predictions, references, valid_masks=None):
    if len(predictions) == 0:
        return 0.0
    n_samples = len(predictions)
    numerator_sum, denominator_sum = 0.0, 0.0
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if valid_masks is not None and valid_masks[i] is not None:
            mask = valid_masks[i]
            pred, ref = pred[mask], ref[mask]
        if len(pred) == 0:
            continue
        n_nodes = pred.shape[0]
        numerator_sum += np.sum((pred - ref) ** 2) / n_nodes
        denominator_sum += np.max(np.abs(ref)) ** 2
    if denominator_sum == 0:
        return float('inf')
    return np.sqrt((numerator_sum / n_samples) / (denominator_sum / n_samples))


def compute_plaid_rrmse_per_component(predictions, references, valid_masks=None):
    if len(predictions) == 0:
        return {'U_x': 0.0, 'U_y': 0.0}
    n_samples = len(predictions)
    component_names = ['U_x', 'U_y']
    rrmse_per_component = {}
    for comp_idx, comp_name in enumerate(component_names):
        num_sum, den_sum = 0.0, 0.0
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if valid_masks is not None and valid_masks[i] is not None:
                mask = valid_masks[i]
                pred_c, ref_c = pred[mask, comp_idx], ref[mask, comp_idx]
            else:
                pred_c, ref_c = pred[:, comp_idx], ref[:, comp_idx]
            if len(pred_c) == 0:
                continue
            n_nodes = pred_c.shape[0]
            num_sum += np.sum((pred_c - ref_c) ** 2) / n_nodes
            den_sum += np.max(np.abs(ref_c)) ** 2
        rrmse = np.sqrt((num_sum / n_samples) / (den_sum / n_samples)) if den_sum > 0 else float('inf')
        rrmse_per_component[comp_name] = float(rrmse)
    return rrmse_per_component


def select_representative_simulations(results_dict, n_samples=3, selection_mode='representative'):
    if 'simulation_metrics' not in results_dict or not results_dict['simulation_metrics']:
        return []
    sims = [(m['metadata']['simulation_idx'], m['overall_physical']['rmse'])
            for m in results_dict['simulation_metrics']]
    sims.sort(key=lambda x: x[1])
    if selection_mode == 'all':
        return [s[0] for s in sims]
    elif selection_mode == 'best':
        return [s[0] for s in sims[:n_samples]]
    elif selection_mode == 'worst':
        return [s[0] for s in sims[-n_samples:]]
    else:
        selected = []
        if len(sims) >= 1: selected.append(sims[0][0])
        if len(sims) >= 2: selected.append(sims[len(sims)//2][0])
        if len(sims) >= 3: selected.append(sims[-1][0])
        return selected[:n_samples]


# =========================================================================
# EVALUATOR CLASS
# =========================================================================

class MeshGraphKANEvaluator:
    """Evaluator for MeshGraphKAN models — mirrors G-PARC ElastoPlasticEvaluator."""

    def __init__(self, model, device='cpu', norm_stats=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.norm_stats = norm_stats
        self.num_static_feats = model.num_static_feats
        self.num_dynamic_feats = model.num_dynamic_feats
        self.simulation_metrics = []

    def denormalize_predictions(self, normalized_data, method='global_max'):
        if method == 'none' or method is None:
            return normalized_data
        if method == 'global_max':
            if self.norm_stats is None:
                return normalized_data
            disp_stats = self.norm_stats.get('displacement', {})
            max_disp = disp_stats.get('max_displacement', 1.0)
            return normalized_data * max_disp
        return normalized_data

    def compute_edge_features_from_pos(self, pos, edge_index):
        """Compute edge features from position tensor and edge_index."""
        src_pos = pos[edge_index[0]]
        dst_pos = pos[edge_index[1]]
        rel_pos = dst_pos - src_pos
        distance = torch.norm(rel_pos, dim=1, keepdim=True)
        return torch.cat([rel_pos, distance], dim=1)

    def generate_rollout(self, simulation, rollout_steps):
        """Generate autoregressive rollout predictions."""
        predictions = []
        sf = self.num_static_feats
        df = self.num_dynamic_feats

        # Initialize from ground truth at t=0
        current_dynamic = simulation[0].x[:, sf:sf + df].clone()

        for step in range(rollout_steps):
            data_t = simulation[step]
            static_feats = data_t.x[:, :sf]
            edge_index = data_t.edge_index
            pos = data_t.pos if hasattr(data_t, 'pos') and data_t.pos is not None else data_t.x[:, :2]

            node_features = torch.cat([static_feats, current_dynamic], dim=-1)
            edge_features = self.compute_edge_features_from_pos(pos, edge_index)

            delta = self.model.model(node_features, edge_features, edge_index)
            new_state = current_dynamic + delta

            predictions.append(new_state)
            current_dynamic = new_state.detach()

        return predictions

    def generate_snapshot_predictions(self, simulation, num_steps):
        """Generate snapshot (single-step) predictions from ground truth."""
        predictions = []
        sf = self.num_static_feats
        df = self.num_dynamic_feats

        for step in range(num_steps):
            data_t = simulation[step]
            static_feats = data_t.x[:, :sf]
            edge_index = data_t.edge_index
            pos = data_t.pos if hasattr(data_t, 'pos') and data_t.pos is not None else data_t.x[:, :2]

            # Use GROUND TRUTH dynamic state at time t
            F_gt = data_t.x[:, sf:sf + df].clone()

            node_features = torch.cat([static_feats, F_gt], dim=-1)
            edge_features = self.compute_edge_features_from_pos(pos, edge_index)

            delta = self.model.model(node_features, edge_features, edge_index)
            F_pred = F_gt + delta

            predictions.append(F_pred)

        return predictions

    def _process_simulation(self, sim_idx, simulation, mode, rollout_steps, denorm_method):
        """Process a single simulation for either rollout or snapshot mode."""
        simulation = [d.to(self.device) for d in simulation]
        initial_data = simulation[0]
        elements = initial_data.elements.detach().cpu().numpy()

        if mode == 'rollout':
            actual_steps = min(rollout_steps, len(simulation))
            preds_raw = self.generate_rollout(simulation, actual_steps)
        else:
            actual_steps = len(simulation) - 1
            preds_raw = self.generate_snapshot_predictions(simulation, actual_steps)

        # Filter unstable predictions
        preds_norm = []
        for p in preds_raw:
            if torch.isfinite(p).all() and p.abs().max() < 50.0:
                preds_norm.append(p.cpu().numpy())
            else:
                if mode == 'rollout':
                    break  # Stop rollout at first instability
                continue  # Skip single snapshot

        if len(preds_norm) == 0:
            return None

        actual_steps = len(preds_norm)

        # Targets
        targs_norm = [simulation[i].y.cpu().numpy() for i in range(actual_steps)]

        # Erosion masks
        erosion_masks = [get_erosion_mask(simulation[i], len(elements)) for i in range(actual_steps)]

        # Denormalize
        preds_phys = [self.denormalize_predictions(p, denorm_method) for p in preds_norm]
        targs_phys = [self.denormalize_predictions(t, denorm_method) for t in targs_norm]

        # Metadata
        metadata = {
            'simulation_idx': sim_idx,
            'case_name': f'simulation_{sim_idx}',
            'num_nodes': initial_data.num_nodes,
            'num_elements': len(elements),
            'max_eroded': max(m.sum() for m in erosion_masks) if erosion_masks else 0,
        }
        if mode == 'rollout':
            metadata['rollout_length'] = actual_steps
        else:
            metadata['num_snapshots'] = actual_steps

        # Per-simulation metrics
        valid_node_masks = [get_valid_node_mask(elements, em) for em in erosion_masks]
        all_p, all_t = [], []
        for t in range(len(preds_phys)):
            mask = valid_node_masks[t]
            if mask.sum() > 0:
                all_p.append(preds_phys[t][mask])
                all_t.append(targs_phys[t][mask])

        if len(all_p) > 0:
            all_p_cat = np.concatenate(all_p, axis=0)
            all_t_cat = np.concatenate(all_t, axis=0)
            rmse = float(np.sqrt(mean_squared_error(all_t_cat, all_p_cat)))
            r2 = float(r2_score(all_t_cat, all_p_cat))
        else:
            rmse, r2 = float('inf'), 0.0

        sim_metric = {
            'metadata': metadata,
            'overall_physical': {'rmse': rmse, 'r2': r2}
        }

        return {
            'preds_phys': preds_phys, 'targs_phys': targs_phys,
            'erosion_masks': erosion_masks, 'metadata': metadata,
            'sim_metric': sim_metric,
        }

    def evaluate(self, simulations, mode='rollout', rollout_steps=37,
                 normalization_method='global_max'):
        """Evaluate model in rollout or snapshot mode."""
        results = {
            'predictions_physical': [], 'targets_physical': [],
            'metadata': [], 'erosion_masks': [],
        }
        self.simulation_metrics = []

        mode_label = 'Rollout' if mode == 'rollout' else 'Snapshot'

        with torch.no_grad():
            for sim_idx, simulation in enumerate(tqdm(simulations, desc=f"{mode_label} predictions")):
                try:
                    out = self._process_simulation(
                        sim_idx, simulation, mode, rollout_steps, normalization_method
                    )
                    if out is None:
                        print(f"  Skipping sim {sim_idx}: unstable")
                        continue

                    results['predictions_physical'].append(out['preds_phys'])
                    results['targets_physical'].append(out['targs_phys'])
                    results['erosion_masks'].append(out['erosion_masks'])
                    results['metadata'].append(out['metadata'])
                    self.simulation_metrics.append(out['sim_metric'])

                except Exception as e:
                    print(f"Error processing simulation {sim_idx}: {e}")
                    import traceback
                    traceback.print_exc()

        results['simulation_metrics'] = self.simulation_metrics
        return results

    def compute_plaid_benchmark_metrics(self, predictions_physical, targets_physical, erosion_masks=None):
        if not predictions_physical:
            return {}
        all_pred, all_targ, all_masks = [], [], []
        for seq_p, seq_t in zip(predictions_physical, targets_physical):
            for p, tg in zip(seq_p, seq_t):
                all_pred.append(p)
                all_targ.append(tg)
                all_masks.append(None)
        rrmse_total = compute_plaid_rrmse(all_pred, all_targ, all_masks)
        rrmse_per_comp = compute_plaid_rrmse_per_component(all_pred, all_targ, all_masks)
        return {
            'RRMSE_total': rrmse_total,
            'RRMSE_Ux': rrmse_per_comp.get('U_x', 0),
            'RRMSE_Uy': rrmse_per_comp.get('U_y', 0),
            'total_error': np.mean(list(rrmse_per_comp.values()))
        }

    def create_mesh_deformation_gifs(self, simulations, results_dict, seq_idx, output_dir,
                                      fps=10, frame_skip=1, eval_mode='rollout'):
        predictions = results_dict['predictions_physical']
        targets = results_dict['targets_physical']
        metadata = results_dict['metadata']
        if seq_idx >= len(predictions):
            return

        seq_pred = predictions[seq_idx]
        seq_targ = targets[seq_idx]
        simulation = simulations[seq_idx]
        case_name = metadata[seq_idx].get('case_name', f'simulation_{seq_idx}')

        if eval_mode == 'snapshot':
            sim_for_viz = simulation[1:len(seq_pred)+1]
        else:
            sim_for_viz = simulation[:len(seq_pred)]

        max_steps = min(len(seq_pred), len(seq_targ), len(sim_for_viz))
        if max_steps < 2:
            return

        first_data = simulation[0]
        if not hasattr(first_data, 'elements'):
            return
        elements = first_data.elements.cpu().numpy()

        print(f"\n{'='*70}")
        print(f"Creating GIFs for {case_name} ({eval_mode} mode)")
        print(f"  Elements: {len(elements)}, Timesteps: {max_steps}")
        print(f"{'='*70}")

        precomputed = precompute_visualization_data(sim_for_viz, seq_pred, seq_targ, elements)

        for gif_type in ['reference', 'deformed', 'error']:
            create_single_gif(gif_type, precomputed, seq_pred, seq_targ, elements,
                             case_name, output_dir, fps, frame_skip, eval_mode)
            print(f"    ✓ {gif_type}_{case_name}.gif")

        if max(precomputed['erosion_counts']) > 0:
            _create_erosion_plot(precomputed, case_name, output_dir)
            print(f"    ✓ erosion_{case_name}.png")

    def plot_comprehensive_metrics(self, results_dict, figsize=(18, 10), eval_mode='rollout'):
        if not self.simulation_metrics:
            return None
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        r2_scores = [m['overall_physical']['r2'] for m in self.simulation_metrics]
        rmse_values = [m['overall_physical']['rmse'] for m in self.simulation_metrics]
        sim_indices = [m['metadata']['simulation_idx'] for m in self.simulation_metrics]
        max_eroded = [m['metadata'].get('max_eroded', 0) for m in self.simulation_metrics]
        mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(r2_scores, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(r2_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r2_scores):.3f}')
        ax1.set_xlabel('R² Score'); ax1.set_ylabel('Frequency')
        ax1.set_title('R² Score Distribution', fontweight='bold'); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(rmse_values, bins=15, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(rmse_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rmse_values):.3e}')
        ax2.set_xlabel('RMSE'); ax2.set_ylabel('Frequency')
        ax2.set_title('RMSE Distribution', fontweight='bold'); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        norm_method = self.norm_stats.get('normalization_method', 'unknown') if self.norm_stats else 'unknown'
        stats_text = f"""
MeshGraphKAN {mode_label.upper()}
{'='*30}
Normalization: {norm_method}

R² Score:
  Mean:   {np.mean(r2_scores):.4f}
  Median: {np.median(r2_scores):.4f}
  Min:    {np.min(r2_scores):.4f}
  Max:    {np.max(r2_scores):.4f}

RMSE:
  Mean:   {np.mean(rmse_values):.3e}
  Median: {np.median(rmse_values):.3e}

Simulations: {len(self.simulation_metrics)}
        """
        ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        ax4 = fig.add_subplot(gs[1, :2])
        scatter = ax4.scatter(r2_scores, rmse_values, c=max_eroded, cmap='YlOrRd',
                             s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('R² Score'); ax4.set_ylabel('RMSE')
        ax4.set_title('R² vs RMSE (colored by max eroded elements)', fontweight='bold')
        ax4.grid(alpha=0.3); plt.colorbar(scatter, ax=ax4).set_label('Max Eroded Elements')

        ax5 = fig.add_subplot(gs[1, 2])
        colors = ['green' if r2 > 0.8 else 'orange' if r2 > 0.5 else 'red' for r2 in r2_scores]
        ax5.barh(range(len(sim_indices)), r2_scores, color=colors, edgecolor='black', alpha=0.7)
        ax5.set_xlabel('R² Score'); ax5.set_ylabel('Simulation Index')
        ax5.set_title('Performance by Simulation', fontweight='bold')
        ax5.axvline(0.8, color='green', linestyle='--', alpha=0.5)
        ax5.axvline(0.5, color='orange', linestyle='--', alpha=0.5); ax5.grid(alpha=0.3, axis='x')

        ax6 = fig.add_subplot(gs[2, 0])
        ax6.scatter(max_eroded, r2_scores, c='steelblue', s=80, alpha=0.7, edgecolors='black')
        ax6.set_xlabel('Max Eroded Elements'); ax6.set_ylabel('R² Score')
        ax6.set_title('R² vs Erosion Level', fontweight='bold'); ax6.grid(alpha=0.3)

        ax7 = fig.add_subplot(gs[2, 1])
        ax7.plot(range(len(rmse_values)), rmse_values, marker='o', linestyle='-',
                linewidth=1.5, markersize=6, color='coral', alpha=0.7)
        ax7.set_xlabel('Simulation Index'); ax7.set_ylabel('RMSE')
        ax7.set_title('RMSE by Simulation', fontweight='bold'); ax7.grid(alpha=0.3)
        ax7.axhline(np.mean(rmse_values), color='red', linestyle='--', alpha=0.5, label='Mean'); ax7.legend(fontsize=9)

        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        sorted_indices = np.argsort(r2_scores)
        best_3 = sorted_indices[-3:][::-1]
        worst_3 = sorted_indices[:3]
        cases_text = f"BEST PERFORMERS\n{'='*25}\n"
        for i, idx in enumerate(best_3, 1):
            cases_text += f"{i}. Sim {sim_indices[idx]}: R²={r2_scores[idx]:.4f}\n"
        cases_text += f"\nWORST PERFORMERS\n{'='*25}\n"
        for i, idx in enumerate(worst_3, 1):
            cases_text += f"{i}. Sim {sim_indices[idx]}: R²={r2_scores[idx]:.4f}\n"
        ax8.text(0.1, 0.5, cases_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        fig.suptitle(f'MeshGraphKAN Performance Analysis ({mode_label})',
                    fontsize=16, fontweight='bold', y=0.98)
        return fig


# =========================================================================
# MAIN EVALUATION
# =========================================================================

def load_test_simulations(test_dir, pattern, max_files):
    import re
    simulations = []
    paths = sorted(list(Path(test_dir).glob(pattern)))
    if max_files:
        paths = paths[:max_files]
    for idx, p in enumerate(paths):
        try:
            sim_data = torch.load(p, weights_only=False)
            match = re.search(r'\d+', p.stem)
            sim_id_int = int(match.group()) if match else idx
            for data in sim_data:
                data.mesh_id = torch.tensor([sim_id_int], dtype=torch.long)
                if not hasattr(data, 'pos') or data.pos is None:
                    data.pos = data.x[:, :2]
            simulations.append(sim_data)
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return simulations


def build_model_from_config(config, device):
    """Build MeshGraphKAN model from saved config."""
    input_dim_nodes = config.get('input_dim_nodes', config.get('num_static_feats', 2) + config.get('num_dynamic_feats', 2))

    kan_model = MeshGraphKAN(
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=config.get('input_dim_edges', 3),
        output_dim=config.get('num_dynamic_feats', 2),
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

    model = MeshGraphKANRollout(
        kan_model,
        num_static_feats=config.get('num_static_feats', 2),
        num_dynamic_feats=config.get('num_dynamic_feats', 2),
    ).to(device)

    return model


def evaluate_meshgraphkan(args):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load checkpoint
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Load config from training output dir
    model_dir = Path(args.model_path).parent
    config_file = model_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded config from {config_file}")
    else:
        print("⚠️  No config.json found, using command-line args")
        config = {
            'hidden_dim': args.hidden_dim,
            'processor_size': args.processor_size,
            'num_harmonics': args.num_harmonics,
            'aggregation': args.aggregation,
            'mlp_activation': args.mlp_activation,
            'num_static_feats': args.num_static_feats,
            'num_dynamic_feats': args.num_dynamic_feats,
            'num_layers_node_processor': args.num_layers_node_processor,
            'num_layers_edge_processor': args.num_layers_edge_processor,
            'num_layers_edge_encoder': args.num_layers_edge_encoder,
            'num_layers_node_decoder': args.num_layers_node_decoder,
        }

    # Load normalization stats
    norm_stats = None
    if args.norm_stats_file and Path(args.norm_stats_file).exists():
        with open(args.norm_stats_file, 'r') as f:
            norm_stats = json.load(f)
        print(f"✓ Loaded norm stats from: {args.norm_stats_file}")
    if norm_stats is None and args.test_dir:
        norm_stats = load_normalization_stats(args.test_dir)
    if norm_stats is None:
        norm_stats = load_normalization_stats_from_checkpoint_dir(model_dir)
    if norm_stats is None:
        print("⚠️  No normalization stats found! Using raw values.")
        norm_stats = {'normalization_method': 'none'}

    norm_method = norm_stats.get('normalization_method', 'none')
    if args.normalization_method == 'auto':
        denorm_method = 'global_max' if norm_method == 'global_max' else 'none'
    else:
        denorm_method = args.normalization_method

    print(f"\n{'='*60}")
    print(f"MeshGraphKAN EVALUATION")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Normalization: {norm_method}")
    print(f"  Denormalization: {denorm_method}")
    print(f"  Processor layers: {config.get('processor_size', '?')}")
    print(f"  Hidden dim: {config.get('hidden_dim', '?')}")
    print(f"  KAN harmonics: {config.get('num_harmonics', '?')}")
    print(f"{'='*60}")

    # Build and load model
    model = build_model_from_config(config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Load test data
    print(f"\nLoading test simulations from: {args.test_dir}")
    simulations = load_test_simulations(args.test_dir, "*.pt", args.max_sequences)
    if not simulations:
        print("No simulations loaded!")
        return
    print(f"Loaded {len(simulations)} simulations")

    # Create evaluator
    evaluator = MeshGraphKANEvaluator(model, device, norm_stats=norm_stats)

    eval_mode = args.eval_mode.lower()

    def safe_fmt(val):
        if isinstance(val, (float, int, np.floating)):
            return f"{val:.4f}"
        return str(val)

    # ROLLOUT
    if eval_mode in ['rollout', 'both']:
        print(f"\n{'='*60}")
        print(f"ROLLOUT EVALUATION (rollout_steps={args.rollout_steps})")
        print(f"{'='*60}")

        rollout_results = evaluator.evaluate(simulations, mode='rollout',
                                              rollout_steps=args.rollout_steps,
                                              normalization_method=denorm_method)
        rollout_metrics = evaluator.compute_plaid_benchmark_metrics(
            rollout_results['predictions_physical'], rollout_results['targets_physical'])

        print(f"\n{'─'*40}")
        print(f"ROLLOUT RESULTS")
        print(f"{'─'*40}")
        print(f"  RRMSE Total: {safe_fmt(rollout_metrics.get('RRMSE_total'))}")
        print(f"  RRMSE U_x:   {safe_fmt(rollout_metrics.get('RRMSE_Ux'))}")
        print(f"  RRMSE U_y:   {safe_fmt(rollout_metrics.get('RRMSE_Uy'))}")

        rollout_metrics['eval_mode'] = 'rollout'
        rollout_metrics['model'] = 'MeshGraphKAN'
        rollout_metrics['parameters'] = total_params
        with open(output_path / 'rollout_metrics.json', 'w') as f:
            json.dump(rollout_metrics, f, indent=2)

        fig = evaluator.plot_comprehensive_metrics(rollout_results, eval_mode='rollout')
        if fig:
            fig.savefig(output_path / 'rollout_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

        if args.create_gifs:
            selected = select_representative_simulations(rollout_results, n_samples=args.num_viz_simulations)
            for idx in selected:
                evaluator.create_mesh_deformation_gifs(
                    simulations, rollout_results, idx, output_path,
                    fps=args.gif_fps, frame_skip=args.gif_frame_skip, eval_mode='rollout')

    # SNAPSHOT
    if eval_mode in ['snapshot', 'both']:
        print(f"\n{'='*60}")
        print(f"SNAPSHOT EVALUATION")
        print(f"{'='*60}")

        snapshot_results = evaluator.evaluate(simulations, mode='snapshot',
                                               normalization_method=denorm_method)
        snapshot_metrics = evaluator.compute_plaid_benchmark_metrics(
            snapshot_results['predictions_physical'], snapshot_results['targets_physical'])

        print(f"\n{'─'*40}")
        print(f"SNAPSHOT RESULTS")
        print(f"{'─'*40}")
        print(f"  RRMSE Total: {safe_fmt(snapshot_metrics.get('RRMSE_total'))}")
        print(f"  RRMSE U_x:   {safe_fmt(snapshot_metrics.get('RRMSE_Ux'))}")
        print(f"  RRMSE U_y:   {safe_fmt(snapshot_metrics.get('RRMSE_Uy'))}")

        snapshot_metrics['eval_mode'] = 'snapshot'
        snapshot_metrics['model'] = 'MeshGraphKAN'
        snapshot_metrics['parameters'] = total_params
        with open(output_path / 'snapshot_metrics.json', 'w') as f:
            json.dump(snapshot_metrics, f, indent=2)

        fig = evaluator.plot_comprehensive_metrics(snapshot_results, eval_mode='snapshot')
        if fig:
            fig.savefig(output_path / 'snapshot_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

        if args.create_gifs:
            selected = select_representative_simulations(snapshot_results, n_samples=args.num_viz_simulations)
            for idx in selected:
                evaluator.create_mesh_deformation_gifs(
                    simulations, snapshot_results, idx, output_path,
                    fps=args.gif_fps, frame_skip=args.gif_frame_skip, eval_mode='snapshot')

    # COMPARISON
    if eval_mode == 'both':
        print(f"\n{'='*60}")
        print(f"SNAPSHOT vs ROLLOUT COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Metric':<20} {'Snapshot':>12} {'Rollout':>12} {'Ratio':>10}")
        print(f"  {'─'*54}")
        for key in ['RRMSE_total', 'RRMSE_Ux', 'RRMSE_Uy']:
            s_val = snapshot_metrics.get(key, 0)
            r_val = rollout_metrics.get(key, 0)
            ratio = r_val / s_val if s_val > 0 else float('inf')
            print(f"  {key:<20} {s_val:>12.4f} {r_val:>12.4f} {ratio:>10.1f}x")

        comparison = {'snapshot': snapshot_metrics, 'rollout': rollout_metrics}
        with open(output_path / 'comparison_metrics.json', 'w') as f:
            json.dump(comparison, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="MeshGraphKAN Evaluation")

    # Paths
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", default="./eval_meshgraphkan")
    parser.add_argument("--norm_stats_file", type=str, default=None)

    # Eval mode
    parser.add_argument("--eval_mode", type=str, default="rollout", choices=['rollout', 'snapshot', 'both'])
    parser.add_argument("--rollout_steps", type=int, default=37)
    parser.add_argument("--max_sequences", type=int, default=10)
    parser.add_argument("--normalization_method", default="auto", choices=['auto', 'global_max', 'none'])

    # Architecture (fallback if no config.json)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--processor_size", type=int, default=4)
    parser.add_argument("--num_harmonics", type=int, default=5)
    parser.add_argument("--aggregation", type=str, default="sum")
    parser.add_argument("--mlp_activation", type=str, default="relu")
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=2)
    parser.add_argument("--num_layers_node_processor", type=int, default=2)
    parser.add_argument("--num_layers_edge_processor", type=int, default=2)
    parser.add_argument("--num_layers_edge_encoder", type=int, default=2)
    parser.add_argument("--num_layers_node_decoder", type=int, default=2)

    # Visualization
    parser.add_argument("--create_gifs", action="store_true")
    parser.add_argument("--num_viz_simulations", type=int, default=3)
    parser.add_argument("--gif_fps", type=int, default=10)
    parser.add_argument("--gif_frame_skip", type=int, default=1)

    args = parser.parse_args()
    evaluate_meshgraphkan(args)


if __name__ == "__main__":
    main()