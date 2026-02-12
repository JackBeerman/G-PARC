#!/usr/bin/env python3
"""
MeshGraphNet Evaluation Script
===============================
Mirrors the G-PARC evaluation script for fair comparison.
Supports ROLLOUT evaluation with identical metrics (PLAID RRMSE),
dashboard, and GIF generation.

Key differences from G-PARC eval:
- MGN predicts deltas (ΔU), not absolute states
- MGN uses its own z-score normalization stats (saved as .pt)
- Autoregressive rollout: U_{t+1} = U_t + unnormalize(model(data_t))
- No MLS operators or boundary enforcement
"""

import argparse
import os
import sys
import re
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from meshgraphnet import MeshGraphNet, normalize, unnormalize

warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# EROSION HANDLING (identical to G-PARC eval)
# ==============================================================================

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


# ==============================================================================
# FAST RENDERING (identical to G-PARC eval)
# ==============================================================================

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


# ==============================================================================
# PRECOMPUTATION (identical to G-PARC eval)
# ==============================================================================

def precompute_visualization_data(simulation, seq_pred, seq_targ, elements):
    max_steps = min(len(seq_pred), len(seq_targ), len(simulation))
    pos_ref = simulation[0].pos.cpu().numpy()
    poly_verts_ref = precompute_element_polygons(pos_ref, elements)

    erosion_masks, erosion_counts, valid_node_masks = [], [], []
    for t in range(max_steps):
        eroded_mask = get_erosion_mask(simulation[t], len(elements))
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
        'max_steps': max_steps, 'pos_ref': pos_ref,
        'x_ref': x_ref, 'y_ref': y_ref,
        'poly_verts_ref': poly_verts_ref,
        'erosion_masks': erosion_masks, 'erosion_counts': erosion_counts,
        'valid_node_masks': valid_node_masks,
        'disp_max': disp_max, 'error_max': error_max,
        'Ux_range': (Ux_min, Ux_max), 'Uy_range': (Uy_min, Uy_max),
        'camera_ref': camera_ref, 'camera_def': camera_def,
    }


# ==============================================================================
# GIF CREATION (identical to G-PARC eval)
# ==============================================================================

def _create_reference_gif(frames, precomputed, seq_pred, seq_targ, elements,
                          case_name, output_dir, fps):
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

    def animate(frame_idx):
        frame = frames[frame_idx]
        for ax in axes:
            ax.clear(); ax.set_xlim(camera[0], camera[1])
            ax.set_ylim(camera[2], camera[3]); ax.set_aspect('equal'); ax.axis('off')
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        d_targ = np.sqrt(U_targ[:, 0]**2 + U_targ[:, 1]**2)
        d_pred = np.sqrt(U_pred[:, 0]**2 + U_pred[:, 1]**2)
        eroded_mask = erosion_masks[frame]
        n_eroded = eroded_mask.sum()
        render_mesh_fast(axes[0], poly_verts, d_targ, elements, eroded_mask, 0, disp_max, cmap, norm, True)
        title = f'Target (t={frame})'
        if n_eroded > 0: title += f' [{n_eroded} eroded]'
        axes[0].set_title(title, fontsize=12)
        render_mesh_fast(axes[1], poly_verts, d_pred, elements, eroded_mask, 0, disp_max, cmap, norm, True)
        axes[1].set_title(f'Prediction (t={frame})', fontsize=12)
        fig.suptitle(f'Reference Config (Rollout): {case_name}', fontsize=14)
        return axes.tolist()

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(Path(output_dir) / f'reference_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def _create_deformed_gif(frames, precomputed, seq_pred, seq_targ, elements,
                         case_name, output_dir, fps):
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

    def animate(frame_idx):
        frame = frames[frame_idx]
        for ax in axes:
            ax.clear(); ax.set_xlim(camera[0], camera[1])
            ax.set_ylim(camera[2], camera[3]); ax.set_aspect('equal'); ax.axis('off')
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        pos_targ_def = np.column_stack([x_ref + U_targ[:, 0], y_ref + U_targ[:, 1]])
        pos_pred_def = np.column_stack([x_ref + U_pred[:, 0], y_ref + U_pred[:, 1]])
        poly_verts_targ = precompute_element_polygons(pos_targ_def, elements)
        poly_verts_pred = precompute_element_polygons(pos_pred_def, elements)
        d_targ = np.sqrt(U_targ[:, 0]**2 + U_targ[:, 1]**2)
        d_pred = np.sqrt(U_pred[:, 0]**2 + U_pred[:, 1]**2)
        eroded_mask = erosion_masks[frame]
        n_eroded = eroded_mask.sum()
        render_mesh_fast(axes[0], poly_verts_targ, d_targ, elements, eroded_mask, 0, disp_max, cmap, norm, False)
        title = f'Target (t={frame})'
        if n_eroded > 0: title += f' [{n_eroded} eroded]'
        axes[0].set_title(title, fontsize=12)
        render_mesh_fast(axes[1], poly_verts_pred, d_pred, elements, eroded_mask, 0, disp_max, cmap, norm, False)
        axes[1].set_title(f'Prediction (t={frame})', fontsize=12)
        fig.suptitle(f'Deformed Config (Rollout): {case_name}', fontsize=14)
        return axes.tolist()

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(Path(output_dir) / f'deformed_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def _create_error_gif(frames, precomputed, seq_pred, seq_targ, elements,
                      case_name, output_dir, fps):
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

    def animate(frame_idx):
        frame = frames[frame_idx]
        ax.clear(); ax.set_xlim(camera[0], camera[1])
        ax.set_ylim(camera[2], camera[3]); ax.set_aspect('equal'); ax.axis('off')
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        error_mag = np.sqrt((U_targ[:, 0] - U_pred[:, 0])**2 +
                           (U_targ[:, 1] - U_pred[:, 1])**2)
        eroded_mask = erosion_masks[frame]
        render_mesh_fast(ax, poly_verts, error_mag, elements, eroded_mask, 0, error_max, cmap, norm, True)
        n_eroded = eroded_mask.sum()
        title = f'Prediction Error - t={frame}'
        if n_eroded > 0: title += f' [{n_eroded} eroded]'
        ax.set_title(title, fontsize=14)
        fig.suptitle(f'Error (Rollout): {case_name}', fontsize=14)
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
    ax.grid(alpha=0.3); ax.set_xlim(0, max_steps - 1)
    ax.set_ylim(0, max(erosion_counts) * 1.1 + 1)
    fig.savefig(Path(output_dir) / f'erosion_{case_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_gifs(precomputed, seq_pred, seq_targ, elements, case_name, output_dir,
                fps=10, frame_skip=1):
    max_steps = precomputed['max_steps']
    frames = list(range(0, max_steps, frame_skip))

    print(f"\n{'='*70}")
    print(f"Creating GIFs for {case_name}")
    print(f"  Elements: {len(elements)}, Timesteps: {max_steps}")
    print(f"{'='*70}")

    _create_reference_gif(frames, precomputed, seq_pred, seq_targ, elements, case_name, output_dir, fps)
    print(f"    ✓ reference_{case_name}.gif")
    _create_deformed_gif(frames, precomputed, seq_pred, seq_targ, elements, case_name, output_dir, fps)
    print(f"    ✓ deformed_{case_name}.gif")
    _create_error_gif(frames, precomputed, seq_pred, seq_targ, elements, case_name, output_dir, fps)
    print(f"    ✓ error_{case_name}.gif")
    if max(precomputed['erosion_counts']) > 0:
        _create_erosion_plot(precomputed, case_name, output_dir)
        print(f"    ✓ erosion_{case_name}.png")
    print(f"{'='*70}\n")


# ==============================================================================
# METRICS (identical to G-PARC eval)
# ==============================================================================

def compute_plaid_rrmse(predictions, references, valid_masks=None):
    if len(predictions) == 0:
        return 0.0
    n_samples = len(predictions)
    numerator_sum, denominator_sum = 0.0, 0.0
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if valid_masks is not None and valid_masks[i] is not None:
            pred, ref = pred[valid_masks[i]], ref[valid_masks[i]]
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
    n_components = predictions[0].shape[1]
    rrmse_per_component = {}
    component_names = ['U_x', 'U_y']
    for comp_idx in range(n_components):
        num_sum, den_sum = 0.0, 0.0
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if valid_masks is not None and valid_masks[i] is not None:
                pred_c = pred[valid_masks[i], comp_idx]
                ref_c = ref[valid_masks[i], comp_idx]
            else:
                pred_c, ref_c = pred[:, comp_idx], ref[:, comp_idx]
            if len(pred_c) == 0:
                continue
            n_nodes = pred_c.shape[0]
            num_sum += np.sum((pred_c - ref_c) ** 2) / n_nodes
            den_sum += np.max(np.abs(ref_c)) ** 2
        rrmse = np.sqrt((num_sum / n_samples) / (den_sum / n_samples)) if den_sum > 0 else float('inf')
        rrmse_per_component[component_names[comp_idx]] = float(rrmse)
    return rrmse_per_component


def select_representative_simulations(sim_metrics, n_samples=3):
    if not sim_metrics:
        return []
    sims = [(m['sim_idx'], m['rmse']) for m in sim_metrics]
    sims.sort(key=lambda x: x[1])
    selected = []
    if len(sims) >= 1: selected.append(sims[0][0])           # best
    if len(sims) >= 2: selected.append(sims[len(sims)//2][0]) # median
    if len(sims) >= 3: selected.append(sims[-1][0])           # worst
    return selected[:n_samples]


# ==============================================================================
# MGN-SPECIFIC: AUTOREGRESSIVE ROLLOUT
# ==============================================================================

def mgn_autoregressive_rollout(model, simulation, stats, device, rollout_steps):
    """
    Autoregressive rollout for MeshGraphNet.
    
    MGN predicts deltas: ΔU = model(data_t)
    State update: U_{t+1} = U_t + unnormalize(ΔU)
    
    Args:
        model: MeshGraphNet model
        simulation: List of Data objects (full simulation)
        stats: Normalization statistics dict (.pt file contents)
        device: torch device
        rollout_steps: Number of steps to predict
        
    Returns:
        predictions: List of displacement arrays [num_nodes, 2] in NORMALIZED space
                     (same space as the data, for fair comparison with targets)
    """
    model.eval()
    
    mean_vec_x = stats['mean_vec_x'].to(device)
    std_vec_x = stats['std_vec_x'].to(device)
    mean_vec_edge = stats['mean_vec_edge'].to(device)
    std_vec_edge = stats['std_vec_edge'].to(device)
    mean_vec_y = stats['mean_vec_y'].to(device)
    std_vec_y = stats['std_vec_y'].to(device)
    
    # Start from first timestep
    current_data = simulation[0].clone().to(device)
    predictions = []
    
    with torch.no_grad():
        for step in range(rollout_steps):
            # Forward pass: predict normalized delta
            pred_normalized = model(
                current_data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge
            )
            
            # Unnormalize the delta prediction
            delta_U = unnormalize(pred_normalized, mean_vec_y, std_vec_y)
            
            # Update displacements: U_{t+1} = U_t + ΔU
            # x = [x_pos, y_pos, U_x, U_y] — positions are static, displacements update
            next_data = current_data.clone()
            next_data.x = current_data.x.clone()
            next_data.x[:, 2:] = current_data.x[:, 2:] + delta_U
            
            # Store the predicted displacement (columns 2: onward)
            predictions.append(next_data.x[:, 2:].cpu())
            
            # Stability check
            if not torch.isfinite(next_data.x).all() or next_data.x[:, 2:].abs().max() > 1e6:
                print(f"    ⚠️  Rollout diverged at step {step}")
                break
            
            current_data = next_data
    
    return predictions


# ==============================================================================
# MGN-SPECIFIC: DENORMALIZATION
# ==============================================================================

def denormalize_displacement(displacement_normalized, norm_stats, method='zscore'):
    """
    Convert normalized displacements to physical units.
    
    Args:
        displacement_normalized: numpy array [num_nodes, 2]
        norm_stats: normalization statistics
        method: 'zscore' or 'global_max' or 'none'
    
    Returns:
        displacement_physical: numpy array [num_nodes, 2] in mm
    """
    if method == 'none':
        return displacement_normalized
    
    if method == 'global_max':
        max_disp = norm_stats.get('max_displacement', 542.1)
        return displacement_normalized * max_disp
    
    if method == 'zscore':
        # For z-score data, displacements in data.x[:, 2:] are already in 
        # the z-score normalized space. We need the original displacement stats.
        # These should be in the normalization_stats.json or computed from data.
        if 'displacement' in norm_stats:
            disp_stats = norm_stats['displacement']
            physical = np.zeros_like(displacement_normalized)
            # Component-wise denormalization
            for i, comp in enumerate(['U_x', 'U_y']):
                if comp in disp_stats:
                    mean = disp_stats[comp].get('mean', 0.0)
                    std = disp_stats[comp].get('std', 1.0)
                    physical[:, i] = displacement_normalized[:, i] * std + mean
                else:
                    physical[:, i] = displacement_normalized[:, i]
            return physical
        else:
            # Fallback: data might already be in physical units
            return displacement_normalized
    
    return displacement_normalized


# ==============================================================================
# DASHBOARD (mirrors G-PARC eval)
# ==============================================================================

def plot_dashboard(sim_metrics, norm_method='zscore', figsize=(18, 10)):
    if not sim_metrics:
        return None

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    r2_scores = [m['r2'] for m in sim_metrics]
    rmse_values = [m['rmse'] for m in sim_metrics]
    sim_indices = [m['sim_idx'] for m in sim_metrics]
    max_eroded = [m.get('max_eroded', 0) for m in sim_metrics]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(r2_scores, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(r2_scores), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(r2_scores):.3f}')
    ax1.set_xlabel('R² Score'); ax1.set_ylabel('Frequency')
    ax1.set_title('R² Score Distribution', fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(rmse_values, bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(rmse_values), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(rmse_values):.3e}')
    ax2.set_xlabel('RMSE'); ax2.set_ylabel('Frequency')
    ax2.set_title('RMSE Distribution', fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    stats_text = f"""
ROLLOUT METRICS (MeshGraphNet)
{'='*30}
Normalization: {norm_method}

R² Score:
  Mean:   {np.mean(r2_scores):.4f}
  Median: {np.median(r2_scores):.4f}
  Std:    {np.std(r2_scores):.4f}
  Min:    {np.min(r2_scores):.4f}
  Max:    {np.max(r2_scores):.4f}

RMSE:
  Mean:   {np.mean(rmse_values):.3e}
  Median: {np.median(rmse_values):.3e}

Simulations: {len(sim_metrics)}
Avg Max Eroded: {np.mean(max_eroded):.0f}
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax4 = fig.add_subplot(gs[1, :2])
    scatter = ax4.scatter(r2_scores, rmse_values, c=max_eroded, cmap='YlOrRd',
                         s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('R² Score'); ax4.set_ylabel('RMSE')
    ax4.set_title('R² vs RMSE (colored by max eroded elements)', fontweight='bold')
    ax4.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax4).set_label('Max Eroded Elements')

    ax5 = fig.add_subplot(gs[1, 2])
    colors = ['green' if r2 > 0.8 else 'orange' if r2 > 0.5 else 'red' for r2 in r2_scores]
    ax5.barh(range(len(sim_indices)), r2_scores, color=colors, edgecolor='black', alpha=0.7)
    ax5.set_xlabel('R² Score'); ax5.set_ylabel('Simulation Index')
    ax5.set_title('Performance by Simulation', fontweight='bold')
    ax5.grid(alpha=0.3, axis='x')

    ax6 = fig.add_subplot(gs[2, 0])
    ax6.scatter(max_eroded, r2_scores, c='steelblue', s=80, alpha=0.7, edgecolors='black')
    ax6.set_xlabel('Max Eroded Elements'); ax6.set_ylabel('R² Score')
    ax6.set_title('R² vs Erosion Level', fontweight='bold'); ax6.grid(alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(range(len(rmse_values)), rmse_values, marker='o', linestyle='-',
            linewidth=1.5, markersize=6, color='coral', alpha=0.7)
    ax7.set_xlabel('Simulation Index'); ax7.set_ylabel('RMSE')
    ax7.set_title('RMSE by Simulation', fontweight='bold'); ax7.grid(alpha=0.3)
    ax7.axhline(np.mean(rmse_values), color='red', linestyle='--', alpha=0.5, label='Mean')
    ax7.legend(fontsize=9)

    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    sorted_idx = np.argsort(r2_scores)
    best_3 = sorted_idx[-3:][::-1]
    worst_3 = sorted_idx[:3]
    txt = f"BEST PERFORMERS\n{'='*25}\n"
    for i, idx in enumerate(best_3, 1):
        txt += f"{i}. Sim {sim_indices[idx]}: R²={r2_scores[idx]:.4f}\n"
    txt += f"\nWORST PERFORMERS\n{'='*25}\n"
    for i, idx in enumerate(worst_3, 1):
        txt += f"{i}. Sim {sim_indices[idx]}: R²={r2_scores[idx]:.4f}\n"
    ax8.text(0.1, 0.5, txt, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    fig.suptitle('Model Performance Analysis (Rollout - MeshGraphNet)',
                fontsize=16, fontweight='bold', y=0.98)
    return fig


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def load_simulations(test_dir, pattern="*.pt", max_files=None):
    """Load simulation files with mesh_id injection."""
    paths = sorted(list(Path(test_dir).glob(pattern)))
    if max_files:
        paths = paths[:max_files]
    
    simulations = []
    for idx, p in enumerate(paths):
        try:
            sim_data = torch.load(p, weights_only=False)
            match = re.search(r'\d+', p.stem)
            sim_id_int = int(match.group()) if match else idx
            for data in sim_data:
                data.mesh_id = torch.tensor([sim_id_int], dtype=torch.long)
            simulations.append(sim_data)
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return simulations


def main():
    parser = argparse.ArgumentParser(description="MeshGraphNet Evaluation")

    # Paths
    parser.add_argument("--model_path", required=True, help="Path to MGN checkpoint (.pt)")
    parser.add_argument("--stats_path", required=True, help="Path to normalization_stats.pt")
    parser.add_argument("--test_dir", required=True, help="Directory with test .pt simulation files")
    parser.add_argument("--output_dir", default="./eval_mgn", help="Output directory")
    parser.add_argument("--norm_stats_json", default=None,
                        help="Path to normalization_stats.json for physical denormalization")

    # Architecture (must match training)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--input_dim_node", type=int, default=4)
    parser.add_argument("--input_dim_edge", type=int, default=3)
    parser.add_argument("--output_dim", type=int, default=2)

    # Eval settings
    parser.add_argument("--max_sequences", type=int, default=10)
    parser.add_argument("--rollout_steps", type=int, default=37)
    parser.add_argument("--denorm_method", default="zscore",
                        choices=['zscore', 'global_max', 'none'],
                        help="How to convert predictions to physical units")

    # Viz
    parser.add_argument("--create_gifs", action="store_true")
    parser.add_argument("--num_viz_simulations", type=int, default=3)
    parser.add_argument("--gif_fps", type=int, default=10)
    parser.add_argument("--gif_frame_skip", type=int, default=1)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # ==================== LOAD MODEL ====================
    print(f"Loading MeshGraphNet from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    model = MeshGraphNet(
        input_dim_node=args.input_dim_node,
        input_dim_edge=args.input_dim_edge,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # ==================== LOAD NORMALIZATION STATS ====================
    print(f"Loading normalization stats from: {args.stats_path}")
    stats = torch.load(args.stats_path, map_location=device)
    print(f"  Node stats: mean={stats['mean_vec_x']}, std={stats['std_vec_x']}")
    print(f"  Target stats: mean={stats['mean_vec_y']}, std={stats['std_vec_y']}")

    # Load physical denormalization stats if available
    phys_norm_stats = {}
    if args.norm_stats_json and Path(args.norm_stats_json).exists():
        with open(args.norm_stats_json) as f:
            phys_norm_stats = json.load(f)
        print(f"  Loaded physical norm stats from: {args.norm_stats_json}")
    
    denorm_method = args.denorm_method

    # ==================== LOAD DATA ====================
    print(f"\nLoading test simulations from: {args.test_dir}")
    simulations = load_simulations(args.test_dir, max_files=args.max_sequences)
    print(f"Loaded {len(simulations)} simulations")

    if not simulations:
        print("No simulations loaded!")
        return

    # ==================== ROLLOUT EVALUATION ====================
    print(f"\n{'='*60}")
    print(f"ROLLOUT EVALUATION (rollout_steps={args.rollout_steps})")
    print(f"  Denormalization: {denorm_method}")
    print(f"{'='*60}")

    all_preds_phys = []
    all_targs_phys = []
    all_erosion_masks = []
    sim_metrics = []

    for sim_idx, simulation in enumerate(tqdm(simulations, desc="Evaluating")):
        try:
            actual_steps = min(args.rollout_steps, len(simulation) - 1)
            if actual_steps < 1:
                continue

            elements = simulation[0].elements.cpu().numpy() if hasattr(simulation[0], 'elements') else None

            # Run autoregressive rollout
            preds_raw = mgn_autoregressive_rollout(
                model, simulation, stats, device, actual_steps
            )

            if len(preds_raw) == 0:
                print(f"  Skipping sim {sim_idx}: rollout failed")
                continue

            actual_steps = len(preds_raw)

            # Get targets: displacement from data.x[:, 2:] at each future timestep
            # Target at step t is simulation[t+1].x[:, 2:] (the next state's displacement)
            targs_raw = [simulation[t + 1].x[:, 2:].cpu() for t in range(actual_steps)]

            # Convert to numpy
            preds_np = [p.numpy() for p in preds_raw]
            targs_np = [t.numpy() for t in targs_raw]

            # Denormalize to physical units
            preds_phys = [denormalize_displacement(p, phys_norm_stats, denorm_method) for p in preds_np]
            targs_phys = [denormalize_displacement(t, phys_norm_stats, denorm_method) for t in targs_np]

            # Erosion masks
            erosion_masks = []
            if elements is not None:
                for t in range(actual_steps):
                    erosion_masks.append(get_erosion_mask(simulation[t + 1], len(elements)))
            else:
                erosion_masks = [np.zeros(0, dtype=bool)] * actual_steps

            all_preds_phys.append(preds_phys)
            all_targs_phys.append(targs_phys)
            all_erosion_masks.append(erosion_masks)

            # Per-simulation metrics
            if elements is not None:
                valid_node_masks = [get_valid_node_mask(elements, em) for em in erosion_masks]
                all_p, all_t = [], []
                for t in range(len(preds_phys)):
                    mask = valid_node_masks[t]
                    if mask.sum() > 0:
                        all_p.append(preds_phys[t][mask])
                        all_t.append(targs_phys[t][mask])
                if all_p:
                    all_p_cat = np.concatenate(all_p, axis=0)
                    all_t_cat = np.concatenate(all_t, axis=0)
                    rmse = float(np.sqrt(mean_squared_error(all_t_cat, all_p_cat)))
                    r2 = float(r2_score(all_t_cat, all_p_cat))
                else:
                    rmse, r2 = float('inf'), 0.0
            else:
                all_p_cat = np.concatenate(preds_phys, axis=0)
                all_t_cat = np.concatenate(targs_phys, axis=0)
                rmse = float(np.sqrt(mean_squared_error(all_t_cat, all_p_cat)))
                r2 = float(r2_score(all_t_cat, all_p_cat))

            max_eroded = max(em.sum() for em in erosion_masks) if erosion_masks and len(erosion_masks[0]) > 0 else 0

            sim_metrics.append({
                'sim_idx': sim_idx,
                'rmse': rmse,
                'r2': r2,
                'rollout_length': actual_steps,
                'num_nodes': simulation[0].num_nodes,
                'max_eroded': int(max_eroded)
            })

        except Exception as e:
            print(f"Error evaluating sim {sim_idx}: {e}")
            import traceback
            traceback.print_exc()

    # ==================== COMPUTE PLAID METRICS ====================
    all_pred_flat, all_targ_flat = [], []
    for seq_p, seq_t in zip(all_preds_phys, all_targs_phys):
        for p, t in zip(seq_p, seq_t):
            all_pred_flat.append(p)
            all_targ_flat.append(t)

    rrmse_total = compute_plaid_rrmse(all_pred_flat, all_targ_flat)
    rrmse_comp = compute_plaid_rrmse_per_component(all_pred_flat, all_targ_flat)

    print(f"\n{'-'*40}")
    print(f"ROLLOUT RESULTS")
    print(f"{'-'*40}")
    print(f"  RRMSE Total: {rrmse_total:.4f}")
    print(f"  RRMSE U_x:   {rrmse_comp.get('U_x', 0):.4f}")
    print(f"  RRMSE U_y:   {rrmse_comp.get('U_y', 0):.4f}")

    # Save metrics
    metrics = {
        'RRMSE_total': float(rrmse_total),
        'RRMSE_Ux': float(rrmse_comp.get('U_x', 0)),
        'RRMSE_Uy': float(rrmse_comp.get('U_y', 0)),
        'total_error': float(np.mean(list(rrmse_comp.values()))),
        'denormalization_method': denorm_method,
        'model': 'MeshGraphNet',
        'num_parameters': num_params,
        'num_simulations': len(simulations),
        'rollout_steps': args.rollout_steps,
        'simulation_metrics': sim_metrics
    }
    with open(output_path / 'rollout_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path / 'rollout_metrics.json'}")

    # ==================== DASHBOARD ====================
    print("Creating dashboard...")
    fig = plot_dashboard(sim_metrics, norm_method=denorm_method)
    if fig:
        fig.savefig(output_path / 'rollout_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved dashboard to {output_path / 'rollout_dashboard.png'}")

    # ==================== GIFS ====================
    if args.create_gifs and simulations:
        print(f"\nCreating GIF visualizations...")
        selected = select_representative_simulations(sim_metrics, args.num_viz_simulations)
        print(f"Selected simulations: {selected}")

        for i, idx in enumerate(selected, 1):
            print(f"\n[{i}/{len(selected)}] Creating GIFs for simulation {idx}...")
            sim = simulations[idx]
            seq_pred = all_preds_phys[idx]
            seq_targ = all_targs_phys[idx]
            elements = sim[0].elements.cpu().numpy() if hasattr(sim[0], 'elements') else None

            if elements is None:
                print(f"  Skipping GIFs for sim {idx}: no element data")
                continue

            # For viz, use simulation[1:] to align with predictions
            sim_for_viz = sim[1:len(seq_pred)+1]
            precomputed = precompute_visualization_data(sim_for_viz, seq_pred, seq_targ, elements)
            create_gifs(precomputed, seq_pred, seq_targ, elements,
                       f'simulation_{idx}', output_path,
                       fps=args.gif_fps, frame_skip=args.gif_frame_skip)

    print(f"\n{'='*60}")
    print(f"✅ MeshGraphNet evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()