#!/usr/bin/env python3
"""
GPARC Evaluation Script - Scheduled Sampling Model
===================================================
Evaluates models trained with scheduled sampling.
Supports both ROLLOUT and SNAPSHOT evaluation modes.
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

from utilities.featureextractor import GraphConvFeatureExtractorV2
from differentiator.differentiator import ElastoPlasticDifferentiator
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from models.globalelasto import GPARC_ElastoPlastic_Numerical

warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# NORMALIZATION UTILITIES (mirroring training script)
# ==============================================================================

def load_normalization_stats(data_dir):
    """
    Load normalization statistics from the data directory.
    Mirrors the training script's load_normalization_stats().
    """
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
        print("   Trying output_dir for normalization_stats.json...")
        return None


def load_normalization_stats_from_checkpoint_dir(checkpoint_dir):
    """
    Try loading normalization_stats.json from the model checkpoint directory.
    (Training script copies it there.)
    """
    stats_file = Path(checkpoint_dir) / "normalization_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"\n✓ Loaded normalization stats from checkpoint dir: {stats_file}")
        print(f"  Method: {stats.get('normalization_method', 'unknown')}")
        return stats
    return None


def get_pos_normalization_params(norm_stats):
    """
    Extract position normalization parameters.
    Mirrors the training script's get_pos_normalization_params().
    """
    if norm_stats is None:
        print("  ⚠️  No norm stats — using hardcoded z-score defaults")
        return [97.2165, 50.2759], [59.3803, 28.4965]
    
    pos_stats = norm_stats['position']
    pos_mean = [pos_stats['x_pos']['mean'], pos_stats['y_pos']['mean']]
    pos_std = [pos_stats['x_pos']['std'], pos_stats['y_pos']['std']]
    return pos_mean, pos_std


# ==============================================================================
# EROSION HANDLING (Ground Truth Only)
# ==============================================================================

EROSION_THRESHOLD = 0.5


def get_erosion_mask(data, num_elements):
    """Get boolean mask of eroded elements from data."""
    if hasattr(data, 'x_element') and data.x_element is not None:
        erosion_status = data.x_element.cpu().numpy().flatten()
        eroded_mask = erosion_status < EROSION_THRESHOLD
    else:
        eroded_mask = np.zeros(num_elements, dtype=bool)
    return eroded_mask


def get_valid_node_mask(elements, eroded_mask):
    """Get mask of nodes belonging to at least one valid element."""
    valid_elements = elements[~eroded_mask]
    if len(valid_elements) == 0:
        return np.zeros(elements.max() + 1, dtype=bool)
    valid_nodes = np.unique(valid_elements.flatten())
    valid_node_mask = np.zeros(elements.max() + 1, dtype=bool)
    valid_node_mask[valid_nodes] = True
    return valid_node_mask


# ==============================================================================
# FAST RENDERING
# ==============================================================================

def precompute_element_polygons(pos, elements):
    """Pre-compute polygon vertices for all elements."""
    return pos[elements]


def render_mesh_fast(ax, poly_verts, node_values, elements, eroded_mask,
                     vmin, vmax, cmap_obj, norm, show_eroded=False):
    """Fast mesh rendering using pre-computed polygon vertices."""
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
# PRECOMPUTATION
# ==============================================================================

def precompute_visualization_data(simulation, seq_pred, seq_targ, elements):
    """Pre-compute all visualization data in a single pass."""
    max_steps = min(len(seq_pred), len(seq_targ), len(simulation))
    num_elements = len(elements)
    
    pos_ref = simulation[0].pos.cpu().numpy()
    poly_verts_ref = precompute_element_polygons(pos_ref, elements)
    
    erosion_masks = []
    erosion_counts = []
    valid_node_masks = []
    
    for t in range(max_steps):
        eroded_mask = get_erosion_mask(simulation[t], num_elements)
        erosion_masks.append(eroded_mask)
        erosion_counts.append(eroded_mask.sum())
        valid_node_masks.append(get_valid_node_mask(elements, eroded_mask))
    
    disp_max = 0
    error_max = 0
    Ux_min, Ux_max = 0, 0
    Uy_min, Uy_max = 0, 0
    
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
        'max_steps': max_steps,
        'pos_ref': pos_ref,
        'x_ref': x_ref,
        'y_ref': y_ref,
        'poly_verts_ref': poly_verts_ref,
        'erosion_masks': erosion_masks,
        'erosion_counts': erosion_counts,
        'valid_node_masks': valid_node_masks,
        'disp_max': disp_max,
        'error_max': error_max,
        'Ux_range': (Ux_min, Ux_max),
        'Uy_range': (Uy_min, Uy_max),
        'camera_ref': camera_ref,
        'camera_def': camera_def,
    }


# ==============================================================================
# GIF CREATION FUNCTIONS
# ==============================================================================

def _create_reference_gif(frames, precomputed, seq_pred, seq_targ, elements,
                          case_name, output_dir, fps, eval_mode='rollout'):
    """Create reference configuration GIF."""
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
            ax.clear()
            ax.set_xlim(camera[0], camera[1])
            ax.set_ylim(camera[2], camera[3])
            ax.set_aspect('equal')
            ax.axis('off')
        
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        d_targ = np.sqrt(U_targ[:, 0]**2 + U_targ[:, 1]**2)
        d_pred = np.sqrt(U_pred[:, 0]**2 + U_pred[:, 1]**2)
        eroded_mask = erosion_masks[frame]
        n_eroded = eroded_mask.sum()
        
        render_mesh_fast(axes[0], poly_verts, d_targ, elements, eroded_mask,
                        0, disp_max, cmap, norm, show_eroded=True)
        title = f'Target (t={frame})'
        if n_eroded > 0:
            title += f' [{n_eroded} eroded]'
        axes[0].set_title(title, fontsize=12)
        
        render_mesh_fast(axes[1], poly_verts, d_pred, elements, eroded_mask,
                        0, disp_max, cmap, norm, show_eroded=True)
        axes[1].set_title(f'Prediction (t={frame})', fontsize=12)
        
        fig.suptitle(f'Reference Config ({mode_label}): {case_name}', fontsize=14)
        return axes.tolist()
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(Path(output_dir) / f'reference_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def _create_deformed_gif(frames, precomputed, seq_pred, seq_targ, elements,
                         case_name, output_dir, fps, eval_mode='rollout'):
    """Create deformed configuration GIF."""
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
            ax.clear()
            ax.set_xlim(camera[0], camera[1])
            ax.set_ylim(camera[2], camera[3])
            ax.set_aspect('equal')
            ax.axis('off')
        
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        
        pos_targ_def = np.column_stack([x_ref + U_targ[:, 0], y_ref + U_targ[:, 1]])
        pos_pred_def = np.column_stack([x_ref + U_pred[:, 0], y_ref + U_pred[:, 1]])
        
        poly_verts_targ = precompute_element_polygons(pos_targ_def, elements)
        poly_verts_pred = precompute_element_polygons(pos_pred_def, elements)
        
        d_targ = np.sqrt(U_targ[:, 0]**2 + U_targ[:, 1]**2)
        d_pred = np.sqrt(U_pred[:, 0]**2 + U_pred[:, 1]**2)
        eroded_mask = erosion_masks[frame]
        n_eroded = eroded_mask.sum()
        
        render_mesh_fast(axes[0], poly_verts_targ, d_targ, elements, eroded_mask,
                        0, disp_max, cmap, norm, show_eroded=False)
        title = f'Target (t={frame})'
        if n_eroded > 0:
            title += f' [{n_eroded} eroded]'
        axes[0].set_title(title, fontsize=12)
        
        render_mesh_fast(axes[1], poly_verts_pred, d_pred, elements, eroded_mask,
                        0, disp_max, cmap, norm, show_eroded=False)
        axes[1].set_title(f'Prediction (t={frame})', fontsize=12)
        
        fig.suptitle(f'Deformed Config ({mode_label}): {case_name}', fontsize=14)
        return axes.tolist()
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(Path(output_dir) / f'deformed_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def _create_error_gif(frames, precomputed, seq_pred, seq_targ, elements,
                      case_name, output_dir, fps, eval_mode='rollout'):
    """Create error visualization GIF."""
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
        ax.clear()
        ax.set_xlim(camera[0], camera[1])
        ax.set_ylim(camera[2], camera[3])
        ax.set_aspect('equal')
        ax.axis('off')
        
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        error_mag = np.sqrt((U_targ[:, 0] - U_pred[:, 0])**2 +
                           (U_targ[:, 1] - U_pred[:, 1])**2)
        eroded_mask = erosion_masks[frame]
        
        render_mesh_fast(ax, poly_verts, error_mag, elements, eroded_mask,
                        0, error_max, cmap, norm, show_eroded=True)
        
        n_eroded = eroded_mask.sum()
        title = f'Prediction Error - t={frame}'
        if n_eroded > 0:
            title += f' [{n_eroded} eroded]'
        ax.set_title(title, fontsize=14)
        fig.suptitle(f'Error ({mode_label}): {case_name}', fontsize=14)
        return [ax]
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(Path(output_dir) / f'error_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def _create_component_gif(frames, precomputed, seq_pred, seq_targ, elements,
                          case_name, output_dir, fps, component=0, eval_mode='rollout'):
    """Create component (Ux or Uy) visualization GIF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    
    comp_name = 'U_x' if component == 0 else 'U_y'
    comp_range = precomputed['Ux_range'] if component == 0 else precomputed['Uy_range']
    vmin, vmax = comp_range
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps.get_cmap('RdBu_r')
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label(comp_name, fontsize=11)
    
    camera = precomputed['camera_ref']
    poly_verts = precomputed['poly_verts_ref']
    erosion_masks = precomputed['erosion_masks']
    
    mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'
    
    def animate(frame_idx):
        frame = frames[frame_idx]
        for ax in axes:
            ax.clear()
            ax.set_xlim(camera[0], camera[1])
            ax.set_ylim(camera[2], camera[3])
            ax.set_aspect('equal')
            ax.axis('off')
        
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        eroded_mask = erosion_masks[frame]
        
        render_mesh_fast(axes[0], poly_verts, U_targ[:, component], elements, eroded_mask,
                        vmin, vmax, cmap, norm, show_eroded=True)
        axes[0].set_title(f'Target (t={frame})', fontsize=12)
        
        render_mesh_fast(axes[1], poly_verts, U_pred[:, component], elements, eroded_mask,
                        vmin, vmax, cmap, norm, show_eroded=True)
        axes[1].set_title(f'Prediction (t={frame})', fontsize=12)
        
        title = 'X-Displacement' if component == 0 else 'Y-Displacement'
        fig.suptitle(f'{title} ({mode_label}): {case_name}', fontsize=14)
        return axes.tolist()
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    suffix = 'ux_component' if component == 0 else 'uy_component'
    anim.save(Path(output_dir) / f'{suffix}_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def _create_erosion_plot(precomputed, case_name, output_dir):
    """Create erosion progression plot."""
    erosion_counts = precomputed['erosion_counts']
    max_steps = precomputed['max_steps']
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(max_steps), erosion_counts, 'r-', linewidth=2, marker='o', markersize=4)
    ax.fill_between(range(max_steps), erosion_counts, alpha=0.3, color='red')
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Eroded Elements', fontsize=12)
    ax.set_title(f'Element Erosion Progression: {case_name}', fontsize=14)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max_steps - 1)
    ax.set_ylim(0, max(erosion_counts) * 1.1 + 1)
    
    fig.savefig(Path(output_dir) / f'erosion_{case_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_single_gif(gif_type, precomputed, seq_pred, seq_targ, elements,
                      case_name, output_dir, fps, frame_skip, eval_mode='rollout'):
    """Create a single GIF."""
    max_steps = precomputed['max_steps']
    frames = list(range(0, max_steps, frame_skip))
    
    if gif_type == 'reference':
        _create_reference_gif(frames, precomputed, seq_pred, seq_targ, elements,
                             case_name, output_dir, fps, eval_mode)
    elif gif_type == 'deformed':
        _create_deformed_gif(frames, precomputed, seq_pred, seq_targ, elements,
                            case_name, output_dir, fps, eval_mode)
    elif gif_type == 'error':
        _create_error_gif(frames, precomputed, seq_pred, seq_targ, elements,
                         case_name, output_dir, fps, eval_mode)
    elif gif_type == 'ux':
        _create_component_gif(frames, precomputed, seq_pred, seq_targ, elements,
                             case_name, output_dir, fps, component=0, eval_mode=eval_mode)
    elif gif_type == 'uy':
        _create_component_gif(frames, precomputed, seq_pred, seq_targ, elements,
                             case_name, output_dir, fps, component=1, eval_mode=eval_mode)


# ==============================================================================
# METRICS FUNCTIONS
# ==============================================================================

def compute_plaid_rrmse(predictions, references, valid_masks=None):
    """
    Compute PLAID RRMSE for vector field outputs.
    
    Paper formula:
        RRMSE_f = sqrt( (1/n*) * sum_i [ (1/N^i) * ||f_ref^i - f_pred^i||_2^2 / ||f_ref^i||_inf^2 ] )
    
    Key: denominator is per-sample (inside the sum), not averaged separately.
    """
    if len(predictions) == 0:
        return 0.0
    
    n_samples = 0
    ratio_sum = 0.0
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if valid_masks is not None and valid_masks[i] is not None:
            mask = valid_masks[i]
            pred = pred[mask]
            ref = ref[mask]
        
        if len(pred) == 0:
            continue
        
        n_nodes = pred.shape[0]
        ref_inf_norm = np.max(np.abs(ref))
        
        if ref_inf_norm == 0:
            continue
        
        mse = np.sum((pred - ref) ** 2) / n_nodes
        ratio_sum += mse / (ref_inf_norm ** 2)
        n_samples += 1
    
    if n_samples == 0:
        return float('inf')
    
    return np.sqrt(ratio_sum / n_samples)


def compute_plaid_rrmse_per_component(predictions, references, valid_masks=None):
    """
    Compute PLAID RRMSE per component (U_x, U_y independently).
    
    Each component treated as a separate field with per-sample normalization:
        RRMSE_f = sqrt( (1/n*) * sum_i [ (1/N^i) * ||f_ref^i - f_pred^i||_2^2 / ||f_ref^i||_inf^2 ] )
    """
    if len(predictions) == 0:
        return {'U_x': 0.0, 'U_y': 0.0}
    
    n_components = predictions[0].shape[1]
    component_names = ['U_x', 'U_y']
    rrmse_per_component = {}
    
    for comp_idx in range(n_components):
        n_samples = 0
        ratio_sum = 0.0
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if valid_masks is not None and valid_masks[i] is not None:
                mask = valid_masks[i]
                pred_c = pred[mask, comp_idx]
                ref_c = ref[mask, comp_idx]
            else:
                pred_c = pred[:, comp_idx]
                ref_c = ref[:, comp_idx]
            
            if len(pred_c) == 0:
                continue
            
            n_nodes = pred_c.shape[0]
            ref_inf_norm = np.max(np.abs(ref_c))
            
            if ref_inf_norm == 0:
                continue
            
            mse = np.sum((pred_c - ref_c) ** 2) / n_nodes
            ratio_sum += mse / (ref_inf_norm ** 2)
            n_samples += 1
        
        rrmse = np.sqrt(ratio_sum / n_samples) if n_samples > 0 else float('inf')
        rrmse_per_component[component_names[comp_idx]] = float(rrmse)
    
    return rrmse_per_component


def select_representative_simulations(results_dict, n_samples=3, selection_mode='representative'):
    """Select simulations for visualization based on performance."""
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
    elif selection_mode == 'random':
        import random
        indices = [s[0] for s in sims]
        return random.sample(indices, min(n_samples, len(indices)))
    else:
        selected = []
        if len(sims) >= 1:
            selected.append(sims[0][0])
        if len(sims) >= 2:
            selected.append(sims[len(sims)//2][0])
        if len(sims) >= 3:
            selected.append(sims[-1][0])
        return selected[:n_samples]


# ==============================================================================
# EVALUATOR CLASS
# ==============================================================================

class ElastoPlasticEvaluator:
    """Evaluator for G-PARC models with scheduled sampling support."""
    
    def __init__(self, model, device='cpu', denormalization_params=None, norm_stats=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.denorm_params = denormalization_params
        self.norm_stats = norm_stats
        self.var_names = ['U_x', 'U_y']
        self.simulation_metrics = []

    def load_denormalization_params(self, metadata_file):
        """Load denormalization parameters from metadata file (z-score legacy)."""
        if not Path(metadata_file).exists():
            return
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        self.denorm_params = {}
        norm_params = metadata.get('original_metadata', {}).get('normalization_statistics',
                               metadata.get('normalization_statistics', {}))
        for var in self.var_names:
            if var in norm_params:
                self.denorm_params[var] = norm_params[var]

    def denormalize_predictions(self, normalized_data, method='global_max'):
        """Convert normalized predictions to physical units."""
        if method == 'none':
            return normalized_data
        
        if method == 'global_max':
            if self.norm_stats is None:
                print("  ⚠️  No norm_stats for global_max denormalization — returning raw")
                return normalized_data
            
            disp_stats = self.norm_stats.get('displacement', {})
            max_disp = disp_stats.get('max_displacement', 1.0)
            physical_data = normalized_data * max_disp
            return physical_data
        
        elif method == 'zscore':
            if self.denorm_params is None:
                return normalized_data
            physical_data = np.zeros_like(normalized_data)
            for i, var_name in enumerate(self.var_names):
                if var_name not in self.denorm_params:
                    physical_data[:, i] = normalized_data[:, i]
                    continue
                params = self.denorm_params[var_name]
                mean, std = params.get('mean', 0.0), params.get('std', 1.0)
                physical_data[:, i] = normalized_data[:, i] * std + mean
            return physical_data
        
        else:
            return normalized_data

    def _prep_simulation(self, simulation):
        """Move Data objects to device and extract mesh_id."""
        simulation = [d.to(self.device) for d in simulation]
        mid = getattr(simulation[0], "mesh_id", None)
        if mid is None:
            raise ValueError("Data object missing mesh_id")
        if torch.is_tensor(mid):
            mesh_id_int = int(mid.view(-1)[0].item())
        else:
            mesh_id_int = int(mid)
        return simulation, mesh_id_int

    def _ensure_mesh_cached(self, initial_data, mesh_id_int):
        """Reinitialize MLS weights/caches when mesh changes."""
        deriv_solver = self.model.derivative_solver
        real_solver = deriv_solver.solver if hasattr(deriv_solver, "solver") else deriv_solver
        
        if getattr(real_solver, "_active_mesh_id", None) == mesh_id_int:
            return
        
        if hasattr(real_solver, "clear_cache"):
            real_solver.clear_cache()
        
        for attr in ("geo_cache", "weights_cache", "damping_cache"):
            if hasattr(real_solver, attr):
                obj = getattr(real_solver, attr)
                if hasattr(obj, "clear"):
                    obj.clear()
        
        if hasattr(real_solver, "initialize_weights"):
            real_solver.initialize_weights(initial_data)
        
        real_solver._active_mesh_id = mesh_id_int

    def generate_rollout(self, initial_data, simulation, rollout_steps):
        """Generate autoregressive rollout predictions."""
        predictions = []
        F_prev = simulation[0].x[:, self.model.num_static_feats:].clone()
    
        for step in range(rollout_steps):
            data_t = simulation[step]
            static_feats = data_t.x[:, :self.model.num_static_feats]
            edge_index = data_t.edge_index
            
            if hasattr(data_t, "mesh_id"):
                edge_index.mesh_id = data_t.mesh_id
    
            F_current = F_prev.clone()
            F_pred = self.model.step(
                static_feats=static_feats,
                dynamic_state=F_current,
                edge_index=edge_index,
                dt=1.0
            )
    
            predictions.append(F_pred)
            F_prev = F_pred
    
        return predictions

    def generate_snapshot_predictions(self, simulation, num_steps):
        """
        Generate snapshot (single-step) predictions.
        
        For each timestep t, feed the model the GROUND TRUTH state at t
        and predict the state at t+1. No error accumulation.
        
        This matches the PLAID benchmark evaluation protocol where each
        (simulation, timestep) is treated as an independent sample.
        """
        predictions = []
        
        for step in range(num_steps):
            data_t = simulation[step]
            static_feats = data_t.x[:, :self.model.num_static_feats]
            edge_index = data_t.edge_index
            
            if hasattr(data_t, "mesh_id"):
                edge_index.mesh_id = data_t.mesh_id
            
            # Use GROUND TRUTH dynamic state at time t
            F_gt = data_t.x[:, self.model.num_static_feats:].clone()
            
            # Predict t+1 from GT at t
            F_pred = self.model.step(
                static_feats=static_feats,
                dynamic_state=F_gt,
                edge_index=edge_index,
                dt=1.0
            )
            
            predictions.append(F_pred)
        
        return predictions

    def evaluate_snapshot_predictions(self, simulations, normalization_method='global_max'):
        """
        Evaluate model in snapshot mode (single-step from GT).
        
        Each timestep is predicted independently from ground truth input.
        Target for step t is simulation[t+1].y (or simulation[t].y depending on data format).
        """
        results = {
            'predictions_physical': [],
            'targets_physical': [],
            'metadata': [],
            'erosion_masks': []
        }
        
        self.simulation_metrics = []
        
        with torch.no_grad():
            for sim_idx, simulation in enumerate(tqdm(simulations, desc="Snapshot predictions")):
                try:
                    simulation, mesh_id_int = self._prep_simulation(simulation)
                    initial_data = simulation[0]
                    elements = initial_data.elements.detach().cpu().numpy()
                    
                    self._ensure_mesh_cached(initial_data, mesh_id_int)
                    
                    # For snapshot: predict from t=0..T-2, target is t=1..T-1
                    # simulation[t].y contains the target for that timestep
                    # (i.e., the state at t+1 that we want to predict)
                    num_steps = len(simulation) - 1  # Can predict from all but last
                    
                    preds_raw = self.generate_snapshot_predictions(simulation, num_steps)
                    
                    # Collect predictions and targets
                    preds_norm = []
                    targs_norm = []
                    erosion_masks_sim = []
                    
                    for t in range(num_steps):
                        pred = preds_raw[t]
                        
                        # Stability check
                        if not (torch.isfinite(pred).all() and pred.abs().max() < 50.0):
                            print(f"  Sim {sim_idx}, step {t}: non-finite prediction, skipping")
                            continue
                        
                        preds_norm.append(pred.cpu().numpy())
                        
                        # Target is simulation[t].y (the GT for the next state)
                        targs_norm.append(simulation[t].y.cpu().numpy())
                        
                        # Erosion from the TARGET timestep (t+1)
                        erosion_masks_sim.append(
                            get_erosion_mask(simulation[t + 1], len(elements))
                        )
                    
                    if len(preds_norm) == 0:
                        print(f"  Skipping sim {sim_idx}: All predictions unstable")
                        continue
                    
                    # Denormalize to physical units
                    preds_phys = [self.denormalize_predictions(p, normalization_method) for p in preds_norm]
                    targs_phys = [self.denormalize_predictions(t, normalization_method) for t in targs_norm]
                    
                    results['predictions_physical'].append(preds_phys)
                    results['targets_physical'].append(targs_phys)
                    results['erosion_masks'].append(erosion_masks_sim)
                    
                    metadata = {
                        'simulation_idx': sim_idx,
                        'case_name': f'simulation_{sim_idx}',
                        'num_snapshots': len(preds_norm),
                        'num_nodes': initial_data.num_nodes,
                        'num_elements': len(elements),
                        'max_eroded': max(m.sum() for m in erosion_masks_sim) if erosion_masks_sim else 0
                    }
                    results['metadata'].append(metadata)
                    
                    # Per-simulation metrics
                    valid_node_masks = [get_valid_node_mask(elements, em) for em in erosion_masks_sim]
                    
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
                    
                    self.simulation_metrics.append({
                        'metadata': metadata,
                        'overall_physical': {'rmse': rmse, 'r2': r2}
                    })
                    
                except Exception as e:
                    print(f"Error processing simulation {sim_idx}: {e}")
                    import traceback
                    traceback.print_exc()
        
        results['simulation_metrics'] = self.simulation_metrics
        return results

    def evaluate_rollout_predictions(self, simulations, rollout_steps=10, normalization_method='global_max'):
        """Evaluate model on rollout predictions."""
        results = {
            'predictions_physical': [],
            'targets_physical': [],
            'metadata': [],
            'erosion_masks': []
        }
        
        self.simulation_metrics = []
        
        with torch.no_grad():
            for sim_idx, simulation in enumerate(tqdm(simulations, desc="Generating rollout predictions")):
                try:
                    simulation, mesh_id_int = self._prep_simulation(simulation)
                    initial_data = simulation[0]
                    elements = initial_data.elements.detach().cpu().numpy()
                    
                    self._ensure_mesh_cached(initial_data, mesh_id_int)
                    
                    actual_steps = min(rollout_steps, len(simulation))
                    preds_raw = self.generate_rollout(initial_data, simulation, actual_steps)
                    
                    preds_norm = []
                    for p in preds_raw:
                        if torch.isfinite(p).all() and p.abs().max() < 50.0:
                            preds_norm.append(p)
                        else:
                            break
                    
                    if len(preds_norm) == 0:
                        print(f"  Skipping sim {sim_idx}: Immediate explosion")
                        continue
                    
                    actual_steps = len(preds_norm)
                    
                    targs_norm = [simulation[i].y.cpu().numpy() for i in range(actual_steps)]
                    erosion_masks = [get_erosion_mask(simulation[i], len(elements)) for i in range(actual_steps)]
                    
                    preds_phys = [self.denormalize_predictions(p.cpu().numpy(), normalization_method) for p in preds_norm]
                    targs_phys = [self.denormalize_predictions(t, normalization_method) for t in targs_norm]
                    
                    results['predictions_physical'].append(preds_phys)
                    results['targets_physical'].append(targs_phys)
                    results['erosion_masks'].append(erosion_masks)
                    
                    metadata = {
                        'simulation_idx': sim_idx,
                        'case_name': f'simulation_{sim_idx}',
                        'rollout_length': len(preds_norm),
                        'num_nodes': initial_data.num_nodes,
                        'num_elements': len(elements),
                        'max_eroded': max(m.sum() for m in erosion_masks)
                    }
                    results['metadata'].append(metadata)
                    
                    valid_node_masks = [get_valid_node_mask(elements, em) for em in erosion_masks]
                    
                    all_p, all_t = [], []
                    for t in range(len(preds_phys)):
                        mask = valid_node_masks[t]
                        if mask.sum() > 0:
                            all_p.append(preds_phys[t][mask])
                            all_t.append(targs_phys[t][mask])
                    
                    if len(all_p) > 0:
                        all_p = np.concatenate(all_p, axis=0)
                        all_t = np.concatenate(all_t, axis=0)
                        rmse = float(np.sqrt(mean_squared_error(all_t, all_p)))
                        r2 = float(r2_score(all_t, all_p))
                    else:
                        rmse, r2 = float('inf'), 0.0
                    
                    self.simulation_metrics.append({
                        'metadata': metadata,
                        'overall_physical': {'rmse': rmse, 'r2': r2}
                    })
                    
                except Exception as e:
                    print(f"Error processing simulation {sim_idx}: {e}")
                    import traceback
                    traceback.print_exc()
        
        results['simulation_metrics'] = self.simulation_metrics
        return results

    def create_mesh_deformation_gifs(self, simulations, results_dict, seq_idx, output_dir,
                                      fps=10, frame_skip=1, parallel=True, n_workers=4,
                                      gif_types=None, eval_mode='rollout'):
        """Create GIFs with fast parallel rendering."""
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
        
        if gif_types is None:
            gif_types = ['reference', 'deformed', 'error', 'ux', 'uy']
        
        print(f"\n{'='*70}")
        print(f"Creating GIFs for {case_name} ({eval_mode} mode)")
        print(f"  Elements: {len(elements)}, Timesteps: {max_steps}")
        print(f"  Frame skip: {frame_skip}, FPS: {fps}, Parallel: {parallel}")
        print(f"{'='*70}")
        
        print("  Pre-computing visualization data...")
        precomputed = precompute_visualization_data(sim_for_viz, seq_pred, seq_targ, elements)
        print(f"  Max eroded: {max(precomputed['erosion_counts'])}")
        
        if parallel and n_workers > 1:
            print(f"  Creating {len(gif_types)} GIFs with {n_workers} workers...")
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for gif_type in gif_types:
                    future = executor.submit(
                        create_single_gif,
                        gif_type, precomputed, seq_pred, seq_targ, elements,
                        case_name, output_dir, fps, frame_skip, eval_mode
                    )
                    futures.append((gif_type, future))
                
                for gif_type, future in futures:
                    try:
                        future.result()
                        print(f"    ✓ {gif_type}_{case_name}.gif")
                    except Exception as e:
                        print(f"    ✗ {gif_type}_{case_name}.gif: {e}")
        else:
            print(f"  Creating {len(gif_types)} GIFs sequentially...")
            for gif_type in gif_types:
                create_single_gif(gif_type, precomputed, seq_pred, seq_targ, elements,
                                 case_name, output_dir, fps, frame_skip, eval_mode)
                print(f"    ✓ {gif_type}_{case_name}.gif")
        
        if max(precomputed['erosion_counts']) > 0:
            _create_erosion_plot(precomputed, case_name, output_dir)
            print(f"    ✓ erosion_{case_name}.png")
        
        print(f"{'='*70}\n")

    def compute_plaid_benchmark_metrics(self, predictions_physical, targets_physical, erosion_masks=None):
        """Compute PLAID benchmark metrics."""
        if not predictions_physical:
            return {}
        
        all_pred, all_targ, all_masks = [], [], []
        
        for i, (seq_p, seq_t) in enumerate(zip(predictions_physical, targets_physical)):
            for t, (p, tg) in enumerate(zip(seq_p, seq_t)):
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

    def plot_comprehensive_metrics(self, results_dict, figsize=(18, 10), eval_mode='rollout'):
        """Create comprehensive metrics dashboard."""
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
        ax1.axvline(np.mean(r2_scores), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(r2_scores):.3f}')
        ax1.set_xlabel('R² Score', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('R² Score Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(rmse_values, bins=15, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(rmse_values), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(rmse_values):.3e}')
        ax2.set_xlabel('RMSE', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('RMSE Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        norm_method = self.norm_stats.get('normalization_method', 'unknown') if self.norm_stats else 'unknown'
        stats_text = f"""
{mode_label.upper()} METRICS
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

Simulations: {len(self.simulation_metrics)}
Avg Max Eroded: {np.mean(max_eroded):.0f}
        """
        ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax4 = fig.add_subplot(gs[1, :2])
        scatter = ax4.scatter(r2_scores, rmse_values, c=max_eroded, cmap='YlOrRd',
                             s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('R² Score', fontsize=11)
        ax4.set_ylabel('RMSE', fontsize=11)
        ax4.set_title('R² vs RMSE (colored by max eroded elements)', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax4).set_label('Max Eroded Elements', fontsize=10)
        
        ax5 = fig.add_subplot(gs[1, 2])
        colors = ['green' if r2 > 0.8 else 'orange' if r2 > 0.5 else 'red' for r2 in r2_scores]
        ax5.barh(range(len(sim_indices)), r2_scores, color=colors, edgecolor='black', alpha=0.7)
        ax5.set_xlabel('R² Score', fontsize=11)
        ax5.set_ylabel('Simulation Index', fontsize=11)
        ax5.set_title('Performance by Simulation', fontsize=12, fontweight='bold')
        ax5.axvline(0.8, color='green', linestyle='--', alpha=0.5)
        ax5.axvline(0.5, color='orange', linestyle='--', alpha=0.5)
        ax5.grid(alpha=0.3, axis='x')
        
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.scatter(max_eroded, r2_scores, c='steelblue', s=80, alpha=0.7, edgecolors='black')
        ax6.set_xlabel('Max Eroded Elements', fontsize=11)
        ax6.set_ylabel('R² Score', fontsize=11)
        ax6.set_title('R² vs Erosion Level', fontsize=12, fontweight='bold')
        ax6.grid(alpha=0.3)
        
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.plot(range(len(rmse_values)), rmse_values, marker='o', linestyle='-',
                linewidth=1.5, markersize=6, color='coral', alpha=0.7)
        ax7.set_xlabel('Simulation Index', fontsize=11)
        ax7.set_ylabel('RMSE', fontsize=11)
        ax7.set_title('RMSE by Simulation', fontsize=12, fontweight='bold')
        ax7.grid(alpha=0.3)
        ax7.axhline(np.mean(rmse_values), color='red', linestyle='--', alpha=0.5, label='Mean')
        ax7.legend(fontsize=9)
        
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
        
        fig.suptitle(f'Model Performance Analysis ({mode_label} - Scheduled Sampling)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig


# ==============================================================================
# MAIN EVALUATION FUNCTION
# ==============================================================================

def load_test_simulations(test_dir, test_files, pattern, max_files):
    """Load test simulations with mesh_id injection."""
    import re
    simulations = []
    paths = [Path(f) for f in test_files] if test_files else sorted(list(Path(test_dir).glob(pattern)))
    if max_files:
        paths = paths[:max_files]
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


def evaluate_elastoplastic(model_path, test_dir, test_files, output_dir, args):
    """Main evaluation function with global max normalization support."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # ==================== LOAD NORMALIZATION STATS ====================
    norm_stats = None
    
    if args.norm_stats_file and Path(args.norm_stats_file).exists():
        with open(args.norm_stats_file, 'r') as f:
            norm_stats = json.load(f)
        print(f"\n✓ Loaded normalization stats from --norm_stats_file: {args.norm_stats_file}")
    
    if norm_stats is None and test_dir:
        norm_stats = load_normalization_stats(test_dir)
    
    if norm_stats is None:
        model_dir = Path(model_path).parent
        norm_stats = load_normalization_stats_from_checkpoint_dir(model_dir)
    
    if norm_stats is None:
        print("\n⚠️  No normalization_stats.json found anywhere!")
        print("   Falling back to hardcoded z-score defaults")
        norm_stats = {
            'normalization_method': 'z_score',
            'position': {
                'x_pos': {'mean': 97.2165, 'std': 59.3803},
                'y_pos': {'mean': 50.2759, 'std': 28.4965}
            }
        }
    
    pos_mean, pos_std = get_pos_normalization_params(norm_stats)
    margin = 2.0
    
    norm_method = norm_stats.get('normalization_method', 'unknown')
    print(f"\n{'='*60}")
    print(f"NORMALIZATION CONFIG")
    print(f"{'='*60}")
    print(f"  Method: {norm_method}")
    print(f"  pos_mean: {pos_mean}")
    print(f"  pos_std: {pos_std}")
    if 'displacement' in norm_stats:
        print(f"  max_displacement: {norm_stats['displacement'].get('max_displacement', 'N/A')}")
    if 'position' in norm_stats:
        print(f"  max_position: {norm_stats['position'].get('max_position', 'N/A')}")
    
    if args.normalization_method == 'auto':
        if norm_method == 'global_max':
            denorm_method = 'global_max'
        elif norm_method in ('z_score', 'zscore'):
            denorm_method = 'zscore'
        else:
            denorm_method = 'none'
        print(f"  Auto-detected denorm method: {denorm_method}")
    else:
        denorm_method = args.normalization_method
        print(f"  Explicit denorm method: {denorm_method}")
    print(f"{'='*60}")
    
    # ==================== BUILD MODEL ====================
    print(f"\nBuilding G-PARC Scheduled Sampling Model...")
    
    max_position = None
    if norm_method == 'global_max' and 'position' in norm_stats:
        max_position = norm_stats['position'].get('max_position', None)
    
    #gradient_solver = SolveGradientsLST(
    #    pos_mean=pos_mean, pos_std=pos_std, boundary_margin=margin,
    #    norm_method=norm_method, max_position=max_position
    #)
    #laplacian_solver = SolveWeightLST2d(
    #    pos_mean=pos_mean, pos_std=pos_std, boundary_margin=margin,
    #    norm_method=norm_method, max_position=max_position
    #)

    print("\nInitializing MLS Operators...")
    gradient_solver = SolveGradientsLST(
        pos_mean=pos_mean,
        pos_std=pos_std,
        norm_method=norm_method,
        max_position=max_position
    )
    #        boundary_margin=margin,
    laplacian_solver = SolveWeightLST2d(
        pos_mean=pos_mean,
        pos_std=pos_std,
        norm_method=norm_method,
        max_position=max_position,
        min_neighbors=5,   # Damp Laplacian at nodes with <5 neighbors
        use_2hop_extension=False
    )
    
    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=args.num_static_feats,
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layer_norm=args.use_layer_norm,
        use_relative_pos=args.use_relative_pos
    )
    
    derivative_solver = ElastoPlasticDifferentiator(
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=args.feature_out_channels,
        list_strain_idx=list(range(args.num_dynamic_feats)),
        list_laplacian_idx=list(range(args.num_dynamic_feats)),
        spade_random_noise=False,
        heads=args.spade_heads,
        concat=args.spade_concat,
        dropout=args.spade_dropout,
        use_von_mises=args.use_von_mises,
        use_volumetric=args.use_volumetric,
        n_state_var=args.n_state_var,
        zero_init=args.zero_init
    )
    
    model = GPARC_ElastoPlastic_Numerical(
        derivative_solver_physics=derivative_solver,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        pos_mean=pos_mean,
        pos_std=pos_std,
        boundary_threshold=0.5,
        clamp_output=not args.no_clamp_output,
        norm_method=norm_method,
        max_position=max_position,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # ==================== LOAD DATA ====================
    print(f"\nLoading test simulations from: {test_dir}")
    simulations = load_test_simulations(test_dir, test_files, "*.pt", args.max_sequences)
    if not simulations:
        print("No simulations loaded!")
        return
    print(f"Loaded {len(simulations)} simulations")
    
    print("Initializing MLS operators...")
    try:
        first_sim = simulations[0][0].to(device)
        if hasattr(model.derivative_solver, 'initialize_weights'):
            model.derivative_solver.initialize_weights(first_sim)
    except Exception as e:
        print(f"Error initializing weights: {e}")
        return
    
    # ==================== CREATE EVALUATOR ====================
    evaluator = ElastoPlasticEvaluator(model, device, norm_stats=norm_stats)
    
    if test_dir and denorm_method == 'zscore':
        norm_file = Path(test_dir).parent / 'normalization_metadata.json'
        if norm_file.exists():
            evaluator.load_denormalization_params(norm_file)
            print(f"Loaded z-score denorm params from {norm_file}")
    
    # ==================== EVALUATION ====================
    eval_mode = args.eval_mode.lower()
    
    # -------------------- ROLLOUT --------------------
    if eval_mode in ['rollout', 'both']:
        print(f"\n{'='*60}")
        print(f"ROLLOUT EVALUATION (rollout_steps={args.rollout_steps})")
        print(f"  Denormalization: {denorm_method}")
        print(f"{'='*60}")
        
        rollout_results = evaluator.evaluate_rollout_predictions(
            simulations,
            rollout_steps=args.rollout_steps,
            normalization_method=denorm_method
        )
        
        rollout_metrics = evaluator.compute_plaid_benchmark_metrics(
            rollout_results['predictions_physical'],
            rollout_results['targets_physical'],
            rollout_results.get('erosion_masks')
        )
        
        print("\n" + "-" * 40)
        print("ROLLOUT RESULTS")
        print("-" * 40)
        
        def safe_fmt(val):
            if isinstance(val, (float, int, np.floating)):
                return f"{val:.4f}"
            return str(val)

        print(f"  RRMSE Total: {safe_fmt(rollout_metrics.get('RRMSE_total', 'N/A'))}")
        print(f"  RRMSE U_x:   {safe_fmt(rollout_metrics.get('RRMSE_Ux', 'N/A'))}")
        print(f"  RRMSE U_y:   {safe_fmt(rollout_metrics.get('RRMSE_Uy', 'N/A'))}")
        
        rollout_metrics['normalization_method'] = norm_method
        rollout_metrics['denormalization_method'] = denorm_method
        rollout_metrics['eval_mode'] = 'rollout'
        if 'displacement' in norm_stats:
            rollout_metrics['max_displacement'] = norm_stats['displacement'].get('max_displacement')
        
        with open(output_path / 'rollout_metrics.json', 'w') as f:
            json.dump(rollout_metrics, f, indent=2)
        print(f"\nSaved rollout metrics to {output_path / 'rollout_metrics.json'}")
        
        print("Creating rollout metrics dashboard...")
        rollout_fig = evaluator.plot_comprehensive_metrics(rollout_results, eval_mode='rollout')
        if rollout_fig:
            rollout_fig.savefig(output_path / 'rollout_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close(rollout_fig)
            print(f"Saved dashboard to {output_path / 'rollout_dashboard.png'}")
        
        if args.create_gifs:
            print(f"\nCreating rollout GIF visualizations...")
            selected_indices = select_representative_simulations(
                rollout_results,
                n_samples=args.num_viz_simulations,
                selection_mode=args.viz_selection_mode
            )
            
            print(f"Selected simulations: {selected_indices}")
            for i, idx in enumerate(selected_indices, 1):
                print(f"\n[{i}/{len(selected_indices)}] Creating rollout GIFs for simulation {idx}...")
                evaluator.create_mesh_deformation_gifs(
                    simulations, rollout_results, idx, output_path,
                    fps=args.gif_fps,
                    frame_skip=args.gif_frame_skip,
                    parallel=False,
                    n_workers=1,
                    gif_types=['reference', 'deformed', 'error'],
                    eval_mode='rollout'
                )
    
    # -------------------- SNAPSHOT --------------------
    if eval_mode in ['snapshot', 'both']:
        print(f"\n{'='*60}")
        print(f"SNAPSHOT EVALUATION (single-step from ground truth)")
        print(f"  Denormalization: {denorm_method}")
        print(f"  Each timestep predicted independently from GT input")
        print(f"{'='*60}")
        
        snapshot_results = evaluator.evaluate_snapshot_predictions(
            simulations,
            normalization_method=denorm_method
        )
        
        snapshot_metrics = evaluator.compute_plaid_benchmark_metrics(
            snapshot_results['predictions_physical'],
            snapshot_results['targets_physical'],
            snapshot_results.get('erosion_masks')
        )
        
        def safe_fmt(val):
            if isinstance(val, (float, int, np.floating)):
                return f"{val:.4f}"
            return str(val)
        
        print("\n" + "-" * 40)
        print("SNAPSHOT RESULTS")
        print("-" * 40)
        print(f"  RRMSE Total: {safe_fmt(snapshot_metrics.get('RRMSE_total', 'N/A'))}")
        print(f"  RRMSE U_x:   {safe_fmt(snapshot_metrics.get('RRMSE_Ux', 'N/A'))}")
        print(f"  RRMSE U_y:   {safe_fmt(snapshot_metrics.get('RRMSE_Uy', 'N/A'))}")
        print(f"  total_error:  {safe_fmt(snapshot_metrics.get('total_error', 'N/A'))}")
        
        n_total_samples = sum(len(seq) for seq in snapshot_results['predictions_physical'])
        print(f"\n  Total snapshot samples: {n_total_samples}")
        print(f"  (= {len(simulations)} simulations × ~{n_total_samples // max(len(simulations), 1)} timesteps)")
        
        snapshot_metrics['normalization_method'] = norm_method
        snapshot_metrics['denormalization_method'] = denorm_method
        snapshot_metrics['eval_mode'] = 'snapshot'
        snapshot_metrics['num_snapshot_samples'] = n_total_samples
        if 'displacement' in norm_stats:
            snapshot_metrics['max_displacement'] = norm_stats['displacement'].get('max_displacement')
        
        with open(output_path / 'snapshot_metrics.json', 'w') as f:
            json.dump(snapshot_metrics, f, indent=2)
        print(f"\nSaved snapshot metrics to {output_path / 'snapshot_metrics.json'}")
        
        print("Creating snapshot metrics dashboard...")
        snapshot_fig = evaluator.plot_comprehensive_metrics(snapshot_results, eval_mode='snapshot')
        if snapshot_fig:
            snapshot_fig.savefig(output_path / 'snapshot_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close(snapshot_fig)
            print(f"Saved dashboard to {output_path / 'snapshot_dashboard.png'}")
        
        if args.create_gifs:
            print(f"\nCreating snapshot GIF visualizations...")
            selected_indices = select_representative_simulations(
                snapshot_results,
                n_samples=args.num_viz_simulations,
                selection_mode=args.viz_selection_mode
            )
            
            print(f"Selected simulations: {selected_indices}")
            for i, idx in enumerate(selected_indices, 1):
                print(f"\n[{i}/{len(selected_indices)}] Creating snapshot GIFs for simulation {idx}...")
                evaluator.create_mesh_deformation_gifs(
                    simulations, snapshot_results, idx, output_path,
                    fps=args.gif_fps,
                    frame_skip=args.gif_frame_skip,
                    parallel=False,
                    n_workers=1,
                    gif_types=['reference', 'deformed', 'error'],
                    eval_mode='snapshot'
                )
    
    # ==================== COMPARISON (both mode) ====================
    if eval_mode == 'both':
        print(f"\n{'='*60}")
        print(f"SNAPSHOT vs ROLLOUT COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Metric':<20} {'Snapshot':>12} {'Rollout':>12} {'Ratio':>10}")
        print(f"  {'-'*54}")
        for key in ['RRMSE_total', 'RRMSE_Ux', 'RRMSE_Uy', 'total_error']:
            s_val = snapshot_metrics.get(key, 0)
            r_val = rollout_metrics.get(key, 0)
            ratio = r_val / s_val if s_val > 0 else float('inf')
            print(f"  {key:<20} {s_val:>12.4f} {r_val:>12.4f} {ratio:>10.1f}x")
        
        comparison = {
            'snapshot': snapshot_metrics,
            'rollout': rollout_metrics,
            'ratio': {
                key: rollout_metrics.get(key, 0) / max(snapshot_metrics.get(key, 0), 1e-12)
                for key in ['RRMSE_total', 'RRMSE_Ux', 'RRMSE_Uy']
            }
        }
        with open(output_path / 'comparison_metrics.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nSaved comparison to {output_path / 'comparison_metrics.json'}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="GPARC Evaluation - Global Max Normalization Support")
    
    # Paths
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--test_files", type=str, nargs='+')
    parser.add_argument("--output_dir", default="./eval_scheduled_sampling")
    parser.add_argument("--norm_stats_file", type=str, default=None,
                        help="Explicit path to normalization_stats.json (overrides auto-detection)")
    
    # Evaluation mode
    parser.add_argument("--eval_mode", type=str, default="rollout",
                        choices=['rollout', 'snapshot', 'both'])
    
    # Architecture (must match training!)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_layer_norm", action="store_true", default=False)
    parser.add_argument("--use_relative_pos", action="store_true", default=False)
    
    # Physics
    parser.add_argument("--no_clamp_output", action="store_true", default=False)
    parser.add_argument("--integrator", type=str, default="euler")
    parser.add_argument("--spade_heads", type=int, default=4)
    parser.add_argument("--spade_concat", action="store_true", default=False)
    parser.add_argument("--spade_dropout", type=float, default=0.1)
    parser.add_argument("--zero_init", action="store_true", default=False)
    
    # Dimensions
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=2)
    parser.add_argument("--n_state_var", type=int, default=0)
    parser.add_argument("--use_von_mises", action="store_true", default=False)
    parser.add_argument("--use_volumetric", action="store_true", default=False)
    
    # Eval settings
    parser.add_argument("--max_sequences", type=int, default=10)
    parser.add_argument("--rollout_steps", type=int, default=37)
    parser.add_argument("--normalization_method", default="auto",
                        choices=['auto', 'global_max', 'zscore', 'none'],
                        help="Denormalization method. 'auto' detects from normalization_stats.json")
    parser.add_argument("--create_gifs", action="store_true")
    
    # Viz settings
    parser.add_argument("--num_viz_simulations", type=int, default=3)
    parser.add_argument("--viz_selection_mode", type=str, default="representative")
    parser.add_argument("--gif_fps", type=int, default=10)
    parser.add_argument("--gif_frame_skip", type=int, default=1)
    
    args = parser.parse_args()
    evaluate_elastoplastic(args.model_path, args.test_dir, args.test_files, args.output_dir, args)


if __name__ == "__main__":
    main()