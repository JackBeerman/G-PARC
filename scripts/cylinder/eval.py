#!/usr/bin/env python3
"""
Evaluation & Visualization for G-PARCv2 Cylinder Flow
======================================================
Loads a trained checkpoint, runs autoregressive rollout on test simulations,
and generates:
  1. Per-variable comparison GIFs: Target | Prediction | |Error|
  2. All-variables combined GIF
  3. Rollout error growth plot (MSE vs timestep)
  4. Per-simulation metrics (MSE, RMSE, RRMSE, R²)

Rendering uses matplotlib tricontourf on the 2D (x,y) mesh positions,
which handles the unstructured cylinder mesh naturally.

Usage:
  python eval_cylinder_v2.py \
    --checkpoint /scratch/jtb3sud/gparcv2/cylinder/best_model.pth \
    --test_dir /standard/.../split_normalized/test \
    --output_dir /scratch/jtb3sud/gparcv2/cylinder/eval
"""

import argparse
import os
import sys
import json
import warnings
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.featureextractor import GraphConvFeatureExtractorV2
from differentiator.cylinder_nospade import CylinderDifferentiator
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from models.cylinder_gparcv2 import GPARC_Cylinder_V2
from data.Cylinder import StreamingKarmanDataset, get_simulation_ids

warnings.filterwarnings("ignore", category=UserWarning)

# Variable names after skipping [3,4,5] (vz, ωx, ωy)
CYLINDER_VAR_NAMES = ['pressure', 'velocity_x', 'velocity_y', 'vorticity_z']

# Colormaps per variable type
VAR_CMAPS = {
    'pressure': 'RdBu_r',
    'velocity_x': 'coolwarm',
    'velocity_y': 'coolwarm',
    'vorticity_z': 'RdBu_r',
}


# =========================================================================
# MODEL RECONSTRUCTION
# =========================================================================

def load_config(checkpoint_dir):
    """Load training config from checkpoint directory."""
    config_path = Path(checkpoint_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def create_model_from_config(config, device='cpu'):
    """Reconstruct model architecture from saved config."""
    num_static = config.get('num_static_feats', 3)
    num_dynamic = config.get('num_dynamic_feats', 4)
    skip_indices = config.get('skip_dynamic_indices', [3, 4, 5])
    velocity_indices = config.get('velocity_indices', [1, 2])
    hidden = config.get('hidden_channels', 128)
    fe_out = config.get('feature_out_channels', 128)
    num_layers = config.get('num_layers', 4)
    dropout = config.get('dropout', 0.0)
    diffusion_type = config.get('diffusion_type', 'mls')
    fusion_hidden = config.get('fusion_hidden_dim', 128)
    global_param_dim = config.get('global_param_dim', 1)
    global_embed_dim = config.get('global_embed_dim', 64)
    integrator = config.get('integrator', 'euler')

    # MLS operators
    gradient_solver = SolveGradientsLST(
        pos_mean=[0.0, 0.0], pos_std=[1.0, 1.0], norm_method='z_score',
    )
    laplacian_solver = SolveWeightLST2d(
        pos_mean=[0.0, 0.0], pos_std=[1.0, 1.0],
        norm_method='z_score', min_neighbors=5,
    )

    # Feature extractor
    fe_in = num_static + num_dynamic
    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=fe_in, hidden_channels=hidden, out_channels=fe_out,
        num_layers=num_layers, dropout=dropout,
        use_layer_norm=True, use_relative_pos=True,
    )

    # Differentiator
    derivative_solver = CylinderDifferentiator(
        num_static_feats=num_static, num_dynamic_feats=num_dynamic,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver, laplacian_solver=laplacian_solver,
        n_fe_features=fe_out, global_embed_dim=global_embed_dim,
        global_param_dim=global_param_dim, velocity_indices=velocity_indices,
        diffusion_type=diffusion_type, fusion_hidden_dim=fusion_hidden,
        zero_init=True, pos_dims=2,
    )

    # Full model
    model = GPARC_Cylinder_V2(
        derivative_solver_physics=derivative_solver,
        integrator_type=integrator,
        num_static_feats=num_static, num_dynamic_feats=num_dynamic,
        skip_dynamic_indices=skip_indices,
        global_param_dim=global_param_dim, global_embed_dim=global_embed_dim,
        clamp_output=not config.get('no_clamp_output', False),
        clamp_max=config.get('clamp_max', 10.0),
    )

    return model


def load_checkpoint(checkpoint_path, config, device):
    """Load model from checkpoint."""
    model = create_model_from_config(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('metrics', {}).get('val_loss', '?')
    print(f"✓ Loaded checkpoint: epoch {epoch}, val_loss {val_loss}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


# =========================================================================
# ROLLOUT
# =========================================================================

def run_rollout(model, simulation, num_steps, device):
    """
    Run autoregressive rollout and collect predictions + targets.

    Returns:
        pred_states: list of [N, D] numpy arrays (model predictions)
        targ_states: list of [N, D] numpy arrays (ground truth)
        positions:   [N, 3] numpy array (static positions)
        reynolds:    float (Reynolds number for this simulation)
    """
    # Move to device and stamp mesh_id
    mesh_id = simulation[0].num_nodes
    for data in simulation:
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        if hasattr(data, 'y') and data.y is not None:
            data.y = data.y.to(device)
        if hasattr(data, 'pos') and data.pos is not None:
            data.pos = data.pos.to(device)
        data.mesh_id = torch.tensor([mesh_id], device=device)

    # Initialize MLS
    deriv = model.derivative_solver
    if hasattr(deriv, 'initialize_weights'):
        sample = simulation[0]
        if not hasattr(sample, 'pos') or sample.pos is None:
            sample.pos = sample.x[:, :model.num_static_feats]
        deriv.initialize_weights(sample)

    # Extract positions and Reynolds
    positions = simulation[0].x[:, :model.num_static_feats].cpu().numpy()
    reynolds = model._extract_global_attrs(simulation[0]).cpu().item()

    # Run rollout
    pred_states = model.rollout(simulation, num_steps, device=device)

    # Extract ground truth targets
    targ_states = []
    for t in range(min(num_steps + 1, len(simulation))):
        targ = model.process_targets(simulation[t].y).cpu().numpy()
        targ_states.append(targ)

    return pred_states, targ_states, positions, reynolds


# =========================================================================
# METRICS
# =========================================================================

def compute_metrics(pred_states, targ_states, var_names):
    """Compute per-variable and overall metrics across all timesteps."""
    max_steps = min(len(pred_states), len(targ_states))

    # Per-timestep MSE
    timestep_mse = []
    per_var_mse = {vn: [] for vn in var_names}

    for t in range(max_steps):
        p, tg = pred_states[t], targ_states[t]
        mse = np.mean((p - tg) ** 2)
        timestep_mse.append(mse)

        for vi, vn in enumerate(var_names):
            if vi < p.shape[1]:
                per_var_mse[vn].append(np.mean((p[:, vi] - tg[:, vi]) ** 2))

    # Aggregate
    all_p = np.concatenate(pred_states[:max_steps], axis=0)
    all_t = np.concatenate(targ_states[:max_steps], axis=0)

    overall_mse = np.mean((all_p - all_t) ** 2)
    overall_rmse = np.sqrt(overall_mse)
    targ_var = np.var(all_t)
    overall_rrmse = overall_rmse / max(np.sqrt(targ_var), 1e-12)

    per_var_r2 = {}
    for vi, vn in enumerate(var_names):
        if vi < all_p.shape[1]:
            per_var_r2[vn] = r2_score(all_t[:, vi], all_p[:, vi])

    return {
        'overall_mse': overall_mse,
        'overall_rmse': overall_rmse,
        'overall_rrmse': overall_rrmse,
        'per_var_r2': per_var_r2,
        'timestep_mse': timestep_mse,
        'per_var_mse': per_var_mse,
    }


# =========================================================================
# VISUALIZATION — TRICONTOURF FOR UNSTRUCTURED MESH
# =========================================================================

def subsample_for_viz(positions, max_nodes=10000, seed=42):
    """
    Subsample node indices for visualization.
    
    For 165k node meshes, tricontourf is extremely slow.
    Subsampling to ~10k nodes gives visually identical results
    since the flow field is smooth.
    
    Returns:
        viz_idx: np.ndarray of selected node indices
    """
    N = positions.shape[0]
    if N <= max_nodes:
        return np.arange(N)
    
    rng = np.random.RandomState(seed)
    viz_idx = rng.choice(N, max_nodes, replace=False)
    viz_idx.sort()  # Keep spatial ordering for cleaner triangulation
    return viz_idx


def build_triangulation(positions, viz_idx=None):
    """
    Build Delaunay triangulation from 2D positions for tricontourf rendering.
    Uses x, y columns (first 2 of potentially 3D positions).
    
    Args:
        positions: [N, 2+] full position array
        viz_idx: optional subset indices for rendering
    """
    if viz_idx is not None:
        positions = positions[viz_idx]
    x, y = positions[:, 0], positions[:, 1]
    return tri.Triangulation(x, y)


def _get_vals(states, frame, var_idx, viz_idx):
    """Helper to extract values, optionally subsampled."""
    if viz_idx is None:
        return states[frame][:, var_idx]
    return states[frame][viz_idx, var_idx]


def _tri_face_avg(triang, node_vals):
    """Average node values to triangle faces for tripcolor shading='flat'."""
    return node_vals[triang.triangles].mean(axis=1)


def create_variable_comparison_gif(
    frames, pred_states, targ_states, triang, var_idx, var_name,
    vmin, vmax, case_name, output_dir, reynolds=None, fps=4,
    viz_idx=None,
):
    """
    Side-by-side Target | Prediction | |Error| GIF using tripcolor.
    Draws triangulation once, updates facecolors each frame → very fast.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    cmap = plt.colormaps.get_cmap(VAR_CMAPS.get(var_name, 'coolwarm'))
    err_cmap = plt.colormaps.get_cmap('hot')

    # Compute global error max
    err_max = 0
    for f in frames:
        err = np.abs(_get_vals(targ_states, f, var_idx, viz_idx) -
                     _get_vals(pred_states, f, var_idx, viz_idx))
        err_max = max(err_max, err.max())
    err_max = max(err_max, 1e-12)

    norm = Normalize(vmin=vmin, vmax=vmax)
    err_norm = Normalize(vmin=0, vmax=err_max)

    title_base = f'{var_name}: {case_name}'
    if reynolds is not None:
        title_base += f'  (Re = {reynolds:.0f})'

    # Initial frame — create tripcolor artists
    f0 = frames[0]
    targ0 = _get_vals(targ_states, f0, var_idx, viz_idx)
    pred0 = _get_vals(pred_states, f0, var_idx, viz_idx)
    err0 = np.abs(targ0 - pred0)

    for ax in axes:
        ax.set_aspect('equal'); ax.axis('off')

    tc0 = axes[0].tripcolor(triang, targ0, cmap=cmap, norm=norm, shading='gouraud')
    tc1 = axes[1].tripcolor(triang, pred0, cmap=cmap, norm=norm, shading='gouraud')
    tc2 = axes[2].tripcolor(triang, err0, cmap=err_cmap, norm=err_norm, shading='gouraud')

    fig.colorbar(tc0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(tc1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(tc2, ax=axes[2], fraction=0.046, pad=0.04)

    def animate(frame_idx):
        frame = frames[frame_idx]
        targ_vals = _get_vals(targ_states, frame, var_idx, viz_idx)
        pred_vals = _get_vals(pred_states, frame, var_idx, viz_idx)
        err_vals = np.abs(targ_vals - pred_vals)

        tc0.set_array(targ_vals)
        tc1.set_array(pred_vals)
        tc2.set_array(err_vals)

        axes[0].set_title(f'Target (t={frame})', fontsize=12)
        axes[1].set_title(f'Prediction (t={frame})', fontsize=12)
        axes[2].set_title(f'|Error| (t={frame})', fontsize=12)
        fig.suptitle(title_base, fontsize=14, fontweight='bold')
        return [tc0, tc1, tc2]

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps, blit=False)
    out_path = Path(output_dir) / f'{var_name}_{case_name}.gif'
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def create_all_variables_gif(
    frames, pred_states, targ_states, triang, var_names, vranges,
    case_name, output_dir, reynolds=None, fps=4, viz_idx=None,
):
    """
    Combined GIF: 2 rows (target / prediction) × n_vars columns.
    Uses tripcolor with gouraud shading, updates arrays in-place.
    """
    n_vars = len(var_names)
    fig, axes = plt.subplots(2, n_vars, figsize=(7 * n_vars, 10))
    if n_vars == 1:
        axes = axes.reshape(2, 1)

    title_base = f'All Variables: {case_name}'
    if reynolds is not None:
        title_base += f'  (Re = {reynolds:.0f})'

    # Create initial tripcolor artists
    f0 = frames[0]
    artists = []  # list of (tc_targ, tc_pred) per variable
    for vi, vn in enumerate(var_names):
        vmin, vmax = vranges[vi]
        n = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.colormaps.get_cmap(VAR_CMAPS.get(vn, 'coolwarm'))

        targ0 = _get_vals(targ_states, f0, vi, viz_idx)
        pred0 = _get_vals(pred_states, f0, vi, viz_idx)

        axes[0, vi].set_aspect('equal'); axes[0, vi].axis('off')
        axes[1, vi].set_aspect('equal'); axes[1, vi].axis('off')

        tc_t = axes[0, vi].tripcolor(triang, targ0, cmap=cmap, norm=n, shading='gouraud')
        tc_p = axes[1, vi].tripcolor(triang, pred0, cmap=cmap, norm=n, shading='gouraud')
        fig.colorbar(tc_t, ax=axes[0, vi], fraction=0.046, pad=0.04)
        fig.colorbar(tc_p, ax=axes[1, vi], fraction=0.046, pad=0.04)
        artists.append((tc_t, tc_p))

    def animate(frame_idx):
        frame = frames[frame_idx]
        for vi, vn in enumerate(var_names):
            targ_vals = _get_vals(targ_states, frame, vi, viz_idx)
            pred_vals = _get_vals(pred_states, frame, vi, viz_idx)
            artists[vi][0].set_array(targ_vals)
            artists[vi][1].set_array(pred_vals)
            axes[0, vi].set_title(f'{vn} Target (t={frame})', fontsize=10)
            axes[1, vi].set_title(f'{vn} Pred (t={frame})', fontsize=10)
        fig.suptitle(f'{title_base} (t={frame})', fontsize=13, fontweight='bold')
        return [a for pair in artists for a in pair]

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps, blit=False)
    out_path = Path(output_dir) / f'all_vars_{case_name}.gif'
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def create_error_evolution_gif(
    frames, pred_states, targ_states, triang, var_names,
    case_name, output_dir, reynolds=None, fps=4, viz_idx=None,
):
    """
    GIF showing per-variable |error| evolving over time.
    Uses tripcolor with gouraud shading, updates arrays in-place.
    """
    n_vars = len(var_names)
    fig, axes = plt.subplots(1, n_vars, figsize=(7 * n_vars, 6))
    if n_vars == 1:
        axes = [axes]
    err_cmap = plt.colormaps.get_cmap('hot')

    # Global error max per variable
    max_errors = [0.0] * n_vars
    for f in frames:
        for vi in range(n_vars):
            err = np.abs(_get_vals(targ_states, f, vi, viz_idx) -
                         _get_vals(pred_states, f, vi, viz_idx))
            max_errors[vi] = max(max_errors[vi], err.max())

    title_base = f'Error Evolution: {case_name}'
    if reynolds is not None:
        title_base += f'  (Re = {reynolds:.0f})'

    # Create initial artists
    f0 = frames[0]
    tc_list = []
    for vi, vn in enumerate(var_names):
        err0 = np.abs(_get_vals(targ_states, f0, vi, viz_idx) -
                      _get_vals(pred_states, f0, vi, viz_idx))
        en = Normalize(vmin=0, vmax=max(max_errors[vi], 1e-12))
        axes[vi].set_aspect('equal'); axes[vi].axis('off')
        tc = axes[vi].tripcolor(triang, err0, cmap=err_cmap, norm=en, shading='gouraud')
        fig.colorbar(tc, ax=axes[vi], fraction=0.046, pad=0.04)
        tc_list.append(tc)

    def animate(frame_idx):
        frame = frames[frame_idx]
        for vi, vn in enumerate(var_names):
            err_vals = np.abs(_get_vals(targ_states, frame, vi, viz_idx) -
                              _get_vals(pred_states, frame, vi, viz_idx))
            tc_list[vi].set_array(err_vals)
            axes[vi].set_title(f'{vn} |Error| (t={frame})', fontsize=10)
        fig.suptitle(f'{title_base} (t={frame})', fontsize=13, fontweight='bold')
        return tc_list

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps, blit=False)
    out_path = Path(output_dir) / f'error_evolution_{case_name}.gif'
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


# =========================================================================
# STATIC PLOTS
# =========================================================================

def plot_rollout_error_growth(all_metrics, var_names, output_path):
    """Plot MSE growth across rollout timesteps, averaged over simulations."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Rollout Error Growth', fontsize=14, fontweight='bold')

    # Overall MSE
    ax = axes[0]
    for sim_name, metrics in all_metrics.items():
        steps = range(len(metrics['timestep_mse']))
        ax.plot(steps, metrics['timestep_mse'], alpha=0.5, label=sim_name)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('MSE')
    ax.set_title('Overall MSE vs Step')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Per-variable MSE (averaged over sims)
    ax = axes[1]
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(var_names)))
    for vi, vn in enumerate(var_names):
        all_curves = []
        for metrics in all_metrics.values():
            if vn in metrics['per_var_mse']:
                all_curves.append(metrics['per_var_mse'][vn])
        if all_curves:
            max_len = max(len(c) for c in all_curves)
            # Pad shorter curves
            padded = [c + [c[-1]] * (max_len - len(c)) for c in all_curves]
            mean_curve = np.mean(padded, axis=0)
            ax.plot(range(max_len), mean_curve, 'o-', label=vn,
                    color=colors[vi], ms=3, linewidth=2)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('MSE')
    ax.set_title('Per-Variable MSE (averaged)')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output_path}")


def plot_prediction_scatter(pred_states, targ_states, var_names, output_path,
                             n_sample=10000):
    """Prediction vs target scatter for each variable."""
    all_p = np.concatenate(pred_states, axis=0)
    all_t = np.concatenate(targ_states, axis=0)

    n_vars = min(all_p.shape[1], len(var_names))
    fig, axes = plt.subplots(1, n_vars, figsize=(6 * n_vars, 5))
    if n_vars == 1:
        axes = [axes]
    fig.suptitle('Prediction vs Target', fontsize=14, fontweight='bold')

    n_sample = min(n_sample, all_p.shape[0])
    idx = np.random.choice(all_p.shape[0], n_sample, replace=False)

    for vi in range(n_vars):
        ax = axes[vi]
        pv, tv = all_p[idx, vi], all_t[idx, vi]
        ax.scatter(tv, pv, s=1, alpha=0.2)
        lo, hi = min(tv.min(), pv.min()), max(tv.max(), pv.max())
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.7, linewidth=2)
        r2 = r2_score(tv, pv) if len(tv) > 1 else 0
        ax.set_title(f'{var_names[vi]}  (R² = {r2:.4f})')
        ax.set_xlabel('Target')
        ax.set_ylabel('Prediction')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate G-PARCv2 Cylinder Flow")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory with test .pt files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: checkpoint_dir/eval)")
    parser.add_argument("--num_rollout_steps", type=int, default=None,
                        help="Number of rollout steps (default: all timesteps - 1)")
    parser.add_argument("--max_sims", type=int, default=None,
                        help="Max simulations to evaluate")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame_skip", type=int, default=2,
                        help="Render every Nth frame in GIFs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_gifs", action="store_true", default=False,
                        help="Skip GIF generation (metrics only)")
    parser.add_argument("--viz_max_nodes", type=int, default=10000,
                        help="Max nodes for GIF rendering (subsample if larger, default 10000)")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_dir = checkpoint_path.parent

    if args.output_dir is None:
        output_dir = checkpoint_dir / "eval"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("G-PARCv2 CYLINDER FLOW EVALUATION")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test dir:   {args.test_dir}")
    print(f"Output:     {output_dir}")
    print(f"Device:     {device}")

    # Load config and model
    config = load_config(checkpoint_dir)
    if config is None:
        print("ERROR: config.json not found in checkpoint directory")
        sys.exit(1)

    var_names = CYLINDER_VAR_NAMES
    num_dynamic = config.get('num_dynamic_feats', 4)
    if num_dynamic < len(var_names):
        var_names = var_names[:num_dynamic]

    print(f"\nVariables ({len(var_names)}): {var_names}")

    model = load_checkpoint(checkpoint_path, config, device)

    # Load test data
    test_dir = Path(args.test_dir)
    test_ids = get_simulation_ids(test_dir, pattern="*.pt")
    if args.max_sims:
        test_ids = test_ids[:args.max_sims]

    print(f"\nFound {len(test_ids)} test simulations")

    # Evaluate each simulation
    all_metrics = {}
    all_pred_states = {}
    all_targ_states = {}

    for sim_idx, sim_id in enumerate(test_ids):
        print(f"\n{'='*70}")
        print(f"Simulation {sim_idx+1}/{len(test_ids)}: {sim_id}")
        print(f"{'='*70}")

        # Load full simulation
        sim_file = test_dir / f"{sim_id}.pt"
        sim_data = torch.load(sim_file, weights_only=False)

        if not isinstance(sim_data, list):
            print(f"  Skipping: unexpected format")
            continue

        T = len(sim_data)
        num_steps = args.num_rollout_steps if args.num_rollout_steps else T - 1
        num_steps = min(num_steps, T - 1)

        print(f"  Nodes: {sim_data[0].num_nodes:,}")
        print(f"  Timesteps: {T}")
        print(f"  Rollout steps: {num_steps}")

        # Run rollout
        with torch.no_grad():
            pred_states, targ_states, positions, reynolds = run_rollout(
                model, sim_data, num_steps, device
            )

        # Extract Reynolds from filename if needed
        re_str = sim_id.replace('_normalized', '').replace('Re_', '')
        try:
            re_val = float(re_str)
        except ValueError:
            re_val = reynolds

        print(f"  Reynolds: {re_val:.0f}")

        case_name = f'Re_{re_val:.0f}'

        # Compute metrics
        metrics = compute_metrics(pred_states, targ_states, var_names)
        all_metrics[case_name] = metrics

        print(f"  Overall MSE:   {metrics['overall_mse']:.6e}")
        print(f"  Overall RMSE:  {metrics['overall_rmse']:.6e}")
        print(f"  Overall RRMSE: {metrics['overall_rrmse']:.4f}")
        for vn, r2 in metrics['per_var_r2'].items():
            print(f"  R² {vn}: {r2:.6f}")

        # Store for scatter plot
        all_pred_states[case_name] = pred_states
        all_targ_states[case_name] = targ_states

        # GIF generation
        if not args.skip_gifs:
            print(f"\n  Generating GIFs...")
            
            # Subsample for rendering speed
            viz_idx = subsample_for_viz(positions, max_nodes=args.viz_max_nodes)
            n_viz = len(viz_idx)
            if n_viz < positions.shape[0]:
                print(f"  Subsampled {positions.shape[0]:,} → {n_viz:,} nodes for rendering")
            
            triang = build_triangulation(positions, viz_idx=viz_idx)
            frames = list(range(0, len(pred_states), args.frame_skip))
            if len(frames) == 0:
                frames = [0]

            # Compute value ranges across all timesteps (on subsampled nodes)
            vranges = []
            for vi in range(len(var_names)):
                all_vals = np.concatenate([
                    np.concatenate([targ_states[f][viz_idx, vi] for f in frames]),
                    np.concatenate([pred_states[f][viz_idx, vi] for f in frames]),
                ])
                vranges.append((np.percentile(all_vals, 1),
                                np.percentile(all_vals, 99)))

            # Per-variable comparison GIFs
            for vi, vn in enumerate(var_names):
                path = create_variable_comparison_gif(
                    frames, pred_states, targ_states, triang, vi, vn,
                    vranges[vi][0], vranges[vi][1],
                    case_name, output_dir, reynolds=re_val, fps=args.fps,
                    viz_idx=viz_idx,
                )
                print(f"    ✓ {path.name}")

            # All-variables combined GIF
            path = create_all_variables_gif(
                frames, pred_states, targ_states, triang, var_names, vranges,
                case_name, output_dir, reynolds=re_val, fps=args.fps,
                viz_idx=viz_idx,
            )
            print(f"    ✓ {path.name}")

            # Error evolution GIF
            path = create_error_evolution_gif(
                frames, pred_states, targ_states, triang, var_names,
                case_name, output_dir, reynolds=re_val, fps=args.fps,
                viz_idx=viz_idx,
            )
            print(f"    ✓ {path.name}")

        # Cleanup
        del sim_data, pred_states, targ_states
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # AGGREGATE RESULTS
    # =========================================================================
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    # Summary table
    print(f"\n{'Simulation':<20} {'MSE':>12} {'RMSE':>12} {'RRMSE':>10}")
    print("-" * 56)
    for case_name, metrics in all_metrics.items():
        print(f"{case_name:<20} {metrics['overall_mse']:>12.6e} "
              f"{metrics['overall_rmse']:>12.6e} {metrics['overall_rrmse']:>10.4f}")

    # Per-variable R² table
    print(f"\n{'Simulation':<20}", end="")
    for vn in var_names:
        print(f" {vn:>14}", end="")
    print()
    print("-" * (20 + 15 * len(var_names)))
    for case_name, metrics in all_metrics.items():
        print(f"{case_name:<20}", end="")
        for vn in var_names:
            r2 = metrics['per_var_r2'].get(vn, float('nan'))
            print(f" {r2:>14.6f}", end="")
        print()

    # Rollout error growth plot
    plot_rollout_error_growth(all_metrics, var_names,
                               output_dir / "rollout_error_growth.png")

    # Combined scatter plot (first sim only to save memory)
    if all_pred_states:
        first_key = list(all_pred_states.keys())[0]
        plot_prediction_scatter(
            all_pred_states[first_key], all_targ_states[first_key],
            var_names, output_dir / f"scatter_{first_key}.png",
        )

    # Save metrics JSON
    metrics_json = {}
    for case_name, metrics in all_metrics.items():
        metrics_json[case_name] = {
            'overall_mse': float(metrics['overall_mse']),
            'overall_rmse': float(metrics['overall_rmse']),
            'overall_rrmse': float(metrics['overall_rrmse']),
            'per_var_r2': {k: float(v) for k, v in metrics['per_var_r2'].items()},
        }
    with open(output_dir / "eval_metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\n✓ Metrics saved to {output_dir / 'eval_metrics.json'}")

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()