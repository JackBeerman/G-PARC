#!/usr/bin/env python3
"""
Shock Tube — Multi-Model Comparison GIFs
==========================================
Generates animated GIFs showing ground truth vs model predictions.

Layout per frame (rows=variables, cols=models):
    Columns: [Ground Truth] [G-PARC w/ MLS] [G-PARC Baseline] [MeshGraphKAN] ...
    Row 0:   density        density          density            density         | colorbar
    Row 1:   x_momentum     x_momentum       x_momentum         x_momentum     | colorbar
    Row 2:   total_energy   total_energy     total_energy       total_energy   | colorbar

Usage:
    python compare_shocktube_gif.py \
        --test_dir /path/to/test \
        --models gparcv2:/path/v2.pth gparcv1:/path/v1.pth mgkan:/path/mgkan.pth \
        --output_dir ./shocktube_gifs \
        --sim_indices 0 1 2 \
        --rollout_steps 40 \
        --fps 6
"""

import argparse, sys, os, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from eval import (
    ST_REGISTRY, ST_VAR_NAMES, ST_NUM_STATIC, ST_NUM_DYNAMIC,
    ST_SKIP_INDICES, ST_KEEP_INDICES, ST_RAW_DYNAMIC,
    st_load_data, st_extract_dynamic, st_apply_skip,
    parse_model_specs, _clear_mls_caches,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================

VAR_NAMES = ['Density (ρ)', 'x-Momentum (ρu)', 'Total Energy (E)']
CMAPS = ['coolwarm', 'coolwarm', 'inferno']
MODEL_COLORS = {
    'Ground Truth':    '#333333',
    'G-PARC with MLS': '#1f77b4',
    'G-PARC Baseline': '#d62728',
    'MeshGraphKAN':    '#ff7f0e',
    'MeshGraphNet':    '#2ca02c',
    'GraphSAGE':       '#9467bd',
}


def detect_grid_size(sim):
    pos = sim[0].x[:, :ST_NUM_STATIC].cpu().numpy()
    n_nodes = pos.shape[0]
    gs = int(np.sqrt(n_nodes))
    if gs * gs != n_nodes:
        for candidate in [64, 32, 128, 48, 96]:
            if candidate * candidate == n_nodes:
                gs = candidate; break
    return gs


def nodes_to_grid(node_values, grid_size):
    D = node_values.shape[1] if node_values.ndim > 1 else 1
    if D == 1:
        return [node_values.reshape(grid_size, grid_size)]
    return [node_values[:, d].reshape(grid_size, grid_size) for d in range(D)]


def extract_global_params_str(sim):
    d = sim[0]; parts = []
    for attr, label in [('global_pressure', 'P'), ('global_density', 'ρ'), ('global_delta_t', 'Δt')]:
        if hasattr(d, attr):
            v = getattr(d, attr)
            parts.append(f"{label}={v.item() if torch.is_tensor(v) else float(v):.3f}")
    if not parts and hasattr(d, 'global_params') and d.global_params.numel() >= 3:
        gp = d.global_params
        parts = [f"P={gp[0]:.3f}", f"ρ={gp[1]:.3f}", f"Δt={gp[2]:.3f}"]
    return ' | '.join(parts) if parts else ''


def extract_gt_list(sim, n_steps):
    gt = []
    for t in range(n_steps):
        if hasattr(sim[t], 'y') and sim[t].y is not None:
            gt.append(st_apply_skip(sim[t].y).cpu().numpy())
        else:
            gt.append(st_extract_dynamic(sim[t + 1].x).cpu().numpy())
    return gt


def compute_gt_ranges(gt_list, n_vars):
    vmin = [np.inf] * n_vars; vmax = [-np.inf] * n_vars
    for gt in gt_list:
        for vi in range(n_vars):
            vmin[vi] = min(vmin[vi], gt[:, vi].min())
            vmax[vi] = max(vmax[vi], gt[:, vi].max())
    for vi in range(n_vars):
        rng = vmax[vi] - vmin[vi]
        vmin[vi] -= 0.02 * rng; vmax[vi] += 0.02 * rng
    return vmin, vmax


def select_frames(n_steps, frame_skip):
    frames = list(range(0, n_steps, max(1, frame_skip)))
    if frames[-1] != n_steps - 1: frames.append(n_steps - 1)
    return frames


# ==============================================================================
# COMPARISON GIF — rows=variables, cols=[GT, models], colorbar per row on right
# ==============================================================================

def create_comparison_gif(sim, model_results, sim_name, grid_size,
                           output_dir, fps=6, frame_skip=1):
    model_names = list(model_results.keys())
    n_vars = len(VAR_NAMES)
    n_cols = 1 + len(model_names)
    n_rows = n_vars

    n_steps = min(len(list(model_results.values())[0]), len(sim) - 1)
    gt_list = extract_gt_list(sim, n_steps)
    vmin, vmax = compute_gt_ranges(gt_list, n_vars)
    all_frames = select_frames(n_steps, frame_skip)
    gp_str = extract_global_params_str(sim)

    col_labels = ['Ground Truth'] + model_names
    col_colors = [MODEL_COLORS.get(c, '#555') for c in col_labels]

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.8 * n_cols + 1.2, 3.0 * n_rows + 0.6),
                              gridspec_kw={'hspace': 0.15, 'wspace': 0.06,
                                           'right': 0.88, 'left': 0.08,
                                           'top': 0.90, 'bottom': 0.03})
    fig.patch.set_facecolor('white')

    ims = []
    for ri in range(n_rows):
        row_ims = []
        for ci in range(n_cols):
            ax = axes[ri, ci]
            im = ax.imshow(np.zeros((grid_size, grid_size)), origin='lower',
                           cmap=CMAPS[ri], vmin=vmin[ri], vmax=vmax[ri],
                           aspect='equal', interpolation='bilinear')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor('#ddd'); sp.set_linewidth(0.5)
            if ri == 0:
                ax.set_title(col_labels[ci], fontsize=10, fontweight='bold',
                             color=col_colors[ci], pad=8)
            if ci == 0:
                ax.set_ylabel(VAR_NAMES[ri], fontsize=10, fontweight='bold',
                              color='#333', rotation=90, labelpad=10)
            row_ims.append(im)
        ims.append(row_ims)

    # Colorbar per row on right
    for ri in range(n_rows):
        pos = axes[ri, -1].get_position()
        cax = fig.add_axes([0.90, pos.y0, 0.012, pos.height])
        sm = plt.cm.ScalarMappable(norm=Normalize(vmin[ri], vmax[ri]), cmap=CMAPS[ri])
        cb = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=7, colors='#444')
        cb.outline.set_edgecolor('#bbb')

    suptitle = fig.suptitle('', fontsize=13, fontweight='bold', color='#222', y=0.96)

    def animate(fi):
        t = all_frames[fi]
        gt_grids = nodes_to_grid(gt_list[t], grid_size)
        for ri in range(n_rows):
            ims[ri][0].set_data(gt_grids[ri])
            for mi, mn in enumerate(model_names):
                pg = nodes_to_grid(model_results[mn][t], grid_size)
                ims[ri][mi + 1].set_data(pg[ri])
        title = f'Rollout Step {t}/{n_steps - 1}'
        if gp_str: title += f'  ({gp_str})'
        suptitle.set_text(title)
        return [im for row in ims for im in row]

    print(f"  Creating comparison GIF: {len(all_frames)} frames @ {fps} fps...")
    anim = FuncAnimation(fig, animate, frames=len(all_frames), interval=1000//fps, blit=False)
    gif_path = output_dir / f'comparison_{sim_name}.gif'
    anim.save(str(gif_path), writer=PillowWriter(fps=fps)); plt.close(fig)
    print(f"  ✓ Saved: {gif_path}")
    return gif_path


# ==============================================================================
# ERROR GIF — rows=variables, cols=models (excluding outliers), colorbar per row
# ==============================================================================

def create_error_gif(sim, model_results, sim_name, grid_size,
                      output_dir, fps=6, frame_skip=1, exclude=None):
    if exclude is None: exclude = []
    model_names = [mn for mn in model_results if mn not in exclude]
    if not model_names:
        print("  ⚠ No models left for error GIF"); return None

    n_vars = len(VAR_NAMES)
    n_models = len(model_names)
    n_steps = min(len(list(model_results.values())[0]), len(sim) - 1)
    gt_list = extract_gt_list(sim, n_steps)

    # Error scale from included models only (95th percentile)
    emax = [0.0] * n_vars
    for mn in model_names:
        for t in range(n_steps):
            err = np.abs(model_results[mn][t] - gt_list[t])
            for vi in range(n_vars):
                emax[vi] = max(emax[vi], np.percentile(err[:, vi], 95))

    all_frames = select_frames(n_steps, frame_skip)

    fig, axes = plt.subplots(n_vars, n_models,
                              figsize=(2.8 * n_models + 1.2, 3.0 * n_vars + 0.6),
                              gridspec_kw={'hspace': 0.15, 'wspace': 0.06,
                                           'right': 0.88, 'left': 0.08,
                                           'top': 0.90, 'bottom': 0.03})
    if n_vars == 1: axes = axes[np.newaxis, :]
    if n_models == 1: axes = axes[:, np.newaxis]
    fig.patch.set_facecolor('white')

    ims = []
    for ri in range(n_vars):
        row_ims = []
        for ci, mn in enumerate(model_names):
            ax = axes[ri, ci]
            im = ax.imshow(np.zeros((grid_size, grid_size)), origin='lower',
                           cmap='hot', vmin=0, vmax=emax[ri],
                           aspect='equal', interpolation='bilinear')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor('#ddd'); sp.set_linewidth(0.5)
            if ri == 0:
                ax.set_title(mn, fontsize=10, fontweight='bold',
                             color=MODEL_COLORS.get(mn, '#555'), pad=8)
            if ci == 0:
                ax.set_ylabel(f'|Err| {VAR_NAMES[ri]}', fontsize=9, fontweight='bold',
                              color='#333', rotation=90, labelpad=10)
            row_ims.append(im)
        ims.append(row_ims)

    for ri in range(n_vars):
        pos = axes[ri, -1].get_position()
        cax = fig.add_axes([0.90, pos.y0, 0.012, pos.height])
        sm = plt.cm.ScalarMappable(norm=Normalize(0, emax[ri]), cmap='hot')
        cb = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=7, colors='#444')
        cb.outline.set_edgecolor('#bbb')

    suptitle = fig.suptitle('', fontsize=13, fontweight='bold', color='#222', y=0.96)

    def animate(fi):
        t = all_frames[fi]
        for ri in range(n_vars):
            for ci, mn in enumerate(model_names):
                err = np.abs(model_results[mn][t] - gt_list[t])
                eg = nodes_to_grid(err, grid_size)
                ims[ri][ci].set_data(eg[ri])
        suptitle.set_text(f'Absolute Error — Step {t}/{n_steps - 1}')
        return [im for row in ims for im in row]

    print(f"  Creating error GIF: {len(all_frames)} frames @ {fps} fps...")
    anim = FuncAnimation(fig, animate, frames=len(all_frames), interval=1000//fps, blit=False)
    gif_path = output_dir / f'error_{sim_name}.gif'
    anim.save(str(gif_path), writer=PillowWriter(fps=fps)); plt.close(fig)
    print(f"  ✓ Saved: {gif_path}")
    return gif_path


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Shock Tube Multi-Model Comparison GIFs")
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--models", nargs='+', required=True)
    parser.add_argument("--output_dir", default="./shocktube_gifs")
    parser.add_argument("--sim_indices", type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument("--rollout_steps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_sims", type=int, default=None)
    parser.add_argument("--error_gif", action="store_true", default=True)
    parser.add_argument("--exclude_error", nargs='*', default=['MeshGraphNet', 'GraphSAGE'],
                        help="Model display names to exclude from error GIF")

    args = parser.parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 60}\nSHOCK TUBE — MULTI-MODEL COMPARISON GIFS\n{'=' * 60}")

    specs = parse_model_specs(args.models, ST_REGISTRY)
    if not specs: print("No valid models found!"); return

    sims = st_load_data(args.test_dir, args.max_sims)
    if not sims: print("No simulations found!"); return

    grid_size = detect_grid_size(sims[0][1])
    print(f"Grid size: {grid_size}×{grid_size}")

    sample = sims[0][1][0]
    if not hasattr(sample, 'pos') or sample.pos is None:
        sample.pos = sample.x[:, :ST_NUM_STATIC]
    sample = sample.to(device)

    models = {}
    for mt, mp in specs.items():
        reg = ST_REGISTRY[mt]
        try:
            print(f"\nLoading {reg['name']}...")
            m = reg['load'](mp, sample, device)
            print(f"  ✓ {sum(p.numel() for p in m.parameters()):,} params")
            models[mt] = {'model': m, 'name': reg['name'], 'rollout': reg['rollout'], 'device': device}
        except Exception as e:
            print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc()

    if not models: print("No models loaded!"); return

    for sim_idx in args.sim_indices:
        if sim_idx >= len(sims):
            print(f"\n⚠ Index {sim_idx} out of range ({len(sims)} sims)"); continue

        sim_name, sim = sims[sim_idx]
        steps = min(args.rollout_steps, len(sim) - 1)
        print(f"\n{'─' * 60}\nSim {sim_idx}: {sim_name} ({steps} steps)")
        gp = extract_global_params_str(sim)
        if gp: print(f"  {gp}")

        model_results = {}
        for mt, minfo in models.items():
            mn = minfo['name']
            print(f"  Rolling out {mn}...")
            try:
                preds = minfo['rollout'](minfo['model'], sim, steps, device)
                preds = [p if isinstance(p, np.ndarray) else p.cpu().numpy() for p in preds]
                model_results[mn] = preds[:steps]
                print(f"    ✓ {len(model_results[mn])} steps")
            except Exception as e:
                print(f"    ✗ {e}"); import traceback; traceback.print_exc()

        if not model_results: continue

        create_comparison_gif(sim, model_results, sim_name, grid_size,
                               output_dir, fps=args.fps, frame_skip=args.frame_skip)

        if args.error_gif:
            create_error_gif(sim, model_results, sim_name, grid_size,
                              output_dir, fps=args.fps, frame_skip=args.frame_skip,
                              exclude=args.exclude_error)

    print(f"\n{'=' * 60}\n✓ All GIFs → {output_dir}\n{'=' * 60}")


if __name__ == "__main__":
    main()