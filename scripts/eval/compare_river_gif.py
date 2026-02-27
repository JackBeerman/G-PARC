#!/usr/bin/env python3
"""
River — Multi-Model Comparison GIFs
=====================================
Generates animated GIFs showing ground truth vs model predictions
on static Eulerian river meshes using HEC-RAS polygon meshes.

Layout per frame (rows=variables, cols=models):
    Columns: [Ground Truth] [G-PARC w/ MLS] [G-PARC Baseline] [MeshGraphKAN] ...
    Row 0:   Depth          Depth            Depth              Depth          | colorbar
    Row 1:   Volume         Volume           Volume             Volume         | colorbar
    Row 2:   Vel_X          Vel_X            Vel_X              Vel_X          | colorbar
    Row 3:   Vel_Y          Vel_Y            Vel_Y              Vel_Y          | colorbar

Usage:
    python compare_river_gif.py \\
        --test_dir /path/to/test \\
        --hec_ras_dir "/standard/sds_baek_energetic/HEC_RAS (River)" \\
        --models gparcv2:/path/v2.pth mgkan:/path/mgkan.pth \\
        --output_dir ./river_gifs \\
        --sim_indices 0 1 2 \\
        --rollout_steps 50 \\
        --fps 6
"""

import argparse, sys, os, json
from pathlib import Path
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from eval import (
    RV_REGISTRY, RV_VAR_NAMES, RV_NUM_STATIC, RV_NUM_DYNAMIC,
    rv_load_data, rv_extract_dynamic,
    parse_model_specs, _clear_mls_caches,
)
from visualizations.mesh_io import load_mesh_for_sim, get_node_positions, HAS_H5PY

# ==============================================================================
# CONSTANTS
# ==============================================================================

VAR_NAMES = ['Depth', 'Volume', 'Vel X', 'Vel Y']
# Publication-friendly colormaps
CMAPS = ['viridis', 'viridis', 'RdBu_r', 'RdBu_r']
MODEL_COLORS = {
    'Ground Truth':    '#333333',
    'G-PARC with MLS': '#1f77b4',
    'G-PARC Baseline': '#d62728',
    'MeshGraphKAN':    '#ff7f0e',
    'MeshGraphNet':    '#2ca02c',
    'GraphSAGE':       '#9467bd',
}

# Physical time per timestep (minutes)
DT_MINUTES = 20


def format_physical_time(step, dt_minutes=DT_MINUTES):
    total_min = step * dt_minutes
    if total_min >= 60:
        hours = total_min / 60
        if hours == int(hours):
            return f'{int(hours)} hr'
        return f'{hours:.1f} hr'
    return f'{total_min} min'


# ==============================================================================
# DENORMALIZATION (optional)
# ==============================================================================

def load_extrema(path):
    if path and Path(path).exists():
        return torch.load(path, weights_only=False)
    return None


def denormalize(arr, extrema, var_idx):
    """Denormalize from [0,1] to physical units."""
    if extrema is None:
        return arr
    y_min = extrema['y_min'][var_idx].item()
    y_max = extrema['y_max'][var_idx].item()
    return arr * (y_max - y_min) + y_min


# ==============================================================================
# RENDERING
# ==============================================================================

def render_on_ax(ax, values, pos, polys, cmap_name, vmin, vmax,
                 bg_color='#f5f5f5'):
    """Render values using PolyCollection (if polys) or scatter fallback."""
    ax.set_facecolor(bg_color)
    norm = Normalize(vmin=vmin, vmax=vmax)

    if polys is not None and len(polys) > 0:
        n_cells = min(len(polys), len(values))
        cmap_obj = plt.colormaps.get_cmap(cmap_name)
        colors = cmap_obj(norm(values[:n_cells]))
        pc = PolyCollection(polys[:n_cells], facecolors=colors,
                            edgecolors='face', linewidths=0.1)
        ax.add_collection(pc)
        ax.autoscale_view()
    else:
        ax.scatter(pos[:, 0], pos[:, 1], c=values, cmap=cmap_name,
                   s=0.5, norm=norm, rasterized=True)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ==============================================================================
# GROUND TRUTH / HELPERS
# ==============================================================================

def extract_gt_list(sim, n_steps, sf, df):
    """Extract GT dynamic variables for each rollout step."""
    gt = []
    for t in range(n_steps):
        if hasattr(sim[t], 'y') and sim[t].y is not None:
            gt.append(sim[t].y[:, :df].cpu().numpy())
        else:
            gt.append(sim[t + 1].x[:, sf:sf + df].cpu().numpy())
    return gt


def compute_gt_ranges(gt_list, var_indices, extrema=None):
    """Compute global value ranges from ground truth only."""
    n_vars = len(var_indices)
    vmin = [np.inf] * n_vars
    vmax = [-np.inf] * n_vars
    for gt in gt_list:
        for ri, vi in enumerate(var_indices):
            vals = gt[:, vi]
            if extrema is not None:
                vals = denormalize(vals, extrema, vi)
            vmin[ri] = min(vmin[ri], vals.min())
            vmax[ri] = max(vmax[ri], vals.max())
    for ri in range(n_vars):
        rng = vmax[ri] - vmin[ri]
        if rng < 1e-10:
            rng = 1.0
        vmin[ri] -= 0.02 * rng
        vmax[ri] += 0.02 * rng
    return vmin, vmax


def select_frames(n_steps, frame_skip):
    frames = list(range(0, n_steps, max(1, frame_skip)))
    if frames[-1] != n_steps - 1:
        frames.append(n_steps - 1)
    return frames


def get_sim_name_str(sim):
    d = sim[0]
    parts = []
    if hasattr(d, 'mesh_id'):
        v = d.mesh_id
        parts.append(f"mesh={v.item() if torch.is_tensor(v) else v}")
    return ' | '.join(parts) if parts else ''


# ==============================================================================
# COMPARISON GIF — PolyCollection rendering
# ==============================================================================

def create_comparison_gif(sim, model_results, sim_name, pos, polys,
                           output_dir, fps=6, frame_skip=1, var_indices=None,
                           extrema=None, sf=9, df=4):
    model_names = list(model_results.keys())

    n_steps = min(len(list(model_results.values())[0]), len(sim) - 1)
    gt_list = extract_gt_list(sim, n_steps, sf, df)
    n_vars_available = gt_list[0].shape[1] if gt_list else df

    if var_indices is None:
        var_indices = list(range(min(len(VAR_NAMES), n_vars_available)))

    display_names = [VAR_NAMES[i] if i < len(VAR_NAMES) else f'Var {i}' for i in var_indices]
    display_cmaps = [CMAPS[i] if i < len(CMAPS) else 'viridis' for i in var_indices]
    n_vars = len(var_indices)
    n_cols = 1 + len(model_names)
    n_rows = n_vars

    vmin, vmax = compute_gt_ranges(gt_list, var_indices, extrema)
    all_frames = select_frames(n_steps, frame_skip)
    gp_str = get_sim_name_str(sim)

    # Determine terrain label
    terrain = 'Iowa River' if 'iw' in sim_name.lower() else 'White River'

    col_labels = ['Ground Truth'] + model_names
    col_colors = [MODEL_COLORS.get(c, '#555') for c in col_labels]

    # Figure sizing — adapt to mesh aspect ratio
    if polys is not None and len(polys) > 0:
        all_pts = np.concatenate(polys)
        aspect = all_pts[:, 0].ptp() / max(all_pts[:, 1].ptp(), 1e-6)
    else:
        aspect = pos[:, 0].ptp() / max(pos[:, 1].ptp(), 1e-6)

    cell_w = max(2.2, min(3.5, 2.2 * aspect))
    cell_h = cell_w / max(aspect, 0.3)
    cell_h = max(1.5, min(3.5, cell_h))

    row_label_w = 0.8
    cbar_w = 1.0
    fig_w = row_label_w + n_cols * cell_w + cbar_w + 0.3
    fig_h = 0.7 + n_rows * cell_h + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    # Compute axes positions manually for better control
    left_margin = row_label_w / fig_w
    top_margin = 0.7 / fig_h
    cw = cell_w / fig_w
    ch = cell_h / fig_h
    y_top = 1.0 - top_margin

    axes = [[None] * n_cols for _ in range(n_rows)]
    for ri in range(n_rows):
        for ci in range(n_cols):
            x = left_margin + ci * cw
            y = y_top - (ri + 1) * ch
            ax = fig.add_axes([x, y, cw * 0.95, ch * 0.93])
            axes[ri][ci] = ax

            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#ddd')
                sp.set_linewidth(0.5)

            if ri == 0:
                ax.set_title(col_labels[ci], fontsize=9, fontweight='bold',
                             color=col_colors[ci], pad=6)
            if ci == 0:
                ax.text(-0.05, 0.5, display_names[ri],
                        transform=ax.transAxes, fontsize=9, fontweight='bold',
                        color='#333', ha='right', va='center', rotation=90)

    # Colorbars per row on right
    for ri in range(n_rows):
        ax_last = axes[ri][-1]
        pos_ax = ax_last.get_position()
        cax = fig.add_axes([pos_ax.x1 + 0.005, pos_ax.y0, 0.012, pos_ax.height])
        sm = plt.cm.ScalarMappable(norm=Normalize(vmin[ri], vmax[ri]),
                                    cmap=display_cmaps[ri])
        cb = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=6, colors='#444')
        cb.outline.set_edgecolor('#bbb')

    suptitle = fig.suptitle('', fontsize=12, fontweight='bold', color='#222', y=0.97)

    def animate(fi):
        t = all_frames[fi]
        gt = gt_list[t]

        for ri, vi in enumerate(var_indices):
            # Ground truth
            ax = axes[ri][0]
            ax.clear()
            gt_vals = gt[:, vi]
            if extrema is not None:
                gt_vals = denormalize(gt_vals, extrema, vi)
            render_on_ax(ax, gt_vals, pos, polys, display_cmaps[ri],
                         vmin[ri], vmax[ri])

            # Row label
            ax.text(-0.05, 0.5, display_names[ri],
                    transform=ax.transAxes, fontsize=9, fontweight='bold',
                    color='#333', ha='right', va='center', rotation=90)

            # Column header (first row only)
            if ri == 0:
                ax.set_title(col_labels[0], fontsize=9, fontweight='bold',
                             color=col_colors[0], pad=6)

            # Models
            for mi, mn in enumerate(model_names):
                ax = axes[ri][mi + 1]
                ax.clear()
                pred_vals = model_results[mn][t][:, vi]
                if extrema is not None:
                    pred_vals = denormalize(pred_vals, extrema, vi)
                render_on_ax(ax, pred_vals, pos, polys, display_cmaps[ri],
                             vmin[ri], vmax[ri])
                if ri == 0:
                    ax.set_title(mn, fontsize=9, fontweight='bold',
                                 color=col_colors[mi + 1], pad=6)

        time_str = format_physical_time(t)
        title = f'{terrain} — Step {t}/{n_steps - 1} (t = {time_str})'
        if gp_str:
            title += f'  [{gp_str}]'
        suptitle.set_text(title)

    print(f"  Creating comparison GIF: {len(all_frames)} frames @ {fps} fps...")
    anim = FuncAnimation(fig, animate, frames=len(all_frames),
                          interval=1000 // fps, blit=False)
    gif_path = output_dir / f'comparison_{sim_name}.gif'
    anim.save(str(gif_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  ✓ Saved: {gif_path}")
    return gif_path


# ==============================================================================
# ERROR GIF
# ==============================================================================

def create_error_gif(sim, model_results, sim_name, pos, polys,
                      output_dir, fps=6, frame_skip=1, exclude=None,
                      var_indices=None, extrema=None, sf=9, df=4):
    if exclude is None:
        exclude = []
    model_names = [mn for mn in model_results if mn not in exclude]
    if not model_names:
        print("  ⚠ No models left for error GIF"); return None

    n_steps = min(len(list(model_results.values())[0]), len(sim) - 1)
    gt_list = extract_gt_list(sim, n_steps, sf, df)
    n_vars_available = gt_list[0].shape[1] if gt_list else df

    if var_indices is None:
        var_indices = list(range(min(len(VAR_NAMES), n_vars_available)))

    display_names = [VAR_NAMES[i] if i < len(VAR_NAMES) else f'Var {i}' for i in var_indices]
    n_vars = len(var_indices)
    n_models = len(model_names)
    terrain = 'Iowa River' if 'iw' in sim_name.lower() else 'White River'

    # Error scale (95th percentile across included models)
    emax = [0.0] * n_vars
    for mn in model_names:
        for t in range(n_steps):
            for ri, vi in enumerate(var_indices):
                gt_v = gt_list[t][:, vi]
                pr_v = model_results[mn][t][:, vi]
                if extrema is not None:
                    gt_v = denormalize(gt_v, extrema, vi)
                    pr_v = denormalize(pr_v, extrema, vi)
                err = np.abs(pr_v - gt_v)
                emax[ri] = max(emax[ri], np.percentile(err, 95))

    all_frames = select_frames(n_steps, frame_skip)

    # Figure sizing
    if polys is not None and len(polys) > 0:
        all_pts = np.concatenate(polys)
        aspect = all_pts[:, 0].ptp() / max(all_pts[:, 1].ptp(), 1e-6)
    else:
        aspect = pos[:, 0].ptp() / max(pos[:, 1].ptp(), 1e-6)

    cell_w = max(2.2, min(3.5, 2.2 * aspect))
    cell_h = cell_w / max(aspect, 0.3)
    cell_h = max(1.5, min(3.5, cell_h))

    fig_w = 0.8 + n_models * cell_w + 1.0
    fig_h = 0.7 + n_vars * cell_h + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    left_margin = 0.8 / fig_w
    cw = cell_w / fig_w
    ch = cell_h / fig_h
    y_top = 1.0 - 0.7 / fig_h

    axes = [[None] * n_models for _ in range(n_vars)]
    for ri in range(n_vars):
        for ci, mn in enumerate(model_names):
            x = left_margin + ci * cw
            y = y_top - (ri + 1) * ch
            ax = fig.add_axes([x, y, cw * 0.95, ch * 0.93])
            axes[ri][ci] = ax
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#ddd'); sp.set_linewidth(0.5)
            if ri == 0:
                ax.set_title(mn, fontsize=9, fontweight='bold',
                             color=MODEL_COLORS.get(mn, '#555'), pad=6)
            if ci == 0:
                ax.text(-0.05, 0.5, f'|Err| {display_names[ri]}',
                        transform=ax.transAxes, fontsize=8, fontweight='bold',
                        color='#333', ha='right', va='center', rotation=90)

    # Colorbars per row
    for ri in range(n_vars):
        ax_last = axes[ri][-1]
        pos_ax = ax_last.get_position()
        cax = fig.add_axes([pos_ax.x1 + 0.005, pos_ax.y0, 0.012, pos_ax.height])
        sm = plt.cm.ScalarMappable(norm=Normalize(0, emax[ri]), cmap='hot')
        cb = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=6, colors='#444')
        cb.outline.set_edgecolor('#bbb')

    suptitle = fig.suptitle('', fontsize=12, fontweight='bold', color='#222', y=0.97)

    def animate(fi):
        t = all_frames[fi]
        gt = gt_list[t]

        for ri, vi in enumerate(var_indices):
            gt_v = gt[:, vi]
            if extrema is not None:
                gt_v = denormalize(gt_v, extrema, vi)

            for ci, mn in enumerate(model_names):
                ax = axes[ri][ci]
                ax.clear()
                pr_v = model_results[mn][t][:, vi]
                if extrema is not None:
                    pr_v = denormalize(pr_v, extrema, vi)
                err = np.abs(pr_v - gt_v)
                render_on_ax(ax, err, pos, polys, 'hot', 0, emax[ri])

                if ri == 0:
                    ax.set_title(mn, fontsize=9, fontweight='bold',
                                 color=MODEL_COLORS.get(mn, '#555'), pad=6)
                if ci == 0:
                    ax.text(-0.05, 0.5, f'|Err| {display_names[ri]}',
                            transform=ax.transAxes, fontsize=8, fontweight='bold',
                            color='#333', ha='right', va='center', rotation=90)

        time_str = format_physical_time(t)
        suptitle.set_text(f'Absolute Error — {terrain} — Step {t}/{n_steps - 1} (t = {time_str})')

    print(f"  Creating error GIF: {len(all_frames)} frames @ {fps} fps...")
    anim = FuncAnimation(fig, animate, frames=len(all_frames),
                          interval=1000 // fps, blit=False)
    gif_path = output_dir / f'error_{sim_name}.gif'
    anim.save(str(gif_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  ✓ Saved: {gif_path}")
    return gif_path


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="River Multi-Model Comparison GIFs")
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--hec_ras_dir", required=True,
                        help='Path to HEC-RAS dir with .hdf mesh files')
    parser.add_argument("--extrema", default=None,
                        help="Path to global_y_extrema.pth for denormalization")
    parser.add_argument("--models", nargs='+', required=True,
                        help="model_key:/path/to/ckpt pairs")
    parser.add_argument("--output_dir", default="./river_gifs")
    parser.add_argument("--sim_indices", type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument("--rollout_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_sims", type=int, default=None)
    parser.add_argument("--error_gif", action="store_true", default=True)
    parser.add_argument("--exclude_error", nargs='*', default=['MeshGraphNet', 'GraphSAGE'],
                        help="Model display names to exclude from error GIF")
    parser.add_argument("--vars", type=int, nargs='*', default=None,
                        help="Variable indices to show (default: all). 0=Depth, 1=Volume, 2=Vel_X, 3=Vel_Y")
    parser.add_argument("--dt_minutes", type=float, default=20,
                        help="Physical time per timestep in minutes")

    args = parser.parse_args()

    global DT_MINUTES
    DT_MINUTES = args.dt_minutes

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    sf = RV_NUM_STATIC  # 9 static features
    df = RV_NUM_DYNAMIC  # 4 dynamic features

    print(f"\n{'=' * 60}\nRIVER — MULTI-MODEL COMPARISON GIFS\n{'=' * 60}")

    # Extrema for denormalization
    extrema = load_extrema(args.extrema)
    if extrema is not None:
        print(f"Loaded extrema for denormalization")

    # Parse model specs
    specs = parse_model_specs(args.models, RV_REGISTRY)
    if not specs:
        print("No valid models found!"); return

    # Load test simulations
    sims = rv_load_data(args.test_dir, args.max_sims)
    if not sims:
        print("No simulations found!"); return

    # Prepare sample for model init
    sample = sims[0][1][0]
    if not hasattr(sample, 'pos') or sample.pos is None:
        sample.pos = sample.x[:, :sf]
    sample = sample.to(device)

    # Load models
    models = {}
    for mt, mp in specs.items():
        reg = RV_REGISTRY[mt]
        try:
            print(f"\nLoading {reg['name']}...")
            m = reg['load'](mp, sample, device)
            print(f"  ✓ {sum(p.numel() for p in m.parameters()):,} params")
            models[mt] = {'model': m, 'name': reg['name'],
                          'rollout': reg['rollout'], 'device': device}
        except Exception as e:
            print(f"  ✗ Failed: {e}"); import traceback; traceback.print_exc()

    if not models:
        print("No models loaded!"); return

    # Generate GIFs per simulation
    for sim_idx in args.sim_indices:
        if sim_idx >= len(sims):
            print(f"\n⚠ Index {sim_idx} out of range ({len(sims)} sims)"); continue

        sim_name, sim = sims[sim_idx]
        steps = min(args.rollout_steps, len(sim) - 1)
        terrain = 'Iowa River' if 'iw' in sim_name.lower() else 'White River'
        print(f"\n{'─' * 60}\nSim {sim_idx}: {sim_name} ({terrain}, {steps} steps)")

        # Get node positions
        pos = get_node_positions(sim)
        print(f"  {pos.shape[0]} nodes")

        # Load HEC-RAS polygon mesh
        polys = load_mesh_for_sim(sim_name, args.hec_ras_dir)
        if polys is None:
            print("  ⚠ No HDF mesh — will use scatter fallback")

        # Clear MLS caches between simulations
        _clear_mls_caches()

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

        if not model_results:
            continue

        create_comparison_gif(sim, model_results, sim_name, pos, polys,
                               output_dir, fps=args.fps, frame_skip=args.frame_skip,
                               var_indices=args.vars, extrema=extrema, sf=sf, df=df)

        if args.error_gif:
            create_error_gif(sim, model_results, sim_name, pos, polys,
                              output_dir, fps=args.fps, frame_skip=args.frame_skip,
                              exclude=args.exclude_error, var_indices=args.vars,
                              extrema=extrema, sf=sf, df=df)

    print(f"\n{'=' * 60}\n✓ All GIFs → {output_dir}\n{'=' * 60}")


if __name__ == "__main__":
    main()
