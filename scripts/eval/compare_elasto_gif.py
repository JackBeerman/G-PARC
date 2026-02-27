#!/usr/bin/env python3
"""
Elastoplastic — Multi-Model Comparison GIFs
=============================================
Generates animated GIFs showing ground truth vs model predictions
on deforming Lagrangian meshes with element erosion.

Layout per frame (rows=variables, cols=models):
    Columns: [Ground Truth] [G-PARC w/ MLS] [G-PARC Baseline] [MeshGraphKAN] ...
    Row 0:   U_x            U_x              U_x                U_x            | colorbar
    Row 1:   U_y            U_y              U_y                U_y            | colorbar

Usage:
    python compare_elasto_gif.py \
        --test_dir /path/to/test \
        --models gparcv2:/path/v2.pth gparcv1:/path/v1.pth mgkan:/path/mgkan.pth \
        --output_dir ./elasto_gifs \
        --sim_indices 0 1 2 \
        --rollout_steps 37 \
        --fps 6
"""

import argparse, sys, os, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from eval import (
    EL_REGISTRY, EL_VAR_NAMES, EL_NUM_STATIC, EL_NUM_DYNAMIC,
    el_load_data, el_extract_dynamic,
    parse_model_specs, _clear_mls_caches,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================

VAR_NAMES = ['Displacement Ux', 'Displacement Uy']
CMAPS = ['RdBu_r', 'RdBu_r']
EROSION_THRESHOLD = -1.0
MODEL_COLORS = {
    'Ground Truth':    '#333333',
    'G-PARC with MLS': '#1f77b4',
    'G-PARC Baseline': '#d62728',
    'MeshGraphKAN':    '#ff7f0e',
    'MeshGraphNet':    '#2ca02c',
    'GraphSAGE':       '#9467bd',
}


# ==============================================================================
# MESH / EROSION HELPERS
# ==============================================================================

def get_erosion_mask(data, num_elements):
    """Boolean mask: True = eroded element."""
    if hasattr(data, 'x_element') and data.x_element is not None:
        erosion = data.x_element.cpu().numpy().flatten()
        return erosion < EROSION_THRESHOLD
    return np.zeros(num_elements, dtype=bool)


def get_elements(sim):
    """Extract element connectivity from first timestep."""
    d = sim[0]
    if hasattr(d, 'elements') and d.elements is not None:
        return d.elements.cpu().numpy() if torch.is_tensor(d.elements) else np.array(d.elements)
    return None


def get_ref_positions(sim):
    """Reference (undeformed) node positions from first timestep."""
    d = sim[0]
    return d.x[:, :EL_NUM_STATIC].cpu().numpy()


def build_polygons(positions, elements, eroded_mask=None):
    """Build polygon vertices for PolyCollection.
    Returns: list of (N_elem, corners, 2) polygon vertex arrays (valid only).
    """
    if eroded_mask is None:
        eroded_mask = np.zeros(len(elements), dtype=bool)
    valid = ~eroded_mask
    valid_elems = elements[valid]
    polys = positions[valid_elems]  # (n_valid, corners_per_elem, 2)
    return polys, valid


def element_avg(node_values, elements, valid_mask):
    """Average node values per element, return for valid elements only."""
    valid_elems = elements[valid_mask]
    return node_values[valid_elems].mean(axis=1)  # (n_valid,)


# ==============================================================================
# GROUND TRUTH EXTRACTION
# ==============================================================================

def extract_gt_list(sim, n_steps):
    """Extract GT dynamic variables for each rollout step."""
    gt = []
    for t in range(n_steps):
        if hasattr(sim[t], 'y') and sim[t].y is not None:
            gt.append(sim[t].y.cpu().numpy())
        else:
            gt.append(el_extract_dynamic(sim[t + 1].x).cpu().numpy())
    return gt


def get_deformed_positions(ref_pos, displacement):
    """Compute deformed positions from reference + displacement."""
    n_vars = displacement.shape[1] if displacement.ndim > 1 else 1
    if n_vars >= 2:
        return ref_pos + displacement[:, :2]
    return ref_pos


def compute_gt_ranges(gt_list, n_vars):
    """Compute global value ranges from ground truth only."""
    vmin = [np.inf] * n_vars
    vmax = [-np.inf] * n_vars
    for gt in gt_list:
        for vi in range(n_vars):
            vmin[vi] = min(vmin[vi], gt[:, vi].min())
            vmax[vi] = max(vmax[vi], gt[:, vi].max())
    for vi in range(n_vars):
        rng = vmax[vi] - vmin[vi]
        vmin[vi] -= 0.02 * rng
        vmax[vi] += 0.02 * rng
    return vmin, vmax


def compute_spatial_bounds(sim, gt_list, n_steps):
    """Compute XY bounds across all deformed configurations."""
    ref_pos = get_ref_positions(sim)
    xmin, xmax = ref_pos[:, 0].min(), ref_pos[:, 0].max()
    ymin, ymax = ref_pos[:, 1].min(), ref_pos[:, 1].max()
    for gt in gt_list:
        dp = get_deformed_positions(ref_pos, gt)
        xmin = min(xmin, dp[:, 0].min())
        xmax = max(xmax, dp[:, 0].max())
        ymin = min(ymin, dp[:, 1].min())
        ymax = max(ymax, dp[:, 1].max())
    pad_x = 0.05 * (xmax - xmin)
    pad_y = 0.05 * (ymax - ymin)
    return (xmin - pad_x, xmax + pad_x), (ymin - pad_y, ymax + pad_y)


def select_frames(n_steps, frame_skip):
    frames = list(range(0, n_steps, max(1, frame_skip)))
    if frames[-1] != n_steps - 1:
        frames.append(n_steps - 1)
    return frames


# ==============================================================================
# RENDER HELPER
# ==============================================================================

def render_mesh(ax, positions, elements, node_values, eroded_mask,
                cmap, vmin, vmax, edgecolor='#ccc', linewidth=0.1):
    """Render mesh with PolyCollection (element-averaged coloring)."""
    polys, valid = build_polygons(positions, elements, eroded_mask)
    vals = element_avg(node_values, elements, valid)
    pc = PolyCollection(polys, array=vals, cmap=cmap, edgecolors=edgecolor,
                        linewidths=linewidth)
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    return pc


# ==============================================================================
# COMPARISON GIF
# ==============================================================================

def create_comparison_gif(sim, model_results, sim_name, elements,
                           output_dir, fps=6, frame_skip=1):
    model_names = list(model_results.keys())
    n_vars = len(VAR_NAMES)
    n_cols = 1 + len(model_names)
    n_rows = n_vars

    n_steps = min(len(list(model_results.values())[0]), len(sim) - 1)
    gt_list = extract_gt_list(sim, n_steps)
    vmin, vmax = compute_gt_ranges(gt_list, n_vars)
    all_frames = select_frames(n_steps, frame_skip)
    ref_pos = get_ref_positions(sim)
    xlim, ylim = compute_spatial_bounds(sim, gt_list, n_steps)

    col_labels = ['Ground Truth'] + model_names
    col_colors = [MODEL_COLORS.get(c, '#555') for c in col_labels]

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.0 * n_cols + 1.2, 3.2 * n_rows + 0.6),
                              gridspec_kw={'hspace': 0.10, 'wspace': 0.06,
                                           'right': 0.88, 'left': 0.06,
                                           'top': 0.90, 'bottom': 0.03})
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.patch.set_facecolor('white')

    # Setup axes
    for ri in range(n_rows):
        for ci in range(n_cols):
            ax = axes[ri, ci]
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#ddd'); sp.set_linewidth(0.5)
            if ri == 0:
                ax.set_title(col_labels[ci], fontsize=10, fontweight='bold',
                             color=col_colors[ci], pad=8)
            if ci == 0:
                ax.set_ylabel(VAR_NAMES[ri], fontsize=10, fontweight='bold',
                              color='#333', rotation=90, labelpad=10)

    # Colorbars per row on right
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
        gt = gt_list[t]
        eroded = get_erosion_mask(sim[t], len(elements))
        gt_pos = get_deformed_positions(ref_pos, gt)

        for ri in range(n_rows):
            # Clear and re-render each subplot
            # Ground truth
            ax = axes[ri, 0]
            ax.collections.clear()
            render_mesh(ax, gt_pos, elements, gt[:, ri], eroded,
                        CMAPS[ri], vmin[ri], vmax[ri])

            # Models
            for mi, mn in enumerate(model_names):
                ax = axes[ri, mi + 1]
                ax.collections.clear()
                pred = model_results[mn][t]
                pred_pos = get_deformed_positions(ref_pos, pred)
                render_mesh(ax, pred_pos, elements, pred[:, ri], eroded,
                            CMAPS[ri], vmin[ri], vmax[ri])

        suptitle.set_text(f'Rollout Step {t}/{n_steps - 1}')

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

def create_error_gif(sim, model_results, sim_name, elements,
                      output_dir, fps=6, frame_skip=1, exclude=None):
    if exclude is None:
        exclude = []
    model_names = [mn for mn in model_results if mn not in exclude]
    if not model_names:
        print("  ⚠ No models left for error GIF"); return None

    n_vars = len(VAR_NAMES)
    n_models = len(model_names)
    n_steps = min(len(list(model_results.values())[0]), len(sim) - 1)
    gt_list = extract_gt_list(sim, n_steps)
    ref_pos = get_ref_positions(sim)
    xlim, ylim = compute_spatial_bounds(sim, gt_list, n_steps)

    # Error scale from included models only (95th percentile)
    emax = [0.0] * n_vars
    for mn in model_names:
        for t in range(n_steps):
            err = np.abs(model_results[mn][t] - gt_list[t])
            for vi in range(n_vars):
                emax[vi] = max(emax[vi], np.percentile(err[:, vi], 95))

    all_frames = select_frames(n_steps, frame_skip)

    fig, axes = plt.subplots(n_vars, n_models,
                              figsize=(3.0 * n_models + 1.2, 3.2 * n_vars + 0.6),
                              gridspec_kw={'hspace': 0.10, 'wspace': 0.06,
                                           'right': 0.88, 'left': 0.06,
                                           'top': 0.90, 'bottom': 0.03})
    if n_vars == 1:
        axes = axes[np.newaxis, :]
    if n_models == 1:
        axes = axes[:, np.newaxis]
    fig.patch.set_facecolor('white')

    for ri in range(n_vars):
        for ci, mn in enumerate(model_names):
            ax = axes[ri, ci]
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#ddd'); sp.set_linewidth(0.5)
            if ri == 0:
                ax.set_title(mn, fontsize=10, fontweight='bold',
                             color=MODEL_COLORS.get(mn, '#555'), pad=8)
            if ci == 0:
                ax.set_ylabel(f'|Err| {VAR_NAMES[ri]}', fontsize=9, fontweight='bold',
                              color='#333', rotation=90, labelpad=10)

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
        gt = gt_list[t]
        eroded = get_erosion_mask(sim[t], len(elements))
        gt_pos = get_deformed_positions(ref_pos, gt)

        for ri in range(n_vars):
            for ci, mn in enumerate(model_names):
                ax = axes[ri, ci]
                ax.collections.clear()
                err = np.abs(model_results[mn][t] - gt)
                # Use GT deformed positions for error overlay
                render_mesh(ax, gt_pos, elements, err[:, ri], eroded,
                            'hot', 0, emax[ri])

        suptitle.set_text(f'Absolute Error — Step {t}/{n_steps - 1}')

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
    parser = argparse.ArgumentParser(description="Elastoplastic Multi-Model Comparison GIFs")
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--models", nargs='+', required=True,
                        help="model_key:/path/to/ckpt pairs")
    parser.add_argument("--output_dir", default="./elasto_gifs")
    parser.add_argument("--sim_indices", type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument("--rollout_steps", type=int, default=37)
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

    print(f"\n{'=' * 60}\nELASTOPLASTIC — MULTI-MODEL COMPARISON GIFS\n{'=' * 60}")

    # Parse model specs
    specs = parse_model_specs(args.models, EL_REGISTRY)
    if not specs:
        print("No valid models found!"); return

    # Load test simulations
    sims = el_load_data(args.test_dir, args.max_sims)
    if not sims:
        print("No simulations found!"); return

    # Get elements from first sim
    elements = get_elements(sims[0][1])
    if elements is None:
        print("ERROR: No element connectivity found in data!"); return
    print(f"Elements: {len(elements)} ({elements.shape[1]}-node)")

    # Prepare sample for model init
    sample = sims[0][1][0]
    if not hasattr(sample, 'pos') or sample.pos is None:
        sample.pos = sample.x[:, :EL_NUM_STATIC]
    sample = sample.to(device)

    # Load models
    models = {}
    for mt, mp in specs.items():
        reg = EL_REGISTRY[mt]
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
        print(f"\n{'─' * 60}\nSim {sim_idx}: {sim_name} ({steps} steps)")

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

        create_comparison_gif(sim, model_results, sim_name, elements,
                               output_dir, fps=args.fps, frame_skip=args.frame_skip)

        if args.error_gif:
            create_error_gif(sim, model_results, sim_name, elements,
                              output_dir, fps=args.fps, frame_skip=args.frame_skip,
                              exclude=args.exclude_error)

    print(f"\n{'=' * 60}\n✓ All GIFs → {output_dir}\n{'=' * 60}")


if __name__ == "__main__":
    main()
