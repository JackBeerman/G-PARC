#!/usr/bin/env python3
"""
River Paper Figures — PolyCollection mesh rendering
=====================================================
Publication-quality figures for river flood simulations.

Separate figures per variable AND per terrain (Iowa River vs White River).
Uses HEC-RAS .hdf mesh files for PolyCollection rendering.

Layout per figure:
    Columns: t_early | t_mid | t_late  (3 timesteps)
    Rows:    GT | G-PARC | G-PARC (w/o MLS) | MeshGraphKAN   [4 rows default]

Usage:
    python paper_figure.py \\
        --test_dir /path/to/test_data \\
        --models gparcv2:/path gparcv1:/path mgkan:/path \\
        --hec_ras_dir "/standard/sds_baek_energetic/HEC_RAS (River)" \\
        --output_dir /scratch/.../figures \\
        --extrema_path /path/to/global_y_extrema.pth \\
        --sim_index 0 --rollout_steps 20 --error_fig

    # Override which models appear (and in what order):
    python paper_figure.py ... --paper_models gparcv2 gparcv1 mgkan mgnet
"""

import argparse, sys, os, json
from pathlib import Path
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("⚠️  h5py not available — will use scatter plots")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.river.eval_comparison import (
    VAR_NAMES, MODEL_LABELS, MODEL_LOADERS, ROLLOUT_FNS,
)
from visualizations.mesh_io import load_mesh_for_sim, get_node_positions, HAS_H5PY
from visualizations.river_viz import VAR_CMAPS, VAR_UNITS_PHYS, VAR_UNITS_NORM

VAR_LABELS = {
    'Depth':      'Depth',
    'Volume':     'Volume',
    'Vel_X':      r'$V_x$',
    'Vel_Y':      r'$V_y$',
    'Velocity_X': r'$V_x$',
    'Velocity_Y': r'$V_y$',
}

# ---------------------------------------------------------------------------
# Paper-specific label overrides
#   gparcv2 (with MLS)  → "G-PARC"
#   gparcv1 (baseline)  → "G-PARC (w/o MLS)"
# ---------------------------------------------------------------------------
PAPER_MODEL_LABELS = {
    'gparcv2':    'G-PARC',
    'gparcv1':    'G-PARC (w/o MLS)',
    'mgkan':      'MeshGraphKAN',
    'mgnet':      'MeshGraphNet',
    'graphsage':  'GraphSAGE',
}

# Default model subset for paper figures (GT row is always included)
DEFAULT_PAPER_MODELS = ['gparcv2', 'gparcv1', 'mgkan']

# Publication-friendly colormaps
PAPER_CMAPS = ['viridis', 'viridis', 'RdBu_r', 'RdBu_r']

# Physical time per timestep (minutes)
DT_MINUTES = 20


# ===========================================================================
# TIME FORMATTING
# ===========================================================================

def format_physical_time(step, dt_minutes=DT_MINUTES):
    """Convert a timestep index to a physical time string."""
    total_min = step * dt_minutes
    if total_min >= 60:
        hours = total_min / 60
        if hours == int(hours):
            return f'{int(hours)} hr'
        return f'{hours:.1f} hr'
    return f'{total_min} min'


# ===========================================================================
# DENORMALIZATION
# ===========================================================================

def denormalize(normalized, extrema, var_idx):
    """Denormalize from [0,1] to physical units."""
    y_min = extrema['y_min'][var_idx].item()
    y_max = extrema['y_max'][var_idx].item()
    return normalized * (y_max - y_min) + y_min


# ===========================================================================
# PATH PARSER
# ===========================================================================

def parse_model_specs(raw_specs, loaders):
    model_specs = {}
    i = 0
    while i < len(raw_specs):
        token = raw_specs[i]
        colon_idx = token.find(':')
        if colon_idx > 0 and token[:colon_idx] in loaders:
            mtype = token[:colon_idx]
            path_parts = [token[colon_idx + 1:]]
            j = i + 1
            while j < len(raw_specs):
                nc = raw_specs[j].find(':')
                if nc > 0 and raw_specs[j][:nc] in loaders:
                    break
                path_parts.append(raw_specs[j])
                j += 1
            mpath = ' '.join(path_parts)
            if Path(mpath).exists():
                model_specs[mtype] = mpath
            else:
                print(f"⚠ Checkpoint not found: {mpath}")
            i = j
        else:
            i += 1
    return model_specs


# ===========================================================================
# RENDER HELPER
# ===========================================================================

def render_on_ax(ax, values, pos, polys, cmap_name, norm, bg_color='#f5f5f5'):
    """Render values using PolyCollection (if polys) or scatter."""
    ax.set_facecolor(bg_color)

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


# ===========================================================================
# SINGLE-VARIABLE FIELD FIGURE
# ===========================================================================

def make_field_figure(gt_list, model_preds, model_order, timesteps,
                      var_idx, var_name, pos, polys,
                      output_path, sim_name='', dpi=300,
                      extrema=None):
    """
    One figure for a single variable on one terrain.

    Columns: 3 timesteps
    Rows:    GT + each model
    Color range from GT only.
    """
    n_times = len(timesteps)
    n_rows = 1 + len(model_order)

    cmap_name = PAPER_CMAPS[var_idx] if var_idx < len(PAPER_CMAPS) else 'viridis'

    # Extract single-variable arrays, denormalize if possible
    def get_var(data_list, t):
        arr = data_list[t][:, var_idx]
        if extrema is not None:
            arr = denormalize(arr, extrema, var_idx)
        return arr

    # Color range from GT only
    gt_vals = [get_var(gt_list, t) for t in timesteps]
    vmin = min(v.min() for v in gt_vals)
    vmax = max(v.max() for v in gt_vals)
    pad = max((vmax - vmin) * 0.02, 1e-8)
    vmin -= pad
    vmax += pad
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Unit label
    unit = VAR_UNITS_PHYS[var_idx] if extrema is not None else '(norm)'
    label = f'{VAR_LABELS.get(var_name, var_name)} ({unit})'

    # Figure sizing — adapt to mesh aspect ratio
    if polys is not None and len(polys) > 0:
        all_pts = np.concatenate(polys)
        xr = all_pts[:, 0].ptp()
        yr = all_pts[:, 1].ptp()
    else:
        xr = pos[:, 0].ptp()
        yr = pos[:, 1].ptp()
    aspect = xr / max(yr, 1e-6)

    cell_w = max(2.5, min(4.0, 2.5 * aspect))
    cell_h = cell_w / max(aspect, 0.3)
    cell_h = max(1.5, min(4.0, cell_h))

    row_label_w = 1.3
    cbar_w = 0.4
    header_h = 0.6

    fig_w = row_label_w + n_times * cell_w + cbar_w + 0.3
    fig_h = header_h + n_rows * cell_h + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    x0 = row_label_w / fig_w
    y_top = 1.0 - header_h / fig_h
    cw = cell_w / fig_w
    ch = cell_h / fig_h

    # Use paper labels (fall back to eval_comparison labels)
    row_labels = ['Ground Truth'] + [
        PAPER_MODEL_LABELS.get(m, MODEL_LABELS.get(m, m))
        for m in model_order
    ]

    for row in range(n_rows):
        data_src = gt_list if row == 0 else model_preds.get(model_order[row - 1], [])

        for ti, t in enumerate(timesteps):
            x = x0 + ti * cw
            y = y_top - (row + 1) * ch

            ax = fig.add_axes([x, y, cw * 0.95, ch * 0.93])

            if t < len(data_src):
                vals = get_var(data_src, t)
            else:
                vals = np.zeros(pos.shape[0])

            render_on_ax(ax, vals, pos, polys, cmap_name, norm)

            # Row label on first column
            if ti == 0:
                ax.text(-0.05, 0.5, row_labels[row],
                        transform=ax.transAxes, fontsize=8, fontweight='bold',
                        ha='right', va='center', rotation=90)

            # Column header on first row
            if row == 0:
                time_str = format_physical_time(t)
                ax.set_title(f't = {time_str}', fontsize=10, fontweight='bold', pad=4)

    # Per-row colorbars (all share the same GT-derived norm)
    cb_x = x0 + n_times * cw + 0.005
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_name)
    sm.set_array([])
    for row in range(n_rows):
        cb_y = y_top - (row + 1) * ch
        cb_h_row = ch * 0.85
        cb_y_padded = cb_y + (ch - cb_h_row) * 0.5  # vertically center in row
        cbar_ax = fig.add_axes([cb_x, cb_y_padded, 0.012, cb_h_row])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(label, fontsize=8, labelpad=4)

    terrain = 'Iowa River' if 'iw' in sim_name.lower() else 'White River'
    fig.text(0.5, 0.97, f'{VAR_LABELS.get(var_name, var_name)} — {terrain}',
             fontsize=11, fontweight='bold', ha='center', va='top')

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# ===========================================================================
# SINGLE-VARIABLE ERROR FIGURE
# ===========================================================================

def make_error_figure(gt_list, model_preds, model_order, timesteps,
                      var_idx, var_name, pos, polys,
                      output_path, sim_name='', dpi=300,
                      extrema=None):
    """Absolute error figure for one variable. No GT row."""
    n_times = len(timesteps)
    n_rows = len(model_order)

    cmap_name = 'hot_r'

    def get_var(data_list, t):
        arr = data_list[t][:, var_idx]
        if extrema is not None:
            arr = denormalize(arr, extrema, var_idx)
        return arr

    # Error range (99th percentile)
    emax = 0
    for m in model_order:
        if m in model_preds:
            for t in timesteps:
                if t < len(model_preds[m]):
                    err = np.abs(get_var(model_preds[m], t) - get_var(gt_list, t))
                    emax = max(emax, float(np.percentile(err, 99)))
    emax = max(emax, 1e-8)
    norm = Normalize(vmin=0, vmax=emax)

    unit = VAR_UNITS_PHYS[var_idx] if extrema is not None else '(norm)'

    # Sizing
    if polys is not None and len(polys) > 0:
        all_pts = np.concatenate(polys)
        aspect = all_pts[:, 0].ptp() / max(all_pts[:, 1].ptp(), 1e-6)
    else:
        aspect = pos[:, 0].ptp() / max(pos[:, 1].ptp(), 1e-6)

    cell_w = max(2.5, min(4.0, 2.5 * aspect))
    cell_h = cell_w / max(aspect, 0.3)
    cell_h = max(1.5, min(4.0, cell_h))

    row_label_w = 1.3
    header_h = 0.6
    fig_w = row_label_w + n_times * cell_w + 0.7
    fig_h = header_h + n_rows * cell_h + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    x0 = row_label_w / fig_w
    y_top = 1.0 - header_h / fig_h
    cw = cell_w / fig_w
    ch = cell_h / fig_h

    for row, mname in enumerate(model_order):
        preds = model_preds.get(mname, [])
        label = PAPER_MODEL_LABELS.get(mname, MODEL_LABELS.get(mname, mname))

        for ti, t in enumerate(timesteps):
            x = x0 + ti * cw
            y = y_top - (row + 1) * ch

            ax = fig.add_axes([x, y, cw * 0.95, ch * 0.93])

            if t < len(preds):
                err = np.abs(get_var(preds, t) - get_var(gt_list, t))
            else:
                err = np.zeros(pos.shape[0])

            render_on_ax(ax, err, pos, polys, cmap_name, norm)

            if ti == 0:
                ax.text(-0.05, 0.5, label,
                        transform=ax.transAxes, fontsize=8, fontweight='bold',
                        ha='right', va='center', rotation=90)
            if row == 0:
                time_str = format_physical_time(t)
                ax.set_title(f't = {time_str}', fontsize=10, fontweight='bold', pad=4)

    # Per-row colorbars (all share the same error norm)
    cb_x = x0 + n_times * cw + 0.005
    var_label = VAR_LABELS.get(var_name, var_name)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_name)
    sm.set_array([])
    for row in range(n_rows):
        cb_y = y_top - (row + 1) * ch
        cb_h_row = ch * 0.85
        cb_y_padded = cb_y + (ch - cb_h_row) * 0.5
        cbar_ax = fig.add_axes([cb_x, cb_y_padded, 0.012, cb_h_row])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(f'|Δ{var_label}| ({unit})', fontsize=8, labelpad=4)

    terrain = 'Iowa River' if 'iw' in sim_name.lower() else 'White River'
    fig.text(0.5, 0.97, f'Absolute Error: {var_label} — {terrain}',
             fontsize=11, fontweight='bold', ha='center', va='top')

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="River Paper Figures")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--hec_ras_dir", type=str, required=True,
                        help="Path to HEC-RAS dir with .hdf mesh files")
    parser.add_argument("--extrema_path", type=str, default=None,
                        help="Path to global_y_extrema.pth for denormalization")
    parser.add_argument("--sim_index_wr", type=int, default=0,
                        help="White River sim index (among WR sims)")
    parser.add_argument("--sim_index_iw", type=int, default=0,
                        help="Iowa River sim index (among IW sims)")
    parser.add_argument("--rollout_steps", type=int, default=20)
    parser.add_argument("--timesteps", type=int, nargs='+', default=None,
                        help="Specific timesteps (default: 3 evenly spaced)")
    parser.add_argument("--variables", type=int, nargs='+', default=None,
                        help="Variable indices to plot (default: all). 0=Depth, 1=Volume, 2=Vel_X, 3=Vel_Y")
    parser.add_argument("--paper_models", type=str, nargs='+', default=None,
                        help="Which models to include in figures and in what order "
                             "(default: gparcv2 gparcv1 mgkan). "
                             "Only models present in --models will appear.")
    parser.add_argument("--dt_minutes", type=float, default=20,
                        help="Physical time per timestep in minutes (default: 20)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--error_fig", action='store_true',
                        help="Also generate absolute-error figures")
    args = parser.parse_args()

    # Update global dt if overridden
    global DT_MINUTES
    DT_MINUTES = args.dt_minutes

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model_specs = parse_model_specs(args.models, MODEL_LOADERS)

    # Determine which models to show (and in what order)
    paper_models = args.paper_models if args.paper_models else DEFAULT_PAPER_MODELS
    # Filter to only those actually provided via --models
    model_order = [m for m in paper_models if m in model_specs]
    skipped = [m for m in paper_models if m not in model_specs]
    if skipped:
        print(f"⚠ Requested but not found in --models: {skipped}")
    print(f"Figure model order: {[PAPER_MODEL_LABELS.get(m, m) for m in model_order]}")

    # Denormalization extrema
    extrema = None
    if args.extrema_path and Path(args.extrema_path).exists():
        extrema = torch.load(args.extrema_path, weights_only=False)
        print(f"Loaded extrema: {args.extrema_path}")

    # Load test data and split by terrain
    test_dir = Path(args.test_dir)
    all_files = sorted(test_dir.glob("*.pt"))
    print(f"Found {len(all_files)} test files")

    wr_files = [f for f in all_files if 'iw' not in f.stem.lower()]
    iw_files = [f for f in all_files if 'iw' in f.stem.lower()]
    print(f"  White River: {len(wr_files)}, Iowa River: {len(iw_files)}")

    # Select one sim per terrain
    sim_selections = {}
    if wr_files and args.sim_index_wr < len(wr_files):
        sim_selections['wr'] = wr_files[args.sim_index_wr]
    if iw_files and args.sim_index_iw < len(iw_files):
        sim_selections['iw'] = iw_files[args.sim_index_iw]

    if not sim_selections:
        print("No simulations found!"); return

    # Variables to plot
    var_indices = args.variables if args.variables else list(range(len(VAR_NAMES)))

    # Process each terrain
    for terrain_key, sim_file in sim_selections.items():
        sim_name = sim_file.stem
        terrain_label = 'Iowa River' if terrain_key == 'iw' else 'White River'
        print(f"\n{'=' * 60}")
        print(f"  {terrain_label}: {sim_name}")
        print(f"{'=' * 60}")

        sim = torch.load(sim_file, weights_only=False)
        if not isinstance(sim, list):
            print(f"  Skipping {sim_name}: not a list"); continue

        sample = sim[0]
        if not hasattr(sample, 'pos') or sample.pos is None:
            sample.pos = sample.x[:, :2]

        sf = 9   # river: 9 static features
        df = 4   # river: 4 dynamic features (depth, volume, vel_x, vel_y)
        max_available = len(sim) - 1
        steps = min(args.rollout_steps, max_available)
        n_nodes = sample.x.size(0)
        pos = sample.pos.cpu().numpy()
        print(f"  {n_nodes} nodes, {len(sim)} timesteps, rollout {steps}")

        # Load mesh
        polys = None
        if args.hec_ras_dir:
            polys = load_mesh_for_sim(sim_name, args.hec_ras_dir)

        # Ground truth
        gt_list = []
        for t in range(steps):
            if hasattr(sim[t], 'y') and sim[t].y is not None:
                gt_list.append(sim[t].y[:, :df].cpu().numpy())
            else:
                gt_list.append(sim[t + 1].x[:, sf:sf + df].cpu().numpy())

        # Timesteps — use terrain-appropriate spacing
        if args.timesteps:
            timesteps = [t for t in args.timesteps if t < steps]
        else:
            # 3 evenly spaced through available range
            timesteps = [steps // 6, steps // 2, steps - 1]
        print(f"  Timesteps: {timesteps} (physical: {[format_physical_time(t) for t in timesteps]})")

        # Load models & rollout  (only those in model_order)
        model_preds = {}
        for mtype in model_order:
            mpath = model_specs[mtype]
            try:
                pname = PAPER_MODEL_LABELS.get(mtype, MODEL_LABELS.get(mtype, mtype))
                print(f"\n  Loading {pname}...")
                model = MODEL_LOADERS[mtype](mpath, device, sf=sf, df=df)
                n_params = sum(p.numel() for p in model.parameters())
                print(f"    {n_params:,} parameters")

                sim_gpu = [d.to(device) for d in sim]
                for d in sim_gpu:
                    if not hasattr(d, 'pos') or d.pos is None:
                        d.pos = d.x[:, :2]

                with torch.no_grad():
                    preds = ROLLOUT_FNS[mtype](model, sim_gpu, steps, device)
                model_preds[mtype] = preds

                # Quick RRMSE on depth
                all_p = np.concatenate([p[:, 0] for p in preds])
                all_g = np.concatenate([g[:, 0] for g in gt_list[:len(preds)]])
                rrmse = (np.sqrt(np.mean((all_p - all_g)**2))
                         / max(np.sqrt(np.mean(all_g**2)), 1e-12))
                print(f"    Depth RRMSE: {rrmse:.4f}")

                del model; torch.cuda.empty_cache()
            except Exception as e:
                pname = PAPER_MODEL_LABELS.get(mtype, MODEL_LABELS.get(mtype, mtype))
                print(f"    ✗ {pname} failed: {e}")
                import traceback; traceback.print_exc()

        active_models = [m for m in model_order if m in model_preds]
        if not active_models:
            print(f"  No models succeeded for {terrain_label}!"); continue

        # Generate figures — one per variable
        print(f"\n  Generating figures for {terrain_label}...")
        for vi in var_indices:
            if vi >= len(VAR_NAMES):
                continue
            vn = VAR_NAMES[vi]
            tag = f"{vn.lower()}_{terrain_key}_{sim_name}"

            for ext in ['png', 'pdf']:
                fpath = output_dir / f'{tag}.{ext}'
                make_field_figure(gt_list, model_preds, active_models, timesteps,
                                  vi, vn, pos, polys, fpath,
                                  sim_name=sim_name, dpi=args.dpi,
                                  extrema=extrema)

            if args.error_fig:
                for ext in ['png', 'pdf']:
                    epath = output_dir / f'{tag}_error.{ext}'
                    make_error_figure(gt_list, model_preds, active_models, timesteps,
                                      vi, vn, pos, polys, epath,
                                      sim_name=sim_name, dpi=args.dpi,
                                      extrema=extrema)

    print(f"\n✓ Done! Figures in {output_dir}")


if __name__ == "__main__":
    main()