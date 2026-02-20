#!/usr/bin/env python3
"""
Shock Tube Paper Figures — One per field variable
===================================================
Generates publication-quality figures, one per field variable.

Each figure:
    Columns: t_early | t_mid | t_late  (3 timesteps)
    Rows:    GT | G-PARCv1 | G-PARCv2 | MeshGraphKAN | MeshGraphNet

Color range is set from GROUND TRUTH only (global min/max across timesteps).

Usage:
    python paper_figure.py \\
        --test_dir /path/to/test_cases_normalized \\
        --models gparcv1:/path gparcv2:/path mgkan:/path mgnet:/path \\
        --output_dir /scratch/.../figures \\
        --sim_index 0 --rollout_steps 40 --error_fig
"""

import argparse, sys, os, json
from pathlib import Path
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.shockchord.eval_comparison import (
    NUM_STATIC, NUM_USED_DYNAMIC, SKIP_INDICES, RAW_DYNAMIC, KEEP_INDICES,
    VAR_NAMES, MODEL_LABELS, LOADERS, ROLLOUT_FNS,
    extract_dynamic, apply_skip, extract_global_params,
)

VAR_LABELS = {
    'density':      r'$\rho$',
    'x_momentum':   r'$\rho u$',
    'total_energy':  r'$E$',
}


# ---------------------------------------------------------------------------
# PATH PARSER (handles spaces)
# ---------------------------------------------------------------------------

def parse_model_specs(raw_specs):
    model_specs = {}
    i = 0
    while i < len(raw_specs):
        token = raw_specs[i]
        colon_idx = token.find(':')
        if colon_idx > 0 and token[:colon_idx] in LOADERS:
            mtype = token[:colon_idx]
            path_parts = [token[colon_idx + 1:]]
            j = i + 1
            while j < len(raw_specs):
                nc = raw_specs[j].find(':')
                if nc > 0 and raw_specs[j][:nc] in LOADERS:
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
# SINGLE-VARIABLE FIELD FIGURE
# ===========================================================================

def make_field_figure(gt_list, model_preds, model_order, timesteps,
                      var_idx, var_name, output_path,
                      sim_name='', dpi=300, cmap='RdBu_r'):
    """
    One figure for a single field variable.

    Columns: 3 timesteps
    Rows:    GT + each model
    Color range: from GT only (global across all shown timesteps)
    """
    n_times = len(timesteps)
    n_rows = 1 + len(model_order)

    n_nodes = gt_list[0].shape[0]
    gs_dim = int(np.sqrt(n_nodes))
    assert gs_dim ** 2 == n_nodes

    # Color range from GT only
    gt_vals = [gt_list[t][:, var_idx] for t in timesteps]
    vmin = min(v.min() for v in gt_vals)
    vmax = max(v.max() for v in gt_vals)
    pad = max((vmax - vmin) * 0.02, 1e-8)
    vmin -= pad
    vmax += pad

    # Layout
    cell_w, cell_h = 2.0, 1.9
    row_label_w = 1.3
    cbar_w = 0.25
    header_h = 0.6

    fig_w = row_label_w + n_times * cell_w + cbar_w + 0.4
    fig_h = header_h + n_rows * cell_h + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # Normalized coordinates
    x0 = row_label_w / fig_w
    y_top = 1.0 - header_h / fig_h
    cw = cell_w / fig_w
    ch = cell_h / fig_h

    row_labels = ['Ground Truth'] + [MODEL_LABELS.get(m, m) for m in model_order]

    for row in range(n_rows):
        data_src = gt_list if row == 0 else model_preds.get(model_order[row - 1], [])

        for ti, t in enumerate(timesteps):
            x = x0 + ti * cw
            y = y_top - (row + 1) * ch

            ax = fig.add_axes([x, y, cw * 0.93, ch * 0.90])

            if t < len(data_src):
                field = data_src[t][:, var_idx].reshape(gs_dim, gs_dim)
            else:
                field = np.full((gs_dim, gs_dim), np.nan)

            im = ax.imshow(field, cmap=cmap, aspect='equal',
                           vmin=vmin, vmax=vmax,
                           origin='lower', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])

            # Row label on first column
            if ti == 0:
                ax.set_ylabel(row_labels[row], fontsize=9, fontweight='bold',
                              rotation=90, labelpad=8)

            # Column header on first row
            if row == 0:
                ax.set_title(f't = {t}', fontsize=10, fontweight='bold', pad=6)

    # Colorbar — full height on right
    cb_x = x0 + n_times * cw + 0.01
    cb_y = y_top - n_rows * ch
    cb_h = n_rows * ch * 0.90
    cbar_ax = fig.add_axes([cb_x, cb_y, 0.015, cb_h])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7)
    var_label = VAR_LABELS.get(var_name, var_name)
    cbar.set_label(var_label, fontsize=10, labelpad=4)

    # Title
    fig.text(0.5, 0.97, f'{var_label}  —  {sim_name}',
             fontsize=12, fontweight='bold', ha='center', va='top')

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# ===========================================================================
# SINGLE-VARIABLE ERROR FIGURE
# ===========================================================================

def make_error_figure(gt_list, model_preds, model_order, timesteps,
                      var_idx, var_name, output_path,
                      sim_name='', dpi=300):
    """Absolute error figure for one variable. No GT row."""
    n_times = len(timesteps)
    n_rows = len(model_order)

    n_nodes = gt_list[0].shape[0]
    gs_dim = int(np.sqrt(n_nodes))

    # Error range (99th percentile across all models)
    emax = 0
    for m in model_order:
        if m in model_preds:
            for t in timesteps:
                if t < len(model_preds[m]):
                    err = np.abs(model_preds[m][t][:, var_idx] - gt_list[t][:, var_idx])
                    emax = max(emax, float(np.percentile(err, 99)))
    emax = max(emax, 1e-8)

    cell_w, cell_h = 2.0, 1.9
    row_label_w = 1.3
    header_h = 0.6

    fig_w = row_label_w + n_times * cell_w + 0.65
    fig_h = header_h + n_rows * cell_h + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    x0 = row_label_w / fig_w
    y_top = 1.0 - header_h / fig_h
    cw = cell_w / fig_w
    ch = cell_h / fig_h

    for row, mname in enumerate(model_order):
        preds = model_preds.get(mname, [])
        label = MODEL_LABELS.get(mname, mname)

        for ti, t in enumerate(timesteps):
            x = x0 + ti * cw
            y = y_top - (row + 1) * ch

            ax = fig.add_axes([x, y, cw * 0.93, ch * 0.90])

            if t < len(preds):
                err = np.abs(preds[t][:, var_idx] - gt_list[t][:, var_idx]).reshape(gs_dim, gs_dim)
            else:
                err = np.full((gs_dim, gs_dim), np.nan)

            im = ax.imshow(err, cmap='hot_r', aspect='equal',
                           vmin=0, vmax=emax,
                           origin='lower', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])

            if ti == 0:
                ax.set_ylabel(label, fontsize=9, fontweight='bold',
                              rotation=90, labelpad=8)
            if row == 0:
                ax.set_title(f't = {t}', fontsize=10, fontweight='bold', pad=6)

    # Colorbar
    var_label = VAR_LABELS.get(var_name, var_name)
    cb_x = x0 + n_times * cw + 0.01
    cb_y = y_top - n_rows * ch
    cb_h = n_rows * ch * 0.90
    cbar_ax = fig.add_axes([cb_x, cb_y, 0.015, cb_h])
    norm = mcolors.Normalize(vmin=0, vmax=emax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='hot_r')
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(f'|Δ{var_label}|', fontsize=10, labelpad=4)

    fig.text(0.5, 0.97, f'Absolute Error: {var_label}  —  {sim_name}',
             fontsize=12, fontweight='bold', ha='center', va='top')

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Shock Tube Paper Figures")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--sim_index", type=int, default=0)
    parser.add_argument("--rollout_steps", type=int, default=40)
    parser.add_argument("--timesteps", type=int, nargs='+', default=None,
                        help="Specific timesteps (default: 3 evenly spaced)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cmap", type=str, default='RdBu_r')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--error_fig", action='store_true',
                        help="Also generate absolute-error figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model_specs = parse_model_specs(args.models)
    model_order = list(model_specs.keys())

    # Load simulation
    test_dir = Path(args.test_dir)
    files = sorted(test_dir.glob("*.pt"))
    if args.sim_index >= len(files):
        print(f"Only {len(files)} files, index {args.sim_index} out of range")
        return

    sim_file = files[args.sim_index]
    sim_name = sim_file.stem
    print(f"Loading simulation: {sim_name}")
    sim = torch.load(sim_file, weights_only=False)
    if not isinstance(sim, list):
        print("Error: expected list of Data objects"); return

    sample_data = sim[0]
    if not hasattr(sample_data, 'pos') or sample_data.pos is None:
        sample_data.pos = sample_data.x[:, :NUM_STATIC]

    steps = min(args.rollout_steps, len(sim) - 1)
    n_nodes = sample_data.x.size(0)
    gs_dim = int(np.sqrt(n_nodes))
    print(f"  {len(sim)} timesteps, {n_nodes} nodes ({gs_dim}×{gs_dim}), rollout {steps} steps")

    # Ground truth
    gt_list = []
    for t in range(steps):
        if hasattr(sim[t], 'y') and sim[t].y is not None:
            gt_list.append(apply_skip(sim[t].y).cpu().numpy())
        else:
            gt_list.append(extract_dynamic(sim[t + 1].x).cpu().numpy())

    # Timesteps
    if args.timesteps:
        timesteps = [t for t in args.timesteps if t < steps]
    else:
        timesteps = [steps // 6, steps // 2, steps - 1]
    print(f"  Timesteps: {timesteps}")

    # Load models & rollout
    model_preds = {}
    for mtype, mpath in model_specs.items():
        try:
            print(f"\n  Loading {MODEL_LABELS.get(mtype, mtype)}...")
            model = LOADERS[mtype](mpath, sample_data, device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    {n_params:,} parameters")

            with torch.no_grad():
                preds = ROLLOUT_FNS[mtype](model, sim, steps, device)
            model_preds[mtype] = preds

            all_p = np.concatenate(preds)
            all_g = np.concatenate(gt_list[:len(preds)])
            rrmse = (np.sqrt(np.mean((all_p - all_g)**2))
                     / max(np.sqrt(np.mean(all_g**2)), 1e-12))
            print(f"    RRMSE: {rrmse:.4f}")

            del model; torch.cuda.empty_cache()
        except Exception as e:
            print(f"    ✗ {MODEL_LABELS.get(mtype, mtype)} failed: {e}")
            import traceback; traceback.print_exc()

    model_order = [m for m in model_order if m in model_preds]
    if not model_order:
        print("No models loaded!"); return

    # Generate one figure per variable
    print(f"\nGenerating figures...")
    for vi, vn in enumerate(VAR_NAMES):
        for ext in ['png', 'pdf']:
            fpath = output_dir / f'{vn}_{sim_name}.{ext}'
            make_field_figure(gt_list, model_preds, model_order, timesteps,
                              vi, vn, fpath,
                              sim_name=sim_name, dpi=args.dpi, cmap=args.cmap)

        if args.error_fig:
            for ext in ['png', 'pdf']:
                epath = output_dir / f'{vn}_error_{sim_name}.{ext}'
                make_error_figure(gt_list, model_preds, model_order, timesteps,
                                  vi, vn, epath,
                                  sim_name=sim_name, dpi=args.dpi)

    print(f"\n✓ Done! {len(VAR_NAMES)} field figures + {'error figures ' if args.error_fig else ''}in {output_dir}")


if __name__ == "__main__":
    main()