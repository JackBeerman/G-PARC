#!/usr/bin/env python3
"""
Shock Tube Paper Figures — One per field variable
===================================================
Generates publication-quality figures, one per field variable.

Each figure:
    Columns: t_early | t_mid | t_late  (3 timesteps)
    Rows:    GT | G-PARC | G-PARC (w/o MLS) | MeshGraphKAN   [default]

Color range is set from GROUND TRUTH only (global min/max across timesteps).
Each row gets its own colorbar.

Usage:
    python paper_figure.py \
        --test_dir /path/to/test_cases_normalized \
        --models gparcv1:/path gparcv2:/path mgkan:/path mgnet:/path \
        --output_dir /scratch/.../figures \
        --sim_index 0 --rollout_steps 40 --error_fig

    # Override which models appear (and in what order):
    python paper_figure.py ... --paper_models gparcv2 gparcv1 mgkan mgnet
"""

import argparse, sys, os, json, re
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
    extract_global_params_from_data,
)

VAR_LABELS = {
    'density':      r'$\rho$',
    'x_momentum':   r'$\rho u$',
    'total_energy':  r'$E$',
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
    'gsage':      'GraphSAGE',
    'graphsage':  'GraphSAGE',
}

# Default model subset for paper figures (GT row is always included)
DEFAULT_PAPER_MODELS = ['gparcv2', 'gparcv1', 'mgkan']

# Denormalization: physical = normalized * (max - min) + min
# From variable_statistics.json global_statistics
DENORM = {
    'density':      {'min': 0.062139, 'max': 2.007674, 'unit': r'kg/m$^3$'},
    'x_momentum':   {'min': -2.600878, 'max': 235.423531, 'unit': r'kg/(m$^2 \cdot$s)'},
    'total_energy': {'min': 12399.199670, 'max': 424145.808785, 'unit': r'J/m$^3$'},
}

def denormalize(arr, var_name):
    """Convert normalized [0,1] array to physical units."""
    if var_name in DENORM:
        d = DENORM[var_name]
        return arr * (d['max'] - d['min']) + d['min']
    return arr


# ---------------------------------------------------------------------------
# TITLE / TIME FORMATTING
# ---------------------------------------------------------------------------

def format_physical_time(step, delta_t):
    """Convert a timestep index to a physical time string."""
    if delta_t is None:
        return f't = {step}'
    phys_time = step * delta_t
    if phys_time < 1e-4:
        return f't = {phys_time:.2e} s'
    elif phys_time < 1.0:
        return f't = {phys_time:.4f} s'
    else:
        return f't = {phys_time:.2f} s'


def format_sim_title(data, sim_name):
    """
    Build a title from physical simulation parameters.

    Priority: parse from filename (always has physical values),
    then try data attributes as fallback.
    Filename format: p_L_143750_rho_L_0.5625_test_with_pos_normalized
    """
    pressure = None
    density = None
    dt = None

    # Parse from filename first (most reliable — always physical values)
    m_p = re.search(r'p_L_(\d+\.?\d*)', sim_name)
    if m_p:
        pressure = float(m_p.group(1))

    m_rho = re.search(r'rho_L_(\d+\.?\d*)', sim_name)
    if m_rho:
        density = float(m_rho.group(1))

    # delta_t from data attributes (physical value)
    for attr in ['delta_t_physical', 'delta_t_numeric']:
        if hasattr(data, attr):
            v = getattr(data, attr)
            dt = v.item() if torch.is_tensor(v) else float(v)
            break

    # Fallback: try data attributes for pressure/density if not in filename
    if pressure is None:
        for attr in ['pressure_numeric', 'pressure_physical']:
            if hasattr(data, attr):
                v = getattr(data, attr)
                pressure = v.item() if torch.is_tensor(v) else float(v)
                break

    if density is None:
        for attr in ['density_numeric', 'density_physical']:
            if hasattr(data, attr):
                v = getattr(data, attr)
                density = v.item() if torch.is_tensor(v) else float(v)
                break

    parts = []
    if pressure is not None:
        parts.append(f'Pressure = {pressure:g} Pa')
    if density is not None:
        parts.append(f'Density = {density:g} kg/m$^3$')
    if dt is not None:
        if dt < 1e-3:
            parts.append(f'$\\Delta t$ = {dt:.2e} s')
        else:
            parts.append(f'$\\Delta t$ = {dt:g} s')

    if parts:
        return ',  '.join(parts)
    return sim_name


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
                print(f"Warning: Checkpoint not found: {mpath}")
            i = j
        else:
            i += 1
    return model_specs


# ===========================================================================
# SINGLE-VARIABLE FIELD FIGURE (per-row colorbars)
# ===========================================================================

def make_field_figure(gt_list, model_preds, model_order, timesteps,
                      var_idx, var_name, output_path,
                      sim_title='', dpi=300, cmap='RdBu_r', delta_t=None):
    """
    One figure for a single field variable.

    Columns: 3 timesteps
    Rows:    GT + each model
    Color range: from GT only (global across all shown timesteps)
    Each row gets its own compact colorbar.
    """
    n_times = len(timesteps)
    n_rows = 1 + len(model_order)

    n_nodes = gt_list[0].shape[0]
    gs_dim = int(np.sqrt(n_nodes))
    assert gs_dim ** 2 == n_nodes

    # Denormalize GT and predictions to physical units
    gt_phys = [denormalize(g, var_name) for g in gt_list]
    model_preds_phys = {}
    for m in model_order:
        if m in model_preds:
            model_preds_phys[m] = [denormalize(p, var_name) for p in model_preds[m]]

    # Color range from GT only (physical units)
    gt_vals = [gt_phys[t][:, var_idx] for t in timesteps]
    vmin = min(v.min() for v in gt_vals)
    vmax = max(v.max() for v in gt_vals)
    pad = max((vmax - vmin) * 0.02, 1e-8)
    vmin -= pad
    vmax += pad

    # Unit string for colorbar
    unit_str = DENORM.get(var_name, {}).get('unit', '')
    var_label = VAR_LABELS.get(var_name, var_name)

    # Layout
    cell_w, cell_h = 2.0, 1.9
    row_label_w = 1.3
    cbar_w = 0.45
    header_h = 0.6

    fig_w = row_label_w + n_times * cell_w + cbar_w + 0.3
    fig_h = header_h + n_rows * cell_h + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # Normalized coordinates
    x0 = row_label_w / fig_w
    y_top = 1.0 - header_h / fig_h
    cw = cell_w / fig_w
    ch = cell_h / fig_h
    cbar_norm_w = 0.015
    cbar_gap = 0.008

    row_labels = ['Ground Truth'] + [
        PAPER_MODEL_LABELS.get(m, MODEL_LABELS.get(m, m))
        for m in model_order
    ]

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for row in range(n_rows):
        data_src = gt_phys if row == 0 else model_preds_phys.get(model_order[row - 1], [])

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
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Row label on first column
            if ti == 0:
                ax.set_ylabel(row_labels[row], fontsize=9, fontweight='bold',
                              rotation=90, labelpad=8)

            # Column header on first row
            if row == 0:
                ax.set_title(format_physical_time(t, delta_t),
                             fontsize=10, fontweight='bold', pad=6)

        # Per-row colorbar
        cb_x = x0 + n_times * cw + cbar_gap
        cb_y = y_top - (row + 1) * ch + ch * 0.05
        cb_h = ch * 0.80

        cbar_ax = fig.add_axes([cb_x, cb_y, cbar_norm_w, cb_h])
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(var_label + ' (' + unit_str + ')', fontsize=7, labelpad=3)

    var_label = VAR_LABELS.get(var_name, var_name)
    fig.text(0.5, 0.97, f'{var_label}    {sim_title}',
             fontsize=11, fontweight='bold', ha='center', va='top')

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# ===========================================================================
# SINGLE-VARIABLE ERROR FIGURE (per-row colorbars)
# ===========================================================================

def make_error_figure(gt_list, model_preds, model_order, timesteps,
                      var_idx, var_name, output_path,
                      sim_title='', dpi=300, delta_t=None):
    """Absolute error figure for one variable. No GT row. Per-row colorbars."""
    n_times = len(timesteps)
    n_rows = len(model_order)

    n_nodes = gt_list[0].shape[0]
    gs_dim = int(np.sqrt(n_nodes))

    # Denormalize GT and predictions to physical units
    gt_phys = [denormalize(g, var_name) for g in gt_list]
    model_preds_phys = {}
    for m in model_order:
        if m in model_preds:
            model_preds_phys[m] = [denormalize(p, var_name) for p in model_preds[m]]

    # Unit string for colorbar
    unit_str = DENORM.get(var_name, {}).get('unit', '')
    var_label = VAR_LABELS.get(var_name, var_name)

    # Per-model error max (99th percentile) in physical units
    global_emax = 0
    for m in model_order:
        if m in model_preds_phys:
            for t in timesteps:
                if t < len(model_preds_phys[m]):
                    err = np.abs(model_preds_phys[m][t][:, var_idx] - gt_phys[t][:, var_idx])
                    global_emax = max(global_emax, float(np.percentile(err, 99)))
    global_emax = max(global_emax, 1e-8)

    cell_w, cell_h = 2.0, 1.9
    row_label_w = 1.3
    cbar_w = 0.45
    header_h = 0.6

    fig_w = row_label_w + n_times * cell_w + cbar_w + 0.3
    fig_h = header_h + n_rows * cell_h + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    x0 = row_label_w / fig_w
    y_top = 1.0 - header_h / fig_h
    cw = cell_w / fig_w
    ch = cell_h / fig_h
    cbar_norm_w = 0.015
    cbar_gap = 0.008

    norm = mcolors.Normalize(vmin=0, vmax=global_emax)

    for row, mname in enumerate(model_order):
        preds = model_preds_phys.get(mname, [])
        label = PAPER_MODEL_LABELS.get(mname, MODEL_LABELS.get(mname, mname))

        for ti, t in enumerate(timesteps):
            x = x0 + ti * cw
            y = y_top - (row + 1) * ch

            ax = fig.add_axes([x, y, cw * 0.93, ch * 0.90])

            if t < len(preds):
                err = np.abs(preds[t][:, var_idx] - gt_phys[t][:, var_idx]).reshape(gs_dim, gs_dim)
            else:
                err = np.full((gs_dim, gs_dim), np.nan)

            im = ax.imshow(err, cmap='hot_r', aspect='equal',
                           vmin=0, vmax=global_emax,
                           origin='lower', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if ti == 0:
                ax.set_ylabel(label, fontsize=9, fontweight='bold',
                              rotation=90, labelpad=8)
            if row == 0:
                ax.set_title(format_physical_time(t, delta_t),
                             fontsize=10, fontweight='bold', pad=6)

        # Per-row colorbar
        cb_x = x0 + n_times * cw + cbar_gap
        cb_y = y_top - (row + 1) * ch + ch * 0.05
        cb_h = ch * 0.80

        cbar_ax = fig.add_axes([cb_x, cb_y, cbar_norm_w, cb_h])
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='hot_r')
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(f'|error| ({unit_str})', fontsize=7, labelpad=3)

    var_label = VAR_LABELS.get(var_name, var_name)
    fig.text(0.5, 0.97, f'Absolute Error: {var_label}    {sim_title}',
             fontsize=11, fontweight='bold', ha='center', va='top')

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
    parser.add_argument("--paper_models", type=str, nargs='+', default=None,
                        help="Which models to include in figures and in what order "
                             "(default: gparcv2 gparcv1 mgkan). "
                             "Only models present in --models will appear.")
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

    # Determine which models to show (and in what order)
    paper_models = args.paper_models if args.paper_models else DEFAULT_PAPER_MODELS
    model_order = [m for m in paper_models if m in model_specs]
    skipped = [m for m in paper_models if m not in model_specs]
    if skipped:
        print(f"⚠ Requested but not found in --models: {skipped}")
    print(f"Figure model order: {[PAPER_MODEL_LABELS.get(m, m) for m in model_order]}")

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
    print(f"  {len(sim)} timesteps, {n_nodes} nodes ({gs_dim}x{gs_dim}), rollout {steps} steps")

    # Build simulation title from physical parameters
    sim_title = format_sim_title(sample_data, sim_name)
    print(f"  Title: {sim_title}")

    # Extract delta_t for physical time labels
    #sim_params = extract_global_params_from_data(sample_data)
    #delta_t = sim_params.get('delta_t', None)
    #if delta_t is not None and delta_t != 0:
    #    print(f"  dt = {delta_t} s")
    #else:
    #    delta_t = None
    #    print(f"  dt not found -- column headers will show step indices")
    # Extract delta_t from normalization metadata (physical value)
    norm_meta_path = Path(args.test_dir).parent / "normalization_metadata.json"
    delta_t = None
    if norm_meta_path.exists():
        with open(norm_meta_path) as f:
            norm_meta = json.load(f)
        case_key = re.sub(r'_test_with_pos_normalized$', '', sim_name)
        case_info = norm_meta.get('original_metadata', {}).get('case_info', {})
        if case_key in case_info:
            delta_t = case_info[case_key]['delta_t']
            print(f"  dt = {delta_t:.3e} s (from normalization metadata)")
        else:
            print(f"  ⚠ case '{case_key}' not found in normalization metadata")
    else:
        print(f"  ⚠ normalization_metadata.json not found at {norm_meta_path}")
    
    if delta_t is None:
        print(f"  dt not found -- column headers will show step indices")

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

    if delta_t is not None:
        phys_times = [format_physical_time(t, delta_t) for t in timesteps]
        print(f"  Timesteps: {timesteps} (physical: {phys_times})")
    else:
        print(f"  Timesteps: {timesteps}")

    # Load models & rollout (only those in model_order)
    model_preds = {}
    for mtype in model_order:
        mpath = model_specs[mtype]
        try:
            pname = PAPER_MODEL_LABELS.get(mtype, MODEL_LABELS.get(mtype, mtype))
            print(f"\n  Loading {pname}...")
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
            pname = PAPER_MODEL_LABELS.get(mtype, MODEL_LABELS.get(mtype, mtype))
            print(f"    Failed: {pname}: {e}")
            import traceback; traceback.print_exc()

    active_models = [m for m in model_order if m in model_preds]
    if not active_models:
        print("No models loaded!"); return

    # Generate one figure per variable
    print(f"\nGenerating figures...")
    for vi, vn in enumerate(VAR_NAMES):
        for ext in ['png', 'pdf']:
            fpath = output_dir / f'{vn}_{sim_name}.{ext}'
            make_field_figure(gt_list, model_preds, active_models, timesteps,
                              vi, vn, fpath,
                              sim_title=sim_title, dpi=args.dpi, cmap=args.cmap,
                              delta_t=delta_t)

        if args.error_fig:
            for ext in ['png', 'pdf']:
                epath = output_dir / f'{vn}_error_{sim_name}.{ext}'
                make_error_figure(gt_list, model_preds, active_models, timesteps,
                                  vi, vn, epath,
                                  sim_title=sim_title, dpi=args.dpi,
                                  delta_t=delta_t)

    print(f"\nDone! {len(VAR_NAMES)} field figures + {'error figures ' if args.error_fig else ''}in {output_dir}")


if __name__ == "__main__":
    main()