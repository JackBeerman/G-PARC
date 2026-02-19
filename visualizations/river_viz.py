"""
visualizations.river_viz
========================
Scalar field GIFs, timeseries plots, and scatter plots for river evaluators.
Supports both PolyCollection (HEC-RAS mesh) and scatter rendering.
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from sklearn.metrics import r2_score

from visualizations.mesh_io import get_node_positions, load_mesh_for_sim, HAS_H5PY

__all__ = [
    'VAR_NAMES', 'VAR_CMAPS', 'VAR_UNITS_NORM', 'VAR_UNITS_PHYS',
    'create_scalar_field_gif', 'create_timeseries_plot',
    'create_scatter_plots', 'create_river_visualizations',
]

VAR_NAMES = ['Depth', 'Volume', 'Vel_X', 'Vel_Y']
VAR_CMAPS = ['Blues', 'Greens', 'RdBu_r', 'RdBu_r']
VAR_UNITS_NORM = ['(norm)', '(norm)', '(norm)', '(norm)']
VAR_UNITS_PHYS = ['m', 'm³', 'm/s', 'm/s']


def create_scalar_field_gif(pos, seq_pred, seq_targ, var_idx, var_name,
                             case_name, output_dir, model_name='Prediction',
                             fps=5, frame_skip=1, polys=None, extrema=None,
                             eval_mode='rollout'):
    """
    Side-by-side GIF: Target | Prediction | |Error| for one scalar variable.
    Uses PolyCollection if mesh polygons available, falls back to scatter.
    """
    max_steps = min(len(seq_pred), len(seq_targ))
    frames = list(range(0, max_steps, frame_skip))

    unit = VAR_UNITS_PHYS[var_idx] if extrema is not None else VAR_UNITS_NORM[var_idx]
    label = f'{var_name} ({unit})'

    # Global color range
    all_vals = np.concatenate([
        np.concatenate([seq_targ[t][:, var_idx] for t in range(max_steps)]),
        np.concatenate([seq_pred[t][:, var_idx] for t in range(max_steps)]),
    ])
    vmin, vmax = all_vals.min(), all_vals.max()
    if vmin == vmax:
        vmax = vmin + 1e-6

    error_vals = np.concatenate([
        np.abs(seq_targ[t][:, var_idx] - seq_pred[t][:, var_idx])
        for t in range(max_steps)
    ])
    err_max = max(np.percentile(error_vals, 99), 1e-6)

    cmap_name = VAR_CMAPS[var_idx] if var_idx < len(VAR_CMAPS) else 'viridis'
    use_polys = polys is not None and len(polys) > 0

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(right=0.88, wspace=0.05)
    cax1 = fig.add_axes([0.90, 0.55, 0.015, 0.35])
    cax2 = fig.add_axes([0.90, 0.10, 0.015, 0.35])

    norm_val = Normalize(vmin=vmin, vmax=vmax)
    norm_err = Normalize(vmin=0, vmax=err_max)

    sm_val = cm.ScalarMappable(cmap=cmap_name, norm=norm_val)
    sm_val.set_array([])
    fig.colorbar(sm_val, cax=cax1).set_label(label, fontsize=9)

    sm_err = cm.ScalarMappable(cmap='hot', norm=norm_err)
    sm_err.set_array([])
    fig.colorbar(sm_err, cax=cax2).set_label('|Error|', fontsize=9)

    x, y = pos[:, 0], pos[:, 1]

    def _render(ax, values, cmap, norm_obj):
        if use_polys:
            n_cells = min(len(polys), len(values))
            cmap_obj = plt.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
            colors = cmap_obj(norm_obj(values[:n_cells]))
            pc = PolyCollection(polys[:n_cells], facecolors=colors,
                                edgecolors='none', linewidths=0)
            ax.add_collection(pc)
            ax.autoscale_view()
        else:
            ax.scatter(x, y, c=values, cmap=cmap, s=1,
                       vmin=norm_obj.vmin, vmax=norm_obj.vmax)
        ax.set_aspect('equal')
        ax.axis('off')

    def animate(frame_idx):
        t = frames[frame_idx]
        for ax in axes:
            ax.clear()
        targ_vals = seq_targ[t][:, var_idx]
        pred_vals = seq_pred[t][:, var_idx]
        err_vals = np.abs(targ_vals - pred_vals)
        _render(axes[0], targ_vals, cmap_name, norm_val)
        axes[0].set_title(f'Target (t={t})', fontsize=11)
        _render(axes[1], pred_vals, cmap_name, norm_val)
        axes[1].set_title(f'{model_name} (t={t})', fontsize=11)
        _render(axes[2], err_vals, 'hot', norm_err)
        axes[2].set_title(f'|Error| (t={t})', fontsize=11)
        fig.suptitle(f'{var_name}: {case_name}', fontsize=13)
        return axes.tolist()

    anim = FuncAnimation(fig, animate, frames=len(frames),
                         interval=1000 // fps, blit=False)
    out_path = Path(output_dir) / f'{eval_mode}_{var_name.lower()}_{case_name}.gif'
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def create_timeseries_plot(seq_pred, seq_targ, case_name, output_dir,
                            var_names=None, eval_mode='rollout'):
    """Per-variable RMSE over time plot."""
    max_steps = min(len(seq_pred), len(seq_targ))
    n_vars = seq_pred[0].shape[1]
    if var_names is None:
        var_names = VAR_NAMES

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for v in range(min(n_vars, 4)):
        rmse_t = [np.sqrt(np.mean((seq_targ[t][:, v] - seq_pred[t][:, v]) ** 2))
                  for t in range(max_steps)]
        axes[v].plot(range(max_steps), rmse_t, 'b-', linewidth=1.5)
        axes[v].set_xlabel('Timestep')
        axes[v].set_ylabel('RMSE')
        axes[v].set_title(var_names[v] if v < len(var_names) else f'Var {v}')
        axes[v].grid(alpha=0.3)

    fig.suptitle(f'Per-Variable RMSE Over Time: {case_name}', fontsize=14)
    fig.tight_layout()
    out_path = Path(output_dir) / f'{eval_mode}_timeseries_{case_name}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def create_scatter_plots(seq_pred, seq_targ, case_name, output_dir,
                          sample_steps=5, var_names=None, eval_mode='rollout'):
    """Pred vs target scatter plots at sampled timesteps."""
    max_steps = min(len(seq_pred), len(seq_targ))
    step_indices = np.linspace(0, max_steps - 1, sample_steps, dtype=int)
    n_vars = seq_pred[0].shape[1]
    if var_names is None:
        var_names = VAR_NAMES

    fig, axes = plt.subplots(n_vars, sample_steps,
                             figsize=(4 * sample_steps, 4 * n_vars))
    if n_vars == 1:
        axes = axes[np.newaxis, :]
    if sample_steps == 1:
        axes = axes[:, np.newaxis]

    for vi in range(n_vars):
        for si, t in enumerate(step_indices):
            ax = axes[vi, si]
            p = seq_pred[t][:, vi]
            tg = seq_targ[t][:, vi]
            ax.scatter(tg, p, s=1, alpha=0.3)
            lims = [min(tg.min(), p.min()), max(tg.max(), p.max())]
            ax.plot(lims, lims, 'r--', linewidth=1)
            ax.set_aspect('equal')
            ax.set_xlabel('Target')
            ax.set_ylabel('Prediction')
            r2 = r2_score(tg, p) if len(tg) > 1 else 0
            name = var_names[vi] if vi < len(var_names) else f'Var {vi}'
            ax.set_title(f'{name} t={t}\nR²={r2:.4f}', fontsize=9)
            ax.grid(alpha=0.3)

    fig.suptitle(f'Prediction vs Target: {case_name}', fontsize=14)
    fig.tight_layout()
    out_path = Path(output_dir) / f'{eval_mode}_scatter_{case_name}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def create_river_visualizations(simulation, seq_pred, seq_targ, case_name,
                                 output_dir, model_name='Prediction',
                                 fps=5, frame_skip=1, eval_mode='rollout',
                                 hec_ras_dir=None, extrema=None):
    """
    Create all river visualizations for a single simulation:
    per-variable GIFs + timeseries + scatter plots.
    """
    pos = get_node_positions(simulation)
    n_vars = min(seq_pred[0].shape[1], len(VAR_NAMES))

    polys = None
    if hec_ras_dir is not None and HAS_H5PY:
        polys = load_mesh_for_sim(case_name, hec_ras_dir)

    render_mode = "PolyCollection" if polys else "scatter"
    print(f"\n  Creating visualizations for {case_name} ({eval_mode}, {render_mode})...")

    for v in range(n_vars):
        path = create_scalar_field_gif(
            pos, seq_pred, seq_targ, v, VAR_NAMES[v],
            case_name, output_dir, model_name=model_name,
            fps=fps, frame_skip=frame_skip, polys=polys, extrema=extrema,
            eval_mode=eval_mode,
        )
        print(f"    ✓ {path.name}")

    path = create_timeseries_plot(seq_pred, seq_targ, case_name, output_dir,
                                   eval_mode=eval_mode)
    print(f"    ✓ {path.name}")

    path = create_scatter_plots(seq_pred, seq_targ, case_name, output_dir,
                                 eval_mode=eval_mode)
    print(f"    ✓ {path.name}")