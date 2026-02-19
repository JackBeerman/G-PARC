"""
visualizations.shocktube_viz
=============================
Grid-based (imshow) GIFs and static plots for shock tube evaluators.
Data is on a structured square grid — rendered as 2D images.

Provides:
  - Per-variable comparison GIF: Target | Prediction | |Error|
  - All-variables combined GIF
  - Error evolution GIF
  - Rollout error growth plot (MSE vs timestep)
  - Prediction-vs-target scatter plot
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.colors import Normalize
from sklearn.metrics import r2_score

__all__ = [
    'SHOCKTUBE_VAR_NAMES',
    'create_variable_comparison_gif',
    'create_all_variables_gif',
    'create_error_evolution_gif',
    'create_shocktube_visualizations',
    'plot_rollout_error_growth',
    'plot_prediction_scatter',
]

SHOCKTUBE_VAR_NAMES = ['density', 'x_momentum', 'total_energy']


# ─── Per-variable comparison GIF ─────────────────────────────────────────

def create_variable_comparison_gif(frames, seq_pred, seq_targ, var_idx, var_name,
                                    grid_size, vmin, vmax, case_name, output_dir,
                                    model_name='Prediction', fps=4,
                                    eval_mode='rollout', global_params_str=None):
    """Side-by-side Target | Prediction | |Error| GIF for one variable, with colorbars."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.subplots_adjust(right=0.95, wspace=0.35)

    mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps.get_cmap('coolwarm')

    err_max = max(np.abs(seq_targ[f][:, var_idx] - seq_pred[f][:, var_idx]).max()
                  for f in frames)
    err_norm = Normalize(vmin=0, vmax=max(err_max, 1e-12))
    err_cmap = plt.colormaps.get_cmap('hot')

    # Initial images for colorbars
    g = grid_size
    targ_2d = seq_targ[frames[0]][:, var_idx].reshape(g, g)
    pred_2d = seq_pred[frames[0]][:, var_idx].reshape(g, g)
    err_2d = np.abs(targ_2d - pred_2d)

    im0 = axes[0].imshow(targ_2d, cmap=cmap, norm=norm, aspect='auto', origin='lower')
    im1 = axes[1].imshow(pred_2d, cmap=cmap, norm=norm, aspect='auto', origin='lower')
    im2 = axes[2].imshow(err_2d, cmap=err_cmap, norm=err_norm, aspect='auto', origin='lower')

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Build title
    title_base = f'{var_name} ({mode_label}): {case_name}'
    if global_params_str:
        title_base += f'\n{global_params_str}'

    def animate(frame_idx):
        frame = frames[frame_idx]
        targ_2d = seq_targ[frame][:, var_idx].reshape(g, g)
        pred_2d = seq_pred[frame][:, var_idx].reshape(g, g)
        err_2d = np.abs(targ_2d - pred_2d)
        im0.set_array(targ_2d)
        im1.set_array(pred_2d)
        im2.set_array(err_2d)
        axes[0].set_title(f'Target (t={frame})', fontsize=12)
        axes[1].set_title(f'{model_name} (t={frame})', fontsize=12)
        axes[2].set_title(f'|Error| (t={frame})', fontsize=12)
        fig.suptitle(title_base, fontsize=13, fontweight='bold')
        return [im0, im1, im2]

    plt.tight_layout(rect=[0, 0, 1, 0.92 if global_params_str else 0.95])
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps, blit=False)
    out_path = Path(output_dir) / f'{eval_mode}_{var_name}_{case_name}.gif'
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


# ─── All-variables combined GIF ──────────────────────────────────────────

def create_all_variables_gif(frames, seq_pred, seq_targ, var_names, grid_size,
                              vranges, case_name, output_dir, model_name='Prediction',
                              fps=4, eval_mode='rollout', global_params_str=None):
    """Combined GIF: 2 rows (target / prediction) × n_vars columns, with colorbars."""
    n_vars = len(var_names)
    fig, axes = plt.subplots(2, n_vars, figsize=(6 * n_vars + 2, 11))
    if n_vars == 1:
        axes = axes.reshape(2, 1)
    mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'
    cmap = plt.colormaps.get_cmap('coolwarm')
    norms = [Normalize(vmin=vr[0], vmax=vr[1]) for vr in vranges]

    g = grid_size
    ims = []
    for vi, vn in enumerate(var_names):
        targ_2d = seq_targ[frames[0]][:, vi].reshape(g, g)
        pred_2d = seq_pred[frames[0]][:, vi].reshape(g, g)
        im_t = axes[0, vi].imshow(targ_2d, cmap=cmap, norm=norms[vi], aspect='auto', origin='lower')
        im_p = axes[1, vi].imshow(pred_2d, cmap=cmap, norm=norms[vi], aspect='auto', origin='lower')
        axes[0, vi].set_xticks([]); axes[0, vi].set_yticks([])
        axes[1, vi].set_xticks([]); axes[1, vi].set_yticks([])
        plt.colorbar(im_t, ax=axes[0, vi], fraction=0.046, pad=0.04)
        plt.colorbar(im_p, ax=axes[1, vi], fraction=0.046, pad=0.04)
        ims.append((im_t, im_p))

    title_base = f'All Variables ({mode_label}): {case_name}'
    if global_params_str:
        title_base += f'\n{global_params_str}'

    def animate(frame_idx):
        frame = frames[frame_idx]
        for vi, vn in enumerate(var_names):
            targ_2d = seq_targ[frame][:, vi].reshape(g, g)
            pred_2d = seq_pred[frame][:, vi].reshape(g, g)
            ims[vi][0].set_array(targ_2d)
            ims[vi][1].set_array(pred_2d)
            axes[0, vi].set_title(f'{vn} Target (t={frame})', fontsize=10)
            axes[1, vi].set_title(f'{vn} Pred (t={frame})', fontsize=10)
        fig.suptitle(f'{title_base} (t={frame})', fontsize=13, fontweight='bold')
        return [im for pair in ims for im in pair]

    plt.tight_layout(rect=[0, 0, 1, 0.91 if global_params_str else 0.94])
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps, blit=False)
    out_path = Path(output_dir) / f'{eval_mode}_all_vars_{case_name}.gif'
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


# ─── Error evolution GIF ─────────────────────────────────────────────────

def create_error_evolution_gif(frames, seq_pred, seq_targ, var_names, grid_size,
                                case_name, output_dir, fps=4, eval_mode='rollout',
                                global_params_str=None):
    """GIF showing per-variable |error| evolving over time, with colorbars."""
    n_vars = len(var_names)
    fig, axes = plt.subplots(1, n_vars, figsize=(6 * n_vars + 2, 5.5))
    if n_vars == 1:
        axes = [axes]
    mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'
    cmap = plt.colormaps.get_cmap('hot')

    g = grid_size
    max_errors = [0.0] * n_vars
    for f in frames:
        for vi in range(n_vars):
            err = np.abs(seq_targ[f][:, vi] - seq_pred[f][:, vi])
            max_errors[vi] = max(max_errors[vi], err.max())
    norms = [Normalize(vmin=0, vmax=max(me, 1e-12)) for me in max_errors]

    ims = []
    for vi, vn in enumerate(var_names):
        err_2d = np.abs(seq_targ[frames[0]][:, vi] - seq_pred[frames[0]][:, vi]).reshape(g, g)
        im = axes[vi].imshow(err_2d, cmap=cmap, norm=norms[vi], aspect='auto', origin='lower')
        axes[vi].set_xticks([]); axes[vi].set_yticks([])
        plt.colorbar(im, ax=axes[vi], fraction=0.046, pad=0.04)
        ims.append(im)

    title_base = f'Error Evolution ({mode_label}): {case_name}'
    if global_params_str:
        title_base += f'\n{global_params_str}'

    def animate(frame_idx):
        frame = frames[frame_idx]
        for vi, vn in enumerate(var_names):
            err_2d = np.abs(seq_targ[frame][:, vi] - seq_pred[frame][:, vi]).reshape(g, g)
            ims[vi].set_array(err_2d)
            axes[vi].set_title(f'{vn} |Error| (t={frame})', fontsize=10)
        fig.suptitle(f'{title_base} (t={frame})', fontsize=13, fontweight='bold')
        return ims

    plt.tight_layout(rect=[0, 0, 1, 0.90 if global_params_str else 0.93])
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps, blit=False)
    out_path = Path(output_dir) / f'{eval_mode}_error_evolution_{case_name}.gif'
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


# ─── High-level convenience ──────────────────────────────────────────────

def create_shocktube_visualizations(seq_pred, seq_targ, sim_idx, output_dir,
                                     var_names=None, model_name='Prediction',
                                     fps=4, frame_skip=1, eval_mode='rollout',
                                     metadata=None):
    """
    Create all shock tube GIFs for one simulation.
    Expects seq_pred / seq_targ as lists of [N, D] arrays.
    N must be a perfect square for grid reshape.

    metadata: optional dict with 'pressure', 'density', 'delta_t' keys
              for global parameter display in titles.
    """
    if var_names is None:
        var_names = SHOCKTUBE_VAR_NAMES

    max_steps = min(len(seq_pred), len(seq_targ))
    if max_steps < 2:
        return

    n_nodes = seq_pred[0].shape[0]
    grid_size = int(np.sqrt(n_nodes))
    if grid_size * grid_size != n_nodes:
        print(f"  ⚠️  Non-square grid ({n_nodes} nodes) for sim_{sim_idx}, skipping GIFs")
        return

    frames = list(range(0, max_steps, frame_skip))
    case_name = f'sim_{sim_idx}'

    # Build global params string for titles
    global_params_str = None
    if metadata:
        parts = []
        if 'pressure' in metadata and metadata['pressure']:
            parts.append(f"P={metadata['pressure']:.1f}")
        if 'density' in metadata and metadata['density']:
            parts.append(f"ρ={metadata['density']:.4f}")
        if 'delta_t' in metadata and metadata['delta_t']:
            parts.append(f"Δt={metadata['delta_t']:.6f}")
        if parts:
            global_params_str = '  |  '.join(parts)

    # Per-variable value ranges
    vranges = []
    for vi in range(len(var_names)):
        vals = np.concatenate([
            np.concatenate([seq_targ[f][:, vi] for f in frames]),
            np.concatenate([seq_pred[f][:, vi] for f in frames]),
        ])
        vranges.append((vals.min(), vals.max()))

    print(f"\n  Creating GIFs for {case_name} ({eval_mode}), {max_steps} steps, grid {grid_size}²")
    if global_params_str:
        print(f"    Global params: {global_params_str}")

    for vi, vn in enumerate(var_names):
        path = create_variable_comparison_gif(
            frames, seq_pred, seq_targ, vi, vn, grid_size,
            vranges[vi][0], vranges[vi][1],
            case_name, output_dir, model_name=model_name,
            fps=fps, eval_mode=eval_mode,
            global_params_str=global_params_str,
        )
        print(f"    ✓ {path.name}")

    path = create_all_variables_gif(
        frames, seq_pred, seq_targ, var_names, grid_size,
        vranges, case_name, output_dir, model_name=model_name,
        fps=fps, eval_mode=eval_mode,
        global_params_str=global_params_str,
    )
    print(f"    ✓ {path.name}")

    path = create_error_evolution_gif(
        frames, seq_pred, seq_targ, var_names, grid_size,
        case_name, output_dir, fps=fps, eval_mode=eval_mode,
        global_params_str=global_params_str,
    )
    print(f"    ✓ {path.name}")


# ─── Static plots ────────────────────────────────────────────────────────

def plot_rollout_error_growth(preds_phys, targs_phys, var_names=None,
                               output_path=None, figsize=(16, 10)):
    """
    Plot MSE growth across rollout timesteps.
    preds_phys / targs_phys: list of lists of [N, D] arrays (sim × timestep).
    """
    if var_names is None:
        var_names = SHOCKTUBE_VAR_NAMES
    if not preds_phys:
        return None

    max_steps = max(len(sp) for sp in preds_phys)
    var_mse = {vn: [[] for _ in range(max_steps)] for vn in var_names}
    overall_mse = [[] for _ in range(max_steps)]

    for sp, st in zip(preds_phys, targs_phys):
        for t in range(len(sp)):
            p, tg = sp[t], st[t]
            overall_mse[t].append(np.mean((p - tg) ** 2))
            for vi, vn in enumerate(var_names):
                if vi < p.shape[1]:
                    var_mse[vn][t].append(np.mean((p[:, vi] - tg[:, vi]) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Rollout Error Growth Over Timesteps', fontsize=14, fontweight='bold')

    # Overall
    ax = axes[0]
    steps = [t for t in range(max_steps) if overall_mse[t]]
    means = [np.mean(overall_mse[t]) for t in steps]
    stds = [np.std(overall_mse[t]) for t in steps]
    ax.errorbar(steps, means, yerr=stds, fmt='o-', capsize=4, color='steelblue')
    ax.set_xlabel('Rollout Step'); ax.set_ylabel('MSE')
    ax.set_title('Overall MSE vs Rollout Step')
    ax.set_yscale('log'); ax.grid(alpha=0.3)

    # Per-variable
    ax = axes[1]
    colors = plt.cm.Set1(np.linspace(0, 0.6, len(var_names)))
    for vi, vn in enumerate(var_names):
        steps_v = [t for t in range(max_steps) if var_mse[vn][t]]
        means_v = [np.mean(var_mse[vn][t]) for t in steps_v]
        ax.plot(steps_v, means_v, 'o-', label=vn, color=colors[vi], ms=5)
    ax.set_xlabel('Rollout Step'); ax.set_ylabel('MSE')
    ax.set_title('Per-Variable MSE vs Rollout Step')
    ax.set_yscale('log'); ax.grid(alpha=0.3); ax.legend()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_prediction_scatter(preds_phys, targs_phys, var_names=None,
                             output_path=None, figsize=(18, 5), n_sample=5000):
    """Prediction vs target scatter for each variable (physical units)."""
    if var_names is None:
        var_names = SHOCKTUBE_VAR_NAMES
    if not preds_phys:
        return None

    all_p = np.concatenate([p for sp in preds_phys for p in sp], axis=0)
    all_t = np.concatenate([t for st in targs_phys for t in st], axis=0)

    n_vars = min(all_p.shape[1], len(var_names))
    fig, axes = plt.subplots(1, n_vars, figsize=figsize)
    if n_vars == 1:
        axes = [axes]
    fig.suptitle('Prediction vs Target (Physical Units)', fontsize=14, fontweight='bold')

    n_sample = min(n_sample, all_p.shape[0])
    idx = np.random.choice(all_p.shape[0], n_sample, replace=False)

    for vi in range(n_vars):
        ax = axes[vi]
        pv, tv = all_p[idx, vi], all_t[idx, vi]
        ax.scatter(tv, pv, s=1, alpha=0.3)
        lo, hi = min(tv.min(), pv.min()), max(tv.max(), pv.max())
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.7)
        r2 = r2_score(tv, pv) if len(tv) > 1 else 0
        ax.set_title(f'{var_names[vi]}  (R² = {r2:.4f})')
        ax.set_xlabel('Target'); ax.set_ylabel('Prediction')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig