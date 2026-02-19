"""
visualizations.elasto_viz
=========================
Mesh deformation GIFs (reference, deformed, error), erosion plots,
and precomputation for elastoplastic evaluators.
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

from visualizations.mesh_io import (
    precompute_element_polygons, get_erosion_mask, get_valid_node_mask,
)

__all__ = [
    'render_mesh_fast', 'precompute_elasto_viz_data',
    'create_reference_gif', 'create_deformed_gif',
    'create_error_gif', 'create_erosion_plot',
    'create_elasto_visualizations',
]


def render_mesh_fast(ax, poly_verts, node_values, elements, eroded_mask,
                     vmin, vmax, cmap_obj, norm, show_eroded=False):
    """Fast element-wise PolyCollection rendering with erosion masking."""
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


def precompute_elasto_viz_data(simulation, seq_pred, seq_targ, elements):
    """Precompute global bounds, erosion masks, and camera extents."""
    max_steps = min(len(seq_pred), len(seq_targ), len(simulation))
    num_elements = len(elements)

    pos_ref = (simulation[0].pos.cpu().numpy()
               if hasattr(simulation[0], 'pos') and simulation[0].pos is not None
               else simulation[0].x[:, :2].cpu().numpy())
    poly_verts_ref = precompute_element_polygons(pos_ref, elements)

    erosion_masks, erosion_counts, valid_node_masks = [], [], []
    for t in range(max_steps):
        eroded_mask = get_erosion_mask(simulation[t], num_elements)
        erosion_masks.append(eroded_mask)
        erosion_counts.append(eroded_mask.sum())
        valid_node_masks.append(get_valid_node_mask(elements, eroded_mask))

    disp_max = error_max = 0
    Ux_min = Ux_max = Uy_min = Uy_max = 0
    x_ref, y_ref = pos_ref[:, 0], pos_ref[:, 1]
    def_x_min, def_x_max = x_ref.min(), x_ref.max()
    def_y_min, def_y_max = y_ref.min(), y_ref.max()

    for t in range(max_steps):
        valid_nodes = valid_node_masks[t]
        if valid_nodes.sum() == 0:
            continue
        U_targ, U_pred = seq_targ[t], seq_pred[t]
        for U in [U_targ, U_pred]:
            u_mag = np.sqrt(U[valid_nodes, 0] ** 2 + U[valid_nodes, 1] ** 2)
            disp_max = max(disp_max, u_mag.max())
        error_mag = np.sqrt(
            (U_targ[valid_nodes, 0] - U_pred[valid_nodes, 0]) ** 2 +
            (U_targ[valid_nodes, 1] - U_pred[valid_nodes, 1]) ** 2
        )
        error_max = max(error_max, error_mag.max())
        for U in [U_targ, U_pred]:
            Ux_min = min(Ux_min, U[valid_nodes, 0].min())
            Ux_max = max(Ux_max, U[valid_nodes, 0].max())
            Uy_min = min(Uy_min, U[valid_nodes, 1].min())
            Uy_max = max(Uy_max, U[valid_nodes, 1].max())
        eroded_mask = erosion_masks[t]
        valid_elements_t = elements[~eroded_mask]
        if len(valid_elements_t) > 0:
            valid_ni = np.unique(valid_elements_t.flatten())
            for U in [U_targ, U_pred]:
                x_def = x_ref[valid_ni] + U[valid_ni, 0]
                y_def = y_ref[valid_ni] + U[valid_ni, 1]
                def_x_min = min(def_x_min, x_def.min())
                def_x_max = max(def_x_max, x_def.max())
                def_y_min = min(def_y_min, y_def.min())
                def_y_max = max(def_y_max, y_def.max())

    pad = 0.1
    pad_x = (x_ref.max() - x_ref.min()) * pad
    pad_y = (y_ref.max() - y_ref.min()) * pad
    camera_ref = (x_ref.min() - pad_x, x_ref.max() + pad_x,
                  y_ref.min() - pad_y, y_ref.max() + pad_y)
    pad_xd = (def_x_max - def_x_min) * pad
    pad_yd = (def_y_max - def_y_min) * pad
    camera_def = (def_x_min - pad_xd, def_x_max + pad_xd,
                  def_y_min - pad_yd, def_y_max + pad_yd)

    return {
        'max_steps': max_steps, 'pos_ref': pos_ref,
        'x_ref': x_ref, 'y_ref': y_ref,
        'poly_verts_ref': poly_verts_ref,
        'erosion_masks': erosion_masks,
        'erosion_counts': erosion_counts,
        'valid_node_masks': valid_node_masks,
        'disp_max': disp_max, 'error_max': error_max,
        'Ux_range': (Ux_min, Ux_max), 'Uy_range': (Uy_min, Uy_max),
        'camera_ref': camera_ref, 'camera_def': camera_def,
    }


# ─── GIF creators ────────────────────────────────────────────────────────

def create_reference_gif(frames, precomputed, seq_pred, seq_targ, elements,
                         case_name, output_dir, model_name='Prediction',
                         fps=10, eval_mode='rollout'):
    """Target vs prediction on the undeformed reference mesh."""
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
            ax.clear(); ax.set_xlim(camera[0], camera[1])
            ax.set_ylim(camera[2], camera[3]); ax.set_aspect('equal'); ax.axis('off')
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        d_targ = np.sqrt(U_targ[:, 0] ** 2 + U_targ[:, 1] ** 2)
        d_pred = np.sqrt(U_pred[:, 0] ** 2 + U_pred[:, 1] ** 2)
        eroded_mask = erosion_masks[frame]
        n_eroded = eroded_mask.sum()
        render_mesh_fast(axes[0], poly_verts, d_targ, elements, eroded_mask,
                         0, disp_max, cmap, norm, show_eroded=True)
        title = f'Target (t={frame})'
        if n_eroded > 0: title += f' [{n_eroded} eroded]'
        axes[0].set_title(title, fontsize=12)
        render_mesh_fast(axes[1], poly_verts, d_pred, elements, eroded_mask,
                         0, disp_max, cmap, norm, show_eroded=True)
        axes[1].set_title(f'{model_name} (t={frame})', fontsize=12)
        fig.suptitle(f'Reference Config ({mode_label}): {case_name}', fontsize=14)
        return axes.tolist()

    anim = FuncAnimation(fig, animate, frames=len(frames),
                         interval=1000 // fps, blit=False)
    anim.save(Path(output_dir) / f'{eval_mode}_reference_{case_name}.gif',
              writer=PillowWriter(fps=fps))
    plt.close(fig)


def create_deformed_gif(frames, precomputed, seq_pred, seq_targ, elements,
                        case_name, output_dir, model_name='Prediction',
                        fps=10, eval_mode='rollout'):
    """Target vs prediction on deformed mesh (displacement applied)."""
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
            ax.clear(); ax.set_xlim(camera[0], camera[1])
            ax.set_ylim(camera[2], camera[3]); ax.set_aspect('equal'); ax.axis('off')
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        pos_targ = np.column_stack([x_ref + U_targ[:, 0], y_ref + U_targ[:, 1]])
        pos_pred = np.column_stack([x_ref + U_pred[:, 0], y_ref + U_pred[:, 1]])
        pv_targ = precompute_element_polygons(pos_targ, elements)
        pv_pred = precompute_element_polygons(pos_pred, elements)
        d_targ = np.sqrt(U_targ[:, 0] ** 2 + U_targ[:, 1] ** 2)
        d_pred = np.sqrt(U_pred[:, 0] ** 2 + U_pred[:, 1] ** 2)
        eroded_mask = erosion_masks[frame]
        n_eroded = eroded_mask.sum()
        render_mesh_fast(axes[0], pv_targ, d_targ, elements, eroded_mask,
                         0, disp_max, cmap, norm, show_eroded=False)
        title = f'Target (t={frame})'
        if n_eroded > 0: title += f' [{n_eroded} eroded]'
        axes[0].set_title(title, fontsize=12)
        render_mesh_fast(axes[1], pv_pred, d_pred, elements, eroded_mask,
                         0, disp_max, cmap, norm, show_eroded=False)
        axes[1].set_title(f'{model_name} (t={frame})', fontsize=12)
        fig.suptitle(f'Deformed Config ({mode_label}): {case_name}', fontsize=14)
        return axes.tolist()

    anim = FuncAnimation(fig, animate, frames=len(frames),
                         interval=1000 // fps, blit=False)
    anim.save(Path(output_dir) / f'{eval_mode}_deformed_{case_name}.gif',
              writer=PillowWriter(fps=fps))
    plt.close(fig)


def create_error_gif(frames, precomputed, seq_pred, seq_targ, elements,
                     case_name, output_dir, fps=10, eval_mode='rollout'):
    """Spatial error magnitude on the reference mesh."""
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
        ax.clear(); ax.set_xlim(camera[0], camera[1])
        ax.set_ylim(camera[2], camera[3]); ax.set_aspect('equal'); ax.axis('off')
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        error_mag = np.sqrt(
            (U_targ[:, 0] - U_pred[:, 0]) ** 2 +
            (U_targ[:, 1] - U_pred[:, 1]) ** 2)
        eroded_mask = erosion_masks[frame]
        render_mesh_fast(ax, poly_verts, error_mag, elements, eroded_mask,
                         0, error_max, cmap, norm, show_eroded=True)
        n_eroded = eroded_mask.sum()
        title = f'Prediction Error - t={frame}'
        if n_eroded > 0: title += f' [{n_eroded} eroded]'
        ax.set_title(title, fontsize=14)
        fig.suptitle(f'Error ({mode_label}): {case_name}', fontsize=14)
        return [ax]

    anim = FuncAnimation(fig, animate, frames=len(frames),
                         interval=1000 // fps, blit=False)
    anim.save(Path(output_dir) / f'{eval_mode}_error_{case_name}.gif',
              writer=PillowWriter(fps=fps))
    plt.close(fig)


def create_erosion_plot(precomputed, case_name, output_dir, eval_mode='rollout'):
    """Static plot of erosion count over time."""
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
    fig.savefig(Path(output_dir) / f'{eval_mode}_erosion_{case_name}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── High-level convenience ──────────────────────────────────────────────

def create_elasto_visualizations(simulation, seq_pred, seq_targ, sim_idx,
                                  output_dir, model_name='Prediction',
                                  fps=10, frame_skip=1, eval_mode='rollout'):
    """
    Create all elastoplastic visualizations (reference, deformed, error GIFs + erosion).
    """
    first_data = simulation[0]
    if not hasattr(first_data, 'elements'):
        print(f"  Skipping sim {sim_idx}: no elements attribute")
        return
    elements = first_data.elements.cpu().numpy()

    if eval_mode == 'snapshot':
        sim_for_viz = simulation[1:len(seq_pred) + 1]
    else:
        sim_for_viz = simulation[:len(seq_pred)]

    max_steps = min(len(seq_pred), len(seq_targ), len(sim_for_viz))
    if max_steps < 2:
        return

    case_name = f'simulation_{sim_idx}'
    print(f"\n{'=' * 70}")
    print(f"Creating GIFs for {case_name} ({eval_mode} mode)")
    print(f"  Elements: {len(elements)}, Timesteps: {max_steps}")
    print(f"{'=' * 70}")

    precomputed = precompute_elasto_viz_data(sim_for_viz, seq_pred, seq_targ, elements)
    frames = list(range(0, max_steps, frame_skip))

    for gif_type, creator in [('reference', create_reference_gif),
                               ('deformed', create_deformed_gif),
                               ('error', create_error_gif)]:
        kwargs = dict(frames=frames, precomputed=precomputed,
                      seq_pred=seq_pred, seq_targ=seq_targ,
                      elements=elements, case_name=case_name,
                      output_dir=output_dir, fps=fps, eval_mode=eval_mode)
        if gif_type != 'error':
            kwargs['model_name'] = model_name
        creator(**kwargs)
        print(f"    ✓ {gif_type}_{case_name}.gif")

    if max(precomputed['erosion_counts']) > 0:
        create_erosion_plot(precomputed, case_name, output_dir, eval_mode=eval_mode)
        print(f"    ✓ erosion_{case_name}.png")