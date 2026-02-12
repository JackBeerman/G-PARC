import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from pathlib import Path
from typing import List, Union, Iterator
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch_geometric.data import Data
from tqdm import tqdm
from neuralop.models import FNOGNO

# ==========================================
# 1) Dataset Definition (Same as training)
# ==========================================

class ElastoPlasticDataset(IterableDataset):
    def __init__(self,
                 directory: Union[str, Path],
                 simulation_ids: List[str] = None,
                 seq_len: int = 10,
                 stride: int = 1,
                 num_static_feats: int = 2,
                 num_dynamic_feats: int = 2,
                 file_pattern: str = "*.pt",
                 use_element_features: bool = False):
        super().__init__()
        self.directory = Path(directory)
        self.seq_len = seq_len
        self.stride = stride
        self.file_pattern = file_pattern
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.use_element_features = use_element_features

        if simulation_ids is None:
            self.simulation_ids = self._discover_simulation_ids()
        else:
            self.simulation_ids = simulation_ids

        self.var_names = ['U_x', 'U_y']
        if self.use_element_features:
            self.element_var_names = ['EROSION_STATUS']

        print(f"Dataset: {self.directory} | {len(self.simulation_ids)} sims")

    def _discover_simulation_ids(self) -> List[str]:
        files = list(self.directory.glob(self.file_pattern))
        return [file.stem for file in files]

    def _extract_id_from_name(self, sim_name: str) -> int:
        import re
        match = re.search(r'\d+', sim_name)
        return int(match.group()) if match else abs(hash(sim_name)) % 100000

    def __iter__(self) -> Iterator[List[Data]]:
        worker_info = get_worker_info()
        if worker_info is None:
            files_to_process = self.simulation_ids
        else:
            total_files = len(self.simulation_ids)
            per_worker = int(math.ceil(total_files / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, total_files)
            files_to_process = self.simulation_ids[start:end]

        for sim_name in files_to_process:
            dataset_file = self.directory / f"{sim_name}.pt"
            sim_id_int = self._extract_id_from_name(sim_name)
            try:
                sim_data = torch.load(dataset_file, weights_only=False)
                if not isinstance(sim_data, list): continue

                for data in sim_data:
                    data.mesh_id = torch.tensor([sim_id_int], dtype=torch.long)

                T = len(sim_data)
                max_start = T - self.seq_len
                if max_start < 0: continue

                for start_idx in range(0, max_start + 1, self.stride):
                    window = []
                    for offset in range(self.seq_len):
                        t = start_idx + offset
                        data_t = sim_data[t].clone()
                        if not hasattr(data_t, 'x') or not hasattr(data_t, 'y'): break
                        window.append(data_t)
                    if len(window) == self.seq_len:
                        yield window
            except Exception as e:
                print(f"Error loading {dataset_file}: {e}")

# ==========================================
# 2) Helper Utilities (Same as training)
# ==========================================

def collate_windows(batch):
    return batch

@torch.no_grad()
def get_node_coords(data_t: Data, num_static_feats: int = 2):
    if hasattr(data_t, "pos") and data_t.pos is not None:
        return data_t.pos
    return data_t.x[:, :num_static_feats]

def get_dyn_state(data_t: Data, num_static_feats: int = 2, num_dynamic_feats: int = 2):
    return data_t.x[:, num_static_feats:num_static_feats + num_dynamic_feats]

def get_next_dyn_target(data_t: Data, num_dynamic_feats: int = 2):
    return data_t.y[:, :num_dynamic_feats] if data_t.y.shape[-1] >= num_dynamic_feats else data_t.y

def make_regular_grid_points(h: int, w: int, device: str):
    ys = torch.linspace(0.0, 1.0, steps=h, device=device)
    xs = torch.linspace(0.0, 1.0, steps=w, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([X, Y], dim=-1)

def rasterize_to_grid(points_xy, values, h: int, w: int):
    xy = points_xy.clamp(0.0, 1.0)
    ix = torch.floor(xy[:, 0] * (w - 1)).long()
    iy = torch.floor(xy[:, 1] * (h - 1)).long()
    idx = iy * w + ix
    C, HW = values.shape[1], h * w
    f_sum = torch.zeros(HW, C, device=values.device, dtype=values.dtype)
    cnt = torch.zeros(HW, 1, device=values.device, dtype=values.dtype)
    f_sum.index_add_(0, idx, values)
    cnt.index_add_(0, idx, torch.ones(values.shape[0], 1, device=values.device))
    return (f_sum / torch.clamp(cnt, min=1.0)).view(h, w, C)

def window_to_samples(window, device, num_static=2, num_dyn=2):
    samples = []
    for data_t in window:
        coords = get_node_coords(data_t, num_static).to(device).float()
        dyn = get_dyn_state(data_t, num_static, num_dyn).to(device).float()
        tgt = get_next_dyn_target(data_t, num_dyn).to(device).float()
        samples.append((coords, dyn, tgt))
    return samples

# ==========================================
# 3) Visualization Helper Functions
# ==========================================

def precompute_element_polygons(pos, elements):
    """Precompute polygon vertices for all elements."""
    return pos[elements]

def render_mesh_fast(ax, poly_verts, node_values, elements, eroded_mask, 
                     vmin, vmax, cmap, norm, show_eroded=True):
    """Fast mesh rendering using PolyCollection."""
    elem_values = node_values[elements].mean(axis=1)
    
    if show_eroded and eroded_mask is not None and eroded_mask.sum() > 0:
        active_mask = ~eroded_mask
        poly_verts_active = poly_verts[active_mask]
        elem_values_active = elem_values[active_mask]
    else:
        poly_verts_active = poly_verts
        elem_values_active = elem_values
    
    if len(poly_verts_active) > 0:
        colors = cmap(norm(elem_values_active))
        pc = PolyCollection(poly_verts_active, facecolors=colors, 
                           edgecolors='none', linewidths=0)
        ax.add_collection(pc)

def precompute_visualization_data(seq_targ, seq_pred, pos_ref, elements, 
                                  erosion_status_seq=None):
    """Precompute data for visualization."""
    max_steps = len(seq_targ)
    
    # Displacement magnitude range
    all_disp = []
    for U in seq_targ:
        all_disp.append(np.sqrt(U[:, 0]**2 + U[:, 1]**2))
    disp_max = np.max([d.max() for d in all_disp])
    
    # Error range
    all_errors = []
    for U_targ, U_pred in zip(seq_targ, seq_pred):
        error = np.sqrt((U_targ[:, 0] - U_pred[:, 0])**2 + 
                       (U_targ[:, 1] - U_pred[:, 1])**2)
        all_errors.append(error)
    error_max = np.max([e.max() for e in all_errors])
    
    # Component ranges
    Ux_all = np.concatenate([U[:, 0] for U in seq_targ])
    Uy_all = np.concatenate([U[:, 1] for U in seq_targ])
    Ux_range = (Ux_all.min(), Ux_all.max())
    Uy_range = (Uy_all.min(), Uy_all.max())
    
    # Reference configuration polygons
    poly_verts_ref = precompute_element_polygons(pos_ref, elements)
    
    # Camera bounds (reference)
    margin = 0.05
    x_min, x_max = pos_ref[:, 0].min(), pos_ref[:, 0].max()
    y_min, y_max = pos_ref[:, 1].min(), pos_ref[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    camera_ref = [x_min - margin*x_range, x_max + margin*x_range,
                  y_min - margin*y_range, y_max + margin*y_range]
    
    # Camera bounds (deformed) - account for displacements
    all_pos_def = []
    for U in seq_targ:
        pos_def = np.column_stack([pos_ref[:, 0] + U[:, 0], 
                                   pos_ref[:, 1] + U[:, 1]])
        all_pos_def.append(pos_def)
    all_pos_def = np.vstack(all_pos_def)
    x_min_def = all_pos_def[:, 0].min()
    x_max_def = all_pos_def[:, 0].max()
    y_min_def = all_pos_def[:, 1].min()
    y_max_def = all_pos_def[:, 1].max()
    x_range_def = x_max_def - x_min_def
    y_range_def = y_max_def - y_min_def
    camera_def = [x_min_def - margin*x_range_def, x_max_def + margin*x_range_def,
                  y_min_def - margin*y_range_def, y_max_def + margin*y_range_def]
    
    # Erosion masks
    erosion_masks = []
    erosion_counts = []
    if erosion_status_seq is not None:
        for status in erosion_status_seq:
            mask = status > 0  # Eroded if status > 0
            erosion_masks.append(mask)
            erosion_counts.append(mask.sum())
    else:
        for _ in range(max_steps):
            erosion_masks.append(np.zeros(len(elements), dtype=bool))
            erosion_counts.append(0)
    
    return {
        'disp_max': disp_max,
        'error_max': error_max,
        'Ux_range': Ux_range,
        'Uy_range': Uy_range,
        'poly_verts_ref': poly_verts_ref,
        'camera_ref': camera_ref,
        'camera_def': camera_def,
        'x_ref': pos_ref[:, 0],
        'y_ref': pos_ref[:, 1],
        'erosion_masks': erosion_masks,
        'erosion_counts': erosion_counts,
        'max_steps': max_steps
    }

# ==========================================
# 4) GIF Creation Functions
# ==========================================

def _create_reference_gif(frames, precomputed, seq_pred, seq_targ, elements,
                          case_name, output_dir, fps, eval_mode='rollout'):
    """Create reference configuration GIF."""
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
            ax.clear()
            ax.set_xlim(camera[0], camera[1])
            ax.set_ylim(camera[2], camera[3])
            ax.set_aspect('equal')
            ax.axis('off')
        
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        d_targ = np.sqrt(U_targ[:, 0]**2 + U_targ[:, 1]**2)
        d_pred = np.sqrt(U_pred[:, 0]**2 + U_pred[:, 1]**2)
        eroded_mask = erosion_masks[frame]
        n_eroded = eroded_mask.sum()
        
        render_mesh_fast(axes[0], poly_verts, d_targ, elements, eroded_mask,
                        0, disp_max, cmap, norm, show_eroded=True)
        title = f'Target (t={frame})'
        if n_eroded > 0:
            title += f' [{n_eroded} eroded]'
        axes[0].set_title(title, fontsize=12)
        
        render_mesh_fast(axes[1], poly_verts, d_pred, elements, eroded_mask,
                        0, disp_max, cmap, norm, show_eroded=True)
        axes[1].set_title(f'Prediction (t={frame})', fontsize=12)
        
        fig.suptitle(f'Reference Config ({mode_label}): {case_name}', fontsize=14)
        return axes.tolist()
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(Path(output_dir) / f'reference_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def _create_deformed_gif(frames, precomputed, seq_pred, seq_targ, elements,
                         case_name, output_dir, fps, eval_mode='rollout'):
    """Create deformed configuration GIF."""
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
            ax.clear()
            ax.set_xlim(camera[0], camera[1])
            ax.set_ylim(camera[2], camera[3])
            ax.set_aspect('equal')
            ax.axis('off')
        
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        
        pos_targ_def = np.column_stack([x_ref + U_targ[:, 0], y_ref + U_targ[:, 1]])
        pos_pred_def = np.column_stack([x_ref + U_pred[:, 0], y_ref + U_pred[:, 1]])
        
        poly_verts_targ = precompute_element_polygons(pos_targ_def, elements)
        poly_verts_pred = precompute_element_polygons(pos_pred_def, elements)
        
        d_targ = np.sqrt(U_targ[:, 0]**2 + U_targ[:, 1]**2)
        d_pred = np.sqrt(U_pred[:, 0]**2 + U_pred[:, 1]**2)
        eroded_mask = erosion_masks[frame]
        n_eroded = eroded_mask.sum()
        
        render_mesh_fast(axes[0], poly_verts_targ, d_targ, elements, eroded_mask,
                        0, disp_max, cmap, norm, show_eroded=False)
        title = f'Target (t={frame})'
        if n_eroded > 0:
            title += f' [{n_eroded} eroded]'
        axes[0].set_title(title, fontsize=12)
        
        render_mesh_fast(axes[1], poly_verts_pred, d_pred, elements, eroded_mask,
                        0, disp_max, cmap, norm, show_eroded=False)
        axes[1].set_title(f'Prediction (t={frame})', fontsize=12)
        
        fig.suptitle(f'Deformed Config ({mode_label}): {case_name}', fontsize=14)
        return axes.tolist()
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(Path(output_dir) / f'deformed_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def _create_error_gif(frames, precomputed, seq_pred, seq_targ, elements,
                      case_name, output_dir, fps, eval_mode='rollout'):
    """Create error visualization GIF."""
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
        ax.clear()
        ax.set_xlim(camera[0], camera[1])
        ax.set_ylim(camera[2], camera[3])
        ax.set_aspect('equal')
        ax.axis('off')
        
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        error_mag = np.sqrt((U_targ[:, 0] - U_pred[:, 0])**2 +
                           (U_targ[:, 1] - U_pred[:, 1])**2)
        eroded_mask = erosion_masks[frame]
        
        render_mesh_fast(ax, poly_verts, error_mag, elements, eroded_mask,
                        0, error_max, cmap, norm, show_eroded=True)
        
        n_eroded = eroded_mask.sum()
        title = f'Prediction Error - t={frame}'
        if n_eroded > 0:
            title += f' [{n_eroded} eroded]'
        ax.set_title(title, fontsize=14)
        fig.suptitle(f'Error ({mode_label}): {case_name}', fontsize=14)
        return [ax]
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(Path(output_dir) / f'error_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def _create_component_gif(frames, precomputed, seq_pred, seq_targ, elements,
                          case_name, output_dir, fps, component=0, eval_mode='rollout'):
    """Create component (Ux or Uy) visualization GIF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    
    comp_name = 'U_x' if component == 0 else 'U_y'
    comp_range = precomputed['Ux_range'] if component == 0 else precomputed['Uy_range']
    vmin, vmax = comp_range
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps.get_cmap('RdBu_r')
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label(comp_name, fontsize=11)
    
    camera = precomputed['camera_ref']
    poly_verts = precomputed['poly_verts_ref']
    erosion_masks = precomputed['erosion_masks']
    
    mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'
    
    def animate(frame_idx):
        frame = frames[frame_idx]
        for ax in axes:
            ax.clear()
            ax.set_xlim(camera[0], camera[1])
            ax.set_ylim(camera[2], camera[3])
            ax.set_aspect('equal')
            ax.axis('off')
        
        U_targ, U_pred = seq_targ[frame], seq_pred[frame]
        eroded_mask = erosion_masks[frame]
        
        render_mesh_fast(axes[0], poly_verts, U_targ[:, component], elements, eroded_mask,
                        vmin, vmax, cmap, norm, show_eroded=True)
        axes[0].set_title(f'Target (t={frame})', fontsize=12)
        
        render_mesh_fast(axes[1], poly_verts, U_pred[:, component], elements, eroded_mask,
                        vmin, vmax, cmap, norm, show_eroded=True)
        axes[1].set_title(f'Prediction (t={frame})', fontsize=12)
        
        title = 'X-Displacement' if component == 0 else 'Y-Displacement'
        fig.suptitle(f'{title} ({mode_label}): {case_name}', fontsize=14)
        return axes.tolist()
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    suffix = 'ux_component' if component == 0 else 'uy_component'
    anim.save(Path(output_dir) / f'{suffix}_{case_name}.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


def create_all_gifs(seq_pred, seq_targ, pos_ref, elements, case_name, 
                    output_dir, fps=5, frame_skip=1, erosion_status_seq=None,
                    eval_mode='rollout'):
    """Create all visualization GIFs for a single simulation."""
    print(f"  Precomputing visualization data...")
    precomputed = precompute_visualization_data(seq_targ, seq_pred, pos_ref, 
                                                elements, erosion_status_seq)
    
    max_steps = precomputed['max_steps']
    frames = list(range(0, max_steps, frame_skip))
    
    print(f"  Creating reference GIF...")
    _create_reference_gif(frames, precomputed, seq_pred, seq_targ, elements,
                         case_name, output_dir, fps, eval_mode)
    
    print(f"  Creating deformed GIF...")
    _create_deformed_gif(frames, precomputed, seq_pred, seq_targ, elements,
                        case_name, output_dir, fps, eval_mode)
    
    print(f"  Creating error GIF...")
    _create_error_gif(frames, precomputed, seq_pred, seq_targ, elements,
                     case_name, output_dir, fps, eval_mode)
    
    print(f"  Creating U_x component GIF...")
    _create_component_gif(frames, precomputed, seq_pred, seq_targ, elements,
                         case_name, output_dir, fps, component=0, eval_mode=eval_mode)
    
    print(f"  Creating U_y component GIF...")
    _create_component_gif(frames, precomputed, seq_pred, seq_targ, elements,
                         case_name, output_dir, fps, component=1, eval_mode=eval_mode)

# ==========================================
# 5) Evaluation Functions
# ==========================================

@torch.no_grad()
def evaluate_one_step(loader, model, device, grid_h, grid_w, grid_p, num_static, num_dyn):
    """One-step prediction evaluation"""
    model.eval()
    
    errors = []
    errors_per_var = {i: [] for i in range(num_dyn)}
    
    for batch_windows in tqdm(loader, desc="One-step eval"):
        for window in batch_windows:
            samples = window_to_samples(window, device, num_static, num_dyn)
            for (coords, dyn, tgt) in samples:
                f_grid = rasterize_to_grid(coords, dyn, grid_h, grid_w)
                pred = model(grid_p, coords, f_grid)
                if pred.ndim == 3: 
                    pred = pred.squeeze(0)
                
                # MSE per sample
                error = torch.mean((pred - tgt) ** 2).item()
                errors.append(error)
                
                # Per-variable MSE
                for i in range(num_dyn):
                    var_error = torch.mean((pred[:, i] - tgt[:, i]) ** 2).item()
                    errors_per_var[i].append(var_error)
    
    results = {
        'mse': np.mean(errors),
        'mse_std': np.std(errors),
        'rmse': np.sqrt(np.mean(errors)),
        'var_mse': {i: np.mean(errors_per_var[i]) for i in range(num_dyn)},
        'var_rmse': {i: np.sqrt(np.mean(errors_per_var[i])) for i in range(num_dyn)}
    }
    
    return results

@torch.no_grad()
def evaluate_autoregressive_with_viz(test_dir, model, device, grid_h, grid_w, grid_p, 
                                     num_static, num_dyn, max_rollout_steps=20, 
                                     num_sims=5, output_dir=None, create_gifs=True):
    """Autoregressive rollout evaluation with visualization"""
    model.eval()
    
    # Load test simulations
    sim_files = sorted(list(Path(test_dir).glob("*.pt")))[:num_sims]
    
    all_rollout_errors = []
    
    for sim_idx, sim_file in enumerate(tqdm(sim_files, desc="Autoregressive rollout")):
        sim_data = torch.load(sim_file, weights_only=False)
        
        # Start from first timestep
        T = min(len(sim_data), max_rollout_steps + 1)
        
        # Get reference positions and elements
        data_0 = sim_data[0]
        pos_ref = get_node_coords(data_0, num_static).cpu().numpy()
        elements = data_0.face.T.cpu().numpy() if hasattr(data_0, 'face') else None
        
        # Get erosion status if available
        erosion_status_seq = None
        if hasattr(data_0, 'elem_attr') and data_0.elem_attr is not None:
            erosion_status_seq = [sim_data[t].elem_attr.cpu().numpy() for t in range(T)]
        
        # Ground truth trajectory
        gt_trajectory = []
        for t in range(T):
            data_t = sim_data[t]
            dyn = get_dyn_state(data_t, num_static, num_dyn).cpu().numpy()
            gt_trajectory.append(dyn)
        
        # Predicted trajectory (autoregressive)
        pred_trajectory = [gt_trajectory[0]]  # Start from ground truth
        current_state = torch.tensor(gt_trajectory[0], device=device, dtype=torch.float32)
        
        rollout_errors = []
        for t in range(1, T):
            # Get coordinates at this timestep
            data_t = sim_data[t]
            coords = get_node_coords(data_t, num_static).to(device).float()
            
            # Predict next state
            f_grid = rasterize_to_grid(coords, current_state, grid_h, grid_w)
            pred = model(grid_p, coords, f_grid)
            if pred.ndim == 3:
                pred = pred.squeeze(0)
            
            # Compute error vs ground truth
            gt = torch.tensor(gt_trajectory[t], device=device, dtype=torch.float32)
            error = torch.mean((pred - gt) ** 2).item()
            rollout_errors.append(error)
            
            # Update state for next prediction
            current_state = pred
            pred_trajectory.append(pred.cpu().numpy())
        
        all_rollout_errors.append(rollout_errors)
        
        # Create GIFs for this simulation
        if create_gifs and output_dir is not None and elements is not None:
            case_name = f"sim_{sim_idx:03d}"
            print(f"\nCreating GIFs for {case_name}...")
            create_all_gifs(pred_trajectory, gt_trajectory, pos_ref, elements,
                          case_name, output_dir, fps=5, frame_skip=1,
                          erosion_status_seq=erosion_status_seq, eval_mode='rollout')
    
    # Compute statistics
    max_len = max(len(e) for e in all_rollout_errors)
    rollout_mse_per_step = []
    for step in range(max_len):
        step_errors = [errors[step] for errors in all_rollout_errors if len(errors) > step]
        rollout_mse_per_step.append(np.mean(step_errors))
    
    results = {
        'rollout_mse_per_step': rollout_mse_per_step,
        'rollout_rmse_per_step': [np.sqrt(e) for e in rollout_mse_per_step],
        'final_mse': rollout_mse_per_step[-1] if rollout_mse_per_step else None,
    }
    
    return results

# ==========================================
# 6) Plotting Functions
# ==========================================

def plot_one_step_results(results, save_path):
    """Plot one-step prediction results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Overall MSE
    ax = axes[0]
    ax.bar(['Overall'], [results['mse']], yerr=[results['mse_std']], capsize=5)
    ax.set_ylabel('MSE')
    ax.set_title('One-Step Prediction Error')
    ax.grid(True, alpha=0.3)
    
    # Per-variable MSE
    ax = axes[1]
    var_names = ['U_x', 'U_y']
    var_mse = [results['var_mse'][i] for i in range(len(var_names))]
    ax.bar(var_names, var_mse)
    ax.set_ylabel('MSE')
    ax.set_title('Per-Variable MSE')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved one-step results to {save_path}")

def plot_autoregressive_results(results, save_path):
    """Plot autoregressive rollout results"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = np.arange(len(results['rollout_mse_per_step']))
    mse = results['rollout_mse_per_step']
    rmse = results['rollout_rmse_per_step']
    
    ax.plot(steps, mse, 'b-', linewidth=2, label='MSE')
    ax.plot(steps, rmse, 'r--', linewidth=2, label='RMSE')
    ax.set_xlabel('Rollout Step', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Autoregressive Rollout Error', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved autoregressive results to {save_path}")

# ==========================================
# 7) Main Evaluation
# ==========================================

def main():
    # --- Config ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BASE_DIR = "/scratch/jtb3sud/processed_elasto_plastic/zscore/normalized"
    CHECKPOINT_PATH = "/scratch/jtb3sud/elasto_fno_gno_results/v1/checkpoint_best.pth"
    RESULTS_DIR = Path("/scratch/jtb3sud/elasto_fno_gno_results/v1/evaluation")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    SEQ_LEN, STRIDE = 2, 1
    NUM_STATIC, NUM_DYN = 2, 2
    GRID_H, GRID_W = 48, 48
    MAX_ROLLOUT_STEPS = 20
    NUM_ROLLOUT_SIMS = 5
    CREATE_GIFS = True
    
    print("="*60)
    print("FNOGNO Evaluation")
    print("="*60)
    
    # --- Load Model ---
    print(f"\nLoading model from: {CHECKPOINT_PATH}")
    in_p_grid = make_regular_grid_points(GRID_H, GRID_W, DEVICE)
    model = FNOGNO(
        in_channels=NUM_DYN,
        out_channels=NUM_DYN,
        fno_n_modes=(12, 12),
        fno_hidden_channels=32,
        fno_n_layers=3,
        gno_coord_dim=2,
        gno_radius=0.05,
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
    ).to(DEVICE)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_loss']:.6e}")
    print(f"  Val loss: {checkpoint['val_loss']:.6e}")
    
    # --- Setup Test Dataset ---
    test_dir = Path(BASE_DIR) / "test"
    test_dataset = ElastoPlasticDataset(
        test_dir, 
        seq_len=SEQ_LEN, 
        stride=STRIDE,
        num_static_feats=NUM_STATIC, 
        num_dynamic_feats=NUM_DYN
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        num_workers=0, 
        collate_fn=collate_windows
    )
    
    # --- One-Step Evaluation ---
    print("\n" + "="*60)
    print("ONE-STEP PREDICTION EVALUATION")
    print("="*60)
    one_step_results = evaluate_one_step(
        test_loader, model, DEVICE, GRID_H, GRID_W, in_p_grid, NUM_STATIC, NUM_DYN
    )
    
    print(f"\nResults:")
    print(f"  MSE:  {one_step_results['mse']:.6e} ± {one_step_results['mse_std']:.6e}")
    print(f"  RMSE: {one_step_results['rmse']:.6e}")
    print(f"\nPer-variable RMSE:")
    print(f"  U_x: {one_step_results['var_rmse'][0]:.6e}")
    print(f"  U_y: {one_step_results['var_rmse'][1]:.6e}")
    
    plot_one_step_results(one_step_results, RESULTS_DIR / "one_step_results.png")
    
    # --- Autoregressive Evaluation ---
    print("\n" + "="*60)
    print("AUTOREGRESSIVE ROLLOUT EVALUATION")
    print("="*60)
    rollout_results = evaluate_autoregressive_with_viz(
        test_dir, model, DEVICE, GRID_H, GRID_W, in_p_grid,
        NUM_STATIC, NUM_DYN, MAX_ROLLOUT_STEPS, NUM_ROLLOUT_SIMS,
        output_dir=RESULTS_DIR if CREATE_GIFS else None,
        create_gifs=CREATE_GIFS
    )
    
    print(f"\nRollout results ({MAX_ROLLOUT_STEPS} steps):")
    print(f"  Initial RMSE: {rollout_results['rollout_rmse_per_step'][0]:.6e}")
    print(f"  Final RMSE:   {rollout_results['rollout_rmse_per_step'][-1]:.6e}")
    print(f"  Error growth: {rollout_results['rollout_rmse_per_step'][-1] / rollout_results['rollout_rmse_per_step'][0]:.2f}x")
    
    plot_autoregressive_results(rollout_results, RESULTS_DIR / "autoregressive_results.png")
    
    # --- Save Results ---
    results_dict = {
        'one_step': one_step_results,
        'autoregressive': {
            'rollout_mse_per_step': rollout_results['rollout_mse_per_step'],
            'rollout_rmse_per_step': rollout_results['rollout_rmse_per_step'],
            'final_mse': rollout_results['final_mse']
        },
        'config': {
            'checkpoint_path': str(CHECKPOINT_PATH),
            'epoch': checkpoint['epoch'],
            'grid_size': (GRID_H, GRID_W),
            'max_rollout_steps': MAX_ROLLOUT_STEPS,
            'num_rollout_sims': NUM_ROLLOUT_SIMS
        }
    }
    
    import json
    with open(RESULTS_DIR / "results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {RESULTS_DIR}")
    if CREATE_GIFS:
        print(f"GIFs saved for {NUM_ROLLOUT_SIMS} simulations")

if __name__ == "__main__":
    main()