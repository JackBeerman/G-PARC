#!/usr/bin/env python3
"""
G-PARCv2 River Evaluation Script
=================================
Evaluates river models trained with scheduled sampling.
Supports ROLLOUT and SNAPSHOT evaluation modes.

River-specific:
  - No erosion (Eulerian fixed mesh)
  - No mesh deformation
  - 4 dynamic variables: Depth, Volume, Vel_X, Vel_Y
  - 9 static features: x, y, Area, Elevation, Slope, Aspect, Curvature, Manning's n, FA
  - Two meshes: White River (mesh_id=0) and Iowa River (mesh_id=1)
  - Quad mesh visualization (structured grid)
"""

import argparse
import os
import sys
import re
from pathlib import Path
import json
import warnings
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("⚠️  h5py not available — will fall back to scatter plots for visualization")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.featureextractor import GraphConvFeatureExtractorV2
from differentiator.riverdifferentiator import RiverDifferentiator
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from models.riverV2 import GPARC_River_V2
from data.Riverdataset import RiverDataset

warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# VARIABLE CONFIGURATION
# ==============================================================================

VAR_NAMES = ['Depth', 'Volume', 'Vel_X', 'Vel_Y']
VAR_CMAPS = ['Blues', 'Greens', 'RdBu_r', 'RdBu_r']
VAR_UNITS_NORM = ['(norm)', '(norm)', '(norm)', '(norm)']
VAR_UNITS_PHYS = ['m', 'm³', 'm/s', 'm/s']


# ==============================================================================
# HEC-RAS MESH LOADING
# ==============================================================================

def load_hec_ras_mesh(hdf_path, sim_id):
    """
    Load HEC-RAS mesh (facepoints + cell indices) from .p0x.hdf file.
    
    Args:
        hdf_path: Path to HDF file
        sim_id: Simulation ID (used to detect Iowa vs White River paths)
    
    Returns:
        facepts: [M, 2] array of face point coordinates
        cells: [C, max_verts] array of cell vertex indices (padded with -1)
    """
    if not HAS_H5PY:
        raise ImportError("h5py required for HEC-RAS mesh loading")
    
    with h5py.File(hdf_path, "r") as f:
        if "iw" in sim_id.lower():
            facepts_path = "Geometry/2D Flow Areas/Flow Area/FacePoints Coordinate"
            cells_path = "Geometry/2D Flow Areas/Flow Area/Cells FacePoint Indexes"
        else:
            facepts_path = "Geometry/2D Flow Areas/Perimeter 1/FacePoints Coordinate"
            cells_path = "Geometry/2D Flow Areas/Perimeter 1/Cells FacePoint Indexes"
        
        facepts = f[facepts_path][:]
        cells = f[cells_path][:]
    
    return facepts, cells


def get_clean_polygons(facepts, cells):
    """Convert HEC-RAS facepts + padded cell indices to polygon vertex arrays."""
    polys = []
    for cell_ids in cells:
        valid_ids = cell_ids[cell_ids >= 0].astype(int)
        pts = facepts[valid_ids]
        polys.append(pts)
    return polys


def load_mesh_for_sim(sim_id, hec_ras_dir):
    """
    Load the appropriate HEC-RAS mesh for a simulation.
    
    Args:
        sim_id: Simulation identifier (e.g., 'H10', 'H348iw')
        hec_ras_dir: Base directory containing HDF files
    
    Returns:
        polys: List of polygon vertex arrays, or None if loading fails
    """
    hec_ras_dir = Path(hec_ras_dir)
    
    if "iw" in sim_id.lower():
        hdf_path = hec_ras_dir / "Flood_GNN.p01.hdf"
    else:
        hdf_path = hec_ras_dir / "Muncie2D_SI.p02.hdf"
    
    if not hdf_path.exists():
        print(f"  ⚠️  HDF file not found: {hdf_path}")
        return None
    
    try:
        facepts, cells = load_hec_ras_mesh(hdf_path, sim_id)
        polys = get_clean_polygons(facepts, cells)
        print(f"  ✓ Loaded mesh: {len(polys)} cells from {hdf_path.name}")
        return polys
    except Exception as e:
        print(f"  ⚠️  Failed to load mesh from {hdf_path}: {e}")
        return None


def load_denorm_extrema(extrema_path):
    """
    Load global y extrema for denormalization.
    
    Returns dict with y_min, y_max tensors (indexed by variable).
    Denormalization: physical = normalized * (y_max - y_min) + y_min
    """
    extrema_path = Path(extrema_path)
    if not extrema_path.exists():
        print(f"  ⚠️  Extrema file not found: {extrema_path}")
        return None
    
    extrema = torch.load(extrema_path, weights_only=False)
    print(f"  ✓ Loaded extrema: y_min={extrema['y_min'].tolist()}, y_max={extrema['y_max'].tolist()}")
    return extrema


def denormalize_array(normalized, var_idx, extrema):
    """Denormalize a single variable: physical = norm * (y_max - y_min) + y_min."""
    if extrema is None:
        return normalized
    y_min = extrema['y_min'][var_idx].item()
    y_max = extrema['y_max'][var_idx].item()
    return normalized * (y_max - y_min) + y_min


def denormalize_all(normalized, extrema):
    """Denormalize all variables in an [N, D] array."""
    if extrema is None:
        return normalized
    physical = np.zeros_like(normalized)
    for v in range(normalized.shape[1]):
        physical[:, v] = denormalize_array(normalized[:, v], v, extrema)
    return physical


# ==============================================================================
# METRICS
# ==============================================================================

def compute_rrmse(predictions, references, valid_masks=None):
    """
    Compute RRMSE (Relative Root Mean Square Error).
    
    RRMSE = sqrt( (1/n) * sum_i [ MSE_i / ||ref_i||_inf^2 ] )
    """
    if len(predictions) == 0:
        return 0.0
    
    n_samples = 0
    ratio_sum = 0.0
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if valid_masks is not None and valid_masks[i] is not None:
            pred, ref = pred[valid_masks[i]], ref[valid_masks[i]]
        if len(pred) == 0:
            continue
        
        n_nodes = pred.shape[0]
        ref_inf = np.max(np.abs(ref))
        if ref_inf == 0:
            continue
        
        mse = np.sum((pred - ref) ** 2) / n_nodes
        ratio_sum += mse / (ref_inf ** 2)
        n_samples += 1
    
    return np.sqrt(ratio_sum / n_samples) if n_samples > 0 else float('inf')


def compute_rrmse_per_variable(predictions, references, var_names=None, valid_masks=None):
    """Compute RRMSE per variable."""
    if len(predictions) == 0:
        return {}
    
    n_vars = predictions[0].shape[1]
    if var_names is None:
        var_names = [f'var_{i}' for i in range(n_vars)]
    
    rrmse = {}
    for c in range(n_vars):
        n_samples, ratio_sum = 0, 0.0
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if valid_masks is not None and valid_masks[i] is not None:
                p, r = pred[valid_masks[i], c], ref[valid_masks[i], c]
            else:
                p, r = pred[:, c], ref[:, c]
            if len(p) == 0:
                continue
            ref_inf = np.max(np.abs(r))
            if ref_inf == 0:
                continue
            mse = np.sum((p - r) ** 2) / len(p)
            ratio_sum += mse / (ref_inf ** 2)
            n_samples += 1
        rrmse[var_names[c]] = float(np.sqrt(ratio_sum / n_samples)) if n_samples > 0 else float('inf')
    
    return rrmse


def compute_per_step_metrics(seq_pred, seq_targ):
    """Compute RMSE and R² at each timestep."""
    steps = min(len(seq_pred), len(seq_targ))
    rmse_per_step = []
    r2_per_step = []
    
    for t in range(steps):
        p, tg = seq_pred[t].flatten(), seq_targ[t].flatten()
        rmse_per_step.append(float(np.sqrt(mean_squared_error(tg, p))))
        r2_per_step.append(float(r2_score(tg, p)))
    
    return rmse_per_step, r2_per_step


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def get_node_positions(simulation):
    """Extract node positions from first timestep."""
    data0 = simulation[0]
    if hasattr(data0, 'pos') and data0.pos is not None:
        return data0.pos.cpu().numpy()
    return data0.x[:, :2].cpu().numpy()


def create_scalar_field_gif(pos, seq_pred, seq_targ, var_idx, var_name,
                             case_name, output_dir, fps=5, frame_skip=1,
                             polys=None, extrema=None):
    """
    Create side-by-side GIF for a scalar variable.
    Uses PolyCollection if mesh polygons available, falls back to scatter.
    If extrema provided, values are already in physical units.
    """
    max_steps = min(len(seq_pred), len(seq_targ))
    frames = list(range(0, max_steps, frame_skip))
    
    # Unit label
    unit = VAR_UNITS_PHYS[var_idx] if extrema is not None else VAR_UNITS_NORM[var_idx]
    label = f'{var_name} ({unit})'
    
    # Global min/max
    all_vals = np.concatenate([
        np.concatenate([seq_targ[t][:, var_idx] for t in range(max_steps)]),
        np.concatenate([seq_pred[t][:, var_idx] for t in range(max_steps)]),
    ])
    vmin, vmax = all_vals.min(), all_vals.max()
    if vmin == vmax:
        vmax = vmin + 1e-6
    
    # Error range
    error_vals = np.concatenate([
        np.abs(seq_targ[t][:, var_idx] - seq_pred[t][:, var_idx])
        for t in range(max_steps)
    ])
    err_max = np.percentile(error_vals, 99)
    if err_max == 0:
        err_max = 1e-6
    
    cmap_name = VAR_CMAPS[var_idx] if var_idx < len(VAR_CMAPS) else 'viridis'
    use_polys = polys is not None and len(polys) > 0
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(right=0.88, wspace=0.05)
    
    # Colorbars
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
        """Render values on mesh using PolyCollection or scatter."""
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
        error_vals = np.abs(targ_vals - pred_vals)
        
        _render(axes[0], targ_vals, cmap_name, norm_val)
        axes[0].set_title(f'Target (t={t})', fontsize=11)
        
        _render(axes[1], pred_vals, cmap_name, norm_val)
        axes[1].set_title(f'Prediction (t={t})', fontsize=11)
        
        _render(axes[2], error_vals, 'hot', norm_err)
        axes[2].set_title(f'|Error| (t={t})', fontsize=11)
        
        fig.suptitle(f'{var_name}: {case_name}', fontsize=13)
        return axes.tolist()
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000//fps, blit=False)
    out_path = Path(output_dir) / f'{var_name.lower()}_{case_name}.gif'
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def create_timeseries_plot(seq_pred, seq_targ, case_name, output_dir):
    """Create per-variable RMSE over time plot."""
    max_steps = min(len(seq_pred), len(seq_targ))
    n_vars = seq_pred[0].shape[1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for v in range(min(n_vars, 4)):
        rmse_t = []
        for t in range(max_steps):
            r = np.sqrt(np.mean((seq_targ[t][:, v] - seq_pred[t][:, v]) ** 2))
            rmse_t.append(r)
        
        axes[v].plot(range(max_steps), rmse_t, 'b-', linewidth=1.5)
        axes[v].set_xlabel('Timestep')
        axes[v].set_ylabel('RMSE')
        axes[v].set_title(VAR_NAMES[v] if v < len(VAR_NAMES) else f'Var {v}')
        axes[v].grid(alpha=0.3)
    
    fig.suptitle(f'Per-Variable RMSE Over Time: {case_name}', fontsize=14)
    fig.tight_layout()
    out_path = Path(output_dir) / f'timeseries_{case_name}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def create_scatter_plots(seq_pred, seq_targ, case_name, output_dir, sample_steps=5):
    """Create pred vs target scatter plots at sampled timesteps."""
    max_steps = min(len(seq_pred), len(seq_targ))
    step_indices = np.linspace(0, max_steps - 1, sample_steps, dtype=int)
    n_vars = seq_pred[0].shape[1]
    
    fig, axes = plt.subplots(n_vars, sample_steps, figsize=(4 * sample_steps, 4 * n_vars))
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
            name = VAR_NAMES[vi] if vi < len(VAR_NAMES) else f'Var {vi}'
            ax.set_title(f'{name} t={t}\nR²={r2:.4f}', fontsize=9)
            ax.grid(alpha=0.3)
    
    fig.suptitle(f'Prediction vs Target: {case_name}', fontsize=14)
    fig.tight_layout()
    out_path = Path(output_dir) / f'scatter_{case_name}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ==============================================================================
# EVALUATOR CLASS
# ==============================================================================

class RiverEvaluator:
    """Evaluator for G-PARCv2 river models."""
    
    def __init__(self, model, device='cpu', extrema=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.var_names = VAR_NAMES
        self.simulation_metrics = []
        self.extrema = extrema  # For denormalization to physical units
        
        if extrema is not None:
            for v in range(min(len(VAR_NAMES), len(extrema['y_min']))):
                y_min = extrema['y_min'][v].item()
                y_max = extrema['y_max'][v].item()
                print(f"    {VAR_NAMES[v]}: [{y_min:.4f}, {y_max:.4f}]")
    
    def _prep_simulation(self, simulation):
        """Move to device, extract mesh_id."""
        simulation = [d.to(self.device) for d in simulation]
        
        # Ensure pos
        for d in simulation:
            if not hasattr(d, 'pos') or d.pos is None:
                d.pos = d.x[:, :2]
        
        mid = getattr(simulation[0], 'mesh_id', None)
        if mid is None:
            mesh_id_int = 0
        elif torch.is_tensor(mid):
            mesh_id_int = int(mid.view(-1)[0].item())
        else:
            mesh_id_int = int(mid)
        
        return simulation, mesh_id_int
    
    def _ensure_mesh_cached(self, initial_data, mesh_id_int):
        """Initialize MLS for this mesh if needed."""
        deriv = self.model.derivative_solver
        
        if getattr(deriv, '_active_mesh_id', None) == mesh_id_int:
            return
        
        if hasattr(deriv, 'initialize_weights'):
            deriv.initialize_weights(initial_data)
        
        deriv._active_mesh_id = mesh_id_int
    
    def generate_rollout(self, simulation, rollout_steps):
        """Autoregressive rollout predictions."""
        sf = self.model.num_static_feats
        df = self.model.num_dynamic_feats
        
        predictions = []
        F_prev = simulation[0].x[:, sf:sf + df].clone()
        
        for step in range(rollout_steps):
            data_t = simulation[step]
            static_feats = data_t.x[:, :sf]
            edge_index = data_t.edge_index
            
            if hasattr(data_t, 'mesh_id'):
                edge_index.mesh_id = data_t.mesh_id
            
            F_pred = self.model.step(
                static_feats=static_feats,
                dynamic_state=F_prev.clone(),
                edge_index=edge_index,
                dt=1.0,
            )
            
            predictions.append(F_pred)
            F_prev = F_pred
        
        return predictions
    
    def generate_snapshot_predictions(self, simulation, num_steps):
        """Single-step predictions from GT at each timestep."""
        sf = self.model.num_static_feats
        df = self.model.num_dynamic_feats
        predictions = []
        
        for step in range(num_steps):
            data_t = simulation[step]
            static_feats = data_t.x[:, :sf]
            edge_index = data_t.edge_index
            
            if hasattr(data_t, 'mesh_id'):
                edge_index.mesh_id = data_t.mesh_id
            
            F_gt = data_t.x[:, sf:sf + df].clone()
            
            F_pred = self.model.step(
                static_feats=static_feats,
                dynamic_state=F_gt,
                edge_index=edge_index,
                dt=1.0,
            )
            
            predictions.append(F_pred)
        
        return predictions
    
    def evaluate_rollout(self, simulations, rollout_steps=50):
        """Evaluate in rollout mode."""
        results = {
            'predictions': [], 'targets': [],
            'metadata': [], 'simulation_metrics': [],
        }
        self.simulation_metrics = []
        
        with torch.no_grad():
            for sim_idx, simulation in enumerate(tqdm(simulations, desc="Rollout")):
                try:
                    simulation, mesh_id = self._prep_simulation(simulation)
                    self._ensure_mesh_cached(simulation[0], mesh_id)
                    
                    actual_steps = min(rollout_steps, len(simulation) - 1)
                    preds_raw = self.generate_rollout(simulation, actual_steps)
                    
                    # Stability check
                    preds = []
                    for p in preds_raw:
                        if torch.isfinite(p).all() and p.abs().max() < 50.0:
                            preds.append(p.cpu().numpy())
                        else:
                            break
                    
                    if len(preds) == 0:
                        print(f"  Skipping sim {sim_idx}: immediate divergence")
                        continue
                    
                    df = self.model.num_dynamic_feats
                    targs = [simulation[t].y[:, :df].cpu().numpy() for t in range(len(preds))]
                    
                    # Denormalize to physical units if extrema available
                    preds_phys = [denormalize_all(p, self.extrema) for p in preds]
                    targs_phys = [denormalize_all(t, self.extrema) for t in targs]
                    
                    results['predictions'].append(preds_phys)
                    results['targets'].append(targs_phys)
                    
                    # Per-simulation metrics (in physical units)
                    all_p = np.concatenate(preds_phys, axis=0)
                    all_t = np.concatenate(targs_phys, axis=0)
                    rmse = float(np.sqrt(mean_squared_error(all_t, all_p)))
                    r2 = float(r2_score(all_t, all_p))
                    
                    sim_name = getattr(simulation[0], 'sim_name', f'simulation_{sim_idx}')
                    if not isinstance(sim_name, str):
                        sim_name = f'simulation_{sim_idx}'
                    
                    meta = {
                        'simulation_idx': sim_idx,
                        'case_name': sim_name,
                        'mesh_id': mesh_id,
                        'rollout_length': len(preds),
                        'num_nodes': simulation[0].num_nodes,
                    }
                    results['metadata'].append(meta)
                    
                    sim_metrics = {
                        'metadata': meta,
                        'overall': {'rmse': rmse, 'r2': r2},
                    }
                    self.simulation_metrics.append(sim_metrics)
                    results['simulation_metrics'].append(sim_metrics)
                    
                except Exception as e:
                    print(f"Error sim {sim_idx}: {e}")
                    import traceback; traceback.print_exc()
        
        return results
    
    def evaluate_snapshot(self, simulations):
        """Evaluate in snapshot mode (single-step from GT)."""
        results = {
            'predictions': [], 'targets': [],
            'metadata': [], 'simulation_metrics': [],
        }
        self.simulation_metrics = []
        
        with torch.no_grad():
            for sim_idx, simulation in enumerate(tqdm(simulations, desc="Snapshot")):
                try:
                    simulation, mesh_id = self._prep_simulation(simulation)
                    self._ensure_mesh_cached(simulation[0], mesh_id)
                    
                    num_steps = len(simulation) - 1
                    preds_raw = self.generate_snapshot_predictions(simulation, num_steps)
                    
                    preds, targs = [], []
                    df = self.model.num_dynamic_feats
                    
                    for t in range(num_steps):
                        p = preds_raw[t]
                        if torch.isfinite(p).all() and p.abs().max() < 50.0:
                            preds.append(p.cpu().numpy())
                            targs.append(simulation[t].y[:, :df].cpu().numpy())
                    
                    if len(preds) == 0:
                        continue
                    
                    # Denormalize to physical units if extrema available
                    preds_phys = [denormalize_all(p, self.extrema) for p in preds]
                    targs_phys = [denormalize_all(t, self.extrema) for t in targs]
                    
                    all_p = np.concatenate(preds_phys, axis=0)
                    all_t = np.concatenate(targs_phys, axis=0)
                    rmse = float(np.sqrt(mean_squared_error(all_t, all_p)))
                    r2 = float(r2_score(all_t, all_p))
                    
                    sim_name = getattr(simulation[0], 'sim_name', f'simulation_{sim_idx}')
                    if not isinstance(sim_name, str):
                        sim_name = f'simulation_{sim_idx}'
                    
                    meta = {
                        'simulation_idx': sim_idx,
                        'case_name': sim_name,
                        'mesh_id': mesh_id,
                        'num_snapshots': len(preds),
                        'num_nodes': simulation[0].num_nodes,
                    }
                    results['predictions'].append(preds_phys)
                    results['targets'].append(targs_phys)
                    results['metadata'].append(meta)
                    
                    sim_metrics = {'metadata': meta, 'overall': {'rmse': rmse, 'r2': r2}}
                    self.simulation_metrics.append(sim_metrics)
                    results['simulation_metrics'].append(sim_metrics)
                    
                except Exception as e:
                    print(f"Error sim {sim_idx}: {e}")
                    import traceback; traceback.print_exc()
        
        return results
    
    def compute_aggregate_metrics(self, results):
        """Compute aggregate RRMSE metrics."""
        all_pred, all_targ = [], []
        for seq_p, seq_t in zip(results['predictions'], results['targets']):
            for p, t in zip(seq_p, seq_t):
                all_pred.append(p)
                all_targ.append(t)
        
        rrmse_total = compute_rrmse(all_pred, all_targ)
        rrmse_per_var = compute_rrmse_per_variable(all_pred, all_targ, VAR_NAMES[:all_pred[0].shape[1]])
        
        return {
            'RRMSE_total': float(rrmse_total),
            **{f'RRMSE_{k}': float(v) for k, v in rrmse_per_var.items()},
            'n_simulations': len(results['predictions']),
            'n_total_samples': len(all_pred),
        }
    
    def create_visualizations(self, simulations, results, sim_idx, output_dir,
                               fps=5, frame_skip=1, eval_mode='rollout',
                               hec_ras_dir=None):
        """Create all visualizations for a simulation."""
        predictions = results['predictions'][sim_idx]
        targets = results['targets'][sim_idx]
        meta = results['metadata'][sim_idx]
        simulation = simulations[meta['simulation_idx']]
        case_name = meta.get('case_name', f'sim_{sim_idx}')
        
        pos = get_node_positions(simulation)
        n_vars = min(predictions[0].shape[1], len(VAR_NAMES))
        
        # Load HEC-RAS mesh polygons if available
        polys = None
        if hec_ras_dir is not None and HAS_H5PY:
            polys = load_mesh_for_sim(case_name, hec_ras_dir)
        
        print(f"\n  Creating visualizations for {case_name} ({eval_mode})...")
        render_mode = "PolyCollection" if polys else "scatter"
        print(f"    Rendering: {render_mode}")
        
        # GIF per variable
        for v in range(n_vars):
            path = create_scalar_field_gif(
                pos, predictions, targets, v, VAR_NAMES[v],
                case_name, output_dir, fps=fps, frame_skip=frame_skip,
                polys=polys, extrema=self.extrema,
            )
            print(f"    ✓ {path.name}")
        
        # Timeseries
        path = create_timeseries_plot(predictions, targets, case_name, output_dir)
        print(f"    ✓ {path.name}")
        
        # Scatter
        path = create_scatter_plots(predictions, targets, case_name, output_dir)
        print(f"    ✓ {path.name}")
    
    def plot_dashboard(self, results, eval_mode='rollout'):
        """Create comprehensive metrics dashboard."""
        if not self.simulation_metrics:
            return None
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        r2_scores = [m['overall']['r2'] for m in self.simulation_metrics]
        rmse_values = [m['overall']['rmse'] for m in self.simulation_metrics]
        mesh_ids = [m['metadata']['mesh_id'] for m in self.simulation_metrics]
        
        mode_label = eval_mode.capitalize()
        
        # R² histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(r2_scores, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(r2_scores), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(r2_scores):.4f}')
        ax1.set_xlabel('R²')
        ax1.set_title('R² Distribution')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        # RMSE histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(rmse_values, bins=15, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(rmse_values), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(rmse_values):.4e}')
        ax2.set_xlabel('RMSE')
        ax2.set_title('RMSE Distribution')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        
        # Stats text
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        stats_text = (
            f"{mode_label.upper()} METRICS\n"
            f"{'='*28}\n\n"
            f"R²:\n"
            f"  Mean:   {np.mean(r2_scores):.4f}\n"
            f"  Median: {np.median(r2_scores):.4f}\n"
            f"  Min:    {np.min(r2_scores):.4f}\n"
            f"  Max:    {np.max(r2_scores):.4f}\n\n"
            f"RMSE:\n"
            f"  Mean:   {np.mean(rmse_values):.4e}\n"
            f"  Median: {np.median(rmse_values):.4e}\n\n"
            f"Simulations: {len(self.simulation_metrics)}\n"
            f"Meshes: {len(set(mesh_ids))}"
        )
        ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # R² vs RMSE colored by mesh
        ax4 = fig.add_subplot(gs[1, 0])
        unique_meshes = sorted(set(mesh_ids))
        colors = ['steelblue', 'coral', 'green', 'purple']
        mesh_labels = {0: 'White River', 1: 'Iowa River'}
        for mi, mid in enumerate(unique_meshes):
            mask = [i for i, m in enumerate(mesh_ids) if m == mid]
            label = mesh_labels.get(mid, f'Mesh {mid}')
            ax4.scatter([r2_scores[i] for i in mask], [rmse_values[i] for i in mask],
                       c=colors[mi % len(colors)], s=80, alpha=0.7, edgecolors='black',
                       label=label)
        ax4.set_xlabel('R²')
        ax4.set_ylabel('RMSE')
        ax4.set_title('R² vs RMSE by Mesh')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        # Per-sim bar chart
        ax5 = fig.add_subplot(gs[1, 1:])
        sim_labels = [f"S{m['metadata']['simulation_idx']}" for m in self.simulation_metrics]
        bar_colors = ['green' if r2 > 0.8 else 'orange' if r2 > 0.5 else 'red' for r2 in r2_scores]
        ax5.bar(range(len(r2_scores)), r2_scores, color=bar_colors, edgecolor='black', alpha=0.7)
        ax5.set_xticks(range(len(sim_labels)))
        ax5.set_xticklabels(sim_labels, rotation=45, fontsize=8)
        ax5.set_ylabel('R²')
        ax5.set_title('R² by Simulation')
        ax5.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
        ax5.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (0.5)')
        ax5.legend(fontsize=8)
        ax5.grid(alpha=0.3, axis='y')
        
        fig.suptitle(f'River Model Performance ({mode_label})', fontsize=14, fontweight='bold')
        return fig


# ==============================================================================
# MAIN
# ==============================================================================

def load_test_simulations(test_dir, pattern, max_files):
    """Load test simulations with mesh_id from filename."""
    simulations = []
    paths = sorted(list(Path(test_dir).glob(pattern)))
    if max_files:
        paths = paths[:max_files]
    
    for idx, p in enumerate(paths):
        try:
            sim_data = torch.load(p, weights_only=False)
            
            # Determine mesh_id from filename
            name = p.stem.lower()
            sim_name = p.stem  # Preserve original case for HDF loading
            if 'iw' in name or 'iowa' in name:
                mesh_id = 1
            else:
                mesh_id = 0
            
            for data in sim_data:
                data.mesh_id = torch.tensor([mesh_id], dtype=torch.long)
                data.sim_name = sim_name
                if not hasattr(data, 'pos') or data.pos is None:
                    data.pos = data.x[:, :2]
            
            simulations.append(sim_data)
            print(f"  Loaded {p.name}: {len(sim_data)} steps, {sim_data[0].num_nodes} nodes, mesh_id={mesh_id}")
        except Exception as e:
            print(f"  Error loading {p}: {e}")
    
    return simulations


def select_representative(results, n_samples=3, mode='representative'):
    """Select simulations for visualization."""
    if not results.get('simulation_metrics'):
        return []
    
    sims = [(i, m['overall']['rmse']) for i, m in enumerate(results['simulation_metrics'])]
    sims.sort(key=lambda x: x[1])
    
    if mode == 'all':
        return [s[0] for s in sims]
    elif mode == 'best':
        return [s[0] for s in sims[:n_samples]]
    elif mode == 'worst':
        return [s[0] for s in sims[-n_samples:]]
    else:
        selected = []
        if len(sims) >= 1: selected.append(sims[0][0])
        if len(sims) >= 2: selected.append(sims[len(sims)//2][0])
        if len(sims) >= 3: selected.append(sims[-1][0])
        return selected[:n_samples]


def main():
    parser = argparse.ArgumentParser(description="G-PARCv2 River Evaluation")
    
    # Paths
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", default="./eval_river")
    
    # Eval mode
    parser.add_argument("--eval_mode", type=str, default="rollout",
                        choices=['rollout', 'snapshot', 'both'])
    
    # Architecture (must match training config!)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_layer_norm", action="store_true", default=True)
    parser.add_argument("--use_relative_pos", action="store_true", default=True)
    
    # Physics
    parser.add_argument("--integrator", type=str, default="euler")
    parser.add_argument("--spade_heads", type=int, default=4)
    parser.add_argument("--spade_concat", action="store_true", default=True)
    parser.add_argument("--spade_dropout", type=float, default=0.1)
    parser.add_argument("--zero_init", action="store_true", default=False)
    
    # Dimensions
    parser.add_argument("--num_static_feats", type=int, default=9)
    parser.add_argument("--num_dynamic_feats", type=int, default=4)
    parser.add_argument("--velocity_indices", type=int, nargs='+', default=[2, 3])
    
    # Eval settings
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--rollout_steps", type=int, default=50)
    parser.add_argument("--create_gifs", action="store_true")
    parser.add_argument("--num_viz_simulations", type=int, default=3)
    parser.add_argument("--viz_selection_mode", type=str, default="representative")
    parser.add_argument("--gif_fps", type=int, default=5)
    parser.add_argument("--gif_frame_skip", type=int, default=1)
    parser.add_argument("--hec_ras_dir", type=str, default=None,
                        help="Path to HEC-RAS base dir with .hdf mesh files "
                             "(e.g., /standard/sds_baek_energetic/HEC_RAS (River)). "
                             "Enables PolyCollection rendering instead of scatter plots.")
    parser.add_argument("--extrema_path", type=str, default=None,
                        help="Path to global_y_extrema.pth for denormalization to physical units")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # ==================== BUILD MODEL ====================
    print(f"\nBuilding G-PARCv2 River model...")
    print(f"  Model: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    gradient_solver = SolveGradientsLST()
    laplacian_solver = SolveWeightLST2d(use_2hop_extension=False)
    
    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=args.num_static_feats,
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layer_norm=args.use_layer_norm,
        use_relative_pos=args.use_relative_pos,
    )
    
    derivative_solver = RiverDifferentiator(
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=args.feature_out_channels,
        list_adv_idx=list(range(args.num_dynamic_feats)),
        list_dif_idx=list(range(args.num_dynamic_feats)),
        velocity_indices=args.velocity_indices,
        spade_random_noise=False,
        heads=args.spade_heads,
        concat=args.spade_concat,
        dropout=args.spade_dropout,
        zero_init=args.zero_init,
    )
    
    model = GPARC_River_V2(
        derivative_solver_physics=derivative_solver,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    
    # ==================== LOAD DATA ====================
    print(f"\nLoading test simulations from: {args.test_dir}")
    simulations = load_test_simulations(args.test_dir, "*.pt", args.max_sequences)
    if not simulations:
        print("No simulations found!")
        return
    print(f"Loaded {len(simulations)} simulations")
    
    # Init MLS
    first_sim = simulations[0][0].to(device)
    if not hasattr(first_sim, 'pos') or first_sim.pos is None:
        first_sim.pos = first_sim.x[:, :2]
    derivative_solver.initialize_weights(first_sim)
    
    # ==================== LOAD DENORMALIZATION ====================
    extrema = None
    if args.extrema_path:
        print(f"\nLoading denormalization extrema...")
        extrema = load_denorm_extrema(args.extrema_path)
    
    # ==================== EVALUATE ====================
    evaluator = RiverEvaluator(model, device, extrema=extrema)
    eval_mode = args.eval_mode.lower()
    
    # ---------- ROLLOUT ----------
    if eval_mode in ['rollout', 'both']:
        print(f"\n{'='*60}")
        print(f"ROLLOUT EVALUATION (steps={args.rollout_steps})")
        print(f"{'='*60}")
        
        rollout_results = evaluator.evaluate_rollout(simulations, rollout_steps=args.rollout_steps)
        rollout_metrics = evaluator.compute_aggregate_metrics(rollout_results)
        
        print(f"\n{'─'*40}")
        print("ROLLOUT RESULTS")
        print(f"{'─'*40}")
        for k, v in rollout_metrics.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        
        rollout_metrics['eval_mode'] = 'rollout'
        rollout_metrics['physical_units'] = args.extrema_path is not None
        with open(output_path / 'rollout_metrics.json', 'w') as f:
            json.dump(rollout_metrics, f, indent=2)
        
        fig = evaluator.plot_dashboard(rollout_results, eval_mode='rollout')
        if fig:
            fig.savefig(output_path / 'rollout_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        if args.create_gifs:
            selected = select_representative(rollout_results, args.num_viz_simulations, args.viz_selection_mode)
            for i, idx in enumerate(selected):
                evaluator.create_visualizations(
                    simulations, rollout_results, idx, output_path,
                    fps=args.gif_fps, frame_skip=args.gif_frame_skip, eval_mode='rollout',
                    hec_ras_dir=args.hec_ras_dir,
                )
    
    # ---------- SNAPSHOT ----------
    if eval_mode in ['snapshot', 'both']:
        print(f"\n{'='*60}")
        print(f"SNAPSHOT EVALUATION (single-step from GT)")
        print(f"{'='*60}")
        
        snapshot_results = evaluator.evaluate_snapshot(simulations)
        snapshot_metrics = evaluator.compute_aggregate_metrics(snapshot_results)
        
        print(f"\n{'─'*40}")
        print("SNAPSHOT RESULTS")
        print(f"{'─'*40}")
        for k, v in snapshot_metrics.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        
        snapshot_metrics['eval_mode'] = 'snapshot'
        snapshot_metrics['physical_units'] = args.extrema_path is not None
        with open(output_path / 'snapshot_metrics.json', 'w') as f:
            json.dump(snapshot_metrics, f, indent=2)
        
        fig = evaluator.plot_dashboard(snapshot_results, eval_mode='snapshot')
        if fig:
            fig.savefig(output_path / 'snapshot_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        if args.create_gifs:
            selected = select_representative(snapshot_results, args.num_viz_simulations, args.viz_selection_mode)
            for i, idx in enumerate(selected):
                evaluator.create_visualizations(
                    simulations, snapshot_results, idx, output_path,
                    fps=args.gif_fps, frame_skip=args.gif_frame_skip, eval_mode='snapshot',
                    hec_ras_dir=args.hec_ras_dir,
                )
    
    # ---------- COMPARISON ----------
    if eval_mode == 'both':
        print(f"\n{'='*60}")
        print("SNAPSHOT vs ROLLOUT COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Metric':<20} {'Snapshot':>12} {'Rollout':>12} {'Ratio':>10}")
        print(f"  {'─'*54}")
        for key in ['RRMSE_total'] + [f'RRMSE_{v}' for v in VAR_NAMES]:
            if key in snapshot_metrics and key in rollout_metrics:
                s, r = snapshot_metrics[key], rollout_metrics[key]
                ratio = r / s if s > 0 else float('inf')
                print(f"  {key:<20} {s:>12.6f} {r:>12.6f} {ratio:>10.1f}x")
        
        comparison = {'snapshot': snapshot_metrics, 'rollout': rollout_metrics}
        with open(output_path / 'comparison_metrics.json', 'w') as f:
            json.dump(comparison, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results in: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()