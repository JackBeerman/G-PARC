#!/usr/bin/env python3
"""
G-PARCv2 Shock Tube Evaluation Script
======================================
Evaluates models trained with scheduled sampling on shock tube physics.
Supports both ROLLOUT and SNAPSHOT evaluation modes.

Produces:
  - Per-simulation and aggregate metrics (RRMSE, R², RMSE, MAE)
  - Global parameter analysis: performance vs delta_t, pressure, density
  - 3D parameter-space visualizations
  - Rollout GIFs showing ground truth vs prediction for each variable
  - Error evolution GIFs
  - Comprehensive dashboards

Usage:
    python eval_shocktube_v2.py \\
        --model_path ./outputs_shocktube_v2/best_model.pth \\
        --test_dir /path/to/test \\
        --output_dir ./eval_shocktube \\
        --eval_mode rollout \\
        --rollout_steps 10 \\
        --create_gifs

    python eval_shocktube_v2.py \\
        --model_path best_model.pth \\
        --test_files sim_001.pt sim_002.pt \\
        --eval_mode both
"""

import argparse
import os
import sys
import re
import json
import warnings
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F_torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.featureextractor import GraphConvFeatureExtractorV2
from differentiator.shocktubedifferentiator import ShockTubeDifferentiator
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from models.shocktube_gparcv2 import GPARC_ShockTube_V2
from data.ShockChorddt import ShockTubeRolloutDataset, get_simulation_ids

warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# NORMALIZATION UTILITIES
# ==============================================================================

def load_normalization_metadata(data_dir):
    """Load normalization_metadata.json from data parent directory."""
    candidates = [
        Path(data_dir).parent / "normalization_metadata.json",
        Path(data_dir) / "normalization_metadata.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, 'r') as f:
                metadata = json.load(f)
            print(f"\n✓ Loaded normalization metadata from: {path}")
            return metadata
    print(f"\n⚠️  normalization_metadata.json not found near {data_dir}")
    return None


def load_normalization_metadata_from_checkpoint(checkpoint_dir):
    """Try loading normalization metadata from the model checkpoint directory."""
    path = Path(checkpoint_dir) / "normalization_metadata.json"
    if path.exists():
        with open(path, 'r') as f:
            metadata = json.load(f)
        print(f"\n✓ Loaded normalization metadata from checkpoint dir: {path}")
        return metadata
    return None


# ==============================================================================
# METRICS
# ==============================================================================

def compute_rrmse(predictions, references, valid_masks=None):
    """
    Compute Relative RMSE (RRMSE).

    RRMSE = sqrt( (1/n*) * Σ_i [ (1/N^i) * ||ref^i - pred^i||_2^2 / ||ref^i||_inf^2 ] )
    """
    if len(predictions) == 0:
        return 0.0

    n_samples = 0
    ratio_sum = 0.0

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if valid_masks is not None and valid_masks[i] is not None:
            pred = pred[valid_masks[i]]
            ref = ref[valid_masks[i]]

        if len(pred) == 0:
            continue

        n_nodes = pred.shape[0]
        ref_inf_norm = np.max(np.abs(ref))
        if ref_inf_norm == 0:
            continue

        mse = np.sum((pred - ref) ** 2) / n_nodes
        ratio_sum += mse / (ref_inf_norm ** 2)
        n_samples += 1

    if n_samples == 0:
        return float('inf')
    return float(np.sqrt(ratio_sum / n_samples))


def compute_rrmse_per_variable(predictions, references, var_names, valid_masks=None):
    """Compute RRMSE for each variable independently."""
    if len(predictions) == 0:
        return {v: 0.0 for v in var_names}

    n_vars = predictions[0].shape[1]
    rrmse_per_var = {}

    for vi in range(min(n_vars, len(var_names))):
        n_samples = 0
        ratio_sum = 0.0
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if valid_masks is not None and valid_masks[i] is not None:
                pred_c = pred[valid_masks[i], vi]
                ref_c = ref[valid_masks[i], vi]
            else:
                pred_c = pred[:, vi]
                ref_c = ref[:, vi]
            if len(pred_c) == 0:
                continue
            n_nodes = pred_c.shape[0]
            ref_inf = np.max(np.abs(ref_c))
            if ref_inf == 0:
                continue
            mse = np.sum((pred_c - ref_c) ** 2) / n_nodes
            ratio_sum += mse / (ref_inf ** 2)
            n_samples += 1
        rrmse_per_var[var_names[vi]] = float(np.sqrt(ratio_sum / n_samples)) if n_samples > 0 else float('inf')

    return rrmse_per_var


# ==============================================================================
# SELECTION HELPERS
# ==============================================================================

def select_representative_simulations(results_dict, n_samples=3, selection_mode='representative'):
    """Select simulations for visualization based on performance."""
    if 'simulation_metrics' not in results_dict or not results_dict['simulation_metrics']:
        return []

    sims = [(m['metadata']['simulation_idx'], m['overall_physical']['rmse'])
            for m in results_dict['simulation_metrics']]
    sims.sort(key=lambda x: x[1])

    if selection_mode == 'all':
        return [s[0] for s in sims]
    elif selection_mode == 'best':
        return [s[0] for s in sims[:n_samples]]
    elif selection_mode == 'worst':
        return [s[0] for s in sims[-n_samples:]]
    else:  # representative
        selected = []
        if len(sims) >= 1:
            selected.append(sims[0][0])
        if len(sims) >= 2:
            selected.append(sims[len(sims) // 2][0])
        if len(sims) >= 3:
            selected.append(sims[-1][0])
        return selected[:n_samples]


# ==============================================================================
# GIF HELPERS
# ==============================================================================

def _create_variable_comparison_gif(frames, seq_pred, seq_targ, var_idx, var_name,
                                     grid_size, vmin, vmax, case_name, output_dir,
                                     fps, eval_mode='rollout'):
    """Create side-by-side target vs prediction GIF for one variable."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(right=0.92)

    mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps.get_cmap('coolwarm')

    err_max = 0
    for f in frames:
        err = np.abs(seq_targ[f][:, var_idx] - seq_pred[f][:, var_idx])
        err_max = max(err_max, err.max())
    err_norm = Normalize(vmin=0, vmax=max(err_max, 1e-12))
    err_cmap = plt.colormaps.get_cmap('hot')

    def animate(frame_idx):
        frame = frames[frame_idx]
        for ax in axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        targ_2d = seq_targ[frame][:, var_idx].reshape(grid_size, grid_size)
        pred_2d = seq_pred[frame][:, var_idx].reshape(grid_size, grid_size)
        err_2d = np.abs(targ_2d - pred_2d)

        axes[0].imshow(targ_2d, cmap=cmap, norm=norm, aspect='auto', origin='lower')
        axes[0].set_title(f'Target (t={frame})', fontsize=12)

        axes[1].imshow(pred_2d, cmap=cmap, norm=norm, aspect='auto', origin='lower')
        axes[1].set_title(f'Prediction (t={frame})', fontsize=12)

        axes[2].imshow(err_2d, cmap=err_cmap, norm=err_norm, aspect='auto', origin='lower')
        axes[2].set_title(f'|Error| (t={frame})', fontsize=12)

        fig.suptitle(f'{var_name} ({mode_label}): {case_name}', fontsize=14)
        return axes.tolist()

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps, blit=False)
    out_path = Path(output_dir) / f'{var_name}_{case_name}.gif'
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"    ✓ {out_path.name}")


def _create_all_variables_gif(frames, seq_pred, seq_targ, var_names, grid_size,
                               vranges, case_name, output_dir, fps, eval_mode='rollout'):
    """Create combined GIF showing all variables at once."""
    n_vars = len(var_names)
    fig, axes = plt.subplots(2, n_vars, figsize=(6 * n_vars, 10))
    if n_vars == 1:
        axes = axes.reshape(2, 1)

    mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'
    cmaps = [plt.colormaps.get_cmap('coolwarm')] * n_vars
    norms = [Normalize(vmin=vr[0], vmax=vr[1]) for vr in vranges]

    def animate(frame_idx):
        frame = frames[frame_idx]
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])

        for vi, vn in enumerate(var_names):
            targ_2d = seq_targ[frame][:, vi].reshape(grid_size, grid_size)
            pred_2d = seq_pred[frame][:, vi].reshape(grid_size, grid_size)

            axes[0, vi].imshow(targ_2d, cmap=cmaps[vi], norm=norms[vi], aspect='auto', origin='lower')
            axes[0, vi].set_title(f'{vn} Target', fontsize=10)
            axes[1, vi].imshow(pred_2d, cmap=cmaps[vi], norm=norms[vi], aspect='auto', origin='lower')
            axes[1, vi].set_title(f'{vn} Pred', fontsize=10)

        fig.suptitle(f'All Variables ({mode_label}) t={frame}: {case_name}', fontsize=14)
        return [ax for row in axes for ax in row]

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps, blit=False)
    out_path = Path(output_dir) / f'all_vars_{case_name}.gif'
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"    ✓ {out_path.name}")


def _create_error_evolution_gif(frames, seq_pred, seq_targ, var_names, grid_size,
                                 case_name, output_dir, fps, eval_mode='rollout'):
    """Create GIF showing error magnitude evolving over time."""
    n_vars = len(var_names)
    fig, axes = plt.subplots(1, n_vars, figsize=(6 * n_vars, 5))
    if n_vars == 1:
        axes = [axes]

    mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'
    cmap = plt.colormaps.get_cmap('hot')

    # Compute max error across all frames for consistent colorbar
    max_errors = [0.0] * n_vars
    for f in frames:
        for vi in range(n_vars):
            err = np.abs(seq_targ[f][:, vi] - seq_pred[f][:, vi])
            max_errors[vi] = max(max_errors[vi], err.max())
    norms = [Normalize(vmin=0, vmax=max(me, 1e-12)) for me in max_errors]

    def animate(frame_idx):
        frame = frames[frame_idx]
        for ax in axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        for vi, vn in enumerate(var_names):
            err_2d = np.abs(seq_targ[frame][:, vi] - seq_pred[frame][:, vi]).reshape(grid_size, grid_size)
            axes[vi].imshow(err_2d, cmap=cmap, norm=norms[vi], aspect='auto', origin='lower')
            axes[vi].set_title(f'{vn} |Error|', fontsize=10)

        fig.suptitle(f'Error Evolution ({mode_label}) t={frame}: {case_name}', fontsize=14)
        return list(axes)

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000 // fps, blit=False)
    out_path = Path(output_dir) / f'error_evolution_{case_name}.gif'
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"    ✓ {out_path.name}")


# ==============================================================================
# EVALUATOR CLASS
# ==============================================================================

class ShockTubeEvaluator:
    """Evaluator for G-PARCv2 shock tube models."""

    # Shock tube dynamic variables (after skipping y_momentum)
    VAR_NAMES = ['density', 'x_momentum', 'total_energy']

    def __init__(self, model, device='cpu', norm_metadata=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.norm_metadata = norm_metadata
        self.simulation_metrics = []
        self.var_names = self.VAR_NAMES

        # Pre-extract denorm parameters
        self.denorm_params = {}
        self.delta_t_denorm = None
        self.pressure_denorm = None
        self.density_denorm = None
        if norm_metadata:
            np_section = norm_metadata.get('normalization_params', {})
            for vn in self.var_names:
                if vn in np_section:
                    self.denorm_params[vn] = np_section[vn]

            gp_section = norm_metadata.get('global_param_normalization', {})
            if 'delta_t' in gp_section:
                self.delta_t_denorm = gp_section['delta_t']
            if 'pressure' in gp_section:
                self.pressure_denorm = gp_section['pressure']
            if 'density' in gp_section:
                self.density_denorm = gp_section['density']

            # Fallback: check normalization_params too
            if self.delta_t_denorm is None and 'delta_t' in np_section:
                self.delta_t_denorm = np_section['delta_t']
            if self.delta_t_denorm is None and 'global_delta_t' in np_section:
                self.delta_t_denorm = np_section['global_delta_t']

            print(f"  Denorm params loaded for variables: {list(self.denorm_params.keys())}")
            print(f"  Delta_t denorm: {'yes' if self.delta_t_denorm else 'no'}")
            print(f"  Pressure denorm: {'yes' if self.pressure_denorm else 'no'}")
            print(f"  Density denorm: {'yes' if self.density_denorm else 'no'}")

    # ------------------------------------------------------------------
    # Denormalization helpers
    # ------------------------------------------------------------------
    def _denorm_minmax(self, val, params):
        if params is None:
            return val
        return val * (params['max'] - params['min']) + params['min']

    def denorm_variable(self, data, var_idx):
        """Denormalize a single variable array [N] or [N,1]."""
        vn = self.var_names[var_idx]
        if vn not in self.denorm_params:
            return data
        p = self.denorm_params[vn]
        return data * (p['max'] - p['min']) + p['min']

    def denorm_all(self, data_np):
        """Denormalize all variables in [N, n_vars] array."""
        out = np.copy(data_np)
        for vi in range(min(out.shape[1], len(self.var_names))):
            out[:, vi] = self.denorm_variable(out[:, vi], vi)
        return out

    def denorm_delta_t(self, norm_val):
        return self._denorm_minmax(norm_val, self.delta_t_denorm)

    def denorm_pressure(self, norm_val):
        return self._denorm_minmax(norm_val, self.pressure_denorm)

    def denorm_density(self, norm_val):
        return self._denorm_minmax(norm_val, self.density_denorm)

    # ------------------------------------------------------------------
    # Simulation preparation
    # ------------------------------------------------------------------
    def _prep_simulation(self, simulation):
        simulation = [d.to(self.device) for d in simulation]
        for data in simulation:
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :self.model.num_static_feats]
            # Ensure global attributes are extracted from global_params
            if hasattr(data, 'global_params') and not hasattr(data, 'global_pressure'):
                gp = data.global_params
                if gp.numel() >= 3:
                    data.global_pressure = gp[0].unsqueeze(0)
                    data.global_density = gp[1].unsqueeze(0)
                    data.global_delta_t = gp[2].unsqueeze(0)
        return simulation

    def _ensure_mesh_cached(self, initial_data):
        """Reinitialise MLS weights for the current mesh."""
        deriv = self.model.derivative_solver
        real = deriv.solver if hasattr(deriv, 'solver') else deriv
        if hasattr(real, 'initialize_weights'):
            real.initialize_weights(initial_data)

    # ------------------------------------------------------------------
    # Prediction generation
    # ------------------------------------------------------------------
    def generate_rollout(self, simulation, rollout_steps):
        """Autoregressive rollout from ground truth t=0."""
        predictions = []
        first_data = simulation[0]

        # Compute global embedding (once, shared across all steps)
        global_attrs = self.model._extract_global_attrs(first_data)
        global_embed = self.model.global_processor(global_attrs)

        # Use delta_t from global params for integration
        dt = first_data.global_delta_t.flatten()[0].item()

        # Initial dynamic state with skip logic
        F_prev = self.model._extract_dynamic(first_data.x)

        for step in range(rollout_steps):
            data_t = simulation[step]
            static_feats = data_t.x[:, :self.model.num_static_feats]
            edge_index = data_t.edge_index

            F_pred = self.model.step(
                static_feats=static_feats,
                dynamic_state=F_prev.clone(),
                edge_index=edge_index,
                global_embed=global_embed,
                dt=dt,
            )
            predictions.append(F_pred)
            F_prev = F_pred

        return predictions

    def generate_snapshot_predictions(self, simulation, num_steps):
        """Single-step from GT at each timestep (no error accumulation)."""
        predictions = []
        first_data = simulation[0]

        # Compute global embedding (once, shared across all steps)
        global_attrs = self.model._extract_global_attrs(first_data)
        global_embed = self.model.global_processor(global_attrs)
        dt = first_data.global_delta_t.flatten()[0].item()

        for step in range(num_steps):
            data_t = simulation[step]
            static_feats = data_t.x[:, :self.model.num_static_feats]
            edge_index = data_t.edge_index

            # Use GROUND TRUTH dynamic state at time t
            F_gt = self.model._extract_dynamic(data_t.x)

            F_pred = self.model.step(
                static_feats=static_feats,
                dynamic_state=F_gt,
                edge_index=edge_index,
                global_embed=global_embed,
                dt=dt,
            )
            predictions.append(F_pred)
        return predictions

    # ------------------------------------------------------------------
    # Main evaluation routines
    # ------------------------------------------------------------------
    def evaluate_rollout_predictions(self, simulations, rollout_steps=10):
        """Evaluate in rollout mode. Returns results dict."""
        results = {
            'predictions_norm': [],
            'targets_norm': [],
            'predictions_physical': [],
            'targets_physical': [],
            'metadata': [],
        }
        self.simulation_metrics = []

        with torch.no_grad():
            for sim_idx, simulation in enumerate(tqdm(simulations, desc="Rollout predictions")):
                try:
                    simulation = self._prep_simulation(simulation)
                    initial_data = simulation[0]
                    self._ensure_mesh_cached(initial_data)

                    actual_steps = min(rollout_steps, len(simulation))
                    preds_raw = self.generate_rollout(simulation, actual_steps)

                    # Stability filter
                    preds_norm = []
                    for p in preds_raw:
                        if torch.isfinite(p).all() and p.abs().max() < 100.0:
                            preds_norm.append(p.cpu().numpy())
                        else:
                            break

                    if not preds_norm:
                        print(f"  Skipping sim {sim_idx}: immediate instability")
                        continue

                    actual_steps = len(preds_norm)
                    targs_norm = [self.model.process_targets(simulation[t].y).cpu().numpy()
                                  for t in range(actual_steps)]

                    preds_phys = [self.denorm_all(p) for p in preds_norm]
                    targs_phys = [self.denorm_all(t) for t in targs_norm]

                    # Extract global params (normalized)
                    norm_dt = float(initial_data.global_delta_t[0]) if hasattr(initial_data, 'global_delta_t') else 0.0
                    norm_pr = float(initial_data.global_pressure[0]) if hasattr(initial_data, 'global_pressure') else 0.0
                    norm_rho = float(initial_data.global_density[0]) if hasattr(initial_data, 'global_density') else 0.0

                    phys_dt = float(self.denorm_delta_t(norm_dt))
                    phys_pr = float(self.denorm_pressure(norm_pr))
                    phys_rho = float(self.denorm_density(norm_rho))

                    metadata = {
                        'simulation_idx': sim_idx,
                        'case_name': f'sim_{sim_idx}',
                        'rollout_length': actual_steps,
                        'num_nodes': initial_data.num_nodes,
                        'delta_t_norm': norm_dt,
                        'pressure_norm': norm_pr,
                        'density_norm': norm_rho,
                        'delta_t': phys_dt,
                        'pressure': phys_pr,
                        'density': phys_rho,
                    }
                    results['predictions_norm'].append(preds_norm)
                    results['targets_norm'].append(targs_norm)
                    results['predictions_physical'].append(preds_phys)
                    results['targets_physical'].append(targs_phys)
                    results['metadata'].append(metadata)

                    # Per-sim aggregate metrics (physical)
                    all_p = np.concatenate(preds_phys, axis=0)
                    all_t = np.concatenate(targs_phys, axis=0)
                    rmse = float(np.sqrt(mean_squared_error(all_t, all_p)))
                    r2 = float(r2_score(all_t.flatten(), all_p.flatten()))

                    per_var = {}
                    for vi, vn in enumerate(self.var_names):
                        pv = all_p[:, vi]
                        tv = all_t[:, vi]
                        per_var[vn] = {
                            'rmse': float(np.sqrt(mean_squared_error(tv, pv))),
                            'r2': float(r2_score(tv, pv)),
                            'mae': float(mean_absolute_error(tv, pv)),
                        }

                    self.simulation_metrics.append({
                        'metadata': metadata,
                        'overall_physical': {'rmse': rmse, 'r2': r2},
                        'per_variable': per_var,
                    })

                except Exception as e:
                    print(f"Error in sim {sim_idx}: {e}")
                    import traceback
                    traceback.print_exc()

        results['simulation_metrics'] = self.simulation_metrics
        return results

    def evaluate_snapshot_predictions(self, simulations):
        """Evaluate in snapshot (single-step) mode."""
        results = {
            'predictions_norm': [],
            'targets_norm': [],
            'predictions_physical': [],
            'targets_physical': [],
            'metadata': [],
        }
        self.simulation_metrics = []

        with torch.no_grad():
            for sim_idx, simulation in enumerate(tqdm(simulations, desc="Snapshot predictions")):
                try:
                    simulation = self._prep_simulation(simulation)
                    initial_data = simulation[0]
                    self._ensure_mesh_cached(initial_data)

                    num_steps = len(simulation) - 1
                    preds_raw = self.generate_snapshot_predictions(simulation, num_steps)

                    preds_norm, targs_norm = [], []
                    for t in range(num_steps):
                        pred = preds_raw[t]
                        if not (torch.isfinite(pred).all() and pred.abs().max() < 100.0):
                            continue
                        preds_norm.append(pred.cpu().numpy())
                        targs_norm.append(self.model.process_targets(simulation[t].y).cpu().numpy())

                    if not preds_norm:
                        continue

                    preds_phys = [self.denorm_all(p) for p in preds_norm]
                    targs_phys = [self.denorm_all(t) for t in targs_norm]

                    norm_dt = float(initial_data.global_delta_t[0]) if hasattr(initial_data, 'global_delta_t') else 0.0
                    norm_pr = float(initial_data.global_pressure[0]) if hasattr(initial_data, 'global_pressure') else 0.0
                    norm_rho = float(initial_data.global_density[0]) if hasattr(initial_data, 'global_density') else 0.0

                    metadata = {
                        'simulation_idx': sim_idx,
                        'case_name': f'sim_{sim_idx}',
                        'num_snapshots': len(preds_norm),
                        'num_nodes': initial_data.num_nodes,
                        'delta_t_norm': norm_dt,
                        'pressure_norm': norm_pr,
                        'density_norm': norm_rho,
                        'delta_t': float(self.denorm_delta_t(norm_dt)),
                        'pressure': float(self.denorm_pressure(norm_pr)),
                        'density': float(self.denorm_density(norm_rho)),
                    }

                    results['predictions_norm'].append(preds_norm)
                    results['targets_norm'].append(targs_norm)
                    results['predictions_physical'].append(preds_phys)
                    results['targets_physical'].append(targs_phys)
                    results['metadata'].append(metadata)

                    all_p = np.concatenate(preds_phys, axis=0)
                    all_t = np.concatenate(targs_phys, axis=0)
                    rmse = float(np.sqrt(mean_squared_error(all_t, all_p)))
                    r2 = float(r2_score(all_t.flatten(), all_p.flatten()))

                    per_var = {}
                    for vi, vn in enumerate(self.var_names):
                        pv, tv = all_p[:, vi], all_t[:, vi]
                        per_var[vn] = {
                            'rmse': float(np.sqrt(mean_squared_error(tv, pv))),
                            'r2': float(r2_score(tv, pv)),
                            'mae': float(mean_absolute_error(tv, pv)),
                        }

                    self.simulation_metrics.append({
                        'metadata': metadata,
                        'overall_physical': {'rmse': rmse, 'r2': r2},
                        'per_variable': per_var,
                    })

                except Exception as e:
                    print(f"Error in snapshot sim {sim_idx}: {e}")
                    import traceback
                    traceback.print_exc()

        results['simulation_metrics'] = self.simulation_metrics
        return results

    # ------------------------------------------------------------------
    # Benchmark metrics
    # ------------------------------------------------------------------
    def compute_benchmark_metrics(self, preds_phys, targs_phys):
        """Compute RRMSE and standard metrics."""
        if not preds_phys:
            return {}

        all_pred, all_targ = [], []
        for sp, st in zip(preds_phys, targs_phys):
            for p, t in zip(sp, st):
                all_pred.append(p)
                all_targ.append(t)

        rrmse_total = compute_rrmse(all_pred, all_targ)
        rrmse_per_var = compute_rrmse_per_variable(all_pred, all_targ, self.var_names)

        all_p = np.concatenate(all_pred, axis=0)
        all_t = np.concatenate(all_targ, axis=0)
        overall_rmse = float(np.sqrt(mean_squared_error(all_t, all_p)))
        overall_r2 = float(r2_score(all_t.flatten(), all_p.flatten()))
        overall_mae = float(mean_absolute_error(all_t, all_p))

        metrics = {
            'RRMSE_total': rrmse_total,
            'overall_RMSE': overall_rmse,
            'overall_R2': overall_r2,
            'overall_MAE': overall_mae,
        }
        for vn in self.var_names:
            metrics[f'RRMSE_{vn}'] = rrmse_per_var.get(vn, float('inf'))

        return metrics

    # ------------------------------------------------------------------
    # GIF creation
    # ------------------------------------------------------------------
    def create_gifs(self, simulations, results_dict, seq_idx, output_dir,
                    fps=4, frame_skip=1, eval_mode='rollout'):
        """Create all GIF types for a given simulation."""
        preds = results_dict['predictions_physical']
        targs = results_dict['targets_physical']
        meta = results_dict['metadata']

        if seq_idx >= len(preds):
            return

        seq_pred = preds[seq_idx]
        seq_targ = targs[seq_idx]
        case_name = meta[seq_idx].get('case_name', f'sim_{seq_idx}')

        max_steps = min(len(seq_pred), len(seq_targ))
        if max_steps < 2:
            return

        n_nodes = seq_pred[0].shape[0]
        grid_size = int(np.sqrt(n_nodes))
        if grid_size * grid_size != n_nodes:
            print(f"  ⚠️  Non-square grid ({n_nodes} nodes) for {case_name}, skipping GIFs")
            return

        frames = list(range(0, max_steps, frame_skip))

        # Pre-compute value ranges per variable
        vranges = []
        for vi in range(len(self.var_names)):
            vmin = min(seq_targ[f][:, vi].min() for f in frames)
            vmax = max(seq_targ[f][:, vi].max() for f in frames)
            pmin = min(seq_pred[f][:, vi].min() for f in frames)
            pmax = max(seq_pred[f][:, vi].max() for f in frames)
            vranges.append((min(vmin, pmin), max(vmax, pmax)))

        print(f"\n  Creating GIFs for {case_name} ({eval_mode}), {max_steps} steps, grid {grid_size}²")

        # Per-variable GIFs
        for vi, vn in enumerate(self.var_names):
            _create_variable_comparison_gif(
                frames, seq_pred, seq_targ, vi, vn, grid_size,
                vranges[vi][0], vranges[vi][1],
                case_name, output_dir, fps, eval_mode,
            )

        # Combined all-vars GIF
        _create_all_variables_gif(
            frames, seq_pred, seq_targ, self.var_names, grid_size,
            vranges, case_name, output_dir, fps, eval_mode,
        )

        # Error evolution GIF
        _create_error_evolution_gif(
            frames, seq_pred, seq_targ, self.var_names, grid_size,
            case_name, output_dir, fps, eval_mode,
        )

    # ------------------------------------------------------------------
    # Comprehensive dashboard
    # ------------------------------------------------------------------
    def plot_comprehensive_metrics(self, results_dict, figsize=(20, 14), eval_mode='rollout'):
        """Create metrics dashboard mirroring the elastoplastic evaluator."""
        if not self.simulation_metrics:
            return None

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
        mode_label = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'

        r2_vals = [m['overall_physical']['r2'] for m in self.simulation_metrics]
        rmse_vals = [m['overall_physical']['rmse'] for m in self.simulation_metrics]
        sim_ids = [m['metadata']['simulation_idx'] for m in self.simulation_metrics]

        # (0,0) R² histogram
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(r2_vals, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(r2_vals), color='red', ls='--', lw=2, label=f'Mean: {np.mean(r2_vals):.3f}')
        ax.set_xlabel('R²'); ax.set_ylabel('Count')
        ax.set_title('R² Distribution', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # (0,1) RMSE histogram
        ax = fig.add_subplot(gs[0, 1])
        ax.hist(rmse_vals, bins=15, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(rmse_vals), color='red', ls='--', lw=2, label=f'Mean: {np.mean(rmse_vals):.3e}')
        ax.set_xlabel('RMSE'); ax.set_ylabel('Count')
        ax.set_title('RMSE Distribution', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # (0,2) Summary text
        ax = fig.add_subplot(gs[0, 2])
        ax.axis('off')
        txt = (
            f"{mode_label.upper()} SUMMARY\n{'='*28}\n"
            f"Simulations: {len(self.simulation_metrics)}\n\n"
            f"R²:\n  Mean:   {np.mean(r2_vals):.4f}\n  Median: {np.median(r2_vals):.4f}\n"
            f"  Min:    {np.min(r2_vals):.4f}\n  Max:    {np.max(r2_vals):.4f}\n\n"
            f"RMSE:\n  Mean:   {np.mean(rmse_vals):.3e}\n  Median: {np.median(rmse_vals):.3e}"
        )
        ax.text(0.05, 0.5, txt, fontsize=10, family='monospace', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # (1, :2) R² vs RMSE scatter colored by delta_t
        delta_ts = [m['metadata'].get('delta_t', 0) for m in self.simulation_metrics]
        ax = fig.add_subplot(gs[1, :2])
        sc = ax.scatter(r2_vals, rmse_vals, c=delta_ts, cmap='viridis', s=80, alpha=0.7, edgecolors='k', lw=0.5)
        ax.set_xlabel('R²'); ax.set_ylabel('RMSE')
        ax.set_title('R² vs RMSE (colored by Δt)', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax).set_label('Δt (physical)')

        # (1, 2) Per-simulation bar chart
        ax = fig.add_subplot(gs[1, 2])
        colors = ['green' if r > 0.8 else 'orange' if r > 0.5 else 'red' for r in r2_vals]
        ax.barh(range(len(sim_ids)), r2_vals, color=colors, edgecolor='k', alpha=0.7)
        ax.set_xlabel('R²'); ax.set_ylabel('Sim Index')
        ax.set_title('Performance by Sim', fontweight='bold')
        ax.axvline(0.8, color='green', ls='--', alpha=0.5)
        ax.axvline(0.5, color='orange', ls='--', alpha=0.5)
        ax.grid(alpha=0.3, axis='x')

        # (2, 0) Per-variable R² boxplot
        ax = fig.add_subplot(gs[2, 0])
        var_r2_data = {vn: [] for vn in self.var_names}
        for m in self.simulation_metrics:
            for vn in self.var_names:
                if vn in m.get('per_variable', {}):
                    var_r2_data[vn].append(m['per_variable'][vn]['r2'])
        bp_data = [var_r2_data[vn] for vn in self.var_names if var_r2_data[vn]]
        bp_labels = [vn for vn in self.var_names if var_r2_data[vn]]
        if bp_data:
            ax.boxplot(bp_data, labels=bp_labels)
        ax.set_ylabel('R²')
        ax.set_title('Per-Variable R²', fontweight='bold')
        ax.grid(alpha=0.3)

        # (2, 1) RMSE timeline
        ax = fig.add_subplot(gs[2, 1])
        ax.plot(range(len(rmse_vals)), rmse_vals, 'o-', color='coral', ms=5, lw=1.5)
        ax.axhline(np.mean(rmse_vals), color='red', ls='--', alpha=0.5, label='Mean')
        ax.set_xlabel('Sim Index'); ax.set_ylabel('RMSE')
        ax.set_title('RMSE by Simulation', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # (2, 2) Best / Worst table
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')
        sorted_idx = np.argsort(r2_vals)
        best_3 = sorted_idx[-3:][::-1]
        worst_3 = sorted_idx[:3]
        txt = f"BEST\n{'='*20}\n"
        for rank, i in enumerate(best_3, 1):
            txt += f"{rank}. Sim {sim_ids[i]}: R²={r2_vals[i]:.4f}\n"
        txt += f"\nWORST\n{'='*20}\n"
        for rank, i in enumerate(worst_3, 1):
            txt += f"{rank}. Sim {sim_ids[i]}: R²={r2_vals[i]:.4f}\n"
        ax.text(0.05, 0.5, txt, fontsize=10, family='monospace', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        fig.suptitle(f'G-PARCv2 Shock Tube — {mode_label} Performance', fontsize=16, fontweight='bold', y=0.99)
        return fig

    # ------------------------------------------------------------------
    # Global parameter analysis
    # ------------------------------------------------------------------
    def plot_global_parameter_analysis(self, figsize=(22, 18)):
        """Comprehensive analysis of performance vs global parameters."""
        if not self.simulation_metrics:
            return None

        df = pd.DataFrame([
            {
                'delta_t': m['metadata']['delta_t'],
                'pressure': m['metadata']['pressure'],
                'density': m['metadata']['density'],
                'overall_r2': m['overall_physical']['r2'],
                'overall_rmse': m['overall_physical']['rmse'],
                **{f'{vn}_r2': m['per_variable'][vn]['r2'] for vn in self.var_names if vn in m.get('per_variable', {})},
                **{f'{vn}_rmse': m['per_variable'][vn]['rmse'] for vn in self.var_names if vn in m.get('per_variable', {})},
            }
            for m in self.simulation_metrics
        ])

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)
        fig.suptitle('Performance vs Global Parameters (Δt, Pressure, Density)', fontsize=16, fontweight='bold')

        params = ['delta_t', 'pressure', 'density']
        param_labels = ['Δt (s)', 'Pressure', 'Density']

        # Row 0: Overall R² vs each parameter
        for col, (param, plabel) in enumerate(zip(params, param_labels)):
            ax = fig.add_subplot(gs[0, col])
            sc = ax.scatter(df[param], df['overall_r2'], c=df['overall_rmse'],
                           cmap='plasma_r', s=60, alpha=0.8, edgecolors='k', lw=0.3)
            ax.set_xlabel(plabel); ax.set_ylabel('R²')
            ax.set_title(f'R² vs {plabel}', fontweight='bold')
            ax.grid(alpha=0.3)
            plt.colorbar(sc, ax=ax, label='RMSE')
            if param == 'delta_t':
                ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))

        # Row 1: Overall RMSE vs each parameter
        for col, (param, plabel) in enumerate(zip(params, param_labels)):
            ax = fig.add_subplot(gs[1, col])
            ax.scatter(df[param], df['overall_rmse'], c='coral', s=60, alpha=0.7, edgecolors='k', lw=0.3)
            ax.set_xlabel(plabel); ax.set_ylabel('RMSE')
            ax.set_title(f'RMSE vs {plabel}', fontweight='bold')
            ax.grid(alpha=0.3)
            if param == 'delta_t':
                ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))

        # Row 2: Per-variable R² vs delta_t
        ax = fig.add_subplot(gs[2, 0])
        markers = ['o', 's', '^']
        colors_var = plt.cm.Set1(np.linspace(0, 0.6, len(self.var_names)))
        for vi, vn in enumerate(self.var_names):
            col_name = f'{vn}_r2'
            if col_name in df.columns:
                ax.scatter(df['delta_t'], df[col_name], marker=markers[vi % 3],
                          color=colors_var[vi], label=vn, s=50, alpha=0.7)
        ax.set_xlabel('Δt (s)'); ax.set_ylabel('R²')
        ax.set_title('Per-Variable R² vs Δt', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))

        # Row 2, col 1: Pressure vs Density colored by R²
        ax = fig.add_subplot(gs[2, 1])
        sc = ax.scatter(df['pressure'], df['density'], c=df['overall_r2'],
                       cmap='viridis', s=60, alpha=0.8, edgecolors='k', lw=0.3)
        ax.set_xlabel('Pressure'); ax.set_ylabel('Density')
        ax.set_title('R² in Pressure–Density Space', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label='R²')

        # Row 2, col 2: Correlation heatmap
        ax = fig.add_subplot(gs[2, 2])
        corr_cols = params + ['overall_r2', 'overall_rmse']
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr = df[corr_cols].corr()
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)
        ax.set_title('Correlation Matrix', fontweight='bold')
        # Annotate
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax)

        # Row 3: 3D scatter (if enough variation)
        try:
            from mpl_toolkits.mplot3d import Axes3D
            ax3d = fig.add_subplot(gs[3, 0], projection='3d')
            sc3d = ax3d.scatter(df['delta_t'], df['pressure'], df['density'],
                               c=df['overall_r2'], cmap='viridis', s=50, alpha=0.8)
            ax3d.set_xlabel('Δt'); ax3d.set_ylabel('Pressure'); ax3d.set_zlabel('Density')
            ax3d.set_title('R² in 3D Param Space', fontweight='bold')
            plt.colorbar(sc3d, ax=ax3d, shrink=0.6, label='R²')
        except Exception:
            ax3d = fig.add_subplot(gs[3, 0])
            ax3d.axis('off')
            ax3d.text(0.5, 0.5, '3D plot unavailable', ha='center', va='center')

        # Row 3, col 1: Aggregated delta_t performance table as bar chart
        ax = fig.add_subplot(gs[3, 1])
        dt_groups = df.groupby(df['delta_t'].round(8))
        dt_means = dt_groups['overall_r2'].mean()
        dt_stds = dt_groups['overall_r2'].std().fillna(0)
        dt_counts = dt_groups['overall_r2'].count()

        x_pos = np.arange(len(dt_means))
        bars = ax.bar(x_pos, dt_means.values, yerr=dt_stds.values,
                     capsize=4, color='steelblue', edgecolor='k', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{v:.2e}\n(n={int(c)})' for v, c in zip(dt_means.index, dt_counts)],
                          fontsize=7, rotation=45, ha='right')
        ax.set_xlabel('Δt'); ax.set_ylabel('Mean R²')
        ax.set_title('Mean R² by Δt Group', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        # Row 3, col 2: Delta_t RMSE grouped
        ax = fig.add_subplot(gs[3, 2])
        rmse_means = dt_groups['overall_rmse'].mean()
        rmse_stds = dt_groups['overall_rmse'].std().fillna(0)

        bars = ax.bar(x_pos, rmse_means.values, yerr=rmse_stds.values,
                     capsize=4, color='coral', edgecolor='k', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{v:.2e}\n(n={int(c)})' for v, c in zip(rmse_means.index, dt_counts)],
                          fontsize=7, rotation=45, ha='right')
        ax.set_xlabel('Δt'); ax.set_ylabel('Mean RMSE')
        ax.set_title('Mean RMSE by Δt Group', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        return fig

    def create_delta_t_performance_table(self):
        """Generate textual delta_t performance summary."""
        if not self.simulation_metrics:
            return "No data."

        df = pd.DataFrame([
            {
                'delta_t': m['metadata']['delta_t'],
                'r2': m['overall_physical']['r2'],
                'rmse': m['overall_physical']['rmse'],
            }
            for m in self.simulation_metrics
        ])

        groups = df.groupby(df['delta_t'].round(8))
        lines = [
            f"\nDelta_t Performance Summary",
            "=" * 70,
            f"{'Δt':>14} {'Count':>6} {'R² mean':>10} {'R² std':>10} {'RMSE mean':>12} {'RMSE std':>12}",
            "-" * 70,
        ]
        for dt_val, grp in sorted(groups, key=lambda x: x[0]):
            lines.append(
                f"{dt_val:>14.6e} {len(grp):>6} "
                f"{grp['r2'].mean():>10.4f} {grp['r2'].std():>10.4f} "
                f"{grp['rmse'].mean():>12.6e} {grp['rmse'].std():>12.6e}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    def plot_rollout_error_growth(self, results_dict, figsize=(16, 10)):
        """Plot how error grows across rollout timesteps."""
        preds = results_dict['predictions_physical']
        targs = results_dict['targets_physical']
        if not preds:
            return None

        max_steps = max(len(sp) for sp in preds)

        # Per-variable MSE at each timestep, aggregated across simulations
        var_mse_by_step = {vn: [[] for _ in range(max_steps)] for vn in self.var_names}
        overall_mse_by_step = [[] for _ in range(max_steps)]

        for sp, st in zip(preds, targs):
            for t in range(len(sp)):
                p, tg = sp[t], st[t]
                overall_mse_by_step[t].append(np.mean((p - tg) ** 2))
                for vi, vn in enumerate(self.var_names):
                    var_mse_by_step[vn][t].append(np.mean((p[:, vi] - tg[:, vi]) ** 2))

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Rollout Error Growth Over Timesteps', fontsize=14, fontweight='bold')

        # Overall
        ax = axes[0]
        steps_with_data = [t for t in range(max_steps) if overall_mse_by_step[t]]
        means = [np.mean(overall_mse_by_step[t]) for t in steps_with_data]
        stds = [np.std(overall_mse_by_step[t]) for t in steps_with_data]
        ax.errorbar(steps_with_data, means, yerr=stds, fmt='o-', capsize=4, color='steelblue')
        ax.set_xlabel('Rollout Step'); ax.set_ylabel('MSE')
        ax.set_title('Overall MSE vs Rollout Step')
        ax.set_yscale('log'); ax.grid(alpha=0.3)

        # Per-variable
        ax = axes[1]
        colors_var = plt.cm.Set1(np.linspace(0, 0.6, len(self.var_names)))
        for vi, vn in enumerate(self.var_names):
            steps = [t for t in range(max_steps) if var_mse_by_step[vn][t]]
            means = [np.mean(var_mse_by_step[vn][t]) for t in steps]
            ax.plot(steps, means, 'o-', label=vn, color=colors_var[vi], ms=5)
        ax.set_xlabel('Rollout Step'); ax.set_ylabel('MSE')
        ax.set_title('Per-Variable MSE vs Rollout Step')
        ax.set_yscale('log'); ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_prediction_scatter(self, results_dict, figsize=(18, 5)):
        """Scatter: prediction vs target for each variable."""
        preds = results_dict['predictions_physical']
        targs = results_dict['targets_physical']
        if not preds:
            return None

        all_p = np.concatenate([p for sp in preds for p in sp], axis=0)
        all_t = np.concatenate([t for st in targs for t in st], axis=0)

        fig, axes = plt.subplots(1, len(self.var_names), figsize=figsize)
        if len(self.var_names) == 1:
            axes = [axes]
        fig.suptitle('Prediction vs Target (Physical Units)', fontsize=14, fontweight='bold')

        n_sample = min(5000, all_p.shape[0])
        idx = np.random.choice(all_p.shape[0], n_sample, replace=False)

        for vi, vn in enumerate(self.var_names):
            ax = axes[vi]
            pv, tv = all_p[idx, vi], all_t[idx, vi]
            ax.scatter(tv, pv, s=1, alpha=0.3)
            lo = min(tv.min(), pv.min())
            hi = max(tv.max(), pv.max())
            ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.7)
            r2 = r2_score(tv, pv)
            ax.set_title(f'{vn}  (R² = {r2:.4f})')
            ax.set_xlabel('Target'); ax.set_ylabel('Prediction')
            ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_test_simulations(test_dir, test_files, pattern, max_files):
    """Load test simulation .pt files."""
    paths = ([Path(f) for f in test_files] if test_files
             else sorted(list(Path(test_dir).glob(pattern))))
    if max_files:
        paths = paths[:max_files]
    simulations = []
    for p in paths:
        try:
            sim_data = torch.load(p, weights_only=False)
            simulations.append(sim_data)
            print(f"  Loaded {p.name}: {len(sim_data)} timesteps")
        except Exception as e:
            print(f"  Error loading {p}: {e}")
    return simulations


# ==============================================================================
# MAIN
# ==============================================================================

def evaluate_shocktube(model_path, test_dir, test_files, output_dir, args):
    """Main evaluation orchestrator."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*70}")
    print(f"G-PARCv2 SHOCK TUBE EVALUATION")
    print(f"{'='*70}")
    print(f"Device:     {device}")
    print(f"Model:      {model_path}")
    print(f"Output:     {output_path}")
    print(f"Eval mode:  {args.eval_mode}")
    print(f"{'='*70}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Load normalization metadata
    norm_metadata = None
    if args.norm_metadata_file and Path(args.norm_metadata_file).exists():
        with open(args.norm_metadata_file, 'r') as f:
            norm_metadata = json.load(f)
        print(f"\n✓ Loaded normalization metadata from: {args.norm_metadata_file}")
    if norm_metadata is None and test_dir:
        norm_metadata = load_normalization_metadata(test_dir)
    if norm_metadata is None:
        norm_metadata = load_normalization_metadata_from_checkpoint(Path(model_path).parent)
    if norm_metadata is None:
        print("\n⚠️  No normalization metadata found — denormalization disabled")

    # Build model (must match training architecture)
    print(f"\nBuilding model...")
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

    derivative_solver = ShockTubeDifferentiator(
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=args.feature_out_channels,
        global_embed_dim=args.global_embed_dim,
        list_adv_idx=list(range(args.num_dynamic_feats)),
        list_dif_idx=list(range(args.num_dynamic_feats)),
        velocity_indices=[args.velocity_index],
        spade_random_noise=args.spade_random_noise,
        heads=args.spade_heads,
        concat=args.spade_concat,
        dropout=args.spade_dropout,
        zero_init=args.zero_init,
    )

    model = GPARC_ShockTube_V2(
        derivative_solver_physics=derivative_solver,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        skip_dynamic_indices=args.skip_dynamic_indices,
        global_param_dim=args.global_param_dim,
        global_embed_dim=args.global_embed_dim,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✓ Model loaded")

    # Load data
    print(f"\nLoading test simulations...")
    simulations = load_test_simulations(test_dir, test_files, "*.pt", args.max_sequences)
    if not simulations:
        print("No simulations loaded!")
        return
    print(f"Loaded {len(simulations)} simulations")

    # Initialize MLS weights
    print("Initializing MLS operators...")
    try:
        first_data = simulations[0][0].to(device)
        if not hasattr(first_data, 'pos') or first_data.pos is None:
            first_data.pos = first_data.x[:, :args.num_static_feats]
        derivative_solver.initialize_weights(first_data)
    except Exception as e:
        print(f"Error initializing MLS weights: {e}")
        return

    # Create evaluator
    evaluator = ShockTubeEvaluator(model, device, norm_metadata=norm_metadata)

    eval_mode = args.eval_mode.lower()

    # ==================== ROLLOUT ====================
    if eval_mode in ['rollout', 'both']:
        print(f"\n{'='*60}")
        print(f"ROLLOUT EVALUATION (steps={args.rollout_steps})")
        print(f"{'='*60}")

        rollout_results = evaluator.evaluate_rollout_predictions(
            simulations, rollout_steps=args.rollout_steps,
        )

        rollout_metrics = evaluator.compute_benchmark_metrics(
            rollout_results['predictions_physical'],
            rollout_results['targets_physical'],
        )

        print(f"\n{'-'*40}")
        print(f"ROLLOUT RESULTS")
        print(f"{'-'*40}")
        for k, v in sorted(rollout_metrics.items()):
            print(f"  {k:>20}: {v:.4f}" if isinstance(v, float) else f"  {k:>20}: {v}")

        print(evaluator.create_delta_t_performance_table())

        with open(output_path / 'rollout_metrics.json', 'w') as f:
            json.dump(rollout_metrics, f, indent=2)

        # Dashboard
        print("\nCreating rollout dashboard...")
        dash_fig = evaluator.plot_comprehensive_metrics(rollout_results, eval_mode='rollout')
        if dash_fig:
            dash_fig.savefig(output_path / 'rollout_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close(dash_fig)

        # Global parameter analysis
        print("Creating global parameter analysis...")
        gpa_fig = evaluator.plot_global_parameter_analysis()
        if gpa_fig:
            gpa_fig.savefig(output_path / 'rollout_global_parameter_analysis.png', dpi=150, bbox_inches='tight')
            plt.close(gpa_fig)

        # Error growth
        print("Creating error growth plot...")
        eg_fig = evaluator.plot_rollout_error_growth(rollout_results)
        if eg_fig:
            eg_fig.savefig(output_path / 'rollout_error_growth.png', dpi=150, bbox_inches='tight')
            plt.close(eg_fig)

        # Scatter
        print("Creating scatter plots...")
        sc_fig = evaluator.plot_prediction_scatter(rollout_results)
        if sc_fig:
            sc_fig.savefig(output_path / 'rollout_scatter.png', dpi=150, bbox_inches='tight')
            plt.close(sc_fig)

        # GIFs
        if args.create_gifs:
            print(f"\nCreating rollout GIF visualizations...")
            selected = select_representative_simulations(
                rollout_results, n_samples=args.num_viz_simulations,
                selection_mode=args.viz_selection_mode,
            )
            print(f"Selected simulations: {selected}")
            for i, idx in enumerate(selected, 1):
                print(f"\n[{i}/{len(selected)}] GIFs for simulation {idx}...")
                evaluator.create_gifs(
                    simulations, rollout_results, idx, output_path,
                    fps=args.gif_fps, frame_skip=args.gif_frame_skip, eval_mode='rollout',
                )

        # Save full results
        serializable_metrics = []
        for m in evaluator.simulation_metrics:
            entry = {**m['metadata'], **m['overall_physical']}
            for vn, vm in m.get('per_variable', {}).items():
                for mk, mv in vm.items():
                    entry[f'{vn}_{mk}'] = mv
            serializable_metrics.append(entry)

        with open(output_path / 'rollout_per_simulation.json', 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

    # ==================== SNAPSHOT ====================
    if eval_mode in ['snapshot', 'both']:
        print(f"\n{'='*60}")
        print(f"SNAPSHOT EVALUATION (single-step from GT)")
        print(f"{'='*60}")

        snapshot_results = evaluator.evaluate_snapshot_predictions(simulations)

        snapshot_metrics = evaluator.compute_benchmark_metrics(
            snapshot_results['predictions_physical'],
            snapshot_results['targets_physical'],
        )

        print(f"\n{'-'*40}")
        print(f"SNAPSHOT RESULTS")
        print(f"{'-'*40}")
        for k, v in sorted(snapshot_metrics.items()):
            print(f"  {k:>20}: {v:.4f}" if isinstance(v, float) else f"  {k:>20}: {v}")

        with open(output_path / 'snapshot_metrics.json', 'w') as f:
            json.dump(snapshot_metrics, f, indent=2)

        print("\nCreating snapshot dashboard...")
        dash_fig = evaluator.plot_comprehensive_metrics(snapshot_results, eval_mode='snapshot')
        if dash_fig:
            dash_fig.savefig(output_path / 'snapshot_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close(dash_fig)

        print("Creating snapshot global parameter analysis...")
        gpa_fig = evaluator.plot_global_parameter_analysis()
        if gpa_fig:
            gpa_fig.savefig(output_path / 'snapshot_global_parameter_analysis.png', dpi=150, bbox_inches='tight')
            plt.close(gpa_fig)

        sc_fig = evaluator.plot_prediction_scatter(snapshot_results)
        if sc_fig:
            sc_fig.savefig(output_path / 'snapshot_scatter.png', dpi=150, bbox_inches='tight')
            plt.close(sc_fig)

        if args.create_gifs:
            print(f"\nCreating snapshot GIFs...")
            selected = select_representative_simulations(
                snapshot_results, n_samples=args.num_viz_simulations,
                selection_mode=args.viz_selection_mode,
            )
            for i, idx in enumerate(selected, 1):
                print(f"\n[{i}/{len(selected)}] Snapshot GIFs for simulation {idx}...")
                evaluator.create_gifs(
                    simulations, snapshot_results, idx, output_path,
                    fps=args.gif_fps, frame_skip=args.gif_frame_skip, eval_mode='snapshot',
                )

        serializable_metrics = []
        for m in evaluator.simulation_metrics:
            entry = {**m['metadata'], **m['overall_physical']}
            for vn, vm in m.get('per_variable', {}).items():
                for mk, mv in vm.items():
                    entry[f'{vn}_{mk}'] = mv
            serializable_metrics.append(entry)
        with open(output_path / 'snapshot_per_simulation.json', 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

    # ==================== COMPARISON ====================
    if eval_mode == 'both':
        print(f"\n{'='*60}")
        print(f"SNAPSHOT vs ROLLOUT COMPARISON")
        print(f"{'='*60}")

        # Reload metrics from saved files
        with open(output_path / 'rollout_metrics.json') as f:
            r_met = json.load(f)
        with open(output_path / 'snapshot_metrics.json') as f:
            s_met = json.load(f)

        print(f"  {'Metric':<25} {'Snapshot':>12} {'Rollout':>12} {'Ratio':>10}")
        print(f"  {'-'*60}")
        for key in ['RRMSE_total', 'overall_R2', 'overall_RMSE']:
            sv = s_met.get(key, 0)
            rv = r_met.get(key, 0)
            ratio = rv / sv if sv != 0 else float('inf')
            print(f"  {key:<25} {sv:>12.4f} {rv:>12.4f} {ratio:>10.2f}x")

        comparison = {'snapshot': s_met, 'rollout': r_met}
        with open(output_path / 'comparison_metrics.json', 'w') as f:
            json.dump(comparison, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="G-PARCv2 Shock Tube Evaluation")

    # Paths
    parser.add_argument("--model_path", required=True)
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--test_dir", type=str)
    input_grp.add_argument("--test_files", type=str, nargs='+')
    parser.add_argument("--output_dir", default="./eval_shocktube")
    parser.add_argument("--norm_metadata_file", type=str, default=None,
                        help="Explicit path to normalization_metadata.json")

    # Evaluation mode
    parser.add_argument("--eval_mode", type=str, default="rollout",
                        choices=['rollout', 'snapshot', 'both'])

    # Architecture (MUST match training script!)
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=3,
                        help="Dynamic features AFTER skipping")
    parser.add_argument("--skip_dynamic_indices", type=int, nargs='+', default=[2])
    parser.add_argument("--velocity_index", type=int, default=1)
    parser.add_argument("--global_param_dim", type=int, default=3)
    parser.add_argument("--global_embed_dim", type=int, default=64)

    # Feature extractor
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_layer_norm", action="store_true", default=True)
    parser.add_argument("--use_relative_pos", action="store_true", default=True)

    # Integrator
    parser.add_argument("--integrator", type=str, default="euler",
                        choices=["euler", "heun", "rk4"])

    # Differentiator (SPADE)
    parser.add_argument("--spade_random_noise", action="store_true", default=False)
    parser.add_argument("--spade_heads", type=int, default=4)
    parser.add_argument("--spade_concat", action="store_true", default=True)
    parser.add_argument("--spade_dropout", type=float, default=0.1)
    parser.add_argument("--zero_init", action="store_true", default=False)

    # Evaluation settings
    parser.add_argument("--max_sequences", type=int, default=30)
    parser.add_argument("--rollout_steps", type=int, default=10)
    parser.add_argument("--create_gifs", action="store_true")
    parser.add_argument("--num_viz_simulations", type=int, default=3)
    parser.add_argument("--viz_selection_mode", type=str, default="representative",
                        choices=['representative', 'best', 'worst', 'all'])
    parser.add_argument("--gif_fps", type=int, default=4)
    parser.add_argument("--gif_frame_skip", type=int, default=1)

    args = parser.parse_args()
    evaluate_shocktube(args.model_path, args.test_dir, args.test_files, args.output_dir, args)


if __name__ == "__main__":
    main()