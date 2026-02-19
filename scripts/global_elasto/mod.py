#!/usr/bin/env python3
"""
GPARC v2 Evaluation Script - Elastoplastic (Scheduled Sampling)
================================================================
Evaluates models trained with scheduled sampling.
Supports both ROLLOUT and SNAPSHOT evaluation modes.

Uses shared visualization and metric modules from visualizations/.
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
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings("ignore", category=UserWarning)

# ── Architecture ──────────────────────────────────────────────────────────
from utilities.featureextractor import GraphConvFeatureExtractorV2
from differentiator.differentiator import ElastoPlasticDifferentiator
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from models.globalelasto import GPARC_ElastoPlastic_Numerical

# ── Shared viz / metrics ─────────────────────────────────────────────────
from visualizations.metrics import (
    compute_rrmse, compute_rrmse_per_variable,
    compute_rrmse_scalar, compute_rrmse_scalar_per_variable,
)
from visualizations.mesh_io import get_erosion_mask, get_valid_node_mask
from visualizations.elasto_viz import create_elasto_visualizations
from visualizations.dashboard import plot_elasto_dashboard
from visualizations.selection import select_representative_simulations


# ==============================================================================
# NORMALIZATION UTILITIES
# ==============================================================================

def load_normalization_stats(data_dir):
    """Load normalization statistics from the data directory."""
    stats_file = Path(data_dir).parent / "normalization_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"\n✓ Loaded normalization stats from: {stats_file}")
        print(f"  Method: {stats.get('normalization_method', 'unknown')}")
        if 'position' in stats and 'displacement' in stats:
            print(f"  max_position: {stats['position']['max_position']:.2f} mm")
            print(f"  max_displacement: {stats['displacement']['max_displacement']:.2f} mm")
        return stats
    else:
        print(f"\n⚠️  No normalization_stats.json found at {stats_file}")
        return None


def load_normalization_stats_from_checkpoint_dir(checkpoint_dir):
    """Try loading normalization_stats.json from the model checkpoint directory."""
    stats_file = Path(checkpoint_dir) / "normalization_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"\n✓ Loaded normalization stats from checkpoint dir: {stats_file}")
        return stats
    return None


def get_pos_normalization_params(norm_stats):
    """Extract position normalization parameters."""
    if norm_stats is None:
        print("  ⚠️  No norm stats — using hardcoded z-score defaults")
        return [97.2165, 50.2759], [59.3803, 28.4965]
    pos_stats = norm_stats['position']
    pos_mean = [pos_stats['x_pos']['mean'], pos_stats['y_pos']['mean']]
    pos_std = [pos_stats['x_pos']['std'], pos_stats['y_pos']['std']]
    return pos_mean, pos_std


# ==============================================================================
# EVALUATOR CLASS
# ==============================================================================

class ElastoPlasticEvaluator:
    """Evaluator for G-PARC elastoplastic models with scheduled sampling."""

    VAR_NAMES = ['U_x', 'U_y']

    def __init__(self, model, device='cpu', denormalization_params=None, norm_stats=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.denorm_params = denormalization_params
        self.norm_stats = norm_stats
        self.var_names = self.VAR_NAMES
        self.simulation_metrics = []

    def load_denormalization_params(self, metadata_file):
        """Load denormalization parameters from metadata file (z-score legacy)."""
        if not Path(metadata_file).exists():
            return
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        self.denorm_params = {}
        norm_params = metadata.get('original_metadata', {}).get('normalization_statistics',
                               metadata.get('normalization_statistics', {}))
        for var in self.var_names:
            if var in norm_params:
                self.denorm_params[var] = norm_params[var]

    def denormalize_predictions(self, normalized_data, method='global_max'):
        """Convert normalized predictions to physical units."""
        if method == 'none':
            return normalized_data
        if method == 'global_max':
            if self.norm_stats is None:
                print("  ⚠️  No norm_stats for global_max denormalization — returning raw")
                return normalized_data
            max_disp = self.norm_stats.get('displacement', {}).get('max_displacement', 1.0)
            return normalized_data * max_disp
        elif method == 'zscore':
            if self.denorm_params is None:
                return normalized_data
            physical_data = np.zeros_like(normalized_data)
            for i, var_name in enumerate(self.var_names):
                if var_name not in self.denorm_params:
                    physical_data[:, i] = normalized_data[:, i]
                    continue
                params = self.denorm_params[var_name]
                mean, std = params.get('mean', 0.0), params.get('std', 1.0)
                physical_data[:, i] = normalized_data[:, i] * std + mean
            return physical_data
        else:
            return normalized_data

    def _prep_simulation(self, simulation):
        """Move Data objects to device and extract mesh_id."""
        simulation = [d.to(self.device) for d in simulation]
        mid = getattr(simulation[0], "mesh_id", None)
        if mid is None:
            raise ValueError("Data object missing mesh_id")
        mesh_id_int = int(mid.view(-1)[0].item()) if torch.is_tensor(mid) else int(mid)
        return simulation, mesh_id_int

    def _ensure_mesh_cached(self, initial_data, mesh_id_int):
        """Reinitialize MLS weights/caches when mesh changes."""
        deriv_solver = self.model.derivative_solver
        real_solver = deriv_solver.solver if hasattr(deriv_solver, "solver") else deriv_solver
        if getattr(real_solver, "_active_mesh_id", None) == mesh_id_int:
            return
        if hasattr(real_solver, "clear_cache"):
            real_solver.clear_cache()
        for attr in ("geo_cache", "weights_cache", "damping_cache"):
            if hasattr(real_solver, attr):
                obj = getattr(real_solver, attr)
                if hasattr(obj, "clear"):
                    obj.clear()
        if hasattr(real_solver, "initialize_weights"):
            real_solver.initialize_weights(initial_data)
        real_solver._active_mesh_id = mesh_id_int

    def generate_rollout(self, initial_data, simulation, rollout_steps):
        """Generate autoregressive rollout predictions."""
        predictions = []
        F_prev = simulation[0].x[:, self.model.num_static_feats:].clone()
        for step in range(rollout_steps):
            data_t = simulation[step]
            static_feats = data_t.x[:, :self.model.num_static_feats]
            edge_index = data_t.edge_index
            if hasattr(data_t, "mesh_id"):
                edge_index.mesh_id = data_t.mesh_id
            F_pred = self.model.step(
                static_feats=static_feats, dynamic_state=F_prev.clone(),
                edge_index=edge_index, dt=1.0)
            predictions.append(F_pred)
            F_prev = F_pred
        return predictions

    def generate_snapshot_predictions(self, simulation, num_steps):
        """Generate snapshot (single-step from GT) predictions."""
        predictions = []
        for step in range(num_steps):
            data_t = simulation[step]
            static_feats = data_t.x[:, :self.model.num_static_feats]
            edge_index = data_t.edge_index
            if hasattr(data_t, "mesh_id"):
                edge_index.mesh_id = data_t.mesh_id
            F_gt = data_t.x[:, self.model.num_static_feats:].clone()
            F_pred = self.model.step(
                static_feats=static_feats, dynamic_state=F_gt,
                edge_index=edge_index, dt=1.0)
            predictions.append(F_pred)
        return predictions

    def _evaluate_common(self, simulations, mode='rollout', rollout_steps=10,
                         normalization_method='global_max'):
        """Shared evaluation logic for rollout and snapshot modes."""
        results = {
            'predictions_physical': [], 'targets_physical': [],
            'metadata': [], 'erosion_masks': []
        }
        self.simulation_metrics = []

        with torch.no_grad():
            for sim_idx, simulation in enumerate(tqdm(simulations, desc=f"{mode.capitalize()} eval")):
                try:
                    simulation, mesh_id_int = self._prep_simulation(simulation)
                    initial_data = simulation[0]
                    elements = initial_data.elements.detach().cpu().numpy()
                    self._ensure_mesh_cached(initial_data, mesh_id_int)

                    if mode == 'rollout':
                        actual_steps = min(rollout_steps, len(simulation))
                        preds_raw = self.generate_rollout(initial_data, simulation, actual_steps)
                    else:
                        actual_steps = len(simulation) - 1
                        preds_raw = self.generate_snapshot_predictions(simulation, actual_steps)

                    # Filter unstable predictions
                    preds_norm = []
                    for p in preds_raw:
                        if torch.isfinite(p).all() and p.abs().max() < 50.0:
                            preds_norm.append(p.cpu().numpy())
                        elif mode == 'rollout':
                            break  # Rollout: stop at first explosion
                        # Snapshot: skip individual bad steps

                    if len(preds_norm) == 0:
                        print(f"  Skipping sim {sim_idx}: unstable predictions")
                        continue

                    actual_steps = len(preds_norm)

                    # Targets and erosion masks
                    if mode == 'snapshot':
                        targs_norm = [simulation[t].y.cpu().numpy() for t in range(actual_steps)]
                        erosion_masks = [get_erosion_mask(simulation[t + 1], len(elements))
                                         for t in range(actual_steps)]
                    else:
                        targs_norm = [simulation[t].y.cpu().numpy() for t in range(actual_steps)]
                        erosion_masks = [get_erosion_mask(simulation[t], len(elements))
                                         for t in range(actual_steps)]

                    # Denormalize
                    preds_phys = [self.denormalize_predictions(p, normalization_method)
                                  for p in preds_norm]
                    targs_phys = [self.denormalize_predictions(t, normalization_method)
                                  for t in targs_norm]

                    results['predictions_physical'].append(preds_phys)
                    results['targets_physical'].append(targs_phys)
                    results['erosion_masks'].append(erosion_masks)

                    metadata = {
                        'simulation_idx': sim_idx,
                        'case_name': f'simulation_{sim_idx}',
                        'rollout_length': actual_steps,
                        'num_nodes': initial_data.num_nodes,
                        'num_elements': len(elements),
                        'max_eroded': max(m.sum() for m in erosion_masks) if erosion_masks else 0
                    }
                    results['metadata'].append(metadata)

                    # Per-simulation metrics (erosion-aware)
                    valid_node_masks = [get_valid_node_mask(elements, em) for em in erosion_masks]
                    all_p, all_t = [], []
                    for t in range(len(preds_phys)):
                        mask = valid_node_masks[t]
                        if mask.sum() > 0:
                            all_p.append(preds_phys[t][mask])
                            all_t.append(targs_phys[t][mask])
                    if all_p:
                        all_p_cat = np.concatenate(all_p, axis=0)
                        all_t_cat = np.concatenate(all_t, axis=0)
                        rmse = float(np.sqrt(mean_squared_error(all_t_cat, all_p_cat)))
                        r2 = float(r2_score(all_t_cat, all_p_cat))
                    else:
                        rmse, r2 = float('inf'), 0.0

                    self.simulation_metrics.append({
                        'metadata': metadata,
                        'overall_physical': {'rmse': rmse, 'r2': r2}
                    })

                except Exception as e:
                    print(f"Error processing simulation {sim_idx}: {e}")
                    import traceback; traceback.print_exc()

        results['simulation_metrics'] = self.simulation_metrics
        return results

    def evaluate_rollout_predictions(self, simulations, rollout_steps=10,
                                      normalization_method='global_max'):
        return self._evaluate_common(simulations, 'rollout', rollout_steps, normalization_method)

    def evaluate_snapshot_predictions(self, simulations, normalization_method='global_max'):
        return self._evaluate_common(simulations, 'snapshot', normalization_method=normalization_method)

    def compute_benchmark_metrics(self, predictions_physical, targets_physical,
                                   erosion_masks=None):
        """Compute RRMSE benchmark metrics using shared functions.
        
        Reports both:
          - Field-level RRMSE (inf-norm denominator, per-timestep)
          - PLAID scalar RRMSE (per-node normalization, matches PLAID leaderboard)
        """
        if not predictions_physical:
            return {}

        # Flatten nested lists: list of sims × list of timesteps → flat list of arrays
        all_pred, all_targ = [], []
        for seq_p, seq_t in zip(predictions_physical, targets_physical):
            for p, t in zip(seq_p, seq_t):
                all_pred.append(p)
                all_targ.append(t)

        # Field-level RRMSE (inf-norm denominator)
        rrmse_total = compute_rrmse(all_pred, all_targ)
        rrmse_per_var = compute_rrmse_per_variable(all_pred, all_targ, self.var_names)

        # PLAID scalar RRMSE (per-node normalization)
        rrmse_scalar_total = compute_rrmse_scalar(all_pred, all_targ)
        rrmse_scalar_per_var = compute_rrmse_scalar_per_variable(all_pred, all_targ, self.var_names)

        return {
            # Field-level (inf-norm)
            'RRMSE_total': rrmse_total,
            'RRMSE_Ux': rrmse_per_var.get('U_x', float('inf')),
            'RRMSE_Uy': rrmse_per_var.get('U_y', float('inf')),
            'total_error': np.mean(list(rrmse_per_var.values())),
            # PLAID scalar (per-node)
            'PLAID_RRMSE_total': rrmse_scalar_total,
            'PLAID_RRMSE_Ux': rrmse_scalar_per_var.get('U_x', float('inf')),
            'PLAID_RRMSE_Uy': rrmse_scalar_per_var.get('U_y', float('inf')),
            'PLAID_total_error': np.mean(list(rrmse_scalar_per_var.values())),
        }


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_test_simulations(test_dir, test_files, pattern, max_files):
    """Load test simulations with mesh_id injection."""
    simulations = []
    paths = [Path(f) for f in test_files] if test_files else sorted(list(Path(test_dir).glob(pattern)))
    if max_files:
        paths = paths[:max_files]
    for idx, p in enumerate(paths):
        try:
            sim_data = torch.load(p, weights_only=False)
            match = re.search(r'\d+', p.stem)
            sim_id_int = int(match.group()) if match else idx
            for data in sim_data:
                data.mesh_id = torch.tensor([sim_id_int], dtype=torch.long)
            simulations.append(sim_data)
            print(f"  {p.name}: {len(sim_data)} timesteps")
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return simulations


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def evaluate_elastoplastic(model_path, test_dir, test_files, output_dir, args):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # ── Load normalization stats ──
    norm_stats = None
    if args.norm_stats_file and Path(args.norm_stats_file).exists():
        with open(args.norm_stats_file, 'r') as f:
            norm_stats = json.load(f)
        print(f"\n✓ Loaded normalization stats from --norm_stats_file: {args.norm_stats_file}")
    if norm_stats is None and test_dir:
        norm_stats = load_normalization_stats(test_dir)
    if norm_stats is None:
        norm_stats = load_normalization_stats_from_checkpoint_dir(Path(model_path).parent)
    if norm_stats is None:
        print("\n⚠️  No normalization_stats.json found! Using hardcoded z-score defaults")
        norm_stats = {
            'normalization_method': 'z_score',
            'position': {
                'x_pos': {'mean': 97.2165, 'std': 59.3803},
                'y_pos': {'mean': 50.2759, 'std': 28.4965}
            }
        }

    pos_mean, pos_std = get_pos_normalization_params(norm_stats)
    norm_method = norm_stats.get('normalization_method', 'unknown')

    if args.normalization_method == 'auto':
        if norm_method == 'global_max':
            denorm_method = 'global_max'
        elif norm_method in ('z_score', 'zscore'):
            denorm_method = 'zscore'
        else:
            denorm_method = 'none'
    else:
        denorm_method = args.normalization_method

    print(f"\n{'='*60}")
    print(f"NORMALIZATION: {norm_method} → denorm: {denorm_method}")
    if 'displacement' in norm_stats:
        print(f"  max_displacement: {norm_stats['displacement'].get('max_displacement', 'N/A')}")
    print(f"{'='*60}")

    # ── Build model ──
    max_position = None
    if norm_method == 'global_max' and 'position' in norm_stats:
        max_position = norm_stats['position'].get('max_position', None)

    gradient_solver = SolveGradientsLST(
        pos_mean=pos_mean, pos_std=pos_std,
        norm_method=norm_method, max_position=max_position)
    laplacian_solver = SolveWeightLST2d(
        pos_mean=pos_mean, pos_std=pos_std,
        norm_method=norm_method, max_position=max_position,
        min_neighbors=5, use_2hop_extension=False)

    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=args.num_static_feats,
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layer_norm=args.use_layer_norm,
        use_relative_pos=args.use_relative_pos)

    derivative_solver = ElastoPlasticDifferentiator(
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=args.feature_out_channels,
        list_strain_idx=list(range(args.num_dynamic_feats)),
        list_laplacian_idx=list(range(args.num_dynamic_feats)),
        spade_random_noise=False,
        heads=args.spade_heads,
        concat=args.spade_concat,
        dropout=args.spade_dropout,
        use_von_mises=args.use_von_mises,
        use_volumetric=args.use_volumetric,
        n_state_var=args.n_state_var,
        zero_init=args.zero_init)

    model = GPARC_ElastoPlastic_Numerical(
        derivative_solver_physics=derivative_solver,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        pos_mean=pos_mean, pos_std=pos_std,
        boundary_threshold=0.5,
        clamp_output=not args.no_clamp_output,
        norm_method=norm_method, max_position=max_position)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device); model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # ── Load data ──
    simulations = load_test_simulations(test_dir, test_files, "*.pt", args.max_sequences)
    if not simulations:
        print("No simulations loaded!"); return
    print(f"Loaded {len(simulations)} simulations")

    # Initialize MLS operators
    try:
        first_sim = simulations[0][0].to(device)
        if hasattr(model.derivative_solver, 'initialize_weights'):
            model.derivative_solver.initialize_weights(first_sim)
    except Exception as e:
        print(f"Error initializing weights: {e}"); return

    # ── Create evaluator ──
    evaluator = ElastoPlasticEvaluator(model, device, norm_stats=norm_stats)

    if test_dir and denorm_method == 'zscore':
        norm_file = Path(test_dir).parent / 'normalization_metadata.json'
        if norm_file.exists():
            evaluator.load_denormalization_params(norm_file)

    # ── Run evaluation ──
    eval_mode = args.eval_mode.lower()
    rollout_metrics = snapshot_metrics = None

    def _run(mode, rollout_steps=None):
        print(f"\n{'='*60}")
        print(f"{mode.upper()} EVALUATION")
        print(f"  Denormalization: {denorm_method}")
        print(f"{'='*60}")

        if mode == 'rollout':
            results = evaluator.evaluate_rollout_predictions(
                simulations, rollout_steps=rollout_steps,
                normalization_method=denorm_method)
        else:
            results = evaluator.evaluate_snapshot_predictions(
                simulations, normalization_method=denorm_method)

        metrics = evaluator.compute_benchmark_metrics(
            results['predictions_physical'], results['targets_physical'],
            results.get('erosion_masks'))

        def safe_fmt(val):
            return f"{val:.4f}" if isinstance(val, (float, int, np.floating)) else str(val)

        print(f"\n{'-'*40}")
        print(f"{mode.upper()} RESULTS")
        print(f"{'-'*40}")
        print(f"  Field-level RRMSE (inf-norm):")
        for k in ['RRMSE_total', 'RRMSE_Ux', 'RRMSE_Uy', 'total_error']:
            print(f"    {k:>15}: {safe_fmt(metrics.get(k, 'N/A'))}")
        print(f"  PLAID scalar RRMSE (per-node):")
        for k in ['PLAID_RRMSE_total', 'PLAID_RRMSE_Ux', 'PLAID_RRMSE_Uy', 'PLAID_total_error']:
            print(f"    {k:>20}: {safe_fmt(metrics.get(k, 'N/A'))}")

        metrics.update({
            'normalization_method': norm_method,
            'denormalization_method': denorm_method,
            'eval_mode': mode, 'model': 'G-PARC_v2', 'parameters': total_params
        })
        if 'displacement' in norm_stats:
            metrics['max_displacement'] = norm_stats['displacement'].get('max_displacement')

        with open(output_path / f'{mode}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Dashboard
        plot_elasto_dashboard(evaluator.simulation_metrics,
                               output_path / f'{mode}_dashboard.png',
                               model_name='G-PARC v2', norm_stats=norm_stats,
                               eval_mode=mode)

        # GIFs
        if args.create_gifs:
            selected = select_representative_simulations(
                evaluator.simulation_metrics, n_samples=args.num_viz_simulations,
                selection_mode=args.viz_selection_mode, sort_key='rmse')
            print(f"Selected simulations for GIFs: {selected}")

            for i, idx in enumerate(selected, 1):
                sim_idx = evaluator.simulation_metrics[idx]['metadata']['simulation_idx']
                print(f"\n[{i}/{len(selected)}] Creating {mode} GIFs for simulation {sim_idx}...")
                create_elasto_visualizations(
                    simulations[sim_idx],
                    results['predictions_physical'][idx],
                    results['targets_physical'][idx],
                    sim_idx, output_path,
                    model_name='G-PARC v2',
                    fps=args.gif_fps,
                    frame_skip=args.gif_frame_skip,
                    eval_mode=mode)

        # Per-simulation JSON
        serializable = []
        for m in evaluator.simulation_metrics:
            entry = {**m['metadata'], **m['overall_physical']}
            serializable.append(entry)
        with open(output_path / f'{mode}_per_simulation.json', 'w') as f:
            json.dump(serializable, f, indent=2)

        return metrics

    if eval_mode in ['rollout', 'both']:
        rollout_metrics = _run('rollout', rollout_steps=args.rollout_steps)
    if eval_mode in ['snapshot', 'both']:
        snapshot_metrics = _run('snapshot')

    # Comparison
    if eval_mode == 'both' and rollout_metrics and snapshot_metrics:
        print(f"\n{'='*60}")
        print(f"SNAPSHOT vs ROLLOUT COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Metric':<20} {'Snapshot':>12} {'Rollout':>12} {'Ratio':>10}")
        print(f"  {'-'*54}")
        for key in ['RRMSE_total', 'RRMSE_Ux', 'RRMSE_Uy', 'total_error',
                    'PLAID_RRMSE_total', 'PLAID_RRMSE_Ux', 'PLAID_RRMSE_Uy', 'PLAID_total_error']:
            s_val = snapshot_metrics.get(key, 0)
            r_val = rollout_metrics.get(key, 0)
            ratio = r_val / s_val if s_val > 0 else float('inf')
            print(f"  {key:<20} {s_val:>12.4f} {r_val:>12.4f} {ratio:>10.1f}x")

        comparison = {
            'snapshot': snapshot_metrics, 'rollout': rollout_metrics,
            'ratio': {k: rollout_metrics.get(k, 0) / max(snapshot_metrics.get(k, 0), 1e-12)
                      for k in ['RRMSE_total', 'RRMSE_Ux', 'RRMSE_Uy']}
        }
        with open(output_path / 'comparison_metrics.json', 'w') as f:
            json.dump(comparison, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Results: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="G-PARC v2 Elastoplastic Evaluation")

    # Paths
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--test_files", type=str, nargs='+')
    parser.add_argument("--output_dir", default="./eval_gparcv2_elasto")
    parser.add_argument("--norm_stats_file", type=str, default=None)

    # Evaluation mode
    parser.add_argument("--eval_mode", type=str, default="rollout",
                        choices=['rollout', 'snapshot', 'both'])

    # Architecture
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_layer_norm", action="store_true", default=False)
    parser.add_argument("--use_relative_pos", action="store_true", default=False)

    # Physics
    parser.add_argument("--no_clamp_output", action="store_true", default=False)
    parser.add_argument("--integrator", type=str, default="euler")
    parser.add_argument("--spade_heads", type=int, default=4)
    parser.add_argument("--spade_concat", action="store_true", default=False)
    parser.add_argument("--spade_dropout", type=float, default=0.1)
    parser.add_argument("--zero_init", action="store_true", default=False)

    # Dimensions
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=2)
    parser.add_argument("--n_state_var", type=int, default=0)
    parser.add_argument("--use_von_mises", action="store_true", default=False)
    parser.add_argument("--use_volumetric", action="store_true", default=False)

    # Eval settings
    parser.add_argument("--max_sequences", type=int, default=10)
    parser.add_argument("--rollout_steps", type=int, default=37)
    parser.add_argument("--normalization_method", default="auto",
                        choices=['auto', 'global_max', 'zscore', 'none'])
    parser.add_argument("--create_gifs", action="store_true")

    # Viz settings
    parser.add_argument("--num_viz_simulations", type=int, default=3)
    parser.add_argument("--viz_selection_mode", type=str, default="representative")
    parser.add_argument("--gif_fps", type=int, default=10)
    parser.add_argument("--gif_frame_skip", type=int, default=1)

    args = parser.parse_args()
    evaluate_elastoplastic(args.model_path, args.test_dir, args.test_files, args.output_dir, args)


if __name__ == "__main__":
    main()