#!/usr/bin/env python3
"""
MeshGraphNet Elastoplastic Evaluation Script
=============================================
Evaluates MeshGraphNet models on elastoplastic data.
Supports both ROLLOUT and SNAPSHOT evaluation modes.

Architecture imported from meshgraphnet.
Visualization / metrics imported from visualizations/.

Mirrors eval_mgkan_elasto.py exactly — only model loading and
forward pass differ.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings("ignore", category=UserWarning)

# ── Architecture ──────────────────────────────────────────────────────────
from meshgraphnet import MeshGraphNet

# ── Shared viz / metrics ─────────────────────────────────────────────────
from visualizations.metrics import (
    compute_rrmse, compute_rrmse_per_variable,
    compute_rrmse_scalar, compute_rrmse_scalar_per_variable,
)
from visualizations.denorm import load_normalization_stats, denormalize_global_max
from visualizations.mesh_io import get_erosion_mask, get_valid_node_mask
from visualizations.elasto_viz import create_elasto_visualizations
from visualizations.dashboard import plot_elasto_dashboard
from visualizations.selection import select_representative_simulations


# =========================================================================
# EVALUATOR
# =========================================================================

class MeshGraphNetEvaluator:
    """Evaluator for MeshGraphNet elastoplastic models."""

    def __init__(self, model, device='cpu', norm_stats=None, sf=2, df=2):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.norm_stats = norm_stats
        self.num_static_feats = sf
        self.num_dynamic_feats = df
        self.simulation_metrics = []

    def denormalize_predictions(self, normalized_data, method='global_max'):
        if method in ('none', None) or self.norm_stats is None:
            return normalized_data
        if method == 'global_max':
            return denormalize_global_max(normalized_data, self.norm_stats)
        return normalized_data

    def generate_rollout(self, simulation, rollout_steps):
        predictions = []
        sf, df = self.num_static_feats, self.num_dynamic_feats
        first = simulation[0]
        static = first.x[:, :sf]
        current_dynamic = first.x[:, sf:sf + df].clone()
        edge_index = first.edge_index
        edge_features = self.model.compute_edge_features(static, edge_index)

        for step in range(rollout_steps):
            node_features = torch.cat([static, current_dynamic], dim=-1)
            pred_delta = self.model(node_features, edge_features, edge_index)
            current_dynamic = (current_dynamic + pred_delta).detach()
            predictions.append(current_dynamic.clone())
        return predictions

    def generate_snapshot_predictions(self, simulation, num_steps):
        predictions = []
        sf, df = self.num_static_feats, self.num_dynamic_feats
        first = simulation[0]
        static = first.x[:, :sf]
        edge_index = first.edge_index
        edge_features = self.model.compute_edge_features(static, edge_index)

        for step in range(num_steps):
            data_t = simulation[step]
            F_gt = data_t.x[:, sf:sf + df].clone()
            node_features = torch.cat([static, F_gt], dim=-1)
            pred_delta = self.model(node_features, edge_features, edge_index)
            predictions.append(F_gt + pred_delta)
        return predictions

    def _process_simulation(self, sim_idx, simulation, mode, rollout_steps, denorm_method):
        simulation = [d.to(self.device) for d in simulation]
        elements = simulation[0].elements.detach().cpu().numpy()

        if mode == 'rollout':
            actual_steps = min(rollout_steps, len(simulation))
            preds_raw = self.generate_rollout(simulation, actual_steps)
        else:
            actual_steps = len(simulation) - 1
            preds_raw = self.generate_snapshot_predictions(simulation, actual_steps)

        preds_norm = []
        for p in preds_raw:
            if torch.isfinite(p).all() and p.abs().max() < 50.0:
                preds_norm.append(p.cpu().numpy())
            elif mode == 'rollout':
                break
            # snapshot: skip unstable frames
        if not preds_norm:
            return None

        actual_steps = len(preds_norm)
        targs_norm = [simulation[i].y.cpu().numpy() for i in range(actual_steps)]
        erosion_masks = [get_erosion_mask(simulation[i], len(elements)) for i in range(actual_steps)]

        preds_phys = [self.denormalize_predictions(p, denorm_method) for p in preds_norm]
        targs_phys = [self.denormalize_predictions(t, denorm_method) for t in targs_norm]

        metadata = {
            'simulation_idx': sim_idx, 'case_name': f'simulation_{sim_idx}',
            'num_nodes': simulation[0].num_nodes, 'num_elements': len(elements),
            'max_eroded': int(max(m.sum() for m in erosion_masks)) if erosion_masks else 0,
        }
        if mode == 'rollout':
            metadata['rollout_length'] = actual_steps
        else:
            metadata['num_snapshots'] = actual_steps

        valid_masks = [get_valid_node_mask(elements, em) for em in erosion_masks]
        all_p = [preds_phys[t][valid_masks[t]] for t in range(len(preds_phys)) if valid_masks[t].sum() > 0]
        all_t = [targs_phys[t][valid_masks[t]] for t in range(len(targs_phys)) if valid_masks[t].sum() > 0]
        if all_p:
            apc, atc = np.concatenate(all_p), np.concatenate(all_t)
            rmse = float(np.sqrt(mean_squared_error(atc, apc)))
            r2 = float(r2_score(atc, apc))
        else:
            rmse, r2 = float('inf'), 0.0

        return {
            'preds_phys': preds_phys, 'targs_phys': targs_phys,
            'erosion_masks': erosion_masks, 'metadata': metadata,
            'sim_metric': {'metadata': metadata, 'overall_physical': {'rmse': rmse, 'r2': r2}},
        }

    def evaluate(self, simulations, mode='rollout', rollout_steps=37,
                 normalization_method='global_max'):
        results = {'predictions_physical': [], 'targets_physical': [],
                    'metadata': [], 'erosion_masks': []}
        self.simulation_metrics = []

        with torch.no_grad():
            for sim_idx, simulation in enumerate(tqdm(simulations,
                                                       desc=f"{'Rollout' if mode=='rollout' else 'Snapshot'}")):
                try:
                    out = self._process_simulation(sim_idx, simulation, mode,
                                                    rollout_steps, normalization_method)
                    if out is None:
                        print(f"  Skipping sim {sim_idx}: unstable"); continue
                    results['predictions_physical'].append(out['preds_phys'])
                    results['targets_physical'].append(out['targs_phys'])
                    results['erosion_masks'].append(out['erosion_masks'])
                    results['metadata'].append(out['metadata'])
                    self.simulation_metrics.append(out['sim_metric'])
                except Exception as e:
                    print(f"Error sim {sim_idx}: {e}")
                    import traceback; traceback.print_exc()

        results['simulation_metrics'] = self.simulation_metrics
        return results

    def compute_plaid_benchmark_metrics(self, preds_phys, targs_phys):
        if not preds_phys:
            return {}
        all_pred = [p for sp in preds_phys for p in sp]
        all_targ = [t for st in targs_phys for t in st]
        rrmse_total = compute_rrmse(all_pred, all_targ)
        rrmse_per = compute_rrmse_per_variable(all_pred, all_targ, ['U_x', 'U_y'])
        # PLAID scalar RRMSE (per-node normalization)
        rrmse_scalar_total = compute_rrmse_scalar(all_pred, all_targ)
        rrmse_scalar_per = compute_rrmse_scalar_per_variable(all_pred, all_targ, ['U_x', 'U_y'])
        return {
            'RRMSE_total': rrmse_total,
            'RRMSE_Ux': rrmse_per.get('U_x', 0),
            'RRMSE_Uy': rrmse_per.get('U_y', 0),
            'total_error': np.mean(list(rrmse_per.values())),
            'PLAID_RRMSE_total': rrmse_scalar_total,
            'PLAID_RRMSE_Ux': rrmse_scalar_per.get('U_x', float('inf')),
            'PLAID_RRMSE_Uy': rrmse_scalar_per.get('U_y', float('inf')),
            'PLAID_total_error': np.mean(list(rrmse_scalar_per.values())),
        }


# =========================================================================
# DATA LOADING
# =========================================================================

def load_test_simulations(test_dir, pattern, max_files):
    simulations = []
    paths = sorted(Path(test_dir).glob(pattern))
    if max_files:
        paths = paths[:max_files]
    for idx, p in enumerate(paths):
        try:
            sim = torch.load(p, weights_only=False)
            match = re.search(r'\d+', p.stem)
            sim_id = int(match.group()) if match else idx
            for d in sim:
                d.mesh_id = torch.tensor([sim_id], dtype=torch.long)
                if not hasattr(d, 'pos') or d.pos is None:
                    d.pos = d.x[:, :2]
            simulations.append(sim)
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return simulations


# =========================================================================
# MAIN
# =========================================================================

def evaluate_meshgraphnet(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_dir = Path(args.model_path).parent
    config = json.load(open(model_dir / "config.json")) if (model_dir / "config.json").exists() else {}

    sf = config.get('num_static_feats', args.num_static_feats)
    df = config.get('num_dynamic_feats', args.num_dynamic_feats)
    hidden_dim = config.get('hidden_dim', args.hidden_dim)
    num_layers = config.get('num_layers', args.num_layers)

    # Normalization
    norm_stats = None
    if args.norm_stats_file and Path(args.norm_stats_file).exists():
        norm_stats = json.load(open(args.norm_stats_file))
    if norm_stats is None:
        norm_stats = load_normalization_stats(args.test_dir)
    if norm_stats is None:
        norm_stats = load_normalization_stats(str(model_dir))
    if norm_stats is None:
        norm_stats = {'normalization_method': 'none'}

    denorm_method = ('global_max' if norm_stats.get('normalization_method') == 'global_max' else 'none') \
        if args.normalization_method == 'auto' else args.normalization_method

    print(f"\n{'='*60}\nMeshGraphNet ELASTOPLASTIC EVALUATION\n{'='*60}")
    print(f"  Device: {device}, Denorm: {denorm_method}")

    model = MeshGraphNet(
        input_dim_node=sf + df, input_dim_edge=3,
        hidden_dim=hidden_dim, output_dim=df, num_layers=num_layers,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    simulations = load_test_simulations(args.test_dir, "*.pt", args.max_sequences)
    if not simulations:
        print("No simulations loaded!"); return
    print(f"Loaded {len(simulations)} simulations")

    evaluator = MeshGraphNetEvaluator(model, device, norm_stats=norm_stats, sf=sf, df=df)

    def _run_mode(mode):
        print(f"\n{'='*60}\n{mode.upper()} EVALUATION\n{'='*60}")
        results = evaluator.evaluate(simulations, mode=mode, rollout_steps=args.rollout_steps,
                                      normalization_method=denorm_method)
        metrics = evaluator.compute_plaid_benchmark_metrics(
            results['predictions_physical'], results['targets_physical'])

        print(f"\n{'─'*40}\n{mode.upper()} RESULTS\n{'─'*40}")
        print(f"  Field-level RRMSE (inf-norm):")
        for k, v in sorted(metrics.items()):
            if k.startswith('RRMSE') or k == 'total_error':
                print(f"    {k:>15}: {v:.4f}" if isinstance(v, float) else f"    {k:>15}: {v}")
        print(f"  PLAID scalar RRMSE (per-node):")
        for k, v in sorted(metrics.items()):
            if k.startswith('PLAID'):
                print(f"    {k:>20}: {v:.4f}" if isinstance(v, float) else f"    {k:>20}: {v}")

        metrics.update({'eval_mode': mode, 'model': 'MeshGraphNet', 'parameters': total_params})
        with open(output_path / f'{mode}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        plot_elasto_dashboard(evaluator.simulation_metrics, output_path / f'{mode}_dashboard.png',
                              model_name='MeshGraphNet', norm_stats=norm_stats, eval_mode=mode)

        if args.create_gifs:
            selected = select_representative_simulations(
                evaluator.simulation_metrics, n_samples=args.num_viz_simulations, sort_key='rmse')
            for idx in selected:
                si = evaluator.simulation_metrics[idx]['metadata']['simulation_idx']
                create_elasto_visualizations(
                    simulations[si], results['predictions_physical'][idx],
                    results['targets_physical'][idx], si, output_path,
                    model_name='MeshGraphNet', fps=args.gif_fps,
                    frame_skip=args.gif_frame_skip, eval_mode=mode)
        return metrics

    rm = sm = None
    if args.eval_mode in ['rollout', 'both']: rm = _run_mode('rollout')
    if args.eval_mode in ['snapshot', 'both']: sm = _run_mode('snapshot')
    if args.eval_mode == 'both' and rm and sm:
        print(f"\n{'='*60}\nSNAPSHOT vs ROLLOUT\n{'='*60}")
        for key in ['RRMSE_total', 'RRMSE_Ux', 'RRMSE_Uy',
                    'PLAID_RRMSE_total', 'PLAID_RRMSE_Ux', 'PLAID_RRMSE_Uy']:
            sv, rv = sm.get(key, 0), rm.get(key, 0)
            ratio = rv / sv if sv > 0 else float('inf')
            print(f"  {key:<20} {sv:>12.4f} {rv:>12.4f} {ratio:>10.1f}x")
        with open(output_path / 'comparison_metrics.json', 'w') as f:
            json.dump({'snapshot': sm, 'rollout': rm}, f, indent=2)
    print(f"\nDone! Results: {output_path}")


def main():
    p = argparse.ArgumentParser(description="MeshGraphNet Elastoplastic Evaluation")
    p.add_argument("--model_path", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--output_dir", default="./eval_meshgraphnet")
    p.add_argument("--norm_stats_file", default=None)
    p.add_argument("--eval_mode", default="rollout", choices=['rollout', 'snapshot', 'both'])
    p.add_argument("--rollout_steps", type=int, default=37)
    p.add_argument("--max_sequences", type=int, default=10)
    p.add_argument("--normalization_method", default="auto", choices=['auto', 'global_max', 'none'])
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=15)
    p.add_argument("--num_static_feats", type=int, default=2)
    p.add_argument("--num_dynamic_feats", type=int, default=2)
    p.add_argument("--create_gifs", action="store_true")
    p.add_argument("--num_viz_simulations", type=int, default=3)
    p.add_argument("--gif_fps", type=int, default=10)
    p.add_argument("--gif_frame_skip", type=int, default=1)
    main_args = p.parse_args()
    evaluate_meshgraphnet(main_args)


if __name__ == "__main__":
    main()