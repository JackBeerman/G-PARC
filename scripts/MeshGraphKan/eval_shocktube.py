#!/usr/bin/env python3
"""
MeshGraphKAN Shock Tube Evaluation Script
==========================================
Evaluates MeshGraphKAN models on shock tube test data.
Produces same outputs as G-PARCv2 eval for direct comparison.

Architecture imported from models.meshgraphkan.
Visualization / metrics imported from visualizations/.
"""

import argparse
import os
import sys
import json
import warnings
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings("ignore", category=UserWarning)

# ── Architecture ──────────────────────────────────────────────────────────
from models.meshgraphkan import (
    MeshGraphKAN, MeshGraphKANShocktubeRollout, build_meshgraphkan,
)

# ── Shared viz / metrics ─────────────────────────────────────────────────
from visualizations.metrics import compute_rrmse, compute_rrmse_per_variable
from visualizations.denorm import denorm_minmax, denorm_all_from_params, load_normalization_metadata
from visualizations.shocktube_viz import (
    SHOCKTUBE_VAR_NAMES, create_shocktube_visualizations,
    plot_rollout_error_growth, plot_prediction_scatter,
)
from visualizations.dashboard import (
    plot_shocktube_dashboard, plot_global_parameter_analysis,
    create_delta_t_performance_table,
)
from visualizations.selection import select_representative_simulations


# =========================================================================
# EVALUATOR
# =========================================================================

class ShockTubeMGKANEvaluator:
    VAR_NAMES = SHOCKTUBE_VAR_NAMES

    def __init__(self, model, device='cpu', norm_metadata=None):
        self.model = model
        self.device = device
        self.model.to(device); self.model.eval()
        self.norm_metadata = norm_metadata
        self.var_names = self.VAR_NAMES
        self.simulation_metrics = []

        self.denorm_params = {}
        self.delta_t_denorm = self.pressure_denorm = self.density_denorm = None
        if norm_metadata:
            np_s = norm_metadata.get('normalization_params', {})
            for vn in self.var_names:
                if vn in np_s:
                    self.denorm_params[vn] = np_s[vn]
            gp = norm_metadata.get('global_param_normalization', {})
            for k in ['delta_t', 'global_delta_t']:
                if k in gp: self.delta_t_denorm = gp[k]; break
            if self.delta_t_denorm is None and 'delta_t' in np_s:
                self.delta_t_denorm = np_s['delta_t']
            if 'pressure' in gp:
                self.pressure_denorm = gp['pressure']
            for k in ['density', 'density_param']:
                if k in gp: self.density_denorm = gp[k]; break

    def denorm_all(self, data_np):
        return denorm_all_from_params(data_np, self.var_names, self.denorm_params)

    def _get_global_metadata(self, data):
        ndt = float(data.global_delta_t[0]) if hasattr(data, 'global_delta_t') else 0.0
        npr = float(data.global_pressure[0]) if hasattr(data, 'global_pressure') else 0.0
        nrho = float(data.global_density[0]) if hasattr(data, 'global_density') else 0.0
        return {
            'delta_t_norm': ndt, 'pressure_norm': npr, 'density_norm': nrho,
            'delta_t': float(denorm_minmax(ndt, self.delta_t_denorm)),
            'pressure': float(denorm_minmax(npr, self.pressure_denorm)),
            'density': float(denorm_minmax(nrho, self.density_denorm)),
        }

    def _prep_simulation(self, simulation):
        simulation = [d.to(self.device) for d in simulation]
        for data in simulation:
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :self.model.num_static_feats]
            if hasattr(data, 'global_params') and not hasattr(data, 'global_pressure'):
                gp = data.global_params
                if gp.numel() >= 3:
                    data.global_pressure = gp[0].unsqueeze(0)
                    data.global_density = gp[1].unsqueeze(0)
                    data.global_delta_t = gp[2].unsqueeze(0)
        return simulation

    def generate_rollout(self, simulation, rollout_steps):
        preds = []
        sf = self.model.num_static_feats
        current = self.model._extract_dynamic(simulation[0].x)
        for step in range(rollout_steps):
            d = simulation[step]
            gf = self.model._extract_global_params(d)
            nf = torch.cat([d.x[:, :sf], current, gf], -1)
            pos = d.pos if hasattr(d, 'pos') and d.pos is not None else d.x[:, :2]
            ef = self.model.compute_edge_features(pos, d.edge_index)
            delta = self.model.model(nf, ef, d.edge_index)
            current = (current + delta).detach()
            preds.append(current.clone())
        return preds

    def generate_snapshot(self, simulation, num_steps):
        preds = []
        sf = self.model.num_static_feats
        for step in range(num_steps):
            d = simulation[step]
            gt = self.model._extract_dynamic(d.x)
            gf = self.model._extract_global_params(d)
            nf = torch.cat([d.x[:, :sf], gt, gf], -1)
            pos = d.pos if hasattr(d, 'pos') and d.pos is not None else d.x[:, :2]
            ef = self.model.compute_edge_features(pos, d.edge_index)
            delta = self.model.model(nf, ef, d.edge_index)
            preds.append(gt + delta)
        return preds

    def evaluate(self, simulations, mode='rollout', rollout_steps=10):
        results = {'predictions_physical': [], 'targets_physical': [],
                    'predictions_norm': [], 'targets_norm': [], 'metadata': []}
        self.simulation_metrics = []

        with torch.no_grad():
            for si, sim in enumerate(tqdm(simulations, desc=f"{mode.capitalize()} eval")):
                try:
                    sim = self._prep_simulation(sim)
                    steps = min(rollout_steps, len(sim)) if mode == 'rollout' else len(sim) - 1
                    raws = self.generate_rollout(sim, steps) if mode == 'rollout' \
                        else self.generate_snapshot(sim, steps)

                    pn = [p.cpu().numpy() for p in raws
                          if torch.isfinite(p).all() and p.abs().max() < 100]
                    if not pn: continue
                    tn = [self.model._apply_skip_to_target(sim[t].y).cpu().numpy()
                          for t in range(len(pn))]
                    pp = [self.denorm_all(p) for p in pn]
                    tp = [self.denorm_all(t) for t in tn]

                    meta = {'simulation_idx': si, 'case_name': f'sim_{si}',
                            'rollout_length': len(pn), 'num_nodes': sim[0].num_nodes,
                            **self._get_global_metadata(sim[0])}
                    results['predictions_norm'].append(pn)
                    results['targets_norm'].append(tn)
                    results['predictions_physical'].append(pp)
                    results['targets_physical'].append(tp)
                    results['metadata'].append(meta)

                    ap, at = np.concatenate(pp), np.concatenate(tp)
                    overall = {'rmse': float(np.sqrt(mean_squared_error(at, ap))),
                               'r2': float(r2_score(at.flatten(), ap.flatten()))}
                    per_var = {}
                    for vi, vn in enumerate(self.var_names):
                        per_var[vn] = {
                            'rmse': float(np.sqrt(mean_squared_error(at[:, vi], ap[:, vi]))),
                            'r2': float(r2_score(at[:, vi], ap[:, vi])),
                            'mae': float(mean_absolute_error(at[:, vi], ap[:, vi])),
                        }
                    self.simulation_metrics.append({
                        'metadata': meta, 'overall_physical': overall, 'per_variable': per_var})
                except Exception as e:
                    print(f"  Error sim {si}: {e}")
                    import traceback; traceback.print_exc()

        results['simulation_metrics'] = self.simulation_metrics
        return results

    def compute_benchmark_metrics(self, preds_phys, targs_phys):
        if not preds_phys: return {}
        ap = [p for sp in preds_phys for p in sp]
        at = [t for st in targs_phys for t in st]
        apc, atc = np.concatenate(ap), np.concatenate(at)
        rrv = compute_rrmse_per_variable(ap, at, self.var_names)
        m = {'RRMSE_total': compute_rrmse(ap, at),
             'overall_RMSE': float(np.sqrt(mean_squared_error(atc, apc))),
             'overall_R2': float(r2_score(atc.flatten(), apc.flatten())),
             'overall_MAE': float(mean_absolute_error(atc, apc))}
        for vn in self.var_names:
            m[f'RRMSE_{vn}'] = rrv.get(vn, float('inf'))
        return m


# =========================================================================
# DATA LOADING
# =========================================================================

def load_test_simulations(test_dir, test_files, pattern, max_files):
    paths = [Path(f) for f in test_files] if test_files else sorted(Path(test_dir).glob(pattern))
    if max_files: paths = paths[:max_files]
    sims = []
    for p in paths:
        try:
            sims.append(torch.load(p, weights_only=False))
            print(f"  {p.name}: {len(sims[-1])} timesteps")
        except Exception as e:
            print(f"  Error: {p}: {e}")
    return sims


# =========================================================================
# MAIN
# =========================================================================

def main():
    p = argparse.ArgumentParser(description="MeshGraphKAN Shock Tube Evaluation")
    p.add_argument("--model_path", required=True)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--test_dir", type=str)
    grp.add_argument("--test_files", type=str, nargs='+')
    p.add_argument("--output_dir", default="./eval_shocktube_mgkan")
    p.add_argument("--norm_metadata_file", default=None)
    p.add_argument("--eval_mode", default="rollout", choices=['rollout', 'snapshot', 'both'])
    p.add_argument("--rollout_steps", type=int, default=10)
    p.add_argument("--max_sequences", type=int, default=30)
    p.add_argument("--create_gifs", action="store_true")
    p.add_argument("--num_viz_simulations", type=int, default=3)
    p.add_argument("--gif_fps", type=int, default=4)
    p.add_argument("--gif_frame_skip", type=int, default=1)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output_dir); output_path.mkdir(exist_ok=True, parents=True)
    model_dir = Path(args.model_path).parent

    print(f"\n{'='*70}\nMeshGraphKAN SHOCK TUBE EVALUATION\n{'='*70}")

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    config = json.load(open(model_dir / "config.json")) if (model_dir / "config.json").exists() else {}

    norm_metadata = None
    if args.norm_metadata_file and Path(args.norm_metadata_file).exists():
        norm_metadata = json.load(open(args.norm_metadata_file))
    if norm_metadata is None and args.test_dir:
        norm_metadata = load_normalization_metadata(args.test_dir)
    if norm_metadata is None:
        norm_metadata = load_normalization_metadata(str(model_dir))

    model = build_meshgraphkan(config, device, domain='shocktube')
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    test_files = args.test_files if hasattr(args, 'test_files') and args.test_files else None
    simulations = load_test_simulations(args.test_dir, test_files, "*.pt", args.max_sequences)
    if not simulations: print("No simulations!"); return

    evaluator = ShockTubeMGKANEvaluator(model, device, norm_metadata=norm_metadata)

    def _run(mode):
        print(f"\n{'='*60}\n{mode.upper()} EVALUATION\n{'='*60}")
        results = evaluator.evaluate(simulations, mode=mode, rollout_steps=args.rollout_steps)
        metrics = evaluator.compute_benchmark_metrics(
            results['predictions_physical'], results['targets_physical'])

        print(f"\n{'-'*40}\n{mode.upper()} RESULTS\n{'-'*40}")
        for k, v in sorted(metrics.items()):
            print(f"  {k:>20}: {v:.4f}" if isinstance(v, float) else f"  {k:>20}: {v}")

        metrics.update({'model': 'MeshGraphKAN', 'parameters': total_params})
        with open(output_path / f'{mode}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        plot_shocktube_dashboard(evaluator.simulation_metrics,
                                  output_path / f'{mode}_dashboard.png',
                                  model_name='MeshGraphKAN', eval_mode=mode)
        plot_global_parameter_analysis(evaluator.simulation_metrics,
                                        output_path / f'{mode}_global_params.png')
        print(create_delta_t_performance_table(evaluator.simulation_metrics))

        fig = plot_prediction_scatter(results['predictions_physical'], results['targets_physical'],
                                       output_path=output_path / f'{mode}_scatter.png')
        if fig: plt.close(fig)
        if mode == 'rollout':
            fig = plot_rollout_error_growth(results['predictions_physical'], results['targets_physical'],
                                             output_path=output_path / f'{mode}_error_growth.png')
            if fig: plt.close(fig)

        if args.create_gifs:
            selected = select_representative_simulations(
                evaluator.simulation_metrics, n_samples=args.num_viz_simulations, sort_key='rmse')
            for idx in selected:
                si = evaluator.simulation_metrics[idx]['metadata']['simulation_idx']
                create_shocktube_visualizations(
                    results['predictions_physical'][idx], results['targets_physical'][idx],
                    si, output_path, model_name='MeshGraphKAN',
                    fps=args.gif_fps, frame_skip=args.gif_frame_skip, eval_mode=mode,
                    metadata=evaluator.simulation_metrics[idx]['metadata'])

        serializable = []
        for m in evaluator.simulation_metrics:
            entry = {**m['metadata'], **m['overall_physical']}
            for vn, vm in m.get('per_variable', {}).items():
                for mk, mv in vm.items(): entry[f'{vn}_{mk}'] = mv
            serializable.append(entry)
        with open(output_path / f'{mode}_per_simulation.json', 'w') as f:
            json.dump(serializable, f, indent=2)
        return metrics

    rm = sm = None
    if args.eval_mode in ['rollout', 'both']: rm = _run('rollout')
    if args.eval_mode in ['snapshot', 'both']: sm = _run('snapshot')
    if args.eval_mode == 'both' and rm and sm:
        print(f"\n{'='*60}\nSNAPSHOT vs ROLLOUT\n{'='*60}")
        for key in ['RRMSE_total', 'overall_R2', 'overall_RMSE']:
            sv, rv = sm.get(key, 0), rm.get(key, 0)
            ratio = rv / sv if sv != 0 else float('inf')
            print(f"  {key:<25} {sv:>12.4f} {rv:>12.4f} {ratio:>10.2f}x")
        with open(output_path / 'comparison_metrics.json', 'w') as f:
            json.dump({'snapshot': sm, 'rollout': rm}, f, indent=2)
    print(f"\nDone! Results: {output_path}")


if __name__ == "__main__":
    main()