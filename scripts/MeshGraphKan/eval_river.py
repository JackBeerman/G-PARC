#!/usr/bin/env python3
"""
MeshGraphKAN River Evaluation Script
=====================================
Evaluates MeshGraphKAN models on river (flood) test data.
Produces metrics compatible with G-PARCv2 comparison.

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
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings("ignore", category=UserWarning)

# ── Architecture ──────────────────────────────────────────────────────────
from models.meshgraphkan import MeshGraphKAN, MeshGraphKANRiverRollout, build_meshgraphkan

# ── Shared viz / metrics ─────────────────────────────────────────────────
from visualizations.metrics import compute_rrmse, compute_rrmse_per_variable
from visualizations.hydrology import nse
from visualizations.denorm import (
    denorm_all_from_params, load_denorm_extrema, denormalize_all,
)
from visualizations.river_viz import VAR_NAMES, create_river_visualizations
from visualizations.selection import select_representative_simulations


# =========================================================================
# EVALUATOR
# =========================================================================

class RiverMGKANEvaluator:
    def __init__(self, model, device='cpu', norm_metadata=None, extrema=None):
        self.model = model
        self.device = device
        self.model.to(device); self.model.eval()
        self.norm_metadata = norm_metadata
        self.extrema = extrema
        self.var_names = VAR_NAMES
        self.simulation_metrics = []

        # Prefer extrema .pth for denormalization (direct y_min/y_max)
        if extrema is not None:
            print(f"  Using extrema denormalization:")
            for v in range(min(len(VAR_NAMES), len(extrema['y_min']))):
                y_min = extrema['y_min'][v].item()
                y_max = extrema['y_max'][v].item()
                print(f"    {VAR_NAMES[v]}: [{y_min:.6f}, {y_max:.6f}]")

        # Fallback: JSON metadata min/max
        self.denorm_params = {}
        if norm_metadata and extrema is None:
            # Try multiple nesting levels
            np_s = norm_metadata.get('normalization_params',
                    norm_metadata.get('normalization_statistics',
                    norm_metadata.get('original_metadata', {}).get('normalization_statistics', {})))
            for vn in self.var_names:
                if vn in np_s:
                    self.denorm_params[vn] = np_s[vn]
            if self.denorm_params:
                print(f"  Using JSON denorm params for: {list(self.denorm_params.keys())}")
            else:
                print(f"  ⚠️  No denorm params found in metadata (keys: {list(np_s.keys()) if np_s else 'empty'})")

    def denorm_all(self, data_np):
        if self.extrema is not None:
            return denormalize_all(data_np, self.extrema)
        return denorm_all_from_params(data_np, self.var_names, self.denorm_params)

    def generate_rollout(self, simulation, rollout_steps):
        sf, df = self.model.num_static_feats, self.model.num_dynamic_feats
        current = simulation[0].x[:, sf:sf + df].clone()
        preds = []
        for step in range(rollout_steps):
            d = simulation[step]
            nf = torch.cat([d.x[:, :sf], current], -1)
            pos = d.pos if hasattr(d, 'pos') and d.pos is not None else d.x[:, :2]
            ef = self.model.compute_edge_features(pos, d.edge_index)
            delta = self.model.model(nf, ef, d.edge_index)
            current = (current + delta).detach()
            preds.append(current.clone())
        return preds

    def generate_snapshot(self, simulation, num_steps):
        sf, df = self.model.num_static_feats, self.model.num_dynamic_feats
        preds = []
        for step in range(num_steps):
            d = simulation[step]
            gt = d.x[:, sf:sf + df].clone()
            nf = torch.cat([d.x[:, :sf], gt], -1)
            pos = d.pos if hasattr(d, 'pos') and d.pos is not None else d.x[:, :2]
            ef = self.model.compute_edge_features(pos, d.edge_index)
            delta = self.model.model(nf, ef, d.edge_index)
            preds.append(gt + delta)
        return preds

    def evaluate(self, simulations, mode='rollout', rollout_steps=50):
        """
        Args:
            simulations: list of (name, [Data, ...]) tuples
        Returns:
            results dict with predictions, targets, sim metadata for viz
        """
        results = {'predictions_norm': [], 'targets_norm': [],
                    'predictions_physical': [], 'targets_physical': [],
                    'sim_names': [], 'sim_data': []}
        self.simulation_metrics = []
        sf, df = self.model.num_static_feats, self.model.num_dynamic_feats

        with torch.no_grad():
            for si, (name, sim) in enumerate(tqdm(simulations, desc=f"{mode} eval")):
                try:
                    sim = [d.to(self.device) for d in sim]
                    steps = min(rollout_steps, len(sim) - 1)
                    raws = self.generate_rollout(sim, steps) if mode == 'rollout' \
                        else self.generate_snapshot(sim, steps)

                    pn = [p.cpu().numpy() for p in raws]
                    tn = [sim[t].y[:, :df].cpu().numpy()
                          if hasattr(sim[t], 'y') and sim[t].y is not None
                          else sim[t + 1].x[:, sf:sf + df].cpu().numpy()
                          for t in range(steps)]

                    pp = [self.denorm_all(p) for p in pn]
                    tp = [self.denorm_all(t) for t in tn]

                    results['predictions_norm'].append(pn)
                    results['targets_norm'].append(tn)
                    results['predictions_physical'].append(pp)
                    results['targets_physical'].append(tp)
                    results['sim_names'].append(name)
                    results['sim_data'].append(sim)

                    ap, at = np.concatenate(pp), np.concatenate(tp)
                    depth_p = np.concatenate([p[:, 0] for p in pp])
                    depth_t = np.concatenate([t[:, 0] for t in tp])

                    per_var = {}
                    for vi, vn in enumerate(self.var_names):
                        if vi < ap.shape[1]:
                            per_var[vn] = {
                                'rmse': float(np.sqrt(np.mean((ap[:, vi] - at[:, vi]) ** 2))),
                                'r2': float(r2_score(at[:, vi], ap[:, vi])),
                            }
                    self.simulation_metrics.append({
                        'sim_idx': si, 'name': name,
                        'rmse': float(np.sqrt(mean_squared_error(at, ap))),
                        'r2': float(r2_score(at.flatten(), ap.flatten())),
                        'depth_nse': float(nse(depth_p, depth_t)),
                        'per_variable': per_var, 'rollout_steps': steps,
                    })
                except Exception as e:
                    print(f"Error sim {si}: {e}")
                    import traceback; traceback.print_exc()
        return results

    def compute_benchmark_metrics(self, preds_phys, targs_phys):
        all_p = [p for sp in preds_phys for p in sp]
        all_t = [t for st in targs_phys for t in st]
        apc, atc = np.concatenate(all_p), np.concatenate(all_t)
        rrv = compute_rrmse_per_variable(all_p, all_t, self.var_names)
        depth_p = np.concatenate([p[:, 0] for p in all_p])
        depth_t = np.concatenate([t[:, 0] for t in all_t])
        m = {'RRMSE_total': compute_rrmse(all_p, all_t),
             'overall_RMSE': float(np.sqrt(mean_squared_error(atc, apc))),
             'overall_R2': float(r2_score(atc.flatten(), apc.flatten())),
             'Depth_NSE': float(nse(depth_p, depth_t))}
        for vn, rv in rrv.items(): m[f'RRMSE_{vn}'] = rv
        return m

    def plot_dashboard(self, results, figsize=(16, 10)):
        if not self.simulation_metrics: return None
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        rmses = [m['rmse'] for m in self.simulation_metrics]
        r2s = [m['r2'] for m in self.simulation_metrics]
        nses = [m['depth_nse'] for m in self.simulation_metrics]

        ax = fig.add_subplot(gs[0, 0])
        ax.bar(range(len(rmses)), rmses, alpha=0.7)
        ax.set_xlabel('Simulation'); ax.set_ylabel('RMSE'); ax.set_title('Per-Simulation RMSE')

        ax = fig.add_subplot(gs[0, 1])
        ax.bar(range(len(r2s)), r2s, alpha=0.7, color='green')
        ax.set_xlabel('Simulation'); ax.set_ylabel('R²'); ax.set_title('Per-Simulation R²')

        ax = fig.add_subplot(gs[0, 2])
        colors = ['green' if n > 0.5 else 'orange' if n > 0 else 'red' for n in nses]
        ax.bar(range(len(nses)), nses, color=colors, alpha=0.7)
        ax.set_xlabel('Simulation'); ax.set_ylabel('NSE'); ax.set_title('Depth NSE')
        ax.axhline(y=0, color='gray', ls='--', alpha=0.5)

        ax = fig.add_subplot(gs[1, 0])
        all_p = [p for sp in results['predictions_physical'] for p in sp]
        all_t = [t for st in results['targets_physical'] for t in st]
        rrv = compute_rrmse_per_variable(all_p, all_t, self.var_names)
        ax.bar(rrv.keys(), rrv.values(), alpha=0.7, color='coral')
        ax.set_ylabel('RRMSE'); ax.set_title('RRMSE per Variable')
        ax.tick_params(axis='x', rotation=45)

        ax = fig.add_subplot(gs[1, 1:])
        max_t = max(len(sp) for sp in results['predictions_physical'])
        rmse_ot = np.full(max_t, np.nan); counts = np.zeros(max_t)
        for sp, st in zip(results['predictions_physical'], results['targets_physical']):
            for t in range(len(sp)):
                r = np.sqrt(np.mean((sp[t] - st[t]) ** 2))
                rmse_ot[t] = np.nanmean([rmse_ot[t], r]) if counts[t] > 0 else r
                counts[t] += 1
        valid = counts > 0
        ax.plot(np.where(valid)[0], rmse_ot[valid], 'b-o', markersize=3)
        ax.set_xlabel('Rollout Step'); ax.set_ylabel('Mean RMSE'); ax.set_title('Error Growth')
        ax.grid(True, alpha=0.3)

        fig.suptitle('MeshGraphKAN River Evaluation', fontsize=14, fontweight='bold')
        return fig


# =========================================================================
# DATA LOADING & MAIN
# =========================================================================

def load_simulations(test_dir, pattern="*.pt", max_sims=None):
    files = sorted(Path(test_dir).glob(pattern))
    if max_sims: files = files[:max_sims]
    result = []
    for f in files:
        data = torch.load(f, weights_only=False)
        if isinstance(data, list) and len(data) > 1:
            result.append((f.stem, data))
    print(f"Loaded {len(result)} simulations from {test_dir}")
    return result


def main():
    p = argparse.ArgumentParser(description="MeshGraphKAN River Evaluation")
    p.add_argument("--model_path", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--norm_metadata", default=None)
    p.add_argument("--output_dir", default="./eval_river_mgkan")
    p.add_argument("--eval_mode", default="rollout", choices=["rollout", "snapshot", "both"])
    p.add_argument("--rollout_steps", type=int, default=50)
    p.add_argument("--max_sequences", type=int, default=None)
    p.add_argument("--device", default="auto")

    # Visualization
    p.add_argument("--create_gifs", action="store_true")
    p.add_argument("--num_viz_simulations", type=int, default=3)
    p.add_argument("--gif_fps", type=int, default=5)
    p.add_argument("--gif_frame_skip", type=int, default=1)
    p.add_argument("--hec_ras_dir", type=str, default=None,
                   help="Path to HEC-RAS geometry dir for PolyCollection rendering")
    p.add_argument("--extrema_path", type=str, default=None,
                   help="Path to extrema .pth file with y_min/y_max for denormalization")

    args = p.parse_args()

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else args.device if args.device != 'auto' else 'cpu')
    output_path = Path(args.output_dir); output_path.mkdir(exist_ok=True, parents=True)

    ckpt_dir = os.path.dirname(args.model_path)
    cfg = json.load(open(os.path.join(ckpt_dir, 'config.json'))) \
        if os.path.exists(os.path.join(ckpt_dir, 'config.json')) else {}

    print(f"\n{'='*60}\nMeshGraphKAN River Evaluation\n{'='*60}")
    print(f"Device: {device}")

    model = build_meshgraphkan(cfg, device, domain='river')
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    total_params = sum(pr.numel() for pr in model.parameters())
    print(f"Parameters: {total_params:,}")

    norm_metadata = json.load(open(args.norm_metadata)) \
        if args.norm_metadata and os.path.exists(args.norm_metadata) else None

    # Load extrema .pth for denormalization (preferred over JSON)
    extrema = None
    if args.extrema_path:
        extrema = load_denorm_extrema(args.extrema_path)
    elif norm_metadata is None:
        # Auto-search for extrema in test_dir or model_dir
        for search_dir in [args.test_dir, os.path.dirname(args.model_path)]:
            for name in ['extrema.pth', 'global_extrema.pth', 'y_extrema.pth']:
                candidate = os.path.join(search_dir, name)
                if os.path.exists(candidate):
                    extrema = load_denorm_extrema(candidate)
                    break
            if extrema is not None:
                break

    simulations = load_simulations(args.test_dir, max_sims=args.max_sequences)
    if not simulations: print("No simulations!"); return

    evaluator = RiverMGKANEvaluator(model, device, norm_metadata=norm_metadata, extrema=extrema)

    for mode in (['rollout', 'snapshot'] if args.eval_mode == 'both' else [args.eval_mode]):
        print(f"\n{'='*60}\n{mode.upper()} EVALUATION (steps={args.rollout_steps})\n{'='*60}")
        results = evaluator.evaluate(simulations, mode=mode, rollout_steps=args.rollout_steps)
        metrics = evaluator.compute_benchmark_metrics(
            results['predictions_physical'], results['targets_physical'])

        print(f"\n{'-'*40}\n{mode.upper()} RESULTS\n{'-'*40}")
        for k, v in sorted(metrics.items()):
            print(f"  {k:>20}: {v:.4f}" if isinstance(v, float) else f"  {k:>20}: {v}")

        metrics.update({'model': 'MeshGraphKAN', 'parameters': total_params,
                        'num_simulations': len(simulations),
                        'simulation_metrics': [{k: v for k, v in m.items() if k != 'per_variable'}
                                                for m in evaluator.simulation_metrics]})
        with open(output_path / f'{mode}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Dashboard
        fig = evaluator.plot_dashboard(results)
        if fig:
            fig.savefig(output_path / f'{mode}_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Dashboard: {mode}_dashboard.png")

        # GIFs
        if args.create_gifs:
            # Build metrics in the format select_representative_simulations expects
            sim_metrics_for_selection = [
                {'metadata': {'simulation_idx': m['sim_idx']},
                 'overall_physical': {'rmse': m['rmse'], 'r2': m['r2']}}
                for m in evaluator.simulation_metrics
            ]
            selected = select_representative_simulations(
                sim_metrics_for_selection, n_samples=args.num_viz_simulations,
                sort_key='rmse')

            print(f"\n  Creating GIFs for {len(selected)} simulations: {selected}")
            for idx in selected:
                name = results['sim_names'][idx]
                sim_data = results['sim_data'][idx]
                seq_pred = results['predictions_physical'][idx]
                seq_targ = results['targets_physical'][idx]

                create_river_visualizations(
                    sim_data, seq_pred, seq_targ, name, output_path,
                    model_name='MeshGraphKAN', fps=args.gif_fps,
                    frame_skip=args.gif_frame_skip, eval_mode=mode,
                    hec_ras_dir=args.hec_ras_dir, extrema=extrema,
                )

    print(f"\nDone! Results: {output_path}")


if __name__ == "__main__":
    main()