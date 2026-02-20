#!/usr/bin/env python3
"""
Erosion Evaluation Script for G-PARCv2
=======================================
Evaluates the joint displacement + erosion head model.

Outputs:
  - Per-simulation erosion metrics (F1, precision, recall)
  - Aggregate metrics across all test sims
  - Erosion comparison plots: GT vs predicted over time
  - Erosion progression curves: GT count vs predicted count
  - Displacement RRMSE (to confirm displacement quality)
  - Summary JSON with all metrics
"""

import argparse
import os
import sys
from pathlib import Path
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.featureextractor import GraphConvFeatureExtractorV2
from differentiator.differentiator import ElastoPlasticDifferentiator
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from models.globalelasto import GPARC_ElastoPlastic_Numerical
from models.erosion_head import ErosionHead


# ==============================================================================
# UTILITIES
# ==============================================================================

def load_normalization_stats(data_dir, checkpoint_dir=None):
    """Load normalization stats from data dir or checkpoint dir."""
    for search_dir in [data_dir, checkpoint_dir]:
        if search_dir is None:
            continue
        for name in ["normalization_stats.json"]:
            stats_file = Path(search_dir) / name
            if not stats_file.exists():
                stats_file = Path(search_dir).parent / name
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
                print(f"✓ Loaded norm stats: {stats_file}")
                return stats
    raise FileNotFoundError("normalization_stats.json not found")


def get_pos_normalization_params(norm_stats):
    ps = norm_stats['position']
    return [ps['x_pos']['mean'], ps['y_pos']['mean']], [ps['x_pos']['std'], ps['y_pos']['std']]


def load_test_simulations(test_dir, max_sims=None):
    """Load all test simulations."""
    test_dir = Path(test_dir)
    files = sorted(test_dir.glob("*.pt"))
    if max_sims:
        files = files[:max_sims]
    
    sims = []
    for f in tqdm(files, desc="Loading test sims"):
        try:
            sim = torch.load(f, weights_only=False)
            if isinstance(sim, list) and len(sim) > 0:
                sims.append((f.stem, sim))
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    
    print(f"Loaded {len(sims)} test simulations")
    return sims


def create_model(args, norm_stats):
    """Build model with erosion head from args."""
    pos_mean, pos_std = get_pos_normalization_params(norm_stats)
    norm_method = norm_stats.get('normalization_method', 'z_score')
    max_position = None
    if norm_method == 'global_max' and 'position' in norm_stats:
        max_position = norm_stats['position'].get('max_position', None)

    gradient_solver = SolveGradientsLST(
        pos_mean=pos_mean, pos_std=pos_std,
        norm_method=norm_method, max_position=max_position
    )
    laplacian_solver = SolveWeightLST2d(
        pos_mean=pos_mean, pos_std=pos_std,
        norm_method=norm_method, max_position=max_position,
        min_neighbors=5
    )

    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=args.num_static_feats,
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layer_norm=args.use_layer_norm,
        use_relative_pos=args.use_relative_pos,
    )

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
        n_state_var=0,
        zero_init=args.zero_init,
    )

    # Erosion head dimensions
    n_explicit = 3 + int(args.use_von_mises) + int(args.use_volumetric) + args.num_dynamic_feats
    erosion_in = args.feature_out_channels + n_explicit + 1

    erosion_head = ErosionHead(
        in_features=erosion_in,
        hidden_dim=args.erosion_hidden_dim,
        num_layers=args.erosion_num_layers,
        dropout=args.erosion_dropout,
    )

    model = GPARC_ElastoPlastic_Numerical(
        derivative_solver_physics=derivative_solver,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        pos_mean=pos_mean,
        pos_std=pos_std,
        boundary_threshold=0.5,
        clamp_output=not args.no_clamp_output,
        norm_method=norm_method,
        max_position=max_position,
        erosion_head=erosion_head,
    )

    return model, norm_method, max_position


def denormalize(data, norm_method, norm_stats):
    """Denormalize displacement to physical units."""
    if norm_method == 'global_max':
        max_disp = norm_stats['displacement']['max_displacement']
        return data * max_disp
    return data


# ==============================================================================
# EROSION METRICS
# ==============================================================================

def compute_erosion_metrics(gt_eroded, pred_eroded):
    """Compute erosion classification metrics."""
    tp = (pred_eroded & gt_eroded).sum()
    fp = (pred_eroded & ~gt_eroded).sum()
    fn = (~pred_eroded & gt_eroded).sum()
    tn = (~pred_eroded & ~gt_eroded).sum()
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    
    return {
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
        'n_gt_eroded': int(gt_eroded.sum()), 'n_pred_eroded': int(pred_eroded.sum()),
    }


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_erosion_comparison(sim_name, pos_ref, elements, gt_erosion_list, pred_erosion_list,
                           timesteps, output_dir):
    """
    Side-by-side GT vs predicted erosion at selected timesteps.
    Top row: GT, Bottom row: predicted.
    """
    n_cols = len(timesteps)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    poly_verts = pos_ref[elements]
    
    for col, t in enumerate(timesteps):
        gt_eroded = gt_erosion_list[t]
        pred_eroded = pred_erosion_list[t]
        
        metrics = compute_erosion_metrics(gt_eroded, pred_eroded)
        
        for row, (eroded, label) in enumerate([
            (gt_eroded, 'Ground Truth'),
            (pred_eroded, 'Predicted'),
        ]):
            ax = axes[row, col]
            
            valid_v = poly_verts[~eroded]
            eroded_v = poly_verts[eroded]
            
            if len(valid_v) > 0:
                ax.add_collection(PolyCollection(
                    valid_v, facecolors='lightblue', edgecolors='k', linewidths=0.05))
            if len(eroded_v) > 0:
                ax.add_collection(PolyCollection(
                    eroded_v, facecolors='red', edgecolors='darkred', linewidths=0.1, alpha=0.8))
            
            ax.set_xlim(pos_ref[:, 0].min(), pos_ref[:, 0].max())
            ax.set_ylim(pos_ref[:, 1].min(), pos_ref[:, 1].max())
            ax.set_aspect('equal')
            ax.axis('off')
            
            if row == 0:
                ax.set_title(f't={t} — GT ({gt_eroded.sum()} eroded)', fontsize=10)
            else:
                ax.set_title(f't={t} — Pred ({pred_eroded.sum()} eroded)\n'
                             f'P={metrics["precision"]:.2f} R={metrics["recall"]:.2f} '
                             f'F1={metrics["f1"]:.2f}', fontsize=10)
        
        if col == 0:
            axes[0, col].set_ylabel('Ground Truth', fontsize=12, rotation=0,
                                    labelpad=80, va='center')
            axes[1, col].set_ylabel('Predicted', fontsize=12, rotation=0,
                                    labelpad=80, va='center')
    
    fig.suptitle(f'{sim_name} — Erosion Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f'erosion_comparison_{sim_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_erosion_progression(sim_name, gt_counts, pred_counts, output_dir):
    """Plot erosion count over time: GT vs predicted."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    t = range(len(gt_counts))
    ax.plot(t, gt_counts, 'r-o', markersize=3, linewidth=2, label='Ground Truth')
    ax.plot(t, pred_counts, 'b--s', markersize=3, linewidth=2, label='Predicted')
    ax.fill_between(t, gt_counts, alpha=0.15, color='red')
    ax.fill_between(t, pred_counts, alpha=0.15, color='blue')
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Eroded Elements', fontsize=12)
    ax.set_title(f'{sim_name} — Erosion Progression', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, len(gt_counts) - 1)
    ax.set_ylim(0, max(max(gt_counts), max(pred_counts)) * 1.1 + 1)
    
    fig.savefig(output_dir / f'erosion_progression_{sim_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_aggregate_summary(all_results, output_dir):
    """Summary plot across all simulations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: F1 per simulation
    ax = axes[0]
    names = [r['sim_name'] for r in all_results]
    f1s = [r['aggregate']['f1'] for r in all_results]
    colors = ['steelblue' if f > 0.3 else 'coral' for f in f1s]
    ax.barh(range(len(names)), f1s, color=colors, edgecolor='k', alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace('simulation_', 's') for n in names], fontsize=8)
    ax.set_xlabel('F1 Score')
    ax.set_title('Erosion F1 per Simulation')
    ax.set_xlim(0, 1)
    ax.axvline(np.mean(f1s), color='red', ls='--', label=f'Mean: {np.mean(f1s):.3f}')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')
    
    # Panel 2: Precision vs Recall scatter
    ax = axes[1]
    precs = [r['aggregate']['precision'] for r in all_results]
    recs = [r['aggregate']['recall'] for r in all_results]
    ax.scatter(recs, precs, c='steelblue', s=60, edgecolors='k', alpha=0.8)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    ax.grid(alpha=0.3)
    
    # Panel 3: GT vs Predicted erosion count at final timestep
    ax = axes[2]
    gt_final = [r['gt_eroded_final'] for r in all_results]
    pred_final = [r['pred_eroded_final'] for r in all_results]
    max_val = max(max(gt_final + [1]), max(pred_final + [1])) * 1.1
    ax.scatter(gt_final, pred_final, c='steelblue', s=60, edgecolors='k', alpha=0.8)
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect')
    ax.set_xlabel('GT Eroded (final step)')
    ax.set_ylabel('Predicted Eroded (final step)')
    ax.set_title('Erosion Count: GT vs Predicted')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle('Erosion Prediction Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'erosion_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EROSION EVALUATION")
    print("=" * 70)
    
    # Load norm stats
    norm_stats = load_normalization_stats(
        args.test_dir, checkpoint_dir=str(Path(args.model_path).parent))
    
    # Build and load model
    model, norm_method, max_position = create_model(args, norm_stats)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint: epoch {checkpoint.get('epoch', '?')}")
    if 'metrics' in checkpoint:
        print(f"  Checkpoint metrics: {checkpoint['metrics']}")
    
    # Load test data
    test_sims = load_test_simulations(args.test_dir, max_sims=args.max_sims)
    
    # Run evaluation
    all_results = []
    global_tp, global_fp, global_fn = 0, 0, 0
    global_disp_errors = []
    
    for sim_idx, (sim_name, sim) in enumerate(test_sims):
        print(f"\n--- {sim_name} ({len(sim)} timesteps) ---")
        
        # Move to device
        for data in sim:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            if hasattr(data, 'pos') and data.pos is not None:
                data.pos = data.pos.to(device)
            if hasattr(data, 'elements'):
                data.elements = data.elements.to(device)
            if hasattr(data, 'x_element') and data.x_element is not None:
                data.x_element = data.x_element.to(device)
        
        # Initialize MLS
        deriv = model.derivative_solver
        if hasattr(deriv, 'initialize_weights'):
            deriv.initialize_weights(sim[0])
        
        elements = sim[0].elements.cpu().numpy()
        pos_ref = sim[0].x[:, :args.num_static_feats].cpu().numpy()
        num_steps = len(sim) - 1
        
        # Rollout with erosion
        with torch.no_grad():
            result = model.rollout(sim, num_steps, device)
        
        if isinstance(result, tuple):
            states, erosion_preds = result
        else:
            states = result
            erosion_preds = [np.zeros(len(elements), dtype=bool)] * num_steps
        
        # Collect per-timestep metrics
        gt_erosion_list = []
        pred_erosion_list = []
        gt_counts = []
        pred_counts = []
        sim_tp, sim_fp, sim_fn = 0, 0, 0
        per_step_metrics = []
        
        for t in range(min(len(states), len(sim))):
            # GT erosion
            if hasattr(sim[t], 'x_element') and sim[t].x_element is not None:
                gt_eroded = (sim[t].x_element.cpu().numpy().flatten() < 0.5)
            else:
                gt_eroded = np.zeros(len(elements), dtype=bool)
            
            # Predicted erosion
            if t > 0 and t - 1 < len(erosion_preds):
                pred_eroded = erosion_preds[t - 1].astype(bool)
            else:
                pred_eroded = np.zeros(len(elements), dtype=bool)
            
            gt_erosion_list.append(gt_eroded)
            pred_erosion_list.append(pred_eroded)
            gt_counts.append(int(gt_eroded.sum()))
            pred_counts.append(int(pred_eroded.sum()))
            
            if gt_eroded.any():
                m = compute_erosion_metrics(gt_eroded, pred_eroded)
                per_step_metrics.append({'timestep': t, **m})
                sim_tp += m['tp']
                sim_fp += m['fp']
                sim_fn += m['fn']
        
        # Aggregate metrics for this sim
        sim_prec = sim_tp / max(sim_tp + sim_fp, 1)
        sim_rec = sim_tp / max(sim_tp + sim_fn, 1)
        sim_f1 = 2 * sim_prec * sim_rec / max(sim_prec + sim_rec, 1e-12)
        
        global_tp += sim_tp
        global_fp += sim_fp
        global_fn += sim_fn
        
        # Displacement RRMSE
        disp_errors = []
        for t in range(1, min(len(states), len(sim))):
            pred = states[t]
            gt = sim[t].x[:, args.num_static_feats:args.num_static_feats + args.num_dynamic_feats].cpu().numpy()
            
            # Mask eroded nodes
            if hasattr(sim[t], 'x_element') and sim[t].x_element is not None:
                gt_er = sim[t].x_element.cpu().numpy().flatten() < 0.5
                eroded_nodes = np.unique(elements[gt_er].flatten())
                valid = np.ones(pred.shape[0], dtype=bool)
                valid[eroded_nodes] = False
            else:
                valid = np.ones(pred.shape[0], dtype=bool)
            
            if valid.sum() > 0:
                diff = pred[valid] - gt[valid]
                disp_errors.append(diff)
        
        if disp_errors:
            all_diff = np.concatenate(disp_errors, axis=0)
            rrmse_num = np.sqrt(np.mean(all_diff ** 2))
            gt_all = []
            for t in range(1, min(len(states), len(sim))):
                gt = sim[t].x[:, args.num_static_feats:args.num_static_feats + args.num_dynamic_feats].cpu().numpy()
                gt_all.append(gt)
            gt_cat = np.concatenate(gt_all, axis=0)
            rrmse_den = np.sqrt(np.mean(gt_cat ** 2))
            rrmse = rrmse_num / max(rrmse_den, 1e-12)
        else:
            rrmse = float('inf')
        
        sim_result = {
            'sim_name': sim_name,
            'num_timesteps': len(sim),
            'aggregate': {
                'precision': float(sim_prec),
                'recall': float(sim_rec),
                'f1': float(sim_f1),
                'tp': sim_tp, 'fp': sim_fp, 'fn': sim_fn,
            },
            'per_step': per_step_metrics,
            'gt_eroded_final': int(gt_counts[-1]) if gt_counts else 0,
            'pred_eroded_final': int(pred_counts[-1]) if pred_counts else 0,
            'displacement_rrmse': float(rrmse),
        }
        all_results.append(sim_result)
        
        has_erosion = any(g > 0 for g in gt_counts)
        print(f"  Erosion F1={sim_f1:.3f} (P={sim_prec:.3f} R={sim_rec:.3f})"
              f"  GT: {gt_counts[-1]} eroded  Pred: {pred_counts[-1]}"
              f"  RRMSE={rrmse:.4f}")
        
        # ---- Visualizations ----
        if has_erosion and sim_idx < args.num_viz:
            # Find key timesteps: first erosion, mid, final
            first_ero_t = next((t for t, g in enumerate(gt_counts) if g > 0), 0)
            last_t = len(gt_counts) - 1
            mid_t = (first_ero_t + last_t) // 2
            viz_timesteps = sorted(set([first_ero_t, mid_t, last_t]))
            
            plot_erosion_comparison(
                sim_name, pos_ref, elements,
                gt_erosion_list, pred_erosion_list,
                viz_timesteps, output_dir
            )
            
            plot_erosion_progression(sim_name, gt_counts, pred_counts, output_dir)
            print(f"  Saved visualizations")
    
    # ---- Global metrics ----
    g_prec = global_tp / max(global_tp + global_fp, 1)
    g_rec = global_tp / max(global_tp + global_fn, 1)
    g_f1 = 2 * g_prec * g_rec / max(g_prec + g_rec, 1e-12)
    
    mean_rrmse = np.mean([r['displacement_rrmse'] for r in all_results
                          if r['displacement_rrmse'] < float('inf')])
    
    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 70}")
    print(f"Simulations evaluated: {len(all_results)}")
    print(f"Global erosion:  F1={g_f1:.4f}  P={g_prec:.4f}  R={g_rec:.4f}")
    print(f"  TP={global_tp}  FP={global_fp}  FN={global_fn}")
    print(f"Mean RRMSE (displacement): {mean_rrmse:.6f}")
    
    per_sim_f1 = [r['aggregate']['f1'] for r in all_results if r['aggregate']['f1'] > 0]
    if per_sim_f1:
        print(f"Per-sim erosion F1: mean={np.mean(per_sim_f1):.3f}, "
              f"min={np.min(per_sim_f1):.3f}, max={np.max(per_sim_f1):.3f}")
    
    # Summary plot
    if len(all_results) > 1:
        plot_aggregate_summary(all_results, output_dir)
    
    # Save JSON
    summary = {
        'global_erosion': {
            'precision': float(g_prec), 'recall': float(g_rec), 'f1': float(g_f1),
            'tp': global_tp, 'fp': global_fp, 'fn': global_fn,
        },
        'mean_displacement_rrmse': float(mean_rrmse),
        'per_simulation': all_results,
        'checkpoint': args.model_path,
        'checkpoint_epoch': checkpoint.get('epoch', None),
    }
    
    with open(output_dir / 'erosion_eval_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {output_dir / 'erosion_eval_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate G-PARC with Erosion Head")
    
    # Paths
    parser.add_argument("--model_path", required=True, help="Path to checkpoint")
    parser.add_argument("--test_dir", required=True, help="Test data directory")
    parser.add_argument("--output_dir", default="./eval_erosion", help="Output directory")
    
    # Architecture (must match training)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_layer_norm", action="store_true", default=True)
    parser.add_argument("--use_relative_pos", action="store_true", default=True)
    parser.add_argument("--no_clamp_output", action="store_true", default=True)
    parser.add_argument("--integrator", type=str, default="euler")
    parser.add_argument("--spade_heads", type=int, default=4)
    parser.add_argument("--spade_concat", action="store_true", default=True)
    parser.add_argument("--spade_dropout", type=float, default=0.1)
    parser.add_argument("--zero_init", action="store_true", default=True)
    parser.add_argument("--use_von_mises", action="store_true", default=True)
    parser.add_argument("--use_volumetric", action="store_true", default=True)
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=2)
    
    # Erosion head (must match training)
    parser.add_argument("--erosion_hidden_dim", type=int, default=64)
    parser.add_argument("--erosion_num_layers", type=int, default=2)
    parser.add_argument("--erosion_dropout", type=float, default=0.1)
    
    # Eval settings
    parser.add_argument("--max_sims", type=int, default=None, help="Max test sims to evaluate")
    parser.add_argument("--num_viz", type=int, default=5, help="Number of sims to visualize")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()