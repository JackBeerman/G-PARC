#!/usr/bin/env python3
"""
Evaluation Script for G-PARCv3 Elastoplastic — Erosion-Aware
=============================================================
Metrics:
  Displacement: RMSE, RRMSE (all nodes, valid-only, erosion-front)
  Erosion: F1, precision, recall, onset timing

Outputs:
  - eval_results.json with all metrics
  - Visualization PNGs (displacement field, erosion comparison)
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import numpy as np
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from utilities.featureextractor import GraphConvFeatureExtractorV2
from differentiator.differentiator_v3 import ElastoPlasticDifferentiatorV3
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from models.globalelasto_v3 import GPARC_ElastoPlastic_V3
from models.erosion_head import ErosionHead


# ===========================================================================
# METRICS
# ===========================================================================

def compute_rrmse(preds, targs):
    diff = np.concatenate([p - t for p, t in zip(preds, targs)])
    rmse = np.sqrt(np.mean(diff ** 2))
    rms_targ = np.sqrt(np.mean(np.concatenate(targs) ** 2))
    return rmse / max(rms_targ, 1e-12)


def get_erosion_front_mask(elements, erosion_elem, num_nodes):
    """
    Nodes at the erosion front: touching both eroded AND valid elements.
    These are the most physically interesting nodes.
    """
    elements_np = elements.cpu().numpy()
    eroded = erosion_elem.cpu().numpy() < 0.5

    nodes_touching_eroded = set()
    nodes_touching_valid = set()

    for m in range(len(elements_np)):
        for node in elements_np[m]:
            if eroded[m]:
                nodes_touching_eroded.add(node)
            else:
                nodes_touching_valid.add(node)

    front_nodes = nodes_touching_eroded & nodes_touching_valid
    mask = np.zeros(num_nodes, dtype=bool)
    for n in front_nodes:
        mask[n] = True
    return mask


# ===========================================================================
# MODEL LOADING
# ===========================================================================

def load_model(args, sample_data, device):
    """Load V3 model from checkpoint."""
    # Load config
    ckpt_dir = Path(args.checkpoint).parent
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"  Loaded config from {config_path}")
    else:
        config = {}

    # Normalization
    norm_stats_path = ckpt_dir / "normalization_stats.json"
    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
    else:
        norm_stats = {}

    pos_stats = norm_stats.get('position', {})
    norm_method = norm_stats.get('normalization_method', 'z_score')
    pos_mean = pos_stats.get('mean')
    pos_std = pos_stats.get('std')
    max_position = None
    if norm_method == 'global_max' and 'position' in norm_stats:
        max_position = norm_stats['position'].get('max_position')

    sf = config.get('num_static_feats', args.num_static_feats)
    df = config.get('num_dynamic_feats', args.num_dynamic_feats)
    feat_out = config.get('feature_out_channels', args.feature_out_channels)

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
        in_channels=sf,
        hidden_channels=config.get('hidden_channels', 128),
        out_channels=feat_out,
        num_layers=config.get('num_layers', 4),
        dropout=config.get('dropout', 0.0),
        use_layer_norm=config.get('use_layer_norm', True),
        use_relative_pos=config.get('use_relative_pos', True),
    )

    derivative_solver = ElastoPlasticDifferentiatorV3(
        num_static_feats=sf,
        num_dynamic_feats=df,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=feat_out,
        list_strain_idx=list(range(df)),
        list_laplacian_idx=list(range(df)),
        spade_random_noise=config.get('spade_random_noise', False),
        heads=config.get('spade_heads', 4),
        concat=config.get('spade_concat', True),
        dropout=config.get('spade_dropout', 0.1),
        use_von_mises=config.get('use_von_mises', True),
        use_volumetric=config.get('use_volumetric', True),
        n_state_var=config.get('n_state_var', 0),
        zero_init=config.get('zero_init', True),
    )

    if not hasattr(sample_data, 'pos') or sample_data.pos is None:
        sample_data.pos = sample_data.x[:, :sf]
    derivative_solver.initialize_weights(sample_data)

    # Erosion head dimensions
    n_explicit = 3 + int(config.get('use_von_mises', True)) + int(config.get('use_volumetric', True))
    n_explicit += df  # Laplacian
    n_explicit += 1   # erosion SPADE channel
    erosion_in = feat_out + n_explicit + 1

    erosion_head = ErosionHead(
        in_features=erosion_in,
        hidden_dim=config.get('erosion_hidden_dim', 64),
        num_layers=config.get('erosion_num_layers', 2),
        dropout=config.get('erosion_dropout', 0.1),
    )

    model = GPARC_ElastoPlastic_V3(
        derivative_solver=derivative_solver,
        erosion_head=erosion_head,
        integrator_type=config.get('integrator', 'euler'),
        num_static_feats=sf,
        num_dynamic_feats=df,
        pos_mean=pos_mean,
        pos_std=pos_std,
        boundary_threshold=0.5,
        clamp_output=config.get('no_clamp_output', True),
        norm_method=norm_method,
        max_position=max_position,
        erosion_threshold=config.get('erosion_threshold', 0.5),
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  ✓ V3 model loaded (epoch {ckpt.get('epoch', '?')})")
    return model


# ===========================================================================
# EVALUATION
# ===========================================================================

def evaluate_simulation(model, sim, device, num_steps=None):
    """Evaluate one simulation."""
    sf = model.num_static_feats
    df = model.num_dynamic_feats

    # Move to device
    for data in sim:
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        if hasattr(data, 'pos') and data.pos is not None:
            data.pos = data.pos.to(device)
        else:
            data.pos = data.x[:, :sf].to(device)
        if hasattr(data, 'elements'):
            data.elements = data.elements.to(device)
        if hasattr(data, 'x_element') and data.x_element is not None:
            data.x_element = data.x_element.to(device)

    steps = min(num_steps or len(sim) - 1, len(sim) - 1)

    with torch.no_grad():
        states, erosion_preds = model.rollout(sim, steps, device=device)

    # GT targets
    gt_states = []
    gt_erosion = []
    for t in range(steps):
        if hasattr(sim[t], 'y') and sim[t].y is not None:
            gt_states.append(sim[t].y.cpu().numpy())
        else:
            gt_states.append(sim[t + 1].x[:, sf:sf + df].cpu().numpy())

        if t + 1 < len(sim) and hasattr(sim[t + 1], 'x_element') and sim[t + 1].x_element is not None:
            gt_erosion.append((sim[t + 1].x_element.cpu().numpy().flatten() < 0.5))
        elif hasattr(sim[t], 'x_element') and sim[t].x_element is not None:
            gt_erosion.append((sim[t].x_element.cpu().numpy().flatten() < 0.5))
        else:
            gt_erosion.append(np.zeros(sim[t].elements.shape[0], dtype=bool))

    # Predicted states start from index 1 (index 0 is initial state)
    pred_states = states[1:steps + 1]

    # ---- Displacement metrics ----
    all_rmse = np.sqrt(np.mean(
        np.concatenate([(p - g).flatten() ** 2
                        for p, g in zip(pred_states, gt_states)])))
    all_rrmse = compute_rrmse(pred_states, gt_states)

    # Valid-only and erosion-front metrics
    elements = sim[0].elements
    valid_rmse_list, front_rmse_list = [], []

    for t, (pred, gt, gt_ero) in enumerate(zip(pred_states, gt_states, gt_erosion)):
        # Current erosion state
        if t < len(erosion_preds) - 1:
            curr_ero = erosion_preds[t + 1]
        else:
            curr_ero = erosion_preds[-1]

        # Valid nodes: not touching any eroded element
        erosion_elem_t = torch.tensor(~gt_ero, dtype=torch.float32)  # 1=valid
        front_mask = get_erosion_front_mask(elements.cpu(), erosion_elem_t, pred.shape[0])

        valid_mask = np.ones(pred.shape[0], dtype=bool)
        eroded_elems = gt_ero
        if eroded_elems.any():
            eroded_nodes = np.unique(elements.cpu().numpy()[eroded_elems].flatten())
            valid_mask[eroded_nodes] = False

        if valid_mask.sum() > 0:
            valid_rmse_list.append(np.sqrt(np.mean((pred[valid_mask] - gt[valid_mask]) ** 2)))
        if front_mask.sum() > 0:
            front_rmse_list.append(np.sqrt(np.mean((pred[front_mask] - gt[front_mask]) ** 2)))

    # ---- Erosion metrics ----
    total_tp, total_fp, total_fn = 0, 0, 0
    first_erosion_gt, first_erosion_pred = None, None

    for t, (pred_ero, gt_ero) in enumerate(zip(erosion_preds[1:], gt_erosion)):
        tp = (pred_ero & gt_ero).sum()
        fp = (pred_ero & ~gt_ero).sum()
        fn = (~pred_ero & gt_ero).sum()
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if first_erosion_gt is None and gt_ero.any():
            first_erosion_gt = t
        if first_erosion_pred is None and pred_ero.any():
            first_erosion_pred = t

    e_prec = total_tp / max(total_tp + total_fp, 1)
    e_rec = total_tp / max(total_tp + total_fn, 1)
    e_f1 = 2 * e_prec * e_rec / max(e_prec + e_rec, 1e-8)

    return {
        'disp_rmse_all': float(all_rmse),
        'disp_rrmse_all': float(all_rrmse),
        'disp_rmse_valid': float(np.mean(valid_rmse_list)) if valid_rmse_list else float('nan'),
        'disp_rmse_front': float(np.mean(front_rmse_list)) if front_rmse_list else float('nan'),
        'erosion_f1': float(e_f1),
        'erosion_precision': float(e_prec),
        'erosion_recall': float(e_rec),
        'erosion_onset_gt': first_erosion_gt,
        'erosion_onset_pred': first_erosion_pred,
        'num_steps': steps,
        'pred_states': pred_states,
        'gt_states': gt_states,
        'erosion_preds': erosion_preds,
        'gt_erosion': gt_erosion,
    }


# ===========================================================================
# VISUALIZATION
# ===========================================================================

def plot_erosion_comparison(sim, results, sim_name, output_dir):
    """Erosion GT vs predicted at key timesteps."""
    elements = sim[0].elements.cpu().numpy()
    pos = sim[0].x[:, :2].cpu().numpy()
    gt_ero = results['gt_erosion']
    pred_ero = results['erosion_preds'][1:]
    T = len(gt_ero)

    # Pick 3 timesteps: first erosion, mid, final
    first_t = results['erosion_onset_gt'] or 0
    mid_t = (first_t + T) // 2
    final_t = T - 1
    timesteps = [first_t, mid_t, final_t]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, t in enumerate(timesteps):
        if t >= T:
            t = T - 1

        for row, (ero_data, label) in enumerate([(gt_ero[t], 'GT'), (pred_ero[t], 'Predicted')]):
            ax = axes[row, col]
            colors = np.where(ero_data, [1, 0, 0, 0.8], [0.2, 0.5, 1.0, 0.5])

            from matplotlib.collections import PolyCollection
            verts = pos[elements]
            pc = PolyCollection(verts, facecolors=colors, edgecolors='gray',
                               linewidths=0.1)
            ax.add_collection(pc)
            ax.autoscale_view()
            ax.set_aspect('equal')
            ax.set_title(f'{label} t={t} ({ero_data.sum()} eroded)', fontsize=10)

    fig.suptitle(f'{sim_name} — Erosion Comparison\n'
                 f'F1={results["erosion_f1"]:.3f}, '
                 f'P={results["erosion_precision"]:.3f}, '
                 f'R={results["erosion_recall"]:.3f}', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / f'erosion_{sim_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate G-PARCv3")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_v3")
    parser.add_argument("--max_sims", type=int, default=None)
    parser.add_argument("--num_viz", type=int, default=5)
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=2)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load test data
    test_dir = Path(args.test_dir)
    files = sorted(test_dir.glob("*.pt"))
    if args.max_sims:
        files = files[:args.max_sims]

    sims = []
    for f in tqdm(files, desc="Loading test data"):
        try:
            sim = torch.load(f, weights_only=False)
            if isinstance(sim, list) and len(sim) > 0:
                sims.append((f.stem, sim))
        except Exception as e:
            print(f"  Error: {f}: {e}")

    print(f"Loaded {len(sims)} test simulations")

    # Load model
    sample_data = sims[0][1][0]
    model = load_model(args, sample_data, device)

    # Evaluate
    all_results = {}

    for idx, (name, sim) in enumerate(tqdm(sims, desc="Evaluating")):
        try:
            r = evaluate_simulation(model, sim, device, args.rollout_steps)
            all_results[name] = {k: v for k, v in r.items()
                                  if k not in ('pred_states', 'gt_states',
                                               'erosion_preds', 'gt_erosion')}

            print(f"  {name}: RRMSE={r['disp_rrmse_all']:.4f}, "
                  f"RMSE_front={r['disp_rmse_front']:.4f}, "
                  f"F1={r['erosion_f1']:.3f}")

            # Visualize first N
            if idx < args.num_viz:
                plot_erosion_comparison(sim, r, name, output_dir)

        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback; traceback.print_exc()

    # Aggregate
    if all_results:
        agg = {}
        for key in ['disp_rmse_all', 'disp_rrmse_all', 'disp_rmse_valid',
                     'disp_rmse_front', 'erosion_f1', 'erosion_precision',
                     'erosion_recall']:
            vals = [r[key] for r in all_results.values()
                    if not np.isnan(r.get(key, np.nan))]
            agg[key] = float(np.mean(vals)) if vals else float('nan')

        print(f"\n{'=' * 60}")
        print("AGGREGATE RESULTS")
        print(f"{'=' * 60}")
        print(f"  Disp RRMSE (all nodes):    {agg['disp_rrmse_all']:.4f}")
        print(f"  Disp RMSE (all nodes):     {agg['disp_rmse_all']:.6f}")
        print(f"  Disp RMSE (valid only):    {agg['disp_rmse_valid']:.6f}")
        print(f"  Disp RMSE (erosion front): {agg['disp_rmse_front']:.6f}")
        print(f"  Erosion F1:                {agg['erosion_f1']:.3f}")
        print(f"  Erosion Precision:         {agg['erosion_precision']:.3f}")
        print(f"  Erosion Recall:            {agg['erosion_recall']:.3f}")

        summary = {'aggregate': agg, 'per_simulation': all_results}
        with open(output_dir / 'eval_results.json', 'w') as f:
            json.dump(summary, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, np.floating) else None)

        print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
