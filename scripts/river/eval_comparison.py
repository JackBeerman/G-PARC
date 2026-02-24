#!/usr/bin/env python3
"""
Unified River Model Comparison Evaluation
==========================================
Evaluates multiple models (G-PARCv1, G-PARCv2, MeshGraphKAN, MeshGraphNet)
on the same test data with identical metrics.

Outputs:
  - Comparison table: RMSE, RRMSE, NSE, CSI, Mass Balance per model
  - Per-variable timeseries: GT vs all models with shading
  - Per-segment breakdown (if segments provided)
  - Summary JSON with all metrics
  - Per-model rollout GIFs (optional)

Usage:
    python eval_comparison.py \
        --test_dir /path/to/test \
        --extrema_path /path/to/extrema.pt \
        --models gparcv2:/path/to/ckpt gparcv1:/path/to/ckpt mgkan:/path/to/ckpt mgnet:/path/to/ckpt \
        --output_dir ./comparison_results
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ==============================================================================
# VARIABLE NAMES & CONSTANTS
# ==============================================================================

VAR_NAMES = ['Depth', 'Volume', 'Velocity_X', 'Velocity_Y']
VAR_UNITS = ['m', 'm³', 'm/s', 'm/s']


# ==============================================================================
# DENORMALIZATION
# ==============================================================================

def load_denorm_extrema(extrema_path):
    extrema_path = Path(extrema_path)
    if not extrema_path.exists():
        print(f"  ⚠️  Extrema not found: {extrema_path}")
        return None
    extrema = torch.load(extrema_path, weights_only=False)
    print(f"  ✓ Loaded extrema: y_min={extrema['y_min'].tolist()}, y_max={extrema['y_max'].tolist()}")
    return extrema


def denormalize_all(normalized, extrema):
    if extrema is None:
        return normalized
    physical = np.zeros_like(normalized)
    for v in range(normalized.shape[1]):
        y_min = extrema['y_min'][v].item()
        y_max = extrema['y_max'][v].item()
        physical[:, v] = normalized[:, v] * (y_max - y_min) + y_min
    return physical


# ==============================================================================
# HYDROLOGY METRICS
# ==============================================================================

def nse(pred, obs):
    """Nash-Sutcliffe Efficiency."""
    num = np.sum((pred - obs) ** 2)
    den = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - num / den if den > 0 else np.nan


def csi(pred, obs, threshold):
    """Critical Success Index for threshold exceedance."""
    hits = np.sum((pred > threshold) & (obs > threshold))
    misses = np.sum((pred <= threshold) & (obs > threshold))
    fa = np.sum((pred > threshold) & (obs <= threshold))
    denom = hits + misses + fa
    return hits / denom if denom > 0 else np.nan


def mass_balance_error(vol_pred, vol_prev, inflow, dt):
    """Squared mass balance error."""
    return (vol_pred - vol_prev - dt * inflow) ** 2


def compute_rrmse(preds, targs):
    """Relative RMSE = RMSE / RMS(target)."""
    diff = np.concatenate([p - t for p, t in zip(preds, targs)])
    rmse = np.sqrt(np.mean(diff ** 2))
    targ_cat = np.concatenate(targs)
    rms_targ = np.sqrt(np.mean(targ_cat ** 2))
    return rmse / max(rms_targ, 1e-12)


def get_important_mask(simulation, num_static_feats, depth_idx, threshold, extrema):
    """Nodes where depth exceeds threshold at ANY timestep."""
    n = simulation[0].x.size(0)
    mask = np.zeros(n, dtype=bool)
    if extrema is None:
        return mask
    y_min = extrema['y_min'][depth_idx].item()
    y_max = extrema['y_max'][depth_idx].item()
    for g in simulation:
        norm_vals = g.x[:, num_static_feats + depth_idx].cpu().numpy()
        phys_vals = norm_vals * (y_max - y_min) + y_min
        mask |= (phys_vals > threshold)
        if hasattr(g, 'y') and g.y is not None and g.y.shape[1] > depth_idx:
            norm_y = g.y[:, depth_idx].cpu().numpy()
            phys_y = norm_y * (y_max - y_min) + y_min
            mask |= (phys_y > threshold)
    return mask


def compute_all_metrics(preds_phys, targs_phys, depth_threshold=0.3):
    """Compute full metric suite for one model on one simulation."""
    depth_p = np.concatenate([p[:, 0] for p in preds_phys])
    depth_t = np.concatenate([t[:, 0] for t in targs_phys])
    
    all_p = np.concatenate(preds_phys)
    all_t = np.concatenate(targs_phys)
    
    # Per-variable RMSE
    per_var = {}
    for vi, vn in enumerate(VAR_NAMES):
        if vi < all_p.shape[1]:
            per_var[vn] = {
                'rmse': float(np.sqrt(np.mean((all_p[:, vi] - all_t[:, vi]) ** 2))),
            }
    
    # Mass balance percentage (volume = index 1)
    mass_balance_pct = np.nan
    if all_p.shape[1] > 1:
        vol_pct_errors = []
        for t in range(len(preds_phys)):
            vol_pred = preds_phys[t][:, 1].sum()
            vol_gt = targs_phys[t][:, 1].sum()
            if abs(vol_gt) > 1e-12:
                vol_pct_errors.append(abs(vol_pred - vol_gt) / abs(vol_gt) * 100)
        if vol_pct_errors:
            mass_balance_pct = float(np.mean(vol_pct_errors))
    
    return {
        'rmse': float(np.sqrt(np.mean((all_p - all_t) ** 2))),
        'rrmse': float(compute_rrmse(preds_phys, targs_phys)),
        'depth_rmse': float(np.sqrt(np.mean((depth_p - depth_t) ** 2))),
        'depth_nse': float(nse(depth_p, depth_t)),
        'depth_csi': float(csi(depth_p, depth_t, depth_threshold)),
        'mass_balance_pct': mass_balance_pct,
        'per_variable': per_var,
    }


def compute_segmented_metrics(preds_phys, targs_phys, segments, important_mask=None,
                               depth_threshold=0.3):
    """Compute per-segment metrics, optionally split by important/non-important nodes."""
    T = len(preds_phys)
    result = {}
    
    groups = {'All_Nodes': None}
    if important_mask is not None:
        groups['Important'] = important_mask
        groups['Non_Important'] = ~important_mask
    
    for grp_name, mask in groups.items():
        seg_metrics = {}
        for seg_str in segments:
            parts = seg_str.split(':')
            s, e = int(parts[0]), int(parts[1])
            e_actual = min(e, T)
            if s >= T:
                continue
            
            key = f"t{s}-{e}"
            depth_pred_list, depth_targ_list = [], []
            
            for t in range(s, e_actual):
                dp = preds_phys[t][:, 0]
                dt_arr = targs_phys[t][:, 0]
                if mask is not None:
                    dp = dp[mask]
                    dt_arr = dt_arr[mask]
                depth_pred_list.append(dp)
                depth_targ_list.append(dt_arr)
            
            if not depth_pred_list:
                continue
            
            dp_cat = np.concatenate(depth_pred_list)
            dg_cat = np.concatenate(depth_targ_list)
            
            seg_metrics[key] = {
                'Time_Range': (s, e_actual - 1),
                'RMSE': float(np.sqrt(np.mean((dp_cat - dg_cat) ** 2))),
                'NSE': float(nse(dp_cat, dg_cat)),
                'CSI': float(csi(dp_cat, dg_cat, depth_threshold)),
            }
        
        result[grp_name] = seg_metrics
    
    return result


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_model_gparcv2(checkpoint_path, device, sf=9, df=4):
    """Load G-PARCv2 river model."""
    from models.riverV2 import GPARC_River_V2
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from differentiator.riverdifferentiator import RiverDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    
    # Load config from checkpoint directory
    ckpt_dir = Path(checkpoint_path).parent
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"  Loaded config from {config_path}")
    else:
        config = {}
    
    hidden = config.get('hidden_channels', 128)
    n_layers = config.get('num_layers', 4)
    feat_out = config.get('feature_out_channels', 128)
    
    gradient_solver = SolveGradientsLST()
    laplacian_solver = SolveWeightLST2d(use_2hop_extension=False)
    
    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=sf+df,
        hidden_channels=hidden,
        out_channels=feat_out,
        num_layers=n_layers,
        use_layer_norm=config.get('use_layer_norm', True),
        use_relative_pos=config.get('use_relative_pos', True),
    )
    
    differentiator = RiverDifferentiator(
        num_static_feats=sf,
        num_dynamic_feats=df,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=feat_out,
    )
    
    model = GPARC_River_V2(
        derivative_solver_physics=differentiator,
        num_static_feats=sf,
        num_dynamic_feats=df,
    )
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  ✓ G-PARCv2 loaded (epoch {ckpt.get('epoch', '?')})")
    return model


def load_model_gparcv1(checkpoint_path, device, sf=9, df=4):
    """Load G-PARCv1 river model."""
    from scripts.river.gparc_new import GPARCRecurrent, FeatureExtractorGNN, DerivativeGNN, IntegralGNN
    
    fe_out = 128
    fe = FeatureExtractorGNN(
        in_channels=sf, hidden_channels=64, out_channels=fe_out,
        depth=2, pool_ratios=0.1, heads=4, concat=True, dropout=0.2
    )
    de = DerivativeGNN(
        in_channels=fe_out + df, out_channels=df,
        heads=4, concat=True, dropout=0.2
    )
    ie = IntegralGNN(
        in_channels=df, out_channels=df,
        heads=4, concat=True, dropout=0.2
    )
    
    model = GPARCRecurrent(fe, de, ie, num_static_feats=sf, num_dynamic_feats=df)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    print(f"  ✓ G-PARCv1 loaded")
    return model


def load_model_mgkan(checkpoint_path, device, sf=9, df=4):
    """Load MeshGraphKAN river model."""
    from scripts.MeshGraphKan.train_river import MeshGraphKAN, MeshGraphKANRollout
    
    ckpt_dir = Path(checkpoint_path).parent
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {'hidden_dim': 128, 'processor_size': 4, 'num_harmonics': 5}
    
    inner = MeshGraphKAN(
        input_dim_nodes=sf + df,
        input_dim_edges=3,
        output_dim=df,
        hidden_dim_processor=config.get('hidden_dim', 128),
        hidden_dim_node_encoder=config.get('hidden_dim', 128),
        hidden_dim_edge_encoder=config.get('hidden_dim', 128),
        hidden_dim_node_decoder=config.get('hidden_dim', 128),
        processor_size=config.get('processor_size', 4),
        num_harmonics=config.get('num_harmonics', 5),
        aggregation=config.get('aggregation', 'sum'),
    )
    
    model = MeshGraphKANRollout(inner, num_static_feats=sf, num_dynamic_feats=df)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  ✓ MeshGraphKAN loaded (epoch {ckpt.get('epoch', '?')})")
    return model


def load_model_mgnet(checkpoint_path, device, sf=9, df=4):
    """Load MeshGraphNet river model."""
    from models.meshgraphnet import MeshGraphNet
    
    ckpt_dir = Path(checkpoint_path).parent
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {'hidden_dim': 128, 'num_layers': 10}
    
    model = MeshGraphNet(
        input_dim_node=sf + df,
        input_dim_edge=3,
        output_dim=df,
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', config.get('processor_size', 10)),
    )
    model.num_static_feats = sf
    model.num_dynamic_feats = df
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    print(f"  ✓ MeshGraphNet loaded (epoch {ckpt.get('epoch', '?')})")
    return model


MODEL_LOADERS = {
    'gparcv2': load_model_gparcv2,
    'gparcv1': load_model_gparcv1,
    'mgkan': load_model_mgkan,
    'mgnet': load_model_mgnet,
}


# ==============================================================================
# ROLLOUT FUNCTIONS (model-specific)
# ==============================================================================

def rollout_gparcv2(model, simulation, rollout_steps, device):
    """G-PARCv2: uses model.step() interface."""
    sf = model.num_static_feats
    df = model.num_dynamic_feats
    
    # Ensure pos is set (MLS needs it)
    for d in simulation:
        if not hasattr(d, 'pos') or d.pos is None:
            d.pos = d.x[:, :2]
    
    # Force MLS re-initialization for each simulation
    # (different river meshes have different topologies)
    deriv = model.derivative_solver
    if hasattr(deriv, '_weights_initialized'):
        deriv._weights_initialized = False
    # Clear ALL solver caches (gradient uses geo_cache, laplacian uses weights_cache)
    if hasattr(deriv, 'gradient_solver'):
        gs = deriv.gradient_solver
        if hasattr(gs, 'clear_caches'):
            gs.clear_caches()
        elif hasattr(gs, 'geo_cache'):
            gs.geo_cache.clear()
    if hasattr(deriv, 'laplacian_solver'):
        ls = deriv.laplacian_solver
        if hasattr(ls, 'clear_caches'):
            ls.clear_caches()
        elif hasattr(ls, 'weights_cache'):
            ls.weights_cache.clear()
    if hasattr(deriv, 'initialize_weights'):
        deriv.initialize_weights(simulation[0])
    
    current = simulation[0].x[:, sf:sf + df].clone()
    static = simulation[0].x[:, :sf]
    edge_index = simulation[0].edge_index
    
    if hasattr(simulation[0], 'mesh_id'):
        edge_index.mesh_id = simulation[0].mesh_id
    
    preds = []
    for step in range(rollout_steps):
        F_pred = model.step(
            static_feats=static,
            dynamic_state=current,
            edge_index=edge_index,
            dt=1.0,
        )
        preds.append(F_pred.cpu().numpy())
        current = F_pred
    return preds


def rollout_gparcv1(model, simulation, rollout_steps, device):
    """G-PARCv1: manual rollout matching forward() logic."""
    sf = model.num_static_feats
    df = model.num_dynamic_feats
    
    static = simulation[0].x[:, :sf]
    edge_index = simulation[0].edge_index
    learned_feats = model.feature_extractor(static, edge_index)
    
    current = simulation[0].x[:, sf:sf + df].clone()
    preds = []
    
    for step in range(rollout_steps):
        Fdot_input = torch.cat([current, learned_feats], dim=-1)
        Fdot = model.derivative_solver(Fdot_input, edge_index)
        Fint = model.integral_solver(Fdot, edge_index)
        current = current + Fint
        preds.append(current.detach().cpu().numpy())
    
    return preds


def rollout_mgkan(model, simulation, rollout_steps, device):
    """MeshGraphKAN: delta-based rollout."""
    sf = model.num_static_feats
    df = model.num_dynamic_feats
    
    first = simulation[0]
    static = first.x[:, :sf]
    current = first.x[:, sf:sf + df].clone()
    edge_index = first.edge_index
    ef = model.compute_edge_features(first)
    
    preds = []
    for step in range(rollout_steps):
        nf = torch.cat([static, current], dim=-1)
        delta = model.model(nf, ef, edge_index)
        current = (current + delta).detach()
        preds.append(current.cpu().numpy())
    return preds


def rollout_mgnet(model, simulation, rollout_steps, device):
    """MeshGraphNet: delta-based rollout."""
    sf = model.num_static_feats
    df = model.num_dynamic_feats
    
    first = simulation[0]
    static = first.x[:, :sf]
    current = first.x[:, sf:sf + df].clone()
    edge_index = first.edge_index
    pos = first.pos if hasattr(first, 'pos') and first.pos is not None else first.x[:, :2]
    ef = model.compute_edge_features(pos, edge_index)
    
    preds = []
    for step in range(rollout_steps):
        nf = torch.cat([static, current], dim=-1)
        delta = model(nf, ef, edge_index)
        current = (current + delta).detach()
        preds.append(current.cpu().numpy())
    return preds


ROLLOUT_FNS = {
    'gparcv2': rollout_gparcv2,
    'gparcv1': rollout_gparcv1,
    'mgkan': rollout_mgkan,
    'mgnet': rollout_mgnet,
}


# ==============================================================================
# VISUALIZATION
# ==============================================================================

MODEL_COLORS = {
    'gparcv2': '#1f77b4',
    'gparcv1': '#ff7f0e',
    'mgkan': '#2ca02c',
    'mgnet': '#d62728',
}

MODEL_LABELS = {
    'gparcv2': 'G-PARCv2',
    'gparcv1': 'G-PARCv1',
    'mgkan': 'MeshGraphKAN',
    'mgnet': 'MeshGraphNet',
}


def plot_timeseries_comparison(all_preds, all_targs, sim_name, output_dir,
                                var_idx=0, var_name='Depth', var_unit='m',
                                important_mask=None):
    """
    Plot spatially-averaged timeseries for one variable.
    GT (black) vs each model (colored) with shading for ±1 std.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # --- Top panel: all nodes ---
    ax = axes[0]
    
    # GT
    gt_list = list(all_targs.values())[0]  # all models have same GT
    gt_mean = [t[:, var_idx].mean() for t in gt_list]
    gt_std = [t[:, var_idx].std() for t in gt_list]
    T = len(gt_mean)
    t_axis = np.arange(T)
    
    ax.plot(t_axis, gt_mean, 'k-', linewidth=2.5, label='Ground Truth', zorder=10)
    ax.fill_between(t_axis,
                     np.array(gt_mean) - np.array(gt_std),
                     np.array(gt_mean) + np.array(gt_std),
                     color='gray', alpha=0.15, label='GT ±1σ')
    
    for model_name, preds in all_preds.items():
        color = MODEL_COLORS.get(model_name, 'purple')
        label = MODEL_LABELS.get(model_name, model_name)
        means = [p[:, var_idx].mean() for p in preds]
        stds = [p[:, var_idx].std() for p in preds]
        ax.plot(t_axis, means, color=color, linewidth=1.5, label=label, alpha=0.9)
        ax.fill_between(t_axis,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         color=color, alpha=0.1)
    
    ax.set_ylabel(f'{var_name} ({var_unit})', fontsize=11)
    ax.set_title(f'{sim_name} — {var_name} (All Nodes)', fontsize=13)
    ax.legend(fontsize=9, ncol=3, loc='upper right')
    ax.grid(alpha=0.3)
    
    # --- Bottom panel: important nodes only ---
    ax = axes[1]
    if important_mask is not None and important_mask.any():
        gt_mean_imp = [t[important_mask, var_idx].mean() for t in gt_list]
        gt_std_imp = [t[important_mask, var_idx].std() for t in gt_list]
        ax.plot(t_axis, gt_mean_imp, 'k-', linewidth=2.5, label='GT', zorder=10)
        ax.fill_between(t_axis,
                         np.array(gt_mean_imp) - np.array(gt_std_imp),
                         np.array(gt_mean_imp) + np.array(gt_std_imp),
                         color='gray', alpha=0.15)
        
        for model_name, preds in all_preds.items():
            color = MODEL_COLORS.get(model_name, 'purple')
            label = MODEL_LABELS.get(model_name, model_name)
            means = [p[important_mask, var_idx].mean() for p in preds]
            stds = [p[important_mask, var_idx].std() for p in preds]
            ax.plot(t_axis, means, color=color, linewidth=1.5, label=label, alpha=0.9)
            ax.fill_between(t_axis,
                             np.array(means) - np.array(stds),
                             np.array(means) + np.array(stds),
                             color=color, alpha=0.1)
        
        ax.set_title(f'{var_name} (Important Nodes)', fontsize=13)
    else:
        ax.text(0.5, 0.5, 'No important nodes detected', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel(f'{var_name} ({var_unit})', fontsize=11)
    ax.legend(fontsize=9, ncol=3, loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / f'timeseries_{var_name.lower()}_{sim_name}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_table(all_metrics, output_dir):
    """Bar chart comparing key metrics across models."""
    model_names = list(all_metrics.keys())
    n = len(model_names)
    
    metrics_to_plot = [
        ('depth_nse', 'Depth NSE', True),     # higher is better
        ('depth_rmse', 'Depth RMSE', False),   # lower is better
        ('rrmse', 'RRMSE', False),             # lower is better
        ('depth_csi', 'Depth CSI', True),      # higher is better
    ]
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4.5 * len(metrics_to_plot), 5))
    
    for ax, (key, title, higher_better) in zip(axes, metrics_to_plot):
        vals = [all_metrics[m].get(key, 0) for m in model_names]
        labels = [MODEL_LABELS.get(m, m) for m in model_names]
        colors = [MODEL_COLORS.get(m, 'gray') for m in model_names]
        
        bars = ax.bar(range(n), vals, color=colors, edgecolor='k', alpha=0.85)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3, axis='y')
        
        # Annotate bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Model Comparison — Aggregate Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_metrics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_nse_per_sim(all_results, output_dir):
    """Per-simulation NSE comparison across models."""
    model_names = list(all_results.keys())
    # Use union of all sim names, sorted
    all_sims = set()
    for m in model_names:
        all_sims.update(all_results[m].keys())
    sim_names = sorted(all_sims)
    n_sims = len(sim_names)
    n_models = len(model_names)
    
    fig, ax = plt.subplots(figsize=(max(12, n_sims * 0.8), 6))
    width = 0.8 / n_models
    
    for i, model_name in enumerate(model_names):
        nses = [all_results[model_name].get(s, {}).get('depth_nse', np.nan)
                for s in sim_names]
        x = np.arange(n_sims) + i * width - 0.4 + width / 2
        ax.bar(x, nses, width=width, label=MODEL_LABELS.get(model_name, model_name),
               color=MODEL_COLORS.get(model_name, 'gray'), edgecolor='k', alpha=0.85)
    
    ax.set_xticks(range(n_sims))
    ax.set_xticklabels([s.replace('simulation_', 's') for s in sim_names],
                        fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Depth NSE', fontsize=11)
    ax.set_title('Per-Simulation Depth NSE', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    ax.axhline(0, color='red', ls='--', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'nse_per_simulation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_test_simulations(test_dir, max_sims=None):
    test_dir = Path(test_dir)
    files = sorted(test_dir.glob("*.pt"))
    if max_sims:
        files = files[:max_sims]
    
    sims = []
    for f in tqdm(files, desc="Loading test data"):
        try:
            sim = torch.load(f, weights_only=False)
            if isinstance(sim, list) and len(sim) > 0:
                sims.append((f.stem, sim))
        except Exception as e:
            print(f"  Error: {f}: {e}")
    print(f"Loaded {len(sims)} test simulations")
    return sims


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified River Model Comparison")
    
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--extrema_path", required=True,
                        help="Path to extrema.pt for denormalization")
    parser.add_argument("--models", nargs='+', required=True,
                        help="model_name:checkpoint_path pairs, e.g. gparcv2:/path/best.pth")
    parser.add_argument("--output_dir", default="./comparison_results")
    parser.add_argument("--rollout_steps", type=int, default=None,
                        help="Max rollout steps (default: full sim length)")
    parser.add_argument("--max_sims", type=int, default=None)
    parser.add_argument("--num_viz", type=int, default=3,
                        help="Number of sims to visualize")
    parser.add_argument("--depth_threshold", type=float, default=0.3)
    parser.add_argument("--segments", type=str, default=None,
                        help="Comma-separated segment ranges, e.g. '0:22,22:64,64:79'")
    parser.add_argument("--num_static_feats", type=int, default=9)
    parser.add_argument("--num_dynamic_feats", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    sf, df = args.num_static_feats, args.num_dynamic_feats
    
    # Parse segments
    segments = args.segments.split(',') if args.segments else None
    
    # Load extrema
    extrema = load_denorm_extrema(args.extrema_path)
    
    # Parse model specs
    model_specs = {}
    for spec in args.models:
        name, path = spec.split(':', 1)
        model_specs[name] = path
    
    print(f"\n{'=' * 70}")
    print("UNIFIED RIVER MODEL COMPARISON")
    print(f"{'=' * 70}")
    print(f"Models: {list(model_specs.keys())}")
    print(f"Device: {device}")
    
    # Load models
    models = {}
    for name, path in model_specs.items():
        print(f"\nLoading {name}: {path}")
        if name not in MODEL_LOADERS:
            print(f"  ⚠️  Unknown model type '{name}', skipping")
            continue
        try:
            models[name] = MODEL_LOADERS[name](path, device, sf, df)
        except Exception as e:
            print(f"  ❌ Failed to load {name}: {e}")
            import traceback; traceback.print_exc()
    
    if not models:
        print("No models loaded. Exiting.")
        return
    
    # Load test data
    test_sims = load_test_simulations(args.test_dir, args.max_sims)
    
    # ---- Evaluate all models on all sims ----
    # Structure: per_sim_results[model_name][sim_name] = metrics dict
    # per_sim_preds[model_name][sim_name] = list of [N, D] arrays
    per_sim_results = {m: {} for m in models}
    per_sim_preds = {m: {} for m in models}
    per_sim_targs = {}
    
    for sim_idx, (sim_name, sim) in enumerate(test_sims):
        print(f"\n{'─' * 50}")
        print(f"Simulation: {sim_name} ({len(sim)} timesteps)")
        
        # Move to device
        sim_gpu = []
        for d in sim:
            d_gpu = d.clone()
            d_gpu.x = d_gpu.x.to(device)
            d_gpu.edge_index = d_gpu.edge_index.to(device)
            if hasattr(d_gpu, 'pos') and d_gpu.pos is not None:
                d_gpu.pos = d_gpu.pos.to(device)
            sim_gpu.append(d_gpu)
        
        steps = min(args.rollout_steps or len(sim) - 1, len(sim) - 1)
        
        # GT targets (same for all models)
        targs_norm = [sim_gpu[t].y[:, :df].cpu().numpy()
                      if hasattr(sim_gpu[t], 'y') and sim_gpu[t].y is not None
                      else sim_gpu[t + 1].x[:, sf:sf + df].cpu().numpy()
                      for t in range(steps)]
        targs_phys = [denormalize_all(t, extrema) for t in targs_norm]
        per_sim_targs[sim_name] = targs_phys
        
        # Important mask
        imp_mask = get_important_mask(sim_gpu, sf, 0, args.depth_threshold, extrema)
        
        for model_name, model in models.items():
            try:
                with torch.no_grad():
                    rollout_fn = ROLLOUT_FNS[model_name]
                    preds_raw = rollout_fn(model, sim_gpu, steps, device)
                
                preds_phys = [denormalize_all(p, extrema) for p in preds_raw]
                per_sim_preds[model_name][sim_name] = preds_phys
                
                # Metrics
                metrics = compute_all_metrics(preds_phys, targs_phys, args.depth_threshold)
                
                # Segmented metrics
                if segments:
                    metrics['segments'] = compute_segmented_metrics(
                        preds_phys, targs_phys, segments, imp_mask, args.depth_threshold)
                
                per_sim_results[model_name][sim_name] = metrics
                
                print(f"  {MODEL_LABELS.get(model_name, model_name):15s} — "
                      f"NSE={metrics['depth_nse']:.4f}  "
                      f"RMSE={metrics['depth_rmse']:.4f}  "
                      f"RRMSE={metrics['rrmse']:.4f}  "
                      f"CSI={metrics['depth_csi']:.3f}  "
                      f"MB%={metrics['mass_balance_pct']:.2f}")
            
            except Exception as e:
                print(f"  ❌ {model_name} failed: {e}")
                import traceback; traceback.print_exc()
        
        # ---- Visualizations for first N sims ----
        if sim_idx < args.num_viz:
            sim_preds = {m: per_sim_preds[m].get(sim_name) for m in models
                         if sim_name in per_sim_preds[m]}
            sim_targs = {m: targs_phys for m in models}
            
            for vi, (vn, vu) in enumerate(zip(VAR_NAMES, VAR_UNITS)):
                if vi < df:
                    plot_timeseries_comparison(
                        sim_preds, sim_targs, sim_name, output_dir,
                        var_idx=vi, var_name=vn, var_unit=vu,
                        important_mask=imp_mask)
    
    # ---- Aggregate metrics ----
    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 70}")
    
    agg_metrics = {}
    for model_name in models:
        sim_metrics = list(per_sim_results[model_name].values())
        if not sim_metrics:
            continue
        
        agg = {}
        for key in ['rmse', 'rrmse', 'depth_rmse', 'depth_nse', 'depth_csi', 'mass_balance_pct']:
            vals = [m[key] for m in sim_metrics if not np.isnan(m.get(key, np.nan))]
            agg[key] = float(np.mean(vals)) if vals else np.nan
        
        agg_metrics[model_name] = agg
    
    # Print comparison table
    print(f"\n{'Model':<18s} {'NSE':>8s} {'RMSE':>10s} {'RRMSE':>8s} {'CSI':>8s} {'MB%':>8s}")
    print("─" * 64)
    for model_name, agg in agg_metrics.items():
        label = MODEL_LABELS.get(model_name, model_name)
        print(f"{label:<18s} {agg.get('depth_nse', 0):>8.4f} "
              f"{agg.get('depth_rmse', 0):>10.4f} "
              f"{agg.get('rrmse', 0):>8.4f} "
              f"{agg.get('depth_csi', 0):>8.3f} "
              f"{agg.get('mass_balance_pct', np.nan):>8.2f}")
    
    # Plots
    if len(agg_metrics) > 1:
        plot_comparison_table(agg_metrics, output_dir)
        plot_nse_per_sim(per_sim_results, output_dir)
    
    # ---- Segment table ----
    if segments:
        seg_keys = segments
        
        for metric_name in ['NSE', 'RMSE', 'CSI']:
            print(f"\n{'=' * 70}")
            print(f"PER-SEGMENT DEPTH {metric_name} (All Nodes)")
            print(f"{'=' * 70}")
            header = f"{'Model':<18s}" + "".join(f"{'t' + s:>12s}" for s in seg_keys) + f"{'Overall':>12s}"
            print(header)
            print("─" * len(header))
            
            for model_name in models:
                label = MODEL_LABELS.get(model_name, model_name)
                row = f"{label:<18s}"
                sim_results = list(per_sim_results[model_name].values())
                
                for seg_str in seg_keys:
                    parts = seg_str.split(':')
                    key = f"t{parts[0]}-{parts[1]}"
                    vals = []
                    for sr in sim_results:
                        if 'segments' in sr and 'All_Nodes' in sr['segments']:
                            seg = sr['segments']['All_Nodes'].get(key, {})
                            if metric_name in seg:
                                vals.append(seg[metric_name])
                    mean_val = np.mean(vals) if vals else np.nan
                    if np.isnan(mean_val):
                        row += f"{'—':>12s}"
                    else:
                        row += f"{mean_val:>12.4f}"
                
                # Overall
                overall_key = {'NSE': 'depth_nse', 'RMSE': 'depth_rmse', 'CSI': 'depth_csi'}
                overall_vals = [m.get(overall_key[metric_name], np.nan)
                               for m in sim_results
                               if not np.isnan(m.get(overall_key[metric_name], np.nan))]
                overall_mean = np.mean(overall_vals) if overall_vals else np.nan
                row += f"{overall_mean:>12.4f}"
                print(row)
        
        # Important nodes only (if available)
        has_important = any(
            'segments' in sr and 'Important' in sr.get('segments', {})
            for model_name in models
            for sr in per_sim_results[model_name].values()
        )
        if has_important:
            for metric_name in ['NSE', 'RMSE', 'CSI']:
                print(f"\n{'=' * 70}")
                print(f"PER-SEGMENT DEPTH {metric_name} (Important Nodes Only)")
                print(f"{'=' * 70}")
                header = f"{'Model':<18s}" + "".join(f"{'t' + s:>12s}" for s in seg_keys)
                print(header)
                print("─" * len(header))
                
                for model_name in models:
                    label = MODEL_LABELS.get(model_name, model_name)
                    row = f"{label:<18s}"
                    sim_results = list(per_sim_results[model_name].values())
                    
                    for seg_str in seg_keys:
                        parts = seg_str.split(':')
                        key = f"t{parts[0]}-{parts[1]}"
                        vals = []
                        for sr in sim_results:
                            if 'segments' in sr and 'Important' in sr['segments']:
                                seg = sr['segments']['Important'].get(key, {})
                                if metric_name in seg:
                                    vals.append(seg[metric_name])
                        mean_val = np.mean(vals) if vals else np.nan
                        if np.isnan(mean_val):
                            row += f"{'—':>12s}"
                        else:
                            row += f"{mean_val:>12.4f}"
                    print(row)
    
    # Save JSON
    summary = {
        'aggregate': {k: v for k, v in agg_metrics.items()},
        'per_simulation': {m: {s: v for s, v in per_sim_results[m].items()} for m in models},
        'models': {m: str(model_specs[m]) for m in models},
    }
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else None)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()