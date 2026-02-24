#!/usr/bin/env python3
"""
Unified Shock Tube Model Comparison Evaluation
================================================
Evaluates multiple models (G-PARCv2, MeshGraphKAN, MeshGraphNet) on the same
test simulations with identical metrics.

Variables: density, x_momentum, total_energy (y_momentum skipped at index 2)
Grid: 64x64 structured (4096 nodes)

Outputs:
  - Comparison table: RMSE, RRMSE, R² per model per variable
  - Per-variable timeseries: spatially-averaged GT vs all models
  - 2D field snapshots at key timesteps
  - Summary JSON
  - Per-simulation RRMSE bar chart

Usage:
    python eval_comparison.py \
        --test_dir /path/to/test \
        --models gparcv2:/path/v2.pth mgkan:/path/kan.pth mgnet:/path/net.pth \
        --output_dir ./comparison \
        --rollout_steps 40
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ===========================================================================
# CONSTANTS
# ===========================================================================

VAR_NAMES = ['density', 'x_momentum', 'total_energy']
NUM_STATIC = 2
NUM_USED_DYNAMIC = 3
SKIP_INDICES = [2]  # skip y_momentum
RAW_DYNAMIC = NUM_USED_DYNAMIC + len(SKIP_INDICES)  # 4

KEEP_INDICES = [i for i in range(RAW_DYNAMIC) if i not in SKIP_INDICES]

MODEL_COLORS = {
    'gparcv1': '#d62728',
    'gparcv2': '#1f77b4',
    'mgkan': '#ff7f0e',
    'mgnet': '#2ca02c',
}
MODEL_LABELS = {
    'gparcv1': 'G-PARCv1',
    'gparcv2': 'G-PARCv2',
    'mgkan': 'MeshGraphKAN',
    'mgnet': 'MeshGraphNet',
}


# ===========================================================================
# UTILITIES
# ===========================================================================

def extract_dynamic(x, sf=NUM_STATIC):
    """Extract used dynamic features from x (skip y_momentum)."""
    raw = x[:, sf:sf + RAW_DYNAMIC]
    return raw[:, KEEP_INDICES]


def extract_global_params(data):
    """Extract [pressure, density_param, delta_t] as [N, 3] broadcast."""
    parts = []
    for attr in ['global_pressure', 'pressure']:
        if hasattr(data, attr):
            parts.append(getattr(data, attr)); break
    else:
        parts.append(torch.zeros(1, device=data.x.device))

    for attr in ['global_density', 'density_param']:
        if hasattr(data, attr):
            parts.append(getattr(data, attr)); break
    else:
        parts.append(torch.zeros(1, device=data.x.device))

    for attr in ['global_delta_t', 'delta_t']:
        if hasattr(data, attr):
            parts.append(getattr(data, attr)); break
    else:
        parts.append(torch.zeros(1, device=data.x.device))

    # Ensure global_params fallback
    if len(parts) < 3 or all(p.item() == 0 for p in parts):
        if hasattr(data, 'global_params') and data.global_params.numel() >= 3:
            gp = data.global_params
            parts = [gp[0].unsqueeze(0), gp[1].unsqueeze(0), gp[2].unsqueeze(0)]

    gp = torch.cat([p.view(1) for p in parts])  # [3]
    return gp.unsqueeze(0).expand(data.x.size(0), -1)  # [N, 3]


def apply_skip(y):
    """Filter target y to keep only used dynamic indices."""
    return y[:, KEEP_INDICES]


def parse_params_from_filename(name):
    """
    Extract pressure and density from filename like:
    p_L_162500_rho_L_0.5_test_with_pos_normalized
    """
    import re
    pressure = None
    density = None
    
    p_match = re.search(r'p_L_([0-9.]+)', name)
    if p_match:
        pressure = float(p_match.group(1))
    
    rho_match = re.search(r'rho_L_([0-9.]+)', name)
    if rho_match:
        density = float(rho_match.group(1))
    
    return pressure, density


def extract_global_params_from_data(data):
    """Extract [pressure, density, delta_t] as raw scalar values from a Data object."""
    vals = {}
    
    def to_float(v):
        """Safely convert tensor or scalar to float."""
        if isinstance(v, torch.Tensor):
            return v.item()
        return float(v)
    
    # Try global_params tensor first (most reliable — always has all 3)
    if hasattr(data, 'global_params'):
        gp = data.global_params
        if torch.is_tensor(gp) and gp.numel() >= 3:
            vals = {
                'pressure': gp[0].item(),
                'density': gp[1].item(),
                'delta_t': gp[2].item(),
            }
            return vals
    
    # Fallback: individual attributes
    for attr in ['global_pressure', 'pressure']:
        if hasattr(data, attr):
            try:
                vals['pressure'] = to_float(getattr(data, attr))
            except (ValueError, TypeError):
                pass
            break
    
    for attr in ['global_density', 'density_param']:
        if hasattr(data, attr):
            try:
                vals['density'] = to_float(getattr(data, attr))
            except (ValueError, TypeError):
                pass
            break
    
    for attr in ['global_delta_t', 'delta_t']:
        if hasattr(data, attr):
            try:
                vals['delta_t'] = to_float(getattr(data, attr))
            except (ValueError, TypeError):
                pass
            break
    
    return vals


# ===========================================================================
# METRICS
# ===========================================================================

def compute_metrics(pred_list, gt_list):
    """Compute per-variable and aggregate metrics from lists of [N, 3] arrays."""
    all_pred = np.concatenate(pred_list, axis=0)  # [T*N, 3]
    all_gt = np.concatenate(gt_list, axis=0)

    metrics = {}

    # Per-variable
    for vi, vn in enumerate(VAR_NAMES):
        p = all_pred[:, vi]
        g = all_gt[:, vi]
        rmse = np.sqrt(np.mean((p - g) ** 2))
        rms_gt = np.sqrt(np.mean(g ** 2))
        rrmse = rmse / max(rms_gt, 1e-12)

        ss_res = np.sum((g - p) ** 2)
        ss_tot = np.sum((g - g.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

        metrics[f'{vn}_rmse'] = float(rmse)
        metrics[f'{vn}_rrmse'] = float(rrmse)
        metrics[f'{vn}_r2'] = float(r2)

    # Aggregate
    diff = all_pred - all_gt
    rmse_all = np.sqrt(np.mean(diff ** 2))
    rms_gt_all = np.sqrt(np.mean(all_gt ** 2))
    metrics['rmse'] = float(rmse_all)
    metrics['rrmse'] = float(rmse_all / max(rms_gt_all, 1e-12))

    return metrics


def compute_per_step_rrmse(pred_list, gt_list):
    """RRMSE per timestep for error accumulation plots."""
    rrmse_steps = []
    for pred, gt in zip(pred_list, gt_list):
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        rms_gt = np.sqrt(np.mean(gt ** 2))
        rrmse_steps.append(rmse / max(rms_gt, 1e-12))
    return rrmse_steps


# ===========================================================================
# MODEL LOADERS
# ===========================================================================

def load_model_gparcv1(ckpt_path, sample_data, device):
    """Load G-PARCv1 shocktube model (FeatureExtractorGNN + DerivativeGNN + IntegralGNN)."""
    from models.shocktube import GPARC
    from utilities.featureextractor import FeatureExtractorGNN
    from differentiator.differentiator import DerivativeGNN
    from integrator.integrator import IntegralGNN

    # Architecture params must match viz_mod.sh / test_mod.py
    sf = NUM_STATIC
    df = NUM_USED_DYNAMIC
    hidden_channels = 32
    feature_out_channels = 32
    depth = 2
    pool_ratios = 0.2
    heads = 2
    dropout = 0.2
    global_embed_dim = 64

    # Derivative solver
    deriv_hidden = 128
    deriv_layers = 3
    deriv_heads = 4
    deriv_dropout = 0.2

    # Integral solver
    integral_hidden = 128
    integral_layers = 3
    integral_heads = 4
    integral_dropout = 0.2

    feature_extractor = FeatureExtractorGNN(
        in_channels=sf,
        hidden_channels=hidden_channels,
        out_channels=feature_out_channels,
        depth=depth,
        pool_ratios=pool_ratios,
        heads=heads,
        concat=True,
        dropout=dropout,
    )

    deriv_in_channels = feature_out_channels + df + global_embed_dim
    derivative_solver = DerivativeGNN(
        in_channels=deriv_in_channels,
        hidden_channels=deriv_hidden,
        out_channels=df,
        num_layers=deriv_layers,
        heads=deriv_heads,
        concat=True,
        dropout=deriv_dropout,
        use_residual=True,
    )

    integral_solver = IntegralGNN(
        in_channels=df,
        hidden_channels=integral_hidden,
        out_channels=df,
        num_layers=integral_layers,
        heads=integral_heads,
        concat=True,
        dropout=integral_dropout,
        use_residual=True,
    )

    model = GPARC(
        feature_extractor=feature_extractor,
        derivative_solver=derivative_solver,
        integral_solver=integral_solver,
        num_static_feats=sf,
        num_dynamic_feats=df,
        skip_dynamic_indices=SKIP_INDICES,
        feature_out_channels=feature_out_channels,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def load_model_gparcv2(ckpt_path, sample_data, device):
    """Load G-PARCv2 shocktube model (ShockTubeDifferentiator + FiLM)."""
    from models.shocktube_gparcv2 import GPARC_ShockTube_V2
    from differentiator.nospade import ShockTubeDifferentiator
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d

    ckpt_dir = Path(ckpt_path).parent
    config_path = ckpt_dir / "config.json"
    config = json.load(open(config_path)) if config_path.exists() else {}

    # Architecture params matching train_gparcv2.sh
    sf = config.get('num_static_feats', NUM_STATIC)
    df = config.get('num_dynamic_feats', NUM_USED_DYNAMIC)
    feat_out = config.get('feature_out_channels', 128)
    hidden = config.get('hidden_channels', 64)
    n_layers = config.get('num_layers', 4)
    global_embed_dim = config.get('global_embed_dim', 64)
    global_param_dim = config.get('global_param_dim', 3)

    norm_stats_path = ckpt_dir / "normalization_stats.json"
    norm_stats = json.load(open(norm_stats_path)) if norm_stats_path.exists() else {}
    pos_stats = norm_stats.get('position', {})
    pos_mean = pos_stats.get('mean')
    pos_std = pos_stats.get('std')

    gradient_solver = SolveGradientsLST(pos_mean=pos_mean, pos_std=pos_std)
    laplacian_solver = SolveWeightLST2d(pos_mean=pos_mean, pos_std=pos_std, use_2hop_extension=False)

    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=sf+df, hidden_channels=hidden, out_channels=feat_out,
        num_layers=n_layers, dropout=config.get('dropout', 0.2),
        use_layer_norm=config.get('use_layer_norm', True),
        use_relative_pos=config.get('use_relative_pos', True),
    )

    derivative_solver = ShockTubeDifferentiator(
        num_static_feats=sf,
        num_dynamic_feats=df,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=feat_out,
        global_embed_dim=global_embed_dim,
        global_param_dim=global_param_dim,
        list_adv_idx=list(range(df)),
        list_dif_idx=list(range(df)),
        velocity_indices=[config.get('velocity_index', 1)],
        diffusion_type=config.get('diffusion_type', 'fd'),
        spade_random_noise=config.get('spade_random_noise', False),
        heads=config.get('spade_heads', 4),
        concat=config.get('spade_concat', True),
        dropout=config.get('spade_dropout', 0.1),
        zero_init=config.get('zero_init', False),
    )

    if not hasattr(sample_data, 'pos') or sample_data.pos is None:
        sample_data.pos = sample_data.x[:, :sf]
    derivative_solver.initialize_weights(sample_data)

    model = GPARC_ShockTube_V2(
        derivative_solver_physics=derivative_solver,
        integrator_type=config.get('integrator', 'euler'),
        num_static_feats=sf,
        num_dynamic_feats=df,
        skip_dynamic_indices=config.get('skip_dynamic_indices', SKIP_INDICES),
        global_param_dim=global_param_dim,
        global_embed_dim=global_embed_dim,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def load_model_mgkan(ckpt_path, sample_data, device):
    """Load MeshGraphKAN shocktube model."""
    # Import from training script (classes defined there)
    sys.path.insert(0, str(Path(__file__).parent.parent / 'MeshGraphKan'))
    from train_shocktube import MeshGraphKAN, MeshGraphKANShocktubeRollout

    ckpt_dir = Path(ckpt_path).parent
    config = json.load(open(ckpt_dir / "config.json")) if (ckpt_dir / "config.json").exists() else {}

    hidden = config.get('hidden_dim_processor', 128)
    proc_size = config.get('processor_size', 15)
    harmonics = config.get('num_harmonics', 5)

    # Input: static(2) + dynamic(3) + global(3) = 8
    base_model = MeshGraphKAN(
        input_dim_nodes=8, input_dim_edges=3, output_dim=NUM_USED_DYNAMIC,
        processor_size=proc_size, hidden_dim_processor=hidden,
        num_harmonics=harmonics,
    )

    wrapper = MeshGraphKANShocktubeRollout(
        model=base_model, num_static_feats=NUM_STATIC,
        num_dynamic_feats=NUM_USED_DYNAMIC,
        skip_dynamic_indices=SKIP_INDICES,
        global_param_dim=3,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    wrapper.load_state_dict(sd)
    wrapper.to(device).eval()
    return wrapper


def load_model_mgnet(ckpt_path, sample_data, device):
    """Load MeshGraphNet shocktube model."""
    sys.path.insert(0, str(Path(__file__).parent.parent / 'MeshGraphNet'))
    from meshgraphnet import MeshGraphNet

    ckpt_dir = Path(ckpt_path).parent
    config = json.load(open(ckpt_dir / "config.json")) if (ckpt_dir / "config.json").exists() else {}

    hidden = config.get('hidden_dim', 128)
    n_layers = config.get('num_layers', 10)

    # Input: static(2) + dynamic(3) + global(3) = 8
    model = MeshGraphNet(
        input_dim_node=8, input_dim_edge=3,
        hidden_dim=hidden, output_dim=NUM_USED_DYNAMIC,
        num_layers=n_layers,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


LOADERS = {
    'gparcv1': load_model_gparcv1,
    'gparcv2': load_model_gparcv2,
    'mgkan': load_model_mgkan,
    'mgnet': load_model_mgnet,
}


# ===========================================================================
# ROLLOUT FUNCTIONS
# ===========================================================================

@torch.no_grad()
def rollout_gparcv1(model, sim, num_steps, device):
    """G-PARCv1 rollout: feature_extractor → DerivativeGNN → IntegralGNN → accumulate."""
    from utilities.embed import GlobalParameterProcessor
    sim = [d.to(device) for d in sim]
    for d in sim:
        if not hasattr(d, 'pos') or d.pos is None:
            d.pos = d.x[:, :NUM_STATIC]
        if hasattr(d, 'global_params') and d.global_params.numel() >= 3:
            gp = d.global_params
            if not hasattr(d, 'global_pressure'):
                d.global_pressure = gp[0].unsqueeze(0)
                d.global_density = gp[1].unsqueeze(0)
                d.global_delta_t = gp[2].unsqueeze(0)

    first = sim[0]
    global_attrs = torch.stack([
        first.global_pressure.flatten()[0],
        first.global_density.flatten()[0],
        first.global_delta_t.flatten()[0],
    ])
    global_embed = model.global_processor(global_attrs)

    # Static features processed once
    static = first.x[:, :NUM_STATIC]
    edge_index = first.edge_index
    learned_static = model.feature_extractor(static, edge_index)
    learned_static = model.feature_norm(learned_static, global_attrs)

    # Initial dynamic
    F_prev = extract_dynamic(first.x)
    predictions = []

    for step in range(min(num_steps, len(sim))):
        data_t = sim[step]

        F_input = model.derivative_norm(F_prev, global_attrs)
        global_context = global_embed.unsqueeze(0).repeat(data_t.num_nodes, 1)
        Fdot_input = torch.cat([learned_static, F_input, global_context], dim=-1)

        Fdot = model.derivative_solver(Fdot_input, data_t.edge_index)
        Fint = model.integral_solver(Fdot, data_t.edge_index)
        F_pred = F_input + Fint

        predictions.append(F_pred.cpu().numpy())
        F_prev = F_pred

    return predictions


@torch.no_grad()
def rollout_gparcv2(model, sim, num_steps, device):
    """G-PARCv2 rollout using FiLM conditioning."""
    sim = [d.to(device) for d in sim]
    for d in sim:
        if not hasattr(d, 'pos') or d.pos is None:
            d.pos = d.x[:, :NUM_STATIC]
        if hasattr(d, 'global_params') and d.global_params.numel() >= 3:
            gp = d.global_params
            if not hasattr(d, 'global_pressure'):
                d.global_pressure = gp[0].unsqueeze(0)
                d.global_density = gp[1].unsqueeze(0)
                d.global_delta_t = gp[2].unsqueeze(0)

    # Init MLS
    deriv = model.derivative_solver if hasattr(model, 'derivative_solver') else \
            getattr(model, 'derivative_solver_physics', None)
    if deriv and hasattr(deriv, 'initialize_weights'):
        deriv.initialize_weights(sim[0])

    first = sim[0]
    global_attrs = model._extract_global_attrs(first)
    global_embed = model.global_processor(global_attrs)

    F_prev = model._extract_dynamic(first.x)
    predictions = []

    for step in range(min(num_steps, len(sim))):
        data_t = sim[step]
        static = data_t.x[:, :NUM_STATIC]

        F_pred = model.step(
            static_feats=static,
            dynamic_state=F_prev.clone(),
            edge_index=data_t.edge_index,
            global_embed=global_embed,
            global_attrs=global_attrs,
            dt=1.0,
        )
        predictions.append(F_pred.cpu().numpy())
        F_prev = F_pred

    return predictions


@torch.no_grad()
def rollout_mgkan(wrapper, sim, num_steps, device):
    """MeshGraphKAN delta-prediction rollout with global param concatenation."""
    sim = [d.to(device) for d in sim]
    for d in sim:
        if not hasattr(d, 'pos') or d.pos is None:
            d.pos = d.x[:, :NUM_STATIC]
        if hasattr(d, 'global_params') and d.global_params.numel() >= 3:
            gp = d.global_params
            if not hasattr(d, 'global_pressure'):
                d.global_pressure = gp[0].unsqueeze(0)
                d.global_density = gp[1].unsqueeze(0)
                d.global_delta_t = gp[2].unsqueeze(0)

    current_dynamic = wrapper._extract_dynamic(sim[0].x)
    predictions = []

    for step in range(min(num_steps, len(sim))):
        data_t = sim[step]
        static = data_t.x[:, :NUM_STATIC]
        global_feats = wrapper._extract_global_params(data_t)

        node_features = torch.cat([static, current_dynamic, global_feats], dim=-1)
        edge_features = wrapper.compute_edge_features(data_t)

        pred_delta = wrapper.model(node_features, edge_features, data_t.edge_index)
        current_dynamic = current_dynamic + pred_delta

        predictions.append(current_dynamic.cpu().numpy())

    return predictions


@torch.no_grad()
def rollout_mgnet(model, sim, num_steps, device):
    """MeshGraphNet delta-prediction rollout with global param concatenation."""
    sim = [d.to(device) for d in sim]
    for d in sim:
        if not hasattr(d, 'pos') or d.pos is None:
            d.pos = d.x[:, :NUM_STATIC]
        if hasattr(d, 'global_params') and d.global_params.numel() >= 3:
            gp = d.global_params
            if not hasattr(d, 'global_pressure'):
                d.global_pressure = gp[0].unsqueeze(0)
                d.global_density = gp[1].unsqueeze(0)
                d.global_delta_t = gp[2].unsqueeze(0)

    current_dynamic = extract_dynamic(sim[0].x)
    predictions = []

    for step in range(min(num_steps, len(sim))):
        data_t = sim[step]
        static = data_t.x[:, :NUM_STATIC]
        pos = data_t.pos if hasattr(data_t, 'pos') and data_t.pos is not None else static
        global_feats = extract_global_params(data_t)

        node_features = torch.cat([static, current_dynamic, global_feats], dim=-1)
        edge_features = model.compute_edge_features(pos, data_t.edge_index)

        pred_delta = model(node_features, edge_features, data_t.edge_index)
        current_dynamic = current_dynamic + pred_delta

        predictions.append(current_dynamic.cpu().numpy())

    return predictions


ROLLOUT_FNS = {
    'gparcv1': rollout_gparcv1,
    'gparcv2': rollout_gparcv2,
    'mgkan': rollout_mgkan,
    'mgnet': rollout_mgnet,
}


# ===========================================================================
# VISUALIZATION
# ===========================================================================

def plot_comparison_table(all_metrics, output_dir):
    """Bar chart comparing RRMSE per variable per model."""
    model_names = list(all_metrics.keys())
    n_models = len(model_names)
    n_vars = len(VAR_NAMES)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_vars)
    width = 0.8 / n_models

    for i, mname in enumerate(model_names):
        vals = [all_metrics[mname].get(f'{vn}_rrmse', 0) for vn in VAR_NAMES]
        bars = ax.bar(x + i * width, vals, width,
                      label=MODEL_LABELS.get(mname, mname),
                      color=MODEL_COLORS.get(mname, f'C{i}'),
                      alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('RRMSE')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(VAR_NAMES)
    ax.legend()
    ax.set_title('Shock Tube Model Comparison — RRMSE per Variable')
    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_rrmse.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_r2_comparison(all_metrics, output_dir):
    """Bar chart of R² per variable per model."""
    model_names = list(all_metrics.keys())
    n_models = len(model_names)
    n_vars = len(VAR_NAMES)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_vars)
    width = 0.8 / n_models

    for i, mname in enumerate(model_names):
        vals = [all_metrics[mname].get(f'{vn}_r2', 0) for vn in VAR_NAMES]
        ax.bar(x + i * width, vals, width,
               label=MODEL_LABELS.get(mname, mname),
               color=MODEL_COLORS.get(mname, f'C{i}'), alpha=0.85)

    ax.set_ylabel('R²')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(VAR_NAMES)
    ax.legend()
    ax.set_title('Shock Tube Model Comparison — R² per Variable')
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_r2.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_error_accumulation(all_per_step, output_dir):
    """RRMSE vs timestep for each model."""
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    for mname, step_rrmse_list in all_per_step.items():
        # Average across simulations
        max_steps = max(len(s) for s in step_rrmse_list)
        avg = np.zeros(max_steps)
        counts = np.zeros(max_steps)
        for s in step_rrmse_list:
            for t, v in enumerate(s):
                avg[t] += v
                counts[t] += 1
        avg = avg / np.maximum(counts, 1)

        axes.plot(avg, label=MODEL_LABELS.get(mname, mname),
                  color=MODEL_COLORS.get(mname), linewidth=2)

    axes.set_xlabel('Rollout Step')
    axes.set_ylabel('RRMSE')
    axes.set_title('Error Accumulation Over Rollout')
    axes.legend()
    axes.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / 'error_accumulation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_timeseries(all_timeseries, output_dir, sim_name):
    """Spatially-averaged timeseries: GT vs each model for one simulation."""
    n_vars = len(VAR_NAMES)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    gt = all_timeseries['gt']  # list of [N, 3]
    T = len(gt)
    timesteps = np.arange(T)

    for vi, vn in enumerate(VAR_NAMES):
        ax = axes[vi]

        # GT
        gt_mean = [g[:, vi].mean() for g in gt]
        gt_std = [g[:, vi].std() for g in gt]
        ax.plot(timesteps, gt_mean, 'k-', linewidth=2, label='GT')
        ax.fill_between(timesteps,
                        [m - s for m, s in zip(gt_mean, gt_std)],
                        [m + s for m, s in zip(gt_mean, gt_std)],
                        alpha=0.15, color='gray')

        # Models
        for mname, preds in all_timeseries.items():
            if mname == 'gt':
                continue
            pred_mean = [p[:, vi].mean() for p in preds]
            pred_std = [p[:, vi].std() for p in preds]
            color = MODEL_COLORS.get(mname, None)
            ax.plot(timesteps[:len(pred_mean)], pred_mean,
                    label=MODEL_LABELS.get(mname, mname),
                    color=color, linewidth=1.5)
            ax.fill_between(timesteps[:len(pred_std)],
                            [m - s for m, s in zip(pred_mean, pred_std)],
                            [m + s for m, s in zip(pred_mean, pred_std)],
                            alpha=0.1, color=color)

        ax.set_ylabel(vn)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Timestep')
    fig.suptitle(f'Spatially-Averaged Timeseries — {sim_name}', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / f'timeseries_{sim_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_sim_rrmse(per_sim_metrics, output_dir):
    """Grouped bar chart of per-simulation RRMSE for each model."""
    model_names = list(per_sim_metrics.keys())
    sim_names = list(per_sim_metrics[model_names[0]].keys())
    n_sims = len(sim_names)
    n_models = len(model_names)

    if n_sims > 30:
        # Too many sims — just show aggregate
        return

    fig, ax = plt.subplots(figsize=(max(12, n_sims * 0.8), 6))
    x = np.arange(n_sims)
    width = 0.8 / n_models

    for i, mname in enumerate(model_names):
        vals = [per_sim_metrics[mname][sn]['rrmse'] for sn in sim_names]
        ax.bar(x + i * width, vals, width,
               label=MODEL_LABELS.get(mname, mname),
               color=MODEL_COLORS.get(mname, f'C{i}'), alpha=0.85)

    ax.set_ylabel('RRMSE')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(sim_names, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.set_title('Per-Simulation RRMSE')
    plt.tight_layout()
    fig.savefig(output_dir / 'per_sim_rrmse.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_rrmse_vs_parameter(per_sim_metrics, sim_params, param_name, output_dir):
    """
    Scatter plot: RRMSE vs a global parameter for each model.
    Shows which parameter regimes each model struggles with.
    """
    model_names = list(per_sim_metrics.keys())
    
    # Collect data points
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mname in model_names:
        x_vals, y_vals = [], []
        for sn, m in per_sim_metrics[mname].items():
            if sn in sim_params and param_name in sim_params[sn]:
                x_vals.append(sim_params[sn][param_name])
                y_vals.append(m['rrmse'])
        
        if x_vals:
            ax.scatter(x_vals, y_vals, 
                      label=MODEL_LABELS.get(mname, mname),
                      color=MODEL_COLORS.get(mname), alpha=0.7, s=40)
            # Trend line
            if len(x_vals) > 2:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_sorted = np.sort(x_vals)
                ax.plot(x_sorted, p(x_sorted), '--',
                       color=MODEL_COLORS.get(mname), alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel(param_name.replace('_', ' ').title())
    ax.set_ylabel('RRMSE')
    ax.set_title(f'Model Performance vs {param_name.replace("_", " ").title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f'rrmse_vs_{param_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_parameter_heatmap(per_sim_metrics, sim_params, output_dir):
    """
    2D heatmap: RRMSE as function of pressure and density for each model.
    One subplot per model.
    """
    model_names = list(per_sim_metrics.keys())
    
    # Check we have both pressure and density
    has_both = any(
        'pressure' in sim_params.get(sn, {}) and 'density' in sim_params.get(sn, {})
        for sn in sim_params
    )
    if not has_both:
        return
    
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    # Get global color range
    all_rrmse = []
    for mname in model_names:
        for sn, m in per_sim_metrics[mname].items():
            all_rrmse.append(m['rrmse'])
    vmin = min(all_rrmse) if all_rrmse else 0
    vmax = np.percentile(all_rrmse, 95) if all_rrmse else 1
    
    for idx, mname in enumerate(model_names):
        ax = axes[idx]
        pressures, densities, rrmses = [], [], []
        
        for sn, m in per_sim_metrics[mname].items():
            if sn in sim_params:
                sp = sim_params[sn]
                if 'pressure' in sp and 'density' in sp:
                    pressures.append(sp['pressure'])
                    densities.append(sp['density'])
                    rrmses.append(m['rrmse'])
        
        if pressures:
            sc = ax.scatter(pressures, densities, c=rrmses,
                          cmap='RdYlGn_r', vmin=vmin, vmax=vmax,
                          s=80, edgecolors='gray', linewidths=0.5)
            plt.colorbar(sc, ax=ax, label='RRMSE')
        
        ax.set_xlabel('Pressure')
        ax.set_ylabel('Density')
        ax.set_title(MODEL_LABELS.get(mname, mname))
    
    fig.suptitle('RRMSE in Parameter Space (Pressure vs Density)', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / 'parameter_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_variable_vs_parameter(per_sim_metrics, sim_params, param_name, output_dir):
    """
    Per-variable RRMSE vs parameter: one subplot per variable, all models overlaid.
    """
    model_names = list(per_sim_metrics.keys())
    n_vars = len(VAR_NAMES)
    
    fig, axes = plt.subplots(1, n_vars, figsize=(6 * n_vars, 5))
    if n_vars == 1:
        axes = [axes]
    
    for vi, vn in enumerate(VAR_NAMES):
        ax = axes[vi]
        
        for mname in model_names:
            x_vals, y_vals = [], []
            for sn, m in per_sim_metrics[mname].items():
                if sn in sim_params and param_name in sim_params[sn]:
                    x_vals.append(sim_params[sn][param_name])
                    y_vals.append(m.get(f'{vn}_rrmse', 0))
            
            if x_vals:
                ax.scatter(x_vals, y_vals,
                          label=MODEL_LABELS.get(mname, mname),
                          color=MODEL_COLORS.get(mname), alpha=0.6, s=30)
        
        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel('RRMSE')
        ax.set_title(vn)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Per-Variable RRMSE vs {param_name.replace("_", " ").title()}', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / f'per_var_rrmse_vs_{param_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Shock Tube Model Comparison")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--models", type=str, nargs='+', required=True,
                        help="model_type:checkpoint_path pairs")
    parser.add_argument("--output_dir", type=str, default="./shocktube_comparison")
    parser.add_argument("--rollout_steps", type=int, default=40)
    parser.add_argument("--max_sims", type=int, default=None)
    parser.add_argument("--num_viz", type=int, default=5,
                        help="Number of sims to generate timeseries plots for")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Parse model specs — handle paths with spaces by reassembling
    # Shell splits "gparcv1:/path/with spaces/file.pth" into multiple args
    # We need to rejoin them: find tokens matching "type:" pattern, everything
    # until the next "type:" token is part of the path
    raw_specs = args.models
    model_specs = {}
    
    i = 0
    while i < len(raw_specs):
        token = raw_specs[i]
        # Check if this token starts with a known model type prefix
        colon_idx = token.find(':')
        if colon_idx > 0 and token[:colon_idx] in LOADERS:
            mtype = token[:colon_idx]
            path_parts = [token[colon_idx + 1:]]
            # Collect subsequent tokens until we hit another model spec or end
            j = i + 1
            while j < len(raw_specs):
                next_colon = raw_specs[j].find(':')
                if next_colon > 0 and raw_specs[j][:next_colon] in LOADERS:
                    break
                path_parts.append(raw_specs[j])
                j += 1
            mpath = ' '.join(path_parts)
            if Path(mpath).exists():
                model_specs[mtype] = mpath
            else:
                print(f"⚠ Checkpoint not found: {mpath}")
            i = j
        else:
            # Orphan token — skip
            i += 1

    if not model_specs:
        print("No valid models specified!")
        return

    print("=" * 70)
    print("SHOCK TUBE MODEL COMPARISON")
    print("=" * 70)
    for mtype, mpath in model_specs.items():
        print(f"  {MODEL_LABELS.get(mtype, mtype):20s} → {mpath}")
    print(f"  Rollout steps: {args.rollout_steps}")
    print("=" * 70)

    # ---- Load test data ----
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
            print(f"  Error loading {f}: {e}")

    print(f"Loaded {len(sims)} test simulations")
    if not sims:
        return

    # ---- Load models ----
    sample_data = sims[0][1][0]
    if not hasattr(sample_data, 'pos') or sample_data.pos is None:
        sample_data.pos = sample_data.x[:, :NUM_STATIC]

    models = {}
    for mtype, mpath in model_specs.items():
        try:
            print(f"\nLoading {MODEL_LABELS.get(mtype, mtype)}...")
            models[mtype] = LOADERS[mtype](mpath, sample_data, device)
            n_params = sum(p.numel() for p in models[mtype].parameters())
            print(f"  ✓ {n_params:,} parameters")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback; traceback.print_exc()

    if not models:
        print("No models loaded successfully!")
        return

    # ---- Evaluate ----
    all_agg_metrics = {}       # model → aggregate metrics
    per_sim_metrics = {}       # model → sim_name → metrics
    per_step_rrmse = {}        # model → list of per-step RRMSE lists
    all_timeseries_data = {}   # sim_name → model → predictions
    sim_params = {}            # sim_name → {pressure, density, delta_t}

    # ---- Extract global parameters per simulation ----
    print("\nExtracting global parameters...")
    for sim_name, sim in sims:
        # Try from data attributes first
        params = extract_global_params_from_data(sim[0])
        
        # Fallback: parse from filename
        if not params or all(v == 0 for v in params.values()):
            p, rho = parse_params_from_filename(sim_name)
            if p is not None:
                params['pressure'] = p
            if rho is not None:
                params['density'] = rho
        
        if params:
            sim_params[sim_name] = params
    
    n_with_params = sum(1 for v in sim_params.values() if v)
    print(f"  {n_with_params}/{len(sims)} simulations have parameter info")
    if sim_params:
        sample = next(iter(sim_params.values()))
        print(f"  Available params: {list(sample.keys())}")

    for mtype, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"Evaluating {MODEL_LABELS.get(mtype, mtype)}")
        print(f"{'=' * 50}")

        rollout_fn = ROLLOUT_FNS[mtype]
        all_preds, all_gts = [], []
        sim_metrics = {}
        step_rrmse_all = []

        for idx, (sim_name, sim) in enumerate(tqdm(sims, desc=mtype)):
            try:
                # GT targets
                steps = min(args.rollout_steps, len(sim) - 1)
                gt_list = []
                for t in range(steps):
                    if hasattr(sim[t], 'y') and sim[t].y is not None:
                        gt_list.append(apply_skip(sim[t].y).cpu().numpy())
                    else:
                        gt_list.append(extract_dynamic(sim[t + 1].x).cpu().numpy())

                # Rollout
                pred_list = rollout_fn(model, sim, steps, device)
                pred_list = pred_list[:len(gt_list)]

                all_preds.extend(pred_list)
                all_gts.extend(gt_list)

                # Per-sim
                sim_m = compute_metrics(pred_list, gt_list)
                sim_metrics[sim_name] = sim_m

                # Per-step RRMSE
                step_rrmse_all.append(compute_per_step_rrmse(pred_list, gt_list))

                # Store timeseries for visualization
                if idx < args.num_viz:
                    if sim_name not in all_timeseries_data:
                        all_timeseries_data[sim_name] = {'gt': gt_list}
                    all_timeseries_data[sim_name][mtype] = pred_list

            except Exception as e:
                print(f"  ✗ {sim_name}: {e}")
                import traceback; traceback.print_exc()

        # Aggregate
        if all_preds:
            agg = compute_metrics(all_preds, all_gts)
            all_agg_metrics[mtype] = agg
            per_sim_metrics[mtype] = sim_metrics
            per_step_rrmse[mtype] = step_rrmse_all

            print(f"\n  Aggregate RRMSE: {agg['rrmse']:.4f}  RMSE: {agg['rmse']:.6f}")
            for vn in VAR_NAMES:
                print(f"    {vn:15s} RRMSE={agg[f'{vn}_rrmse']:.4f}  RMSE={agg[f'{vn}_rmse']:.6f}  R²={agg[f'{vn}_r2']:.4f}")

    # ---- Print comparison table ----
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY — RRMSE")
    print(f"{'=' * 70}")

    header = f"{'Model':20s}"
    for vn in VAR_NAMES:
        header += f" | {vn:>12s}"
    header += f" | {'Overall':>10s}"
    print(header)
    print("-" * len(header))

    for mtype, agg in all_agg_metrics.items():
        row = f"{MODEL_LABELS.get(mtype, mtype):20s}"
        for vn in VAR_NAMES:
            row += f" | {agg.get(f'{vn}_rrmse', float('nan')):12.4f}"
        row += f" | {agg.get('rrmse', float('nan')):10.4f}"
        print(row)

    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY — RMSE")
    print(f"{'=' * 70}")

    header = f"{'Model':20s}"
    for vn in VAR_NAMES:
        header += f" | {vn:>12s}"
    header += f" | {'Overall':>10s}"
    print(header)
    print("-" * len(header))

    for mtype, agg in all_agg_metrics.items():
        row = f"{MODEL_LABELS.get(mtype, mtype):20s}"
        for vn in VAR_NAMES:
            row += f" | {agg.get(f'{vn}_rmse', float('nan')):12.6f}"
        row += f" | {agg.get('rmse', float('nan')):10.6f}"
        print(row)

    # ---- Plots ----
    print("\nGenerating plots...")

    if all_agg_metrics:
        plot_comparison_table(all_agg_metrics, output_dir)
        plot_r2_comparison(all_agg_metrics, output_dir)

    if per_step_rrmse:
        plot_error_accumulation(per_step_rrmse, output_dir)

    if per_sim_metrics:
        plot_per_sim_rrmse(per_sim_metrics, output_dir)

    for sim_name, ts_data in all_timeseries_data.items():
        plot_timeseries(ts_data, output_dir, sim_name)

    # Parameter-conditioned plots
    if sim_params and per_sim_metrics:
        available_params = set()
        for sp in sim_params.values():
            available_params.update(sp.keys())
        
        for param_name in available_params:
            plot_rrmse_vs_parameter(per_sim_metrics, sim_params, param_name, output_dir)
            plot_per_variable_vs_parameter(per_sim_metrics, sim_params, param_name, output_dir)
        
        plot_parameter_heatmap(per_sim_metrics, sim_params, output_dir)

    # ---- Save JSON ----
    results = {
        'aggregate': {k: v for k, v in all_agg_metrics.items()},
        'per_simulation': {
            mtype: {sn: m for sn, m in sm.items()}
            for mtype, sm in per_sim_metrics.items()
        },
        'simulation_parameters': sim_params,
        'rollout_steps': args.rollout_steps,
        'num_simulations': len(sims),
        'models': {k: str(v) for k, v in model_specs.items()},
    }
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    main()