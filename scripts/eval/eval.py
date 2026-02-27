#!/usr/bin/env python3
"""
Unified Multi-Dataset Model Evaluation
========================================
Single script for all G-PARC datasets: shock tube, elastoplastic, river (HEC-RAS).

Core rollout metrics (computed at every timestep):
  RMSE(t)   = sqrt(1/N * Σ(y - ŷ)²)         per variable, over nodes
  RRMSE(t)  = RMSE(t) / RMS(gt)              relative, comparable across variables
  NMSE(t)   = MSE(t) / Var(gt)               = 1 - R²(t), normalized by variance
  SSIM(t)   = structural similarity           spatial structure quality

Table metrics (single numbers):
  RRMSE AUC    = (1/T) Σ_t RRMSE(t)          integrated over rollout
  RRMSE final  = RRMSE(T)                     hardest point
  NMSE AUC     = (1/T) Σ_t NMSE(t)           integrated
  SSIM AUC     = (1/T) Σ_t SSIM(t)           integrated
  R²           = 1 - SS_res/SS_tot            pooled over all steps

Plots (quantitative):
  - RRMSE over time (mean ± std)
  - Per-variable RRMSE over time
  - NMSE over time
  - SSIM over time
  - Per-simulation bar chart (RRMSE AUC)
  - Cross-dataset summary

Usage:
    python eval_unified.py --datasets shocktube \\
        --st_test_dir /path/to/test \\
        --st_models gparcv2:/path/v2.pth gparcv1:/path/v1.pth

    python eval_unified.py --datasets elasto \\
        --el_test_dir /path/to/test --el_norm_stats /path/to/norm.json \\
        --el_models gparcv2:/path/v2.pth gparcv1:/path/v1.pth

    python eval_unified.py --datasets shocktube elasto \\
        --st_test_dir ... --st_models ... \\
        --el_test_dir ... --el_norm_stats ... --el_models ...
"""

import argparse, sys, os, json, time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ==============================================================================
# CONSTANTS & STYLE
# ==============================================================================

EPS = 1e-12

MODEL_COLORS = {
    'G-PARC with MLS':         '#1f77b4',
    'G-PARC MLS (NoSPADE)':    '#17becf',
    'G-PARC Baseline':         '#d62728',
    'MeshGraphKAN':            '#ff7f0e',
    'MeshGraphNet':            '#2ca02c',
    'GraphSAGE':               '#9467bd',
}
MODEL_STYLES = {
    'G-PARC with MLS':         '-',
    'G-PARC MLS (NoSPADE)':    (0, (5, 2)),
    'G-PARC Baseline':         '--',
    'MeshGraphKAN':            '-.',
    'MeshGraphNet':            ':',
    'GraphSAGE':               (0, (3, 1, 1, 1)),
}

# Shock tube
ST_VAR_NAMES = ['density', 'x_momentum', 'total_energy']
ST_NUM_STATIC = 2
ST_NUM_DYNAMIC = 3
ST_SKIP_INDICES = [2]
ST_RAW_DYNAMIC = ST_NUM_DYNAMIC + len(ST_SKIP_INDICES)
ST_KEEP_INDICES = [i for i in range(ST_RAW_DYNAMIC) if i not in ST_SKIP_INDICES]

# Elastoplastic
EL_VAR_NAMES = ['$U_x$', '$U_y$']
EL_NUM_STATIC = 2
EL_NUM_DYNAMIC = 2

# River (HEC-RAS)
RV_VAR_NAMES = ['Depth', 'Volume', 'Velocity_X', 'Velocity_Y']
RV_NUM_STATIC = 9
RV_NUM_DYNAMIC = 4


# ==============================================================================
# MLS CACHE MANAGEMENT
# ==============================================================================

def _clear_mls_caches(model):
    """Clear all MLS caches in the derivative solver, handling both direct and nested solvers."""
    deriv = getattr(model, 'derivative_solver', getattr(model, 'derivative_solver_physics', None))
    if deriv is None:
        return
    if hasattr(deriv, '_weights_initialized'):
        deriv._weights_initialized = False

    # Collect all unique solver objects (direct attrs + nested in list_strain/list_laplacian)
    solvers = set()
    for attr in ['gradient_solver', 'laplacian_solver']:
        s = getattr(deriv, attr, None)
        if s is not None:
            solvers.add(id(s))
            if hasattr(s, 'clear_caches'): s.clear_caches()
            elif hasattr(s, 'geo_cache'): s.geo_cache.clear()
            elif hasattr(s, 'weights_cache'): s.weights_cache.clear()
    # Nested: list_strain[i].gradient_solver, list_laplacian[i].laplacian_solver
    for list_attr, solver_attr in [('list_strain', 'gradient_solver'), ('list_laplacian', 'laplacian_solver')]:
        lst = getattr(deriv, list_attr, None)
        if lst is None:
            continue
        for item in lst:
            if item is None:
                continue
            s = getattr(item, solver_attr, None)
            if s is not None and id(s) not in solvers:
                solvers.add(id(s))
                if hasattr(s, 'clear_caches'): s.clear_caches()
                elif hasattr(s, 'geo_cache'): s.geo_cache.clear()
                elif hasattr(s, 'weights_cache'): s.weights_cache.clear()


# ==============================================================================
# SSIM IMPLEMENTATION (no scikit-image dependency)
# ==============================================================================

def _ssim_1d(x, y, C1=1e-4, C2=9e-4):
    """
    SSIM between two 1D vectors (one field variable across all nodes).
    Uses mean/variance/covariance — the "global SSIM" formulation
    appropriate for unstructured meshes where local windowing isn't defined.

    Args:
        x, y: 1D arrays [N] (prediction and ground truth for one variable)
        C1, C2: stability constants (scaled for normalized data)

    Returns:
        float: SSIM ∈ [-1, 1], higher is better
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sig_x = np.std(x)
    sig_y = np.std(y)
    sig_xy = np.mean((x - mu_x) * (y - mu_y))

    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sig_x**2 + sig_y**2 + C2)
    return float(num / den)


def compute_ssim(pred, gt):
    """
    Mean SSIM across all variables for one timestep.

    Args:
        pred: [N, D] array
        gt:   [N, D] array

    Returns:
        float: mean SSIM across D variables
    """
    D = pred.shape[1] if pred.ndim > 1 else 1
    if D == 1:
        return _ssim_1d(pred.ravel(), gt.ravel())

    ssims = []
    for d in range(D):
        # Scale C1, C2 relative to data range for this variable
        data_range = gt[:, d].max() - gt[:, d].min()
        if data_range < EPS:
            ssims.append(1.0)  # constant field → perfect match if pred is also constant
            continue
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        ssims.append(_ssim_1d(pred[:, d], gt[:, d], C1, C2))
    return float(np.mean(ssims))


# ==============================================================================
# UNIFIED METRICS
# ==============================================================================

def compute_per_step_metrics(pred_list, gt_list):
    """
    All per-step metrics over a rollout.

    Returns dict with [T] arrays:
        'rmse', 'rrmse', 'nmse', 'ssim'
    """
    T = min(len(pred_list), len(gt_list))
    rmse_t  = np.full(T, np.nan)
    rrmse_t = np.full(T, np.nan)
    nmse_t  = np.full(T, np.nan)
    ssim_t  = np.full(T, np.nan)

    for t in range(T):
        p, g = pred_list[t], gt_list[t]
        if np.any(np.isnan(p)):
            continue

        diff = p - g
        mse = np.mean(diff ** 2)
        rmse_val = np.sqrt(mse)
        rms_gt = np.sqrt(np.mean(g ** 2))
        var_gt = np.var(g)

        rmse_t[t]  = rmse_val
        rrmse_t[t] = rmse_val / max(rms_gt, EPS)
        nmse_t[t]  = mse / max(var_gt, EPS)
        ssim_t[t]  = compute_ssim(p, g)

    return {'rmse': rmse_t, 'rrmse': rrmse_t, 'nmse': nmse_t, 'ssim': ssim_t}


def compute_per_step_per_variable(pred_list, gt_list, var_names):
    """Per-step metrics for each variable separately."""
    T = min(len(pred_list), len(gt_list))
    result = {}
    for vi, vn in enumerate(var_names):
        rmse_t  = np.full(T, np.nan)
        rrmse_t = np.full(T, np.nan)
        nmse_t  = np.full(T, np.nan)
        ssim_t  = np.full(T, np.nan)
        for t in range(T):
            p = pred_list[t][:, vi]
            g = gt_list[t][:, vi]
            if np.any(np.isnan(p)):
                continue
            diff = p - g
            mse = np.mean(diff ** 2)
            r = np.sqrt(mse)
            rms_gt = np.sqrt(np.mean(g ** 2))
            var_gt = np.var(g)
            data_range = g.max() - g.min()

            rmse_t[t]  = r
            rrmse_t[t] = r / max(rms_gt, EPS)
            nmse_t[t]  = mse / max(var_gt, EPS)

            if data_range > EPS:
                C1 = (0.01 * data_range) ** 2
                C2 = (0.03 * data_range) ** 2
                ssim_t[t] = _ssim_1d(p, g, C1, C2)
            else:
                ssim_t[t] = 1.0

        result[vn] = {'rmse': rmse_t, 'rrmse': rrmse_t, 'nmse': nmse_t, 'ssim': ssim_t}
    return result


def compute_table_metrics(pred_list, gt_list):
    """
    All table metrics from one simulation rollout.

    Returns dict with scalar metrics + per_step arrays.
    """
    step = compute_per_step_metrics(pred_list, gt_list)

    # Helper: AUC and final from a per-step array
    def auc_and_final(arr):
        valid = arr[~np.isnan(arr)]
        auc = float(np.mean(valid)) if len(valid) > 0 else np.nan
        fin = float(arr[np.where(~np.isnan(arr))[0][-1]]) if len(valid) > 0 else np.nan
        return auc, fin

    rrmse_auc, rrmse_fin = auc_and_final(step['rrmse'])
    rmse_auc, rmse_fin   = auc_and_final(step['rmse'])
    nmse_auc, nmse_fin   = auc_and_final(step['nmse'])
    ssim_auc, ssim_fin   = auc_and_final(step['ssim'])

    # Pooled R²
    all_p = np.concatenate(pred_list, axis=0)
    all_g = np.concatenate(gt_list, axis=0)
    ss_res = np.sum((all_g - all_p) ** 2)
    ss_tot = np.sum((all_g - all_g.mean(axis=0)) ** 2)
    r2_pool = 1.0 - ss_res / max(ss_tot, EPS)

    return {
        'rrmse_auc': rrmse_auc,     'rrmse_final': rrmse_fin,
        'rmse_auc': rmse_auc,       'rmse_final': rmse_fin,
        'nmse_auc': nmse_auc,       'nmse_final': nmse_fin,
        'ssim_auc': ssim_auc,       'ssim_final': ssim_fin,
        'r2_pooled': float(r2_pool),
        'per_step': step,
    }


def aggregate_sim_metrics(sim_list):
    """Aggregate table metrics across simulations → mean ± std + per-step curves."""
    scalar_keys = [
        'rrmse_auc', 'rrmse_final',
        'rmse_auc', 'rmse_final',
        'nmse_auc', 'nmse_final',
        'ssim_auc', 'ssim_final',
        'r2_pooled',
    ]
    agg = {}
    for k in scalar_keys:
        vals = [m[k] for m in sim_list if not np.isnan(m[k])]
        agg[f'{k}_mean'] = float(np.mean(vals)) if vals else np.nan
        agg[f'{k}_std'] = float(np.std(vals)) if vals else np.nan

    # Per-step curves
    step_keys = ['rrmse', 'rmse', 'nmse', 'ssim']
    max_T = max(len(m['per_step']['rrmse']) for m in sim_list) if sim_list else 0

    for sk in step_keys:
        arrs = [m['per_step'][sk] for m in sim_list]
        pad = np.full((len(arrs), max_T), np.nan)
        for i, a in enumerate(arrs):
            pad[i, :len(a)] = a
        agg[f'per_step_{sk}_mean'] = np.nanmean(pad, axis=0)
        agg[f'per_step_{sk}_std'] = np.nanstd(pad, axis=0)

    agg['n_valid'] = sum(1 for m in sim_list if not np.isnan(m['rrmse_final']))
    agg['n_diverged'] = sum(1 for m in sim_list if np.isnan(m['rrmse_final']))
    return agg


# ==============================================================================
# SHOCK TUBE — DATA & MODELS
# ==============================================================================

def st_extract_dynamic(x):
    raw = x[:, ST_NUM_STATIC:ST_NUM_STATIC + ST_RAW_DYNAMIC]
    return raw[:, ST_KEEP_INDICES]

def st_apply_skip(y):
    return y[:, ST_KEEP_INDICES]

def st_extract_global_params(data):
    parts = []
    for attrs in [['global_pressure', 'pressure'],
                  ['global_density', 'density_param'],
                  ['global_delta_t', 'delta_t']]:
        for a in attrs:
            if hasattr(data, a):
                parts.append(getattr(data, a)); break
        else:
            parts.append(torch.zeros(1, device=data.x.device))
    if len(parts) < 3 or all(p.item() == 0 for p in parts):
        if hasattr(data, 'global_params') and data.global_params.numel() >= 3:
            gp = data.global_params
            parts = [gp[0].unsqueeze(0), gp[1].unsqueeze(0), gp[2].unsqueeze(0)]
    gp = torch.cat([p.view(1) for p in parts])
    return gp.unsqueeze(0).expand(data.x.size(0), -1)

def st_load_data(test_dir, max_sims=None):
    files = sorted(Path(test_dir).glob("*.pt"))
    if max_sims: files = files[:max_sims]
    sims = []
    for f in tqdm(files, desc="Loading shock tube data"):
        try:
            sim = torch.load(f, weights_only=False)
            if isinstance(sim, list) and len(sim) > 0:
                sims.append((f.stem, sim))
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    return sims


# --- Shock tube model loaders ---

def st_load_gparcv1(ckpt_path, sample_data, device):
    from models.shocktube import GPARC
    from utilities.featureextractor import FeatureExtractorGNN
    from differentiator.differentiator import DerivativeGNN
    from integrator.integrator import IntegralGNN

    sf, df = ST_NUM_STATIC, ST_NUM_DYNAMIC
    fe = FeatureExtractorGNN(in_channels=sf, hidden_channels=32, out_channels=32,
                              depth=2, pool_ratios=0.2, heads=2, concat=True, dropout=0.2)
    ds = DerivativeGNN(in_channels=32+df+64, hidden_channels=128, out_channels=df,
                        num_layers=3, heads=4, concat=True, dropout=0.2, use_residual=True)
    ig = IntegralGNN(in_channels=df, hidden_channels=128, out_channels=df,
                      num_layers=3, heads=4, concat=True, dropout=0.2, use_residual=True)
    model = GPARC(feature_extractor=fe, derivative_solver=ds, integral_solver=ig,
                   num_static_feats=sf, num_dynamic_feats=df,
                   skip_dynamic_indices=ST_SKIP_INDICES, feature_out_channels=32)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.to(device).eval()
    return model


def st_load_gparcv2(ckpt_path, sample_data, device):
    from models.shocktube_gparcv2 import GPARC_ShockTube_V2
    from differentiator.nospade import ShockTubeDifferentiator
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d

    ckpt_dir = Path(ckpt_path).parent
    config = json.load(open(ckpt_dir/"config.json")) if (ckpt_dir/"config.json").exists() else {}
    ns = json.load(open(ckpt_dir/"normalization_stats.json")) if (ckpt_dir/"normalization_stats.json").exists() else {}
    ps = ns.get('position', {})
    sf = config.get('num_static_feats', ST_NUM_STATIC)
    df = config.get('num_dynamic_feats', ST_NUM_DYNAMIC)
    feat_out = config.get('feature_out_channels', 128)

    gs = SolveGradientsLST(pos_mean=ps.get('mean'), pos_std=ps.get('std'))
    ls = SolveWeightLST2d(pos_mean=ps.get('mean'), pos_std=ps.get('std'), use_2hop_extension=False)
    fe = GraphConvFeatureExtractorV2(
        in_channels=sf+df, hidden_channels=config.get('hidden_channels', 64),
        out_channels=feat_out, num_layers=config.get('num_layers', 4),
        dropout=config.get('dropout', 0.2),
        use_layer_norm=config.get('use_layer_norm', True),
        use_relative_pos=config.get('use_relative_pos', True))
    ds = ShockTubeDifferentiator(
        num_static_feats=sf, num_dynamic_feats=df, feature_extractor=fe,
        gradient_solver=gs, laplacian_solver=ls, n_fe_features=feat_out,
        global_embed_dim=config.get('global_embed_dim', 64),
        global_param_dim=config.get('global_param_dim', 3),
        list_adv_idx=list(range(df)), list_dif_idx=list(range(df)),
        velocity_indices=[config.get('velocity_index', 1)],
        diffusion_type=config.get('diffusion_type', 'fd'),
        spade_random_noise=config.get('spade_random_noise', False),
        heads=config.get('spade_heads', 4), concat=config.get('spade_concat', True),
        dropout=config.get('spade_dropout', 0.1), zero_init=config.get('zero_init', False))

    if not hasattr(sample_data, 'pos') or sample_data.pos is None:
        sample_data.pos = sample_data.x[:, :sf]
    ds.initialize_weights(sample_data)

    model = GPARC_ShockTube_V2(
        derivative_solver_physics=ds, integrator_type=config.get('integrator', 'euler'),
        num_static_feats=sf, num_dynamic_feats=df,
        skip_dynamic_indices=config.get('skip_dynamic_indices', ST_SKIP_INDICES),
        global_param_dim=config.get('global_param_dim', 3),
        global_embed_dim=config.get('global_embed_dim', 64))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.to(device).eval()
    return model


def st_load_mgkan(ckpt_path, sample_data, device):
    sys.path.insert(0, str(Path(__file__).parent.parent / 'MeshGraphKan'))
    from train_shocktube import MeshGraphKAN, MeshGraphKANShocktubeRollout
    ckpt_dir = Path(ckpt_path).parent
    config = json.load(open(ckpt_dir/"config.json")) if (ckpt_dir/"config.json").exists() else {}
    base = MeshGraphKAN(input_dim_nodes=8, input_dim_edges=3, output_dim=ST_NUM_DYNAMIC,
                         processor_size=config.get('processor_size', 15),
                         hidden_dim_processor=config.get('hidden_dim_processor', 128),
                         num_harmonics=config.get('num_harmonics', 5))
    wrapper = MeshGraphKANShocktubeRollout(model=base, num_static_feats=ST_NUM_STATIC,
                                            num_dynamic_feats=ST_NUM_DYNAMIC,
                                            skip_dynamic_indices=ST_SKIP_INDICES, global_param_dim=3)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    wrapper.load_state_dict(ckpt.get('model_state_dict', ckpt))
    wrapper.to(device).eval()
    return wrapper


def st_load_mgnet(ckpt_path, sample_data, device):
    sys.path.insert(0, str(Path(__file__).parent.parent / 'MeshGraphNet'))
    from meshgraphnet import MeshGraphNet
    ckpt_dir = Path(ckpt_path).parent
    config = json.load(open(ckpt_dir/"config.json")) if (ckpt_dir/"config.json").exists() else {}
    model = MeshGraphNet(input_dim_node=8, input_dim_edge=3,
                          hidden_dim=config.get('hidden_dim', 128),
                          output_dim=ST_NUM_DYNAMIC, num_layers=config.get('num_layers', 10))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.to(device).eval()
    return model


def st_load_gsage(ckpt_path, sample_data, device):
    sys.path.insert(0, str(Path(__file__).parent.parent / 'GraphSAGE'))
    from models.graphsage import ShocktubeGNN
    ckpt_dir = Path(ckpt_path).parent
    config = json.load(open(ckpt_dir/"config.json")) if (ckpt_dir/"config.json").exists() else {}
    model = ShocktubeGNN(in_channels=config.get('in_channels', 6),
                          out_channels=config.get('out_channels', 4),
                          hidden_channels=config.get('hidden_channels', 177),
                          num_layers=config.get('num_layers', 8),
                          dropout=config.get('dropout', 0.0))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.to(device).eval()
    return model


# --- Shock tube rollouts ---

@torch.no_grad()
def st_rollout_gparcv1(model, sim, num_steps, device):
    sim = [d.to(device) for d in sim]
    for d in sim:
        if not hasattr(d, 'pos') or d.pos is None: d.pos = d.x[:, :ST_NUM_STATIC]
        if hasattr(d, 'global_params') and d.global_params.numel() >= 3:
            gp = d.global_params
            if not hasattr(d, 'global_pressure'):
                d.global_pressure = gp[0].unsqueeze(0)
                d.global_density = gp[1].unsqueeze(0)
                d.global_delta_t = gp[2].unsqueeze(0)
    first = sim[0]
    ga = torch.stack([first.global_pressure.flatten()[0],
                      first.global_density.flatten()[0],
                      first.global_delta_t.flatten()[0]])
    ge = model.global_processor(ga)
    ls = model.feature_extractor(first.x[:, :ST_NUM_STATIC], first.edge_index)
    ls = model.feature_norm(ls, ga)
    F = st_extract_dynamic(first.x)
    preds = []
    for step in range(min(num_steps, len(sim))):
        Fi = model.derivative_norm(F, ga)
        gc = ge.unsqueeze(0).repeat(sim[step].num_nodes, 1)
        Fd = model.derivative_solver(torch.cat([ls, Fi, gc], dim=-1), sim[step].edge_index)
        Fint = model.integral_solver(Fd, sim[step].edge_index)
        F = Fi + Fint
        preds.append(F.cpu().numpy())
    return preds


@torch.no_grad()
def st_rollout_gparcv2(model, sim, num_steps, device):
    sim = [d.to(device) for d in sim]
    for d in sim:
        if not hasattr(d, 'pos') or d.pos is None: d.pos = d.x[:, :ST_NUM_STATIC]
        if hasattr(d, 'global_params') and d.global_params.numel() >= 3:
            gp = d.global_params
            if not hasattr(d, 'global_pressure'):
                d.global_pressure = gp[0].unsqueeze(0)
                d.global_density = gp[1].unsqueeze(0)
                d.global_delta_t = gp[2].unsqueeze(0)
    deriv = getattr(model, 'derivative_solver', getattr(model, 'derivative_solver_physics', None))
    # Clear MLS caches for each new simulation (mesh topology varies)
    _clear_mls_caches(model)
    if deriv and hasattr(deriv, 'initialize_weights'): deriv.initialize_weights(sim[0])
    ga = model._extract_global_attrs(sim[0])
    ge = model.global_processor(ga)
    F = model._extract_dynamic(sim[0].x)
    preds = []
    for step in range(min(num_steps, len(sim))):
        F = model.step(static_feats=sim[step].x[:, :ST_NUM_STATIC],
                       dynamic_state=F.clone(), edge_index=sim[step].edge_index,
                       global_embed=ge, global_attrs=ga, dt=1.0)
        preds.append(F.cpu().numpy())
    return preds


@torch.no_grad()
def st_rollout_mgkan(wrapper, sim, num_steps, device):
    sim = [d.to(device) for d in sim]
    for d in sim:
        if not hasattr(d, 'pos') or d.pos is None: d.pos = d.x[:, :ST_NUM_STATIC]
        if hasattr(d, 'global_params') and d.global_params.numel() >= 3:
            gp = d.global_params
            if not hasattr(d, 'global_pressure'):
                d.global_pressure = gp[0].unsqueeze(0)
                d.global_density = gp[1].unsqueeze(0)
                d.global_delta_t = gp[2].unsqueeze(0)
    cur = wrapper._extract_dynamic(sim[0].x)
    preds = []
    for step in range(min(num_steps, len(sim))):
        d = sim[step]
        nf = torch.cat([d.x[:, :ST_NUM_STATIC], cur, wrapper._extract_global_params(d)], dim=-1)
        ef = wrapper.compute_edge_features(d)
        cur = cur + wrapper.model(nf, ef, d.edge_index)
        preds.append(cur.cpu().numpy())
    return preds


@torch.no_grad()
def st_rollout_mgnet(model, sim, num_steps, device):
    sim = [d.to(device) for d in sim]
    for d in sim:
        if not hasattr(d, 'pos') or d.pos is None: d.pos = d.x[:, :ST_NUM_STATIC]
        if hasattr(d, 'global_params') and d.global_params.numel() >= 3:
            gp = d.global_params
            if not hasattr(d, 'global_pressure'):
                d.global_pressure = gp[0].unsqueeze(0)
                d.global_density = gp[1].unsqueeze(0)
                d.global_delta_t = gp[2].unsqueeze(0)
    cur = st_extract_dynamic(sim[0].x)
    preds = []
    for step in range(min(num_steps, len(sim))):
        d = sim[step]
        pos = d.pos if hasattr(d, 'pos') and d.pos is not None else d.x[:, :ST_NUM_STATIC]
        nf = torch.cat([d.x[:, :ST_NUM_STATIC], cur, st_extract_global_params(d)], dim=-1)
        cur = cur + model(nf, model.compute_edge_features(pos, d.edge_index), d.edge_index)
        preds.append(cur.cpu().numpy())
    return preds


@torch.no_grad()
def st_rollout_gsage(model, sim, num_steps, device):
    from models.graphsage import compute_edge_attr
    sim = [d.to(device) for d in sim]
    for d in sim:
        if not hasattr(d, 'pos') or d.pos is None: d.pos = d.x[:, :ST_NUM_STATIC]
    cur = sim[0].x[:, ST_NUM_STATIC:ST_NUM_STATIC + ST_RAW_DYNAMIC]
    preds = []
    for step in range(min(num_steps, len(sim))):
        d = sim[step]
        nf = torch.cat([d.x[:, :ST_NUM_STATIC], cur], dim=-1)
        cur = cur + model(nf, d.edge_index, edge_attr=compute_edge_attr(d.pos, d.edge_index))
        preds.append(cur[:, ST_KEEP_INDICES].cpu().numpy())
    return preds


ST_REGISTRY = {
    'gparcv1': {'name': 'G-PARC Baseline',  'load': st_load_gparcv1, 'rollout': st_rollout_gparcv1},
    'gparcv2': {'name': 'G-PARC with MLS',  'load': st_load_gparcv2, 'rollout': st_rollout_gparcv2},
    'mgkan':   {'name': 'MeshGraphKAN',      'load': st_load_mgkan,   'rollout': st_rollout_mgkan},
    'mgnet':   {'name': 'MeshGraphNet',      'load': st_load_mgnet,   'rollout': st_rollout_mgnet},
    'gsage':   {'name': 'GraphSAGE',         'load': st_load_gsage,   'rollout': st_rollout_gsage},
}


# ==============================================================================
# ELASTOPLASTIC — DATA & MODELS
# ==============================================================================

def el_load_data(test_dir, max_sims=None):
    files = sorted(Path(test_dir).glob("simulation_*.pt"))
    if max_sims: files = files[:max_sims]
    sims = []
    for f in tqdm(files, desc="Loading elastoplastic data"):
        sims.append((f.stem, torch.load(f, weights_only=False)))
    return sims

def el_load_norm_stats(path):
    with open(path) as f: return json.load(f)


def el_load_gparcv2(ckpt_path, norm_stats, sample_data, device):
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from differentiator.differentiator import ElastoPlasticDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from models.globalelasto import GPARC_ElastoPlastic_Numerical
    ckpt_dir = Path(ckpt_path).parent
    cfg = json.load(open(ckpt_dir/"config.json")) if (ckpt_dir/"config.json").exists() else {}
    ps = norm_stats['position']
    pos_mean = [ps['x_pos']['mean'], ps['y_pos']['mean']]
    pos_std = [ps['x_pos']['std'], ps['y_pos']['std']]
    nm = norm_stats.get('normalization_method', 'global_max')
    mp = ps.get('max_position', 200.0)
    sf, df, foc = cfg.get('num_static_feats', 2), cfg.get('num_dynamic_feats', 2), cfg.get('feature_out_channels', 128)
    gs = SolveGradientsLST(pos_mean=pos_mean, pos_std=pos_std, norm_method=nm, max_position=mp)
    ls = SolveWeightLST2d(pos_mean=pos_mean, pos_std=pos_std, norm_method=nm, max_position=mp, min_neighbors=5, use_2hop_extension=False)
    fe = GraphConvFeatureExtractorV2(in_channels=sf, hidden_channels=cfg.get('hidden_channels', 128), out_channels=foc, num_layers=cfg.get('num_layers', 4), dropout=cfg.get('dropout', 0.0), use_layer_norm=cfg.get('use_layer_norm', True), use_relative_pos=cfg.get('use_relative_pos', True))
    diffs = ElastoPlasticDifferentiator(num_static_feats=sf, num_dynamic_feats=df, feature_extractor=fe, gradient_solver=gs, laplacian_solver=ls, n_fe_features=foc, list_strain_idx=cfg.get('list_strain_idx', [0,1]), list_laplacian_idx=cfg.get('list_laplacian_idx', [0,1]), spade_random_noise=cfg.get('spade_random_noise', False), heads=cfg.get('spade_heads', 4), concat=cfg.get('spade_concat', True), dropout=cfg.get('spade_dropout', 0.1), use_von_mises=cfg.get('use_von_mises', True), use_volumetric=cfg.get('use_volumetric', True), n_state_var=cfg.get('n_state_var', 0), zero_init=cfg.get('zero_init', True))
    diffs.initialize_weights(sample_data)
    model = GPARC_ElastoPlastic_Numerical(derivative_solver_physics=diffs, integrator_type=cfg.get('integrator', 'euler'), num_static_feats=sf, num_dynamic_feats=df, pos_mean=pos_mean, pos_std=pos_std, boundary_threshold=cfg.get('boundary_threshold', 0.5), clamp_output=cfg.get('clamp_output', False), norm_method=nm, max_position=mp).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    return model


def el_load_gparcv2_nospade(ckpt_path, norm_stats, sample_data, device):
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from differentiator.nospadeelasto import ElastoPlasticDifferentiatorNoSPADE
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from models.globalelasto import GPARC_ElastoPlastic_Numerical
    ckpt_dir = Path(ckpt_path).parent
    cfg = json.load(open(ckpt_dir/"config.json")) if (ckpt_dir/"config.json").exists() else {}
    ps = norm_stats['position']
    pos_mean = [ps['x_pos']['mean'], ps['y_pos']['mean']]
    pos_std = [ps['x_pos']['std'], ps['y_pos']['std']]
    nm = norm_stats.get('normalization_method', 'global_max')
    mp = ps.get('max_position', 200.0)
    sf, df, foc = cfg.get('num_static_feats', 2), cfg.get('num_dynamic_feats', 2), cfg.get('feature_out_channels', 128)
    gs = SolveGradientsLST(pos_mean=pos_mean, pos_std=pos_std, norm_method=nm, max_position=mp)
    ls = SolveWeightLST2d(pos_mean=pos_mean, pos_std=pos_std, norm_method=nm, max_position=mp, min_neighbors=5, use_2hop_extension=False)
    fe = GraphConvFeatureExtractorV2(in_channels=sf+df, hidden_channels=cfg.get('hidden_channels', 128), out_channels=foc, num_layers=cfg.get('num_layers', 4), dropout=cfg.get('dropout', 0.0), use_layer_norm=cfg.get('use_layer_norm', True), use_relative_pos=cfg.get('use_relative_pos', True))
    diffs = ElastoPlasticDifferentiatorNoSPADE(num_static_feats=sf, num_dynamic_feats=df, feature_extractor=fe, gradient_solver=gs, laplacian_solver=ls, n_fe_features=foc, list_strain_idx=cfg.get('list_strain_idx', [0,1]), list_laplacian_idx=cfg.get('list_laplacian_idx', [0,1]), use_von_mises=cfg.get('use_von_mises', True), use_volumetric=cfg.get('use_volumetric', True), n_state_var=cfg.get('n_state_var', 0))
    diffs.initialize_weights(sample_data)
    model = GPARC_ElastoPlastic_Numerical(derivative_solver_physics=diffs, integrator_type=cfg.get('integrator', 'euler'), num_static_feats=sf, num_dynamic_feats=df, pos_mean=pos_mean, pos_std=pos_std, boundary_threshold=cfg.get('boundary_threshold', 0.5), clamp_output=cfg.get('clamp_output', False), norm_method=nm, max_position=mp).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    return model


def el_load_gparcv1(ckpt_path, norm_stats, sample_data, device):
    from utilities.featureextractor import FeatureExtractorGNN
    from differentiator.differentiator import DerivativeGNN
    from integrator.integrator import IntegralGNN
    from models.parcv1_elasto import GPARC
    ckpt_dir = Path(ckpt_path).parent
    cfg = json.load(open(ckpt_dir/"config.json")) if (ckpt_dir/"config.json").exists() else {}
    sf, df, foc = cfg.get('num_static_feats', 2), cfg.get('num_dynamic_feats', 2), cfg.get('feature_out_channels', 128)
    fe = FeatureExtractorGNN(in_channels=sf, hidden_channels=cfg.get('hidden_channels', 128), out_channels=foc, depth=cfg.get('depth', 3), pool_ratios=cfg.get('pool_ratios', 0.2), heads=cfg.get('heads', 3), concat=True, dropout=cfg.get('dropout', 0.1))
    ds = DerivativeGNN(in_channels=foc+df, hidden_channels=cfg.get('deriv_hidden_channels', 12), out_channels=df, num_layers=cfg.get('deriv_num_layers', 3), heads=cfg.get('deriv_heads', 3), concat=True, dropout=cfg.get('deriv_dropout', 0.1), use_residual=cfg.get('deriv_use_residual', True))
    ig = IntegralGNN(in_channels=df, hidden_channels=cfg.get('integral_hidden_channels', 128), out_channels=df, num_layers=cfg.get('integral_num_layers', 3), heads=cfg.get('integral_heads', 4), concat=True, dropout=cfg.get('integral_dropout', 0.1), use_residual=cfg.get('integral_use_residual', True))
    skip = cfg.get('skip_dynamic_indices', [])
    if isinstance(skip, str): skip = [int(x) for x in skip.split(',') if x.strip()] if skip else []
    model = GPARC(feature_extractor=fe, derivative_solver=ds, integral_solver=ig, num_static_feats=sf, num_dynamic_feats=df, skip_dynamic_indices=skip, feature_out_channels=foc).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    return model


def el_load_mgkan(ckpt_path, norm_stats, sample_data, device):
    from models.meshgraphkan import MeshGraphKAN, MeshGraphKANElastoRollout
    kan = MeshGraphKAN(input_dim_nodes=4, input_dim_edges=3, output_dim=2, processor_size=4, mlp_activation_fn='relu', num_layers_node_processor=2, num_layers_edge_processor=2, hidden_dim_processor=128, hidden_dim_node_encoder=128, hidden_dim_edge_encoder=128, num_layers_edge_encoder=2, hidden_dim_node_decoder=128, num_layers_node_decoder=2, aggregation='sum', num_harmonics=5)
    model = MeshGraphKANElastoRollout(kan, num_static_feats=2, num_dynamic_feats=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    return model


def el_load_mgn(ckpt_path, norm_stats, sample_data, device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MeshGraphNet'))
    from meshgraphnet import MeshGraphNet
    model = MeshGraphNet(input_dim_node=4, input_dim_edge=3, hidden_dim=128, output_dim=2, num_layers=4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    return model


def el_load_gsage(ckpt_path, norm_stats, sample_data, device):
    from models.graphsage import load_model as _load
    return _load('elasto', ckpt_path, device=device)


# --- Elastoplastic rollouts ---

@torch.no_grad()
def el_rollout_gparcv2(model, simulation, num_steps, device):
    from torch_geometric.data import Data
    # Move to device temporarily (don't mutate originals)
    sim_gpu = []
    for d in simulation:
        dg = d.clone()
        dg.x = dg.x.to(device); dg.y = dg.y.to(device); dg.edge_index = dg.edge_index.to(device)
        dg.pos = dg.pos.to(device) if hasattr(dg, 'pos') and dg.pos is not None else dg.x[:, :2]
        sim_gpu.append(dg)
    # Clear MLS caches for each new simulation (mesh topology varies)
    _clear_mls_caches(model)
    deriv = model.derivative_solver
    if hasattr(deriv, 'initialize_weights'): deriv.initialize_weights(sim_gpu[0])
    sf, df = model.num_static_feats, model.num_dynamic_feats
    static = sim_gpu[0].x[:, :sf]; cur = sim_gpu[0].x[:, sf:sf+df].clone()
    inp = Data(x=torch.cat([static, cur], -1), edge_index=sim_gpu[0].edge_index, pos=static, y=sim_gpu[0].y)
    for a in ['elements', 'x_element', 'y_element', 'mesh_id']:
        if hasattr(sim_gpu[0], a): setattr(inp, a, getattr(sim_gpu[0], a))
    disps = [cur.cpu().numpy()]
    for t in range(num_steps):
        inp.x = torch.cat([static, cur], -1); inp.y = sim_gpu[t].y
        for a in ['x_element', 'y_element']:
            if hasattr(sim_gpu[t], a): setattr(inp, a, getattr(sim_gpu[t], a))
        cur = model([inp], dt=1.0, teacher_forcing_ratio=0.0)[0]
        disps.append(cur.cpu().numpy())
    del sim_gpu, inp, static, cur
    return disps


@torch.no_grad()
def el_rollout_gparcv1(model, simulation, num_steps, device):
    sim_gpu = []
    for d in simulation:
        dg = d.clone()
        dg.x = dg.x.to(device); dg.y = dg.y.to(device); dg.edge_index = dg.edge_index.to(device)
        sim_gpu.append(dg)
    sf, df = model.num_static_feats, model.num_dynamic_feats
    cur = sim_gpu[0].x[:, sf:sf+df].clone()
    disps = [cur.cpu().numpy()]
    for t in range(num_steps):
        dt = sim_gpu[t].clone(); dt.x = dt.x.clone(); dt.x[:, sf:sf+df] = cur
        cur = model([dt])[0]; disps.append(cur.cpu().numpy())
    del sim_gpu, cur
    return disps


@torch.no_grad()
def el_rollout_mgkan(model, simulation, num_steps, device):
    first = simulation[0]; sf, df = model.num_static_feats, model.num_dynamic_feats
    static = first.x[:, :sf].to(device); cur = first.x[:, sf:sf+df].clone().to(device)
    ei = first.edge_index.to(device); ef = model.compute_edge_features(static, ei)
    disps = [cur.cpu().numpy()]
    for t in range(num_steps):
        cur = cur + model.model(torch.cat([static, cur], -1), ef, ei)
        disps.append(cur.cpu().numpy())
    return disps


@torch.no_grad()
def el_rollout_mgn(model, simulation, num_steps, device):
    first = simulation[0]; static = first.x[:, :2].to(device)
    cur = first.x[:, 2:4].clone().to(device); ei = first.edge_index.to(device)
    ef = model.compute_edge_features(static, ei)
    disps = [cur.cpu().numpy()]
    for t in range(num_steps):
        cur = cur + model(torch.cat([static, cur], -1), ef, ei)
        disps.append(cur.cpu().numpy())
    return disps


@torch.no_grad()
def el_rollout_gsage(model, simulation, num_steps, device):
    from models.graphsage import compute_edge_attr
    first = simulation[0]; sf, df = model.num_static_feats, model.num_dynamic_feats
    static = first.x[:, :sf].to(device); cur = first.x[:, sf:sf+df].clone().to(device)
    ei = first.edge_index.to(device)
    pos = first.pos.to(device) if hasattr(first, 'pos') and first.pos is not None else static
    ef = compute_edge_attr(pos, ei)
    disps = [cur.cpu().numpy()]
    for t in range(num_steps):
        cur = cur + model(torch.cat([static, cur], -1), ei, edge_attr=ef)
        disps.append(cur.cpu().numpy())
    return disps


EL_REGISTRY = {
    'gparcv2':         {'name': 'G-PARC with MLS', 'load': el_load_gparcv2, 'rollout': el_rollout_gparcv2},
    'gparcv2_nospade': {'name': 'G-PARC MLS (NoSPADE)', 'load': el_load_gparcv2_nospade, 'rollout': el_rollout_gparcv2},
    'gparcv1':         {'name': 'G-PARC Baseline',  'load': el_load_gparcv1, 'rollout': el_rollout_gparcv1},
    'mgkan':           {'name': 'MeshGraphKAN',      'load': el_load_mgkan,   'rollout': el_rollout_mgkan},
    'mgn':             {'name': 'MeshGraphNet',      'load': el_load_mgn,     'rollout': el_rollout_mgn},
    'graphsage':       {'name': 'GraphSAGE',         'load': el_load_gsage,   'rollout': el_rollout_gsage},
}


# ==============================================================================
# RIVER (HEC-RAS) — DATA & MODELS
# ==============================================================================

def rv_load_data(test_dir, max_sims=None):
    files = sorted(Path(test_dir).glob("*.pt"))
    if max_sims: files = files[:max_sims]
    sims = []
    for f in tqdm(files, desc="Loading river data"):
        try:
            sim = torch.load(f, weights_only=False)
            if isinstance(sim, list) and len(sim) > 0:
                sims.append((f.stem, sim))
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    return sims


def rv_load_extrema(path):
    if path is None or not Path(path).exists():
        print(f"  ⚠ Extrema not found: {path}")
        return None
    extrema = torch.load(path, weights_only=False)
    print(f"  ✓ Loaded extrema: y_min={extrema['y_min'].tolist()}, y_max={extrema['y_max'].tolist()}")
    return extrema


def rv_denormalize(normalized, extrema):
    """Denormalize [N, D] array using min-max extrema."""
    if extrema is None:
        return normalized
    out = np.zeros_like(normalized)
    for v in range(normalized.shape[1]):
        y_min = extrema['y_min'][v].item()
        y_max = extrema['y_max'][v].item()
        out[:, v] = normalized[:, v] * (y_max - y_min) + y_min
    return out


# ---- Model loaders (signature: ckpt_path, sample_data, device) ----

def rv_load_gparcv2(ckpt_path, sample_data, device):
    from models.riverV2 import GPARC_River_V2
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from differentiator.riverdifferentiator import RiverDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    sf, df = RV_NUM_STATIC, RV_NUM_DYNAMIC

    ckpt_dir = Path(ckpt_path).parent
    config_path = ckpt_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f: config = json.load(f)

    hidden = config.get('hidden_channels', 128)
    n_layers = config.get('num_layers', 4)
    feat_out = config.get('feature_out_channels', 128)

    fe = GraphConvFeatureExtractorV2(
        in_channels=sf + df, hidden_channels=hidden, out_channels=feat_out,
        num_layers=n_layers,
        use_layer_norm=config.get('use_layer_norm', True),
        use_relative_pos=config.get('use_relative_pos', True),
    )
    diff = RiverDifferentiator(
        num_static_feats=sf, num_dynamic_feats=df,
        feature_extractor=fe,
        gradient_solver=SolveGradientsLST(),
        laplacian_solver=SolveWeightLST2d(use_2hop_extension=False),
        n_fe_features=feat_out,
    )
    model = GPARC_River_V2(derivative_solver_physics=diff,
                            num_static_feats=sf, num_dynamic_feats=df)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


def rv_load_gparcv1(ckpt_path, sample_data, device):
    from scripts.river.gparc_new import GPARCRecurrent, FeatureExtractorGNN, DerivativeGNN, IntegralGNN
    sf, df = RV_NUM_STATIC, RV_NUM_DYNAMIC
    fe_out = 128
    fe = FeatureExtractorGNN(in_channels=sf, hidden_channels=64, out_channels=fe_out,
                              depth=2, pool_ratios=0.1, heads=4, concat=True, dropout=0.2)
    de = DerivativeGNN(in_channels=fe_out + df, out_channels=df,
                        heads=4, concat=True, dropout=0.2)
    ie = IntegralGNN(in_channels=df, out_channels=df,
                      heads=4, concat=True, dropout=0.2)
    model = GPARCRecurrent(fe, de, ie, num_static_feats=sf, num_dynamic_feats=df)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    return model.to(device).eval()


def rv_load_mgkan(ckpt_path, sample_data, device):
    from scripts.MeshGraphKan.train_river import MeshGraphKAN, MeshGraphKANRollout
    sf, df = RV_NUM_STATIC, RV_NUM_DYNAMIC
    ckpt_dir = Path(ckpt_path).parent
    config_path = ckpt_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f: config = json.load(f)
    inner = MeshGraphKAN(
        input_dim_nodes=sf + df, input_dim_edges=3, output_dim=df,
        hidden_dim_processor=config.get('hidden_dim', 128),
        hidden_dim_node_encoder=config.get('hidden_dim', 128),
        hidden_dim_edge_encoder=config.get('hidden_dim', 128),
        hidden_dim_node_decoder=config.get('hidden_dim', 128),
        processor_size=config.get('processor_size', 4),
        num_harmonics=config.get('num_harmonics', 5),
        aggregation=config.get('aggregation', 'sum'),
    )
    model = MeshGraphKANRollout(inner, num_static_feats=sf, num_dynamic_feats=df)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


def rv_load_mgnet(ckpt_path, sample_data, device):
    from models.meshgraphnet import MeshGraphNet
    sf, df = RV_NUM_STATIC, RV_NUM_DYNAMIC
    ckpt_dir = Path(ckpt_path).parent
    config_path = ckpt_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f: config = json.load(f)
    model = MeshGraphNet(
        input_dim_node=sf + df, input_dim_edge=3, output_dim=df,
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', config.get('processor_size', 10)),
    )
    model.num_static_feats = sf
    model.num_dynamic_feats = df
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    return model.to(device).eval()


def rv_load_gsage(ckpt_path, sample_data, device):
    from models.graphsage import load_model as load_gsage
    sf, df = RV_NUM_STATIC, RV_NUM_DYNAMIC
    return load_gsage('river', ckpt_path, device=device,
                      in_channels=sf + df, out_channels=df)


# ---- Rollout functions ----

def rv_rollout_gparcv2(model, simulation, num_steps, device):
    sf, df = RV_NUM_STATIC, RV_NUM_DYNAMIC
    simulation = [d.to(device) for d in simulation]
    # Ensure pos is set
    for d in simulation:
        if not hasattr(d, 'pos') or d.pos is None:
            d.pos = d.x[:, :2]
    # Clear MLS caches for each new simulation
    _clear_mls_caches(model)
    deriv = model.derivative_solver
    if hasattr(deriv, 'initialize_weights'):
        deriv.initialize_weights(simulation[0])

    static = simulation[0].x[:, :sf]
    current = simulation[0].x[:, sf:sf + df].clone()
    ei = simulation[0].edge_index
    if hasattr(simulation[0], 'mesh_id'):
        ei.mesh_id = simulation[0].mesh_id
    preds = []
    for _ in range(num_steps):
        F_pred = model.step(static_feats=static, dynamic_state=current,
                            edge_index=ei, dt=1.0)
        preds.append(F_pred.detach().cpu().numpy())
        current = F_pred.detach()
    return preds


def rv_rollout_gparcv1(model, simulation, num_steps, device):
    sf, df = RV_NUM_STATIC, RV_NUM_DYNAMIC
    simulation = [d.to(device) for d in simulation]
    static = simulation[0].x[:, :sf]
    ei = simulation[0].edge_index
    learned = model.feature_extractor(static, ei)
    current = simulation[0].x[:, sf:sf + df].clone()
    preds = []
    for _ in range(num_steps):
        Fdot = model.derivative_solver(torch.cat([current, learned], -1), ei)
        Fint = model.integral_solver(Fdot, ei)
        current = current + Fint
        preds.append(current.detach().cpu().numpy())
    return preds


def rv_rollout_mgkan(model, simulation, num_steps, device):
    sf, df = RV_NUM_STATIC, RV_NUM_DYNAMIC
    simulation = [d.to(device) for d in simulation]
    first = simulation[0]
    static = first.x[:, :sf]
    current = first.x[:, sf:sf + df].clone()
    ei = first.edge_index
    ef = model.compute_edge_features(first)
    preds = []
    for _ in range(num_steps):
        delta = model.model(torch.cat([static, current], -1), ef, ei)
        current = (current + delta).detach()
        preds.append(current.cpu().numpy())
    return preds


def rv_rollout_mgnet(model, simulation, num_steps, device):
    sf, df = RV_NUM_STATIC, RV_NUM_DYNAMIC
    simulation = [d.to(device) for d in simulation]
    first = simulation[0]
    static = first.x[:, :sf]
    current = first.x[:, sf:sf + df].clone()
    ei = first.edge_index
    pos = first.pos if hasattr(first, 'pos') and first.pos is not None else first.x[:, :2]
    ef = model.compute_edge_features(pos, ei)
    preds = []
    for _ in range(num_steps):
        delta = model(torch.cat([static, current], -1), ef, ei)
        current = (current + delta).detach()
        preds.append(current.cpu().numpy())
    return preds


def rv_rollout_gsage(model, simulation, num_steps, device):
    sf, df = RV_NUM_STATIC, RV_NUM_DYNAMIC
    simulation = [d.to(device) for d in simulation]
    first = simulation[0]
    static = first.x[:, :sf]
    current = first.x[:, sf:sf + df].clone()
    ei = first.edge_index
    ef = model.compute_edge_features(first)
    preds = []
    for _ in range(num_steps):
        delta = model(torch.cat([static, current], -1), ei, edge_attr=ef)
        current = (current + delta).detach()
        preds.append(current.cpu().numpy())
    return preds


RV_REGISTRY = {
    'gparcv2': {'name': 'G-PARC with MLS', 'load': rv_load_gparcv2, 'rollout': rv_rollout_gparcv2},
    'gparcv1': {'name': 'G-PARC Baseline',  'load': rv_load_gparcv1, 'rollout': rv_rollout_gparcv1},
    'mgkan':   {'name': 'MeshGraphKAN',      'load': rv_load_mgkan,   'rollout': rv_rollout_mgkan},
    'mgnet':   {'name': 'MeshGraphNet',      'load': rv_load_mgnet,   'rollout': rv_rollout_mgnet},
    'gsage':   {'name': 'GraphSAGE',         'load': rv_load_gsage,   'rollout': rv_rollout_gsage},
}


# ==============================================================================
# QUANTITATIVE PLOTS
# ==============================================================================

def _plot_metric_over_time(model_agg, metric_key, ylabel, output_path, title,
                            higher_is_better=False):
    """Generic: plot any per-step metric over time with mean ± std."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for mn, a in model_agg.items():
        mean = a[f'per_step_{metric_key}_mean']
        std = a[f'per_step_{metric_key}_std']
        t = np.arange(len(mean))
        c = MODEL_COLORS.get(mn); s = MODEL_STYLES.get(mn, '-')
        ax.plot(t, mean, linestyle=s, color=c, linewidth=2, label=mn)
        ax.fill_between(t, mean - std, mean + std, alpha=0.15, color=c)
    ax.set_xlabel('Rollout Step', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_var_over_time(model_var_agg, var_names, metric_key, ylabel,
                            output_path, title):
    """Multi-panel per-variable metric over time."""
    nv = len(var_names)
    fig, axes = plt.subplots(1, nv, figsize=(6*nv, 5), sharey=True)
    if nv == 1: axes = [axes]
    for vi, vn in enumerate(var_names):
        ax = axes[vi]
        for mn, vdict in model_var_agg.items():
            if vn not in vdict: continue
            a = vdict[vn]
            mean = a[f'per_step_{metric_key}_mean']
            std = a[f'per_step_{metric_key}_std']
            t = np.arange(len(mean))
            c = MODEL_COLORS.get(mn); s = MODEL_STYLES.get(mn, '-')
            ax.plot(t, mean, linestyle=s, color=c, linewidth=2, label=mn)
            ax.fill_between(t, mean - std, mean + std, alpha=0.15, color=c)
        ax.set_xlabel('Rollout Step', fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(vn, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)
    fig.suptitle(title, fontsize=13); plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_sim_bars(per_sim_data, output_path, title="Per-Simulation RRMSE AUC"):
    """Grouped bars: RRMSE AUC per simulation per model."""
    mnames = list(per_sim_data.keys())
    snames = list(per_sim_data[mnames[0]].keys())
    if len(snames) > 30: return
    fig, ax = plt.subplots(figsize=(max(12, len(snames)*0.8), 6))
    x = np.arange(len(snames)); w = 0.8 / len(mnames)
    for i, mn in enumerate(mnames):
        vals = [per_sim_data[mn][sn]['rrmse_auc'] for sn in snames]
        ax.bar(x + i*w, vals, w, label=mn, color=MODEL_COLORS.get(mn, f'C{i}'), alpha=0.85)
    ax.set_ylabel('RRMSE AUC'); ax.set_xticks(x + w*(len(mnames)-1)/2)
    ax.set_xticklabels([s[:20] for s in snames], rotation=45, ha='right', fontsize=7)
    ax.legend(); ax.set_title(title); plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_cross_dataset_summary(dataset_results, output_dir):
    """Summary: RRMSE AUC across datasets."""
    datasets = list(dataset_results.keys())
    all_m = sorted(set(m for dr in dataset_results.values() for m in dr))
    fig, ax = plt.subplots(figsize=(max(10, len(datasets)*3), 6))
    x = np.arange(len(datasets)); w = 0.8 / len(all_m)
    for i, mn in enumerate(all_m):
        vals = [dataset_results[ds].get(mn, {}).get('rrmse_auc_mean', 0) for ds in datasets]
        ax.bar(x + i*w, vals, w, label=mn, color=MODEL_COLORS.get(mn, f'C{i}'), alpha=0.85)
    ax.set_ylabel('RRMSE AUC', fontsize=12); ax.set_xticks(x + w*(len(all_m)-1)/2)
    ax.set_xticklabels(datasets, fontsize=11); ax.legend(fontsize=9)
    ax.set_title('Cross-Dataset: RRMSE AUC', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
    fig.savefig(output_dir / 'cross_dataset_summary.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {output_dir / 'cross_dataset_summary.png'}")


# ==============================================================================
# TABLE OUTPUT
# ==============================================================================

def print_table(model_agg, dataset_name):
    """Formatted comparison table with all metrics."""
    print(f"\n{'=' * 110}")
    print(f"  {dataset_name} — MODEL COMPARISON")
    print(f"{'=' * 110}")
    hdr = (f"{'Model':22s} | {'RRMSE AUC':>10s} | {'RRMSE fin':>10s} | "
           f"{'NMSE AUC':>9s} | {'SSIM AUC':>9s} | {'R²':>8s} | "
           f"{'RMSE AUC':>9s} | {'n':>3s}")
    print(hdr); print("-" * len(hdr))

    for mn, a in model_agg.items():
        print(f"{mn:22s} | "
              f"{a['rrmse_auc_mean']:10.4f} | {a['rrmse_final_mean']:10.4f} | "
              f"{a['nmse_auc_mean']:9.4f} | {a['ssim_auc_mean']:9.4f} | "
              f"{a['r2_pooled_mean']:8.4f} | {a['rmse_auc_mean']:9.4f} | "
              f"{a['n_valid']:3d}")
    print()
    for mn, a in model_agg.items():
        print(f"{'  (±std)':22s} | "
              f"±{a['rrmse_auc_std']:9.4f} | ±{a['rrmse_final_std']:9.4f} | "
              f"±{a['nmse_auc_std']:8.4f} | ±{a['ssim_auc_std']:8.4f} | "
              f"±{a['r2_pooled_std']:7.4f} | ±{a['rmse_auc_std']:8.4f} |")
    print(f"{'=' * 110}")


def print_per_variable_table(all_model_var_agg, var_names, dataset_name):
    """Formatted per-variable comparison table."""
    if not all_model_var_agg:
        return
    print(f"\n{'=' * 100}")
    print(f"  {dataset_name} — PER-VARIABLE METRICS")
    print(f"{'=' * 100}")

    model_names = list(all_model_var_agg.keys())
    hdr = f"{'Model':22s} | {'Variable':16s} | {'RRMSE AUC':>10s} | {'RRMSE fin':>10s} | {'NMSE AUC':>9s} | {'SSIM AUC':>9s} | {'RMSE AUC':>9s}"
    print(hdr)
    print("-" * len(hdr))

    for mn in model_names:
        vagg = all_model_var_agg[mn]
        for vi, vn in enumerate(var_names):
            if vn not in vagg or not vagg[vn]:
                continue
            a = vagg[vn]
            label = mn if vi == 0 else ""
            rrmse_auc = a.get('rrmse_auc_mean', np.nan)
            rrmse_fin = a.get('rrmse_final_mean', np.nan)
            nmse_auc = a.get('nmse_auc_mean', np.nan)
            ssim_auc = a.get('ssim_auc_mean', np.nan)
            rmse_auc = a.get('rmse_auc_mean', np.nan)
            # Clean display name (strip $ for printing)
            vn_clean = vn.replace('$', '').replace('\\', '')
            print(f"{label:22s} | {vn_clean:16s} | "
                  f"{rrmse_auc:10.4f} | {rrmse_fin:10.4f} | "
                  f"{nmse_auc:9.4f} | {ssim_auc:9.4f} | {rmse_auc:9.4f}")
        if mn != model_names[-1]:
            print("-" * len(hdr))

    print(f"{'=' * 100}")


def generate_latex_table(model_agg, caption, label):
    """LaTeX table with all metrics, best RRMSE AUC bolded."""
    lines = [
        r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{l c c c c c}",
        r"\toprule",
        r"Model & RRMSE$_\text{AUC}$ & RRMSE$_\text{final}$ & NMSE$_\text{AUC}$ & SSIM$_\text{AUC}$ & $R^2$ \\",
        r"\midrule",
    ]
    best_auc = min((a['rrmse_auc_mean'] for a in model_agg.values()
                    if not np.isnan(a['rrmse_auc_mean'])), default=np.inf)
    best_ssim = max((a['ssim_auc_mean'] for a in model_agg.values()
                     if not np.isnan(a['ssim_auc_mean'])), default=-np.inf)

    for mn, a in model_agg.items():
        def f(k): return f"${a[f'{k}_mean']:.4f} \\pm {a[f'{k}_std']:.4f}$"
        rrmse_s = f('rrmse_auc')
        if abs(a['rrmse_auc_mean'] - best_auc) < 1e-8: rrmse_s = f"\\textbf{{{rrmse_s}}}"
        ssim_s = f('ssim_auc')
        if abs(a['ssim_auc_mean'] - best_ssim) < 1e-8: ssim_s = f"\\textbf{{{ssim_s}}}"
        lines.append(f"  {mn} & {rrmse_s} & {f('rrmse_final')} & {f('nmse_auc')} & {ssim_s} & ${a['r2_pooled_mean']:.4f}$ \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


# ==============================================================================
# DATASET RUNNERS
# ==============================================================================

def parse_model_specs(raw_specs, registry):
    specs = {}; i = 0
    while i < len(raw_specs):
        token = raw_specs[i]; colon = token.find(':')
        if colon > 0 and token[:colon] in registry:
            mtype = token[:colon]; parts = [token[colon+1:]]; j = i + 1
            while j < len(raw_specs):
                nc = raw_specs[j].find(':')
                if nc > 0 and raw_specs[j][:nc] in registry: break
                parts.append(raw_specs[j]); j += 1
            mp = ' '.join(parts)
            if Path(mp).exists(): specs[mtype] = mp
            else: print(f"  ⚠ Not found: {mp}")
            i = j
        else: i += 1
    return specs


def _run_dataset(dataset_name, sims, models, rollout_steps, var_names,
                  extract_gt_fn, output_dir, num_viz=3):
    """
    Generic evaluation loop for any dataset.

    Args:
        extract_gt_fn: callable(sim, steps) → (gt_list, pred_ready_sim)
            Returns list of [N, D] GT arrays for each step
    """
    all_model_agg = {}
    all_model_var_agg = {}
    all_per_sim = {}

    for mtype, minfo in models.items():
        mname = minfo['name']
        print(f"\n{'=' * 50}\nEvaluating {mname}\n{'=' * 50}")

        sim_metrics, sim_var_metrics, per_sim = [], [], {}

        for idx, (sim_name, sim) in enumerate(tqdm(sims, desc=mname)):
            try:
                steps = min(rollout_steps, len(sim) - 1) if rollout_steps else len(sim) - 1
                gt_list = extract_gt_fn(sim, steps)
                pred_list = minfo['rollout'](minfo['model'], sim, steps, minfo['device'])
                pred_list = pred_list[:len(gt_list)]

                sm = compute_table_metrics(pred_list, gt_list)
                svm = compute_per_step_per_variable(pred_list, gt_list, var_names)
                sim_metrics.append(sm)
                sim_var_metrics.append(svm)
                per_sim[sim_name] = sm

            except Exception as e:
                print(f"  ✗ {sim_name}: {e}")
                import traceback; traceback.print_exc()
            finally:
                # Free GPU memory between simulations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if sim_metrics:
            all_model_agg[mname] = aggregate_sim_metrics(sim_metrics)
            # Per-variable: compute full scalar metrics per variable per simulation
            vagg = {}
            for vn in var_names:
                v_list = []
                for svm_dict in sim_var_metrics:
                    if vn not in svm_dict:
                        continue
                    vd = svm_dict[vn]
                    # Compute scalar AUC/final for each metric
                    entry = {'per_step': vd}
                    for mk in ['rrmse', 'rmse', 'nmse', 'ssim']:
                        arr = vd[mk]
                        valid = arr[~np.isnan(arr)]
                        auc = float(np.mean(valid)) if len(valid) > 0 else np.nan
                        fin = float(arr[np.where(~np.isnan(arr))[0][-1]]) if len(valid) > 0 else np.nan
                        entry[f'{mk}_auc'] = auc
                        entry[f'{mk}_final'] = fin
                    entry['r2_pooled'] = 0.0  # not meaningful per-variable without full concat
                    v_list.append(entry)
                vagg[vn] = aggregate_sim_metrics(v_list) if v_list else {}
            all_model_var_agg[mname] = vagg
            all_per_sim[mname] = per_sim

            a = all_model_agg[mname]
            print(f"  RRMSE AUC={a['rrmse_auc_mean']:.4f}  "
                  f"final={a['rrmse_final_mean']:.4f}  "
                  f"NMSE={a['nmse_auc_mean']:.4f}  "
                  f"SSIM={a['ssim_auc_mean']:.4f}  "
                  f"R²={a['r2_pooled_mean']:.4f}")

    # --- Output ---
    print_table(all_model_agg, dataset_name)
    print_per_variable_table(all_model_var_agg, var_names, dataset_name)

    d = output_dir; d.mkdir(parents=True, exist_ok=True)

    # Metric-over-time plots
    for metric, ylabel in [('rrmse', 'RRMSE'), ('nmse', 'NMSE'), ('ssim', 'SSIM')]:
        _plot_metric_over_time(all_model_agg, metric, ylabel,
                                d / f'{metric}_over_time.png',
                                f'{dataset_name}: {ylabel} Over Rollout')

    # Per-variable plots for RRMSE, NMSE, SSIM
    for metric, ylabel in [('rrmse', 'RRMSE'), ('nmse', 'NMSE'), ('ssim', 'SSIM')]:
        plot_per_var_over_time(all_model_var_agg, var_names, metric, ylabel,
                               d / f'{metric}_per_variable.png',
                               f'{dataset_name}: Per-Variable {ylabel}')

    # Per-sim bars
    if all_per_sim:
        plot_per_sim_bars(all_per_sim, d / 'per_sim_rrmse_auc.png',
                          f'{dataset_name}: Per-Simulation RRMSE AUC')

    # LaTeX
    latex = generate_latex_table(all_model_agg,
                                  f"{dataset_name} model comparison",
                                  f"tab:{dataset_name.lower().replace(' ', '_')}")
    with open(d / 'table.tex', 'w') as f: f.write(latex)
    print(f"  LaTeX → {d / 'table.tex'}")

    # JSON
    # Build per-variable aggregate for JSON (strip numpy arrays)
    per_var_json = {}
    for mname, vagg in all_model_var_agg.items():
        per_var_json[mname] = {}
        for vn, va in vagg.items():
            per_var_json[mname][vn] = {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv)
                                        for kk, vv in va.items() if not isinstance(vv, np.ndarray)}

    results = {
        'aggregate': {k: {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv)
                          for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
                      for k, v in all_model_agg.items()},
        'per_variable': per_var_json,
        'n_simulations': len(sims),
    }
    with open(d / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)

    return all_model_agg


def run_shocktube(args, device, output_dir):
    print(f"\n{'#' * 70}\n  SHOCK TUBE EVALUATION\n{'#' * 70}")
    specs = parse_model_specs(args.st_models, ST_REGISTRY)
    if not specs: print("No valid shock tube models!"); return {}

    sims = st_load_data(args.st_test_dir, args.max_sims)
    if not sims: return {}
    sample = sims[0][1][0]
    if not hasattr(sample, 'pos') or sample.pos is None: sample.pos = sample.x[:, :ST_NUM_STATIC]
    sample = sample.to(device)

    models = {}
    for mt, mp in specs.items():
        try:
            reg = ST_REGISTRY[mt]
            print(f"Loading {reg['name']}...")
            m = reg['load'](mp, sample, device)
            print(f"  ✓ {sum(p.numel() for p in m.parameters()):,} params")
            models[mt] = {'model': m, 'name': reg['name'], 'rollout': reg['rollout'], 'device': device}
        except Exception as e:
            print(f"  ✗ {e}"); import traceback; traceback.print_exc()
    if not models: return {}

    def extract_gt(sim, steps):
        gt = []
        for t in range(steps):
            if hasattr(sim[t], 'y') and sim[t].y is not None:
                gt.append(st_apply_skip(sim[t].y).cpu().numpy())
            else:
                gt.append(st_extract_dynamic(sim[t+1].x).cpu().numpy())
        return gt

    return _run_dataset('Shock Tube', sims, models, args.st_rollout_steps,
                         ST_VAR_NAMES, extract_gt, output_dir / 'shocktube',
                         args.num_viz)


def run_elasto(args, device, output_dir):
    print(f"\n{'#' * 70}\n  ELASTOPLASTIC EVALUATION\n{'#' * 70}")
    specs = parse_model_specs(args.el_models, EL_REGISTRY)
    if not specs: print("No valid elasto models!"); return {}

    norm_stats = el_load_norm_stats(args.el_norm_stats)
    max_disp = norm_stats['displacement']['max_displacement']
    print(f"max_disp = {max_disp:.1f} mm")

    sims = el_load_data(args.el_test_dir, args.max_sims)
    if not sims: return {}
    sample = sims[0][1][0]
    sample.pos = sample.x[:, :2].to(device); sample.x = sample.x.to(device)
    sample.edge_index = sample.edge_index.to(device)
    if hasattr(sample, 'y'): sample.y = sample.y.to(device)

    models = {}
    for mt, mp in specs.items():
        try:
            reg = EL_REGISTRY[mt]
            print(f"Loading {reg['name']}...")
            m = reg['load'](mp, norm_stats, sample, device)
            print(f"  ✓ {sum(p.numel() for p in m.parameters()):,} params")
            models[mt] = {'model': m, 'name': reg['name'], 'rollout': reg['rollout'], 'device': device}
        except Exception as e:
            print(f"  ✗ {e}"); import traceback; traceback.print_exc()
    if not models: return {}

    def extract_gt(sim, steps):
        # Physical-unit GT displacements (skip t=0 initial state)
        return [sim[t+1].x[:, 2:4].cpu().numpy() * max_disp for t in range(steps)]

    # Patch rollouts to also return physical units
    orig_models = {}
    for mt, minfo in models.items():
        orig_fn = minfo['rollout']
        def make_wrapper(fn):
            def wrapped(model, simulation, num_steps, device):
                disps = fn(model, simulation, num_steps, device)
                # Skip initial state, convert to physical
                return [d * max_disp for d in disps[1:]]
            return wrapped
        orig_models[mt] = {**minfo, 'rollout': make_wrapper(orig_fn)}

    return _run_dataset('Elastoplastic', sims, orig_models, None,
                         EL_VAR_NAMES, extract_gt, output_dir / 'elastoplastic',
                         args.num_viz)


def run_river(args, device, output_dir):
    print(f"\n{'#' * 70}\n  RIVER (HEC-RAS) EVALUATION\n{'#' * 70}")
    specs = parse_model_specs(args.rv_models, RV_REGISTRY)
    if not specs: print("No valid river models!"); return {}

    extrema = rv_load_extrema(args.rv_extrema)

    sims = rv_load_data(args.rv_test_dir, args.max_sims)
    if not sims: return {}
    sample = sims[0][1][0]
    if not hasattr(sample, 'pos') or sample.pos is None:
        sample.pos = sample.x[:, :2]
    sample = sample.to(device)

    models = {}
    for mt, mp in specs.items():
        try:
            reg = RV_REGISTRY[mt]
            print(f"Loading {reg['name']}...")
            m = reg['load'](mp, sample, device)
            print(f"  ✓ {sum(p.numel() for p in m.parameters()):,} params")
            models[mt] = {'model': m, 'name': reg['name'], 'rollout': reg['rollout'], 'device': device}
        except Exception as e:
            print(f"  ✗ {e}"); import traceback; traceback.print_exc()
    if not models: return {}

    sf, df = RV_NUM_STATIC, RV_NUM_DYNAMIC

    def extract_gt(sim, steps):
        """Extract GT targets, denormalized to physical units."""
        gt = []
        for t in range(steps):
            if hasattr(sim[t], 'y') and sim[t].y is not None and sim[t].y.shape[1] >= df:
                gt_norm = sim[t].y[:, :df].cpu().numpy()
            else:
                gt_norm = sim[t + 1].x[:, sf:sf + df].cpu().numpy()
            gt.append(rv_denormalize(gt_norm, extrema))
        return gt

    # Wrap rollouts to denormalize predictions
    wrapped_models = {}
    for mt, minfo in models.items():
        orig_fn = minfo['rollout']
        def make_wrapper(fn):
            def wrapped(model, simulation, num_steps, device):
                preds_norm = fn(model, simulation, num_steps, device)
                return [rv_denormalize(p, extrema) for p in preds_norm]
            return wrapped
        wrapped_models[mt] = {**minfo, 'rollout': make_wrapper(orig_fn)}

    return _run_dataset('River', sims, wrapped_models, args.rv_rollout_steps,
                         RV_VAR_NAMES, extract_gt, output_dir / 'river',
                         args.num_viz)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified multi-dataset evaluation",
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs='+', required=True, choices=['shocktube', 'elasto', 'river'])
    parser.add_argument("--output_dir", default="./unified_results")
    parser.add_argument("--max_sims", type=int, default=None)
    parser.add_argument("--num_viz", type=int, default=3)
    parser.add_argument("--device", default="cuda")

    # Shock tube
    parser.add_argument("--st_test_dir"); parser.add_argument("--st_models", nargs='+', default=[])
    parser.add_argument("--st_rollout_steps", type=int, default=40)

    # Elastoplastic
    parser.add_argument("--el_test_dir"); parser.add_argument("--el_norm_stats")
    parser.add_argument("--el_models", nargs='+', default=[])

    # River
    parser.add_argument("--rv_test_dir"); parser.add_argument("--rv_extrema")
    parser.add_argument("--rv_models", nargs='+', default=[])
    parser.add_argument("--rv_rollout_steps", type=int, default=None,
                        help="Max rollout steps for river (default: full sim length)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"{'=' * 70}\nUNIFIED EVALUATION | datasets={args.datasets} | device={device}\n{'=' * 70}")

    results = {}
    if 'shocktube' in args.datasets:
        if args.st_test_dir and args.st_models:
            results['Shock Tube'] = run_shocktube(args, device, output_dir)
        else: print("ERROR: need --st_test_dir and --st_models")

    if 'elasto' in args.datasets:
        if args.el_test_dir and args.el_norm_stats and args.el_models:
            results['Elastoplastic'] = run_elasto(args, device, output_dir)
        else: print("ERROR: need --el_test_dir, --el_norm_stats, --el_models")

    if 'river' in args.datasets:
        if args.rv_test_dir and args.rv_models:
            results['River'] = run_river(args, device, output_dir)
        else: print("ERROR: need --rv_test_dir and --rv_models")

    valid = {k: v for k, v in results.items() if v}
    if len(valid) > 1: plot_cross_dataset_summary(valid, output_dir)

    print(f"\n{'=' * 70}\n✓ All results → {output_dir}\n{'=' * 70}")


if __name__ == "__main__":
    main()