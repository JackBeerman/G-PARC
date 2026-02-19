#!/usr/bin/env python3
"""
Multi-Model Comparison for Elastoplastic Simulations
=====================================================

Computes RMSE and RRMSE over rollout timesteps for multiple models,
generates publication-quality time-series plots with mean ± std shading,
and outputs per-simulation and aggregate metrics.

Supported models:
  - gparcv2:  G-PARCv2 (GraphConv + Physics Operators)
  - gparcv1:  G-PARCv1 (GraphUNet + Learned Integration)
  - mgkan:    MeshGraphKAN
  - mgn:      MeshGraphNet

Adding new models:
  1. Add a load function: load_<name>(ckpt_path, norm_stats, sample_data, device) → model
  2. Add a rollout function: rollout_<name>(model, simulation, num_steps, device, **kw) → list of [N,2]
  3. Register in MODEL_REGISTRY at bottom of this file

Usage:
    python compare_elasto.py \\
        --test_dir /path/to/test \\
        --norm_stats /path/to/normalization_stats.json \\
        --models gparcv2 gparcv1 mgkan \\
        --gparcv2_ckpt /path/to/best_model.pth \\
        --gparcv1_ckpt /path/to/modelseq20_ep250.pth \\
        --mgkan_ckpt /path/to/best_model.pth \\
        --max_sims 20 \\
        --output_dir ./comparison_results
"""

import argparse
import sys
import os
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# UTILITIES
# ==============================================================================

def load_norm_stats(path):
    with open(path) as f:
        return json.load(f)


def load_simulations(test_dir, pattern="simulation_*.pt", max_sims=None):
    files = sorted(Path(test_dir).glob(pattern))
    if max_sims:
        files = files[:max_sims]
    print(f"Loading {len(files)} simulations from {test_dir}...")
    sims = []
    for f in files:
        sims.append((torch.load(f, weights_only=False), f.stem))
    return sims


def denorm_displacement(u_norm, norm_stats):
    """Normalized → physical displacement (mm)."""
    method = norm_stats.get('normalization_method', 'global_max')
    if method == 'global_max':
        return u_norm * norm_stats['displacement']['max_displacement']
    raise ValueError(f"Unknown norm method: {method}")


# ==============================================================================
# MODEL LOADERS
# ==============================================================================

def load_gparcv2(ckpt_path, norm_stats, sample_data, device, config_path=None):
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from differentiator.differentiator import ElastoPlasticDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from models.globalelasto import GPARC_ElastoPlastic_Numerical

    # Load architecture from config if available
    cfg = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"    Loaded v2 config from {config_path}")

    ps = norm_stats['position']
    pos_mean = [ps['x_pos']['mean'], ps['y_pos']['mean']]
    pos_std = [ps['x_pos']['std'], ps['y_pos']['std']]
    nm = norm_stats.get('normalization_method', 'global_max')
    mp = ps.get('max_position', 200.0)

    sf = cfg.get('num_static_feats', 2)
    df = cfg.get('num_dynamic_feats', 2)
    foc = cfg.get('feature_out_channels', 128)

    gs = SolveGradientsLST(pos_mean=pos_mean, pos_std=pos_std,
                            norm_method=nm, max_position=mp)
    ls = SolveWeightLST2d(pos_mean=pos_mean, pos_std=pos_std,
                           norm_method=nm, max_position=mp,
                           min_neighbors=5, use_2hop_extension=False)

    fe = GraphConvFeatureExtractorV2(
        in_channels=sf,
        hidden_channels=cfg.get('hidden_channels', 128),
        out_channels=foc,
        num_layers=cfg.get('num_layers', 4),
        dropout=cfg.get('dropout', 0.0),
        use_layer_norm=cfg.get('use_layer_norm', True),
        use_relative_pos=cfg.get('use_relative_pos', True))

    ds = ElastoPlasticDifferentiator(
        num_static_feats=sf, num_dynamic_feats=df,
        feature_extractor=fe, gradient_solver=gs, laplacian_solver=ls,
        n_fe_features=foc,
        list_strain_idx=cfg.get('list_strain_idx', [0, 1]),
        list_laplacian_idx=cfg.get('list_laplacian_idx', [0, 1]),
        spade_random_noise=cfg.get('spade_random_noise', False),
        heads=cfg.get('spade_heads', 4),
        concat=cfg.get('spade_concat', True),
        dropout=cfg.get('spade_dropout', 0.1),
        use_von_mises=cfg.get('use_von_mises', True),
        use_volumetric=cfg.get('use_volumetric', True),
        n_state_var=cfg.get('n_state_var', 0),
        zero_init=cfg.get('zero_init', True))
    ds.initialize_weights(sample_data)

    model = GPARC_ElastoPlastic_Numerical(
        derivative_solver_physics=ds,
        integrator_type=cfg.get('integrator', 'euler'),
        num_static_feats=sf, num_dynamic_feats=df,
        pos_mean=pos_mean, pos_std=pos_std,
        boundary_threshold=cfg.get('boundary_threshold', 0.5),
        clamp_output=cfg.get('clamp_output', False),
        norm_method=nm, max_position=mp).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"    G-PARCv2 loaded (epoch {ckpt.get('epoch', '?')})")
    return model


def load_gparcv1(ckpt_path, norm_stats, sample_data, device, config_path=None):
    from utilities.featureextractor import FeatureExtractorGNN
    from differentiator.differentiator import DerivativeGNN
    from integrator.integrator import IntegralGNN
    from models.parcv1_elasto import GPARC

    # Load architecture from config if available, else use defaults
    cfg = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"    Loaded v1 config from {config_path}")

    sf = cfg.get('num_static_feats', 2)
    df = cfg.get('num_dynamic_feats', 2)
    foc = cfg.get('feature_out_channels', 128)

    fe = FeatureExtractorGNN(
        in_channels=sf,
        hidden_channels=cfg.get('hidden_channels', 128),
        out_channels=foc,
        depth=cfg.get('depth', 3),
        pool_ratios=cfg.get('pool_ratios', 0.2),
        heads=cfg.get('heads', 3),
        concat=True,
        dropout=cfg.get('dropout', 0.1))

    ds = DerivativeGNN(
        in_channels=foc + df,
        hidden_channels=cfg.get('deriv_hidden_channels', 12),
        out_channels=df,
        num_layers=cfg.get('deriv_num_layers', 3),
        heads=cfg.get('deriv_heads', 3),
        concat=True,
        dropout=cfg.get('deriv_dropout', 0.1),
        use_residual=cfg.get('deriv_use_residual', True))

    ig = IntegralGNN(
        in_channels=df,
        hidden_channels=cfg.get('integral_hidden_channels', 128),
        out_channels=df,
        num_layers=cfg.get('integral_num_layers', 3),
        heads=cfg.get('integral_heads', 4),
        concat=True,
        dropout=cfg.get('integral_dropout', 0.1),
        use_residual=cfg.get('integral_use_residual', True))

    skip = cfg.get('skip_dynamic_indices', [])
    if isinstance(skip, str):
        skip = [int(x) for x in skip.split(',') if x.strip()] if skip else []

    model = GPARC(
        feature_extractor=fe, derivative_solver=ds, integral_solver=ig,
        num_static_feats=sf, num_dynamic_feats=df,
        skip_dynamic_indices=skip, feature_out_channels=foc).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"    G-PARCv1 loaded (epoch {ckpt.get('epoch', '?')})")
    return model


def load_mgkan(ckpt_path, norm_stats, sample_data, device):
    from train_meshgraphkan import MeshGraphKAN, MeshGraphKANRollout

    kan = MeshGraphKAN(
        input_dim_nodes=4, input_dim_edges=3, output_dim=2,
        processor_size=4, mlp_activation_fn='relu',
        num_layers_node_processor=2, num_layers_edge_processor=2,
        hidden_dim_processor=128, hidden_dim_node_encoder=128,
        hidden_dim_edge_encoder=128, num_layers_edge_encoder=2,
        hidden_dim_node_decoder=128, num_layers_node_decoder=2,
        aggregation='sum', num_harmonics=5)

    model = MeshGraphKANRollout(kan, num_static_feats=2, num_dynamic_feats=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"    MeshGraphKAN loaded (epoch {ckpt.get('epoch', '?')})")
    return model


def load_mgn(ckpt_path, norm_stats, sample_data, device):
    from meshgraphnet import MeshGraphNet

    model = MeshGraphNet(
        input_dim_node=4, input_dim_edge=3,
        hidden_dim=128, output_dim=2, num_layers=4).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    stats_path = ckpt_path.replace('best_model.pt', 'normalization_stats.pt')
    stats = torch.load(stats_path, map_location=device, weights_only=False)
    print(f"    MeshGraphNet loaded (epoch {ckpt.get('epoch', '?')})")
    return model, stats


# ==============================================================================
# ROLLOUT FUNCTIONS
# ==============================================================================

def rollout_gparcv2(model, simulation, num_steps, device, **kw):
    from torch_geometric.data import Data

    for d in simulation:
        d.x = d.x.to(device)
        d.y = d.y.to(device)
        d.edge_index = d.edge_index.to(device)
        if hasattr(d, 'pos') and d.pos is not None:
            d.pos = d.pos.to(device)
        else:
            d.pos = d.x[:, :2]

    deriv = model.derivative_solver
    if hasattr(deriv, 'initialize_weights'):
        deriv.initialize_weights(simulation[0])

    sf, df = model.num_static_feats, model.num_dynamic_feats
    static = simulation[0].x[:, :sf]
    current = simulation[0].x[:, sf:sf + df].clone()
    edge_index = simulation[0].edge_index

    inp = Data(x=torch.cat([static, current], dim=-1),
               edge_index=edge_index, pos=static, y=simulation[0].y)
    for attr in ['elements', 'x_element', 'y_element', 'mesh_id']:
        if hasattr(simulation[0], attr):
            setattr(inp, attr, getattr(simulation[0], attr))

    disps = [current.cpu().numpy()]
    with torch.no_grad():
        for t in range(num_steps):
            inp.x = torch.cat([static, current], dim=-1)
            inp.y = simulation[t].y
            if hasattr(simulation[t], 'x_element'):
                inp.x_element = simulation[t].x_element
            if hasattr(simulation[t], 'y_element'):
                inp.y_element = simulation[t].y_element
            preds = model([inp], dt=1.0, teacher_forcing_ratio=0.0)
            current = preds[0]
            disps.append(current.cpu().numpy())
    return disps


def rollout_gparcv1(model, simulation, num_steps, device, **kw):
    for d in simulation:
        d.x = d.x.to(device)
        d.y = d.y.to(device)
        d.edge_index = d.edge_index.to(device)

    sf, df = model.num_static_feats, model.num_dynamic_feats
    current = simulation[0].x[:, sf:sf + df].clone()
    disps = [current.cpu().numpy()]

    with torch.no_grad():
        for t in range(num_steps):
            data_t = simulation[t].clone()
            data_t.x = data_t.x.clone()
            data_t.x[:, sf:sf + df] = current
            preds = model([data_t])
            current = preds[0]
            disps.append(current.cpu().numpy())
    return disps


def rollout_mgkan(model, simulation, num_steps, device, **kw):
    from torch_geometric.data import Data

    first = simulation[0]
    sf, df = model.num_static_feats, model.num_dynamic_feats
    static = first.x[:, :sf].to(device)
    current = first.x[:, sf:sf + df].clone().to(device)
    edge_index = first.edge_index.to(device)

    disps = [current.cpu().numpy()]
    with torch.no_grad():
        for t in range(num_steps):
            feats = torch.cat([static, current], dim=-1)
            data = Data(x=feats, edge_index=edge_index,
                        pos=static, y=torch.zeros_like(current))
            edge_feat = model.compute_edge_features(data)
            delta = model.model(feats, edge_feat, edge_index)
            current = current + delta
            disps.append(current.cpu().numpy())
    return disps


def rollout_mgn(model, simulation, num_steps, device, **kw):
    from torch_geometric.data import Data

    stats = kw['stats']
    mean_x = stats['mean_vec_x'].to(device)
    std_x = stats['std_vec_x'].to(device)
    mean_edge = stats['mean_vec_edge'].to(device)
    std_edge = stats['std_vec_edge'].to(device)
    mean_y = stats['mean_vec_y'].to(device)
    std_y = stats['std_vec_y'].to(device)

    first = simulation[0]
    static = first.x[:, :2].to(device)
    current = first.x[:, 2:4].clone().to(device)
    edge_index = first.edge_index.to(device)

    disps = [current.cpu().numpy()]
    with torch.no_grad():
        for t in range(num_steps):
            feats = torch.cat([static, current], dim=-1)
            data = Data(x=feats, pos=static, edge_index=edge_index)
            pred_norm = model(data, mean_x, std_x, mean_edge, std_edge)
            delta = pred_norm * std_y + mean_y
            current = current + delta
            disps.append(current.cpu().numpy())
    return disps


# ==============================================================================
# METRICS
# ==============================================================================

def compute_timestep_metrics(gt_disps, pred_disps, max_disp):
    """
    Compute per-timestep RMSE and RRMSE (physical units).

    Args:
        gt_disps: list of [N, 2] ground truth (normalized)
        pred_disps: list of [N, 2] predictions (normalized)
        max_disp: denorm factor

    Returns:
        rmse_t: [T] array of RMSE in mm
        rrmse_t: [T] array of RRMSE (dimensionless)
        ux_rmse_t, uy_rmse_t: per-component RMSE
    """
    T = min(len(gt_disps), len(pred_disps))
    rmse_t = np.zeros(T)
    rrmse_t = np.zeros(T)
    ux_rmse_t = np.zeros(T)
    uy_rmse_t = np.zeros(T)

    for t in range(T):
        gt = gt_disps[t] * max_disp
        pr = pred_disps[t] * max_disp

        if np.any(np.isnan(pr)):
            rmse_t[t] = np.nan
            rrmse_t[t] = np.nan
            ux_rmse_t[t] = np.nan
            uy_rmse_t[t] = np.nan
            continue

        diff = pr - gt
        mse = np.mean(diff ** 2)
        rmse_t[t] = np.sqrt(mse)

        gt_rms = np.sqrt(np.mean(gt ** 2)) + 1e-10
        rrmse_t[t] = rmse_t[t] / gt_rms

        ux_rmse_t[t] = np.sqrt(np.mean(diff[:, 0] ** 2))
        uy_rmse_t[t] = np.sqrt(np.mean(diff[:, 1] ** 2))

    return rmse_t, rrmse_t, ux_rmse_t, uy_rmse_t


def precompute_shape_derivatives(pos, elements):
    """
    Precompute shape function derivatives for a triangular mesh.
    These depend only on the reference geometry and are constant
    across all timesteps.

    Returns:
        dict with dN[0-2]_d[xy] arrays, each [M], plus node indices
    """
    n0, n1, n2 = elements[:, 0], elements[:, 1], elements[:, 2]

    x0, y0 = pos[n0, 0], pos[n0, 1]
    x1, y1 = pos[n1, 0], pos[n1, 1]
    x2, y2 = pos[n2, 0], pos[n2, 1]

    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    inv_det = np.where(np.abs(det) > 1e-15, 1.0 / det, 0.0)

    return {
        'n0': n0, 'n1': n1, 'n2': n2,
        'dN0_dx': (y1 - y2) * inv_det,
        'dN1_dx': (y2 - y0) * inv_det,
        'dN2_dx': (y0 - y1) * inv_det,
        'dN0_dy': (x2 - x1) * inv_det,
        'dN1_dy': (x0 - x2) * inv_det,
        'dN2_dy': (x1 - x0) * inv_det,
        'n_nodes': pos.shape[0],
    }


def compute_strain_fields_numpy(disp, shape_cache):
    """
    Compute strain field from displacement using precomputed shape derivatives.
    Fully vectorized — no Python loops over elements.

    Args:
        disp: [N, 2] displacement (physical units)
        shape_cache: precomputed dict from precompute_shape_derivatives()

    Returns:
        node_strain: dict with keys 'eps_xx', 'eps_yy', 'eps_xy',
                     'von_mises', 'volumetric' — each [N] arrays
    """
    c = shape_cache
    n0, n1, n2 = c['n0'], c['n1'], c['n2']
    n_nodes = c['n_nodes']

    # Element displacement components
    ux0, ux1, ux2 = disp[n0, 0], disp[n1, 0], disp[n2, 0]
    uy0, uy1, uy2 = disp[n0, 1], disp[n1, 1], disp[n2, 1]

    # Displacement gradients: [M]
    dux_dx = c['dN0_dx'] * ux0 + c['dN1_dx'] * ux1 + c['dN2_dx'] * ux2
    dux_dy = c['dN0_dy'] * ux0 + c['dN1_dy'] * ux1 + c['dN2_dy'] * ux2
    duy_dx = c['dN0_dx'] * uy0 + c['dN1_dx'] * uy1 + c['dN2_dx'] * uy2
    duy_dy = c['dN0_dy'] * uy0 + c['dN1_dy'] * uy1 + c['dN2_dy'] * uy2

    # Element strain
    eps_xx_e = dux_dx
    eps_yy_e = duy_dy
    eps_xy_e = 0.5 * (dux_dy + duy_dx)

    # Scatter to nodes
    eps_xx_n = np.zeros(n_nodes)
    eps_yy_n = np.zeros(n_nodes)
    eps_xy_n = np.zeros(n_nodes)
    count = np.zeros(n_nodes)

    for ni, exx, eyy, exy in [(n0, eps_xx_e, eps_yy_e, eps_xy_e),
                                (n1, eps_xx_e, eps_yy_e, eps_xy_e),
                                (n2, eps_xx_e, eps_yy_e, eps_xy_e)]:
        np.add.at(eps_xx_n, ni, exx)
        np.add.at(eps_yy_n, ni, eyy)
        np.add.at(eps_xy_n, ni, exy)
        np.add.at(count, ni, 1)

    mask = count > 0
    eps_xx_n[mask] /= count[mask]
    eps_yy_n[mask] /= count[mask]
    eps_xy_n[mask] /= count[mask]

    volumetric = eps_xx_n + eps_yy_n
    vm_sq = eps_xx_n**2 + eps_yy_n**2 + eps_xx_n * eps_yy_n + 3 * eps_xy_n**2
    von_mises = np.sqrt(np.maximum(vm_sq, 0.0))

    return {
        'eps_xx': eps_xx_n,
        'eps_yy': eps_yy_n,
        'eps_xy': eps_xy_n,
        'von_mises': von_mises,
        'volumetric': volumetric,
    }


def compute_strain_metrics(gt_disps, pred_disps, shape_cache, max_disp,
                           high_strain_percentile=80):
    """
    Compute strain-based and peak displacement metrics over rollout.

    Args:
        gt_disps: list of [N, 2] GT displacement (normalized)
        pred_disps: list of [N, 2] predicted displacement (normalized)
        shape_cache: precomputed shape function derivatives
        max_disp: denorm factor for displacement
        high_strain_percentile: percentile to define "high-strain" nodes

    Returns:
        dict with per-timestep arrays:
            'vm_rmse': von Mises strain RMSE over all nodes
            'vm_rmse_high': von Mises strain RMSE on high-strain nodes only
            'peak_disp_gt': peak ||U|| in GT per timestep
            'peak_disp_pred': peak ||U|| in prediction per timestep
            'strain_energy_gt': total strain energy density (GT)
            'strain_energy_pred': total strain energy density (pred)
            'plastic_zone_csi': CSI for high-strain zone capture
    """
    T = min(len(gt_disps), len(pred_disps))

    vm_rmse_t = np.zeros(T)
    vm_rmse_high_t = np.zeros(T)
    peak_gt_t = np.zeros(T)
    peak_pred_t = np.zeros(T)
    se_gt_t = np.zeros(T)
    se_pred_t = np.zeros(T)
    csi_t = np.zeros(T)

    for t in range(T):
        gt_phys = gt_disps[t] * max_disp
        pr_phys = pred_disps[t] * max_disp

        if np.any(np.isnan(pr_phys)):
            vm_rmse_t[t] = np.nan
            vm_rmse_high_t[t] = np.nan
            peak_gt_t[t] = np.nan
            peak_pred_t[t] = np.nan
            se_gt_t[t] = np.nan
            se_pred_t[t] = np.nan
            csi_t[t] = np.nan
            continue

        # Peak displacement magnitude
        gt_mag = np.sqrt(gt_phys[:, 0]**2 + gt_phys[:, 1]**2)
        pr_mag = np.sqrt(pr_phys[:, 0]**2 + pr_phys[:, 1]**2)
        peak_gt_t[t] = np.max(gt_mag)
        peak_pred_t[t] = np.max(pr_mag)

        # Strain fields (uses precomputed shape derivatives — fast)
        gt_strain = compute_strain_fields_numpy(gt_phys, shape_cache)
        pr_strain = compute_strain_fields_numpy(pr_phys, shape_cache)

        gt_vm = gt_strain['von_mises']
        pr_vm = pr_strain['von_mises']

        # Von Mises RMSE — all nodes
        vm_rmse_t[t] = np.sqrt(np.mean((pr_vm - gt_vm) ** 2))

        # High-strain region: nodes where GT von Mises > percentile threshold
        threshold = np.percentile(gt_vm, high_strain_percentile)
        high_mask = gt_vm > threshold

        if np.sum(high_mask) > 0:
            vm_rmse_high_t[t] = np.sqrt(
                np.mean((pr_vm[high_mask] - gt_vm[high_mask]) ** 2)
            )

            # CSI: does prediction also show high strain where GT does?
            pred_high = pr_vm > threshold
            hits = np.sum(pred_high & high_mask)
            misses = np.sum(~pred_high & high_mask)
            fa = np.sum(pred_high & ~high_mask)
            denom = hits + misses + fa
            csi_t[t] = hits / denom if denom > 0 else np.nan
        else:
            vm_rmse_high_t[t] = 0.0
            csi_t[t] = np.nan

        # Strain energy density: W = 0.5 * (σ_xx*ε_xx + σ_yy*ε_yy + 2*σ_xy*ε_xy)
        # For comparison purposes use W ≈ 0.5 * (ε_xx² + ε_yy² + 2*ε_xy²)
        # (proportional to actual energy, same E and ν for both)
        def strain_energy(s):
            return 0.5 * np.sum(
                s['eps_xx']**2 + s['eps_yy']**2 + 2 * s['eps_xy']**2
            )

        se_gt_t[t] = strain_energy(gt_strain)
        se_pred_t[t] = strain_energy(pr_strain)

    return {
        'vm_rmse': vm_rmse_t,
        'vm_rmse_high': vm_rmse_high_t,
        'peak_disp_gt': peak_gt_t,
        'peak_disp_pred': peak_pred_t,
        'strain_energy_gt': se_gt_t,
        'strain_energy_pred': se_pred_t,
        'plastic_zone_csi': csi_t,
    }


# ==============================================================================
# PLOTTING
# ==============================================================================

MODEL_COLORS = {
    'G-PARCv2': '#1f77b4',
    'G-PARCv1': '#ff7f0e',
    'MeshGraphKAN': '#2ca02c',
    'MeshGraphNet': '#d62728',
}

MODEL_STYLES = {
    'G-PARCv2': '-',
    'G-PARCv1': '--',
    'MeshGraphKAN': '-.',
    'MeshGraphNet': ':',
}


def plot_metric_over_time(all_metrics, metric_key, ylabel, title,
                          output_path, max_disp=None):
    """
    Plot a metric over time for all models.
    Mean line + shaded std band across simulations.

    Args:
        all_metrics: dict[model_name] → list of 1D arrays (one per sim)
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, sim_arrays in all_metrics.items():
        # Stack: [n_sims, T]
        max_t = max(len(a) for a in sim_arrays)
        padded = np.full((len(sim_arrays), max_t), np.nan)
        for i, a in enumerate(sim_arrays):
            padded[i, :len(a)] = a

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        t_axis = np.arange(max_t)

        color = MODEL_COLORS.get(name, 'gray')
        style = MODEL_STYLES.get(name, '-')

        ax.plot(t_axis, mean, style, color=color, linewidth=2, label=name)
        ax.fill_between(t_axis, mean - std, mean + std,
                         alpha=0.15, color=color)

    ax.set_xlabel('Rollout Timestep', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_component_comparison(all_ux_metrics, all_uy_metrics,
                               output_path):
    """Plot Ux and Uy RMSE side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for name in all_ux_metrics:
        color = MODEL_COLORS.get(name, 'gray')
        style = MODEL_STYLES.get(name, '-')

        for ax, metrics, comp in [
            (ax1, all_ux_metrics[name], '$U_x$'),
            (ax2, all_uy_metrics[name], '$U_y$')
        ]:
            max_t = max(len(a) for a in metrics)
            padded = np.full((len(metrics), max_t), np.nan)
            for i, a in enumerate(metrics):
                padded[i, :len(a)] = a
            mean = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)
            t_axis = np.arange(max_t)

            ax.plot(t_axis, mean, style, color=color, linewidth=2, label=name)
            ax.fill_between(t_axis, mean - std, mean + std,
                             alpha=0.15, color=color)

    for ax, comp in [(ax1, '$U_x$'), (ax2, '$U_y$')]:
        ax.set_xlabel('Rollout Timestep', fontsize=12)
        ax.set_ylabel('RMSE (mm)', fontsize=12)
        ax.set_title(f'{comp} Displacement RMSE', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary_table(summary, output_path):
    """Create a summary bar chart of final-timestep RRMSE."""
    names = list(summary.keys())
    rrmses = [summary[n]['rrmse_final_mean'] for n in names]
    colors = [MODEL_COLORS.get(n, 'gray') for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, rrmses, color=colors, edgecolor='black', alpha=0.8)

    for bar, val in zip(bars, rrmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('RRMSE (final timestep)', fontsize=12)
    ax.set_title('Model Comparison — Final RRMSE', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_peak_displacement(all_peak_gt, all_peak_pred, output_path):
    """
    Plot peak displacement magnitude: GT vs each model over time.
    Shows whether models capture localization or smooth it out.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # GT (use first model's GT since it's the same for all)
    gt_name = list(all_peak_gt.keys())[0]
    gt_arrays = all_peak_gt[gt_name]
    max_t = max(len(a) for a in gt_arrays)
    padded = np.full((len(gt_arrays), max_t), np.nan)
    for i, a in enumerate(gt_arrays):
        padded[i, :len(a)] = a
    gt_mean = np.nanmean(padded, axis=0)
    gt_std = np.nanstd(padded, axis=0)
    t_axis = np.arange(max_t)

    ax.plot(t_axis, gt_mean, 'k-', linewidth=2.5, label='Ground Truth')
    ax.fill_between(t_axis, gt_mean - gt_std, gt_mean + gt_std,
                     alpha=0.1, color='black')

    for name, sim_arrays in all_peak_pred.items():
        max_t_m = max(len(a) for a in sim_arrays)
        padded = np.full((len(sim_arrays), max_t_m), np.nan)
        for i, a in enumerate(sim_arrays):
            padded[i, :len(a)] = a
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        t_ax = np.arange(max_t_m)

        color = MODEL_COLORS.get(name, 'gray')
        style = MODEL_STYLES.get(name, '-')
        ax.plot(t_ax, mean, style, color=color, linewidth=2, label=name)
        ax.fill_between(t_ax, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel('Rollout Timestep', fontsize=12)
    ax.set_ylabel('Peak $\\|\\mathbf{U}\\|$ (mm)', fontsize=12)
    ax.set_title('Peak Displacement Magnitude Over Rollout', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_strain_energy(all_se_gt, all_se_pred, output_path):
    """Plot total strain energy: GT vs predictions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    gt_name = list(all_se_gt.keys())[0]
    gt_arrays = all_se_gt[gt_name]
    max_t = max(len(a) for a in gt_arrays)
    padded = np.full((len(gt_arrays), max_t), np.nan)
    for i, a in enumerate(gt_arrays):
        padded[i, :len(a)] = a
    gt_mean = np.nanmean(padded, axis=0)
    t_axis = np.arange(max_t)
    ax.plot(t_axis, gt_mean, 'k-', linewidth=2.5, label='Ground Truth')

    for name, sim_arrays in all_se_pred.items():
        max_t_m = max(len(a) for a in sim_arrays)
        padded = np.full((len(sim_arrays), max_t_m), np.nan)
        for i, a in enumerate(sim_arrays):
            padded[i, :len(a)] = a
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        t_ax = np.arange(max_t_m)

        color = MODEL_COLORS.get(name, 'gray')
        style = MODEL_STYLES.get(name, '-')
        ax.plot(t_ax, mean, style, color=color, linewidth=2, label=name)
        ax.fill_between(t_ax, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel('Rollout Timestep', fontsize=12)
    ax.set_ylabel('Strain Energy (arb. units)', fontsize=12)
    ax.set_title('Total Strain Energy Over Rollout', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_strain_dashboard(all_vm_rmse, all_vm_rmse_high, all_csi, output_path):
    """3-panel strain dashboard: VM RMSE, high-strain RMSE, plastic zone CSI."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    panels = [
        (ax1, all_vm_rmse, 'Von Mises Strain RMSE', 'VM Strain RMSE'),
        (ax2, all_vm_rmse_high, 'High-Strain Region RMSE', 'VM Strain RMSE\n(top 20% nodes)'),
        (ax3, all_csi, 'Plastic Zone CSI', 'CSI'),
    ]

    for ax, data, title, ylabel in panels:
        for name, sim_arrays in data.items():
            if not sim_arrays:
                continue
            max_t = max(len(a) for a in sim_arrays)
            padded = np.full((len(sim_arrays), max_t), np.nan)
            for i, a in enumerate(sim_arrays):
                padded[i, :len(a)] = a
            mean = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)
            t_ax = np.arange(max_t)

            color = MODEL_COLORS.get(name, 'gray')
            style = MODEL_STYLES.get(name, '-')
            ax.plot(t_ax, mean, style, color=color, linewidth=2, label=name)
            ax.fill_between(t_ax, mean - std, mean + std, alpha=0.15,
                             color=color)

        ax.set_xlabel('Rollout Timestep', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ==============================================================================
# MODEL REGISTRY
# ==============================================================================

MODEL_REGISTRY = {
    'gparcv2': {
        'display_name': 'G-PARCv2',
        'load_fn': load_gparcv2,
        'rollout_fn': rollout_gparcv2,
        'ckpt_arg': 'gparcv2_ckpt',
    },
    'gparcv1': {
        'display_name': 'G-PARCv1',
        'load_fn': load_gparcv1,
        'rollout_fn': rollout_gparcv1,
        'ckpt_arg': 'gparcv1_ckpt',
    },
    'mgkan': {
        'display_name': 'MeshGraphKAN',
        'load_fn': load_mgkan,
        'rollout_fn': rollout_mgkan,
        'ckpt_arg': 'mgkan_ckpt',
    },
    'mgn': {
        'display_name': 'MeshGraphNet',
        'load_fn': load_mgn,
        'rollout_fn': rollout_mgn,
        'ckpt_arg': 'mgn_ckpt',
        'extra_kw_key': 'stats',  # rollout needs stats
    },
}


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model comparison for elastoplastic simulations"
    )

    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--norm_stats", required=True)
    parser.add_argument("--output_dir", default="./comparison_results")
    parser.add_argument("--models", nargs='+', default=['gparcv2'],
                        help="Models to compare: gparcv2 gparcv1 mgkan mgn")
    parser.add_argument("--max_sims", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    # Model checkpoint paths
    parser.add_argument("--gparcv2_ckpt", type=str, default=None)
    parser.add_argument("--gparcv2_config", type=str, default=None,
                        help="Path to G-PARCv2 config.json (auto-detect from ckpt dir if not set)")
    parser.add_argument("--gparcv1_ckpt", type=str, default=None)
    parser.add_argument("--gparcv1_config", type=str, default=None,
                        help="Path to G-PARCv1 config.json (auto-detect from ckpt dir if not set)")
    parser.add_argument("--mgkan_ckpt", type=str, default=None)
    parser.add_argument("--mgn_ckpt", type=str, default=None)
    parser.add_argument("--mgn_stats", type=str, default=None,
                        help="Path to MGN normalization_stats.pt")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    norm_stats = load_norm_stats(args.norm_stats)
    max_disp = norm_stats['displacement']['max_displacement']
    print(f"Normalization: {norm_stats['normalization_method']}, max_disp={max_disp:.1f} mm")

    # Load simulations
    sim_data = load_simulations(args.test_dir, max_sims=args.max_sims)

    # Prepare sample data for model init
    sample_sim = sim_data[0][0]
    sample = sample_sim[0]
    sample.pos = sample.x[:, :2].to(device)
    sample.x = sample.x.to(device)
    sample.edge_index = sample.edge_index.to(device)
    if hasattr(sample, 'y'):
        sample.y = sample.y.to(device)

    # Load models
    print(f"\nLoading models: {args.models}")
    loaded_models = {}

    # Auto-detect config.json from checkpoint directory
    def resolve_config(explicit_path, ckpt_path):
        if explicit_path and Path(explicit_path).exists():
            return explicit_path
        if ckpt_path:
            auto = Path(ckpt_path).parent / "config.json"
            if auto.exists():
                return str(auto)
        return None

    for model_key in args.models:
        if model_key not in MODEL_REGISTRY:
            print(f"  ⚠️  Unknown model: {model_key}, skipping")
            continue

        reg = MODEL_REGISTRY[model_key]
        ckpt_path = getattr(args, reg['ckpt_arg'], None)
        if ckpt_path is None or not Path(ckpt_path).exists():
            print(f"  ⚠️  No checkpoint for {model_key}, skipping")
            continue

        print(f"\n  Loading {reg['display_name']}...")

        # Build kwargs for loader
        load_kwargs = {}
        if model_key in ('gparcv1', 'gparcv2'):
            config_arg = f"{model_key}_config"
            config_path = resolve_config(
                getattr(args, config_arg, None), ckpt_path
            )
            load_kwargs['config_path'] = config_path

        result = reg['load_fn'](ckpt_path, norm_stats, sample, device, **load_kwargs)

        # Handle models that return (model, extra_data)
        if isinstance(result, tuple):
            model_obj, extra = result
            loaded_models[model_key] = {
                'model': model_obj,
                'display_name': reg['display_name'],
                'rollout_fn': reg['rollout_fn'],
                'extra_kw': {reg.get('extra_kw_key', 'extra'): extra},
            }
        else:
            loaded_models[model_key] = {
                'model': result,
                'display_name': reg['display_name'],
                'rollout_fn': reg['rollout_fn'],
                'extra_kw': {},
            }

    if not loaded_models:
        print("ERROR: No models loaded.")
        return

    print(f"\n{'=' * 60}")
    print(f"RUNNING ROLLOUTS ({len(sim_data)} simulations, {len(loaded_models)} models)")
    print(f"{'=' * 60}")

    # Storage: model_name → list of metric arrays
    all_rmse = {v['display_name']: [] for v in loaded_models.values()}
    all_rrmse = {v['display_name']: [] for v in loaded_models.values()}
    all_ux_rmse = {v['display_name']: [] for v in loaded_models.values()}
    all_uy_rmse = {v['display_name']: [] for v in loaded_models.values()}
    # Strain / deformation metrics
    all_vm_rmse = {v['display_name']: [] for v in loaded_models.values()}
    all_vm_rmse_high = {v['display_name']: [] for v in loaded_models.values()}
    all_peak_gt = {v['display_name']: [] for v in loaded_models.values()}
    all_peak_pred = {v['display_name']: [] for v in loaded_models.values()}
    all_se_gt = {v['display_name']: [] for v in loaded_models.values()}
    all_se_pred = {v['display_name']: [] for v in loaded_models.values()}
    all_csi = {v['display_name']: [] for v in loaded_models.values()}
    per_sim_results = []

    for sim_idx, (simulation, sim_name) in enumerate(tqdm(sim_data, desc="Simulations")):
        total_steps = len(simulation) - 1

        # Ground truth: cumulative displacements [N, 2] at each step
        gt_disps = [simulation[t].x[:, 2:4].cpu().numpy() for t in range(total_steps + 1)]

        # Mesh data for strain computation
        pos = simulation[0].x[:, :2].cpu().numpy()
        has_elements = hasattr(simulation[0], 'elements') and simulation[0].elements is not None
        elements = simulation[0].elements.cpu().numpy() if has_elements else None

        # Precompute shape function derivatives ONCE per simulation mesh
        # (depends only on reference geometry, reused for all models + timesteps)
        shape_cache = None
        if elements is not None:
            # Denormalize positions for correct gradient computation
            norm_method = norm_stats.get('normalization_method', 'global_max')
            if norm_method == 'global_max':
                max_pos = norm_stats['position'].get('max_position', 200.0)
                pos_phys = pos * max_pos
            else:
                pos_phys = pos
            shape_cache = precompute_shape_derivatives(pos_phys, elements)

        sim_result = {'name': sim_name, 'models': {}}

        for model_key, info in loaded_models.items():
            name = info['display_name']
            try:
                t0 = time.time()
                pred_disps = info['rollout_fn'](
                    info['model'], simulation, total_steps,
                    device, **info['extra_kw']
                )
                elapsed = time.time() - t0

                rmse_t, rrmse_t, ux_t, uy_t = compute_timestep_metrics(
                    gt_disps, pred_disps, max_disp
                )

                all_rmse[name].append(rmse_t)
                all_rrmse[name].append(rrmse_t)
                all_ux_rmse[name].append(ux_t)
                all_uy_rmse[name].append(uy_t)

                # Strain metrics (requires triangulation)
                strain_m = {}
                if shape_cache is not None:
                    strain_m = compute_strain_metrics(
                        gt_disps, pred_disps, shape_cache, max_disp,
                        high_strain_percentile=99,
                    )
                    all_vm_rmse[name].append(strain_m['vm_rmse'])
                    all_vm_rmse_high[name].append(strain_m['vm_rmse_high'])
                    all_peak_gt[name].append(strain_m['peak_disp_gt'])
                    all_peak_pred[name].append(strain_m['peak_disp_pred'])
                    all_se_gt[name].append(strain_m['strain_energy_gt'])
                    all_se_pred[name].append(strain_m['strain_energy_pred'])
                    all_csi[name].append(strain_m['plastic_zone_csi'])

                # Final timestep metrics
                final_rmse = rmse_t[-1] if not np.isnan(rmse_t[-1]) else np.nan
                final_rrmse = rrmse_t[-1] if not np.isnan(rrmse_t[-1]) else np.nan

                sim_entry = {
                    'rmse_final': float(final_rmse),
                    'rrmse_final': float(final_rrmse),
                    'time_s': float(elapsed),
                    'diverged': bool(np.any(np.isnan(rmse_t))),
                }
                if strain_m:
                    vm_final = strain_m['vm_rmse'][-1]
                    vm_high_final = strain_m['vm_rmse_high'][-1]
                    csi_final = strain_m['plastic_zone_csi'][-1]
                    sim_entry['vm_rmse_final'] = float(vm_final) if not np.isnan(vm_final) else None
                    sim_entry['vm_rmse_high_final'] = float(vm_high_final) if not np.isnan(vm_high_final) else None
                    sim_entry['plastic_zone_csi_final'] = float(csi_final) if not np.isnan(csi_final) else None

                sim_result['models'][name] = sim_entry

            except Exception as e:
                print(f"  Error {name} on {sim_name}: {e}")
                import traceback; traceback.print_exc()

        per_sim_results.append(sim_result)

    # ==============================================================
    # AGGREGATE + PRINT
    # ==============================================================
    print(f"\n{'=' * 60}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 60}")

    summary = {}
    for name in all_rmse:
        if not all_rmse[name]:
            continue

        final_rmses = [a[-1] for a in all_rmse[name] if not np.isnan(a[-1])]
        final_rrmses = [a[-1] for a in all_rrmse[name] if not np.isnan(a[-1])]
        n_diverged = sum(1 for a in all_rmse[name] if np.any(np.isnan(a)))

        summary[name] = {
            'rmse_final_mean': float(np.mean(final_rmses)) if final_rmses else np.nan,
            'rmse_final_std': float(np.std(final_rmses)) if final_rmses else np.nan,
            'rrmse_final_mean': float(np.mean(final_rrmses)) if final_rrmses else np.nan,
            'rrmse_final_std': float(np.std(final_rrmses)) if final_rrmses else np.nan,
            'n_valid': len(final_rmses),
            'n_diverged': n_diverged,
        }

        # Strain aggregate metrics
        if all_vm_rmse.get(name):
            vm_finals = [a[-1] for a in all_vm_rmse[name] if not np.isnan(a[-1])]
            vm_high_finals = [a[-1] for a in all_vm_rmse_high[name] if not np.isnan(a[-1])]
            csi_finals = [a[-1] for a in all_csi[name] if not np.isnan(a[-1])]

            if vm_finals:
                summary[name]['vm_rmse_final_mean'] = float(np.mean(vm_finals))
                summary[name]['vm_rmse_final_std'] = float(np.std(vm_finals))
            if vm_high_finals:
                summary[name]['vm_rmse_high_final_mean'] = float(np.mean(vm_high_finals))
                summary[name]['vm_rmse_high_final_std'] = float(np.std(vm_high_finals))
            if csi_finals:
                summary[name]['plastic_zone_csi_mean'] = float(np.mean(csi_finals))
                summary[name]['plastic_zone_csi_std'] = float(np.std(csi_finals))

        s = summary[name]
        print(
            f"\n  {name}:"
            f"\n    RMSE (final):  {s['rmse_final_mean']:.4f} ± {s['rmse_final_std']:.4f} mm"
            f"\n    RRMSE (final): {s['rrmse_final_mean']:.4f} ± {s['rrmse_final_std']:.4f}"
            f"\n    Valid: {s['n_valid']}, Diverged: {s['n_diverged']}"
        )
        if 'vm_rmse_final_mean' in s:
            print(
                f"    VM Strain RMSE (final):      {s['vm_rmse_final_mean']:.6f} ± {s['vm_rmse_final_std']:.6f}"
            )
        if 'vm_rmse_high_final_mean' in s:
            print(
                f"    VM Strain RMSE High (final): {s['vm_rmse_high_final_mean']:.6f} ± {s['vm_rmse_high_final_std']:.6f}"
            )
        if 'plastic_zone_csi_mean' in s:
            print(
                f"    Plastic Zone CSI (final):    {s['plastic_zone_csi_mean']:.4f} ± {s['plastic_zone_csi_std']:.4f}"
            )

    # ==============================================================
    # PLOTS
    # ==============================================================
    print(f"\nGenerating plots...")

    plot_metric_over_time(
        all_rmse, 'RMSE', 'RMSE (mm)',
        'Displacement RMSE Over Rollout',
        output_dir / 'rmse_over_time.png', max_disp,
    )

    plot_metric_over_time(
        all_rrmse, 'RRMSE', 'RRMSE',
        'Relative RMSE Over Rollout',
        output_dir / 'rrmse_over_time.png', max_disp,
    )

    plot_component_comparison(
        all_ux_rmse, all_uy_rmse,
        output_dir / 'component_rmse.png',
    )

    if len(summary) > 1:
        plot_summary_table(summary, output_dir / 'model_comparison_bar.png')

    # Strain / deformation plots
    has_strain = any(len(v) > 0 for v in all_vm_rmse.values())
    if has_strain:
        plot_strain_dashboard(
            all_vm_rmse, all_vm_rmse_high, all_csi,
            output_dir / 'strain_dashboard.png',
        )
        plot_peak_displacement(
            all_peak_gt, all_peak_pred,
            output_dir / 'peak_displacement.png',
        )
        plot_strain_energy(
            all_se_gt, all_se_pred,
            output_dir / 'strain_energy.png',
        )
        plot_metric_over_time(
            all_vm_rmse, 'VM_RMSE', 'Von Mises Strain RMSE',
            'Von Mises Strain RMSE Over Rollout',
            output_dir / 'vm_strain_rmse_over_time.png',
        )

    # ==============================================================
    # SAVE JSON
    # ==============================================================
    output_data = {
        'summary': summary,
        'per_simulation': per_sim_results,
        'config': {
            'models': list(loaded_models.keys()),
            'n_simulations': len(sim_data),
            'max_disp_mm': max_disp,
        },
    }
    json_path = output_dir / 'comparison_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=lambda x:
                  float(x) if isinstance(x, (np.floating, np.integer)) else
                  None if (isinstance(x, float) and np.isnan(x)) else str(x))
    print(f"  Saved: {json_path}")

    print(f"\n{'=' * 60}")
    print(f"Comparison complete! Results in: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()