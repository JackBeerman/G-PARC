#!/usr/bin/env python3
"""
Publication Figure: Model Comparison — Displacement Magnitude on Reference Mesh
================================================================================

Layout:
    Row 0: Ground Truth
    Row 1: G-PARC (ours)
    Row 2: MeshGraphNet
    Row 3: MeshGraphKAN

    Columns: 4 evenly spaced rollout timesteps

Each cell shows displacement magnitude ||U|| = sqrt(Ux^2 + Uy^2) on the
undeformed (reference) mesh using tripcolor with gouraud shading.
Per-row colorbar, color range from GT global max across ALL timesteps.

Usage:
    python model_comparison_figure.py \
        --sim_index 0 \
        --output_dir ./paper_figures
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
import matplotlib.tri as tri
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from pathlib import Path
from tqdm import tqdm
import warnings

# ============================================================
# CONFIGURATION — UPDATE THESE PATHS
# ============================================================

# Test data
TEST_DIR = "/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/train"
NORM_STATS_PATH = "/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/normalization_stats.json"

# ============================================================
# Add project root to path
# ============================================================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

warnings.filterwarnings("ignore", category=UserWarning)

def load_norm_stats(path):
    """Load normalization statistics."""
    with open(path) as f:
        return json.load(f)


def load_simulation(test_dir, sim_index):
    """Load a single test simulation."""
    sim_files = sorted(Path(test_dir).glob("simulation_*.pt"))
    if sim_index >= len(sim_files):
        raise ValueError(f"sim_index {sim_index} >= {len(sim_files)} available")
    sim_file = sim_files[sim_index]
    print(f"Loading: {sim_file.name}")
    simulation = torch.load(sim_file, weights_only=False)
    return simulation, sim_file.stem


def get_mesh_data(simulation):
    """Extract reference positions and elements from simulation."""
    first = simulation[0]
    pos = first.x[:, :2].cpu().numpy()
    elements = first.elements.cpu().numpy()
    edge_index = first.edge_index
    return pos, elements, edge_index


def get_ground_truth_displacements(simulation, timesteps):
    """Get ground truth displacement at specified timesteps."""
    displacements = []
    for t in timesteps:
        u = simulation[t].x[:, 2:4].cpu().numpy()
        mag = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
        displacements.append(mag)
    return displacements


def get_erosion_mask(simulation, timestep):
    """Get eroded element mask at a given timestep."""
    data = simulation[timestep]
    if hasattr(data, 'x_element') and data.x_element is not None:
        erosion = data.x_element.squeeze().cpu().numpy()
        return erosion < 0.5
    return None


# ============================================================
# MODEL LOADING
# ============================================================

def load_gparcv2(checkpoint_path, norm_stats, sample_data, device):
    """Load G-PARCv2 model."""
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from differentiator.differentiator import ElastoPlasticDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from models.globalelasto import GPARC_ElastoPlastic_Numerical

    pos_stats = norm_stats['position']
    pos_mean = [pos_stats['x_pos']['mean'], pos_stats['y_pos']['mean']]
    pos_std = [pos_stats['x_pos']['std'], pos_stats['y_pos']['std']]
    norm_method = norm_stats.get('normalization_method', 'global_max')
    max_position = pos_stats.get('max_position', 200.0)

    ckpt_dir = Path(checkpoint_path).parent
    cfg = {}
    if (ckpt_dir / "config.json").exists():
        with open(ckpt_dir / "config.json") as f:
            cfg = json.load(f)
        print(f"  Loaded v2 config from {ckpt_dir / 'config.json'}")

    sf = cfg.get('num_static_feats', 2)
    df = cfg.get('num_dynamic_feats', 2)
    foc = cfg.get('feature_out_channels', 128)

    gradient_solver = SolveGradientsLST(
        pos_mean=pos_mean, pos_std=pos_std,
        norm_method=norm_method, max_position=max_position
    )
    laplacian_solver = SolveWeightLST2d(
        pos_mean=pos_mean, pos_std=pos_std,
        norm_method=norm_method, max_position=max_position,
        min_neighbors=5, use_2hop_extension=False
    )

    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=sf,
        hidden_channels=cfg.get('hidden_channels', 128),
        out_channels=foc,
        num_layers=cfg.get('num_layers', 4),
        dropout=cfg.get('dropout', 0.0),
        use_layer_norm=cfg.get('use_layer_norm', True),
        use_relative_pos=cfg.get('use_relative_pos', True),
    )

    derivative_solver = ElastoPlasticDifferentiator(
        num_static_feats=sf, num_dynamic_feats=df,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
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
        zero_init=cfg.get('zero_init', True),
    )
    derivative_solver.initialize_weights(sample_data)

    model = GPARC_ElastoPlastic_Numerical(
        derivative_solver_physics=derivative_solver,
        integrator_type=cfg.get('integrator', 'euler'),
        num_static_feats=sf, num_dynamic_feats=df,
        pos_mean=pos_mean, pos_std=pos_std,
        boundary_threshold=cfg.get('boundary_threshold', 0.5),
        clamp_output=cfg.get('clamp_output', False),
        norm_method=norm_method, max_position=max_position,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  G-PARCv2 loaded (epoch {ckpt.get('epoch', '?')})")
    return model


def load_gparcv1(checkpoint_path, norm_stats, sample_data, device):
    """Load G-PARCv1 model."""
    from utilities.featureextractor import FeatureExtractorGNN
    from differentiator.differentiator import DerivativeGNN
    from integrator.integrator import IntegralGNN
    from models.parcv1_elasto import GPARC

    ckpt_dir = Path(checkpoint_path).parent
    cfg = {}
    if (ckpt_dir / "config.json").exists():
        with open(ckpt_dir / "config.json") as f:
            cfg = json.load(f)
        print(f"  Loaded v1 config from {ckpt_dir / 'config.json'}")

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

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  G-PARCv1 loaded (epoch {ckpt.get('epoch', '?')})")
    return model


def load_meshgraphnet(checkpoint_path, device):
    """Load MeshGraphNet model."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MeshGraphNet'))
    from meshgraphnet import MeshGraphNet

    model = MeshGraphNet(
        input_dim_node=4, input_dim_edge=3,
        hidden_dim=128, output_dim=2, num_layers=4
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  MeshGraphNet loaded (epoch {ckpt['epoch']})")
    return model


def load_meshgraphkan(checkpoint_path, device):
    """Load MeshGraphKAN model."""
    from models.meshgraphkan import MeshGraphKAN, MeshGraphKANElastoRollout

    kan_model = MeshGraphKAN(
        input_dim_nodes=4, input_dim_edges=3,
        output_dim=2, processor_size=4,
        mlp_activation_fn='relu',
        num_layers_node_processor=2, num_layers_edge_processor=2,
        hidden_dim_processor=128, hidden_dim_node_encoder=128,
        hidden_dim_edge_encoder=128, num_layers_edge_encoder=2,
        hidden_dim_node_decoder=128, num_layers_node_decoder=2,
        aggregation='sum', num_harmonics=5,
    )

    model = MeshGraphKANElastoRollout(
        kan_model, num_static_feats=2, num_dynamic_feats=2
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  MeshGraphKAN loaded (epoch {ckpt['epoch']})")
    return model


def load_graphsage(checkpoint_path, device):
    """Load GraphSAGE elastoplastic model."""
    from models.graphsage import load_model as load_gsage
    model = load_gsage('elasto', checkpoint_path, device=device)
    return model


# ============================================================
# ROLLOUT
# ============================================================

def rollout_gparcv2(model, simulation, num_steps, device):
    """Run G-PARCv2 rollout, return cumulative displacements at each step."""
    for data in simulation:
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.edge_index = data.edge_index.to(device)
        if hasattr(data, 'pos') and data.pos is not None:
            data.pos = data.pos.to(device)
        else:
            data.pos = data.x[:, :2]

    deriv = model.derivative_solver
    if hasattr(deriv, 'initialize_weights'):
        deriv.initialize_weights(simulation[0])

    sf, df = model.num_static_feats, model.num_dynamic_feats
    static = simulation[0].x[:, :sf]
    current_dynamic = simulation[0].x[:, sf:sf + df].clone()
    edge_index = simulation[0].edge_index

    from torch_geometric.data import Data
    input_data = Data(
        x=torch.cat([static, current_dynamic], dim=-1),
        edge_index=edge_index,
        pos=static,
        y=simulation[0].y,
    )
    for attr in ['elements', 'x_element', 'y_element', 'mesh_id']:
        if hasattr(simulation[0], attr):
            setattr(input_data, attr, getattr(simulation[0], attr))

    cumulative_disps = [current_dynamic.cpu().numpy()]

    with torch.no_grad():
        for t in range(num_steps):
            input_data.x = torch.cat([static, current_dynamic], dim=-1)
            input_data.y = simulation[t].y
            if hasattr(simulation[t], 'x_element'):
                input_data.x_element = simulation[t].x_element
            if hasattr(simulation[t], 'y_element'):
                input_data.y_element = simulation[t].y_element

            seq = [input_data]
            preds = model(seq, dt=1.0, teacher_forcing_ratio=0.0)
            current_dynamic = preds[0]
            cumulative_disps.append(current_dynamic.cpu().numpy())

    return cumulative_disps


def rollout_gparcv1(model, simulation, num_steps, device):
    """Run G-PARCv1 rollout."""
    for data in simulation:
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.edge_index = data.edge_index.to(device)

    sf, df = model.num_static_feats, model.num_dynamic_feats
    current_dynamic = simulation[0].x[:, sf:sf + df].clone()
    cumulative_disps = [current_dynamic.cpu().numpy()]

    with torch.no_grad():
        for t in range(num_steps):
            data_t = simulation[t].clone()
            data_t.x = data_t.x.clone()
            data_t.x[:, sf:sf + df] = current_dynamic
            preds = model([data_t])
            current_dynamic = preds[0]
            cumulative_disps.append(current_dynamic.cpu().numpy())

    return cumulative_disps


def rollout_mgn(model, simulation, num_steps, device):
    """Run MeshGraphNet rollout."""
    first = simulation[0]
    static = first.x[:, :2].to(device)
    current_dynamic = first.x[:, 2:4].clone().to(device)
    edge_index = first.edge_index.to(device)

    edge_feat = model.compute_edge_features(static, edge_index)

    cumulative_disps = [current_dynamic.cpu().numpy()]

    t0 = time.time()
    with torch.no_grad():
        for t in range(num_steps):
            if t % 5 == 0:
                print(f"    MGN step {t}/{num_steps} ({time.time()-t0:.1f}s)")
            node_feats = torch.cat([static, current_dynamic], dim=-1)
            delta = model(node_feats, edge_feat, edge_index)
            current_dynamic = current_dynamic + delta
            cumulative_disps.append(current_dynamic.cpu().numpy())

    print(f"    MGN done ({time.time()-t0:.1f}s)")
    return cumulative_disps


def rollout_mgkan(model, simulation, num_steps, device):
    """Run MeshGraphKAN rollout."""
    first = simulation[0]
    sf, df = model.num_static_feats, model.num_dynamic_feats
    static = first.x[:, :sf].to(device)
    current_dynamic = first.x[:, sf:sf + df].clone().to(device)
    edge_index = first.edge_index.to(device)

    edge_feat = model.compute_edge_features(static, edge_index)

    cumulative_disps = [current_dynamic.cpu().numpy()]

    t0 = time.time()
    with torch.no_grad():
        for t in range(num_steps):
            if t % 5 == 0:
                print(f"    MGKAN step {t}/{num_steps} ({time.time()-t0:.1f}s)")
            node_feats = torch.cat([static, current_dynamic], dim=-1)
            delta = model.model(node_feats, edge_feat, edge_index)
            current_dynamic = current_dynamic + delta
            cumulative_disps.append(current_dynamic.cpu().numpy())

    print(f"    MGKAN done ({time.time()-t0:.1f}s)")
    return cumulative_disps


def rollout_graphsage(model, simulation, num_steps, device):
    """Run GraphSAGE rollout."""
    from models.graphsage import compute_edge_attr

    first = simulation[0]
    sf = model.num_static_feats
    df = model.num_dynamic_feats
    static = first.x[:, :sf].to(device)
    current_dynamic = first.x[:, sf:sf + df].clone().to(device)
    edge_index = first.edge_index.to(device)

    pos = first.pos.to(device) if hasattr(first, 'pos') and first.pos is not None else static
    edge_feat = compute_edge_attr(pos, edge_index)

    cumulative_disps = [current_dynamic.cpu().numpy()]

    t0 = time.time()
    with torch.no_grad():
        for t in range(num_steps):
            if t % 5 == 0:
                print(f"    GraphSAGE step {t}/{num_steps} ({time.time()-t0:.1f}s)")
            node_feats = torch.cat([static, current_dynamic], dim=-1)
            delta = model(node_feats, edge_index, edge_attr=edge_feat)
            current_dynamic = current_dynamic + delta
            cumulative_disps.append(current_dynamic.cpu().numpy())

    print(f"    GraphSAGE done ({time.time()-t0:.1f}s)")
    return cumulative_disps


# ============================================================
# DENORMALIZATION
# ============================================================

def denormalize_displacement(u_norm, norm_stats):
    """Convert normalized displacement back to physical units (mm)."""
    method = norm_stats.get('normalization_method', 'global_max')
    if method == 'global_max':
        max_disp = norm_stats['displacement']['max_displacement']
        return u_norm * max_disp
    else:
        raise ValueError(f"Unsupported norm method: {method}")


# ============================================================
# MESH RENDERING
# ============================================================

def render_mesh_poly(ax, pos, elements, node_values, cmap, norm, erosion_mask=None):
    """Render mesh using PolyCollection (element-averaged coloring)."""
    if erosion_mask is not None and erosion_mask.any():
        valid_mask = ~erosion_mask
    else:
        valid_mask = np.ones(len(elements), dtype=bool)

    valid_elements = elements[valid_mask]

    if len(valid_elements) == 0:
        return

    has_nan = np.any(np.isnan(node_values))

    polygons = pos[valid_elements]
    elem_values = node_values[valid_elements].mean(axis=1)

    if has_nan:
        nan_elems = np.isnan(elem_values)
        valid_elems = ~nan_elems

        if valid_elems.any():
            cmap_obj = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
            colors = cmap_obj(norm(elem_values[valid_elems]))
            pc = PolyCollection(polygons[valid_elems], facecolors=colors,
                                edgecolors=(0, 0, 0, 0.08), linewidths=0.15)
            ax.add_collection(pc)

        if nan_elems.any():
            pc_nan = PolyCollection(polygons[nan_elems], facecolors='lightgray',
                                    edgecolors=(0, 0, 0, 0.08), linewidths=0.15, alpha=0.5)
            ax.add_collection(pc_nan)
    else:
        cmap_obj = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
        colors = cmap_obj(norm(elem_values))
        pc = PolyCollection(polygons, facecolors=colors,
                            edgecolors=(0, 0, 0, 0.08), linewidths=0.15)
        ax.add_collection(pc)

    if erosion_mask is not None and erosion_mask.any():
        eroded_elements = elements[erosion_mask]
        eroded_polygons = pos[eroded_elements]
        pc_eroded = PolyCollection(eroded_polygons, facecolors='white',
                                   edgecolors='lightgray', linewidths=0.1, alpha=0.3)
        ax.add_collection(pc_eroded)


# ============================================================
# FIGURE CREATION — per-row colorbars, GT global max
# ============================================================

def create_comparison_figure(
    pos, elements, gt_disps, model_disps, model_names,
    timesteps, sim_name, norm_stats, output_dir,
    erosion_masks=None, dpi=300,
    deformed=False, gt_disp_vectors=None, model_disp_vectors=None,
):
    """
    Create publication comparison figure with per-row colorbars.
    Color range from GT global max across ALL timesteps.
    
    If deformed=True, renders on deformed mesh (pos + displacement).
    gt_disp_vectors / model_disp_vectors: dict of [N,2] arrays at each timestep.
    """
    n_rows = 1 + len(model_names)
    n_cols = len(timesteps)
    row_labels = ['Ground Truth'] + model_names

    # Denormalize positions
    method = norm_stats.get('normalization_method', 'global_max')
    if method == 'global_max':
        max_pos = norm_stats['position']['max_position']
        pos_phys = pos * max_pos
    else:
        pos_phys = pos

    # Color range from GT global max across ALL shown timesteps
    vmin = 0
    vmax = float(np.percentile(np.concatenate([m.ravel() for m in gt_disps]), 99))
    color_norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.cm.jet
    print(f"  GT color vmax = {vmax:.4f}")

    # Compute deformed bounds if needed
    if deformed and gt_disp_vectors is not None:
        all_deformed_pos = []
        for col_idx in range(n_cols):
            all_deformed_pos.append(pos_phys + gt_disp_vectors[col_idx])
            for name in model_names:
                if name in model_disp_vectors:
                    all_deformed_pos.append(pos_phys + model_disp_vectors[name][col_idx])
        all_pos_stack = np.concatenate(all_deformed_pos, axis=0)
        xlim = (all_pos_stack[:, 0].min() - 2, all_pos_stack[:, 0].max() + 2)
        ylim = (all_pos_stack[:, 1].min() - 2, all_pos_stack[:, 1].max() + 2)
    else:
        xlim = (pos_phys[:, 0].min() - 1, pos_phys[:, 0].max() + 1)
        ylim = (pos_phys[:, 1].min() - 1, pos_phys[:, 1].max() + 1)

    # Layout with manual axes placement for per-row colorbars
    cell_w, cell_h = 4.0, 3.2
    row_label_w = 1.2
    cbar_w = 0.5
    header_h = 0.6

    fig_w = row_label_w + n_cols * cell_w + cbar_w + 0.3
    fig_h = header_h + n_rows * cell_h + 0.3

    config_label = 'Deformed' if deformed else 'Reference'
    suffix = '_deformed' if deformed else ''

    for fmt in ['png', 'pdf']:
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        x0 = row_label_w / fig_w
        y_top = 1.0 - header_h / fig_h
        cw = cell_w / fig_w
        ch = cell_h / fig_h
        cbar_norm_w = 0.012
        cbar_gap = 0.008

        for row in range(n_rows):
            for col_idx, t in enumerate(timesteps):
                emask = erosion_masks.get(t, None) if erosion_masks else None

                x = x0 + col_idx * cw
                y = y_top - (row + 1) * ch

                ax = fig.add_axes([x, y, cw * 0.95, ch * 0.90])

                if row == 0:
                    values = gt_disps[col_idx]
                    disp_vec = gt_disp_vectors[col_idx] if gt_disp_vectors else None
                else:
                    name = model_names[row - 1]
                    values = model_disps[name][col_idx]
                    disp_vec = model_disp_vectors[name][col_idx] if model_disp_vectors and name in model_disp_vectors else None

                # Choose reference or deformed mesh positions
                if deformed and disp_vec is not None:
                    render_pos = pos_phys + disp_vec
                else:
                    render_pos = pos_phys

                render_mesh_poly(ax, render_pos, elements, values, cmap, color_norm, emask)

                ax.set_aspect('equal')
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                if row == 0:
                    ax.set_title(f't = {t}', fontsize=11, fontweight='bold', pad=6)

                if col_idx == 0:
                    ax.set_ylabel(row_labels[row], fontsize=10, fontweight='bold',
                                  rotation=90, labelpad=8)

            # Per-row colorbar
            cb_x = x0 + n_cols * cw + cbar_gap
            cb_y = y_top - (row + 1) * ch + ch * 0.05
            cb_h = ch * 0.80

            cbar_ax = fig.add_axes([cb_x, cb_y, cbar_norm_w, cb_h])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=6)

            # Label on middle row only
            if row == n_rows // 2:
                cbar.set_label('||U|| (mm)', fontsize=8, labelpad=3)

        # Title
        fig.text(0.5, 0.97,
                 f'Displacement Magnitude ({config_label} Config) \u2014 {sim_name}',
                 fontsize=12, fontweight='bold', ha='center', va='top')

        out_path = Path(output_dir) / f'model_comparison{suffix}_{sim_name}.{fmt}'
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Model comparison figure for paper")
    parser.add_argument("--sim_index", type=int, default=0, help="Test simulation index")
    parser.add_argument("--output_dir", type=str, default="./paper_figures")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_timesteps", type=int, default=4,
                        help="Number of evenly spaced timesteps to show")
    parser.add_argument("--models", type=str, nargs='+',
                        default=['gparcv2'],
                        help="Models to include: gparcv2 gparcv1 mgn mgkan graphsage")

    parser.add_argument("--test_dir", type=str, default=TEST_DIR)
    parser.add_argument("--norm_stats", type=str, default=NORM_STATS_PATH)
    parser.add_argument("--gparcv2_ckpt", type=str, default=None)
    parser.add_argument("--gparcv1_ckpt", type=str, default=None)
    parser.add_argument("--mgn_ckpt", type=str, default=None)
    parser.add_argument("--mgkan_ckpt", type=str, default=None)
    parser.add_argument("--graphsage_ckpt", type=str, default=None)
    parser.add_argument("--deformed", action='store_true',
                        help="Show deformed configuration (mesh moves with displacement)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load normalization stats
    norm_stats = load_norm_stats(args.norm_stats)
    max_disp = norm_stats['displacement']['max_displacement']
    print(f"Normalization: {norm_stats['normalization_method']}, max_disp={max_disp:.1f} mm")

    # Load simulation
    simulation, sim_name = load_simulation(args.test_dir, args.sim_index)
    pos, elements, edge_index = get_mesh_data(simulation)
    total_steps = len(simulation) - 1
    print(f"Simulation: {sim_name}, nodes={pos.shape[0]}, steps={total_steps}")

    # Pick evenly spaced timesteps
    timesteps = np.linspace(0, total_steps - 1, args.num_timesteps, dtype=int).tolist()
    print(f"Timesteps: {timesteps}")

    # Ground truth: displacement magnitude in physical units
    gt_disps_norm = []
    gt_disp_vectors = []
    erosion_masks = {}
    for t in timesteps:
        u = simulation[t].x[:, 2:4].cpu().numpy()
        u_phys = u * max_disp
        mag = np.sqrt(u_phys[:, 0]**2 + u_phys[:, 1]**2)
        gt_disps_norm.append(mag)
        gt_disp_vectors.append(u_phys)

        emask = get_erosion_mask(simulation, t)
        if emask is not None:
            erosion_masks[t] = emask

    # Load models and run rollouts
    sample_data = simulation[0]
    sample_data.pos = sample_data.x[:, :2]
    rollout_steps = total_steps

    model_names = []
    all_disps = {}

    selected = [m.lower() for m in args.models]
    print(f"\nSelected models: {selected}")

    if 'gparcv2' in selected:
        print("\n  Loading G-PARCv2...")
        gparcv2_model = load_gparcv2(args.gparcv2_ckpt, norm_stats, sample_data, device)
        print("  G-PARCv2 rollout...")
        gparcv2_disps = rollout_gparcv2(gparcv2_model, simulation, rollout_steps, device)
        model_names.append('G-PARC with MLS')
        all_disps['G-PARC with MLS'] = gparcv2_disps

    if 'gparcv1' in selected:
        print("\n  Loading G-PARCv1...")
        gparcv1_model = load_gparcv1(args.gparcv1_ckpt, norm_stats, sample_data, device)
        print("  G-PARCv1 rollout...")
        gparcv1_disps = rollout_gparcv1(gparcv1_model, simulation, rollout_steps, device)
        model_names.append('G-PARC Baseline')
        all_disps['G-PARC Baseline'] = gparcv1_disps

    if 'mgn' in selected:
        print("\n  Loading MeshGraphNet...")
        mgn_model = load_meshgraphnet(args.mgn_ckpt, device)
        print("  MeshGraphNet rollout...")
        mgn_disps = rollout_mgn(mgn_model, simulation, rollout_steps, device)
        model_names.append('MeshGraphNet')
        all_disps['MeshGraphNet'] = mgn_disps

    if 'mgkan' in selected:
        print("\n  Loading MeshGraphKAN...")
        mgkan_model = load_meshgraphkan(args.mgkan_ckpt, device)
        print("  MeshGraphKAN rollout...")
        mgkan_disps = rollout_mgkan(mgkan_model, simulation, rollout_steps, device)
        model_names.append('MeshGraphKAN')
        all_disps['MeshGraphKAN'] = mgkan_disps

    if 'graphsage' in selected:
        print("\n  Loading GraphSAGE...")
        gsage_model = load_graphsage(args.graphsage_ckpt, device)
        print("  GraphSAGE rollout...")
        gsage_disps = rollout_graphsage(gsage_model, simulation, rollout_steps, device)
        model_names.append('GraphSAGE')
        all_disps['GraphSAGE'] = gsage_disps

    if len(model_names) == 0:
        print("ERROR: No valid models selected. Use --models gparcv2 gparcv1 mgn mgkan graphsage")
        return

    # Extract displacement magnitudes and vectors at selected timesteps (physical units)
    model_disps = {}
    model_disp_vectors = {}
    for name in model_names:
        disps = all_disps[name]
        mags = []
        vecs = []
        for t in timesteps:
            u = disps[t]
            u_phys = u * max_disp
            mag = np.sqrt(u_phys[:, 0]**2 + u_phys[:, 1]**2)
            if np.any(np.isnan(mag)):
                nan_pct = 100 * np.sum(np.isnan(mag)) / len(mag)
                print(f"  Warning: {name} has {nan_pct:.1f}% NaN nodes at t={t}")
            mags.append(mag)
            vecs.append(u_phys)
        model_disps[name] = mags
        model_disp_vectors[name] = vecs

    # Create figure (reference configuration)
    print("\nCreating comparison figure (reference config)...")
    create_comparison_figure(
        pos=pos, elements=elements,
        gt_disps=gt_disps_norm,
        model_disps=model_disps,
        model_names=model_names,
        timesteps=timesteps,
        sim_name=sim_name,
        norm_stats=norm_stats,
        output_dir=output_dir,
        erosion_masks=erosion_masks,
        dpi=args.dpi,
        deformed=False,
    )

    # Create deformed configuration figure if requested
    if args.deformed:
        print("\nCreating comparison figure (deformed config)...")
        create_comparison_figure(
            pos=pos, elements=elements,
            gt_disps=gt_disps_norm,
            model_disps=model_disps,
            model_names=model_names,
            timesteps=timesteps,
            sim_name=sim_name,
            norm_stats=norm_stats,
            output_dir=output_dir,
            erosion_masks=erosion_masks,
            dpi=args.dpi,
            deformed=True,
            gt_disp_vectors=gt_disp_vectors,
            model_disp_vectors=model_disp_vectors,
        )

    # Print RRMSE summary
    print("\n" + "="*60)
    print("PER-SIMULATION ROLLOUT ERROR (last timestep)")
    print("="*60)
    t_final = timesteps[-1]
    gt_u = simulation[t_final].x[:, 2:4].cpu().numpy() * max_disp
    gt_norm_val = np.sqrt(np.mean(gt_u**2))

    for name in model_names:
        disps = all_disps[name]
        pred_u = disps[t_final] * max_disp
        rmse = np.sqrt(np.nanmean((pred_u - gt_u)**2))
        rrmse = rmse / (gt_norm_val + 1e-10)
        print(f"  {name:20s}: RMSE={rmse:.4f} mm, RRMSE={rrmse:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()