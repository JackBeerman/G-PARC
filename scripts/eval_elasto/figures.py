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

Each cell shows displacement magnitude ||U|| = sqrt(Ux² + Uy²) on the
undeformed (reference) mesh using tripcolor with gouraud shading.
Shared colorbar per row, consistent across all panels.

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

# Model checkpoints
GPARC_CHECKPOINT = "/scratch/jtb3sud/elasto_graphconv_V2/global_max_v3_lapdamp/best_model.pth"
MGN_CHECKPOINT = "/scratch/jtb3sud/meshgraphnet_baseline/best_model.pt"
MGKAN_CHECKPOINT = "/scratch/jtb3sud/elasto_meshgraphkan/run1/best_model.pth"

# MGN normalization stats (computed during MGN training)
MGN_STATS_PATH = "/scratch/jtb3sud/meshgraphnet_baseline/normalization_stats.pt"

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
    pos = first.x[:, :2].cpu().numpy()        # reference positions
    elements = first.elements.cpu().numpy()     # triangulation
    edge_index = first.edge_index
    return pos, elements, edge_index


def get_ground_truth_displacements(simulation, timesteps):
    """
    Get ground truth displacement at specified timesteps.
    Returns displacement magnitude at each timestep.
    
    For rollout: cumulative displacement = sum of increments up to timestep t.
    simulation[t].x[:, 2:4] contains cumulative displacement at timestep t.
    simulation[t].y contains the INCREMENT from t to t+1.
    """
    displacements = []
    for t in timesteps:
        # x[:, 2:4] = [Ux_cumulative, Uy_cumulative] at timestep t
        u = simulation[t].x[:, 2:4].cpu().numpy()
        mag = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
        displacements.append(mag)
    return displacements


def get_erosion_mask(simulation, timestep):
    """Get eroded element mask at a given timestep."""
    data = simulation[timestep]
    if hasattr(data, 'x_element') and data.x_element is not None:
        erosion = data.x_element.squeeze().cpu().numpy()
        return erosion < 0.5  # True = eroded
    return None


# ============================================================
# MODEL LOADING
# ============================================================

def load_gparc(checkpoint_path, norm_stats, sample_data, device):
    """Load G-PARC model."""
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from differentiator.differentiator import ElastoPlasticDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from models.globalelasto import GPARC_ElastoPlastic_Numerical

    pos_stats = norm_stats['position']
    pos_mean = [pos_stats['x_pos']['mean'], pos_stats['y_pos']['mean']]
    pos_std = [pos_stats['x_pos']['std'], pos_stats['y_pos']['std']]
    norm_method = norm_stats.get('normalization_method', 'global_max')
    max_position = pos_stats.get('max_position', 200.0)

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
        in_channels=2, hidden_channels=128, out_channels=128,
        num_layers=4, dropout=0.0, use_layer_norm=True, use_relative_pos=True
    )

    derivative_solver = ElastoPlasticDifferentiator(
        num_static_feats=2, num_dynamic_feats=2,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=128,
        list_strain_idx=[0, 1], list_laplacian_idx=[0, 1],
        spade_random_noise=False, heads=4, concat=True,
        dropout=0.1, use_von_mises=True, use_volumetric=True,
        n_state_var=0, zero_init=True
    )
    derivative_solver.initialize_weights(sample_data)

    model = GPARC_ElastoPlastic_Numerical(
        derivative_solver_physics=derivative_solver,
        integrator_type='euler',
        num_static_feats=2, num_dynamic_feats=2,
        pos_mean=pos_mean, pos_std=pos_std,
        boundary_threshold=0.5, clamp_output=False,
        norm_method=norm_method, max_position=max_position,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  G-PARC loaded (epoch {ckpt['epoch']})")
    return model


def load_meshgraphnet(checkpoint_path, stats_path, device):
    """Load MeshGraphNet model."""
    from meshgraphnet import MeshGraphNet

    model = MeshGraphNet(
        input_dim_node=4, input_dim_edge=3,
        hidden_dim=128, output_dim=2, num_layers=4
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load normalization stats
    stats = torch.load(stats_path, map_location=device)
    print(f"  MeshGraphNet loaded (epoch {ckpt['epoch']})")
    return model, stats


def load_meshgraphkan(checkpoint_path, device):
    """Load MeshGraphKAN model."""
    # Import the classes defined in the training script
    from train_meshgraphkan import MeshGraphKAN, MeshGraphKANRollout

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

    model = MeshGraphKANRollout(
        kan_model, num_static_feats=2, num_dynamic_feats=2
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  MeshGraphKAN loaded (epoch {ckpt['epoch']})")
    return model


# ============================================================
# ROLLOUT — shared logic for all models
# ============================================================

def rollout_gparc(model, simulation, num_steps, device):
    """Run G-PARC rollout, return cumulative displacements at each step."""
    for data in simulation:
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.edge_index = data.edge_index.to(device)
        if hasattr(data, 'pos') and data.pos is not None:
            data.pos = data.pos.to(device)
        else:
            data.pos = data.x[:, :2]

    # Initialize MLS weights
    deriv = model.derivative_solver
    if hasattr(deriv, 'initialize_weights'):
        deriv.initialize_weights(simulation[0])

    sf, df = model.num_static_feats, model.num_dynamic_feats
    static = simulation[0].x[:, :sf]
    current_dynamic = simulation[0].x[:, sf:sf + df].clone()
    edge_index = simulation[0].edge_index

    # Create a SINGLE reusable Data object (so cache keys stay stable)
    from torch_geometric.data import Data
    input_data = Data(
        x=torch.cat([static, current_dynamic], dim=-1),
        edge_index=edge_index,
        pos=static,
        y=simulation[0].y,
    )
    # Copy element data from first frame
    for attr in ['elements', 'x_element', 'y_element', 'mesh_id']:
        if hasattr(simulation[0], attr):
            setattr(input_data, attr, getattr(simulation[0], attr))

    cumulative_disps = [current_dynamic.cpu().numpy()]
    first_nan_step = None
    first_diverge_step = None
    DIVERGE_THRESHOLD = 1.0  # normalized displacement > 1.0 is already 100% of max_disp

    with torch.no_grad():
        for t in range(num_steps):
            # Update the reusable data object in-place
            input_data.x = torch.cat([static, current_dynamic], dim=-1)
            input_data.y = simulation[t].y
            if hasattr(simulation[t], 'x_element'):
                input_data.x_element = simulation[t].x_element
            if hasattr(simulation[t], 'y_element'):
                input_data.y_element = simulation[t].y_element

            seq = [input_data]
            preds = model(seq, dt=1.0, teacher_forcing_ratio=0.0)
            F_next = preds[0]  # model returns NEXT STATE, not delta

            # Early divergence detection (before NaN)
            f_max = F_next.abs().max().item()
            if first_diverge_step is None and f_max > DIVERGE_THRESHOLD:
                first_diverge_step = t
                # Find the worst nodes
                node_max = F_next.abs().max(dim=1).values
                worst_nodes = torch.argsort(node_max, descending=True)[:20]
                
                # Get neighbor counts for these nodes
                row = edge_index[0]
                neighbor_count = torch.zeros(static.shape[0], device=device, dtype=torch.long)
                neighbor_count.index_add_(0, row, torch.ones(row.shape[0], device=device, dtype=torch.long))
                
                print(f"  🟡 DIVERGENCE at step {t}: max|F_next|={f_max:.4e}")
                print(f"     Top 20 worst nodes (by |F_next|):")
                print(f"     {'Node':>6} {'Neighbors':>9} {'|F_next|':>12} {'pos_x':>8} {'pos_y':>8} {'Fx':>12} {'Fy':>12}")
                for n in worst_nodes:
                    n = n.item()
                    nc = neighbor_count[n].item()
                    dm = node_max[n].item()
                    px, py = static[n, 0].item(), static[n, 1].item()
                    dx, dy = F_next[n, 0].item(), F_next[n, 1].item()
                    print(f"     {n:>6} {nc:>9} {dm:>12.4e} {px:>8.4f} {py:>8.4f} {dx:>12.4e} {dy:>12.4e}")

            # NaN detection
            if torch.any(torch.isnan(F_next)) or torch.any(torch.isinf(F_next)):
                if first_nan_step is None:
                    first_nan_step = t
                    nan_count = torch.isnan(F_next).any(dim=1).sum().item()
                    inf_count = torch.isinf(F_next).any(dim=1).sum().item()
                    print(f"  🔴 FIRST NaN/Inf at step {t}: "
                          f"{nan_count} NaN nodes, {inf_count} Inf nodes")

            current_dynamic = F_next  # next state IS the cumulative displacement
            cumulative_disps.append(current_dynamic.cpu().numpy())

    if first_nan_step is not None:
        print(f"  ⚠️  NaN first appeared at step {first_nan_step}/{num_steps}")
    elif first_diverge_step is not None:
        print(f"  ⚠️  Divergence at step {first_diverge_step} but no NaN")
    else:
        print(f"  ✓ Rollout clean — no NaN or divergence in {num_steps} steps")

    return cumulative_disps  # list of [N, 2] arrays


def rollout_mgn(model, simulation, num_steps, device, stats):
    """Run MeshGraphNet rollout."""
    mean_x = stats['mean_vec_x'].to(device)
    std_x = stats['std_vec_x'].to(device)
    mean_edge = stats['mean_vec_edge'].to(device)
    std_edge = stats['std_vec_edge'].to(device)
    mean_y = stats['mean_vec_y'].to(device)
    std_y = stats['std_vec_y'].to(device)

    first = simulation[0]
    static = first.x[:, :2].to(device)
    current_dynamic = first.x[:, 2:4].clone().to(device)
    edge_index = first.edge_index.to(device)
    pos = first.x[:, :2].to(device)

    cumulative_disps = [current_dynamic.cpu().numpy()]

    from torch_geometric.data import Data

    t0 = time.time()
    with torch.no_grad():
        for t in range(num_steps):
            if t % 5 == 0:
                print(f"    MGN step {t}/{num_steps} ({time.time()-t0:.1f}s)")
            node_feats = torch.cat([static, current_dynamic], dim=-1)
            data = Data(x=node_feats, pos=pos, edge_index=edge_index)

            pred_norm = model(data, mean_x, std_x, mean_edge, std_edge)
            # Unnormalize prediction
            delta = pred_norm * std_y + mean_y

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

    cumulative_disps = [current_dynamic.cpu().numpy()]

    from torch_geometric.data import Data

    t0 = time.time()
    with torch.no_grad():
        for t in range(num_steps):
            if t % 5 == 0:
                print(f"    MGKAN step {t}/{num_steps} ({time.time()-t0:.1f}s)")
            node_feats = torch.cat([static, current_dynamic], dim=-1)
            data = Data(
                x=node_feats, edge_index=edge_index,
                pos=static, y=torch.zeros_like(current_dynamic)
            )

            edge_feat = model.compute_edge_features(data)
            delta = model.model(node_feats, edge_feat, edge_index)

            current_dynamic = current_dynamic + delta
            cumulative_disps.append(current_dynamic.cpu().numpy())

    print(f"    MGKAN done ({time.time()-t0:.1f}s)")
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
# FIGURE CREATION
# ============================================================

def render_mesh_poly(ax, pos, elements, node_values, cmap, norm, erosion_mask=None):
    """
    Render mesh using PolyCollection (element-averaged coloring).
    Handles NaN values gracefully by showing those elements in gray.
    """
    # Filter eroded elements
    if erosion_mask is not None and erosion_mask.any():
        valid_mask = ~erosion_mask
    else:
        valid_mask = np.ones(len(elements), dtype=bool)

    valid_elements = elements[valid_mask]

    if len(valid_elements) == 0:
        return

    # Check for NaN in node values
    has_nan = np.any(np.isnan(node_values))

    # Build polygons and element-averaged values
    polygons = pos[valid_elements]  # [M, 3, 2]
    elem_values = node_values[valid_elements].mean(axis=1)  # [M]

    if has_nan:
        # Separate NaN and valid elements
        nan_elems = np.isnan(elem_values)
        valid_elems = ~nan_elems

        if valid_elems.any():
            cmap_obj = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
            colors = cmap_obj(norm(elem_values[valid_elems]))
            pc = PolyCollection(polygons[valid_elems], facecolors=colors,
                                edgecolors='face', linewidths=0.1)
            ax.add_collection(pc)

        if nan_elems.any():
            pc_nan = PolyCollection(polygons[nan_elems], facecolors='lightgray',
                                    edgecolors='gray', linewidths=0.1, alpha=0.5)
            ax.add_collection(pc_nan)
    else:
        cmap_obj = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
        colors = cmap_obj(norm(elem_values))
        pc = PolyCollection(polygons, facecolors=colors,
                            edgecolors='face', linewidths=0.1)
        ax.add_collection(pc)

    # Show eroded elements as white/transparent
    if erosion_mask is not None and erosion_mask.any():
        eroded_elements = elements[erosion_mask]
        eroded_polygons = pos[eroded_elements]
        pc_eroded = PolyCollection(eroded_polygons, facecolors='white',
                                   edgecolors='lightgray', linewidths=0.1, alpha=0.3)
        ax.add_collection(pc_eroded)


def create_comparison_figure(
    pos, elements, gt_disps, model_disps, model_names,
    timesteps, sim_name, norm_stats, output_dir,
    erosion_masks=None, dpi=300
):
    """
    Create the publication comparison figure.
    Uses PolyCollection rendering and GT-only colorbar range.
    """
    n_rows = 1 + len(model_names)  # GT + models
    n_cols = len(timesteps)
    row_labels = ['Ground Truth'] + model_names

    # Denormalize positions for display
    method = norm_stats.get('normalization_method', 'global_max')
    if method == 'global_max':
        max_pos = norm_stats['position']['max_position']
        pos_phys = pos * max_pos
    else:
        pos_phys = pos

    # Colorbar range from GROUND TRUTH ONLY (not diluted by diverging baselines)
    vmin = 0
    vmax = max(np.nanmax(m) for m in gt_disps)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.jet

    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.2 * n_rows),
        squeeze=False
    )

    for col_idx, t in enumerate(timesteps):
        emask = erosion_masks.get(t, None) if erosion_masks else None

        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]

            if row_idx == 0:
                values = gt_disps[col_idx]
            else:
                name = model_names[row_idx - 1]
                values = model_disps[name][col_idx]

            # Render with PolyCollection
            render_mesh_poly(ax, pos_phys, elements, values, cmap, norm, emask)

            ax.set_aspect('equal')
            ax.set_xlim(pos_phys[:, 0].min() - 1, pos_phys[:, 0].max() + 1)
            ax.set_ylim(pos_phys[:, 1].min() - 1, pos_phys[:, 1].max() + 1)

            # Remove all axes
            ax.set_xticks([])
            ax.set_yticks([])

            # Column titles (timestep)
            if row_idx == 0:
                ax.set_title(f't = {t}', fontsize=11, fontweight='bold')

            # Row labels
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=10, fontweight='bold')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Displacement Magnitude (mm)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(
        left=0.06, right=0.90, top=0.93, bottom=0.05,
        wspace=0.08, hspace=0.12
    )
    fig.suptitle(
        f'Rollout Displacement Comparison — {sim_name}',
        fontsize=13, fontweight='bold', y=0.97
    )

    # Save PNG
    out_path = Path(output_dir) / f'model_comparison_{sim_name}.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Saved: {out_path}")

    # Save PDF for LaTeX
    pdf_path = Path(output_dir) / f'model_comparison_{sim_name}.pdf'
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows), squeeze=False)
    for col_idx, t in enumerate(timesteps):
        emask = erosion_masks.get(t, None) if erosion_masks else None
        for row_idx in range(n_rows):
            ax = axes2[row_idx, col_idx]
            values = gt_disps[col_idx] if row_idx == 0 else model_disps[model_names[row_idx - 1]][col_idx]
            render_mesh_poly(ax, pos_phys, elements, values, cmap, norm, emask)
            ax.set_aspect('equal')
            ax.set_xlim(pos_phys[:, 0].min() - 1, pos_phys[:, 0].max() + 1)
            ax.set_ylim(pos_phys[:, 1].min() - 1, pos_phys[:, 1].max() + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f't = {t}', fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=10, fontweight='bold')

    cbar_ax2 = fig2.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar2 = fig2.colorbar(sm, cax=cbar_ax2)
    cbar2.set_label('Displacement Magnitude (mm)', fontsize=10)
    cbar2.ax.tick_params(labelsize=8)
    plt.subplots_adjust(left=0.06, right=0.90, top=0.93, bottom=0.05, wspace=0.08, hspace=0.12)
    fig2.suptitle(f'Rollout Displacement Comparison — {sim_name}', fontsize=13, fontweight='bold', y=0.97)
    fig2.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig2)
    print(f"✅ Saved: {pdf_path}")


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
                        default=['gparc'],
                        help="Models to include: gparc mgn mgkan (e.g. --models gparc mgn)")

    # Override paths from command line if desired
    parser.add_argument("--test_dir", type=str, default=TEST_DIR)
    parser.add_argument("--norm_stats", type=str, default=NORM_STATS_PATH)
    parser.add_argument("--gparc_ckpt", type=str, default=GPARC_CHECKPOINT)
    parser.add_argument("--mgn_ckpt", type=str, default=MGN_CHECKPOINT)
    parser.add_argument("--mgkan_ckpt", type=str, default=MGKAN_CHECKPOINT)
    parser.add_argument("--mgn_stats", type=str, default=MGN_STATS_PATH)

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
    total_steps = len(simulation) - 1  # -1 because last frame has no target
    print(f"Simulation: {sim_name}, nodes={pos.shape[0]}, steps={total_steps}")

    # Pick evenly spaced timesteps
    timesteps = np.linspace(0, total_steps - 1, args.num_timesteps, dtype=int).tolist()
    print(f"Timesteps: {timesteps}")

    # Ground truth: cumulative displacement at each timestep
    gt_disps_norm = []
    erosion_masks = {}
    for t in timesteps:
        u = simulation[t].x[:, 2:4].cpu().numpy()
        u_phys = u * max_disp
        mag = np.sqrt(u_phys[:, 0]**2 + u_phys[:, 1]**2)
        gt_disps_norm.append(mag)

        emask = get_erosion_mask(simulation, t)
        if emask is not None:
            erosion_masks[t] = emask

    # Model registry: key -> (display_name, load_fn, rollout_fn)
    sample_data = simulation[0]
    sample_data.pos = sample_data.x[:, :2]
    rollout_steps = total_steps

    # Storage for loaded models and rollout results
    model_names = []
    all_disps = {}

    selected = [m.lower() for m in args.models]
    print(f"\nSelected models: {selected}")

    if 'gparc' in selected:
        print("\n  Loading G-PARC...")
        gparc_model = load_gparc(args.gparc_ckpt, norm_stats, sample_data, device)
        print("  G-PARC rollout...")
        gparc_disps = rollout_gparc(gparc_model, simulation, rollout_steps, device)
        model_names.append('G-PARC (ours)')
        all_disps['G-PARC (ours)'] = gparc_disps

    if 'mgn' in selected:
        print("\n  Loading MeshGraphNet...")
        mgn_model, mgn_stats = load_meshgraphnet(args.mgn_ckpt, args.mgn_stats, device)
        print("  MeshGraphNet rollout...")
        mgn_disps = rollout_mgn(mgn_model, simulation, rollout_steps, device, mgn_stats)
        model_names.append('MeshGraphNet')
        all_disps['MeshGraphNet'] = mgn_disps

    if 'mgkan' in selected:
        print("\n  Loading MeshGraphKAN...")
        mgkan_model = load_meshgraphkan(args.mgkan_ckpt, device)
        print("  MeshGraphKAN rollout...")
        mgkan_disps = rollout_mgkan(mgkan_model, simulation, rollout_steps, device)
        model_names.append('MeshGraphKAN')
        all_disps['MeshGraphKAN'] = mgkan_disps

    if len(model_names) == 0:
        print("ERROR: No valid models selected. Use --models gparc mgn mgkan")
        return

    # Extract displacement magnitudes at selected timesteps (physical units)
    model_disps = {}
    for name in model_names:
        disps = all_disps[name]
        mags = []
        for t in timesteps:
            u = disps[t]  # [N, 2] normalized
            u_phys = u * max_disp
            mag = np.sqrt(u_phys[:, 0]**2 + u_phys[:, 1]**2)
            if np.any(np.isnan(mag)):
                nan_pct = 100 * np.sum(np.isnan(mag)) / len(mag)
                print(f"  ⚠️  {name} has {nan_pct:.1f}% NaN nodes at t={t}")
            mags.append(mag)
        model_disps[name] = mags

    # Create figure
    print("\nCreating comparison figure...")
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
    )

    # Print RRMSE summary for the selected simulation
    print("\n" + "="*60)
    print("PER-SIMULATION ROLLOUT ERROR (last timestep)")
    print("="*60)
    t_final = timesteps[-1]
    gt_u = simulation[t_final].x[:, 2:4].cpu().numpy() * max_disp
    gt_norm = np.sqrt(np.mean(gt_u**2))

    for name in model_names:
        disps = all_disps[name]
        pred_u = disps[t_final] * max_disp
        rmse = np.sqrt(np.nanmean((pred_u - gt_u)**2))
        rrmse = rmse / (gt_norm + 1e-10)
        print(f"  {name:20s}: RMSE={rmse:.4f} mm, RRMSE={rrmse:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()