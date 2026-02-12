#!/usr/bin/env python3
"""
NaN Source Diagnostic for G-PARC Rollout
=========================================
Finds exactly which node(s) and timestep produce the first NaN,
and checks if they're boundary nodes with low MLS neighbor counts.

Usage:
    python diagnose_nan_rollout.py --sim_index 0
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ============================================================
# CONFIGURATION — same as figures.py
# ============================================================
TEST_DIR = "/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/test"
NORM_STATS_PATH = "/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/normalization_stats.json"
GPARC_CHECKPOINT = "/scratch/jtb3sud/elasto_graphconv_V2/global_max_v3_lapdamp/best_model.pth"


def load_norm_stats(path):
    with open(path) as f:
        return json.load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_dir", type=str, default=TEST_DIR)
    parser.add_argument("--norm_stats", type=str, default=NORM_STATS_PATH)
    parser.add_argument("--gparc_ckpt", type=str, default=GPARC_CHECKPOINT)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load
    norm_stats = load_norm_stats(args.norm_stats)
    sim_files = sorted(Path(args.test_dir).glob("simulation_*.pt"))
    sim_file = sim_files[args.sim_index]
    print(f"Loading: {sim_file.name}")
    simulation = torch.load(sim_file, weights_only=False)

    # Load G-PARC
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from differentiator.differentiator import ElastoPlasticDifferentiator
    from differentiator.fix import SolveGradientsLST, SolveWeightLST2d
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
        min_neighbors=5
    )

    sample_data = simulation[0]
    sample_data.pos = sample_data.x[:, :2]

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

    ckpt = torch.load(args.gparc_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"G-PARC loaded (epoch {ckpt['epoch']})")

    # Move simulation to device
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

    # ============================================================
    # Get MLS neighbor counts for analysis
    # ============================================================
    pos_ref = simulation[0].x[:, :2]
    edge_index = simulation[0].edge_index

    # Count neighbors per node
    dst_nodes = edge_index[1].cpu().numpy()
    neighbor_counts = np.bincount(dst_nodes, minlength=pos_ref.shape[0])

    # Identify boundary nodes (from static features or position)
    pos_np = pos_ref.cpu().numpy()
    max_disp = norm_stats['displacement']['max_displacement']

    print(f"\nMesh: {pos_ref.shape[0]} nodes, {edge_index.shape[1]} edges")
    print(f"Neighbor count range: {neighbor_counts.min()} - {neighbor_counts.max()}")
    print(f"Nodes with ≤3 neighbors: {(neighbor_counts <= 3).sum()}")
    print(f"Nodes with ≤4 neighbors: {(neighbor_counts <= 4).sum()}")
    print(f"Nodes with ≤5 neighbors: {(neighbor_counts <= 5).sum()}")

    # ============================================================
    # Run rollout step by step, checking for NaN
    # ============================================================
    from torch_geometric.data import Data

    sf, df = model.num_static_feats, model.num_dynamic_feats
    static = simulation[0].x[:, :sf]
    current_dynamic = simulation[0].x[:, sf:sf + df].clone()
    num_steps = len(simulation) - 1

    print(f"\n{'='*70}")
    print(f"ROLLOUT NaN DIAGNOSTIC ({num_steps} steps)")
    print(f"{'='*70}")

    first_nan_step = None

    with torch.no_grad():
        for t in range(num_steps):
            data_t = simulation[t]
            input_data = Data(
                x=torch.cat([static, current_dynamic], dim=-1),
                edge_index=edge_index,
                pos=static,
                y=data_t.y,
            )
            for attr in ['elements', 'x_element', 'y_element', 'mesh_id']:
                if hasattr(data_t, attr):
                    setattr(input_data, attr, getattr(data_t, attr))

            seq = [input_data]
            preds = model(seq, dt=1.0, teacher_forcing_ratio=0.0)
            delta = preds[0]

            # Check delta for NaN/Inf BEFORE accumulation
            delta_np = delta.cpu().numpy()
            nan_mask = np.isnan(delta_np)
            inf_mask = np.isinf(delta_np)

            if nan_mask.any() or inf_mask.any():
                nan_nodes = np.where(nan_mask.any(axis=1))[0]
                inf_nodes = np.where(inf_mask.any(axis=1))[0]
                bad_nodes = np.unique(np.concatenate([nan_nodes, inf_nodes]))

                if first_nan_step is None:
                    first_nan_step = t
                    print(f"\n🔴 FIRST NaN/Inf at step {t}")
                    print(f"   NaN nodes: {len(nan_nodes)}, Inf nodes: {len(inf_nodes)}")
                    print(f"   Total bad: {len(bad_nodes)} / {delta_np.shape[0]}")

                    print(f"\n   First 10 bad nodes:")
                    print(f"   {'Node':>6} {'Neighbors':>9} {'delta_Ux':>12} {'delta_Uy':>12} {'pos_x':>8} {'pos_y':>8}")
                    for node_id in bad_nodes[:10]:
                        nc = neighbor_counts[node_id]
                        dx, dy = delta_np[node_id]
                        px, py = pos_np[node_id]
                        print(f"   {node_id:6d} {nc:9d} {dx:12.4f} {dy:12.4f} {px:8.4f} {py:8.4f}")

                    # Check current_dynamic at these nodes BEFORE this step
                    cd_np = current_dynamic.cpu().numpy()
                    print(f"\n   State at bad nodes BEFORE step {t}:")
                    print(f"   {'Node':>6} {'Ux':>12} {'Uy':>12} {'|U|':>12}")
                    for node_id in bad_nodes[:10]:
                        ux, uy = cd_np[node_id]
                        mag = np.sqrt(ux**2 + uy**2)
                        print(f"   {node_id:6d} {ux:12.6f} {uy:12.6f} {mag:12.6f}")

                    # Check if bad nodes are near domain boundary
                    print(f"\n   Position analysis of bad nodes:")
                    x_min, x_max = pos_np[:, 0].min(), pos_np[:, 0].max()
                    y_min, y_max = pos_np[:, 1].min(), pos_np[:, 1].max()
                    for node_id in bad_nodes[:10]:
                        px, py = pos_np[node_id]
                        on_left = abs(px - x_min) < 0.02
                        on_right = abs(px - x_max) < 0.02
                        on_bottom = abs(py - y_min) < 0.02
                        on_top = abs(py - y_max) < 0.02
                        boundary = []
                        if on_left: boundary.append("LEFT")
                        if on_right: boundary.append("RIGHT")
                        if on_bottom: boundary.append("BOTTOM")
                        if on_top: boundary.append("TOP")
                        loc = ", ".join(boundary) if boundary else "INTERIOR"
                        print(f"   Node {node_id}: ({px:.4f}, {py:.4f}) -> {loc}")
                else:
                    print(f"   Step {t}: {len(bad_nodes)} bad nodes (propagating)")

                # Stop after a few propagation steps
                if t > first_nan_step + 3:
                    print(f"\n   ... NaN propagating. Stopping diagnostic.")
                    break
            else:
                # Print summary stats for healthy steps
                max_delta = np.abs(delta_np).max()
                cd_np = current_dynamic.cpu().numpy()
                max_disp_val = np.sqrt(cd_np[:, 0]**2 + cd_np[:, 1]**2).max()
                if t % 5 == 0 or t == num_steps - 1:
                    print(f"   Step {t:3d}: max|delta|={max_delta:.6f}, max|U|={max_disp_val:.6f} ✓")

            current_dynamic = current_dynamic + delta

    if first_nan_step is None:
        print(f"\n✅ No NaN detected in full {num_steps}-step rollout!")
    else:
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"First NaN at step: {t}")
        print(f"Total steps: {num_steps}")
        print(f"Steps before failure: {first_nan_step}")

        # Analyze the bad nodes
        nan_neighbor_counts = neighbor_counts[bad_nodes]
        print(f"\nBad node neighbor count distribution:")
        for nc_val in sorted(np.unique(nan_neighbor_counts)):
            count = (nan_neighbor_counts == nc_val).sum()
            print(f"  {nc_val} neighbors: {count} nodes")

        low_neighbor = (nan_neighbor_counts <= 5).sum()
        print(f"\nBad nodes with ≤5 neighbors: {low_neighbor}/{len(bad_nodes)} ({100*low_neighbor/len(bad_nodes):.1f}%)")


if __name__ == "__main__":
    main()
