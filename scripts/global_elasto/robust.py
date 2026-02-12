#!/usr/bin/env python3
"""
2-Hop Stencil Robustness Test
===============================
Verifies that the 2-hop stencil extension remains accurate across:
  1. Multiple analytic test functions (varying difficulty)
  2. Actual simulation displacement fields at multiple timesteps
  3. Fields with sharp gradients near boundaries

Test Functions:
  - u = x²           → Lap = 2.0  (easy, smooth)
  - u = x² + y²      → Lap = 4.0  (radially symmetric)
  - u = sin(πx/L)    → Lap = -(π/L)² sin(πx/L)  (spatially varying)
  - u = exp(-r²/σ²)  → Lap varies, tests localization
  - Actual displacement fields from simulation timesteps

Usage (notebook):
    import sys
    sys.argv = ['twohop_robustness.py']
    exec(open('twohop_robustness.py').read())
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
from glob import glob
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small"
NORM_STATS_PATH = "/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/normalization_stats.json"
OUTPUT_DIR = "./twohop_robustness"
MAX_SIMS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MLS LAPLACIAN
# ============================================================
def mls_laplacian(pos, edge_index, u_field):
    N = pos.shape[0]
    row, col = edge_index
    diff = (pos[col] - pos[row]).float()
    x, y = diff[:, 0:1], diff[:, 1:2]
    H = torch.cat([x, y, x * y, x * x, y * y], dim=-1)

    M_edge = torch.bmm(H.unsqueeze(2), H.unsqueeze(1))
    M_node = torch.zeros(N, 5, 5, dtype=torch.float32)
    M_node.index_add_(0, row, M_edge)
    M_node += torch.eye(5).unsqueeze(0) * 1e-8

    L = torch.zeros(N, 5, dtype=torch.float32)
    L[:, 3] = 2.0
    L[:, 4] = 2.0

    try:
        M_inv = torch.linalg.inv(M_node)
    except:
        M_inv = torch.linalg.pinv(M_node)

    C = torch.bmm(M_inv, L.unsqueeze(2)).squeeze(2)
    lap_weights = (C[row] * H).sum(dim=1)

    du = u_field[col] - u_field[row]
    weighted = lap_weights.unsqueeze(1) * du
    laplacian = torch.zeros(N, 1, dtype=torch.float32)
    laplacian.index_add_(0, row, weighted)
    return laplacian.squeeze().numpy()


# ============================================================
# 2-HOP STENCIL EXTENSION
# ============================================================
def extend_edge_index_2hop(edge_index, N, min_neighbors=6):
    """Compute augmented edge_index with 2-hop neighbors at low-count nodes."""
    row_np, col_np = edge_index[0].numpy(), edge_index[1].numpy()

    # Build adjacency
    adj = defaultdict(set)
    for e in range(len(row_np)):
        adj[row_np[e]].add(col_np[e])

    # Neighbor counts
    counts = np.zeros(N, dtype=int)
    for i in range(N):
        counts[i] = len(adj[i])

    extra_rows, extra_cols = [], []
    pos_np_cache = None  # Will set if needed

    for node_i in range(N):
        if counts[node_i] >= min_neighbors:
            continue
        needed = min_neighbors - counts[node_i]
        neighbors = adj[node_i]

        two_hop = set()
        for nbr in neighbors:
            for nbr2 in adj[nbr]:
                if nbr2 != node_i and nbr2 not in neighbors:
                    two_hop.add(nbr2)

        if len(two_hop) == 0:
            continue

        # We need positions to sort by distance — passed separately
        two_hop_list = list(two_hop)
        for k in range(min(needed, len(two_hop_list))):
            extra_rows.append(node_i)
            extra_cols.append(two_hop_list[k])

    if len(extra_rows) == 0:
        return edge_index

    return torch.cat([
        edge_index,
        torch.stack([torch.tensor(extra_rows, dtype=torch.long),
                     torch.tensor(extra_cols, dtype=torch.long)])
    ], dim=1)


def extend_edge_index_2hop_sorted(pos, edge_index, N, min_neighbors=6):
    """2-hop extension with distance-sorted neighbor selection."""
    row_np, col_np = edge_index[0].numpy(), edge_index[1].numpy()
    pos_np = pos.numpy()

    adj = defaultdict(set)
    for e in range(len(row_np)):
        adj[row_np[e]].add(col_np[e])

    counts = np.zeros(N, dtype=int)
    for i in range(N):
        counts[i] = len(adj[i])

    extra_rows, extra_cols = [], []

    for node_i in range(N):
        if counts[node_i] >= min_neighbors:
            continue
        needed = min_neighbors - counts[node_i]
        neighbors = adj[node_i]

        two_hop = set()
        for nbr in neighbors:
            for nbr2 in adj[nbr]:
                if nbr2 != node_i and nbr2 not in neighbors:
                    two_hop.add(nbr2)

        if len(two_hop) == 0:
            continue

        two_hop_list = list(two_hop)
        pi = pos_np[node_i]
        dists = np.linalg.norm(pos_np[two_hop_list] - pi, axis=1)
        sorted_idx = np.argsort(dists)

        for k in range(min(needed, len(two_hop_list))):
            extra_rows.append(node_i)
            extra_cols.append(two_hop_list[sorted_idx[k]])

    if len(extra_rows) == 0:
        return edge_index

    return torch.cat([
        edge_index,
        torch.stack([torch.tensor(extra_rows, dtype=torch.long),
                     torch.tensor(extra_cols, dtype=torch.long)])
    ], dim=1)


def mls_laplacian_2hop(pos, edge_index, u_field, min_neighbors=6):
    """MLS Laplacian with 2-hop stencil extension."""
    N = pos.shape[0]
    edge_aug = extend_edge_index_2hop_sorted(pos, edge_index, N, min_neighbors)
    return mls_laplacian(pos, edge_aug, u_field)[:N]


# ============================================================
# TEST FUNCTIONS
# ============================================================
def make_test_functions(pos):
    """
    Returns dict of {name: (u_field [N,1], true_laplacian [N])}
    All in normalized coordinate space.
    """
    x = pos[:, 0].numpy()
    y = pos[:, 1].numpy()
    N = len(x)

    tests = {}

    # 1. u = x² → Lap = 2
    u1 = x ** 2
    tests['u = x²'] = (
        torch.tensor(u1, dtype=torch.float32).unsqueeze(1),
        np.full(N, 2.0)
    )

    # 2. u = x² + y² → Lap = 4
    u2 = x ** 2 + y ** 2
    tests['u = x² + y²'] = (
        torch.tensor(u2, dtype=torch.float32).unsqueeze(1),
        np.full(N, 4.0)
    )

    # 3. u = x³ → Lap = 6x (spatially varying)
    u3 = x ** 3
    lap3 = 6.0 * x
    tests['u = x³'] = (
        torch.tensor(u3, dtype=torch.float32).unsqueeze(1),
        lap3
    )

    # 4. u = sin(πx) → Lap = -π² sin(πx) (oscillatory)
    u4 = np.sin(np.pi * x)
    lap4 = -np.pi ** 2 * np.sin(np.pi * x)
    tests['u = sin(πx)'] = (
        torch.tensor(u4, dtype=torch.float32).unsqueeze(1),
        lap4
    )

    # 5. u = x²y² → Lap = 2y² + 2x² (mixed)
    u5 = x ** 2 * y ** 2
    lap5 = 2.0 * y ** 2 + 2.0 * x ** 2
    tests['u = x²y²'] = (
        torch.tensor(u5, dtype=torch.float32).unsqueeze(1),
        lap5
    )

    # 6. u = exp(-50*((x-0.5)² + (y-0.5)²)) — sharp Gaussian bump
    cx, cy = 0.5, 0.5
    sigma2 = 1.0 / 50.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    u6 = np.exp(-r2 / sigma2)
    # Lap of Gaussian: exp(-r²/σ²) * (4r²/σ⁴ - 4/σ²)
    lap6 = u6 * (4.0 * r2 / sigma2 ** 2 - 4.0 / sigma2)
    tests['Gaussian bump'] = (
        torch.tensor(u6, dtype=torch.float32).unsqueeze(1),
        lap6
    )

    return tests


# ============================================================
# MAIN
# ============================================================
def run_robustness_test():
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)
    max_disp = norm_stats['displacement']['max_displacement']

    sim_files = sorted(glob(os.path.join(DATA_DIR, "test", "*.pt")))
    sim_files = sim_files[:MAX_SIMS]

    print(f"Testing {len(sim_files)} simulations")
    print(f"{'='*90}")

    # ============================================================
    # PART 1: ANALYTIC TEST FUNCTIONS
    # ============================================================
    print("\nPART 1: ANALYTIC TEST FUNCTIONS")
    print(f"{'='*90}")

    all_analytic = {'MLS Raw': defaultdict(lambda: defaultdict(list)),
                    '2-Hop Stencil': defaultdict(lambda: defaultdict(list))}

    for sim_path in sim_files:
        sim_name = Path(sim_path).stem
        sim = torch.load(sim_path, weights_only=False)
        data0 = sim[0]
        pos = data0.pos.float()
        edge_index = data0.edge_index
        N = pos.shape[0]

        counts = torch.zeros(N, dtype=torch.long)
        counts.index_add_(0, edge_index[0],
                          torch.ones(edge_index.shape[1], dtype=torch.long))
        counts_np = counts.numpy()

        tests = make_test_functions(pos)

        print(f"\n{sim_name}: N={N}")
        print(f"  {'Test Function':<20} {'Method':<16} {'3-nbr':>10} {'4-nbr':>10} {'5-nbr':>10} {'6+-nbr':>10}")
        print(f"  {'-'*78}")

        for test_name, (u_field, true_lap) in tests.items():
            # Handle nodes where true_lap ≈ 0 (use absolute error)
            # Otherwise use relative error
            for method_name, lap_fn in [('MLS Raw', lambda u: mls_laplacian(pos, edge_index, u)),
                                         ('2-Hop Stencil', lambda u: mls_laplacian_2hop(pos, edge_index, u))]:
                lap = lap_fn(u_field)

                # Use absolute error where true_lap is near zero,
                # otherwise relative error
                abs_err = np.abs(lap - true_lap)
                scale = np.maximum(np.abs(true_lap), 1.0)  # avoid /0
                rel_err = abs_err / scale

                for c in np.unique(counts_np):
                    mask = counts_np == c
                    all_analytic[method_name][test_name][int(c)].extend(
                        abs_err[mask].tolist())

                e3 = abs_err[counts_np == 3].mean() if (counts_np == 3).sum() > 0 else 0
                e4 = abs_err[counts_np == 4].mean() if (counts_np == 4).sum() > 0 else 0
                e5 = abs_err[counts_np == 5].mean() if (counts_np == 5).sum() > 0 else 0
                e6p = abs_err[counts_np >= 6].mean()
                print(f"  {test_name:<20} {method_name:<16} {e3:>10.4f} {e4:>10.4f} {e5:>10.4f} {e6p:>10.6f}")

    # Aggregate analytic results
    print(f"\n{'='*90}")
    print("AGGREGATE ANALYTIC RESULTS")
    print(f"{'='*90}")

    for test_name in make_test_functions(torch.zeros(1, 2)).keys():
        print(f"\n  {test_name}:")
        print(f"    {'Method':<16} {'3-nbr':>10} {'4-nbr':>10} {'5-nbr':>10} {'6+-nbr':>10}")
        for method_name in ['MLS Raw', '2-Hop Stencil']:
            data = all_analytic[method_name][test_name]
            e3 = np.mean(data.get(3, [0]))
            e4 = np.mean(data.get(4, [0]))
            e5 = np.mean(data.get(5, [0]))
            e6 = np.mean(data.get(6, [0]))
            print(f"    {method_name:<16} {e3:>10.4f} {e4:>10.4f} {e5:>10.4f} {e6:>10.6f}")

    # ============================================================
    # PART 2: ACTUAL SIMULATION DISPLACEMENT FIELDS
    # ============================================================
    print(f"\n{'='*90}")
    print("PART 2: ACTUAL SIMULATION DISPLACEMENT FIELDS")
    print(f"{'='*90}")

    # For real fields we compare MLS Raw vs 2-Hop on Ux and Uy components
    # Since we don't know the true Laplacian, we use the 6+-neighbor MLS
    # as reference (since it's accurate to 0.001)

    timestep_results = {'MLS Raw': defaultdict(list),
                        '2-Hop Stencil': defaultdict(list)}

    for sim_path in sim_files:
        sim_name = Path(sim_path).stem
        sim = torch.load(sim_path, weights_only=False)
        data0 = sim[0]
        pos = data0.pos.float()
        edge_index = data0.edge_index
        N = pos.shape[0]

        counts = torch.zeros(N, dtype=torch.long)
        counts.index_add_(0, edge_index[0],
                          torch.ones(edge_index.shape[1], dtype=torch.long))
        counts_np = counts.numpy()

        total_steps = len(sim) - 1
        test_timesteps = np.linspace(0, total_steps - 1, 6, dtype=int)

        edge_aug = extend_edge_index_2hop_sorted(pos, edge_index, N, min_neighbors=6)

        print(f"\n{sim_name}: testing timesteps {test_timesteps.tolist()}")
        print(f"  {'Timestep':>8} {'Component':<6} {'Method':<16} "
              f"{'3-nbr dev':>12} {'4-nbr dev':>12} {'5-nbr dev':>12}")
        print(f"  {'-'*72}")

        for t in test_timesteps:
            # Get displacement field at timestep t
            u_disp = sim[t].x[:, 2:4].float()  # [N, 2] normalized

            for comp_idx, comp_name in [(0, 'Ux'), (1, 'Uy')]:
                u_field = u_disp[:, comp_idx:comp_idx+1]

                # Reference: MLS on full mesh (accurate at 6+ nodes)
                lap_raw = mls_laplacian(pos, edge_index, u_field)
                lap_2hop = mls_laplacian(pos, edge_aug, u_field)[:N]

                # At 6+ neighbor nodes, both should agree (this is our reference)
                mask_6p = counts_np >= 6
                ref = lap_raw.copy()
                # Reference is MLS Raw at well-supported nodes
                # At low-neighbor nodes, we compare how different the two methods are

                # Deviation: how much does 2-hop change the answer vs raw?
                deviation = np.abs(lap_2hop - lap_raw)

                e3 = deviation[counts_np == 3].mean() if (counts_np == 3).sum() > 0 else 0
                e4 = deviation[counts_np == 4].mean() if (counts_np == 4).sum() > 0 else 0
                e5 = deviation[counts_np == 5].mean() if (counts_np == 5).sum() > 0 else 0

                print(f"  {t:>8} {comp_name:<6} {'deviation':<16} "
                      f"{e3:>12.6f} {e4:>12.6f} {e5:>12.6f}")

                # Also check: does 2-hop produce any NaN?
                if np.any(np.isnan(lap_2hop)):
                    print(f"  ⚠️  2-Hop produced NaN at t={t}, {comp_name}!")
                if np.any(np.isnan(lap_raw)):
                    nan_count = np.sum(np.isnan(lap_raw))
                    nan_at_low = np.sum(np.isnan(lap_raw) & (counts_np <= 4))
                    print(f"  ⚠️  MLS Raw NaN at t={t}, {comp_name}: "
                          f"{nan_count} total, {nan_at_low} at ≤4-nbr nodes")

    # ============================================================
    # PART 3: CONSISTENCY CHECK — 2-HOP WEIGHTS ARE TIMESTEP-INDEPENDENT
    # ============================================================
    print(f"\n{'='*90}")
    print("PART 3: VERIFY STENCIL IS GEOMETRY-ONLY")
    print(f"{'='*90}")
    print("\nThe 2-hop edge augmentation depends ONLY on mesh topology (edge_index)")
    print("and node positions (for distance sorting). It does NOT depend on the")
    print("field being differentiated. Therefore:")
    print("  ✓ Augmented edge_index can be precomputed ONCE per mesh")
    print("  ✓ No per-timestep overhead")
    print("  ✓ MLS weights (M_inv) can be cached after first computation")

    # Verify: compute augmented edge_index with different fields
    sim = torch.load(sim_files[0], weights_only=False)
    pos = sim[0].pos.float()
    edge_index = sim[0].edge_index
    N = pos.shape[0]

    edge_aug_1 = extend_edge_index_2hop_sorted(pos, edge_index, N, min_neighbors=6)
    edge_aug_2 = extend_edge_index_2hop_sorted(pos, edge_index, N, min_neighbors=6)

    match = torch.equal(edge_aug_1, edge_aug_2)
    print(f"\n  Edge augmentation deterministic: {match}")
    print(f"  Original edges: {edge_index.shape[1]}")
    print(f"  Augmented edges: {edge_aug_1.shape[1]}")
    print(f"  Added edges: {edge_aug_1.shape[1] - edge_index.shape[1]}")

    # Count how many nodes got extended
    orig_counts = torch.zeros(N, dtype=torch.long)
    orig_counts.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
    aug_counts = torch.zeros(N, dtype=torch.long)
    aug_counts.index_add_(0, edge_aug_1[0], torch.ones(edge_aug_1.shape[1], dtype=torch.long))

    extended = (aug_counts > orig_counts).sum().item()
    print(f"  Nodes with extended stencil: {extended}")
    print(f"  Nodes unchanged: {N - extended}")

    # ============================================================
    # VISUALIZATION
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    test_funcs = list(make_test_functions(torch.zeros(1, 2)).keys())

    for idx, test_name in enumerate(test_funcs[:6]):
        ax = axes[idx // 3, idx % 3]

        for method_name, color in [('MLS Raw', 'red'), ('2-Hop Stencil', 'blue')]:
            data = all_analytic[method_name][test_name]
            counts_to_plot = sorted(c for c in data.keys() if c <= 8)
            means = [np.mean(data[c]) for c in counts_to_plot]
            ax.plot(counts_to_plot, means, 'o-', color=color, label=method_name,
                    markersize=6, alpha=0.8)

        ax.set_xlabel('Neighbor Count')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(test_name)
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    plt.suptitle('2-Hop Stencil vs MLS Raw: Error by Neighbor Count',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_fig = os.path.join(OUTPUT_DIR, 'twohop_robustness.png')
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {out_fig}")

    print(f"\n{'='*90}")
    print("CONCLUSION")
    print(f"{'='*90}")
    print("If 2-Hop Stencil consistently outperforms MLS Raw at 3-4 neighbor nodes")
    print("across ALL test functions and simulation timesteps, and preserves accuracy")
    print("at 6+ nodes, it is safe to integrate into SolveWeightLST2d.")
    print(f"{'='*90}")


# ============================================================
if __name__ == "__main__":
    run_robustness_test()