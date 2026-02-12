#!/usr/bin/env python3
"""
Ghost Node Refinement Diagnostic
==================================
Tests improved ghost node strategies to close the 4-neighbor accuracy gap.

Current ghost node approach:
  - 3-nbr nodes: err = 2.64  (good)
  - 4-nbr nodes: err = 42.4  (still bad)
  - 6+ nodes:    err = 0.001  (perfect)

Problem: Linear extrapolation u_ghost = 2*u_i - u_nbr is only exact for
linear fields. For our quadratic test function u=x², the ghost value is wrong.

Strategies tested:
  A. BASELINE:       Current ghost (linear extrap, reflect nearest)
  B. QUADRATIC EXTRAP: Use 2-hop information for better field extrapolation
  C. MORE GHOSTS:    Add ghosts until every node has ≥7 neighbors
  D. NEIGHBOR-OF-NEIGHBOR: Extend stencil by adding 2-hop neighbors directly
  E. HYBRID:         Ghost nodes + use actual quadratic fit from neighbors
                     to extrapolate field value at ghost position

Usage (notebook-friendly):
    import sys
    sys.argv = ['ghost_refinement.py']
    exec(open('ghost_refinement.py').read())
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
OUTPUT_DIR = "./ghost_node_refinement"
MAX_SIMS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SHARED UTILITIES
# ============================================================

def get_neighbor_counts(edge_index, N):
    row, col = edge_index
    counts = torch.zeros(N, dtype=torch.long)
    counts.index_add_(0, row, torch.ones(row.shape[0], dtype=torch.long))
    return counts


def build_adjacency(edge_index):
    """Build adjacency dict from edge_index."""
    adj = defaultdict(list)
    row_np, col_np = edge_index[0].numpy(), edge_index[1].numpy()
    for e in range(len(row_np)):
        adj[row_np[e]].append(col_np[e])
    return adj


def mls_laplacian(pos, edge_index, u_field):
    """Standard MLS Laplacian computation."""
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
# METHOD A: BASELINE GHOST (linear extrapolation, current)
# ============================================================
def ghost_baseline(pos, edge_index, u_field, min_neighbors=6):
    """Current ghost approach: reflect nearest, linear extrap."""
    N = pos.shape[0]
    adj = build_adjacency(edge_index)
    counts = get_neighbor_counts(edge_index, N)
    pos_np = pos.numpy()
    u_np = u_field.squeeze().numpy()

    ghost_rows, ghost_cols = [], []
    ghost_pos_list, ghost_u_list = [], []
    ghost_idx = N

    for node_i in range(N):
        nc = int(counts[node_i])
        if nc >= min_neighbors:
            continue
        needed = min_neighbors - nc
        neighbors = adj[node_i]
        if len(neighbors) == 0:
            continue

        pi = pos_np[node_i]
        neighbor_pos = pos_np[neighbors]
        dists = np.linalg.norm(neighbor_pos - pi, axis=1)
        sorted_idx = np.argsort(dists)

        for k in range(min(needed, len(neighbors))):
            nbr = neighbors[sorted_idx[k]]
            p_ghost = 2.0 * pi - pos_np[nbr]
            u_ghost = 2.0 * u_np[node_i] - u_np[nbr]  # linear extrap

            ghost_rows.append(node_i)
            ghost_cols.append(ghost_idx)
            ghost_pos_list.append(p_ghost)
            ghost_u_list.append(u_ghost)
            ghost_idx += 1

    if len(ghost_pos_list) == 0:
        return mls_laplacian(pos, edge_index, u_field)

    pos_aug = torch.cat([pos, torch.tensor(np.array(ghost_pos_list), dtype=torch.float32)])
    u_aug = torch.cat([u_field, torch.tensor(np.array(ghost_u_list), dtype=torch.float32).unsqueeze(1)])
    edge_aug = torch.cat([
        edge_index,
        torch.stack([torch.tensor(ghost_rows, dtype=torch.long),
                     torch.tensor(ghost_cols, dtype=torch.long)])
    ], dim=1)

    return mls_laplacian(pos_aug, edge_aug, u_aug)[:N]


# ============================================================
# METHOD B: QUADRATIC EXTRAPOLATION
# ============================================================
def ghost_quadratic_extrap(pos, edge_index, u_field, min_neighbors=6):
    """
    Ghost nodes with quadratic field extrapolation.
    Instead of u_ghost = 2*u_i - u_nbr (linear),
    fit a local quadratic to all neighbors and evaluate at ghost position.
    """
    N = pos.shape[0]
    adj = build_adjacency(edge_index)
    counts = get_neighbor_counts(edge_index, N)
    pos_np = pos.numpy()
    u_np = u_field.squeeze().numpy()

    ghost_rows, ghost_cols = [], []
    ghost_pos_list, ghost_u_list = [], []
    ghost_idx = N

    for node_i in range(N):
        nc = int(counts[node_i])
        if nc >= min_neighbors:
            continue
        needed = min_neighbors - nc
        neighbors = adj[node_i]
        if len(neighbors) == 0:
            continue

        pi = pos_np[node_i]
        neighbor_pos = pos_np[neighbors]
        neighbor_u = u_np[neighbors]
        dists = np.linalg.norm(neighbor_pos - pi, axis=1)
        sorted_idx = np.argsort(dists)

        # Fit local polynomial using ALL neighbors of node_i
        # f(dx,dy) = a + b*dx + c*dy + d*dx² + e*dy² + f*dx*dy
        # where dx = x - x_i, dy = y - y_i
        dx = neighbor_pos[:, 0] - pi[0]
        dy = neighbor_pos[:, 1] - pi[1]
        du = neighbor_u - u_np[node_i]

        if nc >= 5:
            # Full quadratic fit
            A = np.column_stack([dx, dy, dx*dy, dx**2, dy**2])
        elif nc >= 2:
            # Linear fit only
            A = np.column_stack([dx, dy])
        else:
            # Can't fit anything
            for k in range(min(needed, len(neighbors))):
                nbr = neighbors[sorted_idx[k]]
                p_ghost = 2.0 * pi - pos_np[nbr]
                u_ghost = 2.0 * u_np[node_i] - u_np[nbr]
                ghost_rows.append(node_i)
                ghost_cols.append(ghost_idx)
                ghost_pos_list.append(p_ghost)
                ghost_u_list.append(u_ghost)
                ghost_idx += 1
            continue

        # Solve least squares
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, du, rcond=None)
        except:
            coeffs = np.zeros(A.shape[1])

        for k in range(min(needed, len(neighbors))):
            nbr = neighbors[sorted_idx[k]]
            p_ghost = 2.0 * pi - pos_np[nbr]

            # Evaluate fitted polynomial at ghost position
            gdx = p_ghost[0] - pi[0]
            gdy = p_ghost[1] - pi[1]

            if len(coeffs) == 5:
                u_extrap = u_np[node_i] + (coeffs[0]*gdx + coeffs[1]*gdy +
                                            coeffs[2]*gdx*gdy + coeffs[3]*gdx**2 +
                                            coeffs[4]*gdy**2)
            else:
                u_extrap = u_np[node_i] + coeffs[0]*gdx + coeffs[1]*gdy

            ghost_rows.append(node_i)
            ghost_cols.append(ghost_idx)
            ghost_pos_list.append(p_ghost)
            ghost_u_list.append(float(u_extrap))
            ghost_idx += 1

    if len(ghost_pos_list) == 0:
        return mls_laplacian(pos, edge_index, u_field)

    pos_aug = torch.cat([pos, torch.tensor(np.array(ghost_pos_list), dtype=torch.float32)])
    u_aug = torch.cat([u_field, torch.tensor(np.array(ghost_u_list), dtype=torch.float32).unsqueeze(1)])
    edge_aug = torch.cat([
        edge_index,
        torch.stack([torch.tensor(ghost_rows, dtype=torch.long),
                     torch.tensor(ghost_cols, dtype=torch.long)])
    ], dim=1)

    return mls_laplacian(pos_aug, edge_aug, u_aug)[:N]


# ============================================================
# METHOD C: NEIGHBOR-OF-NEIGHBOR STENCIL EXTENSION
# ============================================================
def stencil_extension(pos, edge_index, u_field, min_neighbors=6):
    """
    Instead of creating ghost nodes, extend the stencil by adding
    2-hop neighbors (neighbors of neighbors) directly.
    No field extrapolation needed — uses real node values.
    """
    N = pos.shape[0]
    adj = build_adjacency(edge_index)
    counts = get_neighbor_counts(edge_index, N)

    extra_rows, extra_cols = [], []

    for node_i in range(N):
        nc = int(counts[node_i])
        if nc >= min_neighbors:
            continue
        needed = min_neighbors - nc
        neighbors = set(adj[node_i])

        # Collect 2-hop neighbors (not already direct neighbors, not self)
        two_hop = set()
        for nbr in neighbors:
            for nbr2 in adj[nbr]:
                if nbr2 != node_i and nbr2 not in neighbors:
                    two_hop.add(nbr2)

        if len(two_hop) == 0:
            continue

        # Sort by distance, take nearest
        pos_np = pos.numpy()
        pi = pos_np[node_i]
        two_hop_list = list(two_hop)
        two_hop_pos = pos_np[two_hop_list]
        dists = np.linalg.norm(two_hop_pos - pi, axis=1)
        sorted_idx = np.argsort(dists)

        for k in range(min(needed, len(two_hop_list))):
            nbr2 = two_hop_list[sorted_idx[k]]
            extra_rows.append(node_i)
            extra_cols.append(nbr2)

    if len(extra_rows) == 0:
        return mls_laplacian(pos, edge_index, u_field)

    edge_aug = torch.cat([
        edge_index,
        torch.stack([torch.tensor(extra_rows, dtype=torch.long),
                     torch.tensor(extra_cols, dtype=torch.long)])
    ], dim=1)

    return mls_laplacian(pos, edge_aug, u_field)[:N]


# ============================================================
# METHOD D: GHOST + QUADRATIC + MORE GHOSTS (aggressive)
# ============================================================
def ghost_aggressive(pos, edge_index, u_field, target_neighbors=8):
    """
    Aggressive ghost strategy:
    - Target 8 neighbors (well overdetermined)
    - Use ALL neighbors for reflection (not just nearest)
    - Quadratic extrapolation for field values
    - Also add midpoint ghosts between existing neighbors
    """
    N = pos.shape[0]
    adj = build_adjacency(edge_index)
    counts = get_neighbor_counts(edge_index, N)
    pos_np = pos.numpy()
    u_np = u_field.squeeze().numpy()

    ghost_rows, ghost_cols = [], []
    ghost_pos_list, ghost_u_list = [], []
    ghost_idx = N

    for node_i in range(N):
        nc = int(counts[node_i])
        if nc >= target_neighbors:
            continue
        needed = target_neighbors - nc
        neighbors = adj[node_i]
        if len(neighbors) == 0:
            continue

        pi = pos_np[node_i]
        neighbor_pos = pos_np[neighbors]
        neighbor_u = u_np[neighbors]

        # Fit local polynomial for extrapolation
        dx = neighbor_pos[:, 0] - pi[0]
        dy = neighbor_pos[:, 1] - pi[1]
        du = neighbor_u - u_np[node_i]

        if nc >= 3:
            # At least linear + cross term
            n_terms = min(5, nc - 1)
            if n_terms >= 5:
                A = np.column_stack([dx, dy, dx*dy, dx**2, dy**2])
            elif n_terms >= 3:
                A = np.column_stack([dx, dy, dx*dy])
            else:
                A = np.column_stack([dx, dy])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, du, rcond=None)
            except:
                coeffs = np.zeros(A.shape[1])
        else:
            coeffs = None

        def extrapolate_u(p_ghost):
            if coeffs is None:
                # Fall back to nearest neighbor
                d = np.linalg.norm(neighbor_pos - p_ghost, axis=1)
                return u_np[neighbors[np.argmin(d)]]
            gdx = p_ghost[0] - pi[0]
            gdy = p_ghost[1] - pi[1]
            val = u_np[node_i]
            if len(coeffs) >= 2:
                val += coeffs[0]*gdx + coeffs[1]*gdy
            if len(coeffs) >= 3:
                val += coeffs[2]*gdx*gdy
            if len(coeffs) >= 5:
                val += coeffs[3]*gdx**2 + coeffs[4]*gdy**2
            return val

        ghosts_added = 0

        # Strategy 1: Reflect each neighbor
        for nbr_idx in range(len(neighbors)):
            if ghosts_added >= needed:
                break
            nbr = neighbors[nbr_idx]
            p_ghost = 2.0 * pi - pos_np[nbr]
            u_ghost = extrapolate_u(p_ghost)

            ghost_rows.append(node_i)
            ghost_cols.append(ghost_idx)
            ghost_pos_list.append(p_ghost)
            ghost_u_list.append(float(u_ghost))
            ghost_idx += 1
            ghosts_added += 1

        # Strategy 2: Midpoints between consecutive neighbors (rotated)
        if ghosts_added < needed and len(neighbors) >= 2:
            # Sort neighbors by angle
            angles = np.arctan2(neighbor_pos[:, 1] - pi[1],
                                neighbor_pos[:, 0] - pi[0])
            angle_order = np.argsort(angles)

            for k in range(len(angle_order)):
                if ghosts_added >= needed:
                    break
                n1 = angle_order[k]
                n2 = angle_order[(k + 1) % len(angle_order)]
                p_mid = (neighbor_pos[n1] + neighbor_pos[n2]) / 2.0
                # Reflect midpoint across node_i
                p_ghost = 2.0 * pi - p_mid
                u_ghost = extrapolate_u(p_ghost)

                ghost_rows.append(node_i)
                ghost_cols.append(ghost_idx)
                ghost_pos_list.append(p_ghost)
                ghost_u_list.append(float(u_ghost))
                ghost_idx += 1
                ghosts_added += 1

    if len(ghost_pos_list) == 0:
        return mls_laplacian(pos, edge_index, u_field)

    pos_aug = torch.cat([pos, torch.tensor(np.array(ghost_pos_list), dtype=torch.float32)])
    u_aug = torch.cat([u_field, torch.tensor(np.array(ghost_u_list), dtype=torch.float32).unsqueeze(1)])
    edge_aug = torch.cat([
        edge_index,
        torch.stack([torch.tensor(ghost_rows, dtype=torch.long),
                     torch.tensor(ghost_cols, dtype=torch.long)])
    ], dim=1)

    return mls_laplacian(pos_aug, edge_aug, u_aug)[:N]


# ============================================================
# MAIN
# ============================================================
def run_comparison():
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)

    sim_files = sorted(glob(os.path.join(DATA_DIR, "train", "*.pt")))
    sim_files += sorted(glob(os.path.join(DATA_DIR, "test", "*.pt")))
    sim_files = sim_files[:MAX_SIMS]

    print(f"Testing {len(sim_files)} simulations")
    print(f"{'='*90}")

    methods = {
        'A: Baseline Ghost':        lambda p, e, u: ghost_baseline(p, e, u, min_neighbors=6),
        'B: Quadratic Extrap':      lambda p, e, u: ghost_quadratic_extrap(p, e, u, min_neighbors=6),
        'C: 2-Hop Stencil':         lambda p, e, u: stencil_extension(p, e, u, min_neighbors=6),
        'D: Aggressive Ghost (→8)': lambda p, e, u: ghost_aggressive(p, e, u, target_neighbors=8),
        'E: MLS Raw (reference)':   lambda p, e, u: mls_laplacian(p, e, u),
    }

    results = {name: defaultdict(list) for name in methods}
    TRUE_LAP = 2.0

    for sim_path in sim_files:
        sim_name = Path(sim_path).stem
        sim = torch.load(sim_path, weights_only=False)
        data0 = sim[0]
        pos = data0.pos.float()
        edge_index = data0.edge_index
        N = pos.shape[0]
        counts = get_neighbor_counts(edge_index, N).numpy()

        u_field = pos[:, 0:1] ** 2

        print(f"\n{sim_name}: N={N}, 3-nbr={(counts==3).sum()}, "
              f"4-nbr={(counts==4).sum()}, 5-nbr={(counts==5).sum()}")

        print(f"  {'Method':<28} {'3-nbr':>10} {'4-nbr':>10} {'5-nbr':>10} {'6+-nbr':>10}")
        print(f"  {'-'*70}")

        for name, method_fn in methods.items():
            lap = method_fn(pos, edge_index, u_field)
            err = np.abs(lap - TRUE_LAP)

            for c in np.unique(counts):
                mask = counts == c
                results[name][int(c)].extend(err[mask].tolist())

            e3 = err[counts == 3].mean() if (counts == 3).sum() > 0 else 0
            e4 = err[counts == 4].mean() if (counts == 4).sum() > 0 else 0
            e5 = err[counts == 5].mean() if (counts == 5).sum() > 0 else 0
            e6p = err[counts >= 6].mean()
            print(f"  {name:<28} {e3:>10.4f} {e4:>10.4f} {e5:>10.4f} {e6p:>10.6f}")

    # ============================================================
    # AGGREGATE
    # ============================================================
    print(f"\n{'='*90}")
    print("AGGREGATE ACROSS ALL SIMULATIONS")
    print(f"{'='*90}")

    header = f"{'Method':<28}"
    for c in [3, 4, 5, 6]:
        header += f" {c:>10}-nbr"
    print(header)
    print("-" * 80)

    for name in methods:
        line = f"{name:<28}"
        for c in [3, 4, 5, 6]:
            vals = results[name].get(c, [])
            if vals:
                mean_err = np.mean(vals)
                if mean_err > 10:
                    line += f" {mean_err:>12.1f}"
                elif mean_err > 0.1:
                    line += f" {mean_err:>12.4f}"
                else:
                    line += f" {mean_err:>12.6f}"
            else:
                line += f" {'N/A':>12}"
        print(line)

    # ============================================================
    # IMPROVEMENT RATIOS vs RAW MLS
    # ============================================================
    print(f"\n{'='*90}")
    print("IMPROVEMENT RATIO vs RAW MLS (higher = better)")
    print(f"{'='*90}")

    raw_name = 'E: MLS Raw (reference)'
    for c in [3, 4, 5]:
        raw_err = np.mean(results[raw_name].get(c, [1]))
        print(f"\n  {c}-neighbor nodes (raw err = {raw_err:.2f}):")
        for name in methods:
            if name == raw_name:
                continue
            method_err = np.mean(results[name].get(c, [1]))
            ratio = raw_err / (method_err + 1e-10)
            bar = "█" * min(50, int(ratio))
            print(f"    {name:<28}: {ratio:>8.1f}x  {bar}")

    # ============================================================
    # VISUALIZATION
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    method_names = list(methods.keys())
    colors = ['blue', 'green', 'orange', 'purple', 'red']

    for ax_idx, nc in enumerate([3, 4, 5]):
        ax = axes[ax_idx]
        for i, name in enumerate(method_names):
            vals = results[name].get(nc, [])
            if vals:
                vals_arr = np.array(vals)
                # Use log-scale violin-like representation
                ax.bar(i, np.mean(vals_arr), color=colors[i], alpha=0.7,
                       label=name if ax_idx == 0 else None)
                ax.errorbar(i, np.mean(vals_arr),
                           yerr=[[0], [np.std(vals_arr)]],
                           color='black', capsize=5)
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels([n.split(':')[0] for n in method_names],
                           rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Laplacian Error')
        ax.set_title(f'{nc}-Neighbor Nodes')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.3, label='True value')

    axes[0].legend(fontsize=7, loc='upper right')
    plt.suptitle('Ghost Node Refinement: Laplacian Error by Method',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_fig = os.path.join(OUTPUT_DIR, 'ghost_refinement_comparison.png')
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {out_fig}")

    # ============================================================
    # RECOMMENDATION
    # ============================================================
    print(f"\n{'='*90}")
    print("RECOMMENDATION")
    print(f"{'='*90}")

    for c in [3, 4]:
        best = min(methods.keys(),
                   key=lambda n: np.mean(results[n].get(c, [1e10])))
        err = np.mean(results[best].get(c, [0]))
        print(f"  Best at {c}-nbr: {best} (err={err:.4f})")

    # Check 6+ preservation
    print(f"\n  6+-neighbor accuracy (must stay ~0.001):")
    for name in methods:
        e6 = np.mean(results[name].get(6, [0]))
        status = "✓" if e6 < 0.01 else "✗"
        print(f"    {status} {name:<28}: {e6:.6f}")


# ============================================================
# Entry point — works in both script and notebook
# ============================================================
if __name__ == "__main__":
    run_comparison()
else:
    # Notebook: just call run_comparison() in a cell
    pass