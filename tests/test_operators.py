#!/usr/bin/env python3
"""
G-PARC Operator Tests
=====================
Tests for shared differential operators in differentiator/hop.py.
These underpin all models — if they break, everything breaks.

Usage:
    python tests/test_operators.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch_geometric.data import Data
import traceback

# ============================================================
# Helpers
# ============================================================

def make_grid(nx=8, ny=8):
    """Uniform Cartesian grid with cardinal (4-neighbor) edges."""
    N = nx * ny
    pos = torch.zeros(N, 2)
    for i in range(N):
        pos[i, 0] = (i % nx) / (nx - 1)
        pos[i, 1] = (i // nx) / (ny - 1)
    
    edges = []
    for i in range(N):
        r, c = i // nx, i % nx
        if r > 0:      edges.append([i, i - nx])
        if r < ny - 1: edges.append([i, i + nx])
        if c > 0:      edges.append([i, i - 1])
        if c < nx - 1: edges.append([i, i + 1])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(pos=pos, edge_index=edge_index)
    data.num_nodes = N
    return pos, edge_index, N, data


def interior_mask(N, nx, ny):
    """Boolean mask for interior nodes (not on boundary)."""
    mask = torch.zeros(N, dtype=torch.bool)
    for i in range(N):
        r, c = i // nx, i % nx
        if 0 < r < ny - 1 and 0 < c < nx - 1:
            mask[i] = True
    return mask


passed = 0
failed = 0
errors = []

def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()
        errors.append(name)
        failed += 1


# ============================================================
# Gradient Tests
# ============================================================

def test_gradient_linear():
    """MLS gradient of u=x should be (1, 0) everywhere."""
    from differentiator.hop import SolveGradientsLST
    pos, edge_index, N, data = make_grid()
    solver = SolveGradientsLST()
    
    u = pos[:, 0:1]
    grads = solver.solve_single_variable(pos, edge_index, u)
    
    err_x = (grads[:, 0] - 1.0).abs().max().item()
    err_y = grads[:, 1].abs().max().item()
    assert err_x < 1e-3, f"∂x/∂x error {err_x}"
    assert err_y < 1e-3, f"∂x/∂y error {err_y}"


def test_gradient_quadratic():
    """MLS gradient of u=x² should be (2x, 0)."""
    from differentiator.hop import SolveGradientsLST
    pos, edge_index, N, data = make_grid()
    solver = SolveGradientsLST()
    
    u = pos[:, 0:1] ** 2
    grads = solver.solve_single_variable(pos, edge_index, u)
    
    expected_gx = 2 * pos[:, 0]
    err = (grads[:, 0] - expected_gx).abs().mean().item()
    assert err < 0.05, f"∂(x²)/∂x mean error {err}"


def test_gradient_multivariable():
    """Gradient solver forward() handles multi-column fields."""
    from differentiator.hop import SolveGradientsLST
    pos, edge_index, N, data = make_grid()
    solver = SolveGradientsLST()
    
    field = torch.cat([pos[:, 0:1], pos[:, 1:2]], dim=1)  # [N, 2]
    grads = solver(data, field)  # list of 2 gradient tensors
    
    assert len(grads) == 2, f"Expected 2 gradient components, got {len(grads)}"
    assert grads[0].shape == (N, 2), f"Grad shape {grads[0].shape}"


def test_gradient_cache():
    """Gradient solver caches geometry — second call should be faster/same result."""
    from differentiator.hop import SolveGradientsLST
    pos, edge_index, N, data = make_grid()
    solver = SolveGradientsLST()
    
    u = pos[:, 0:1]
    g1 = solver.solve_single_variable(pos, edge_index, u, cache_key="test")
    g2 = solver.solve_single_variable(pos, edge_index, u, cache_key="test")
    assert torch.allclose(g1, g2), "Cached vs uncached results differ"


# ============================================================
# FD Laplacian Tests
# ============================================================

def test_fd_laplacian_quadratic():
    """FD Laplacian of u=x²+y² should be 4.0 at interior nodes."""
    from differentiator.hop import FDLaplacian
    pos, edge_index, N, data = make_grid(8, 8)
    
    u = pos[:, 0:1]**2 + pos[:, 1:2]**2
    fd = FDLaplacian()
    lap = fd(u, data)
    
    mask = interior_mask(N, 8, 8)
    err = (lap[mask, 0] - 4.0).abs().max().item()
    assert err < 1e-4, f"FD Laplacian interior error {err}"


def test_fd_laplacian_linear():
    """FD Laplacian of u=x should be 0 everywhere."""
    from differentiator.hop import FDLaplacian
    pos, edge_index, N, data = make_grid(8, 8)
    
    u = pos[:, 0:1]
    fd = FDLaplacian()
    lap = fd(u, data)
    
    mask = interior_mask(N, 8, 8)
    err = lap[mask, 0].abs().max().item()
    assert err < 1e-4, f"FD Laplacian of linear field = {err} (should be 0)"


def test_fd_laplacian_multichannel():
    """FD Laplacian handles multi-channel input."""
    from differentiator.hop import FDLaplacian
    pos, edge_index, N, data = make_grid(8, 8)
    
    u = torch.randn(N, 3)
    fd = FDLaplacian()
    lap = fd(u, data)
    assert lap.shape == (N, 3), f"Multi-channel shape {lap.shape}"


def test_fd_laplacian_cache():
    """FD Laplacian caches h² across calls."""
    from differentiator.hop import FDLaplacian
    pos, edge_index, N, data = make_grid()
    
    fd = FDLaplacian()
    u = pos[:, 0:1]**2
    _ = fd(u, data)
    assert len(fd._h_sq_cache) == 1, "Cache should have one entry"
    _ = fd(u, data)
    assert len(fd._h_sq_cache) == 1, "Cache should still have one entry"


# ============================================================
# MLS Laplacian Tests
# ============================================================

def test_mls_laplacian_with_2hop():
    """MLS Laplacian with 2-hop on unstructured-like grid doesn't crash."""
    from differentiator.hop import SolveWeightLST2d, apply_laplacian
    pos, edge_index, N, data = make_grid(8, 8)
    
    solver = SolveWeightLST2d(use_2hop_extension=True)
    weights = solver(data)
    
    u = pos[:, 0:1]**2 + pos[:, 1:2]**2
    lap = apply_laplacian(data, u, weights)
    assert lap.shape == (N, 1), f"Shape {lap.shape}"
    assert not torch.isnan(lap).any(), "NaN in MLS Laplacian"


def test_mls_laplacian_no_2hop():
    """MLS Laplacian without 2-hop produces damped but valid output."""
    from differentiator.hop import SolveWeightLST2d, apply_laplacian
    pos, edge_index, N, data = make_grid(8, 8)
    
    solver = SolveWeightLST2d(use_2hop_extension=False)
    weights = solver(data)
    
    u = torch.randn(N, 1)
    lap = apply_laplacian(data, u, weights)
    assert lap.shape == (N, 1)
    assert not torch.isnan(lap).any(), "NaN in MLS Laplacian (no 2-hop)"


# ============================================================
# Diffusion Operator Tests
# ============================================================

def test_diffusion_fd_interface():
    """DiffusionFD has same interface as DiffusionMLS."""
    from differentiator.hop import DiffusionFD, DiffusionMLS, SolveWeightLST2d
    pos, edge_index, N, data = make_grid()
    
    u = torch.randn(N, 1)
    
    out_fd = DiffusionFD()(u, data)
    assert out_fd.shape == (N, 1)
    
    out_mls = DiffusionMLS(SolveWeightLST2d(use_2hop_extension=False))(u, data)
    assert out_mls.shape == (N, 1)


# ============================================================
# Advection Tests
# ============================================================

def test_advection_constant_velocity():
    """Advection of u=x with v=(1,0) should give ∂u/∂x = 1 everywhere."""
    from differentiator.hop import AdvectionMLS, SolveGradientsLST
    pos, edge_index, N, data = make_grid()
    
    grad_solver = SolveGradientsLST()
    adv = AdvectionMLS(grad_solver)
    
    u = pos[:, 0:1]  # u = x
    velocity = torch.ones(N, 2)
    velocity[:, 1] = 0.0  # v = (1, 0)
    
    result = adv(u, velocity, data)
    mask = interior_mask(N, 8, 8)
    err = (result[mask, 0] - 1.0).abs().max().item()
    assert err < 0.01, f"Advection error {err}"


def test_advection_zero_velocity():
    """Advection with zero velocity should give zero."""
    from differentiator.hop import AdvectionMLS, SolveGradientsLST
    pos, edge_index, N, data = make_grid()
    
    grad_solver = SolveGradientsLST()
    adv = AdvectionMLS(grad_solver)
    
    u = torch.randn(N, 1)
    velocity = torch.zeros(N, 2)
    
    result = adv(u, velocity, data)
    assert result.abs().max().item() < 1e-6, "Advection with v=0 should be 0"


# ============================================================
# Strain Tests
# ============================================================

def test_strain_mls():
    """StrainMLS produces correct number of features for 2D displacement."""
    from differentiator.hop import StrainMLS, SolveGradientsLST
    pos, edge_index, N, data = make_grid()
    
    grad_solver = SolveGradientsLST()
    strain = StrainMLS(grad_solver, use_von_mises=True, use_volumetric=True, n_dimensions=2)
    
    # 2D displacement field
    displacement = torch.randn(N, 2)
    result = strain(displacement, data)
    
    # eps_xx, eps_yy, eps_xy + von_mises + volumetric = 5
    assert result.shape == (N, 5), f"Strain shape {result.shape}, expected (N, 5)"


def test_strain_no_extras():
    """StrainMLS without von_mises/volumetric gives 3 features."""
    from differentiator.hop import StrainMLS, SolveGradientsLST
    pos, edge_index, N, data = make_grid()
    
    grad_solver = SolveGradientsLST()
    strain = StrainMLS(grad_solver, use_von_mises=False, use_volumetric=False, n_dimensions=2)
    
    displacement = torch.randn(N, 2)
    result = strain(displacement, data)
    assert result.shape == (N, 3), f"Strain shape {result.shape}, expected (N, 3)"


# ============================================================
# Neighbor Damping / 2-hop Extension
# ============================================================

def test_neighbor_damping():
    """Damping values match expected schedule."""
    from differentiator.hop import compute_neighbor_damping
    pos, edge_index, N, data = make_grid(4, 4)
    
    damping = compute_neighbor_damping(edge_index, N, min_neighbors=5)
    assert damping.shape == (N,)
    assert damping.min() >= 0.0
    assert damping.max() <= 1.0


def test_2hop_extension():
    """2-hop extension adds edges for low-neighbor nodes."""
    from differentiator.hop import compute_2hop_extension
    pos, edge_index, N, data = make_grid(4, 4)
    
    aug = compute_2hop_extension(pos, edge_index, min_neighbors=6)
    assert aug.shape[1] >= edge_index.shape[1], "Should add edges, not remove"


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("G-PARC Operator Tests")
    print("=" * 50 + "\n")
    
    print("--- Gradient Solver ---")
    run_test("Gradient linear (u=x)", test_gradient_linear)
    run_test("Gradient quadratic (u=x²)", test_gradient_quadratic)
    run_test("Gradient multi-variable", test_gradient_multivariable)
    run_test("Gradient cache consistency", test_gradient_cache)
    
    print("\n--- FD Laplacian ---")
    run_test("FD Laplacian quadratic (u=x²+y²)", test_fd_laplacian_quadratic)
    run_test("FD Laplacian linear (u=x, expect 0)", test_fd_laplacian_linear)
    run_test("FD Laplacian multi-channel", test_fd_laplacian_multichannel)
    run_test("FD Laplacian cache", test_fd_laplacian_cache)
    
    print("\n--- MLS Laplacian ---")
    run_test("MLS Laplacian with 2-hop", test_mls_laplacian_with_2hop)
    run_test("MLS Laplacian no 2-hop", test_mls_laplacian_no_2hop)
    
    print("\n--- Diffusion ---")
    run_test("DiffusionFD/MLS interface parity", test_diffusion_fd_interface)
    
    print("\n--- Advection ---")
    run_test("Advection constant velocity", test_advection_constant_velocity)
    run_test("Advection zero velocity", test_advection_zero_velocity)
    
    print("\n--- Strain ---")
    run_test("StrainMLS 2D with extras", test_strain_mls)
    run_test("StrainMLS 2D no extras", test_strain_no_extras)
    
    print("\n--- Utilities ---")
    run_test("Neighbor damping", test_neighbor_damping)
    run_test("2-hop extension", test_2hop_extension)
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print(f"Failed: {', '.join(errors)}")
    print(f"{'=' * 50}\n")
    
    sys.exit(1 if failed > 0 else 0)