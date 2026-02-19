#!/usr/bin/env python3
"""
G-PARCv2 Shock Tube — Smoke Tests
===================================
Run after any code change to verify nothing is broken.
No GPU or real data required — uses synthetic 8x8 grid.

Usage:
    python tests/test_shocktube_v2.py

Tests:
  1. FDLaplacian accuracy on known analytic field
  2. DiffusionFD / DiffusionMLS interface parity
  3. ShockTubeDifferentiator forward pass (all diffusion_type modes)
  4. GPARC_ShockTube_V2 full forward + rollout
  5. State tensor dimensions (the [72] = 2+64+3+3 layout)
  6. Gradient through full model (backprop doesn't crash)
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# ============================================================
# Synthetic 8x8 grid with cardinal connectivity
# ============================================================

def make_grid(nx=8, ny=8):
    """Create a uniform Cartesian grid with cardinal (4-neighbor) edges."""
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
    return pos, edge_index, N


def make_fake_data(pos, edge_index, N, num_dynamic_raw=4):
    """Create a fake PyG Data object mimicking shock tube format."""
    static = pos  # [N, 2]
    dynamic = torch.randn(N, num_dynamic_raw)  # density, x_mom, y_mom, energy
    x = torch.cat([static, dynamic], dim=1)  # [N, 6]
    y = torch.randn(N, num_dynamic_raw)
    
    data = Data(x=x, y=y, edge_index=edge_index, pos=pos)
    data.num_nodes = N
    data.global_pressure = torch.tensor([0.5])
    data.global_density = torch.tensor([0.3])
    data.global_delta_t = torch.tensor([0.01])
    return data


# ============================================================
# Test runner
# ============================================================

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
# Tests
# ============================================================

def test_fd_laplacian_accuracy():
    """FD Laplacian on u = x²+y² should give 4.0 at interior nodes."""
    from differentiator.hop import FDLaplacian
    
    pos, edge_index, N = make_grid(8, 8)
    mesh = Data(pos=pos, edge_index=edge_index)
    mesh.num_nodes = N
    
    u = (pos[:, 0:1]**2 + pos[:, 1:2]**2)
    
    fd = FDLaplacian()
    lap = fd(u, mesh)
    
    # Interior nodes (not on boundary) should be ~4.0
    # On 8x8 grid, interior = rows 1-6, cols 1-6 = 36 nodes
    interior_mask = torch.zeros(N, dtype=torch.bool)
    for i in range(N):
        r, c = i // 8, i % 8
        if 0 < r < 7 and 0 < c < 7:
            interior_mask[i] = True
    
    interior_lap = lap[interior_mask, 0]
    err = (interior_lap - 4.0).abs().max().item()
    assert err < 1e-4, f"Interior Laplacian error {err:.6f} > 1e-4"


def test_diffusion_fd_interface():
    """DiffusionFD has same interface as DiffusionMLS."""
    from differentiator.hop import DiffusionFD, DiffusionMLS, SolveWeightLST2d
    
    pos, edge_index, N = make_grid(8, 8)
    mesh = Data(pos=pos, edge_index=edge_index)
    mesh.num_nodes = N
    
    u = torch.randn(N, 1)
    
    # FD
    dfd = DiffusionFD()
    out_fd = dfd(u, mesh)
    assert out_fd.shape == (N, 1), f"DiffusionFD shape {out_fd.shape} != ({N}, 1)"
    
    # MLS (for interface comparison only — values will differ on this grid)
    lap_solver = SolveWeightLST2d(use_2hop_extension=False)
    dmls = DiffusionMLS(lap_solver)
    out_mls = dmls(u, mesh)
    assert out_mls.shape == (N, 1), f"DiffusionMLS shape {out_mls.shape} != ({N}, 1)"


def test_differentiator_all_modes():
    """ShockTubeDifferentiator forward pass with fd, mls, and none."""
    from differentiator.shocktubedifferentiator import ShockTubeDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    
    pos, edge_index, N = make_grid(8, 8)
    mesh = Data(pos=pos, edge_index=edge_index)
    mesh.num_nodes = N
    
    for mode in ['fd', 'mls', 'none']:
        grad_solver = SolveGradientsLST()
        lap_solver = SolveWeightLST2d(use_2hop_extension=False)
        
        fe = GraphConvFeatureExtractorV2(
            in_channels=2, hidden_channels=32, out_channels=64, num_layers=2
        )
        
        diff = ShockTubeDifferentiator(
            num_static_feats=2,
            num_dynamic_feats=3,
            feature_extractor=fe,
            gradient_solver=grad_solver,
            laplacian_solver=lap_solver,
            n_fe_features=64,
            global_embed_dim=32,
            global_param_dim=3,
            diffusion_type=mode,
        )
        diff.initialize_weights(mesh)
        
        # Build state tensor: [pos(2) | global_embed(32) | raw_global(3) | dynamic(3)] = [N, 40]
        sf, ge, gp, df = 2, 32, 3, 3
        state = torch.randn(N, sf + ge + gp + df)
        state[:, :2] = pos  # positions
        
        out = diff(state, edge_index)
        assert out.shape == (N, 3), f"mode={mode}: output shape {out.shape} != ({N}, 3)"


def test_full_model_forward():
    """GPARC_ShockTube_V2 full forward pass with synthetic data."""
    from differentiator.shocktubedifferentiator import ShockTubeDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from models.shocktube_gparcv2 import GPARC_ShockTube_V2
    
    pos, edge_index, N = make_grid(8, 8)
    
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=False)
    
    fe = GraphConvFeatureExtractorV2(
        in_channels=2, hidden_channels=32, out_channels=64, num_layers=2
    )
    
    diff = ShockTubeDifferentiator(
        num_static_feats=2, num_dynamic_feats=3,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=64, global_embed_dim=32, global_param_dim=3,
        diffusion_type='fd',
    )
    
    sample_mesh = Data(pos=pos, edge_index=edge_index)
    sample_mesh.num_nodes = N
    diff.initialize_weights(sample_mesh)
    
    model = GPARC_ShockTube_V2(
        derivative_solver_physics=diff,
        integrator_type='euler',
        num_static_feats=2, num_dynamic_feats=3,
        skip_dynamic_indices=[2],
        global_param_dim=3, global_embed_dim=32,
    )
    
    # Create a 3-step sequence
    seq = [make_fake_data(pos, edge_index, N) for _ in range(3)]
    
    preds = model(seq, teacher_forcing_ratio=0.0)
    assert len(preds) == 3, f"Expected 3 predictions, got {len(preds)}"
    for i, p in enumerate(preds):
        assert p.shape == (N, 3), f"Pred {i} shape {p.shape} != ({N}, 3)"


def test_state_tensor_layout():
    """Verify the [pos | global_embed | raw_global | dynamic] layout is [N, 72] for default dims."""
    from models.shocktube_gparcv2 import GPARC_ShockTube_V2
    from differentiator.shocktubedifferentiator import ShockTubeDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    
    pos, edge_index, N = make_grid(4, 4)
    
    fe = GraphConvFeatureExtractorV2(in_channels=2, hidden_channels=32, out_channels=128, num_layers=2)
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=False)
    
    diff = ShockTubeDifferentiator(
        num_static_feats=2, num_dynamic_feats=3,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=128, global_embed_dim=64, global_param_dim=3,
        diffusion_type='fd',
    )
    
    sample_mesh = Data(pos=pos, edge_index=edge_index)
    sample_mesh.num_nodes = N
    diff.initialize_weights(sample_mesh)
    
    # Build state tensor manually to check dimensions
    # [pos(2) + global_embed(64) + raw_global(3) + dynamic(3)] = 72
    state = torch.randn(N, 72)
    state[:, :2] = pos
    
    out = diff(state, edge_index)
    assert out.shape == (N, 3), f"Output shape {out.shape} != ({N}, 3) for state dim 72"


def test_backward_pass():
    """Verify gradients flow through the full model."""
    from differentiator.shocktubedifferentiator import ShockTubeDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from models.shocktube_gparcv2 import GPARC_ShockTube_V2
    
    pos, edge_index, N = make_grid(8, 8)
    
    fe = GraphConvFeatureExtractorV2(in_channels=2, hidden_channels=32, out_channels=64, num_layers=2)
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=False)
    
    diff = ShockTubeDifferentiator(
        num_static_feats=2, num_dynamic_feats=3,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=64, global_embed_dim=32, global_param_dim=3,
        diffusion_type='fd',
    )
    
    sample_mesh = Data(pos=pos, edge_index=edge_index)
    sample_mesh.num_nodes = N
    diff.initialize_weights(sample_mesh)
    
    model = GPARC_ShockTube_V2(
        derivative_solver_physics=diff,
        integrator_type='euler',
        num_static_feats=2, num_dynamic_feats=3,
        skip_dynamic_indices=[2],
        global_param_dim=3, global_embed_dim=32,
    )
    model.train()
    
    seq = [make_fake_data(pos, edge_index, N) for _ in range(2)]
    preds = model(seq, teacher_forcing_ratio=0.0)
    
    # Compute loss and backprop
    target = torch.randn(N, 3)
    loss = sum(F.mse_loss(p, target) for p in preds) / len(preds)
    loss.backward()
    
    # Check that at least some parameters got gradients
    grads_found = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in model.parameters())
    assert grads_found > 0, "No gradients found in any parameters"


def test_rollout():
    """Verify model.rollout() runs without error."""
    from differentiator.shocktubedifferentiator import ShockTubeDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from models.shocktube_gparcv2 import GPARC_ShockTube_V2
    
    pos, edge_index, N = make_grid(8, 8)
    
    fe = GraphConvFeatureExtractorV2(in_channels=2, hidden_channels=32, out_channels=64, num_layers=2)
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=False)
    
    diff = ShockTubeDifferentiator(
        num_static_feats=2, num_dynamic_feats=3,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=64, global_embed_dim=32, global_param_dim=3,
        diffusion_type='fd',
    )
    
    sample_mesh = Data(pos=pos, edge_index=edge_index)
    sample_mesh.num_nodes = N
    diff.initialize_weights(sample_mesh)
    
    model = GPARC_ShockTube_V2(
        derivative_solver_physics=diff,
        integrator_type='euler',
        num_static_feats=2, num_dynamic_feats=3,
        skip_dynamic_indices=[2],
        global_param_dim=3, global_embed_dim=32,
    )
    model.eval()
    
    sim = [make_fake_data(pos, edge_index, N) for _ in range(5)]
    states = model.rollout(sim, num_steps=3)
    
    assert len(states) == 4, f"Expected 4 states (initial + 3 steps), got {len(states)}"
    assert states[0].shape == (N, 3), f"State shape {states[0].shape} != ({N}, 3)"


def test_process_targets():
    """Verify process_targets correctly skips y_momentum."""
    from models.shocktube_gparcv2 import GPARC_ShockTube_V2
    
    # Minimal model just for process_targets
    model = GPARC_ShockTube_V2.__new__(GPARC_ShockTube_V2)
    model.num_dynamic_feats = 3
    model.skip_dynamic_indices = [2]
    model.num_static_feats = 2
    
    # Raw y has 4 dynamic features: [density, x_mom, y_mom, energy]
    y = torch.tensor([[1.0, 2.0, 99.0, 4.0],
                      [5.0, 6.0, 99.0, 8.0]])
    
    result = model.process_targets(y)
    expected = torch.tensor([[1.0, 2.0, 4.0],
                             [5.0, 6.0, 8.0]])
    
    assert torch.allclose(result, expected), f"process_targets wrong: {result} != {expected}"


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("G-PARCv2 Shock Tube — Smoke Tests")
    print("=" * 50 + "\n")
    
    run_test("FD Laplacian accuracy (u=x²+y², expect 4.0)", test_fd_laplacian_accuracy)
    run_test("DiffusionFD / DiffusionMLS interface parity", test_diffusion_fd_interface)
    run_test("Differentiator forward (fd, mls, none)", test_differentiator_all_modes)
    run_test("Full model forward pass", test_full_model_forward)
    run_test("State tensor layout [N, 72]", test_state_tensor_layout)
    run_test("Backward pass (gradients flow)", test_backward_pass)
    run_test("Rollout inference", test_rollout)
    run_test("process_targets skips y_momentum", test_process_targets)
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print(f"Failed: {', '.join(errors)}")
    print(f"{'=' * 50}\n")
    
    sys.exit(1 if failed > 0 else 0)
