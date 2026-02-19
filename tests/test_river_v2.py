#!/usr/bin/env python3
"""
G-PARCv2 River — Smoke Tests
==============================
No GPU or real data required.

Usage:
    python tests/test_river_v2.py
"""

import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def make_grid(nx=8, ny=8, num_static=9, num_dynamic=4):
    """Create synthetic river-like data on a grid."""
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


def make_river_data(pos, edge_index, N, num_static=9, num_dynamic=4):
    """Fake river Data object with 9 static + 4 dynamic features."""
    static = torch.randn(N, num_static)
    static[:, :2] = pos  # first 2 are positions
    dynamic = torch.randn(N, num_dynamic)
    x = torch.cat([static, dynamic], dim=1)
    y = torch.randn(N, num_dynamic)
    
    data = Data(x=x, y=y, edge_index=edge_index, pos=pos)
    data.num_nodes = N
    data.mesh_id = torch.tensor([0])
    return data


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


def test_river_differentiator_forward():
    """RiverDifferentiator forward pass."""
    from differentiator.riverdifferentiator import RiverDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    
    pos, edge_index, N = make_grid()
    mesh = Data(pos=pos, edge_index=edge_index)
    mesh.num_nodes = N
    
    fe = GraphConvFeatureExtractorV2(in_channels=9, hidden_channels=32, out_channels=64, num_layers=2)
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=False)
    
    diff = RiverDifferentiator(
        num_static_feats=9, num_dynamic_feats=4,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=64,
        velocity_indices=[2, 3],
    )
    diff.initialize_weights(mesh)
    
    state = torch.randn(N, 9 + 4)
    state[:, :2] = pos
    out = diff(state, edge_index)
    assert out.shape == (N, 4), f"Output shape {out.shape}"


def test_river_model_forward():
    """GPARC_River_V2 full forward pass."""
    from differentiator.riverdifferentiator import RiverDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from models.riverV2 import GPARC_River_V2
    
    pos, edge_index, N = make_grid()
    mesh = Data(pos=pos, edge_index=edge_index)
    mesh.num_nodes = N
    
    fe = GraphConvFeatureExtractorV2(in_channels=9, hidden_channels=32, out_channels=64, num_layers=2)
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=False)
    
    diff = RiverDifferentiator(
        num_static_feats=9, num_dynamic_feats=4,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=64,
    )
    diff.initialize_weights(mesh)
    
    model = GPARC_River_V2(
        derivative_solver_physics=diff,
        integrator_type='euler',
        num_static_feats=9, num_dynamic_feats=4,
    )
    
    seq = [make_river_data(pos, edge_index, N) for _ in range(3)]
    preds = model(seq, dt=1.0, teacher_forcing_ratio=0.0)
    
    assert len(preds) == 3
    for p in preds:
        assert p.shape == (N, 4), f"Pred shape {p.shape}"


def test_river_backward():
    """Gradients flow through river model."""
    from differentiator.riverdifferentiator import RiverDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from models.riverV2 import GPARC_River_V2
    
    pos, edge_index, N = make_grid()
    mesh = Data(pos=pos, edge_index=edge_index)
    mesh.num_nodes = N
    
    fe = GraphConvFeatureExtractorV2(in_channels=9, hidden_channels=32, out_channels=64, num_layers=2)
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=False)
    
    diff = RiverDifferentiator(
        num_static_feats=9, num_dynamic_feats=4,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=64,
    )
    diff.initialize_weights(mesh)
    
    model = GPARC_River_V2(
        derivative_solver_physics=diff, integrator_type='euler',
        num_static_feats=9, num_dynamic_feats=4,
    )
    model.train()
    
    seq = [make_river_data(pos, edge_index, N) for _ in range(2)]
    preds = model(seq, dt=1.0)
    
    loss = sum(F.mse_loss(p, torch.randn(N, 4)) for p in preds) / len(preds)
    loss.backward()
    
    grads_found = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grads_found > 0, "No gradients found"


def test_river_rollout():
    """River model rollout inference."""
    from differentiator.riverdifferentiator import RiverDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from models.riverV2 import GPARC_River_V2
    
    pos, edge_index, N = make_grid()
    mesh = Data(pos=pos, edge_index=edge_index)
    mesh.num_nodes = N
    
    fe = GraphConvFeatureExtractorV2(in_channels=9, hidden_channels=32, out_channels=64, num_layers=2)
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=False)
    
    diff = RiverDifferentiator(
        num_static_feats=9, num_dynamic_feats=4,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=64,
    )
    diff.initialize_weights(mesh)
    
    model = GPARC_River_V2(
        derivative_solver_physics=diff, integrator_type='euler',
        num_static_feats=9, num_dynamic_feats=4,
    )
    model.eval()
    
    sim = [make_river_data(pos, edge_index, N) for _ in range(5)]
    states = model.rollout(sim, num_steps=3)
    
    assert len(states) == 4, f"Expected 4 states, got {len(states)}"
    assert states[0].shape == (N, 4)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("G-PARCv2 River — Smoke Tests")
    print("=" * 50 + "\n")
    
    run_test("RiverDifferentiator forward", test_river_differentiator_forward)
    run_test("River model forward", test_river_model_forward)
    run_test("River backward pass", test_river_backward)
    run_test("River rollout", test_river_rollout)
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print(f"Failed: {', '.join(errors)}")
    print(f"{'=' * 50}\n")
    
    sys.exit(1 if failed > 0 else 0)
