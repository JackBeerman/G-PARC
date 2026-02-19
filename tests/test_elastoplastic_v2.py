#!/usr/bin/env python3
"""
G-PARCv2 Elastoplastic — Smoke Tests
======================================
No GPU or real data required.

Usage:
    python tests/test_elastoplastic_v2.py
"""

import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def make_grid(nx=8, ny=8):
    """Uniform grid mimicking PLAID mesh structure."""
    N = nx * ny
    pos = torch.zeros(N, 2)
    for i in range(N):
        pos[i, 0] = (i % nx) / (nx - 1)
        pos[i, 1] = (i // nx) / (ny - 1)
    
    edges = []
    for i in range(N):
        r, c = i // nx, i % nx
        # Cardinal + diagonal = 8 neighbors for interior
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc_ = r + dr, c + dc
                if 0 <= nr < ny and 0 <= nc_ < nx:
                    j = nr * nx + nc_
                    edges.append([i, j])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return pos, edge_index, N


def make_elasto_data(pos, edge_index, N, num_static=2, num_dynamic=2):
    """Fake elastoplastic Data: 2 static (pos) + 2 dynamic (displacement)."""
    x = torch.cat([pos, torch.randn(N, num_dynamic)], dim=1)
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


def test_elasto_differentiator_forward():
    """ElastoPlasticDifferentiator forward pass."""
    from differentiator.differentiator import ElastoPlasticDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    
    pos, edge_index, N = make_grid()
    mesh = Data(pos=pos, edge_index=edge_index)
    mesh.num_nodes = N
    
    fe = GraphConvFeatureExtractorV2(in_channels=2, hidden_channels=32, out_channels=64, num_layers=2)
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=True)
    
    diff = ElastoPlasticDifferentiator(
        num_static_feats=2, num_dynamic_feats=2,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=64,
        list_strain_idx=[0, 1],
        list_laplacian_idx=[0, 1],
    )
    diff.initialize_weights(mesh)
    
    state = torch.cat([pos, torch.randn(N, 2)], dim=1)
    out = diff(state, edge_index)
    assert out.shape == (N, 2), f"Output shape {out.shape}"


def test_elasto_model_forward():
    """GPARC_ElastoPlastic_Numerical full forward pass."""
    from differentiator.differentiator import ElastoPlasticDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from models.globalelasto import GPARC_ElastoPlastic_Numerical
    
    pos, edge_index, N = make_grid()
    mesh = Data(pos=pos, edge_index=edge_index)
    mesh.num_nodes = N
    
    fe = GraphConvFeatureExtractorV2(in_channels=2, hidden_channels=32, out_channels=64, num_layers=2)
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=True)
    
    diff = ElastoPlasticDifferentiator(
        num_static_feats=2, num_dynamic_feats=2,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=64,
        list_strain_idx=[0, 1], list_laplacian_idx=[0, 1],
    )
    diff.initialize_weights(mesh)
    
    model = GPARC_ElastoPlastic_Numerical(
        derivative_solver_physics=diff,
        integrator_type='euler',
        num_static_feats=2, num_dynamic_feats=2,
    )
    
    seq = [make_elasto_data(pos, edge_index, N) for _ in range(3)]
    preds = model(seq, dt=1.0, teacher_forcing_ratio=0.0)
    
    assert len(preds) == 3
    for p in preds:
        assert p.shape == (N, 2), f"Pred shape {p.shape}"


def test_elasto_backward():
    """Gradients flow through elastoplastic model."""
    from differentiator.differentiator import ElastoPlasticDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from models.globalelasto import GPARC_ElastoPlastic_Numerical
    
    pos, edge_index, N = make_grid()
    mesh = Data(pos=pos, edge_index=edge_index)
    mesh.num_nodes = N
    
    fe = GraphConvFeatureExtractorV2(in_channels=2, hidden_channels=32, out_channels=64, num_layers=2)
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=True)
    
    diff = ElastoPlasticDifferentiator(
        num_static_feats=2, num_dynamic_feats=2,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=64,
        list_strain_idx=[0, 1], list_laplacian_idx=[0, 1],
    )
    diff.initialize_weights(mesh)
    
    model = GPARC_ElastoPlastic_Numerical(
        derivative_solver_physics=diff, integrator_type='euler',
        num_static_feats=2, num_dynamic_feats=2,
    )
    model.train()
    
    seq = [make_elasto_data(pos, edge_index, N) for _ in range(2)]
    preds = model(seq, dt=1.0)
    
    loss = sum(F.mse_loss(p, torch.randn(N, 2)) for p in preds) / len(preds)
    loss.backward()
    
    grads_found = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grads_found > 0, "No gradients found"


def test_elasto_rollout():
    """Elastoplastic rollout inference."""
    from differentiator.differentiator import ElastoPlasticDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    from models.globalelasto import GPARC_ElastoPlastic_Numerical
    
    pos, edge_index, N = make_grid()
    mesh = Data(pos=pos, edge_index=edge_index)
    mesh.num_nodes = N
    
    fe = GraphConvFeatureExtractorV2(in_channels=2, hidden_channels=32, out_channels=64, num_layers=2)
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=True)
    
    diff = ElastoPlasticDifferentiator(
        num_static_feats=2, num_dynamic_feats=2,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=64,
        list_strain_idx=[0, 1], list_laplacian_idx=[0, 1],
    )
    diff.initialize_weights(mesh)
    
    model = GPARC_ElastoPlastic_Numerical(
        derivative_solver_physics=diff, integrator_type='euler',
        num_static_feats=2, num_dynamic_feats=2,
    )
    model.eval()
    
    sim = [make_elasto_data(pos, edge_index, N) for _ in range(5)]
    states = model.rollout(sim, num_steps=3)
    
    assert len(states) == 4, f"Expected 4 states, got {len(states)}"
    assert states[0].shape == (N, 2)


def test_elasto_dirichlet_bc():
    """Dirichlet BC enforcement freezes left-wall nodes."""
    from models.globalelasto import GPARC_ElastoPlastic_Numerical
    from differentiator.differentiator import ElastoPlasticDifferentiator
    from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
    from utilities.featureextractor import GraphConvFeatureExtractorV2
    
    pos, edge_index, N = make_grid()
    mesh = Data(pos=pos, edge_index=edge_index)
    mesh.num_nodes = N
    
    fe = GraphConvFeatureExtractorV2(in_channels=2, hidden_channels=32, out_channels=64, num_layers=2)
    grad_solver = SolveGradientsLST()
    lap_solver = SolveWeightLST2d(use_2hop_extension=True)
    
    diff = ElastoPlasticDifferentiator(
        num_static_feats=2, num_dynamic_feats=2,
        feature_extractor=fe,
        gradient_solver=grad_solver, laplacian_solver=lap_solver,
        n_fe_features=64,
        list_strain_idx=[0, 1], list_laplacian_idx=[0, 1],
    )
    diff.initialize_weights(mesh)
    
    # pos_mean/std so boundary detection works
    # Positions are [0,1], so physical coords = pos * std + mean
    # Set boundary_threshold = 0.1 and use identity normalization
    model = GPARC_ElastoPlastic_Numerical(
        derivative_solver_physics=diff, integrator_type='euler',
        num_static_feats=2, num_dynamic_feats=2,
        pos_mean=[0.0, 0.0], pos_std=[1.0, 1.0],
        boundary_threshold=0.05,
    )
    
    static = pos
    F_current = torch.ones(N, 2)  # all ones
    F_next = torch.randn(N, 2)    # random
    
    result = model._enforce_dirichlet_bc(F_next, F_current, static)
    
    # Left wall: x < 0.05 → should keep F_current (ones)
    left_mask = pos[:, 0] < 0.05
    if left_mask.any():
        assert torch.allclose(result[left_mask], F_current[left_mask]), \
            "Left wall should be frozen to F_current"


def test_elasto_process_targets():
    """process_targets correctly handles skip_dynamic_indices."""
    from models.globalelasto import GPARC_ElastoPlastic_Numerical
    
    model = GPARC_ElastoPlastic_Numerical.__new__(GPARC_ElastoPlastic_Numerical)
    model.num_dynamic_feats = 2
    model.skip_dynamic_indices = []
    
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = model.process_targets(y)
    assert torch.allclose(result, y), "No skip should return identity"


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("G-PARCv2 Elastoplastic — Smoke Tests")
    print("=" * 50 + "\n")
    
    run_test("ElastoPlastic differentiator forward", test_elasto_differentiator_forward)
    run_test("ElastoPlastic model forward", test_elasto_model_forward)
    run_test("ElastoPlastic backward pass", test_elasto_backward)
    run_test("ElastoPlastic rollout", test_elasto_rollout)
    run_test("Dirichlet BC enforcement", test_elasto_dirichlet_bc)
    run_test("process_targets", test_elasto_process_targets)
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print(f"Failed: {', '.join(errors)}")
    print(f"{'=' * 50}\n")
    
    sys.exit(1 if failed > 0 else 0)
