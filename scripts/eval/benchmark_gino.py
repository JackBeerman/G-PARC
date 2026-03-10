#!/usr/bin/env python3
"""
GNO / GINO Memory & Parameter Benchmark
=========================================
Loads real G-PARC test data to get exact mesh dimensions, then
instantiates GNO and GINO and measures parameter count + peak GPU
memory for a single forward pass.

GNO (Li et al. 2020): stacked kernel integral transforms on point clouds.
GINO (Li et al. 2023): input GNO -> FNO (latent grid) -> output GNO.
FNO-GNO excluded: requires pre-gridded data, cannot handle irregular meshes.

Usage:
    python benchmark_gino.py \
        --st_test_dir /path/to/shocktube/test \
        --el_test_dir /path/to/elasto/test \
        --rv_test_dir /path/to/river/test

Output: gino_benchmark_results.json
"""

import argparse
import json
import os
import traceback

import torch
import torch.nn.functional as F

from neuralop.models.gino import GINO
from neuralop.layers.gno_block import GNOBlock
from neuralop.layers.channel_mlp import ChannelMLP


# ============================================================
# MODEL CREATION
# ============================================================

class GNO(torch.nn.Module):
    """
    Graph Neural Operator (Li et al. 2020) built from GNOBlock layers.
    Architecture: Lifting MLP -> N x GNOBlock -> Projection MLP
    Operates directly on irregular point clouds via kernel integral transforms.
    """
    def __init__(self, in_channels, out_channels, coord_dim=2,
                 hidden_channels=64, n_layers=4, radius=0.033,
                 channel_mlp_hidden_layers=[128, 128],
                 use_open3d=False, use_torch_scatter=False):
        super().__init__()
        self.lifting = ChannelMLP(
            in_channels=in_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels * 2,
            n_layers=2,
            n_dim=1,
        )
        self.gno_layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.gno_layers.append(GNOBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                coord_dim=coord_dim,
                radius=radius,
                channel_mlp_layers=channel_mlp_hidden_layers,
                use_open3d_neighbor_search=use_open3d,
                use_torch_scatter_reduce=use_torch_scatter,
            ))
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels * 2,
            n_layers=2,
            n_dim=1,
        )

    def forward(self, pos, x):
        """
        pos: (n_nodes, coord_dim) — node positions
        x:   (batch, n_nodes, in_channels) — input features
        """
        # Lifting: (batch, n_nodes, in_ch) -> (batch, n_nodes, hidden)
        h = self.lifting(x.permute(0, 2, 1)).permute(0, 2, 1)
        # GNO layers
        for gno in self.gno_layers:
            h_new = gno(y=pos, x=pos, f_y=h)
            h = h + h_new  # residual
        # Projection: (batch, n_nodes, hidden) -> (batch, n_nodes, out_ch)
        out = self.projection(h.permute(0, 2, 1)).permute(0, 2, 1)
        return out


def make_gno(in_ch, out_ch, coord_dim=2, gno_radius=0.033):
    return GNO(
        in_channels=in_ch,
        out_channels=out_ch,
        coord_dim=coord_dim,
        hidden_channels=64,
        n_layers=4,
        radius=gno_radius,
        channel_mlp_hidden_layers=[128, 128],
        use_open3d=False,
        use_torch_scatter=False,
    )


# ============================================================
# DATA LOADING — extract graph dimensions from real test data
# ============================================================

def load_sample(test_dir):
    """Load first .pt file from a test directory and extract dimensions."""
    pt_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.pt')])
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {test_dir}")

    data = torch.load(os.path.join(test_dir, pt_files[0]), weights_only=False)

    # Handle different data formats
    if isinstance(data, list):
        # List of PyG Data objects (one per timestep)
        sample = data[0]
    elif isinstance(data, dict):
        if 'frames' in data:
            sample = data['frames'][0]
        else:
            sample = data
    else:
        sample = data

    info = {}

    # Node count
    if hasattr(sample, 'x') and sample.x is not None:
        info['n_nodes'] = sample.x.size(0)
        info['x_channels'] = sample.x.size(1)
    elif hasattr(sample, 'pos') and sample.pos is not None:
        info['n_nodes'] = sample.pos.size(0)
    else:
        raise ValueError("Cannot determine node count from sample")

    # Edge count
    if hasattr(sample, 'edge_index') and sample.edge_index is not None:
        info['n_edges'] = sample.edge_index.size(1)
    else:
        info['n_edges'] = 0

    # Position dimensionality
    if hasattr(sample, 'pos') and sample.pos is not None:
        info['coord_dim'] = sample.pos.size(1)
    else:
        info['coord_dim'] = 2  # default

    # Y (target) channels
    if hasattr(sample, 'y') and sample.y is not None:
        info['y_channels'] = sample.y.size(1) if sample.y.dim() > 1 else 1
    else:
        info['y_channels'] = None

    info['file'] = pt_files[0]
    info['sample'] = sample

    return info


def describe_dataset(name, info):
    print(f"  {name}:")
    print(f"    File: {info['file']}")
    print(f"    Nodes: {info['n_nodes']:,}")
    print(f"    Edges: {info['n_edges']:,}")
    print(f"    x channels: {info.get('x_channels', '?')}")
    print(f"    y channels: {info.get('y_channels', '?')}")
    print(f"    coord_dim: {info['coord_dim']}")
    print(f"    Avg degree: {info['n_edges'] / max(info['n_nodes'], 1):.1f}")


# ============================================================
# MODEL CREATION
# ============================================================

def make_gino(in_ch, out_ch, coord_dim=2, grid_res=32, gno_radius=0.033,
              fno_hidden=64, fno_layers=4):
    return GINO(
        in_channels=in_ch,
        out_channels=out_ch,
        gno_coord_dim=coord_dim,
        fno_in_channels=in_ch,
        fno_n_modes=tuple([16] * coord_dim),
        fno_hidden_channels=fno_hidden,
        fno_n_layers=fno_layers,
        in_gno_radius=gno_radius,
        out_gno_radius=gno_radius,
        in_gno_transform_type='nonlinear_kernelonly',
        in_gno_channel_mlp_hidden_layers=[80, 80, 80],
        out_gno_channel_mlp_hidden_layers=[128, 64],
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
    )


# ============================================================
# BENCHMARK
# ============================================================

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    breakdown = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n > 0:
            breakdown[name] = {
                'params': n,
                'pct': round(100.0 * n / max(total, 1), 1),
            }

    return total, round(size_mb, 2), breakdown


def try_forward(model, inputs, model_name, dataset_name, device):
    """Single forward pass with memory measurement. Handles OOM."""
    result = {
        'model': model_name,
        'dataset': dataset_name,
        'params': None,
        'size_mb': None,
        'peak_gpu_mb': None,
        'status': None,
        'breakdown': None,
    }

    total, size_mb, breakdown = count_params(model)
    result['params'] = total
    result['size_mb'] = size_mb
    result['breakdown'] = breakdown
    print(f"    Params: {total:,} ({size_mb:.2f} MB)")
    for name, info in breakdown.items():
        print(f"      {name}: {info['params']:,} ({info['pct']:.1f}%)")

    try:
        model = model.to(device)
        gpu_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                gpu_inputs[k] = v.to(device)
            else:
                gpu_inputs[k] = v

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        # --- Single forward pass for memory measurement ---
        with torch.no_grad():
            out = model(**gpu_inputs)

        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        result['peak_gpu_mb'] = round(peak_mb, 1)
        out_shape = tuple(out.shape) if hasattr(out, 'shape') else '?'
        print(f"    Peak GPU: {peak_mb:.0f} MB")
        print(f"    Output: {out_shape}")

        # --- Timing: warmup + timed steps ---
        n_warmup = 3
        n_timed = 10

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(**gpu_inputs)
        torch.cuda.synchronize(device)

        # Timed forward passes
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            for _ in range(n_timed):
                _ = model(**gpu_inputs)
        end.record()
        torch.cuda.synchronize(device)

        step_ms = start.elapsed_time(end) / n_timed
        result['step_ms'] = round(step_ms, 2)
        print(f"    Step time: {step_ms:.2f} ms")

        # Estimate rollout times for each dataset
        rollout_steps = {'Shock Tube': 42, 'Elastoplastic': 39, 'River': 50}
        n_steps = rollout_steps.get(dataset_name, 40)
        rollout_s = (step_ms * n_steps) / 1000.0
        result['rollout_steps'] = n_steps
        result['rollout_s'] = round(rollout_s, 3)
        print(f"    Rollout ({n_steps} steps): {rollout_s:.3f} s")

        result['status'] = 'OK'

    except torch.cuda.OutOfMemoryError:
        result['status'] = 'OOM'
        result['peak_gpu_mb'] = '>80000'
        print(f"    *** OUT OF MEMORY on {torch.cuda.get_device_name()} ***")

    except Exception as e:
        result['status'] = f'ERROR: {str(e)[:200]}'
        print(f"    *** Error: {e} ***")
        traceback.print_exc()

    finally:
        model.cpu()
        del model
        torch.cuda.empty_cache()

    return result


def benchmark_dataset(ds_name, info, device, grid_res=32, fno_hidden=64, fno_layers=4):
    """Benchmark GINO on one dataset."""
    n = info['n_nodes']
    cd = info['coord_dim']

    # Determine in/out channels from the data
    # For neural operators: in_channels = dynamic field channels
    # We'll use x_channels as in and y_channels as out
    in_ch = info.get('x_channels', 4)
    out_ch = info.get('y_channels', in_ch)

    # If y_channels not available, assume same as x
    if out_ch is None:
        out_ch = in_ch

    print(f"\n  Config: {n:,} nodes, {info['n_edges']:,} edges, "
          f"{in_ch} in -> {out_ch} out, {cd}D coords, "
          f"{grid_res}x{grid_res} latent grid")

    # Build dummy inputs matching real data dimensions
    # Use positions from real data if available
    sample = info.get('sample')
    if sample is not None and hasattr(sample, 'pos') and sample.pos is not None:
        input_pos = sample.pos.float().cpu()
        # Normalize to [0,1] for GNO radius to make sense
        for d in range(cd):
            pmin = input_pos[:, d].min()
            pmax = input_pos[:, d].max()
            if pmax > pmin:
                input_pos[:, d] = (input_pos[:, d] - pmin) / (pmax - pmin)
    elif sample is not None and hasattr(sample, 'x') and sample.x is not None:
        # Some datasets store positions in x[:, :2]
        input_pos = sample.x[:, :cd].float().cpu()
        for d in range(cd):
            pmin = input_pos[:, d].min()
            pmax = input_pos[:, d].max()
            if pmax > pmin:
                input_pos[:, d] = (input_pos[:, d] - pmin) / (pmax - pmin)
    else:
        input_pos = torch.rand(n, cd)

    input_f = torch.randn(n, in_ch)
    output_pos = input_pos.clone()

    # Compute appropriate GNO radius from mesh spacing
    # Average spacing ≈ 1/sqrt(N) for 2D, radius should cover ~5-10 neighbors
    avg_spacing = 1.0 / (n ** (1.0 / cd))
    gno_radius = avg_spacing * 3.0  # ~3x spacing captures ~10-30 neighbors
    print(f"  Avg mesh spacing: {avg_spacing:.4f}, GNO radius: {gno_radius:.4f}")

    grid_1d = torch.linspace(0, 1, grid_res)
    latent_grid = torch.stack(
        torch.meshgrid(*[grid_1d] * cd, indexing='ij'), dim=-1
    )

    results = []

    # ---- GNO ----
    # Graph Neural Operator (Li et al. 2020): operates directly on irregular
    # point clouds via stacked kernel integral transforms. No latent grid.
    # Memory scales with n_nodes^2 within radius (neighbor search).
    print(f"\n  [GNO]")
    try:
        gno = make_gno(in_ch, out_ch, cd, gno_radius=gno_radius)
        gno_inputs = {
            'pos': input_pos,              # (n_nodes, coord_dim)
            'x': input_f.unsqueeze(0),     # (1, n_nodes, in_channels)
        }
        r = try_forward(gno, gno_inputs, 'GNO', ds_name, device)
        r['n_nodes'] = n
        r['n_edges'] = info['n_edges']
        r['in_channels'] = in_ch
        r['out_channels'] = out_ch
        if r.get('rollout_s') and r['rollout_s'] > 0:
            r['node_steps_per_s'] = round(n * r['rollout_steps'] / r['rollout_s'], 0)
        results.append(r)
    except Exception as e:
        print(f"    Setup error: {e}")
        traceback.print_exc()
        results.append({'model': 'GNO', 'dataset': ds_name,
                        'status': f'SETUP_ERROR: {str(e)[:200]}'})

    # ---- GINO ----
    # GINO handles irregular meshes natively via GNO→FNO→GNO pipeline:
    #   1. Input GNO: rasterizes from irregular mesh nodes → regular latent grid
    #   2. FNO blocks: spectral convolution on regular grid
    #   3. Output GNO: maps from latent grid → output query points (mesh nodes)
    # forward(input_geom, latent_queries, output_queries, x=...)
    # The GNO neighbor search (pairwise distances within gno_radius) is the
    # memory bottleneck — it scales with n_nodes × n_grid_points.
    print(f"\n  [GINO]")
    try:
        gino = make_gino(in_ch, out_ch, cd, grid_res, gno_radius=gno_radius,
                         fno_hidden=fno_hidden, fno_layers=fno_layers)
        gino_inputs = {
            'input_geom': input_pos.unsqueeze(0),       # (1, n_nodes, coord_dim) — real mesh positions
            'latent_queries': latent_grid.unsqueeze(0),  # (1, grid, grid, coord_dim) — regular FNO grid
            'output_queries': output_pos.unsqueeze(0),   # (1, n_nodes, coord_dim) — output at same mesh
            'x': input_f.unsqueeze(0),                   # (1, n_nodes, in_channels) — field values on mesh
        }
        r = try_forward(gino, gino_inputs, 'GINO', ds_name, device)
        r['n_nodes'] = n
        r['n_edges'] = info['n_edges']
        r['in_channels'] = in_ch
        r['out_channels'] = out_ch
        if r.get('rollout_s') and r['rollout_s'] > 0:
            r['node_steps_per_s'] = round(n * r['rollout_steps'] / r['rollout_s'], 0)
        results.append(r)
    except Exception as e:
        print(f"    Setup error: {e}")
        traceback.print_exc()
        results.append({'model': 'GINO', 'dataset': ds_name,
                        'status': f'SETUP_ERROR: {str(e)[:200]}'})

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GINO Memory Benchmark on G-PARC Datasets")
    parser.add_argument("--st_test_dir", type=str, default='',
                        help="Shock tube test directory")
    parser.add_argument("--el_test_dir", type=str, default='',
                        help="Elastoplastic test directory")
    parser.add_argument("--rv_test_dir", type=str, default='',
                        help="River test directory")
    parser.add_argument("--grid_res", type=int, default=32,
                        help="Latent FNO grid resolution (default: 32)")
    parser.add_argument("--fno_hidden", type=int, default=16,
                        help="FNO hidden channels for GINO (16 -> ~250K params)")
    parser.add_argument("--fno_layers", type=int, default=4,
                        help="Number of FNO layers for GINO")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("  GNO / GINO MEMORY BENCHMARK")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU Memory: {props.total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: No GPU available")
    print(f"  Latent grid: {args.grid_res}x{args.grid_res}")
    print(f"  GINO FNO: hidden={args.fno_hidden}, layers={args.fno_layers}")
    print()

    # ---- Load real data dimensions ----
    datasets = {}

    if args.st_test_dir and os.path.isdir(args.st_test_dir):
        print("Loading Shock Tube sample...")
        datasets['Shock Tube'] = load_sample(args.st_test_dir)
        describe_dataset('Shock Tube', datasets['Shock Tube'])

    if args.el_test_dir and os.path.isdir(args.el_test_dir):
        print("Loading Elastoplastic sample...")
        datasets['Elastoplastic'] = load_sample(args.el_test_dir)
        describe_dataset('Elastoplastic', datasets['Elastoplastic'])

    if args.rv_test_dir and os.path.isdir(args.rv_test_dir):
        print("Loading River sample...")
        datasets['River'] = load_sample(args.rv_test_dir)
        describe_dataset('River', datasets['River'])

    if not datasets:
        print("ERROR: No valid test directories provided.")
        print("Usage: python benchmark_no_models.py "
              "--st_test_dir ... --el_test_dir ... --rv_test_dir ...")
        return

    # ---- Run benchmarks ----
    all_results = []

    for ds_name, info in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"  {ds_name} ({info['n_nodes']:,} nodes, "
              f"{info['n_edges']:,} edges)")
        print(f"{'=' * 60}")

        ds_results = benchmark_dataset(
            ds_name, info, device, grid_res=args.grid_res,
            fno_hidden=args.fno_hidden, fno_layers=args.fno_layers)
        all_results.extend(ds_results)

    # ---- Summary ----
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Model':>10s} | {'Dataset':>15s} | {'Nodes':>8s} | "
          f"{'Edges':>8s} | {'Params':>12s} | {'Size(MB)':>8s} | "
          f"{'GPU(MB)':>10s} | Status")
    print("-" * 90)
    for r in all_results:
        params = f"{r.get('params', '?'):,}" if r.get('params') else '?'
        size = str(r.get('size_mb', '?'))
        gpu = str(r.get('peak_gpu_mb', '?'))
        nodes = str(r.get('n_nodes', '?'))
        edges = str(r.get('n_edges', '?'))
        print(f"{r.get('model','?'):>10s} | {r.get('dataset','?'):>15s} | "
              f"{nodes:>8s} | {edges:>8s} | {params:>12s} | "
              f"{size:>8s} | {gpu:>10s} | {r.get('status', '?')}")
    print(f"{'=' * 90}")

    # Compare with G-PARC
    print(f"\n  For reference, G-PARC on these datasets:")
    print(f"    Shock Tube:    249K params,  220 MB GPU")
    print(f"    Elastoplastic: 285K params,  528 MB GPU")
    print(f"    River:         224K params,  704 MB GPU")

    # Save
    # Remove non-serializable sample objects
    serializable = []
    for r in all_results:
        r2 = {k: v for k, v in r.items() if k != 'sample'}
        serializable.append(r2)

    out_path = 'gino_benchmark_results.json'
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == '__main__':
    main()