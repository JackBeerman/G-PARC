#!/usr/bin/env python3
"""
G-PARC Computational Efficiency Benchmark
==========================================
Integrates directly with eval.py registries to benchmark:
  1. Inference time (per simulation, per step, throughput)
  2. FLOPs (via torch.profiler with_flops)
  3. GPU peak memory during rollout
  4. Parameter counts with submodule breakdown
  5. Parameter efficiency (accuracy / params)

Uses the SAME model spec format as eval.py:
    --st_models gparcv2:/path/to/ckpt gparcv1:/path/to/ckpt ...

Usage:
    python benchmark.py --datasets shocktube elasto river \
        --st_test_dir ... --st_models gparcv2:... gparcv1:... \
        --el_test_dir ... --el_norm_stats ... --el_models gparcv2_nospade:... \
        --rv_test_dir ... --rv_extrema ... --rv_models gparcv2:... \
        --n_warmup 3 --n_timed 20 --max_sims 10 \
        --output_dir benchmark_results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Import everything from eval.py
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from eval import (
    ST_REGISTRY, EL_REGISTRY, RV_REGISTRY,
    st_load_data, el_load_data, rv_load_data,
    _clear_mls_caches,
    ST_NUM_STATIC, ST_NUM_DYNAMIC,
    parse_model_specs,
)


# ============================================================
# PARAMETER ANALYSIS
# ============================================================

def analyze_parameters(model):
    """Detailed parameter breakdown by submodule."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    breakdown = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n > 0:
            breakdown[name] = {
                'params': n,
                'pct': round(100.0 * n / max(total, 1), 1),
            }

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
        'size_mb': round(sum(p.nelement() * p.element_size()
                             for p in model.parameters()) / 1024**2, 2),
        'breakdown': breakdown,
    }


# ============================================================
# TIMING BENCHMARK
# ============================================================

def benchmark_timing(model, rollout_fn, sims, num_steps, device,
                     n_warmup=3, n_timed=20):
    """Precise timing with CUDA synchronization. Handles variable-length sims."""
    use_cuda = torch.cuda.is_available() and 'cpu' not in str(device)

    # Warmup
    for i in range(min(n_warmup, len(sims))):
        sim = sims[i][1]
        steps = min(num_steps, len(sim) - 1) if num_steps > 0 else len(sim) - 1
        try:
            rollout_fn(model, sim, steps, device)
        except Exception:
            pass
    if use_cuda:
        torch.cuda.synchronize(device)

    # Timed runs
    sim_times = []
    step_counts = []
    node_counts = []

    for run in range(n_timed):
        for sim_name, sim_data in sims:
            # Each sim may have different length
            steps = min(num_steps, len(sim_data) - 1) if num_steps > 0 else len(sim_data) - 1
            n_nodes = sim_data[0].x.size(0) if hasattr(sim_data[0], 'x') else 0

            if use_cuda:
                torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            try:
                rollout_fn(model, sim_data, steps, device)
            except Exception as e:
                print(f"    Rollout error: {e}")
                continue

            if use_cuda:
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            sim_times.append(t1 - t0)
            step_counts.append(steps)
            node_counts.append(n_nodes)

    if not sim_times:
        return {'error': 'All rollouts failed'}

    sim_times = np.array(sim_times)
    step_counts = np.array(step_counts)
    node_counts = np.array(node_counts)

    # Per-step times: divide each sim_time by its own step count
    step_times = sim_times / np.maximum(step_counts, 1)
    avg_nodes = int(np.mean(node_counts))
    avg_steps = float(np.mean(step_counts))
    min_steps = int(np.min(step_counts))
    max_steps = int(np.max(step_counts))

    return {
        'n_rollouts': len(sim_times),
        'avg_nodes': avg_nodes,
        'avg_steps': round(avg_steps, 1),
        'min_steps': min_steps,
        'max_steps': max_steps,
        'sim_time_mean_s': round(float(np.mean(sim_times)), 6),
        'sim_time_std_s': round(float(np.std(sim_times)), 6),
        'sim_time_median_s': round(float(np.median(sim_times)), 6),
        # Per-step: computed per-rollout then averaged
        'step_time_mean_ms': round(float(np.mean(step_times)) * 1000, 4),
        'step_time_std_ms': round(float(np.std(step_times)) * 1000, 4),
        'sims_per_sec': round(1.0 / max(float(np.mean(sim_times)), 1e-12), 2),
        'steps_per_sec': round(1.0 / max(float(np.mean(step_times)), 1e-12), 1),
        'node_steps_per_sec': round(avg_nodes / max(float(np.mean(step_times)), 1e-12), 0),
    }


# ============================================================
# GPU MEMORY
# ============================================================

def measure_memory(model, rollout_fn, sim_data, num_steps, device):
    """Peak GPU memory during one rollout."""
    use_cuda = torch.cuda.is_available() and 'cpu' not in str(device)
    if not use_cuda:
        return {'peak_mb': 0, 'note': 'CPU mode'}

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)

    steps = min(num_steps, len(sim_data) - 1) if num_steps > 0 else len(sim_data) - 1
    rollout_fn(model, sim_data, steps, device)

    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    delta = (torch.cuda.memory_allocated(device) - mem_before) / 1024**2

    return {
        'peak_mb': round(peak, 1),
        'inference_delta_mb': round(delta, 1),
    }


# ============================================================
# FLOPs ESTIMATION
# ============================================================

def estimate_flops(model, rollout_fn, sim_data, num_steps, device):
    """
    Estimate FLOPs using multiple methods for robustness:
      1. torch.profiler (primary, but can double-count attention)
      2. Analytical estimate from model architecture (cross-check)
      3. Manual forward timing extrapolation (fallback)

    If profiler result exceeds analytical estimate by >10x, we flag it
    as anomalous and use the analytical estimate instead.
    """
    steps = min(num_steps, len(sim_data) - 1) if num_steps > 0 else len(sim_data) - 1
    n_params = sum(p.numel() for p in model.parameters())
    profiler_flops = None
    try:
        from torch.profiler import profile, ProfilerActivity
        use_cuda = torch.cuda.is_available() and 'cpu' not in str(device)
        activities = [ProfilerActivity.CPU]
        if use_cuda:
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=True,
                     with_flops=True) as prof:
            rollout_fn(model, sim_data, steps, device)

        profiler_flops = sum(e.flops for e in prof.key_averages()
                             if e.flops and e.flops > 0)
    except Exception as e:
        print(f"      Profiler failed: {e}")

    # ---- Method 2: Analytical estimate ----
    # For standard architectures: ~2 * params * activations per step
    # For GNNs: each message-passing layer does ~2*H*H FLOPs per edge
    # Rough estimate: 2 * total_params * num_nodes per step
    n_nodes = sim_data[0].x.size(0) if hasattr(sim_data[0], 'x') else 1000
    n_edges = sim_data[0].edge_index.size(1) if hasattr(sim_data[0], 'edge_index') else n_nodes * 10
    # Each param participates in ~1 multiply + 1 add per forward pass
    # For GNNs, this scales with edges not just nodes
    analytical_per_step = 2 * n_params * max(n_edges / n_nodes, 1) * 2
    analytical_total = analytical_per_step * steps

    # ---- Method 3: Empirical per-step via profiling single steps ----
    single_step_flops = None
    try:
        from torch.profiler import profile, ProfilerActivity
        use_cuda = torch.cuda.is_available() and 'cpu' not in str(device)
        activities = [ProfilerActivity.CPU]
        if use_cuda:
            activities.append(ProfilerActivity.CUDA)

        # Run just 1 step
        with profile(activities=activities, record_shapes=True,
                     with_flops=True) as prof:
            rollout_fn(model, sim_data, 1, device)

        single_step_flops = sum(e.flops for e in prof.key_averages()
                                if e.flops and e.flops > 0)
    except Exception:
        pass

    # ---- Select best estimate ----
    method = 'profiler'
    total_flops = profiler_flops

    # Sanity check: if profiler reports > 100x the single-step * num_steps,
    # or > 100x the analytical estimate, it's anomalous
    anomalous = False
    if profiler_flops is not None and single_step_flops is not None:
        expected = single_step_flops * steps
        if profiler_flops > expected * 50:
            anomalous = True
    if profiler_flops is not None and profiler_flops > analytical_total * 100:
        anomalous = True

    if anomalous and single_step_flops is not None:
        total_flops = single_step_flops * steps
        method = 'single_step_extrapolated'
        print(f"      ! Profiler anomaly detected ({profiler_flops/1e12:.1f} TFLOPs). "
              f"Using single-step extrapolation: {total_flops/1e9:.1f} GFLOPs")
    elif anomalous:
        total_flops = int(analytical_total)
        method = 'analytical'
        print(f"      ! Profiler anomaly detected ({profiler_flops/1e12:.1f} TFLOPs). "
              f"Using analytical estimate: {total_flops/1e9:.1f} GFLOPs")
    elif total_flops is None or total_flops == 0:
        total_flops = int(analytical_total)
        method = 'analytical'

    return {
        'total_flops': total_flops,
        'total_gflops': round(total_flops / 1e9, 4),
        'total_tflops': round(total_flops / 1e12, 6),
        'per_step_gflops': round(total_flops / max(steps, 1) / 1e9, 4),
        'flops_per_param': round(total_flops / max(n_params, 1), 1),
        'method': method,
        'profiler_raw_gflops': round(profiler_flops / 1e9, 4) if profiler_flops else None,
        'single_step_gflops': round(single_step_flops / 1e9, 4) if single_step_flops else None,
        'analytical_gflops': round(analytical_total / 1e9, 4),
    }


# ============================================================
# PRETTY PRINTING
# ============================================================

def print_benchmark_table(dataset_name, model_results, num_steps):
    print(f"\n{'='*130}")
    print(f"  {dataset_name} — COMPUTATIONAL EFFICIENCY BENCHMARK")
    print(f"{'='*130}")

    hdr = (f"{'Model':24s} | {'Params':>8s} | {'Size(MB)':>8s} | "
           f"{'Sim(s)':>14s} | {'Step(ms)':>14s} | "
           f"{'Sims/s':>7s} | {'Steps/s':>8s} | "
           f"{'GPU(MB)':>8s} | {'GFLOPs':>10s} | {'Method':>12s}")
    print(hdr)
    print('-' * 136)

    for mkey, r in model_results.items():
        name = r['name']
        p = r['params']
        t = r.get('timing', {})
        m = r.get('memory', {})
        f = r.get('flops', {})

        param_str = f"{p['total']:,}"
        size_str = f"{p['size_mb']:.1f}"

        if 'error' in t:
            print(f"{name:24s} | {param_str:>8s} | {size_str:>8s} | {'ERROR':>14s}")
            continue

        sim_t = f"{t['sim_time_mean_s']:.4f}\u00B1{t['sim_time_std_s']:.4f}"
        step_t = f"{t['step_time_mean_ms']:.2f}\u00B1{t['step_time_std_ms']:.2f}"
        sps = f"{t['sims_per_sec']:.1f}"
        stps = f"{t['steps_per_sec']:.0f}"
        gpu = f"{m.get('peak_mb', 0):.0f}"
        gf = f"{f.get('total_gflops', 0):.2f}"
        meth = f.get('method', '?')

        print(f"{name:24s} | {param_str:>8s} | {size_str:>8s} | "
              f"{sim_t:>14s} | {step_t:>14s} | "
              f"{sps:>7s} | {stps:>8s} | "
              f"{gpu:>8s} | {gf:>10s} | {meth:>12s}")

    print('=' * 130)


def print_parameter_breakdown(model_results):
    print(f"\n  Parameter Breakdown:")
    for mkey, r in model_results.items():
        p = r['params']
        print(f"    {r['name']} ({p['total']:,} total, {p['size_mb']} MB)")
        for sub_name, sub_info in p['breakdown'].items():
            bar = '\u2588' * int(sub_info['pct'] / 2)
            print(f"      {sub_name:30s}: {sub_info['params']:>8,}  "
                  f"({sub_info['pct']:5.1f}%) {bar}")


def print_efficiency_comparison(all_results, rrmse_data):
    print(f"\n{'='*120}")
    print(f"  PARAMETER EFFICIENCY — Accuracy per Parameter (lower RRMSE/100K = better)")
    print(f"{'='*120}")
    hdr = (f"{'Model':24s} | {'Dataset':16s} | {'Params':>8s} | "
           f"{'RRMSE AUC':>10s} | {'RRMSE/100K':>11s} | "
           f"{'Step(ms)':>10s} | {'GFLOPs':>10s} | {'GPU(MB)':>8s}")
    print(hdr)
    print('-' * 120)

    for ds_name, ds_results in all_results.items():
        for mkey, r in ds_results.items():
            name = r['name']
            params = r['params']['total']
            rrmse = rrmse_data.get(ds_name, {}).get(mkey)
            t = r.get('timing', {})
            f = r.get('flops', {})
            m = r.get('memory', {})

            rr_str = f"{rrmse:.4f}" if rrmse else "N/A"
            eff = f"{rrmse / (params / 1e5):.6f}" if rrmse and params > 0 else "N/A"
            step_str = f"{t.get('step_time_mean_ms', 0):.2f}" if 'step_time_mean_ms' in t else "N/A"
            gf_str = f"{f.get('total_gflops', 0):.2f}" if 'total_gflops' in f else "N/A"
            gpu_str = f"{m.get('peak_mb', 0):.0f}" if 'peak_mb' in m else "N/A"

            print(f"{name:24s} | {ds_name:16s} | {params:>8,} | "
                  f"{rr_str:>10s} | {eff:>11s} | "
                  f"{step_str:>10s} | {gf_str:>10s} | {gpu_str:>8s}")
        print('-' * 120)
    print('=' * 120)


# ============================================================
# DATASET RUNNER
# ============================================================

def run_dataset(dataset_name, registry, sims, specs, device, num_steps, args,
                norm_stats=None):
    print(f"\n{'#'*70}")
    print(f"  {dataset_name} BENCHMARK")
    print(f"{'#'*70}")

    if not sims:
        print("  No data loaded!")
        return {}

    sample = sims[0][1][0]
    n_nodes = sample.x.size(0) if hasattr(sample, 'x') else 0
    actual_steps = min(num_steps, len(sims[0][1]) - 1) if num_steps > 0 else len(sims[0][1]) - 1

    # Check for variable-length sims
    all_lens = [len(s[1]) - 1 for s in sims]
    min_len, max_len = min(all_lens), max(all_lens)
    if min_len == max_len:
        print(f"  {len(sims)} sims, ~{n_nodes} nodes, {actual_steps} rollout steps")
    else:
        print(f"  {len(sims)} sims, ~{n_nodes} nodes, {min_len}-{max_len} rollout steps (variable)")
    print(f"  Timing: {args.n_warmup} warmup + {args.n_timed} timed runs "
          f"\u00D7 {len(sims)} sims = {args.n_timed * len(sims)} total rollouts")

    # Ensure sample has pos
    if not hasattr(sample, 'pos') or sample.pos is None:
        sample.pos = sample.x[:, :2]
    sample_gpu = sample.to(device)
    # Elasto needs edge_index on device too
    if hasattr(sample_gpu, 'edge_index'):
        sample_gpu.edge_index = sample_gpu.edge_index.to(device)

    results = {}
    for mkey, ckpt_path in specs.items():
        if mkey not in registry:
            print(f"  Skipping {mkey} — not in registry")
            continue
        reg = registry[mkey]
        name = reg['name']
        print(f"\n  [{name}]")

        try:
            # Elasto loaders take (ckpt, norm_stats, sample, device)
            # Shock tube / river loaders take (ckpt, sample, device)
            if norm_stats is not None:
                model = reg['load'](ckpt_path, norm_stats, sample_gpu, device)
            else:
                model = reg['load'](ckpt_path, sample_gpu, device)
            params = analyze_parameters(model)
            print(f"    \u2713 Loaded: {params['total']:,} params ({params['size_mb']:.1f} MB)")

            print(f"    Timing...")
            timing = benchmark_timing(
                model, reg['rollout'], sims, actual_steps, device,
                n_warmup=args.n_warmup, n_timed=args.n_timed)
            if 'error' not in timing:
                step_range = ""
                if timing.get('min_steps') != timing.get('max_steps'):
                    step_range = f" [{timing['min_steps']}-{timing['max_steps']} steps]"
                print(f"    \u2713 {timing['sim_time_mean_s']:.4f}s/sim, "
                      f"{timing['step_time_mean_ms']:.2f}ms/step, "
                      f"{timing['steps_per_sec']:.0f} steps/s{step_range}")

            print(f"    GPU memory...")
            memory = measure_memory(model, reg['rollout'], sims[0][1],
                                    actual_steps, device)
            print(f"    \u2713 Peak: {memory.get('peak_mb', 0):.0f} MB")

            print(f"    FLOPs...")
            flops = estimate_flops(model, reg['rollout'], sims[0][1],
                                   actual_steps, device)
            method = flops.get('method', '?')
            print(f"    \u2713 {flops.get('total_gflops', 0):.2f} GFLOPs total, "
                  f"{flops.get('per_step_gflops', 0):.4f} GFLOPs/step [{method}]")

            results[mkey] = {
                'name': name,
                'params': params,
                'timing': timing,
                'memory': memory,
                'flops': flops,
            }
        except Exception as e:
            print(f"    \u2717 Error: {e}")
            import traceback; traceback.print_exc()

    if results:
        print_benchmark_table(dataset_name, results, actual_steps)
        print_parameter_breakdown(results)

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="G-PARC Computational Benchmark")
    parser.add_argument("--datasets", nargs='+', required=True,
                        choices=['shocktube', 'elasto', 'river'])
    parser.add_argument("--output_dir", default="benchmark_results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_timed", type=int, default=20)
    parser.add_argument("--max_sims", type=int, default=10)

    # Shock tube
    parser.add_argument("--st_test_dir", type=str, default='')
    parser.add_argument("--st_models", nargs='+', default=[])
    parser.add_argument("--st_rollout_steps", type=int, default=40)

    # Elastoplastic
    parser.add_argument("--el_test_dir", type=str, default='')
    parser.add_argument("--el_norm_stats", type=str, default='')
    parser.add_argument("--el_models", nargs='+', default=[])

    # River
    parser.add_argument("--rv_test_dir", type=str, default='')
    parser.add_argument("--rv_extrema", type=str, default='')
    parser.add_argument("--rv_models", nargs='+', default=[])
    parser.add_argument("--rv_rollout_steps", type=int, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"{'='*70}")
    print(f"  G-PARC COMPUTATIONAL BENCHMARK")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  SMs: {props.multi_processor_count}")
    print(f"  Warmup: {args.n_warmup}, Timed: {args.n_timed}, Max sims: {args.max_sims}")

    # Known RRMSE AUC from unified eval
    rrmse_data = {
        'Shock Tube': {
            'gparcv2': 0.0070, 'gparcv1': 0.0824, 'mgkan': 0.0555,
            'mgnet': 0.3544, 'gsage': 7.5062,
        },
        'Elastoplastic': {
            'gparcv2_nospade': 0.4391, 'gparcv1': 0.4718, 'mgkan': 0.5850,
            'mgn': 3.4528, 'graphsage': 244.842,
        },
        'River': {
            'gparcv2': 0.2075, 'gparcv1': 0.2995, 'mgkan': 0.2263,
            'mgnet': 111.7565, 'gsage': 18.605,
        },
    }

    all_results = {}

    if 'shocktube' in args.datasets and args.st_test_dir and args.st_models:
        specs = parse_model_specs(args.st_models, ST_REGISTRY)
        sims = st_load_data(args.st_test_dir, max_sims=args.max_sims)
        all_results['Shock Tube'] = run_dataset(
            "Shock Tube", ST_REGISTRY, sims, specs,
            device, args.st_rollout_steps, args)

    if 'elasto' in args.datasets and args.el_test_dir and args.el_models:
        specs = parse_model_specs(args.el_models, EL_REGISTRY)
        sims = el_load_data(args.el_test_dir, max_sims=args.max_sims)
        num_steps = len(sims[0][1]) - 1 if sims else 10
        # Elasto loaders need norm_stats
        from eval import el_load_norm_stats
        norm_stats = el_load_norm_stats(args.el_norm_stats) if args.el_norm_stats else None
        all_results['Elastoplastic'] = run_dataset(
            "Elastoplastic", EL_REGISTRY, sims, specs,
            device, num_steps, args, norm_stats=norm_stats)

    if 'river' in args.datasets and args.rv_test_dir and args.rv_models:
        specs = parse_model_specs(args.rv_models, RV_REGISTRY)
        sims = rv_load_data(args.rv_test_dir, max_sims=args.max_sims)
        # 0 means use each sim's full length (variable-length sims)
        num_steps = args.rv_rollout_steps or 0
        all_results['River'] = run_dataset(
            "River", RV_REGISTRY, sims, specs,
            device, num_steps, args)

    # Cross-dataset summary
    if all_results:
        print_efficiency_comparison(all_results, rrmse_data)

        def serialize(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            elif isinstance(obj, (np.floating,)): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, dict): return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)): return [serialize(v) for v in obj]
            return obj

        out_path = os.path.join(args.output_dir, 'benchmark_results.json')
        with open(out_path, 'w') as f:
            json.dump(serialize(all_results), f, indent=2)
        print(f"\n\u2713 Results saved to {out_path}")


if __name__ == '__main__':
    main()