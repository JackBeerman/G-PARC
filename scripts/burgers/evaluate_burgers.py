#!/usr/bin/env python3
"""
G-PARCv2 Burgers Evaluation Script (Updated)
=============================================
Matches updated train_burgers.py architecture:
  - hop.py operators (SolveGradientsLST, DiffusionFD)
  - GraphConvFeatureExtractorV2
  - BurgersDifferentiator with diffusion_type + FiLM
  - dt=1.0

Features:
  - Rollout evaluation with RRMSE / R² / MAE metrics
  - GIF generation with fixed colorbars
  - Per-Reynolds analysis
  - Divergence detection
"""

import argparse
import os
import sys
from pathlib import Path
import json
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.featureextractor import GraphConvFeatureExtractorV2
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from differentiator.burgers_differentiator import BurgersDifferentiator
from models.burgers import GPARC_Burgers_Numerical

warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# EVALUATOR
# ==============================================================================

class BurgersEvaluator:
    VAR_NAMES = ['u', 'v']

    def __init__(self, model, device='cpu', denormalization_params=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.denorm_params = denormalization_params or {}
        self.simulation_metrics = []

    # ------------------------------------------------------------------
    # Denormalization
    # ------------------------------------------------------------------
    def load_denormalization_params(self, metadata_file):
        if not Path(metadata_file).exists():
            print(f"Warning: Metadata not found: {metadata_file}")
            return
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        stats = metadata.get('normalization_statistics', {})
        key_map = {'u': 'U_x', 'v': 'U_y'}

        for var in self.VAR_NAMES:
            meta_key = key_map.get(var, var)
            if meta_key in stats:
                self.denorm_params[var] = stats[meta_key]

        if self.denorm_params:
            print(f"  Denorm params loaded for: {list(self.denorm_params.keys())}")
        else:
            print("  No denorm params found — reporting normalized metrics")

    def denormalize(self, data_np, method='zscore'):
        if not self.denorm_params:
            return data_np
        out = np.zeros_like(data_np)
        for i, var in enumerate(self.VAR_NAMES):
            if var not in self.denorm_params:
                out[:, i] = data_np[:, i]
                continue
            p = self.denorm_params[var]
            if method == 'zscore':
                out[:, i] = data_np[:, i] * p.get('std', 1.0) + p.get('mean', 0.0)
            else:
                out[:, i] = data_np[:, i]
        return out

    # ------------------------------------------------------------------
    # Rollout generation
    # ------------------------------------------------------------------
    def generate_rollout(self, simulation, rollout_steps):
        """Autoregressive rollout from ground truth t=0."""
        predictions = []
        initial = simulation[0]

        static_feats = initial.x[:, :self.model.num_static_feats]
        edge_index = initial.edge_index
        F_prev = initial.x[:, self.model.num_static_feats:]

        for step in range(rollout_steps):
            F_next = self.model.integrator(
                derivative_fn=self.model.derivative_solver,
                static_feats=static_feats,
                dynamic_state=F_prev,
                edge_index=edge_index,
                dt=1.0,
            )
            predictions.append(F_next)
            F_prev = F_next

        return predictions

    def generate_snapshot_predictions(self, simulation, num_steps):
        """Single-step from GT at each timestep (no error accumulation)."""
        predictions = []
        for step in range(num_steps):
            data_t = simulation[step]
            static_feats = data_t.x[:, :self.model.num_static_feats]
            edge_index = data_t.edge_index
            dynamic_gt = data_t.x[:, self.model.num_static_feats:]

            F_pred = self.model.integrator(
                derivative_fn=self.model.derivative_solver,
                static_feats=static_feats,
                dynamic_state=dynamic_gt,
                edge_index=edge_index,
                dt=1.0,
            )
            predictions.append(F_pred)
        return predictions

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def evaluate(self, simulations, rollout_steps=20, mode='rollout'):
        all_preds = []
        all_targs = []
        metadata = []
        self.simulation_metrics = []

        print(f"\n{'=' * 60}")
        print(f"EVALUATION ({mode} mode, {rollout_steps} steps)")
        print(f"{'=' * 60}")

        with torch.no_grad():
            for sim_idx, simulation in enumerate(
                tqdm(simulations, desc=f"{mode.capitalize()} predictions")
            ):
                try:
                    # Move to device + reorder features
                    for data in simulation:
                        pos = data.x[:, 0:2]
                        vel = data.x[:, 2:4]
                        re = data.x[:, 4:5]
                        data.x = torch.cat([pos, re, vel], dim=1)

                        data.x = data.x.to(self.device)
                        data.edge_index = data.edge_index.to(self.device)
                        data.y = data.y.to(self.device)
                        if hasattr(data, 'pos'):
                            data.pos = data.pos.to(self.device)

                    # Initialize MLS on first sim
                    if sim_idx == 0:
                        self.model.derivative_solver.initialize_weights(
                            simulation[0]
                        )

                    max_steps = len(simulation) - 1  # y[t] = state at t+1
                    steps = min(rollout_steps, max_steps)

                    if mode == 'rollout':
                        preds_norm = self.generate_rollout(simulation, steps)
                    else:
                        preds_norm = self.generate_snapshot_predictions(
                            simulation, steps
                        )

                    targs_norm = [
                        simulation[i].y.cpu().numpy() for i in range(steps)
                    ]

                    # Convert and check for issues
                    preds_phys = []
                    targs_phys = []
                    has_nan = False
                    has_inf = False
                    first_issue_step = -1

                    for step_idx, (p_t, t_np) in enumerate(
                        zip(preds_norm, targs_norm)
                    ):
                        p_np = p_t.cpu().numpy()

                        if np.isnan(p_np).any() and not has_nan:
                            has_nan = True
                            first_issue_step = step_idx
                        if np.isinf(p_np).any() and not has_inf:
                            has_inf = True
                            if first_issue_step == -1:
                                first_issue_step = step_idx

                        preds_phys.append(self.denormalize(p_np))
                        targs_phys.append(self.denormalize(t_np))

                    is_diverged = has_nan or has_inf
                    re_val = simulation[0].x[0, 2].item()

                    # Range report
                    p_all = np.concatenate(preds_phys, axis=0)
                    t_all = np.concatenate(targs_phys, axis=0)
                    print(
                        f"\n[Sim {sim_idx}] Re={re_val:.4f}"
                    )
                    print(
                        f"  Target:     [{t_all.min():.4f}, {t_all.max():.4f}]"
                        f"  mean={t_all.mean():.4f}"
                    )
                    print(
                        f"  Prediction: [{np.nanmin(p_all):.4e}, "
                        f"{np.nanmax(p_all):.4e}]"
                        f"  mean={np.nanmean(p_all):.4e}"
                    )
                    if is_diverged:
                        print(
                            f"  ⚠️  DIVERGED at step {first_issue_step}"
                        )

                    all_preds.append(preds_phys)
                    all_targs.append(targs_phys)

                    sim_meta = {
                        'id': sim_idx,
                        'name': f"sim_{sim_idx}"
                        + ("_DIVERGED" if is_diverged else ""),
                        'reynolds': re_val,
                        'diverged': is_diverged,
                        'divergence_step': first_issue_step
                        if is_diverged
                        else -1,
                    }
                    metadata.append(sim_meta)

                    if not is_diverged:
                        self._track_metrics(preds_phys, targs_phys, sim_meta)

                except Exception as e:
                    print(f"Error on sim {sim_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        return all_preds, all_targs, metadata

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _track_metrics(self, preds, targs, meta):
        p_flat = np.concatenate(preds, axis=0)
        t_flat = np.concatenate(targs, axis=0)

        # Per-variable metrics
        var_metrics = {}
        for i, var in enumerate(self.VAR_NAMES):
            p_v = p_flat[:, i]
            t_v = t_flat[:, i]
            denom = np.sqrt(np.mean(t_v ** 2)) + 1e-10
            var_metrics[var] = {
                'rmse': float(np.sqrt(mean_squared_error(t_v, p_v))),
                'rrmse': float(
                    np.sqrt(mean_squared_error(t_v, p_v)) / denom
                ),
                'mae': float(mean_absolute_error(t_v, p_v)),
                'r2': float(r2_score(t_v, p_v)),
            }

        overall_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(t_flat, p_flat))),
            'mae': float(mean_absolute_error(t_flat, p_flat)),
            'r2': float(r2_score(t_flat, p_flat)),
        }

        self.simulation_metrics.append(
            {
                'meta': meta,
                'overall': overall_metrics,
                'per_variable': var_metrics,
            }
        )

    def print_summary(self):
        if not self.simulation_metrics:
            print("\nNo valid simulations for metrics.")
            return

        print(f"\n{'=' * 60}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 60}")

        for m in self.simulation_metrics:
            name = m['meta']['name']
            re = m['meta']['reynolds']
            o = m['overall']
            print(
                f"  {name:30s}  Re={re:.4f}  "
                f"RMSE={o['rmse']:.6f}  R²={o['r2']:.4f}"
            )

        rmses = [m['overall']['rmse'] for m in self.simulation_metrics]
        r2s = [m['overall']['r2'] for m in self.simulation_metrics]
        print(f"\n  Average RMSE: {np.mean(rmses):.6f}")
        print(f"  Average R²:   {np.mean(r2s):.4f}")
        print(f"  Valid sims:   {len(self.simulation_metrics)}")

    # ------------------------------------------------------------------
    # GIF creation
    # ------------------------------------------------------------------
    def create_gif(self, preds, targs, meta, output_path):
        """Side-by-side velocity magnitude GIF with fixed colorbars."""
        N = preds[0].shape[0]
        S = int(np.sqrt(N))
        steps = len(preds)

        diverged = meta.get('diverged', False)
        div_step = meta.get('divergence_step', -1)

        title = f"Burgers Rollout (Re={meta['reynolds']:.2f})"
        if diverged:
            title += f" [DIVERGED @ step {div_step}]"

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, color='red' if diverged else 'black')

        mag_t = [
            np.sqrt(t[:, 0] ** 2 + t[:, 1] ** 2).reshape(S, S)
            for t in targs
        ]
        mag_p = [
            np.sqrt(
                np.nan_to_num(p[:, 0], 0) ** 2
                + np.nan_to_num(p[:, 1], 0) ** 2
            ).reshape(S, S)
            for p in preds
        ]

        all_t = np.array(mag_t)
        vmin, vmax = all_t.min(), all_t.max()
        if vmax - vmin < 1e-6:
            vmax = vmin + 0.1

        im1 = axes[0].imshow(
            mag_t[0], vmin=vmin, vmax=vmax, cmap='magma', origin='lower'
        )
        axes[0].set_title('Target (t=0)')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(
            mag_p[0], vmin=vmin, vmax=vmax, cmap='magma', origin='lower'
        )
        axes[1].set_title('Prediction (t=0)')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        def update(frame):
            im1.set_data(mag_t[frame])
            im2.set_data(mag_p[frame])

            t_r = f"[{mag_t[frame].min():.3f}, {mag_t[frame].max():.3f}]"
            axes[0].set_title(f'Target (t={frame})\n{t_r}')

            p_min = np.nanmin(mag_p[frame])
            p_max = np.nanmax(mag_p[frame])
            if diverged and frame >= div_step:
                axes[1].set_title(
                    f'Prediction (t={frame}) **DIVERGED**\n'
                    f'[{p_min:.2e}, {p_max:.2e}]',
                    color='red',
                )
            else:
                axes[1].set_title(
                    f'Prediction (t={frame})\n[{p_min:.3f}, {p_max:.3f}]'
                )
            return im1, im2

        anim = FuncAnimation(
            fig, update, frames=steps, interval=100, blit=False
        )
        anim.save(output_path, writer=PillowWriter(fps=10))
        plt.close(fig)
        print(f"  Saved GIF: {output_path}")

    # ------------------------------------------------------------------
    # Per-Reynolds summary plot
    # ------------------------------------------------------------------
    def plot_reynolds_summary(self, output_path):
        """R² vs Reynolds number scatter plot."""
        if len(self.simulation_metrics) < 2:
            return

        re_vals = [m['meta']['reynolds'] for m in self.simulation_metrics]
        r2_vals = [m['overall']['r2'] for m in self.simulation_metrics]
        rmse_vals = [m['overall']['rmse'] for m in self.simulation_metrics]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.scatter(re_vals, r2_vals, c='steelblue', s=40, alpha=0.7)
        ax1.set_xlabel('Reynolds Number (normalized)')
        ax1.set_ylabel('R²')
        ax1.set_title('R² vs Reynolds')
        ax1.grid(True, alpha=0.3)

        ax2.scatter(re_vals, rmse_vals, c='coral', s=40, alpha=0.7)
        ax2.set_xlabel('Reynolds Number (normalized)')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE vs Reynolds')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved Reynolds summary: {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def load_simulations(data_dir, pattern="*.pt", limit=None):
    files = sorted(list(Path(data_dir).glob(pattern)))
    if limit:
        files = files[:limit]
    print(f"Loading {len(files)} simulations from {data_dir}...")
    return [torch.load(f, weights_only=False) for f in files]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate G-PARCv2 Burgers model"
    )

    # Paths
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="eval_burgers")
    parser.add_argument("--metadata_file", default=None,
                        help="Path to normalization_metadata.json")

    # Architecture (must match training)
    parser.add_argument("--num_static_feats", type=int, default=3)
    parser.add_argument("--num_dynamic_feats", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--num_fe_layers", type=int, default=4)
    parser.add_argument("--spade_heads", type=int, default=2)
    parser.add_argument("--diffusion_type", type=str, default="fd",
                        choices=["fd", "mls", "none"])
    parser.add_argument("--use_film", action="store_true", default=True)
    parser.add_argument("--no_film", dest="use_film", action="store_false")

    # Eval
    parser.add_argument("--integrator", default="euler")
    parser.add_argument("--mode", default="rollout",
                        choices=["rollout", "snapshot"])
    parser.add_argument("--rollout_steps", type=int, default=50)
    parser.add_argument("--max_sequences", type=int, default=10)
    parser.add_argument("--create_gifs", action="store_true")
    parser.add_argument("--num_gifs", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("G-PARCv2 BURGERS EVALUATION")
    print("=" * 70)
    print(f"Device:     {device}")
    print(f"Model:      {args.model_path}")
    print(f"Test dir:   {args.test_dir}")
    print(f"Mode:       {args.mode}")
    print(f"Steps:      {args.rollout_steps}")
    print(f"Diffusion:  {args.diffusion_type}")
    print(f"FiLM:       {args.use_film}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Build model (must match training architecture exactly)
    # ------------------------------------------------------------------
    print("\nBuilding model...")

    gradient_solver = SolveGradientsLST()

    laplacian_solver = None
    if args.diffusion_type == 'mls':
        laplacian_solver = SolveWeightLST2d(use_2hop_extension=True)

    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=args.num_static_feats,
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        num_layers=args.num_fe_layers,
    )

    differentiator = BurgersDifferentiator(
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=args.feature_out_channels,
        spade_heads=args.spade_heads,
        zero_init=True,
        diffusion_type=args.diffusion_type,
        use_film=args.use_film,
    )

    model = GPARC_Burgers_Numerical(
        derivative_solver=differentiator,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
    )

    # Load weights
    checkpoint = torch.load(
        args.model_path, map_location=device, weights_only=False
    )
    load_result = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False
    )

    if load_result.missing_keys:
        print(f"  Missing keys: {len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        unexpected_real = [
            k for k in load_result.unexpected_keys if 'static_' not in k
        ]
        if unexpected_real:
            print(f"  ⚠️  Unexpected keys: {unexpected_real}")

    print("✓ Model loaded")

    # Param count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading test simulations...")
    sims = load_simulations(args.test_dir, limit=args.max_sequences)
    if not sims:
        print("No simulations loaded.")
        return

    # ------------------------------------------------------------------
    # Initialize MLS from first sim (after reorder)
    # ------------------------------------------------------------------
    sample = sims[0][0]
    pos = sample.x[:, 0:2]
    vel = sample.x[:, 2:4]
    re = sample.x[:, 4:5]
    sample.x = torch.cat([pos, re, vel], dim=1).to(device)
    sample.edge_index = sample.edge_index.to(device)
    if hasattr(sample, 'pos'):
        sample.pos = sample.pos.to(device)
    model.derivative_solver.initialize_weights(sample)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    evaluator = BurgersEvaluator(model, device)

    # Load denorm params
    if args.metadata_file:
        meta_path = args.metadata_file
    else:
        meta_path = Path(args.test_dir).parent / "normalization_metadata.json"
    evaluator.load_denormalization_params(meta_path)

    preds, targs, metadata = evaluator.evaluate(
        sims, args.rollout_steps, mode=args.mode
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    evaluator.print_summary()

    # Save metrics
    metrics_path = Path(args.output_dir) / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(
            {
                'simulation_metrics': evaluator.simulation_metrics,
                'metadata': metadata,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n  Saved metrics: {metrics_path}")

    # Reynolds summary plot
    evaluator.plot_reynolds_summary(
        Path(args.output_dir) / "reynolds_summary.png"
    )

    # GIFs
    if args.create_gifs:
        n_gifs = min(args.num_gifs, len(preds))
        print(f"\nGenerating {n_gifs} GIFs...")
        for i in range(n_gifs):
            gif_path = Path(args.output_dir) / f"{metadata[i]['name']}.gif"
            evaluator.create_gif(preds[i], targs[i], metadata[i], gif_path)

    print("\n✓ Evaluation complete")


if __name__ == "__main__":
    main()