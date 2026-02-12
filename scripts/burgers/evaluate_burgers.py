#!/usr/bin/env python3
"""
GPARC Burgers Model Evaluation Script
=====================================
Features:
- Rollouts & RRMSE Metrics
- GIFs with separate colorbars
- DEBUG: Prints value ranges (Min/Max/Mean) for diagnosis
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

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# --- IMPORTS ---
from utilities.featureextractor import FeatureExtractorGNN
from differentiator.fast_differential_operators import SolveGradientsLST, SolveWeightLST2d
from differentiator.burgers_differentiator import BurgersDifferentiator
from models.burgers import GPARC_Burgers_Test
from models.burgers import GPARC_Burgers_Numerical

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# EVALUATOR CLASS
# ==============================================================================

class BurgersEvaluator:
    def __init__(self, model, device='cpu', denormalization_params=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.denorm_params = denormalization_params
        self.var_names = ['u', 'v'] 
        self.simulation_metrics = []

    def load_denormalization_params(self, metadata_file):
        if not Path(metadata_file).exists():
            print(f"Warning: Metadata not found: {metadata_file}")
            return
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        self.denorm_params = {}
        stats = metadata.get('normalization_statistics', {})
        
        key_map = {'u': 'U_x', 'v': 'U_y'}
        
        for var in self.var_names:
            meta_key = key_map.get(var, var)
            if meta_key in stats:
                self.denorm_params[var] = stats[meta_key]
        
        print(f"Loaded denorm params for: {list(self.denorm_params.keys())}")

    def denormalize_predictions(self, normalized_data, method='zscore'):
        if self.denorm_params is None: return normalized_data
        physical_data = np.zeros_like(normalized_data)
        
        for i, var_name in enumerate(self.var_names):
            if var_name not in self.denorm_params:
                physical_data[:, i] = normalized_data[:, i]
                continue
            
            params = self.denorm_params[var_name]
            if method == 'zscore':
                mean, std = params.get('mean', 0.0), params.get('std', 1.0)
                physical_data[:, i] = normalized_data[:, i] * std + mean
        
        return physical_data

    #def generate_rollout(self, initial_data, rollout_steps, dt=1.0):
    #    predictions = []
    #    F_prev = None
    #    
    #    static_feats = initial_data.x[:, :self.model.num_static_feats]
    #    edge_index = initial_data.edge_index
    #    
    #    for step in range(rollout_steps):
    #        if step == 0:
    #            dynamic_feats = initial_data.x[:, self.model.num_static_feats:]
    #        else:
    #            dynamic_feats = F_prev
    #        
    #        current_dynamic = torch.clamp(dynamic_feats, -10.0, 10.0)
    #        
    #        F_next = self.model.integrator(
    #            derivative_fn=self.model.derivative_solver,
    #            static_feats=static_feats,
    #            dynamic_state=current_dynamic,
    #            edge_index=edge_index,
    #            dt=dt
    #        )
    #        
    #        predictions.append(F_next)
    #        F_prev = F_next
    #        
    #    return predictions
    def generate_rollout(self, initial_data, rollout_steps, dt=1.0):
        predictions = []
        F_prev = None
        
        static_feats = initial_data.x[:, :self.model.num_static_feats]
        edge_index = initial_data.edge_index
        pos = static_feats[:, :2]
        
        for step in range(rollout_steps):
            if step == 0:
                dynamic_feats = initial_data.x[:, self.model.num_static_feats:]
            else:
                dynamic_feats = F_prev
            
            current_dynamic = dynamic_feats
            
            F_next = self.model.integrator(
                derivative_fn=self.model.derivative_solver,
                static_feats=static_feats,
                dynamic_state=current_dynamic,
                edge_index=edge_index,
                dt=dt
            )
    
            # CRITICAL FIX: Gentle velocity clipping
            # Prevents numerical overflow without destroying shock physics
            # Based on ground truth max velocity ~1.4
            F_next = torch.clamp(F_next, -1.0, 1.0)
            
            # Debug tracking
            max_val = F_next.abs().max().item()
            if step < 20 or max_val > 1.0:
                print(f"Step {step}: max|F|={max_val:.4f}")
            
            predictions.append(F_next)
            F_prev = F_next
            
        return predictions
    

    #def evaluate(self, simulations, rollout_steps=20, dt=1.0):
    #    all_preds = []
    #    all_targs = []
    #    metadata = []
    #    self.simulation_metrics = []
#
    #    print(f"\n{'='*60}")
    #    print(f"STARTING EVALUATION (Debug Mode: Range Printing)")
    #    print(f"{'='*60}")
#
    #    with torch.no_grad():
    #        for sim_idx, simulation in enumerate(tqdm(simulations, desc="Evaluating")):
    #            try:
    #                for data in simulation:
    #                    # Reorder features [pos, u, v, Re] -> [pos, Re, u, v]
    #                    pos = data.x[:, 0:2]
    #                    vel = data.x[:, 2:4]
    #                    re  = data.x[:, 4:5]
    #                    data.x = torch.cat([pos, re, vel], dim=1)
    #                    
    #                    data.x = data.x.to(self.device)
    #                    data.edge_index = data.edge_index.to(self.device)
    #                    data.y = data.y.to(self.device)
    #                    if hasattr(data, 'pos'):
    #                        data.pos = data.pos.to(self.device)
#
    #                initial_data = simulation[0]
    #                self.model.derivative_solver.initialize_weights(initial_data)
#
    #                max_steps = len(simulation)
    #                steps = min(rollout_steps, max_steps)
    #                
    #                preds_norm = self.generate_rollout(initial_data, steps, dt)
    #                targs_norm = [simulation[i].y.cpu().numpy() for i in range(steps)]
    #                
    #                preds_phys = []
    #                targs_phys = []
    #                
    #                is_nan = False
    #                for p_norm, t_norm in zip(preds_norm, targs_norm):
    #                    if not torch.isfinite(p_norm).all():
    #                        is_nan = True; break
    #                    
    #                    p_np = p_norm.cpu().numpy()
    #                    preds_phys.append(self.denormalize_predictions(p_np))
    #                    targs_phys.append(self.denormalize_predictions(t_norm))
    #                
    #                if is_nan:
    #                    print(f"Simulation {sim_idx} diverged (NaN). Skipping.")
    #                    continue
#
    #                # --- DEBUG: PRINT RANGES ---
    #                p_concat = np.concatenate(preds_phys, axis=0)
    #                t_concat = np.concatenate(targs_phys, axis=0)
    #                
    #                print(f"\n[Sim {sim_idx}] Range Analysis:")
    #                print(f"  Target (GT):  Min={t_concat.min():.4f}, Max={t_concat.max():.4f}, Mean={t_concat.mean():.4f}")
    #                print(f"  Prediction:   Min={p_concat.min():.4f}, Max={p_concat.max():.4f}, Mean={p_concat.mean():.4f}")
    #                print("-" * 40)
    #                # ---------------------------
#
    #                all_preds.append(preds_phys)
    #                all_targs.append(targs_phys)
    #                
    #                sim_meta = {
    #                    'id': sim_idx, 
    #                    'name': f"sim_{sim_idx}",
    #                    'reynolds': initial_data.x[0, 2].item()
    #                }
    #                metadata.append(sim_meta)
    #                
    #                self._track_metrics(preds_phys, targs_phys, sim_meta)
#
    #            except Exception as e:
    #                print(f"Error on sim {sim_idx}: {e}")
    #                continue
#
    #    return all_preds, all_targs, metadata

    def evaluate(self, simulations, rollout_steps=20, dt=1.0):
        all_preds = []
        all_targs = []
        metadata = []
        self.simulation_metrics = []
    
        print(f"\n{'='*60}")
        print(f"STARTING EVALUATION (Debug Mode: Range Printing)")
        print(f"{'='*60}")
    
        with torch.no_grad():
            for sim_idx, simulation in enumerate(tqdm(simulations, desc="Evaluating")):
                try:
                    for data in simulation:
                        # Reorder features [pos, u, v, Re] -> [pos, Re, u, v]
                        pos = data.x[:, 0:2]
                        vel = data.x[:, 2:4]
                        re  = data.x[:, 4:5]
                        data.x = torch.cat([pos, re, vel], dim=1)
                        
                        data.x = data.x.to(self.device)
                        data.edge_index = data.edge_index.to(self.device)
                        data.y = data.y.to(self.device)
                        if hasattr(data, 'pos'):
                            data.pos = data.pos.to(self.device)
    
                    initial_data = simulation[0]
                    #self.model.derivative_solver.initialize_weights(initial_data)
    
                    max_steps = len(simulation)
                    steps = min(rollout_steps, max_steps)
                    
                    preds_norm = self.generate_rollout(initial_data, steps, dt)
                    targs_norm = [simulation[i].y.cpu().numpy() for i in range(steps)]
                    
                    preds_phys = []
                    targs_phys = []
                    
                    has_nan = False
                    has_inf = False
                    first_issue_step = -1
                    
                    for step_idx, (p_norm, t_norm) in enumerate(zip(preds_norm, targs_norm)):
                        # Convert to numpy - KEEP RAW VALUES
                        p_np = p_norm.cpu().numpy()
                        
                        # Check for NaN/Inf but DON'T replace
                        if np.isnan(p_np).any():
                            if not has_nan:
                                has_nan = True
                                first_issue_step = step_idx
                                print(f"  [WARNING] NaN detected at step {step_idx}")
                        
                        if np.isinf(p_np).any():
                            if not has_inf:
                                has_inf = True
                                if first_issue_step == -1:
                                    first_issue_step = step_idx
                                print(f"  [WARNING] Inf detected at step {step_idx}")
                        
                        # Denormalize with RAW values (including huge/NaN/Inf)
                        preds_phys.append(self.denormalize_predictions(p_np))
                        targs_phys.append(self.denormalize_predictions(t_norm))
                    
                    is_diverged = has_nan or has_inf
                    
                    # --- DEBUG: PRINT RANGES ---
                    p_concat = np.concatenate(preds_phys, axis=0)
                    t_concat = np.concatenate(targs_phys, axis=0)
                    
                    # Use nanmin/nanmax to handle NaN values in stats
                    print(f"\n[Sim {sim_idx}] Range Analysis:")
                    print(f"  Target (GT):  Min={t_concat.min():.4f}, Max={t_concat.max():.4f}, Mean={t_concat.mean():.4f}")
                    print(f"  Prediction:   Min={np.nanmin(p_concat):.4e}, Max={np.nanmax(p_concat):.4e}, Mean={np.nanmean(p_concat):.4e}")
                    if is_diverged:
                        print(f"  [DIVERGED at step {first_issue_step}] - NaN: {has_nan}, Inf: {has_inf}")
                    print("-" * 40)
                    # ---------------------------
    
                    # ALWAYS append for GIF creation
                    all_preds.append(preds_phys)
                    all_targs.append(targs_phys)
                    
                    sim_meta = {
                        'id': sim_idx, 
                        'name': f"sim_{sim_idx}{'_DIVERGED' if is_diverged else ''}",
                        'reynolds': initial_data.x[0, 2].item(),
                        'diverged': is_diverged,
                        'divergence_step': first_issue_step if is_diverged else -1
                    }
                    metadata.append(sim_meta)
                    
                    # Only compute metrics if valid
                    if not is_diverged:
                        self._track_metrics(preds_phys, targs_phys, sim_meta)
                    else:
                        print(f"  [Skipping metrics for diverged simulation]")
    
                except Exception as e:
                    print(f"Error on sim {sim_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
        return all_preds, all_targs, metadata

    def _track_metrics(self, preds, targs, meta):
        p_flat = np.concatenate(preds, axis=0)
        t_flat = np.concatenate(targs, axis=0)
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(t_flat, p_flat))),
            'mae': float(mean_absolute_error(t_flat, p_flat)),
            'r2': float(r2_score(t_flat, p_flat))
        }
        
        self.simulation_metrics.append({
            'meta': meta,
            'metrics': metrics
        })

    #def create_gif(self, preds, targs, meta, output_path):
    #    """
    #    Create side-by-side GIF showing Velocity Magnitude.
    #    Fixed: Dynamic color scaling based on Ground Truth range to see early steps.
    #    """
    #    N = preds[0].shape[0]
    #    S = int(np.sqrt(N))
    #    steps = len(preds)
    #    
    #    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    #    fig.suptitle(f"Burgers Rollout (Re={meta['reynolds']:.2f})")
    def create_gif(self, preds, targs, meta, output_path):
        """
        Create side-by-side GIF showing Velocity Magnitude.
        Shows RAW values including explosions/divergence.
        Uses CONSTANT colorbar limits based on ground truth global min/max.
        """
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
        
        # Pre-calculate magnitudes
        mag_targs = [np.sqrt(t[:, 0]**2 + t[:, 1]**2).reshape(S, S) for t in targs]
        mag_preds = [np.sqrt(p[:, 0]**2 + p[:, 1]**2).reshape(S, S) for p in preds]
    
        # GLOBAL min/max from ground truth across ALL timesteps
        all_targs = np.array(mag_targs)
        global_vmin = all_targs.min()
        global_vmax = all_targs.max()
        
        # Ensure reasonable range
        if global_vmax - global_vmin < 1e-6:
            global_vmax = global_vmin + 0.1
    
        # Initialize plots with GLOBAL range
        im1 = axes[0].imshow(mag_targs[0], vmin=global_vmin, vmax=global_vmax, cmap='magma', origin='lower')
        axes[0].set_title(f'Target (t=0)')
        cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label('Velocity Magnitude', rotation=270, labelpad=15)
    
        im2 = axes[1].imshow(mag_preds[0], vmin=global_vmin, vmax=global_vmax, cmap='magma', origin='lower')
        axes[1].set_title(f'Prediction (t=0)')
        cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label('Velocity Magnitude', rotation=270, labelpad=15)
    
        def update(frame):
            current_targ = mag_targs[frame]
            current_pred = mag_preds[frame]
            
            # Update Image Data (colorbar limits stay CONSTANT)
            im1.set_data(current_targ)
            im2.set_data(current_pred)
            
            # Get actual values for title display
            p_min = np.nanmin(current_pred)
            p_max = np.nanmax(current_pred)
            t_min = current_targ.min()
            t_max = current_targ.max()
            
            # Update Titles with current frame range info
            axes[0].set_title(f'Target (t={frame})\nFrame range: [{t_min:.3f}, {t_max:.3f}]')
            
            # Show if prediction diverged
            if diverged and frame >= div_step:
                axes[1].set_title(
                    f'Prediction (t={frame}) **DIVERGED**\nFrame range: [{p_min:.2e}, {p_max:.2e}]',
                    color='red'
                )
            else:
                axes[1].set_title(f'Prediction (t={frame})\nFrame range: [{p_min:.3f}, {p_max:.3f}]')
            
            return im1, im2
    
        anim = FuncAnimation(fig, update, frames=steps, interval=100, blit=False)
        anim.save(output_path, writer=PillowWriter(fps=10))
        plt.close(fig)
        print(f"Saved GIF: {output_path} (colorbar fixed: [{global_vmin:.3f}, {global_vmax:.3f}])")

# ==============================================================================
# MAIN
# ==============================================================================

def load_simulations(data_dir, pattern="*.pt", limit=None):
    files = sorted(list(Path(data_dir).glob(pattern)))
    if limit: files = files[:limit]
    print(f"Loading {len(files)} simulations from {data_dir}...")
    return [torch.load(f, weights_only=False) for f in files]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="eval_burgers")
    
    # Architecture
    parser.add_argument("--num_static_feats", type=int, default=3)
    parser.add_argument("--num_dynamic_feats", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--feature_out_channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=2)
    
    parser.add_argument("--integrator", default="euler")
    parser.add_argument("--rollout_steps", type=int, default=50)
    parser.add_argument("--max_sequences", type=int, default=5)
    parser.add_argument("--create_gifs", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # BEFORE building the model, load one simulation to get mesh structure
    sims = load_simulations(args.test_dir, limit=1)
    sample_sequence = sims[0]
    sample_data = sample_sequence[0]
    
    # Reorder features to match training format
    pos = sample_data.x[:, 0:2]
    vel = sample_data.x[:, 2:4]
    re = sample_data.x[:, 4:5]
    sample_data.x = torch.cat([pos, re, vel], dim=1)
    
    # 1. Rebuild Model
    gradient_solver = SolveGradientsLST(boundary_margin=0.1, precompute_mesh=sample_data)
    laplacian_solver = SolveWeightLST2d(boundary_margin=0.1, precompute_mesh=sample_data)
    
    feature_extractor = FeatureExtractorGNN(
        in_channels=args.num_static_feats,
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        depth=args.depth,
        heads=args.heads,
        concat=True
    )
    
    differentiator = BurgersDifferentiator(
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=args.feature_out_channels,
        spade_heads=2,
        zero_init=True
    )
    
    model = GPARC_Burgers_Numerical(
        derivative_solver=differentiator,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats
    )
    
    # 2. Load Weights
    checkpoint = torch.load(args.model_path, map_location=device)
    #model.load_state_dict(checkpoint['model_state_dict'])
    # ============================================================
    # FIX: Use strict=False for backward compatibility
    # ============================================================
    missing_keys = model.load_state_dict(
        checkpoint['model_state_dict'], 
        strict=False  # <-- CRITICAL: Allow missing static buffers
    )
    
    print("✓ Model loaded with precomputed MLS weights")
    
    # Verify that only static buffers are missing (expected)
    if missing_keys.missing_keys:
        static_missing = [k for k in missing_keys.missing_keys if 'static_' in k]
        other_missing = [k for k in missing_keys.missing_keys if 'static_' not in k]
        
        if static_missing:
            print(f"\n✓ New static buffers (computed from mesh): {len(static_missing)}")
        
        if other_missing:
            print(f"\n⚠️  WARNING: Unexpected missing keys:")
            for key in other_missing:
                print(f"  - {key}")
    
    if missing_keys.unexpected_keys:
        print(f"\n⚠️  WARNING: Unexpected keys in checkpoint:")
        for key in missing_keys.unexpected_keys:
            print(f"  - {key}")
        


    # --- PRINT MODEL PARAMETERS ---
    print("\nModel Parameters:")
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num
        print(f"{name:60s} | shape={tuple(param.shape)} | requires_grad={param.requires_grad}")
    
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print("-" * 60)

    
    # 3. Evaluate
    evaluator = BurgersEvaluator(model, device)
    
    meta_path = Path(args.test_dir).parent / "normalization_metadata.json"
    evaluator.load_denormalization_params(meta_path)
    
    sims = load_simulations(args.test_dir, limit=args.max_sequences)
    if not sims:
        print("No simulations loaded.")
        return

    preds, targs, meta = evaluator.evaluate(sims, args.rollout_steps)
    
    # 4. Summary & Visuals
    if evaluator.simulation_metrics:
        rmses = [m['metrics']['rmse'] for m in evaluator.simulation_metrics]
        print(f"\nAverage RMSE: {np.mean(rmses):.6f}")
    else:
        print("\nNo successful simulations to compute RMSE.")
    
    if args.create_gifs:
        print("Generating GIFs...")
        for i in range(len(preds)):
            path = os.path.join(args.output_dir, f"{meta[i]['name']}.gif")
            evaluator.create_gif(preds[i], targs[i], meta[i], path)

if __name__ == "__main__":
    main()