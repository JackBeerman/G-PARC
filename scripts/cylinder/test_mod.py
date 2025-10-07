#!/usr/bin/env python3
"""
GPARC Model Rollout Evaluation Script for Cylinder Flow
========================================================

Evaluates trained GPARC model using rollout prediction where the model
receives only initial conditions and predicts the entire sequence.
Analyzes performance across different Reynolds numbers.

Usage:
    python evaluate_cylinder_rollout.py --model_path best_model.pth --test_dir /path/to/test --output_dir ./evaluation
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import json
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.Cylinder import KarmanVortexRolloutDataset, get_simulation_ids
from utilities.featureextractor import RobustFeatureExtractorGNN
from utilities.embed import SimulationConditionedLayerNorm, GlobalParameterProcessor
from differentiator.differentiator import DerivativeGNN
from integrator.integrator import IntegralGNN
from models.cylinder import GPARC


class CylinderRolloutEvaluator:
    """Evaluator for cylinder flow GPARC model using rollout prediction."""
    
    def __init__(self, model, device='cpu', denorm_params=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.denorm_params = denorm_params
        
        # All 7 dynamic features from your cylinder flow data
        self.var_names = [
            'pressure',
            'x_velocity', 
            'y_velocity',
            'z_velocity',
            'x_vorticity', 
            'y_vorticity',
            'z_vorticity'
        ]
        
        # Only use features that aren't skipped
        if hasattr(model, 'skip_dynamic_indices') and model.skip_dynamic_indices:
            self.var_names = [v for i, v in enumerate(self.var_names) if i not in model.skip_dynamic_indices]
        
        print(f"Evaluating {len(self.var_names)} variables: {self.var_names}")
        
        self.reynolds_metrics = defaultdict(list)
    
    def load_denormalization_params(self, metadata_file):
        """Load denormalization parameters from metadata JSON."""
        if not Path(metadata_file).exists():
            print(f"Warning: Metadata file not found: {metadata_file}")
            return
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.denorm_params = {}
        norm_params = metadata.get('normalization_params', {})
        
        # Load variable parameters
        for var in self.var_names:
            if var in norm_params:
                self.denorm_params[var] = norm_params[var]
        
        # Load Reynolds parameters
        if 'global_param_normalization' in metadata:
            if 'reynolds' in metadata['global_param_normalization']:
                self.denorm_params['reynolds'] = metadata['global_param_normalization']['reynolds']
        elif 'reynolds' in norm_params:
            self.denorm_params['reynolds'] = norm_params['reynolds']
        
        print(f"Loaded denormalization for: {list(self.denorm_params.keys())}")
    
    def denormalize_value(self, normalized_val, param_name):
        """Denormalize a value given its parameter name."""
        if self.denorm_params is None or param_name not in self.denorm_params:
            return normalized_val
        
        params = self.denorm_params[param_name]
        return normalized_val * (params['max'] - params['min']) + params['min']
    
    def denormalize_predictions(self, normalized_data, var_idx):
        """Denormalize predictions for a specific variable."""
        return self.denormalize_value(normalized_data, self.var_names[var_idx])
    
    def extract_global_params(self, data, sim_idx=None):
        """Extract Reynolds number from data object."""
        if hasattr(data, 'global_params'):
            if data.global_params.numel() == 1:
                return data
            data.global_params = data.global_params[0:1]
            return data
        
        if hasattr(data, 'reynolds'):
            data.global_params = data.reynolds.unsqueeze(0) if data.reynolds.dim() == 0 else data.reynolds[0:1]
            return data
        
        # Default fallback
        data.global_params = torch.tensor([100.0], device=data.x.device)
        if sim_idx is not None:
            print(f"Warning: No Reynolds found for sim {sim_idx}, using default 100")
        return data
    
    def process_targets(self, target_y, skip_indices):
        """Apply variable skipping to targets."""
        if not skip_indices:
            return target_y
        
        keep_indices = [i for i in range(target_y.shape[1]) if i not in skip_indices]
        return target_y[:, keep_indices]
    
    def generate_rollout(self, initial_data, rollout_steps):
        """Generate rollout predictions from initial conditions."""
        predictions = []
        F_prev = None
        
        # Extract and process global parameters
        global_attrs = initial_data.global_params.flatten()
        global_embed = self.model.global_processor(global_attrs)
        
        # Compute static features once
        static_feats = initial_data.x[:, :self.model.num_static_feats]
        edge_index = initial_data.edge_index
        learned_static = self.model.feature_extractor(static_feats, edge_index)
        learned_static = self.model.feature_norm(learned_static, global_attrs)
        
        # Rollout loop
        for step in range(rollout_steps):
            if step == 0:
                # First step: use ground truth
                all_dynamic = initial_data.x[:, 
                    self.model.num_static_feats:
                    self.model.num_static_feats + self.model.num_dynamic_feats + len(self.model.skip_dynamic_indices)
                ]
                keep_indices = [i for i in range(all_dynamic.shape[1]) if i not in self.model.skip_dynamic_indices]
                dynamic_feats = all_dynamic[:, keep_indices]
            else:
                # Use previous prediction
                dynamic_feats = F_prev
            
            # Normalize and add global context
            F_norm = self.model.derivative_norm(dynamic_feats, global_attrs)
            global_context = global_embed.unsqueeze(0).repeat(initial_data.num_nodes, 1)
            
            # Forward pass
            Fdot_input = torch.cat([learned_static, F_norm, global_context], dim=-1)
            Fdot = self.model.derivative_solver(Fdot_input, edge_index)
            Fint = self.model.integral_solver(Fdot, edge_index)
            F_pred = F_norm + Fint
            
            predictions.append(F_pred)
            F_prev = F_pred
        
        return predictions
    
    def evaluate_rollout_predictions(self, simulations, rollout_steps=10):
        """Generate and evaluate rollout predictions."""
        all_predictions = []
        all_targets = []
        metadata = []
        
        with torch.no_grad():
            for sim_idx, simulation in enumerate(tqdm(simulations, desc="Rollout predictions")):
                # Move to device
                for data in simulation:
                    data.x = data.x.to(self.device)
                    data.y = data.y.to(self.device)
                    data.edge_index = data.edge_index.to(self.device)
                    data = self.extract_global_params(data, sim_idx)
                    data.global_params = data.global_params.to(self.device)
                    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                        data.edge_attr = data.edge_attr.to(self.device)
                
                initial_data = simulation[0]
                
                # Extract Reynolds number
                norm_reynolds = float(initial_data.global_params[0])
                phys_reynolds = self.denormalize_value(norm_reynolds, 'reynolds')
                case_name = getattr(initial_data, 'case_name', f'sim_{sim_idx}')
                
                # Extract positions for visualization (static features should contain positions)
                pos_2d = initial_data.x[:, :self.model.num_static_feats].cpu().numpy()
                if pos_2d.shape[1] < 2:
                    print(f"Warning: Not enough position features for sim {sim_idx}")
                    pos_2d = None
                
                # Generate predictions
                actual_steps = min(rollout_steps, len(simulation))
                rollout_preds = self.generate_rollout(initial_data, actual_steps)
                
                # Collect targets
                rollout_targs = []
                for i in range(actual_steps):
                    target_y = simulation[i].y.cpu()
                    target_y = self.process_targets(target_y, self.model.skip_dynamic_indices)
                    rollout_targs.append(target_y)
                
                all_predictions.append([p.cpu() for p in rollout_preds])
                all_targets.append(rollout_targs)
                
                # Store metadata with positions
                reynolds_str = f"{int(phys_reynolds)}"
                sim_meta = {
                    'simulation_idx': sim_idx,
                    'case_name': case_name,
                    'reynolds': phys_reynolds,
                    'reynolds_str': reynolds_str,
                    'rollout_length': len(rollout_preds),
                    'available_targets': len(simulation),
                    'positions': pos_2d
                }
                metadata.append(sim_meta)
                
                # Track performance
                self._track_reynolds_performance(rollout_preds, rollout_targs, reynolds_str)
        
        print(f"\nGenerated predictions for {len(all_predictions)} simulations")
        return all_predictions, all_targets, metadata
    
    def _track_reynolds_performance(self, predictions, targets, reynolds_str):
        """Track metrics for specific Reynolds number."""
        all_preds = np.concatenate([p.cpu().numpy() for p in predictions], axis=0)
        all_targs = np.concatenate([t.cpu().numpy() for t in targets], axis=0)
        
        # Overall metrics
        pred_flat = all_preds.flatten()
        targ_flat = all_targs.flatten()
        overall_metrics = {
            'mse': float(mean_squared_error(targ_flat, pred_flat)),
            'mae': float(mean_absolute_error(targ_flat, pred_flat)),
            'rmse': float(np.sqrt(mean_squared_error(targ_flat, pred_flat))),
            'r2': float(r2_score(targ_flat, pred_flat))
        }
        
        # Per-variable metrics
        var_metrics = {}
        for i, var_name in enumerate(self.var_names):
            var_metrics[var_name] = {
                'mse': float(mean_squared_error(all_targs[:, i], all_preds[:, i])),
                'mae': float(mean_absolute_error(all_targs[:, i], all_preds[:, i])),
                'rmse': float(np.sqrt(mean_squared_error(all_targs[:, i], all_preds[:, i]))),
                'r2': float(r2_score(all_targs[:, i], all_preds[:, i]))
            }
        
        self.reynolds_metrics[reynolds_str].append({
            'overall': overall_metrics,
            'variables': var_metrics
        })
    
    def compute_rollout_metrics(self, predictions, targets):
        """Compute overall metrics."""
        all_preds = []
        all_targs = []
        
        for seq_pred, seq_targ in zip(predictions, targets):
            for step_pred, step_targ in zip(seq_pred, seq_targ):
                all_preds.append(step_pred.numpy())
                all_targs.append(step_targ.numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targs = np.concatenate(all_targs, axis=0)
        
        metrics = {}
        
        # Per-variable metrics
        for i, var_name in enumerate(self.var_names):
            pred_var = all_preds[:, i]
            targ_var = all_targs[:, i]
            
            pred_phys = self.denormalize_predictions(pred_var, i)
            targ_phys = self.denormalize_predictions(targ_var, i)
            
            metrics[var_name] = {
                'mse': float(mean_squared_error(targ_var, pred_var)),
                'mae': float(mean_absolute_error(targ_var, pred_var)),
                'rmse': float(np.sqrt(mean_squared_error(targ_var, pred_var))),
                'r2': float(r2_score(targ_var, pred_var)),
                'mse_physical': float(mean_squared_error(targ_phys, pred_phys)),
                'mae_physical': float(mean_absolute_error(targ_phys, pred_phys)),
                'rmse_physical': float(np.sqrt(mean_squared_error(targ_phys, pred_phys)))
            }
        
        # Overall metrics
        metrics['overall'] = {
            'mse': float(mean_squared_error(all_targs.flatten(), all_preds.flatten())),
            'mae': float(mean_absolute_error(all_targs.flatten(), all_preds.flatten())),
            'rmse': float(np.sqrt(mean_squared_error(all_targs.flatten(), all_preds.flatten()))),
            'r2': float(r2_score(all_targs.flatten(), all_preds.flatten()))
        }
        
        return metrics
    
    def analyze_reynolds_performance(self):
        """Analyze performance across Reynolds numbers."""
        analysis = {}
        
        for reynolds_str, metrics_list in self.reynolds_metrics.items():
            if not metrics_list:
                continue
            
            overall_metrics = [m['overall'] for m in metrics_list]
            var_metrics = {v: [m['variables'][v] for m in metrics_list if v in m['variables']] 
                          for v in self.var_names}
            
            analysis[reynolds_str] = {
                'num_simulations': len(metrics_list),
                'reynolds_value': float(reynolds_str),
                'overall': self._compute_statistics(overall_metrics),
                'variables': {v: self._compute_statistics(var_metrics[v]) for v in self.var_names if var_metrics[v]}
            }
        
        return analysis
    
    def _compute_statistics(self, metrics_list):
        """Compute statistics for metric list."""
        if not metrics_list:
            return {}
        
        stats = {}
        for metric_name in metrics_list[0].keys():
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            if values:
                stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        return stats
    
    def plot_reynolds_performance(self, figsize=(18, 12)):
        """Plot performance vs Reynolds number."""
        analysis = self.analyze_reynolds_performance()
        if not analysis:
            return None
        
        # Extract data
        reynolds_vals = []
        r2_means, r2_stds = [], []
        rmse_means, rmse_stds = [], []
        var_r2 = {v: {'means': [], 'stds': []} for v in self.var_names}
        var_rmse = {v: {'means': [], 'stds': []} for v in self.var_names}
        
        for re_str, data in sorted(analysis.items(), key=lambda x: x[1]['reynolds_value']):
            reynolds_vals.append(data['reynolds_value'])
            r2_means.append(data['overall']['r2']['mean'])
            r2_stds.append(data['overall']['r2']['std'])
            rmse_means.append(data['overall']['rmse']['mean'])
            rmse_stds.append(data['overall']['rmse']['std'])
            
            for v in self.var_names:
                if v in data['variables']:
                    var_r2[v]['means'].append(data['variables'][v]['r2']['mean'])
                    var_r2[v]['stds'].append(data['variables'][v]['r2']['std'])
                    var_rmse[v]['means'].append(data['variables'][v]['rmse']['mean'])
                    var_rmse[v]['stds'].append(data['variables'][v]['rmse']['std'])
        
        reynolds_vals = np.array(reynolds_vals)
        r2_means = np.array(r2_means)
        r2_stds = np.array(r2_stds)
        rmse_means = np.array(rmse_means)
        rmse_stds = np.array(rmse_stds)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Performance vs Reynolds Number', fontsize=16)
        
        # Overall R²
        ax = axes[0, 0]
        ax.errorbar(reynolds_vals, r2_means, yerr=r2_stds, fmt='o-', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel('Reynolds Number')
        ax.set_ylabel('R² Score')
        ax.set_title('Overall Performance (R²)')
        ax.grid(True, alpha=0.3)
        
        # Overall RMSE
        ax = axes[0, 1]
        ax.errorbar(reynolds_vals, rmse_means, yerr=rmse_stds, fmt='s-', capsize=5, linewidth=2, markersize=8, color='red')
        ax.set_xlabel('Reynolds Number')
        ax.set_ylabel('RMSE')
        ax.set_title('Overall Error (RMSE)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Variable R²
        ax = axes[1, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.var_names)))
        for i, v in enumerate(self.var_names):
            ax.errorbar(reynolds_vals, var_r2[v]['means'], yerr=var_r2[v]['stds'], 
                       fmt='o-', capsize=3, label=v.replace('_', ' ').title(), color=colors[i])
        ax.set_xlabel('Reynolds Number')
        ax.set_ylabel('R² Score')
        ax.set_title('Variable-Specific R²')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Variable RMSE
        ax = axes[1, 1]
        for i, v in enumerate(self.var_names):
            ax.errorbar(reynolds_vals, var_rmse[v]['means'], yerr=var_rmse[v]['stds'],
                       fmt='s-', capsize=3, label=v.replace('_', ' ').title(), color=colors[i])
        ax.set_xlabel('Reynolds Number')
        ax.set_ylabel('RMSE')
        ax.set_title('Variable-Specific RMSE')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_scatter(self, predictions, targets, figsize=(20, 14)):
        """Create prediction vs target scatter plots."""
        all_preds = np.concatenate([np.concatenate([p.numpy() for p in seq], 0) for seq in predictions], 0)
        all_targs = np.concatenate([np.concatenate([t.numpy() for t in seq], 0) for seq in targets], 0)
        
        n_vars = len(self.var_names)
        
        # Determine subplot layout
        if n_vars == 1:
            nrows, ncols = 1, 1
        elif n_vars == 2:
            nrows, ncols = 1, 2
        elif n_vars == 3:
            nrows, ncols = 1, 3
        elif n_vars <= 4:
            nrows, ncols = 2, 2
        elif n_vars <= 6:
            nrows, ncols = 2, 3
        else:
            nrows, ncols = 3, 3
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_vars == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        for i, var_name in enumerate(self.var_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            pred_var = self.denormalize_predictions(all_preds[:, i], i)
            targ_var = self.denormalize_predictions(all_targs[:, i], i)
            
            # Sample for visualization
            n_sample = min(3000, len(pred_var))
            idx = np.random.choice(len(pred_var), n_sample, replace=False)
            
            ax.scatter(targ_var[idx], pred_var[idx], alpha=0.5, s=1)
            
            min_val = min(targ_var.min(), pred_var.min())
            max_val = max(targ_var.max(), pred_var.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            r2 = r2_score(targ_var, pred_var)
            rmse = np.sqrt(mean_squared_error(targ_var, pred_var))
            ax.set_title(f'{var_name.replace("_", " ").title()}\nR²={r2:.3f}, RMSE={rmse:.3e}')
            ax.set_xlabel('Target')
            ax.set_ylabel('Prediction')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(n_vars, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_rollout_gif(self, predictions, targets, metadata, seq_idx, output_dir):
        """Create GIF showing ground truth (top row) and predictions (bottom row) side by side."""
        try:
            from matplotlib.animation import PillowWriter
        except ImportError:
            print("PillowWriter not available")
            return
        
        if seq_idx >= len(predictions):
            return
        
        seq_pred = predictions[seq_idx]
        seq_targ = targets[seq_idx]
        seq_meta = metadata[seq_idx]
        
        case_name = seq_meta['case_name']
        # Ensure Reynolds is denormalized
        reynolds_val = seq_meta['reynolds']  # This should already be physical value
        max_steps = min(len(seq_pred), len(seq_targ))
        
        if max_steps < 2:
            return
        
        # Get node positions
        n_nodes = seq_pred[0].shape[0]
        print(f"  Creating GIF for {case_name}: {n_nodes} nodes, {len(self.var_names)} variables")
        
        pos_2d = metadata[seq_idx].get('positions', None)
        if pos_2d is None:
            print(f"  Warning: No position data found, skipping GIF for {case_name}")
            return
        
        # Determine subplot layout: 2 rows (GT + Pred) x n_vars columns
        n_vars = len(self.var_names)
        nrows = 2
        ncols = n_vars
        
        # ===== COMBINED GT + PREDICTION GIF =====
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 10))
        fig.suptitle(f'{case_name} (Re={reynolds_val:.1f}) - Top: Ground Truth, Bottom: Predictions', fontsize=14)
        
        # Handle single variable case
        if n_vars == 1:
            axes = axes.reshape(2, 1)
        
        scatters_gt = []
        scatters_pred = []
        
        for i, var_name in enumerate(self.var_names):
            # Ground Truth (top row)
            ax_gt = axes[0, i]
            ax_gt.set_title(f'{var_name.replace("_", " ").title()} - GT')
            ax_gt.set_xlabel('X')
            ax_gt.set_ylabel('Y')
            ax_gt.set_aspect('equal')
            
            # Get GT data range (independent from predictions)
            all_gt_data = np.concatenate([self.denormalize_predictions(seq_targ[t][:, i].numpy(), i) 
                                          for t in range(max_steps)])
            vmin_gt, vmax_gt = np.percentile(all_gt_data, [2, 98])
            
            # Initial GT frame
            gt_data = self.denormalize_predictions(seq_targ[0][:, i].numpy(), i)
            scatter_gt = ax_gt.scatter(pos_2d[:, 0], pos_2d[:, 1], c=gt_data, 
                                       cmap='viridis', s=0.5, vmin=vmin_gt, vmax=vmax_gt, alpha=0.8)
            plt.colorbar(scatter_gt, ax=ax_gt, fraction=0.046)
            scatters_gt.append(scatter_gt)
            
            # Predictions (bottom row)
            ax_pred = axes[1, i]
            ax_pred.set_title(f'{var_name.replace("_", " ").title()} - Pred')
            ax_pred.set_xlabel('X')
            ax_pred.set_ylabel('Y')
            ax_pred.set_aspect('equal')
            
            # Get Prediction data range (independent from GT)
            all_pred_data = np.concatenate([self.denormalize_predictions(seq_pred[t][:, i].numpy(), i) 
                                            for t in range(max_steps)])
            vmin_pred, vmax_pred = np.percentile(all_pred_data, [2, 98])
            
            # Initial prediction frame
            pred_data = self.denormalize_predictions(seq_pred[0][:, i].numpy(), i)
            scatter_pred = ax_pred.scatter(pos_2d[:, 0], pos_2d[:, 1], c=pred_data, 
                                           cmap='viridis', s=0.5, vmin=vmin_pred, vmax=vmax_pred, alpha=0.8)
            plt.colorbar(scatter_pred, ax=ax_pred, fraction=0.046)
            scatters_pred.append(scatter_pred)
        
        plt.tight_layout()
        
        def animate_combined(frame):
            for i in range(len(self.var_names)):
                # Update GT
                gt_data = self.denormalize_predictions(seq_targ[frame][:, i].numpy(), i)
                scatters_gt[i].set_array(gt_data)
                
                # Update Prediction
                pred_data = self.denormalize_predictions(seq_pred[frame][:, i].numpy(), i)
                scatters_pred[i].set_array(pred_data)
            
            fig.suptitle(f'{case_name} (Re={reynolds_val:.1f}) - Step {frame}/{max_steps-1} - Top: GT, Bottom: Pred', fontsize=14)
            return scatters_gt + scatters_pred
        
        anim = FuncAnimation(fig, animate_combined, frames=max_steps, interval=500, blit=False)
        gif_path = output_dir / f'rollout_combined_{case_name}_Re{int(reynolds_val)}.gif'
        anim.save(gif_path, writer=PillowWriter(fps=2))
        plt.close(fig)
        print(f"  Saved: {gif_path.name}")
        
        # ===== OPTIONAL: ERROR GIF (kept separate as it's useful) =====
        fig, axes_err = plt.subplots(1, ncols, figsize=(5*ncols, 5))
        fig.suptitle(f'{case_name} (Re={reynolds_val:.1f}) - Absolute Error', fontsize=14)
        
        if n_vars == 1:
            axes_err = np.array([axes_err])
        
        scatters_err = []
        
        for i, var_name in enumerate(self.var_names):
            ax = axes_err[i]
            ax.set_title(f'{var_name.replace("_", " ").title()}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            
            # Calculate max error across all timesteps
            max_err = 0
            for t in range(max_steps):
                gt = self.denormalize_predictions(seq_targ[t][:, i].numpy(), i)
                pred = self.denormalize_predictions(seq_pred[t][:, i].numpy(), i)
                max_err = max(max_err, np.abs(pred - gt).max())
            
            # Initial error frame
            gt_data = self.denormalize_predictions(seq_targ[0][:, i].numpy(), i)
            pred_data = self.denormalize_predictions(seq_pred[0][:, i].numpy(), i)
            err_data = np.abs(pred_data - gt_data)
            
            scatter = ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c=err_data, 
                               cmap='Reds', s=0.5, vmin=0, vmax=max_err, alpha=0.8)
            plt.colorbar(scatter, ax=ax, fraction=0.046)
            scatters_err.append(scatter)
        
        plt.tight_layout()
        
        def animate_err(frame):
            for i in range(len(self.var_names)):
                gt_data = self.denormalize_predictions(seq_targ[frame][:, i].numpy(), i)
                pred_data = self.denormalize_predictions(seq_pred[frame][:, i].numpy(), i)
                err_data = np.abs(pred_data - gt_data)
                scatters_err[i].set_array(err_data)
            fig.suptitle(f'{case_name} (Re={reynolds_val:.1f}) - Error Step {frame}/{max_steps-1}', fontsize=14)
            return scatters_err
        
        anim_err = FuncAnimation(fig, animate_err, frames=max_steps, interval=500, blit=False)
        gif_path_err = output_dir / f'rollout_error_{case_name}_Re{int(reynolds_val)}.gif'
        anim_err.save(gif_path_err, writer=PillowWriter(fps=2))
        plt.close(fig)
        print(f"  Saved: {gif_path_err.name}")


def load_simulations(test_dir=None, test_files=None, max_files=None):
    """Load simulation files."""
    simulations = []
    
    if test_files:
        file_paths = [Path(f) for f in test_files]
        if max_files:
            file_paths = file_paths[:max_files]
        print(f"Loading {len(file_paths)} specific files")
    elif test_dir:
        file_paths = list(Path(test_dir).glob("*.pt"))
        if max_files:
            file_paths = file_paths[:max_files]
        print(f"Loading {len(file_paths)} files from {test_dir}")
    else:
        raise ValueError("Provide test_dir or test_files")
    
    for path in file_paths:
        try:
            sim_data = torch.load(path, weights_only=False)
            simulations.append(sim_data)
            print(f"  Loaded {path.name}: {len(sim_data)} timesteps")
        except Exception as e:
            print(f"  Error loading {path}: {e}")
    
    if not simulations:
        raise ValueError("No simulations loaded")
    
    return simulations


def select_diverse_simulations(predictions, targets, n_samples=3):
    """Select best, median, worst performing simulations."""
    performances = []
    
    for sim_idx, (pred_seq, targ_seq) in enumerate(zip(predictions, targets)):
        if not pred_seq or not targ_seq:
            continue
        
        total_mse = 0
        total_weight = 0
        for step_idx, (pred, targ) in enumerate(zip(pred_seq, targ_seq)):
            mse = np.mean((pred.numpy() - targ.numpy()) ** 2)
            weight = 1.0 + 0.1 * step_idx
            total_mse += mse * weight
            total_weight += weight
        
        if total_weight > 0:
            performances.append((sim_idx, total_mse / total_weight))
    
    if not performances:
        return list(range(min(n_samples, len(predictions))))
    
    performances.sort(key=lambda x: x[1])
    
    selected = []
    if performances:
        selected.append(performances[0][0])  # Best
    if len(performances) >= 2:
        selected.append(performances[len(performances)//2][0])  # Median
    if len(performances) >= 3:
        selected.append(performances[-1][0])  # Worst
    
    return selected[:n_samples]


def main():
    parser = argparse.ArgumentParser()
    
    # Required
    parser.add_argument("--model_path", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--test_dir")
    input_group.add_argument("--test_files", nargs='+')
    parser.add_argument("--output_dir", default="./rollout_evaluation")
    
    # Model architecture (must match training)
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=3)
    parser.add_argument("--skip_dynamic_indices", type=int, nargs='+', default=[])
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    #parser.add_argument("--pool_ratios", type=float, default=0.1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--deriv_hidden_channels", type=int, default=128)
    parser.add_argument("--deriv_num_layers", type=int, default=4)
    parser.add_argument("--deriv_heads", type=int, default=8)
    parser.add_argument("--deriv_dropout", type=float, default=0.3)
    parser.add_argument("--deriv_use_residual", action="store_true", default=True)
    
    parser.add_argument("--integral_hidden_channels", type=int, default=128)
    parser.add_argument("--integral_num_layers", type=int, default=4)
    parser.add_argument("--integral_heads", type=int, default=8)
    parser.add_argument("--integral_dropout", type=float, default=0.3)
    parser.add_argument("--integral_use_residual", action="store_true", default=True)
    
    # Evaluation
    parser.add_argument("--rollout_steps", type=int, default=10)
    parser.add_argument("--max_sequences", type=int, default=30)
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Device: {device}")
    print(f"Loading model from: {args.model_path}")
    
    # Build model
    checkpoint = torch.load(args.model_path, map_location=device)
    
    global_embed_dim = 64
    
    feature_extractor = RobustFeatureExtractorGNN(
        in_channels=args.num_static_feats,
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        depth=args.depth,
        #pool_ratios=args.pool_ratios,
        heads=args.heads,
        #concat=True,
        dropout=args.dropout
    )
    
    deriv_in_channels = args.feature_out_channels + args.num_dynamic_feats + global_embed_dim
    
    derivative_solver = DerivativeGNN(
        in_channels=deriv_in_channels,
        hidden_channels=args.deriv_hidden_channels,
        out_channels=args.num_dynamic_feats,
        num_layers=args.deriv_num_layers,
        heads=args.deriv_heads,
        concat=True,
        dropout=args.deriv_dropout,
        use_residual=args.deriv_use_residual
    )
    
    integral_solver = IntegralGNN(
        in_channels=args.num_dynamic_feats,
        hidden_channels=args.integral_hidden_channels,
        out_channels=args.num_dynamic_feats,
        num_layers=args.integral_num_layers,
        heads=args.integral_heads,
        concat=True,
        dropout=args.integral_dropout,
        use_residual=args.integral_use_residual
    )
    
    model = GPARC(
        feature_extractor=feature_extractor,
        derivative_solver=derivative_solver,
        integral_solver=integral_solver,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        skip_dynamic_indices=args.skip_dynamic_indices,
        feature_out_channels=args.feature_out_channels
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Model loaded successfully")
    
    # Load test data
    simulations = load_simulations(
        test_dir=args.test_dir,
        test_files=args.test_files,
        max_files=args.max_sequences
    )
    
    # Initialize evaluator
    evaluator = CylinderRolloutEvaluator(model, device)
    
    # Load denormalization parameters
    if args.test_dir:
        norm_file = Path(args.test_dir).parent / 'normalization_metadata.json'
    elif args.test_files:
        norm_file = Path(args.test_files[0]).parent / 'normalization_metadata.json'
        if not norm_file.exists():
            norm_file = Path(args.test_files[0]).parent.parent / 'normalization_metadata.json'
    else:
        norm_file = None
    
    if norm_file and norm_file.exists():
        evaluator.load_denormalization_params(norm_file)
    else:
        print(f"Warning: Normalization metadata not found at {norm_file}")
    
    # Generate predictions
    print(f"\nGenerating rollout predictions ({args.rollout_steps} steps)...")
    predictions, targets, metadata = evaluator.evaluate_rollout_predictions(
        simulations, rollout_steps=args.rollout_steps
    )
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = evaluator.compute_rollout_metrics(predictions, targets)
    reynolds_analysis = evaluator.analyze_reynolds_performance()
    
    # Select simulations for visualization
    selected_indices = select_diverse_simulations(predictions, targets, n_samples=3)
    print(f"\nSelected simulations for visualization: {selected_indices}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Reynolds performance plot
    fig = evaluator.plot_reynolds_performance()
    if fig:
        fig.savefig(output_path / 'reynolds_performance.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  Saved: reynolds_performance.png")
    
    # Scatter plot
    fig = evaluator.plot_scatter(predictions, targets)
    fig.savefig(output_path / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: prediction_scatter.png")
    
    # Create GIFs for selected simulations
    print("\nCreating GIFs...")
    for sim_idx in selected_indices:
        evaluator.create_rollout_gif(predictions, targets, metadata, sim_idx, output_path)
    
    # Save results
    # Convert numpy arrays to lists for JSON serialization
    metadata_serializable = []
    for meta in metadata:
        meta_copy = meta.copy()
        if 'positions' in meta_copy and meta_copy['positions'] is not None:
            meta_copy['positions'] = meta_copy['positions'].tolist()
        metadata_serializable.append(meta_copy)
    
    results = {
        'metrics': metrics,
        'reynolds_analysis': reynolds_analysis,
        'metadata': metadata_serializable,
        'model_info': {
            'model_path': str(args.model_path),
            'num_simulations': len(predictions),
            'rollout_steps': args.rollout_steps,
            'num_dynamic_feats': args.num_dynamic_feats,
            'skip_dynamic_indices': args.skip_dynamic_indices
        }
    }
    
    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print("\nOverall Metrics:")
    print(f"  R² Score: {metrics['overall']['r2']:.4f}")
    print(f"  RMSE:     {metrics['overall']['rmse']:.6f}")
    print(f"  MAE:      {metrics['overall']['mae']:.6f}")
    
    print("\nPer-Variable Performance:")
    for var_name in evaluator.var_names:
        if var_name in metrics:
            print(f"\n  {var_name.replace('_', ' ').title()}:")
            print(f"    R²:   {metrics[var_name]['r2']:.4f}")
            print(f"    RMSE: {metrics[var_name]['rmse']:.6f}")
    
    if reynolds_analysis:
        print("\nReynolds Number Analysis:")
        for re_str, data in sorted(reynolds_analysis.items(), key=lambda x: x[1]['reynolds_value']):
            print(f"\n  Re={data['reynolds_value']:.0f} ({data['num_simulations']} sims):")
            print(f"    R²:   {data['overall']['r2']['mean']:.4f} ± {data['overall']['r2']['std']:.4f}")
            print(f"    RMSE: {data['overall']['rmse']['mean']:.6f} ± {data['overall']['rmse']['std']:.6f}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()