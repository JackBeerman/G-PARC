#!/usr/bin/env python3
"""
Tennis Motion Prediction Model Rollout Evaluation Script
========================================================

This script evaluates a trained tennis GPARC model using rollout prediction mode,
with support for multi-step context evaluation.

Usage:
    # Single-step rollout (give t=0, predict all):
    python evaluate_tennis_model.py --model_path best_model.pth --test_dir /path/to/test --num_context_steps 0
    
    # Multi-step context (give t=0,1,2, predict rest):
    python evaluate_tennis_model.py --model_path best_model.pth --test_dir /path/to/test --num_context_steps 3
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict
import json
import pickle

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project paths
debug_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, debug_path)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.featureextractor import FeatureExtractorGNN
from utilities.embed import GlobalParameterProcessor
from differentiator.differentiator import DerivativeGNN
from integrator.integrator import IntegralGNN
from models.tennisv2 import GPARC


class TennisRolloutEvaluator:
    """Evaluator for tennis motion models with context support."""
    
    def __init__(self, model, device='cpu', denormalization_params=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.denorm_params = denormalization_params
        
        self.joint_names = [
            "Pelvis", "R Hip", "R Knee", "R Ankle", "L Hip", "L Knee", "L Ankle", 
            "Spine", "Thorax", "Neck", "Head", "L Shoulder", "L Elbow", "L Wrist", 
            "R Shoulder", "R Elbow", "R Wrist"
        ]
        
        self.skeleton_edges = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), 
            (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), 
            (8, 14), (14, 15), (15, 16)
        ]
        
        self.player_metrics = defaultdict(list)
    
    def load_denormalization_params(self, norm_stats_file):
        """Load denormalization parameters."""
        if not Path(norm_stats_file).exists():
            print(f"Warning: Normalization stats not found: {norm_stats_file}")
            return
            
        with open(norm_stats_file, 'rb') as f:
            self.denorm_params = pickle.load(f)
        print(f"Loaded denormalization parameters")
    
    def denormalize_positions(self, normalized_positions):
        """Convert normalized positions back to physical units."""
        if self.denorm_params is None or 'position_mean' not in self.denorm_params:
            return normalized_positions
        
        original_shape = normalized_positions.shape
        pos_flat = normalized_positions.reshape(-1, 3)
        
        denormalized = (pos_flat * self.denorm_params['position_std'] + 
                       self.denorm_params['position_mean'])
        
        return denormalized.reshape(original_shape)
    
    def _extract_global_attributes(self, data, serve_idx=None):
        """Extract global attributes from data object."""
        if hasattr(data, 'server_id'):
            data.global_server_id = data.server_id.unsqueeze(0) if data.server_id.dim() == 0 else data.server_id
            data.global_serve_number = data.serve_number.unsqueeze(0) if data.serve_number.dim() == 0 else data.serve_number
            data.global_set_number = data.set_number.unsqueeze(0) if data.set_number.dim() == 0 else data.set_number
            data.global_game_number = data.game_number.unsqueeze(0) if data.game_number.dim() == 0 else data.game_number
            data.global_point_number = data.point_number.unsqueeze(0) if data.point_number.dim() == 0 else data.point_number
        elif not hasattr(data, 'global_server_id'):
            data.global_server_id = torch.tensor([0], device=data.x.device, dtype=torch.long)
            data.global_serve_number = torch.tensor([1.0], device=data.x.device)
            data.global_set_number = torch.tensor([1.0], device=data.x.device)
            data.global_game_number = torch.tensor([1.0], device=data.x.device)
            data.global_point_number = torch.tensor([1.0], device=data.x.device)
        
        return data
    
    def generate_rollout_with_context(self, initial_sequence, rollout_steps, num_context_steps=0):
        """
        Generate rollout prediction with optional multi-step context.
        
        Args:
            initial_sequence: List of Data objects
            rollout_steps: Total steps to predict
            num_context_steps: Number of context steps (0 = single-step rollout)
        """
        predictions = []
        F_prev = None
        
        first_data = initial_sequence[0]
        global_attrs = torch.stack([
            first_data.global_server_id.flatten()[0].float(),
            first_data.global_serve_number.flatten()[0],
            first_data.global_set_number.flatten()[0],
            first_data.global_game_number.flatten()[0],
            first_data.global_point_number.flatten()[0]
        ])
        
        global_embed = self.model.global_processor(global_attrs)
        edge_index_0 = first_data.edge_index
        
        # Extract initial features
        all_dynamic_feats_0 = first_data.x[:, 
            self.model.num_static_feats:
            self.model.num_static_feats + self.model.num_dynamic_feats + len(self.model.skip_dynamic_indices)
        ]
        keep_indices = [i for i in range(all_dynamic_feats_0.shape[1]) if i not in self.model.skip_dynamic_indices]
        initial_dynamic_feats = all_dynamic_feats_0[:, keep_indices]
        
        learned_features = self.model.feature_extractor(initial_dynamic_feats, edge_index_0)
        learned_features = self.model.feature_norm(learned_features, global_attrs)
        
        # Process context steps (use ground truth)
        for ctx_step in range(num_context_steps):
            if ctx_step >= len(initial_sequence):
                break
                
            data_t = initial_sequence[ctx_step]
            
            all_dynamic_feats = data_t.x[:, 
                self.model.num_static_feats:
                self.model.num_static_feats + self.model.num_dynamic_feats + len(self.model.skip_dynamic_indices)
            ]
            keep_indices = [i for i in range(all_dynamic_feats.shape[1]) if i not in self.model.skip_dynamic_indices]
            dynamic_feats_t = all_dynamic_feats[:, keep_indices]
            
            F_prev_used = self.model.derivative_norm(dynamic_feats_t, global_attrs)
            global_context = global_embed.unsqueeze(0).repeat(data_t.num_nodes, 1)
            
            Fdot_input = torch.cat([learned_features, F_prev_used, global_context], dim=-1)
            Fdot = self.model.derivative_solver(Fdot_input, edge_index_0)
            Fint = self.model.integral_solver(Fdot, edge_index_0)
            F_pred = F_prev_used + Fint
            
            predictions.append(F_pred)
            F_prev = F_pred
        
        # Predict remaining steps
        remaining_steps = rollout_steps - num_context_steps
        for step in range(remaining_steps):
            dynamic_feats_t = F_prev
            F_prev_used = self.model.derivative_norm(dynamic_feats_t, global_attrs)
            global_context = global_embed.unsqueeze(0).repeat(first_data.num_nodes, 1)
            
            Fdot_input = torch.cat([learned_features, F_prev_used, global_context], dim=-1)
            Fdot = self.model.derivative_solver(Fdot_input, edge_index_0)
            Fint = self.model.integral_solver(Fdot, edge_index_0)
            F_pred = F_prev_used + Fint
            
            predictions.append(F_pred)
            F_prev = F_pred
        
        return predictions
    
    def evaluate_rollout_predictions_with_context(self, serves, rollout_steps=10, num_context_steps=0):
        """Generate rollout predictions with context support."""
        all_predictions = []
        all_targets = []
        metadata = []
        
        with torch.no_grad():
            for serve_idx, serve in enumerate(tqdm(serves, desc="Generating predictions")):
                
                for data in serve:
                    data.x = data.x.to(self.device)
                    data.y = data.y.to(self.device)
                    data.edge_index = data.edge_index.to(self.device)
                    data = self._extract_global_attributes(data, serve_idx)
                    data.global_server_id = data.global_server_id.to(self.device)
                    data.global_serve_number = data.global_serve_number.to(self.device)
                    data.global_set_number = data.global_set_number.to(self.device)
                    data.global_game_number = data.global_game_number.to(self.device)
                    data.global_point_number = data.global_point_number.to(self.device)
                
                first_data = serve[0]
                server_id = int(first_data.global_server_id[0])
                serve_name = f"serve_{serve_idx}"
                
                max_available_steps = len(serve)
                actual_rollout_steps = min(rollout_steps, max_available_steps)
                
                if num_context_steps >= actual_rollout_steps:
                    effective_context_steps = max(1, actual_rollout_steps - 1)
                else:
                    effective_context_steps = num_context_steps
                
                rollout_predictions = self.generate_rollout_with_context(
                    serve, 
                    rollout_steps=actual_rollout_steps,
                    num_context_steps=effective_context_steps
                )
                
                rollout_targets = [serve[i].y.cpu() for i in range(actual_rollout_steps)]
                
                all_predictions.append([pred.cpu() for pred in rollout_predictions])
                all_targets.append(rollout_targets)
                
                serve_metadata = {
                    'serve_idx': serve_idx,
                    'serve_name': serve_name,
                    'server_id': server_id,
                    'rollout_length': len(rollout_predictions),
                    'num_context_steps': effective_context_steps
                }
                metadata.append(serve_metadata)
                
                self._track_player_performance(rollout_predictions, rollout_targets, server_id)
        
        print(f"\nGenerated predictions for {len(all_predictions)} serves")
        return all_predictions, all_targets, metadata
    
    def _track_player_performance(self, predictions, targets, player_id):
        """Track performance metrics per player."""
        all_preds = np.stack([p.cpu().numpy() for p in predictions], axis=0)
        all_targs = np.stack([t.cpu().numpy() for t in targets], axis=0)
        
        pred_flat = all_preds.flatten()
        target_flat = all_targs.flatten()
        
        overall_metrics = {
            'mse': float(mean_squared_error(target_flat, pred_flat)),
            'mae': float(mean_absolute_error(target_flat, pred_flat)),
            'rmse': float(np.sqrt(mean_squared_error(target_flat, pred_flat))),
            'r2': float(r2_score(target_flat, pred_flat))
        }
        
        self.player_metrics[player_id].append({'overall': overall_metrics})
    
    def compute_rollout_metrics(self, predictions, targets):
        """Compute overall metrics."""
        all_preds = []
        all_targs = []
        
        for seq_pred, seq_targ in zip(predictions, targets):
            for step_pred, step_targ in zip(seq_pred, seq_targ):
                all_preds.append(step_pred.numpy())
                all_targs.append(step_targ.numpy())
        
        all_preds = np.stack(all_preds, axis=0)
        all_targs = np.stack(all_targs, axis=0)
        
        pred_flat = all_preds.flatten()
        target_flat = all_targs.flatten()
        
        metrics = {
            'overall': {
                'mse': float(mean_squared_error(target_flat, pred_flat)),
                'mae': float(mean_absolute_error(target_flat, pred_flat)),
                'rmse': float(np.sqrt(mean_squared_error(target_flat, pred_flat))),
                'r2': float(r2_score(target_flat, pred_flat))
            }
        }
        
        return metrics
    
    def plot_rollout_evolution(self, predictions, targets, metadata, seq_idx=0, figsize=(20, 10)):
        """Plot how prediction accuracy evolves over rollout timesteps."""
        if seq_idx >= len(predictions):
            return None
        
        seq_pred = predictions[seq_idx]
        seq_targ = targets[seq_idx]
        seq_meta = metadata[seq_idx]
        
        serve_name = seq_meta['serve_name']
        player_id = seq_meta['server_id']
        max_steps = min(len(seq_pred), len(seq_targ))
        
        # Calculate errors over time
        timestep_errors = []
        
        for t in range(max_steps):
            pred_array = seq_pred[t].numpy()
            targ_array = seq_targ[t].numpy()
            
            # Compute error per joint
            joint_errors = []
            for j in range(pred_array.shape[0]):
                if pred_array.shape[1] >= 3:
                    pred_joint = pred_array[j, :3]
                    targ_joint = targ_array[j, :3]
                else:
                    pred_joint = pred_array[j, :]
                    targ_joint = targ_array[j, :]
                
                error = np.linalg.norm(pred_joint - targ_joint)
                joint_errors.append(error)
            
            overall_error = np.mean(joint_errors)
            timestep_errors.append(overall_error)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle(f'Rollout Evolution: {serve_name} (Player {player_id})', fontsize=16)
        
        ax.plot(range(max_steps), timestep_errors, 'bo-', linewidth=2, markersize=6)
        ax.set_xlabel('Rollout Timestep')
        ax.set_ylabel('Mean Position Error')
        ax.set_title('Overall Prediction Error vs Time')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at context boundary if applicable
        if seq_meta.get('num_context_steps', 0) > 0:
            ctx_steps = seq_meta['num_context_steps']
            ax.axvline(x=ctx_steps, color='r', linestyle='--', linewidth=2, 
                      label=f'Context boundary (t={ctx_steps})')
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_summary_plot(self, predictions, targets, metadata, figsize=(15, 10)):
        """Create a summary plot with multiple metrics."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Tennis Rollout Evaluation Summary', fontsize=16)
        
        # Plot 1: Error distribution across all serves
        ax = axes[0, 0]
        all_errors = []
        for seq_pred, seq_targ in zip(predictions, targets):
            for step_pred, step_targ in zip(seq_pred, seq_targ):
                pred_array = step_pred.numpy()
                targ_array = step_targ.numpy()
                error = np.linalg.norm(pred_array - targ_array, axis=-1).mean()
                all_errors.append(error)
        
        ax.hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Mean Position Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.axvline(np.mean(all_errors), color='r', linestyle='--', 
                  label=f'Mean: {np.mean(all_errors):.4f}')
        ax.legend()
        
        # Plot 2: Error vs timestep (averaged across all serves)
        ax = axes[0, 1]
        max_len = max(len(p) for p in predictions)
        timestep_errors = [[] for _ in range(max_len)]
        
        for seq_pred, seq_targ in zip(predictions, targets):
            for t, (pred, targ) in enumerate(zip(seq_pred, seq_targ)):
                error = np.linalg.norm(pred.numpy() - targ.numpy(), axis=-1).mean()
                timestep_errors[t].append(error)
        
        mean_errors = [np.mean(errors) if errors else 0 for errors in timestep_errors]
        std_errors = [np.std(errors) if errors else 0 for errors in timestep_errors]
        
        ax.plot(range(len(mean_errors)), mean_errors, 'b-', linewidth=2)
        ax.fill_between(range(len(mean_errors)), 
                        np.array(mean_errors) - np.array(std_errors),
                        np.array(mean_errors) + np.array(std_errors),
                        alpha=0.3)
        ax.set_xlabel('Rollout Timestep')
        ax.set_ylabel('Mean Position Error')
        ax.set_title('Error Evolution (Mean ± Std)')
        ax.grid(True, alpha=0.3)
        
        # Add context boundary if applicable
        if metadata[0].get('num_context_steps', 0) > 0:
            ctx_steps = metadata[0]['num_context_steps']
            ax.axvline(x=ctx_steps, color='r', linestyle='--', linewidth=2,
                      label=f'Context boundary')
            ax.legend()
        
        # Plot 3: Number of serves by player
        ax = axes[1, 0]
        player_counts = {}
        for meta in metadata:
            pid = meta['server_id']
            player_counts[pid] = player_counts.get(pid, 0) + 1
        
        players = sorted(player_counts.keys())
        counts = [player_counts[p] for p in players]
        
        ax.bar(range(len(players)), counts)
        ax.set_xlabel('Player ID')
        ax.set_ylabel('Number of Serves')
        ax.set_title('Serves per Player')
        ax.set_xticks(range(len(players)))
        ax.set_xticklabels(players)
        
        # Plot 4: Overall metrics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Compute overall metrics
        all_preds_flat = np.concatenate([p.numpy().flatten() for seq in predictions for p in seq])
        all_targs_flat = np.concatenate([t.numpy().flatten() for seq in targets for t in seq])
        
        r2 = r2_score(all_targs_flat, all_preds_flat)
        rmse = np.sqrt(mean_squared_error(all_targs_flat, all_preds_flat))
        mae = mean_absolute_error(all_targs_flat, all_preds_flat)
        
        metrics_text = f"""
        Overall Performance Metrics:
        
        R² Score:  {r2:.4f}
        RMSE:      {rmse:.6f}
        MAE:       {mae:.6f}
        
        Dataset Info:
        Total Serves:  {len(predictions)}
        Total Steps:   {sum(len(p) for p in predictions)}
        Context Steps: {metadata[0].get('num_context_steps', 0)}
        """
        
        ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig


def load_test_serves(test_dir=None, test_files=None, max_files=None):
    """Load test serve files."""
    serves = []
    
    if test_files is not None:
        test_file_paths = [Path(f) for f in test_files]
        if max_files:
            test_file_paths = test_file_paths[:max_files]
        
        for file_path in test_file_paths:
            if not file_path.exists():
                continue
            serve_data = torch.load(file_path, weights_only=False)
            serves.append(serve_data)
    
    elif test_dir is not None:
        test_dir = Path(test_dir)
        test_file_paths = list(test_dir.glob("*.pt"))
        if max_files:
            test_file_paths = test_file_paths[:max_files]
        
        for file_path in test_file_paths:
            serve_data = torch.load(file_path, weights_only=False)
            serves.append(serve_data)
    
    return serves


def evaluate_tennis_rollout(model_path, test_dir, test_files, output_dir, args):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    trained_context_steps = checkpoint.get('num_context_steps', 0)
    if trained_context_steps > 0:
        print(f"Model was trained with {trained_context_steps} context steps")
    
    # Create model
    global_embed_dim = 64
    feature_extractor = FeatureExtractorGNN(
        in_channels=args.num_dynamic_feats,
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        depth=args.depth,
        pool_ratios=args.pool_ratios,
        heads=args.heads,
        concat=True,
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
        skip_dynamic_indices=getattr(args, 'skip_dynamic_indices', []),
        feature_out_channels=args.feature_out_channels
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Model loaded successfully!")
    
    # Load serves
    serves = load_test_serves(
        test_dir=test_dir,
        test_files=test_files,
        max_files=args.max_sequences
    )
    print(f"Loaded {len(serves)} test serves")
    
    # Initialize evaluator
    evaluator = TennisRolloutEvaluator(model, device)
    
    # Load normalization params
    if test_dir:
        norm_stats_file = Path(test_dir).parent / 'normalization_stats.pkl'
    elif test_files:
        norm_stats_file = Path(test_files[0]).parent.parent / 'normalization_stats.pkl'
    else:
        norm_stats_file = None
    
    if norm_stats_file and norm_stats_file.exists():
        evaluator.load_denormalization_params(norm_stats_file)
    
    # Generate predictions
    print(f"\nGenerating predictions ({args.rollout_steps} steps, {args.num_context_steps} context)...")
    predictions, targets, metadata = evaluator.evaluate_rollout_predictions_with_context(
        serves, 
        rollout_steps=args.rollout_steps,
        num_context_steps=args.num_context_steps
    )
    
    # Compute metrics
    print("Computing metrics...")
    metrics = evaluator.compute_rollout_metrics(predictions, targets)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Summary plot
    fig_summary = evaluator.create_summary_plot(predictions, targets, metadata)
    fig_summary.savefig(output_path / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig_summary)
    print("  ✓ Saved evaluation_summary.png")
    
    # Individual serve plots (first 3)
    for i in range(min(3, len(predictions))):
        fig = evaluator.plot_rollout_evolution(predictions, targets, metadata, seq_idx=i)
        if fig:
            fig.savefig(output_path / f'rollout_evolution_serve_{i}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
    print(f"  ✓ Saved {min(3, len(predictions))} individual rollout plots")
    
    # Save results
    results = {
        'metrics': metrics,
        'metadata': metadata,
        'model_info': {
            'model_path': str(model_path),
            'test_serves': len(predictions),
            'rollout_steps': args.rollout_steps,
            'num_context_steps': args.num_context_steps,
            'trained_with_context_steps': trained_context_steps
        }
    }
    
    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to: {output_path}")
    print(f"\nOverall Performance:")
    print(f"  R² = {metrics['overall']['r2']:.4f}")
    print(f"  RMSE = {metrics['overall']['rmse']:.6f}")
    print(f"  MAE = {metrics['overall']['mae']:.6f}")
    
    return metrics, evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tennis GPARC model")
    
    parser.add_argument("--model_path", type=str, required=True)
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--test_dir", type=str)
    input_group.add_argument("--test_files", type=str, nargs='+')
    
    parser.add_argument("--output_dir", type=str, default="./evaluation")
    parser.add_argument("--num_static_feats", type=int, default=0)
    parser.add_argument("--num_dynamic_feats", type=int, default=6)
    parser.add_argument("--skip_dynamic_indices", type=int, nargs='+', default=[])
    
    # Feature extractor
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--pool_ratios", type=float, default=0.1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # Derivative solver
    parser.add_argument("--deriv_hidden_channels", type=int, default=128)
    parser.add_argument("--deriv_num_layers", type=int, default=4)
    parser.add_argument("--deriv_heads", type=int, default=8)
    parser.add_argument("--deriv_dropout", type=float, default=0.3)
    parser.add_argument("--deriv_use_residual", action="store_true", default=True)
    
    # Integral solver
    parser.add_argument("--integral_hidden_channels", type=int, default=128)
    parser.add_argument("--integral_num_layers", type=int, default=4)
    parser.add_argument("--integral_heads", type=int, default=8)
    parser.add_argument("--integral_dropout", type=float, default=0.3)
    parser.add_argument("--integral_use_residual", action="store_true", default=True)
    
    # Evaluation settings
    parser.add_argument("--max_sequences", type=int, default=30)
    parser.add_argument("--rollout_steps", type=int, default=10)
    parser.add_argument("--num_context_steps", type=int, default=0,
                       help="Context steps (0=single-step, 3+=multi-step)")
    
    args = parser.parse_args()
    
    evaluate_tennis_rollout(args.model_path, args.test_dir, args.test_files, 
                           args.output_dir, args)


if __name__ == "__main__":
    main()