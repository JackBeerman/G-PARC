#!/usr/bin/env python3
"""
Tennis Motion Prediction Model Rollout Evaluation Script
========================================================

This script evaluates a trained tennis GPARC model using rollout prediction mode,
where the model receives only initial conditions and predicts the entire sequence.

Usage:
    # Test all files in a directory:
    python evaluate_tennis_model.py --model_path best_model.pth --test_dir /path/to/test --output_dir ./evaluation
    
    # Test specific serve files:
    python evaluate_tennis_model.py --model_path best_model.pth --test_files serve1.pt serve2.pt serve3.pt --output_dir ./evaluation
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Union, Dict, Any
import pickle
from collections import defaultdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GraphUNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import imageio
import io
from mpl_toolkits.mplot3d import Axes3D

debug_path = os.path.join(os.path.dirname(__file__), '..', '..')
print(f"Script location: {__file__}")
print(f"Adding to path: {os.path.abspath(debug_path)}")
print(f"Files in that directory: {os.listdir(debug_path) if os.path.exists(debug_path) else 'Directory not found'}")
sys.path.insert(0, debug_path)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
#from data.tennisDataset import TennisServeRolloutDataset, get_serve_ids, create_datasets_from_folders
# Import the tennis model components
from utilities.featureextractor import FeatureExtractorGNN
from utilities.embed import SimulationConditionedLayerNorm, GlobalParameterProcessor, GlobalModulatedGNN
from utilities.trainer import train_and_validate, load_model, plot_loss_curves
from differentiator.differentiator import DerivativeGNN
from integrator.integrator import IntegralGNN
from models.tennisv2 import GPARC

################################################################################
# TENNIS MOTION ROLLOUT EVALUATOR
################################################################################

class TennisRolloutEvaluator:
    """
    Evaluator for tennis motion models using rollout prediction mode.
    Adapted from shock tube evaluator for tennis serve motion prediction.
    """
    
    def __init__(self, model, device='cpu', denormalization_params=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.denorm_params = denormalization_params
        
        # Tennis joint names
        self.joint_names = [
            "Pelvis", "R Hip", "R Knee", "R Ankle", "L Hip", "L Knee", "L Ankle", 
            "Spine", "Thorax", "Neck", "Head", "L Shoulder", "L Elbow", "L Wrist", 
            "R Shoulder", "R Elbow", "R Wrist"
        ]
        
        # Tennis skeleton connections
        self.skeleton_edges = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), 
            (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), 
            (8, 14), (14, 15), (15, 16)
        ]
        
        # For player-specific performance tracking
        self.player_metrics = defaultdict(list)
    
    def load_denormalization_params(self, norm_stats_file):
        """Load denormalization parameters from normalization stats."""
        if not Path(norm_stats_file).exists():
            print(f"Warning: Normalization stats file not found: {norm_stats_file}")
            return
            
        with open(norm_stats_file, 'rb') as f:
            self.denorm_params = pickle.load(f)
        
        print(f"Loaded denormalization parameters: {list(self.denorm_params.keys())}")
    
    def denormalize_positions(self, normalized_positions):
        """Convert normalized positions back to physical units."""
        if self.denorm_params is None or 'position_mean' not in self.denorm_params:
            return normalized_positions
        
        original_shape = normalized_positions.shape
        pos_flat = normalized_positions.reshape(-1, 3)
        
        # Denormalize using stored statistics
        denormalized = (pos_flat * self.denorm_params['position_std'] + 
                       self.denorm_params['position_mean'])
        
        return denormalized.reshape(original_shape)
    
    def _extract_global_attributes(self, data, serve_idx=None):
        """Extract global attributes from tennis data object."""
        # Check for already processed global attributes
        if hasattr(data, 'global_server_id') and hasattr(data, 'global_serve_number'):
            return data
        
        # Extract from individual attributes
        if hasattr(data, 'server_id') and hasattr(data, 'serve_number'):
            data.global_server_id = data.server_id.unsqueeze(0) if data.server_id.dim() == 0 else data.server_id
            data.global_serve_number = data.serve_number.unsqueeze(0) if data.serve_number.dim() == 0 else data.serve_number
            data.global_set_number = data.set_number.unsqueeze(0) if data.set_number.dim() == 0 else data.set_number
            data.global_game_number = data.game_number.unsqueeze(0) if data.game_number.dim() == 0 else data.game_number
            data.global_point_number = data.point_number.unsqueeze(0) if data.point_number.dim() == 0 else data.point_number
            
            return data
        
        # Default values as fallback
        data.global_server_id = torch.tensor([0], device=data.x.device, dtype=torch.long)
        data.global_serve_number = torch.tensor([1.0], device=data.x.device)
        data.global_set_number = torch.tensor([1.0], device=data.x.device)
        data.global_game_number = torch.tensor([1.0], device=data.x.device)
        data.global_point_number = torch.tensor([1.0], device=data.x.device)
        
        if serve_idx is not None:
            print(f"Warning: No global attributes found in serve {serve_idx}, using default values")
        
        return data
    
    def generate_rollout(self, initial_data, rollout_steps):
        """
        Generate a rollout prediction from initial conditions.
        Adapted from shock tube evaluator for tennis motion.
        """
        predictions = []
        F_prev = None
    
        # Extract global parameters for tennis
        global_attrs = torch.stack([
            initial_data.global_server_id.flatten()[0].float(),
            initial_data.global_serve_number.flatten()[0],
            initial_data.global_set_number.flatten()[0],
            initial_data.global_game_number.flatten()[0],
            initial_data.global_point_number.flatten()[0]
        ])
        
        # Process global parameters once
        global_embed = self.model.global_processor(global_attrs)
    
        edge_index_0 = initial_data.edge_index
        
        # Extract initial dynamic features to pass through feature extractor
        # This extracts the features once at the beginning
        all_dynamic_feats_0 = initial_data.x[:, 
            self.model.num_static_feats:
            self.model.num_static_feats + self.model.num_dynamic_feats + len(self.model.skip_dynamic_indices)
        ]
        keep_indices = [i for i in range(all_dynamic_feats_0.shape[1]) if i not in self.model.skip_dynamic_indices]
        initial_dynamic_feats = all_dynamic_feats_0[:, keep_indices]
        
        # Extract learned features from initial dynamic state (computed once)
        learned_features = self.model.feature_extractor(initial_dynamic_feats, edge_index_0)
        learned_features = self.model.feature_norm(learned_features, global_attrs)
    
        # Debug initial features
        print(f"Debug - Initial dynamic features: mean={initial_dynamic_feats.mean():.6f}, std={initial_dynamic_feats.std():.6f}")
        print(f"Debug - Learned features: mean={learned_features.mean():.6f}, std={learned_features.std():.6f}")
    
        # Rollout loop
        for step in range(rollout_steps):
            if step == 0:
                # First step: use ground truth dynamic features (velocities)
                all_dynamic_feats = initial_data.x[:, 
                    self.model.num_static_feats:
                    self.model.num_static_feats + self.model.num_dynamic_feats + len(self.model.skip_dynamic_indices)
                ]
                keep_indices = [i for i in range(all_dynamic_feats.shape[1]) if i not in self.model.skip_dynamic_indices]
                dynamic_feats_t = all_dynamic_feats[:, keep_indices]
    
                assert dynamic_feats_t.shape[1] == self.model.num_dynamic_feats, (
                    f"Expected {self.model.num_dynamic_feats} dynamic features after skipping, "
                    f"but got {dynamic_feats_t.shape[1]}"
                )
                
                print(f"Debug Step {step} (ground truth): dynamic features mean={dynamic_feats_t.mean():.6f}, std={dynamic_feats_t.std():.6f}")
            else:
                # Use previous prediction
                dynamic_feats_t = F_prev
                print(f"Debug Step {step} (using prev pred): dynamic features mean={dynamic_feats_t.mean():.6f}, std={dynamic_feats_t.std():.6f}")
    
            # Normalize dynamic features
            F_prev_used = dynamic_feats_t if F_prev is None else F_prev
            F_prev_used = self.model.derivative_norm(F_prev_used, global_attrs)
            
            print(f"Debug Step {step}: after normalization mean={F_prev_used.mean():.6f}, std={F_prev_used.std():.6f}")
    
            # Broadcast global embedding to all nodes
            global_context = global_embed.unsqueeze(0).repeat(initial_data.num_nodes, 1)
    
            # Concatenate all features: learned features + normalized dynamic + global context
            Fdot_input = torch.cat([learned_features, F_prev_used, global_context], dim=-1)
            print(f"Debug Step {step}: Fdot_input mean={Fdot_input.mean():.6f}, std={Fdot_input.std():.6f}")
    
            # Forward through derivative and integral solvers
            Fdot = self.model.derivative_solver(Fdot_input, edge_index_0)
            print(f"Debug Step {step}: Fdot (derivative) mean={Fdot.mean():.6f}, std={Fdot.std():.6f}")
            print(f"Debug Step {step}: Fdot min={Fdot.min():.6f}, max={Fdot.max():.6f}")
            
            Fint = self.model.integral_solver(Fdot, edge_index_0)
            print(f"Debug Step {step}: Fint (integral) mean={Fint.mean():.6f}, std={Fint.std():.6f}")
            print(f"Debug Step {step}: Fint min={Fint.min():.6f}, max={Fint.max():.6f}")
            
            F_pred = F_prev_used + Fint
            print(f"Debug Step {step}: F_pred (final) mean={F_pred.mean():.6f}, std={F_pred.std():.6f}")
            print(f"Debug Step {step}: F_pred min={F_pred.min():.6f}, max={F_pred.max():.6f}")
            
            # Check if prediction is changing
            if step > 0:
                diff = torch.abs(F_pred - F_prev).mean()
                print(f"Debug Step {step}: Change from previous = {diff:.8f}")
                if diff < 1e-8:
                    print(f"WARNING: Prediction barely changed at step {step}!")
            
            print(f"Debug Step {step}: ---")
    
            predictions.append(F_pred)
            F_prev = F_pred
    
        return predictions

    def evaluate_rollout_predictions(self, serves, rollout_steps=10):
        """
        Generate rollout predictions from loaded serve files.
        Enhanced to track player-specific performance.
        """
        all_predictions = []
        all_targets = []
        metadata = []
        
        with torch.no_grad():
            for serve_idx, serve in enumerate(tqdm(serves, desc="Generating rollout predictions")):
                
                # Move serve to device and extract global attributes
                for data in serve:
                    data.x = data.x.to(self.device)
                    data.y = data.y.to(self.device)
                    data.edge_index = data.edge_index.to(self.device)
                    
                    # Extract global attributes using the helper function
                    data = self._extract_global_attributes(data, serve_idx)
                    
                    # Move global attributes to device
                    data.global_server_id = data.global_server_id.to(self.device)
                    data.global_serve_number = data.global_serve_number.to(self.device)
                    data.global_set_number = data.global_set_number.to(self.device)
                    data.global_game_number = data.global_game_number.to(self.device)
                    data.global_point_number = data.global_point_number.to(self.device)
                
                # Use only the first timestep as initial condition
                initial_data = serve[0]
                
                # Extract serve metadata
                server_id = int(initial_data.global_server_id[0])
                serve_number = float(initial_data.global_serve_number[0])
                set_number = float(initial_data.global_set_number[0])
                game_number = float(initial_data.global_game_number[0])
                point_number = float(initial_data.global_point_number[0])
                serve_name = f"serve_{serve_idx}"
                
                # Determine how many steps to predict
                max_available_steps = len(serve)
                actual_rollout_steps = min(rollout_steps, max_available_steps)
                
                # Generate rollout predictions
                rollout_predictions = self.generate_rollout(
                    initial_data, 
                    rollout_steps=actual_rollout_steps
                )
                
                # Collect ground truth targets for comparison
                rollout_targets = []
                for i in range(actual_rollout_steps):
                    target_y = serve[i].y.cpu()
                    rollout_targets.append(target_y)
                
                all_predictions.append([pred.cpu() for pred in rollout_predictions])
                all_targets.append(rollout_targets)
                
                # Extract metadata
                serve_metadata = {
                    'serve_idx': serve_idx,
                    'serve_name': serve_name,
                    'server_id': server_id,
                    'serve_number': serve_number,
                    'set_number': set_number,
                    'game_number': game_number,
                    'point_number': point_number,
                    'rollout_length': len(rollout_predictions),
                    'available_targets': len(serve),
                    'skip_dynamic_indices': self.model.skip_dynamic_indices,
                    'num_dynamic_feats': self.model.num_dynamic_feats
                }
                metadata.append(serve_metadata)
                
                # Track player-specific performance
                self._track_player_performance(rollout_predictions, rollout_targets, server_id)
        
        print(f"\nGenerated rollout predictions for {len(all_predictions)} serves")
        return all_predictions, all_targets, metadata
    
    def _track_player_performance(self, predictions, targets, player_id):
        """Track performance metrics for specific players."""
        # Compute overall performance for this serve
        all_preds = []
        all_targs = []
        
        for step_pred, step_targ in zip(predictions, targets):
            all_preds.append(step_pred.cpu().numpy())
            all_targs.append(step_targ.cpu().numpy())
        
        # Debug: Check individual array shapes before stacking
        print(f"Debug - Individual prediction shapes: {[p.shape for p in all_preds[:3]]}")
        print(f"Debug - Individual target shapes: {[t.shape for t in all_targs[:3]]}")
        
        # Check if all arrays have the same shape before stacking
        pred_shapes = [p.shape for p in all_preds]
        targ_shapes = [t.shape for t in all_targs]
        
        if len(set(pred_shapes)) > 1:
            print(f"Warning: Inconsistent prediction shapes: {set(pred_shapes)}")
            # Use concatenate as fallback
            all_preds = np.concatenate(all_preds, axis=0)
        else:
            # Stack to preserve (timesteps, nodes, features) structure
            all_preds = np.stack(all_preds, axis=0)
        
        if len(set(targ_shapes)) > 1:
            print(f"Warning: Inconsistent target shapes: {set(targ_shapes)}")
            all_targs = np.concatenate(all_targs, axis=0)
        else:
            all_targs = np.stack(all_targs, axis=0)
        
        print(f"Debug - Final stacked shapes: preds={all_preds.shape}, targets={all_targs.shape}")
        
        # Overall metrics
        pred_flat = all_preds.flatten()
        target_flat = all_targs.flatten()
        
        overall_metrics = {
            'mse': float(mean_squared_error(target_flat, pred_flat)),
            'mae': float(mean_absolute_error(target_flat, pred_flat)),
            'rmse': float(np.sqrt(mean_squared_error(target_flat, pred_flat))),
            'r2': float(r2_score(target_flat, pred_flat))
        }
        
        # Per-joint metrics - handle the actual data structure safely
        joint_metrics = {}
        
        try:
            if all_preds.ndim == 3:
                # 3D structure: (timesteps, joints, features)
                print(f"Debug - Processing 3D data: {all_preds.shape}")
                num_joints = min(all_preds.shape[1], len(self.joint_names))
                for j in range(num_joints):
                    joint_name = self.joint_names[j]
                    pred_joint = all_preds[:, j, :]  # All features for this joint across time
                    target_joint = all_targs[:, j, :]
                    
                    joint_metrics[joint_name] = {
                        'mse': float(mean_squared_error(target_joint.flatten(), pred_joint.flatten())),
                        'mae': float(mean_absolute_error(target_joint.flatten(), pred_joint.flatten())),
                        'rmse': float(np.sqrt(mean_squared_error(target_joint.flatten(), pred_joint.flatten()))),
                        'r2': float(r2_score(target_joint.flatten(), pred_joint.flatten()))
                    }
            
            elif all_preds.ndim == 2:
                # 2D structure: (total_samples, features) - features might be arranged by joint
                print(f"Debug - Processing 2D data: {all_preds.shape}")
                total_features = all_preds.shape[1]
                
                # Method 1: Features per joint (e.g., 3 features per joint for x,y,z)
                if total_features % len(self.joint_names) == 0:
                    features_per_joint = total_features // len(self.joint_names)
                    print(f"Debug - Assuming {features_per_joint} features per joint")
                    
                    for j, joint_name in enumerate(self.joint_names):
                        start_idx = j * features_per_joint
                        end_idx = (j + 1) * features_per_joint
                        
                        pred_joint = all_preds[:, start_idx:end_idx]
                        target_joint = all_targs[:, start_idx:end_idx]
                        
                        joint_metrics[joint_name] = {
                            'mse': float(mean_squared_error(target_joint.flatten(), pred_joint.flatten())),
                            'mae': float(mean_absolute_error(target_joint.flatten(), pred_joint.flatten())),
                            'rmse': float(np.sqrt(mean_squared_error(target_joint.flatten(), pred_joint.flatten()))),
                            'r2': float(r2_score(target_joint.flatten(), pred_joint.flatten()))
                        }
                
                # Method 2: Features are arranged differently, just compute overall metrics per feature
                else:
                    print(f"Debug - Features don't divide evenly by joints, computing feature-wise metrics")
                    for feat_idx in range(min(total_features, len(self.joint_names))):
                        joint_name = self.joint_names[feat_idx] if feat_idx < len(self.joint_names) else f"Feature_{feat_idx}"
                        
                        pred_feat = all_preds[:, feat_idx]
                        target_feat = all_targs[:, feat_idx]
                        
                        joint_metrics[joint_name] = {
                            'mse': float(mean_squared_error(target_feat, pred_feat)),
                            'mae': float(mean_absolute_error(target_feat, pred_feat)),
                            'rmse': float(np.sqrt(mean_squared_error(target_feat, pred_feat))),
                            'r2': float(r2_score(target_feat, pred_feat))
                        }
            
            else:
                print(f"Warning: Unexpected data dimensionality: {all_preds.ndim}D")
                # Just use overall metrics
        
        except Exception as e:
            print(f"Error in joint metrics calculation: {e}")
            print(f"Data shapes - preds: {all_preds.shape}, targets: {all_targs.shape}")
            # Continue with just overall metrics
        
        # Store metrics for this player
        self.player_metrics[player_id].append({
            'overall': overall_metrics,
            'joints': joint_metrics
        })
    
    def compute_rollout_metrics(self, predictions, targets):
        """Compute metrics for rollout predictions."""
        metrics = {}
        
        # Flatten all data
        all_preds = []
        all_targs = []
        
        for seq_pred, seq_targ in zip(predictions, targets):
            for step_pred, step_targ in zip(seq_pred, seq_targ):
                all_preds.append(step_pred.numpy())
                all_targs.append(step_targ.numpy())
        
        # Stack to preserve (timesteps, nodes, features) structure
        all_preds = np.stack(all_preds, axis=0)
        all_targs = np.stack(all_targs, axis=0)
        
        # Per-joint metrics
        for j, joint_name in enumerate(self.joint_names):
            pred_joint = all_preds[:, j, :]  # All coordinates for this joint
            target_joint = all_targs[:, j, :]
            
            pred_joint_phys = self.denormalize_positions(pred_joint)
            target_joint_phys = self.denormalize_positions(target_joint)
            
            metrics[joint_name] = {
                'mse': float(mean_squared_error(target_joint.flatten(), pred_joint.flatten())),
                'mae': float(mean_absolute_error(target_joint.flatten(), pred_joint.flatten())),
                'rmse': float(np.sqrt(mean_squared_error(target_joint.flatten(), pred_joint.flatten()))),
                'r2': float(r2_score(target_joint.flatten(), pred_joint.flatten())),
                'mse_physical': float(mean_squared_error(target_joint_phys.flatten(), pred_joint_phys.flatten())),
                'mae_physical': float(mean_absolute_error(target_joint_phys.flatten(), pred_joint_phys.flatten())),
                'rmse_physical': float(np.sqrt(mean_squared_error(target_joint_phys.flatten(), pred_joint_phys.flatten())))
            }
        
        # Overall metrics
        pred_flat = all_preds.flatten()
        target_flat = all_targs.flatten()
        
        metrics['overall'] = {
            'mse': float(mean_squared_error(target_flat, pred_flat)),
            'mae': float(mean_absolute_error(target_flat, pred_flat)),
            'rmse': float(np.sqrt(mean_squared_error(target_flat, pred_flat))),
            'r2': float(r2_score(target_flat, pred_flat))
        }
        
        return metrics
    
    def analyze_player_performance(self):
        """Analyze performance across different players."""
        if not self.player_metrics:
            return {}
        
        player_analysis = {}
        
        for player_id, metrics_list in self.player_metrics.items():
            if not metrics_list:
                continue
                
            # Aggregate metrics across all serves for this player
            overall_metrics = []
            joint_metrics = {joint_name: [] for joint_name in self.joint_names}
            
            for serve_metrics in metrics_list:
                overall_metrics.append(serve_metrics['overall'])
                for joint_name in self.joint_names:
                    if joint_name in serve_metrics['joints']:
                        joint_metrics[joint_name].append(serve_metrics['joints'][joint_name])
            
            # Compute statistics (mean, std, min, max)
            analysis = {
                'num_serves': len(metrics_list),
                'player_id': int(player_id),
                'overall': self._compute_metric_statistics(overall_metrics),
                'joints': {}
            }
            
            for joint_name in self.joint_names:
                if joint_metrics[joint_name]:
                    analysis['joints'][joint_name] = self._compute_metric_statistics(joint_metrics[joint_name])
            
            player_analysis[str(player_id)] = analysis
        
        return player_analysis
    
    def _compute_metric_statistics(self, metrics_list):
        """Compute statistics (mean, std, min, max) for a list of metric dictionaries."""
        if not metrics_list:
            return {}
        
        metric_names = metrics_list[0].keys()
        stats = {}
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            if values:
                stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return stats
    
    def plot_player_performance_analysis(self, figsize=(20, 12)):
        """Create comprehensive plots showing performance vs players."""
        player_analysis = self.analyze_player_performance()
        
        if not player_analysis:
            print("No player analysis data available for plotting.")
            return None
        
        # Extract data for plotting
        player_ids = []
        overall_r2_means = []
        overall_r2_stds = []
        overall_rmse_means = []
        overall_rmse_stds = []
        
        joint_r2_data = {joint_name: {'means': [], 'stds': []} for joint_name in self.joint_names}
        joint_rmse_data = {joint_name: {'means': [], 'stds': []} for joint_name in self.joint_names}
        
        # Sort by player ID
        sorted_items = sorted(player_analysis.items(), key=lambda x: int(x[0]))
        
        for player_id_str, analysis in sorted_items:
            player_ids.append(int(player_id_str))
            
            # Overall metrics
            overall_r2_means.append(analysis['overall']['r2']['mean'])
            overall_r2_stds.append(analysis['overall']['r2']['std'])
            overall_rmse_means.append(analysis['overall']['rmse']['mean'])
            overall_rmse_stds.append(analysis['overall']['rmse']['std'])
            
            # Joint-specific metrics (sample a few key joints)
            key_joints = ['R Wrist', 'L Wrist', 'R Elbow', 'L Elbow']
            for joint_name in key_joints:
                if joint_name in analysis['joints']:
                    joint_r2_data[joint_name]['means'].append(analysis['joints'][joint_name]['r2']['mean'])
                    joint_r2_data[joint_name]['stds'].append(analysis['joints'][joint_name]['r2']['std'])
                    joint_rmse_data[joint_name]['means'].append(analysis['joints'][joint_name]['rmse']['mean'])
                    joint_rmse_data[joint_name]['stds'].append(analysis['joints'][joint_name]['rmse']['std'])
                else:
                    joint_r2_data[joint_name]['means'].append(0)
                    joint_r2_data[joint_name]['stds'].append(0)
                    joint_rmse_data[joint_name]['means'].append(0)
                    joint_rmse_data[joint_name]['stds'].append(0)
        
        # Convert to numpy arrays
        player_ids = np.array(player_ids)
        overall_r2_means = np.array(overall_r2_means)
        overall_r2_stds = np.array(overall_r2_stds)
        overall_rmse_means = np.array(overall_rmse_means)
        overall_rmse_stds = np.array(overall_rmse_stds)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Performance vs Players', fontsize=16)
        
        # Overall R² vs player
        ax = axes[0, 0]
        ax.errorbar(player_ids, overall_r2_means, yerr=overall_r2_stds, 
                   fmt='o', markersize=8, capsize=5, capthick=2, linewidth=0, 
                   elinewidth=2, color='blue', alpha=0.7)
        ax.set_xlabel('Player ID')
        ax.set_ylabel('R² Score')
        ax.set_title('Overall Model Performance (R²)')
        ax.grid(True, alpha=0.3)
        
        # Overall RMSE vs player
        ax = axes[0, 1]
        ax.errorbar(player_ids, overall_rmse_means, yerr=overall_rmse_stds,
                   fmt='s', markersize=8, capsize=5, capthick=2, linewidth=0,
                   elinewidth=2, color='red', alpha=0.7)
        ax.set_xlabel('Player ID')
        ax.set_ylabel('RMSE')
        ax.set_title('Overall Model Error (RMSE)')
        ax.grid(True, alpha=0.3)
        
        # Key joint R² vs player
        ax = axes[1, 0]
        colors = plt.cm.Set1(np.linspace(0, 1, 4))
        markers = ['o', 's', '^', 'D']
        key_joints = ['R Wrist', 'L Wrist', 'R Elbow', 'L Elbow']
        
        for i, joint_name in enumerate(key_joints):
            marker = markers[i % len(markers)]
            ax.errorbar(player_ids, joint_r2_data[joint_name]['means'], 
                       yerr=joint_r2_data[joint_name]['stds'],
                       fmt=marker, markersize=6, capsize=3, linewidth=0,
                       elinewidth=1.5, label=joint_name,
                       color=colors[i], alpha=0.8)
        ax.set_xlabel('Player ID')
        ax.set_ylabel('R² Score')
        ax.set_title('Key Joint Performance (R²)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Key joint RMSE vs player
        ax = axes[1, 1]
        for i, joint_name in enumerate(key_joints):
            marker = markers[i % len(markers)]
            ax.errorbar(player_ids, joint_rmse_data[joint_name]['means'],
                       yerr=joint_rmse_data[joint_name]['stds'],
                       fmt=marker, markersize=6, capsize=3, linewidth=0,
                       elinewidth=1.5, label=joint_name,
                       color=colors[i], alpha=0.8)
        ax.set_xlabel('Player ID')
        ax.set_ylabel('RMSE')
        ax.set_title('Key Joint Error (RMSE)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_rollout_animation(self, predictions, targets, metadata, seq_idx, output_dir, performance_category=None):
        """Create animated GIFs showing rollout evolution for tennis motion."""
        try:
            from matplotlib.animation import PillowWriter
        except ImportError:
            print("Warning: PillowWriter not available. Skipping GIF creation.")
            return
        
        if seq_idx >= len(predictions):
            return
        
        seq_pred = predictions[seq_idx]
        seq_targ = targets[seq_idx]
        seq_meta = metadata[seq_idx]
        
        serve_name = seq_meta['serve_name']
        player_id = seq_meta['server_id']
        max_steps = min(len(seq_pred), len(seq_targ))
        
        if max_steps < 2:
            print(f"Skipping GIF for {serve_name}: insufficient timesteps")
            return
        
        # Create title with performance category
        title_suffix = f" ({performance_category.replace('_', ' ').title()})" if performance_category else ""
        base_title = f'Serve Motion Rollout: {serve_name} (Player {player_id}){title_suffix}'
        
        # Set up plot limits
        all_positions = []
        for t in range(max_steps):
            pred_pos = self.denormalize_positions(seq_pred[t].numpy())
            targ_pos = self.denormalize_positions(seq_targ[t].numpy())
            all_positions.extend([pred_pos, targ_pos])
        
        all_positions = np.array(all_positions)
        min_coords = all_positions.min(axis=(0, 1))
        max_coords = all_positions.max(axis=(0, 1))
        center = (max_coords + min_coords) / 2
        max_range = (max_coords - min_coords).max() * 0.6
        
        # Create figure
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(base_title, fontsize=14)
        
        # Ground truth subplot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title('Ground Truth')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        
        # Prediction subplot  
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title('Prediction')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
        
        def animate(frame):
            """Animation function for updating the plots."""
            ax1.clear()
            ax2.clear()
            
            # Get denormalized positions
            gt_pos = self.denormalize_positions(seq_targ[frame].numpy())
            pred_pos = self.denormalize_positions(seq_pred[frame].numpy())
            
            # Plot ground truth
            ax1.scatter(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 
                       c='blue', s=50, alpha=0.8)
            for start, end in self.skeleton_edges:
                ax1.plot([gt_pos[start, 0], gt_pos[end, 0]],
                        [gt_pos[start, 1], gt_pos[end, 1]],
                        [gt_pos[start, 2], gt_pos[end, 2]], 
                        color='blue', alpha=0.6, linewidth=2)
            
            # Plot prediction
            ax2.scatter(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
                       c='red', s=50, alpha=0.8)
            for start, end in self.skeleton_edges:
                ax2.plot([pred_pos[start, 0], pred_pos[end, 0]],
                        [pred_pos[start, 1], pred_pos[end, 1]],
                        [pred_pos[start, 2], pred_pos[end, 2]], 
                        color='red', alpha=0.6, linewidth=2)
            
            # Set consistent limits and view
            for ax in [ax1, ax2]:
                ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
                ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
                ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
                ax.view_init(elev=20, azim=45)
            
            # Update title with timestep
            title_with_timestep = f'{base_title} (Timestep {frame}/{max_steps-1})'
            fig.suptitle(title_with_timestep, fontsize=14)
        
        # Create animation
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, animate, frames=max_steps, interval=500, blit=False)
        
        # Save as GIF
        gif_filename = f'tennis_rollout_{serve_name}_player{player_id}'
        if performance_category:
            gif_filename = f'tennis_rollout_{performance_category}_{serve_name}_player{player_id}'
        gif_path = output_dir / f'{gif_filename}.gif'
        
        writer = PillowWriter(fps=2)
        anim.save(gif_path, writer=writer)
        plt.close(fig)
        
        print(f"Saved tennis rollout GIF: {gif_path}")
    
    def create_joint_error_analysis(self, predictions, targets, metadata, figsize=(20, 15)):
        """Create analysis of per-joint prediction errors."""
        if len(predictions) == 0:
            return None
        
        # Calculate per-joint errors across all serves
        joint_errors = {joint_name: [] for joint_name in self.joint_names}
        
        for seq_pred, seq_targ in zip(predictions, targets):
            for step_pred, step_targ in zip(seq_pred, seq_targ):
                pred_array = step_pred.numpy()  # Shape: (num_joints, num_features)
                targ_array = step_targ.numpy()  # Shape: (num_joints, num_features)
                
                # Calculate error for each joint (assuming features are x,y,z coordinates)
                for j, joint_name in enumerate(self.joint_names):
                    if j < pred_array.shape[0]:  # Check bounds
                        if pred_array.shape[1] >= 3:  # Assuming x,y,z coordinates
                            pred_joint = pred_array[j, :3]  # First 3 features as x,y,z
                            targ_joint = targ_array[j, :3]
                            error = np.linalg.norm(pred_joint - targ_joint)  # Euclidean distance
                        else:
                            # Fallback: use all features for this joint
                            pred_joint = pred_array[j, :]
                            targ_joint = targ_array[j, :]
                            error = np.linalg.norm(pred_joint - targ_joint)
                        joint_errors[joint_name].append(error)
        
        # Create visualization
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('Per-Joint Prediction Error Analysis', fontsize=16)
        axes = axes.flatten()
        
        # Box plot of joint errors
        ax = axes[0]
        joint_names_short = [name.replace(' ', '\n') for name in self.joint_names]
        error_data = [joint_errors[joint_name] for joint_name in self.joint_names if joint_errors[joint_name]]
        
        if error_data:
            bp = ax.boxplot(error_data, labels=joint_names_short[:len(error_data)], patch_artist=True)
            ax.set_title('Error Distribution by Joint')
            ax.set_ylabel('Position Error (physical units)')
            ax.tick_params(axis='x', rotation=45)
            
            # Color boxes by joint type
            colors = ['lightblue' if 'Wrist' in name or 'Elbow' in name else 
                     'lightgreen' if 'Hip' in name or 'Knee' in name or 'Ankle' in name else
                     'lightcoral' for name in self.joint_names[:len(error_data)]]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
        
        # Mean error by joint
        ax = axes[1]
        mean_errors = [np.mean(joint_errors[joint_name]) if joint_errors[joint_name] else 0 
                      for joint_name in self.joint_names]
        colors = ['lightblue' if 'Wrist' in name or 'Elbow' in name else 
                 'lightgreen' if 'Hip' in name or 'Knee' in name or 'Ankle' in name else
                 'lightcoral' for name in self.joint_names]
        bars = ax.bar(range(len(self.joint_names)), mean_errors, color=colors)
        ax.set_title('Mean Error by Joint')
        ax.set_ylabel('Mean Position Error')
        ax.set_xticks(range(len(self.joint_names)))
        ax.set_xticklabels(joint_names_short, rotation=45)
        
        # Error correlation heatmap (sample of joints)
        ax = axes[2]
        key_joints = ['R Wrist', 'L Wrist', 'R Elbow', 'L Elbow', 'R Hip', 'L Hip']
        key_joint_errors = []
        valid_key_joints = []
        
        for joint in key_joints:
            if joint in joint_errors and joint_errors[joint]:
                key_joint_errors.append(joint_errors[joint])
                valid_key_joints.append(joint)
        
        if len(key_joint_errors) > 1:
            # Pad arrays to same length
            max_len = max(len(errors) for errors in key_joint_errors)
            padded_errors = []
            for errors in key_joint_errors:
                padded = np.array(errors + [0] * (max_len - len(errors)))
                padded_errors.append(padded)
            
            key_joint_errors = np.array(padded_errors)
            correlation_matrix = np.corrcoef(key_joint_errors)
            
            im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(valid_key_joints)))
            ax.set_yticks(range(len(valid_key_joints)))
            ax.set_xticklabels([j.replace(' ', '\n') for j in valid_key_joints], rotation=45)
            ax.set_yticklabels([j.replace(' ', '\n') for j in valid_key_joints])
            ax.set_title('Error Correlation (Key Joints)')
            plt.colorbar(im, ax=ax)
        
        # Error vs timestep for key joints
        ax = axes[3]
        timestep_errors = {joint: [] for joint in key_joints}
        max_timesteps = 0
        
        for seq_pred, seq_targ in zip(predictions, targets):
            max_timesteps = max(max_timesteps, len(seq_pred))
            for t, (step_pred, step_targ) in enumerate(zip(seq_pred, seq_targ)):
                pred_array = step_pred.numpy()
                targ_array = step_targ.numpy()
                
                for joint in key_joints:
                    if joint in self.joint_names:
                        j = self.joint_names.index(joint)
                        if j < pred_array.shape[0]:
                            if pred_array.shape[1] >= 3:
                                pred_joint = pred_array[j, :3]
                                targ_joint = targ_array[j, :3]
                            else:
                                pred_joint = pred_array[j, :]
                                targ_joint = targ_array[j, :]
                            error = np.linalg.norm(pred_joint - targ_joint)
                            
                            if len(timestep_errors[joint]) <= t:
                                timestep_errors[joint].extend([[] for _ in range(t + 1 - len(timestep_errors[joint]))])
                            timestep_errors[joint][t].append(error)
        
        for joint in key_joints:
            if joint in timestep_errors and any(timestep_errors[joint]):
                mean_errors_by_time = [np.mean(errors) if errors else 0 
                                     for errors in timestep_errors[joint]]
                if any(mean_errors_by_time):
                    ax.plot(range(len(mean_errors_by_time)), mean_errors_by_time, 
                           'o-', label=joint, linewidth=2)
        
        ax.set_xlabel('Rollout Timestep')
        ax.set_ylabel('Mean Position Error')
        ax.set_title('Error Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Upper vs lower body error comparison
        ax = axes[4]
        upper_joints = ['Head', 'Neck', 'Thorax', 'L Shoulder', 'R Shoulder', 
                       'L Elbow', 'R Elbow', 'L Wrist', 'R Wrist']
        lower_joints = ['Pelvis', 'L Hip', 'R Hip', 'L Knee', 'R Knee', 
                       'L Ankle', 'R Ankle']
        
        upper_errors = []
        lower_errors = []
        
        for joint_name in self.joint_names:
            if joint_name in upper_joints and joint_errors[joint_name]:
                upper_errors.extend(joint_errors[joint_name])
            elif joint_name in lower_joints and joint_errors[joint_name]:
                lower_errors.extend(joint_errors[joint_name])
        
        if upper_errors and lower_errors:
            ax.boxplot([upper_errors, lower_errors], labels=['Upper Body', 'Lower Body'])
            ax.set_title('Upper vs Lower Body Error')
            ax.set_ylabel('Position Error')
        
        # Remove unused subplot
        axes[5].set_visible(False)
        
        plt.tight_layout()
        return fig

    def plot_rollout_evolution(self, predictions, targets, metadata, seq_idx=0, figsize=(20, 10)):
        """Plot how prediction accuracy evolves over rollout timesteps for tennis motion."""
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
        joint_errors_over_time = {joint_name: [] for joint_name in self.joint_names}
        
        for t in range(max_steps):
            pred_array = seq_pred[t].numpy()  # Shape: (num_joints, num_features)
            targ_array = seq_targ[t].numpy()  # Shape: (num_joints, num_features)
            
            # Overall error (assuming first 3 features are x,y,z coordinates)
            joint_errors = []
            for j in range(min(pred_array.shape[0], len(self.joint_names))):
                if pred_array.shape[1] >= 3:
                    pred_joint = pred_array[j, :3]  # x,y,z coordinates
                    targ_joint = targ_array[j, :3]
                else:
                    pred_joint = pred_array[j, :]
                    targ_joint = targ_array[j, :]
                
                error = np.linalg.norm(pred_joint - targ_joint)
                joint_errors.append(error)
                
                # Store per-joint errors
                if j < len(self.joint_names):
                    joint_errors_over_time[self.joint_names[j]].append(error)
            
            overall_error = np.mean(joint_errors)
            timestep_errors.append(overall_error)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Rollout Evolution: {serve_name} (Player {player_id})', fontsize=16)
        
        # Overall error over time
        ax = axes[0, 0]
        ax.plot(range(max_steps), timestep_errors, 'bo-', linewidth=2, markersize=6)
        ax.set_xlabel('Rollout Timestep')
        ax.set_ylabel('Mean Position Error')
        ax.set_title('Overall Prediction Error vs Time')
        ax.grid(True, alpha=0.3)
        
        # Key joint errors over time
        ax = axes[0, 1]
        key_joints = ['R Wrist', 'L Wrist', 'R Elbow', 'L Elbow']
        colors = plt.cm.Set1(np.linspace(0, 1, len(key_joints)))
        
        for i, joint_name in enumerate(key_joints):
            if joint_name in joint_errors_over_time and joint_errors_over_time[joint_name]:
                ax.plot(range(len(joint_errors_over_time[joint_name])), 
                       joint_errors_over_time[joint_name], 
                       'o-', label=joint_name, color=colors[i], linewidth=2)
        
        ax.set_xlabel('Rollout Timestep')
        ax.set_ylabel('Position Error')
        ax.set_title('Key Joint Errors vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final pose comparison (3D scatter if we have x,y,z coordinates)
        ax = axes[1, 0]
        ax.remove()
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        
        final_pred = seq_pred[-1].numpy()
        final_targ = seq_targ[-1].numpy()
        
        if final_pred.shape[1] >= 3:  # Check if we have x,y,z coordinates
            # Extract x,y,z coordinates (assuming first 3 features)
            pred_pos = final_pred[:, :3]
            targ_pos = final_targ[:, :3]
            
            # Apply denormalization if available
            pred_pos_phys = self.denormalize_positions(pred_pos)
            targ_pos_phys = self.denormalize_positions(targ_pos)
            
            ax.scatter(targ_pos_phys[:, 0], targ_pos_phys[:, 1], targ_pos_phys[:, 2], 
                      c='blue', s=50, alpha=0.8, label='Ground Truth')
            ax.scatter(pred_pos_phys[:, 0], pred_pos_phys[:, 1], pred_pos_phys[:, 2], 
                      c='red', s=50, alpha=0.8, label='Prediction')
            
            # Draw skeleton for both (if we have enough joints)
            if pred_pos_phys.shape[0] >= max(max(edge) for edge in self.skeleton_edges) + 1:
                for start, end in self.skeleton_edges:
                    if start < pred_pos_phys.shape[0] and end < pred_pos_phys.shape[0]:
                        ax.plot([targ_pos_phys[start, 0], targ_pos_phys[end, 0]],
                               [targ_pos_phys[start, 1], targ_pos_phys[end, 1]],
                               [targ_pos_phys[start, 2], targ_pos_phys[end, 2]], 
                               color='blue', alpha=0.4, linewidth=1)
                        ax.plot([pred_pos_phys[start, 0], pred_pos_phys[end, 0]],
                               [pred_pos_phys[start, 1], pred_pos_phys[end, 1]],
                               [pred_pos_phys[start, 2], pred_pos_phys[end, 2]], 
                               color='red', alpha=0.4, linewidth=1)
            
            ax.set_title(f'Final Pose (t={max_steps-1})')
            ax.legend()
        else:
            # Fallback: show feature comparison
            ax.remove()
            ax = fig.add_subplot(2, 2, 3)
            ax.plot(final_targ.flatten(), 'b-', label='Ground Truth', alpha=0.7)
            ax.plot(final_pred.flatten(), 'r-', label='Prediction', alpha=0.7)
            ax.set_title(f'Final Features Comparison (t={max_steps-1})')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Feature Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Error heatmap for final timestep
        ax = axes[1, 1]
        final_errors = []
        for j in range(min(final_pred.shape[0], len(self.joint_names))):
            if final_pred.shape[1] >= 3:
                pred_joint = final_pred[j, :3]
                targ_joint = final_targ[j, :3]
            else:
                pred_joint = final_pred[j, :]
                targ_joint = final_targ[j, :]
            error = np.linalg.norm(pred_joint - targ_joint)
            final_errors.append(error)
        
        bars = ax.bar(range(len(final_errors)), final_errors)
        ax.set_xlabel('Joint')
        ax.set_ylabel('Position Error')
        ax.set_title('Final Timestep Error by Joint')
        ax.set_xticks(range(len(final_errors)))
        joint_labels = [self.joint_names[j].replace(' ', '\n') if j < len(self.joint_names) else f'J{j}' 
                       for j in range(len(final_errors))]
        ax.set_xticklabels(joint_labels, rotation=45, fontsize=8)
        
        # Color bars by error magnitude
        if final_errors:
            max_error = max(final_errors)
            for bar, error in zip(bars, final_errors):
                color_intensity = error / max_error if max_error > 0 else 0
                bar.set_color(plt.cm.Reds(color_intensity))
        
        plt.tight_layout()
        return fig

    def plot_prediction_vs_target_scatter(self, predictions, targets, figsize=(15, 10)):
        """Create scatter plots comparing predictions vs targets for tennis positions."""
        all_preds = []
        all_targs = []
        
        for seq_pred, seq_targ in zip(predictions, targets):
            for step_pred, step_targ in zip(seq_pred, seq_targ):
                pred_pos = self.denormalize_positions(step_pred.numpy())
                targ_pos = self.denormalize_positions(step_targ.numpy())
                all_preds.append(pred_pos)
                all_targs.append(targ_pos)
        
        all_preds = np.array(all_preds)
        all_targs = np.array(all_targs)
        
        # Create subplots for different coordinate dimensions and key joints
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Tennis Motion: Prediction vs Target', fontsize=16)
        
        # X, Y, Z coordinates (all joints combined)
        coord_names = ['X', 'Y', 'Z']
        for i, coord_name in enumerate(coord_names):
            ax = axes[0, i]
            
            # Flatten all joints for this coordinate
            pred_coord = all_preds[:, :, i].flatten()
            targ_coord = all_targs[:, :, i].flatten()
            
            # Sample for visualization
            n_points = min(5000, len(pred_coord))
            indices = np.random.choice(len(pred_coord), n_points, replace=False)
            
            pred_sample = pred_coord[indices]
            targ_sample = targ_coord[indices]
            
            ax.scatter(targ_sample, pred_sample, alpha=0.3, s=1)
            
            min_val = min(targ_sample.min(), pred_sample.min())
            max_val = max(targ_sample.max(), pred_sample.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            r2 = r2_score(targ_sample, pred_sample)
            ax.set_title(f'{coord_name} Coordinate (R² = {r2:.3f})')
            ax.set_xlabel('Target')
            ax.set_ylabel('Prediction')
            ax.grid(True, alpha=0.3)
        
        # Key joints (all coordinates combined)
        key_joints = ['R Wrist', 'L Wrist', 'R Elbow']
        for i, joint_name in enumerate(key_joints):
            ax = axes[1, i]
            
            if joint_name in self.joint_names:
                j = self.joint_names.index(joint_name)
                
                # All coordinates for this joint
                pred_joint = all_preds[:, j, :].flatten()
                targ_joint = all_targs[:, j, :].flatten()
                
                # Sample for visualization
                n_points = min(3000, len(pred_joint))
                indices = np.random.choice(len(pred_joint), n_points, replace=False)
                
                pred_sample = pred_joint[indices]
                targ_sample = targ_joint[indices]
                
                ax.scatter(targ_sample, pred_sample, alpha=0.5, s=2)
                
                min_val = min(targ_sample.min(), pred_sample.min())
                max_val = max(targ_sample.max(), pred_sample.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                r2 = r2_score(targ_sample, pred_sample)
                ax.set_title(f'{joint_name} (R² = {r2:.3f})')
                ax.set_xlabel('Target Position')
                ax.set_ylabel('Predicted Position')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


################################################################################
# UTILITY FUNCTIONS FOR TENNIS EVALUATION
################################################################################

def select_serves_by_performance(predictions, targets, metadata, n_samples=3):
    """Select serves for visualization based on rollout prediction performance."""
    if len(predictions) == 0:
        return []
    
    # Calculate per-serve rollout performance
    serve_performances = []
    
    for serve_idx, (pred_seq, targ_seq) in enumerate(zip(predictions, targets)):
        if len(pred_seq) == 0 or len(targ_seq) == 0:
            continue
            
        # Calculate cumulative error over the rollout sequence
        total_error = 0
        total_points = 0
        
        # Weight later timesteps more heavily since rollout error accumulates
        for step_idx, (pred_step, targ_step) in enumerate(zip(pred_seq, targ_seq)):
            pred_pos = pred_step.numpy()
            targ_pos = targ_step.numpy()
            
            # Calculate mean position error for this timestep
            step_error = np.mean([np.linalg.norm(pred_pos[j] - targ_pos[j]) 
                                for j in range(pred_pos.shape[0])])
            
            # Weight by timestep (later steps weighted more)
            weight = 1.0 + 0.2 * step_idx  # Gradually increasing weight
            total_error += step_error * weight
            total_points += weight
        
        # Average weighted error for this serve
        if total_points > 0:
            avg_weighted_error = total_error / total_points
            serve_performances.append((serve_idx, avg_weighted_error))
    
    if len(serve_performances) == 0:
        return list(range(min(n_samples, len(predictions))))
    
    # Sort by performance (lower error = better performance)
    serve_performances.sort(key=lambda x: x[1])
    
    # Select best, median, worst (or available subset)
    selected_indices = []
    n_available = len(serve_performances)
    
    if n_available >= 1:
        # Best performing (lowest error)
        selected_indices.append(serve_performances[0][0])
    
    if n_available >= 2 and n_samples >= 2:
        # Median performing
        median_idx = serve_performances[n_available // 2][0]
        selected_indices.append(median_idx)
    
    if n_available >= 3 and n_samples >= 3:
        # Worst performing (highest error)
        selected_indices.append(serve_performances[-1][0])
    
    # Fill remaining slots if needed
    while len(selected_indices) < min(n_samples, n_available):
        for serve_idx, _ in serve_performances:
            if serve_idx not in selected_indices:
                selected_indices.append(serve_idx)
                break
    
    return selected_indices[:n_samples]


def get_serve_performance_category(serve_idx, predictions, targets):
    """Get performance category label for a serve."""
    if len(predictions) == 0:
        return "unknown"
    
    # Calculate performance for all serves
    all_performances = []
    for pred_seq, targ_seq in zip(predictions, targets):
        if len(pred_seq) == 0 or len(targ_seq) == 0:
            continue
        total_error = 0
        total_points = 0
        for step_idx, (pred_step, targ_step) in enumerate(zip(pred_seq, targ_seq)):
            pred_pos = pred_step.numpy()
            targ_pos = targ_step.numpy()
            step_error = np.mean([np.linalg.norm(pred_pos[j] - targ_pos[j]) 
                                for j in range(pred_pos.shape[0])])
            weight = 1.0 + 0.2 * step_idx
            total_error += step_error * weight
            total_points += weight
        if total_points > 0:
            all_performances.append(total_error / total_points)
    
    if len(all_performances) == 0:
        return "unknown"
    
    # Calculate current serve's performance
    pred_seq = predictions[serve_idx]
    targ_seq = targets[serve_idx]
    total_error = 0
    total_points = 0
    for step_idx, (pred_step, targ_step) in enumerate(zip(pred_seq, targ_seq)):
        pred_pos = pred_step.numpy()
        targ_pos = targ_step.numpy()
        step_error = np.mean([np.linalg.norm(pred_pos[j] - targ_pos[j]) 
                            for j in range(pred_pos.shape[0])])
        weight = 1.0 + 0.2 * step_idx
        total_error += step_error * weight
        total_points += weight
    
    if total_points == 0:
        return "unknown"
    
    current_performance = total_error / total_points
    
    # Determine category based on percentiles
    sorted_perfs = sorted(all_performances)
    n_serves = len(sorted_perfs)
    
    # Find percentile rank
    rank = sorted_perfs.index(min(sorted_perfs, key=lambda x: abs(x - current_performance)))
    percentile = rank / n_serves
    
    if percentile <= 0.33:
        return "best_performance"
    elif percentile <= 0.67:
        return "median_performance"
    else:
        return "worst_performance"


def load_test_serves(test_dir=None, test_files=None, file_pattern="*.pt", max_files=None):
    """Load test serve files for rollout evaluation."""
    serves = []
    
    if test_files is not None:
        # Load specific files passed via command line
        test_file_paths = [Path(f) for f in test_files]
        
        if max_files:
            test_file_paths = test_file_paths[:max_files]
        
        print(f"Loading {len(test_file_paths)} specific test files for rollout evaluation")
        
        for file_path in test_file_paths:
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue
                
            try:
                serve_data = torch.load(file_path, weights_only=False)
                serves.append(serve_data)
                print(f"  Loaded {file_path.name}: {len(serve_data)} timesteps")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    elif test_dir is not None:
        # Load all files from directory
        test_dir = Path(test_dir)
        test_file_paths = list(test_dir.glob(file_pattern))
        
        if max_files:
            test_file_paths = test_file_paths[:max_files]
        
        print(f"Loading {len(test_file_paths)} test files from directory for rollout evaluation")
        
        for file_path in test_file_paths:
            try:
                serve_data = torch.load(file_path, weights_only=False)
                serves.append(serve_data)
                print(f"  Loaded {file_path.name}: {len(serve_data)} timesteps")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    else:
        raise ValueError("Either test_dir or test_files must be provided")
    
    if len(serves) == 0:
        raise ValueError("No serve files were successfully loaded")
    
    return serves


################################################################################
# MAIN EVALUATION FUNCTION FOR TENNIS
################################################################################

def evaluate_tennis_rollout(model_path, test_dir, test_files, output_dir, args):
    """Evaluate tennis GPARC model using rollout prediction mode."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # The global embedding dimension from GlobalParameterProcessor is 64
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
    
    # Dynamically calculate the input channels for the DerivativeGNN
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
    
    # Load test serves
    serves = load_test_serves(
        test_dir=test_dir,
        test_files=test_files,
        file_pattern="*.pt", 
        max_files=args.max_sequences
    )
    
    # Initialize evaluator
    evaluator = TennisRolloutEvaluator(model, device)
    
    # Load denormalization params
    if test_dir:
        norm_stats_file = Path(test_dir).parent / 'normalization_stats.pkl'
    elif test_files and len(test_files) > 0:
        first_file_dir = Path(test_files[0]).parent
        norm_stats_file = first_file_dir / 'normalization_stats.pkl'
        if not norm_stats_file.exists():
            norm_stats_file = first_file_dir.parent / 'normalization_stats.pkl'
    else:
        norm_stats_file = None
    
    if norm_stats_file and norm_stats_file.exists():
        evaluator.load_denormalization_params(norm_stats_file)
    
    # Generate rollout predictions
    print(f"\nGenerating rollout predictions ({args.rollout_steps} steps)...")
    predictions, targets, metadata = evaluator.evaluate_rollout_predictions(
        serves, rollout_steps=args.rollout_steps
    )
    
    # Compute metrics
    print("Computing metrics...")
    metrics = evaluator.compute_rollout_metrics(predictions, targets)
    
    # Analyze player performance
    print("Analyzing performance across different players...")
    player_analysis = evaluator.analyze_player_performance()
    
    # Select diverse serves for visualization based on performance
    print("Selecting representative serves for visualization...")
    selected_indices = select_serves_by_performance(predictions, targets, metadata, n_samples=3)
    
    print(f"Selected serves for visualization:")
    for idx in selected_indices:
        serve_name = metadata[idx]['serve_name']
        player_id = metadata[idx]['server_id']
        performance_category = get_serve_performance_category(idx, predictions, targets)
        print(f"  - Serve {idx} ({serve_name}, Player {player_id}): {performance_category}")
    
    # Generate visualizations
    print("Creating visualizations...")
    
    # Player performance analysis
    print("Creating player performance analysis...")
    fig_player = evaluator.plot_player_performance_analysis()
    if fig_player:
        fig_player.savefig(output_path / 'player_performance_analysis.png', 
                          dpi=300, bbox_inches='tight')
        plt.close(fig_player)
    
    # Joint error analysis
    print("Creating joint error analysis...")
    fig_joint = evaluator.create_joint_error_analysis(predictions, targets, metadata)
    if fig_joint:
        fig_joint.savefig(output_path / 'joint_error_analysis.png', 
                         dpi=300, bbox_inches='tight')
        plt.close(fig_joint)
    
    # Static plots for selected serves
    for i, serve_idx in enumerate(selected_indices):
        fig = evaluator.plot_rollout_evolution(predictions, targets, metadata, serve_idx)
        if fig:
            serve_name = metadata[serve_idx]['serve_name']
            player_id = metadata[serve_idx]['server_id']
            performance_cat = get_serve_performance_category(serve_idx, predictions, targets).replace(' ', '_')
            fig.savefig(output_path / f'rollout_evolution_{performance_cat}_{serve_name}_player{player_id}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    fig = evaluator.plot_prediction_vs_target_scatter(predictions, targets)
    fig.savefig(output_path / 'rollout_scatter.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create animated GIFs for selected serves
    print("Creating animated GIFs...")
    for serve_idx in selected_indices:
        print(f"Creating GIF for serve {serve_idx}...")
        performance_cat = get_serve_performance_category(serve_idx, predictions, targets)
        evaluator.create_rollout_animation(predictions, targets, metadata, serve_idx, output_path, performance_cat)
    
    # Save results with player analysis
    results = {
        'metrics': metrics,
        'player_analysis': player_analysis,
        'metadata': metadata,
        'model_info': {
            'model_path': str(model_path),
            'test_serves': len(predictions),
            'rollout_steps': args.rollout_steps,
            'device': str(device),
            'skip_dynamic_indices': model.skip_dynamic_indices,
            'num_dynamic_feats': model.num_dynamic_feats,
            'test_source': 'specific_files' if test_files else 'directory',
            'test_files': test_files if test_files else None,
            'test_dir': str(test_dir) if test_dir else None
        }
    }
    
    with open(output_path / 'tennis_rollout_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\nTennis rollout evaluation complete! Results saved to: {output_path}")
    print("\nModel Performance Summary:")
    print("="*50)
    
    # Overall performance
    overall = metrics['overall']
    print(f"Overall: R²={overall['r2']:.4f}, RMSE={overall['rmse']:.6f}")
    
    # Key joint performance
    key_joints = ['R Wrist', 'L Wrist', 'R Elbow', 'L Elbow']
    for joint_name in key_joints:
        if joint_name in metrics:
            r2 = metrics[joint_name]['r2']
            rmse = metrics[joint_name]['rmse']
            print(f"{joint_name}: R²={r2:.4f}, RMSE={rmse:.6f}")
    
    # Player analysis summary
    if player_analysis:
        print(f"\nPlayer Analysis Summary:")
        print("="*50)
        player_ids = sorted([int(pid) for pid in player_analysis.keys()])
        
        print(f"Analyzed {len(player_ids)} different players:")
        for player_id in player_ids:
            pid_str = str(player_id)
            if pid_str in player_analysis:
                analysis = player_analysis[pid_str]
                overall_r2 = analysis['overall']['r2']['mean']
                overall_rmse = analysis['overall']['rmse']['mean']
                n_serves = analysis['num_serves']
                print(f"  Player {player_id}: R²={overall_r2:.4f}, RMSE={overall_rmse:.6f} ({n_serves} serves)")
    
    return metrics, evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tennis GPARC model with rollout prediction")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--test_dir", type=str,
                           help="Directory containing test dataset files")
    input_group.add_argument("--test_files", type=str, nargs='+',
                           help="Specific test serve files to evaluate")
    
    parser.add_argument("--output_dir", type=str, default="./tennis_rollout_evaluation",
                       help="Output directory for evaluation results")
    
    # Model architecture - Feature extractor (MUST MATCH TRAINING SCRIPT)
    parser.add_argument("--num_static_feats", type=int, default=0,  # 17 joints * 3 coordinates
                       help="Number of static features (joint positions)")
    parser.add_argument("--num_dynamic_feats", type=int, default=6,  # 17 joints * 3 velocities
                       help="Number of dynamic features (joint velocities)")
    parser.add_argument("--skip_dynamic_indices", type=int, nargs='+', default=[],
                       help="Indices of dynamic features to skip")
    parser.add_argument("--hidden_channels", type=int, default=64,
                       help="Hidden channels in feature extractor")
    parser.add_argument("--feature_out_channels", type=int, default=128,
                       help="Output channels from feature extractor")
    parser.add_argument("--depth", type=int, default=2,
                       help="Depth of GraphUNet")
    parser.add_argument("--pool_ratios", type=float, default=0.1,
                       help="Pool ratios for GraphUNet")
    parser.add_argument("--heads", type=int, default=4,
                       help="Number of attention heads in feature extractor")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate in feature extractor")
    
    # Derivative solver arguments (MUST MATCH TRAINING SCRIPT)
    parser.add_argument("--deriv_hidden_channels", type=int, default=128,
                       help="Hidden channels in derivative solver")
    parser.add_argument("--deriv_num_layers", type=int, default=4,
                       help="Number of layers in derivative solver")
    parser.add_argument("--deriv_heads", type=int, default=8,
                       help="Number of attention heads in derivative solver")
    parser.add_argument("--deriv_dropout", type=float, default=0.3,
                       help="Dropout rate in derivative solver")
    parser.add_argument("--deriv_use_residual", action="store_true", default=True,
                       help="Use residual connections in derivative solver")
    
    # Integral solver arguments (MUST MATCH TRAINING SCRIPT)
    parser.add_argument("--integral_hidden_channels", type=int, default=128,
                       help="Hidden channels in integral solver")
    parser.add_argument("--integral_num_layers", type=int, default=4,
                       help="Number of layers in integral solver")
    parser.add_argument("--integral_heads", type=int, default=8,
                       help="Number of attention heads in integral solver")
    parser.add_argument("--integral_dropout", type=float, default=0.3,
                       help="Dropout rate in integral solver")
    parser.add_argument("--integral_use_residual", action="store_true", default=True,
                       help="Use residual connections in integral solver")
    
    # Evaluation settings
    parser.add_argument("--max_sequences", type=int, default=30,
                       help="Maximum number of serves to evaluate")
    parser.add_argument("--rollout_steps", type=int, default=10,
                       help="Number of timesteps to predict in rollout")
    
    args = parser.parse_args()
    
    evaluate_tennis_rollout(args.model_path, args.test_dir, args.test_files, args.output_dir, args)


if __name__ == "__main__":
    main()