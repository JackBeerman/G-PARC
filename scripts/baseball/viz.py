#!/usr/bin/env python3
"""
Baseball Pitch Prediction Model Rollout Evaluation Script
=========================================================

This script evaluates a trained baseball GPARC model using rollout prediction mode,
with support for multi-step context evaluation.

Usage:
    # Single-step rollout (give t=0, predict all):
    python evaluate_baseball_model.py --model_path best_model.pth --test_dir /path/to/test --num_context_steps 0
    
    # Multi-step context (give t=0,1,2, predict rest):
    python evaluate_baseball_model.py --model_path best_model.pth --test_dir /path/to/test --num_context_steps 3
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
from models.baseball import BaseballGPARC


class BaseballRolloutEvaluator:
    """Evaluator for baseball pitching models with context support."""
    
    def __init__(self, model, device='cpu', denormalization_params=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.denorm_params = denormalization_params
        
        # Baseball joint names (18 joints)
        self.joint_names = [
            "centerofmass", "elbow_jc", "glove_elbow_jc", "glove_hand_jc",
            "glove_shoulder_jc", "glove_wrist_jc", "hand_jc", "lead_ankle_jc",
            "lead_hip", "lead_knee_jc", "rear_ankle_jc", "rear_hip",
            "rear_knee_jc", "shoulder_jc", "thorax_ap", "thorax_dist",
            "thorax_prox", "wrist_jc"
        ]
        
        # Baseball skeleton edges (kinematic chain)
        self.skeleton_edges = [
            # Spine/Torso chain
            (14, 15), (15, 16), (16, 0),  # thorax_ap -> thorax_dist -> thorax_prox -> centerofmass
            
            # Throwing arm chain
            (16, 13), (13, 1), (1, 17), (17, 6),  # thorax_prox -> shoulder_jc -> elbow_jc -> wrist_jc -> hand_jc
            
            # Glove arm chain
            (16, 4), (4, 2), (2, 5), (5, 3),  # thorax_prox -> glove_shoulder_jc -> glove_elbow_jc -> glove_wrist_jc -> glove_hand_jc
            
            # Lead leg chain
            (0, 8), (8, 9), (9, 7),  # centerofmass -> lead_hip -> lead_knee_jc -> lead_ankle_jc
            
            # Rear leg chain
            (0, 11), (11, 12), (12, 10)  # centerofmass -> rear_hip -> rear_knee_jc -> rear_ankle_jc
        ]
        
        self.pitcher_metrics = defaultdict(list)
    
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
    
    def _extract_global_attributes(self, data, pitch_idx=None):
        """Extract global attributes from data object."""
        if hasattr(data, 'pitch_speed'):
            data.global_pitch_speed = data.pitch_speed.unsqueeze(0) if data.pitch_speed.dim() == 0 else data.pitch_speed
        elif not hasattr(data, 'global_pitch_speed'):
            data.global_pitch_speed = torch.tensor([85.0], device=data.x.device, dtype=torch.float32)
        
        # Create global_feats for model compatibility
        if not hasattr(data, 'global_feats'):
            data.global_feats = data.global_pitch_speed.clone()
        
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
        
        # Extract global attributes - baseball only has pitch_speed
        global_attrs = first_data.global_pitch_speed.flatten()
        if global_attrs.dim() == 0:
            global_attrs = global_attrs.unsqueeze(0)
        
        global_embed = self.model.global_processor(global_attrs)
        edge_index_0 = first_data.edge_index
        
        # Extract initial features (9 dynamic features: pos + vel + angles)
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
    
    def evaluate_rollout_predictions_with_context(self, pitches, rollout_steps=10, num_context_steps=0):
        """Generate rollout predictions with context support."""
        all_predictions = []
        all_targets = []
        metadata = []
        
        with torch.no_grad():
            for pitch_idx, pitch in enumerate(tqdm(pitches, desc="Generating predictions")):
                
                for data in pitch:
                    data.x = data.x.to(self.device)
                    data.y = data.y.to(self.device)
                    data.edge_index = data.edge_index.to(self.device)
                    data = self._extract_global_attributes(data, pitch_idx)
                    data.global_pitch_speed = data.global_pitch_speed.to(self.device)
                    if hasattr(data, 'global_feats'):
                        data.global_feats = data.global_feats.to(self.device)
                
                first_data = pitch[0]
                pitch_speed = float(first_data.global_pitch_speed[0])
                pitch_name = f"pitch_{pitch_idx}"
                
                max_available_steps = len(pitch)
                actual_rollout_steps = min(rollout_steps, max_available_steps)
                
                if num_context_steps >= actual_rollout_steps:
                    effective_context_steps = max(1, actual_rollout_steps - 1)
                else:
                    effective_context_steps = num_context_steps
                
                rollout_predictions = self.generate_rollout_with_context(
                    pitch, 
                    rollout_steps=actual_rollout_steps,
                    num_context_steps=effective_context_steps
                )
                
                rollout_targets = [pitch[i].y.cpu() for i in range(actual_rollout_steps)]
                
                all_predictions.append([pred.cpu() for pred in rollout_predictions])
                all_targets.append(rollout_targets)
                
                pitch_metadata = {
                    'pitch_idx': pitch_idx,
                    'pitch_name': pitch_name,
                    'pitch_speed': pitch_speed,
                    'rollout_length': len(rollout_predictions),
                    'num_context_steps': effective_context_steps
                }
                metadata.append(pitch_metadata)
                
                self._track_pitcher_performance(rollout_predictions, rollout_targets, pitch_speed)
        
        print(f"\nGenerated predictions for {len(all_predictions)} pitches")
        return all_predictions, all_targets, metadata
    
    def _track_pitcher_performance(self, predictions, targets, pitch_speed):
        """Track performance metrics per pitch speed range."""
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
        
        # Bin by pitch speed
        speed_bin = int(pitch_speed / 10) * 10  # Round to nearest 10 mph
        self.pitcher_metrics[speed_bin].append({'overall': overall_metrics})
    
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
    
    def create_3d_pitch_animation(self, predictions, targets, metadata, seq_idx=0, 
                                   output_path='pitch_rollout.gif', fps=20, view_angle="side"):
        """
        Create a 3D animated GIF showing side-by-side comparison of predicted vs ground truth pitch motion.
        Uses the same color-coding and styling as the original visualization.
        
        Args:
            predictions: List of prediction sequences
            targets: List of target sequences
            metadata: List of metadata dicts
            seq_idx: Index of sequence to animate
            output_path: Path to save GIF
            fps: Frames per second
            view_angle: "side", "front", or "3d"
        """
        if seq_idx >= len(predictions):
            print(f"Sequence {seq_idx} not found")
            return None
        
        seq_pred = predictions[seq_idx]
        seq_targ = targets[seq_idx]
        seq_meta = metadata[seq_idx]
        
        pitch_name = seq_meta['pitch_name']
        pitch_speed = seq_meta['pitch_speed']
        max_steps = min(len(seq_pred), len(seq_targ))
        
        print(f"🎬 Generating 3D animation for {pitch_name} ({pitch_speed:.1f} mph)...")
        
        # Set view angle
        if view_angle == "side":
            elev, azim = 0, 90
        elif view_angle == "front":
            elev, azim = 0, 0
        else:  # 3d
            elev, azim = 20, 45
        
        # Calculate axis limits SEPARATELY for ground truth and predictions
        all_targ_coords = []
        all_pred_coords = []
        
        for t in range(max_steps):
            targ_positions = seq_targ[t].numpy()[:, :3]
            pred_positions = seq_pred[t].numpy()[:, :3]
            
            # Check if we need denormalization
            if np.abs(targ_positions).max() < 10:
                targ_positions = self.denormalize_positions(targ_positions)
                pred_positions = self.denormalize_positions(pred_positions)
            
            all_targ_coords.extend(targ_positions.tolist())
            all_pred_coords.extend(pred_positions.tolist())
        
        all_targ_coords = np.array(all_targ_coords)
        all_pred_coords = np.array(all_pred_coords)
        
        # Calculate separate ranges for ground truth
        targ_max_range = np.array([
            all_targ_coords[:, 0].max() - all_targ_coords[:, 0].min(),
            all_targ_coords[:, 1].max() - all_targ_coords[:, 1].min(),
            all_targ_coords[:, 2].max() - all_targ_coords[:, 2].min()
        ]).max() / 2.0
        targ_mid = all_targ_coords.mean(axis=0)
        
        # Calculate separate ranges for predictions
        pred_max_range = np.array([
            all_pred_coords[:, 0].max() - all_pred_coords[:, 0].min(),
            all_pred_coords[:, 1].max() - all_pred_coords[:, 1].min(),
            all_pred_coords[:, 2].max() - all_pred_coords[:, 2].min()
        ]).max() / 2.0
        pred_mid = all_pred_coords.mean(axis=0)
        
        print(f"  Ground truth range: {targ_max_range:.3f}m, center: [{targ_mid[0]:.2f}, {targ_mid[1]:.2f}, {targ_mid[2]:.2f}]")
        print(f"  Prediction range: {pred_max_range:.3f}m, center: [{pred_mid[0]:.2f}, {pred_mid[1]:.2f}, {pred_mid[2]:.2f}]")
        
        # Create animation
        with imageio.get_writer(output_path, mode='I', fps=fps, loop=0) as writer:
            for t in tqdm(range(max_steps), desc=f"Rendering {pitch_name}"):
                fig = plt.figure(figsize=(20, 10))
                
                # Ground Truth subplot
                ax1 = fig.add_subplot(121, projection='3d')
                # Prediction subplot
                ax2 = fig.add_subplot(122, projection='3d')
                
                # Extract positions (first 3 features are positions)
                targ_positions = seq_targ[t].numpy()[:, :3]
                pred_positions = seq_pred[t].numpy()[:, :3]
                
                # Check if we need denormalization
                if np.abs(targ_positions).max() < 10:
                    targ_positions = self.denormalize_positions(targ_positions)
                    pred_positions = self.denormalize_positions(pred_positions)
                
                # Create pose dictionaries for easier access
                targ_pose = {name: targ_positions[i] for i, name in enumerate(self.joint_names[:targ_positions.shape[0]])}
                pred_pose = {name: pred_positions[i] for i, name in enumerate(self.joint_names[:pred_positions.shape[0]])}
                
                # Helper function to get joint color and size (matching original code)
                def get_joint_style(joint_name):
                    if any(x in joint_name for x in ['shoulder', 'elbow', 'wrist', 'hand']) and 'glove' not in joint_name:
                        return 'red', 120  # Throwing arm
                    elif 'glove' in joint_name:
                        return 'blue', 120  # Glove arm
                    elif 'lead' in joint_name:
                        return 'green', 100  # Lead leg
                    elif 'rear' in joint_name:
                        return 'orange', 100  # Rear leg
                    elif 'thorax' in joint_name:
                        return 'purple', 110  # Torso
                    else:
                        return 'gray', 80  # Other joints
                
                # Plot Ground Truth
                for joint_name, pos in targ_pose.items():
                    color, size = get_joint_style(joint_name)
                    ax1.scatter(*pos, c=color, s=size, marker='o', 
                               edgecolors='black', linewidths=1.5, alpha=0.9)
                
                # Plot skeleton for ground truth
                for joint1_name, joint2_name in [(self.joint_names[j1], self.joint_names[j2]) 
                                                  for j1, j2 in self.skeleton_edges]:
                    if joint1_name in targ_pose and joint2_name in targ_pose:
                        pos1 = targ_pose[joint1_name]
                        pos2 = targ_pose[joint2_name]
                        
                        # Different line styles for different body parts
                        if 'hip' in joint1_name and 'hip' in joint2_name:
                            ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                                   'c-', linewidth=4, alpha=0.8)
                        elif 'thorax' in joint1_name and 'thorax' in joint2_name:
                            ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                                   'k-', linewidth=3.5, alpha=0.7)
                        else:
                            ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                                   'k-', linewidth=2.5, alpha=0.7)
                
                # Plot Predictions
                for joint_name, pos in pred_pose.items():
                    color, size = get_joint_style(joint_name)
                    ax2.scatter(*pos, c=color, s=size, marker='o', 
                               edgecolors='black', linewidths=1.5, alpha=0.9)
                
                # Plot skeleton for predictions
                for joint1_name, joint2_name in [(self.joint_names[j1], self.joint_names[j2]) 
                                                  for j1, j2 in self.skeleton_edges]:
                    if joint1_name in pred_pose and joint2_name in pred_pose:
                        pos1 = pred_pose[joint1_name]
                        pos2 = pred_pose[joint2_name]
                        
                        if 'hip' in joint1_name and 'hip' in joint2_name:
                            ax2.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                                   'c-', linewidth=4, alpha=0.8)
                        elif 'thorax' in joint1_name and 'thorax' in joint2_name:
                            ax2.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                                   'k-', linewidth=3.5, alpha=0.7)
                        else:
                            ax2.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                                   'k-', linewidth=2.5, alpha=0.7)
                
                # Calculate error
                position_error = np.linalg.norm(pred_positions - targ_positions, axis=1).mean()
                
                # Set up plots with SEPARATE limits for each
                # Ground truth axes
                ax1.set_xlim(targ_mid[0] - targ_max_range, targ_mid[0] + targ_max_range)
                ax1.set_ylim(targ_mid[1] - targ_max_range, targ_mid[1] + targ_max_range)
                ax1.set_zlim(targ_mid[2] - targ_max_range, targ_mid[2] + targ_max_range)
                ax1.set_xlabel("X (m)", fontweight='bold')
                ax1.set_ylabel("Y (m)", fontweight='bold')
                ax1.set_zlabel("Z (m)", fontweight='bold')
                ax1.set_title(f"GROUND TRUTH\nFrame {t+1}/{max_steps}", 
                           fontweight='bold', fontsize=12)
                ax1.view_init(elev=elev, azim=azim)
                ax1.grid(True, alpha=0.3)
                
                # Prediction axes
                ax2.set_xlim(pred_mid[0] - pred_max_range, pred_mid[0] + pred_max_range)
                ax2.set_ylim(pred_mid[1] - pred_max_range, pred_mid[1] + pred_max_range)
                ax2.set_zlim(pred_mid[2] - pred_max_range, pred_mid[2] + pred_max_range)
                ax2.set_xlabel("X (m)", fontweight='bold')
                ax2.set_ylabel("Y (m)", fontweight='bold')
                ax2.set_zlabel("Z (m)", fontweight='bold')
                ax2.set_title(f"PREDICTION\nFrame {t+1}/{max_steps}", 
                           fontweight='bold', fontsize=12)
                ax2.view_init(elev=elev, azim=azim)
                ax2.grid(True, alpha=0.3)
                
                # Add overall title with metadata
                fig.suptitle(f'Pitching Motion - {view_angle.upper()} VIEW\n{pitch_name} | {pitch_speed:.1f} mph | Position Error: {position_error:.4f}m',
                           fontweight='bold', fontsize=16)
                
                # Save frame using BytesIO (same as original)
                import io
                with io.BytesIO() as buf:
                    plt.savefig(buf, format='png', dpi=90)
                    buf.seek(0)
                    writer.append_data(imageio.v2.imread(buf))
                plt.close(fig)
        
        print(f"✅ Animation saved to '{output_path}'")
        return output_path
    
    def plot_rollout_evolution(self, predictions, targets, metadata, seq_idx=0, figsize=(20, 10)):
        """Plot how prediction accuracy evolves over rollout timesteps."""
        if seq_idx >= len(predictions):
            return None
        
        seq_pred = predictions[seq_idx]
        seq_targ = targets[seq_idx]
        seq_meta = metadata[seq_idx]
        
        pitch_name = seq_meta['pitch_name']
        pitch_speed = seq_meta['pitch_speed']
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
        fig.suptitle(f'Rollout Evolution: {pitch_name} ({pitch_speed:.1f} mph)', fontsize=16)
        
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
        fig.suptitle('Baseball Pitch Rollout Evaluation Summary', fontsize=16)
        
        # Plot 1: Error distribution across all pitches
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
        
        # Plot 2: Error vs timestep (averaged across all pitches)
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
        
        # Plot 3: Distribution of pitch speeds
        ax = axes[1, 0]
        pitch_speeds = [meta['pitch_speed'] for meta in metadata]
        
        ax.hist(pitch_speeds, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Pitch Speed (mph)')
        ax.set_ylabel('Number of Pitches')
        ax.set_title('Pitch Speed Distribution')
        ax.axvline(np.mean(pitch_speeds), color='r', linestyle='--',
                  label=f'Mean: {np.mean(pitch_speeds):.1f} mph')
        ax.legend()
        
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
        Total Pitches:    {len(predictions)}
        Total Steps:      {sum(len(p) for p in predictions)}
        Context Steps:    {metadata[0].get('num_context_steps', 0)}
        Avg Pitch Speed:  {np.mean(pitch_speeds):.1f} mph
        """
        
        ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig


def load_test_pitches(test_dir=None, test_files=None, max_files=None):
    """Load test pitch files."""
    pitches = []
    
    if test_files is not None:
        test_file_paths = [Path(f) for f in test_files]
        if max_files:
            test_file_paths = test_file_paths[:max_files]
        
        for file_path in test_file_paths:
            if not file_path.exists():
                continue
            pitch_data = torch.load(file_path, weights_only=False)
            pitches.append(pitch_data)
    
    elif test_dir is not None:
        test_dir = Path(test_dir)
        test_file_paths = list(test_dir.glob("*.pt"))
        if max_files:
            test_file_paths = test_file_paths[:max_files]
        
        for file_path in test_file_paths:
            pitch_data = torch.load(file_path, weights_only=False)
            pitches.append(pitch_data)
    
    return pitches


def evaluate_baseball_rollout(model_path, test_dir, test_files, output_dir, args):
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
    
    model = BaseballGPARC(
        feature_extractor=feature_extractor,
        derivative_solver=derivative_solver,
        integral_solver=integral_solver,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        skip_dynamic_indices=getattr(args, 'skip_dynamic_indices', []),
        feature_out_channels=args.feature_out_channels,
        num_global_feats=1  # Just pitch_speed
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Model loaded successfully!")
    
    # Load pitches
    pitches = load_test_pitches(
        test_dir=test_dir,
        test_files=test_files,
        max_files=args.max_sequences
    )
    print(f"Loaded {len(pitches)} test pitches")
    
    # Initialize evaluator
    evaluator = BaseballRolloutEvaluator(model, device)
    
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
        pitches, 
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
    
    # Individual pitch plots (first 3)
    for i in range(min(3, len(predictions))):
        fig = evaluator.plot_rollout_evolution(predictions, targets, metadata, seq_idx=i)
        if fig:
            fig.savefig(output_path / f'rollout_evolution_pitch_{i}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
    print(f"  ✓ Saved {min(3, len(predictions))} individual rollout plots")
    
    # Create 3D animated GIFs for first few pitches
    print("\nCreating 3D pitch animations...")
    num_animations = min(args.num_animations, len(predictions))
    for i in range(num_animations):
        gif_path = output_path / f'pitch_rollout_{i}_{args.view_angle}.gif'
        evaluator.create_3d_pitch_animation(
            predictions, targets, metadata, 
            seq_idx=i, 
            output_path=str(gif_path),
            fps=args.animation_fps,
            view_angle=args.view_angle
        )
    print(f"  ✓ Created {num_animations} pitch animations ({args.view_angle} view)")
    
    # Save results
    results = {
        'metrics': metrics,
        'metadata': metadata,
        'model_info': {
            'model_path': str(model_path),
            'test_pitches': len(predictions),
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
    parser = argparse.ArgumentParser(description="Evaluate Baseball GPARC model")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--test_dir", type=str,
                            help="Directory containing test pitch files")
    input_group.add_argument("--test_files", type=str, nargs='+',
                            help="Specific test pitch files")
    
    parser.add_argument("--output_dir", type=str, default="./evaluation_baseball",
                       help="Output directory for evaluation results")
    parser.add_argument("--num_static_feats", type=int, default=0,
                       help="Number of static features (0 for baseball)")
    parser.add_argument("--num_dynamic_feats", type=int, default=9,
                       help="Number of dynamic features (9: pos+vel+angles)")
    parser.add_argument("--skip_dynamic_indices", type=int, nargs='+', default=[],
                       help="Indices to skip in dynamic features")
    
    # Feature extractor
    parser.add_argument("--hidden_channels", type=int, default=128,
                       help="Hidden channels in feature extractor")
    parser.add_argument("--feature_out_channels", type=int, default=256,
                       help="Output channels from feature extractor")
    parser.add_argument("--depth", type=int, default=3,
                       help="Depth of GraphUNet")
    parser.add_argument("--pool_ratios", type=float, default=0.1,
                       help="Pool ratios for GraphUNet")
    parser.add_argument("--heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate")
    
    # Derivative solver
    parser.add_argument("--deriv_hidden_channels", type=int, default=256,
                       help="Hidden channels in derivative solver")
    parser.add_argument("--deriv_num_layers", type=int, default=4,
                       help="Number of layers in derivative solver")
    parser.add_argument("--deriv_heads", type=int, default=8,
                       help="Number of attention heads in derivative solver")
    parser.add_argument("--deriv_dropout", type=float, default=0.3,
                       help="Dropout rate in derivative solver")
    parser.add_argument("--deriv_use_residual", action="store_true", default=True,
                       help="Use residual connections in derivative solver")
    
    # Integral solver
    parser.add_argument("--integral_hidden_channels", type=int, default=256,
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
                       help="Maximum number of test sequences to evaluate")
    parser.add_argument("--rollout_steps", type=int, default=10,
                       help="Number of rollout steps to predict")
    parser.add_argument("--num_context_steps", type=int, default=0,
                       help="Number of context steps (0=single-step, 3+=multi-step)")
    
    # Animation settings
    parser.add_argument("--num_animations", type=int, default=5,
                       help="Number of pitch animations to create")
    parser.add_argument("--animation_fps", type=int, default=20,
                       help="Frames per second for animations")
    parser.add_argument("--view_angle", type=str, default="3d", 
                       choices=["side", "front", "3d"],
                       help="Camera view angle: 'side' (catcher view), 'front' (pitcher view), '3d' (angled)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_context_steps < 0:
        parser.error("num_context_steps must be >= 0")
    
    if args.rollout_steps < 1:
        parser.error("rollout_steps must be >= 1")
    
    # Log configuration
    print("=" * 80)
    print("BASEBALL PITCH ROLLOUT EVALUATION")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Model: {args.model_path}")
    if args.test_dir:
        print(f"  Test directory: {args.test_dir}")
    else:
        print(f"  Test files: {len(args.test_files)} files")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Rollout steps: {args.rollout_steps}")
    print(f"  Context steps: {args.num_context_steps}")
    print(f"  Dynamic features: {args.num_dynamic_feats} (pos+vel+angles)")
    print(f"  Max sequences: {args.max_sequences}")
    print(f"  Animations: {args.num_animations} at {args.animation_fps} fps ({args.view_angle} view)")
    print()
    
    # Run evaluation
    evaluate_baseball_rollout(args.model_path, args.test_dir, args.test_files, 
                             args.output_dir, args)


if __name__ == "__main__":
    main()