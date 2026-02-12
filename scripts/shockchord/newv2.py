#!/usr/bin/env python3
"""
GPARC Model Rollout Evaluation Script for Shock Tube Physics (Updated with Delta_t Analysis)
====================================================================

This script evaluates a trained GPARC model using rollout prediction mode,
where the model receives only initial conditions and predicts the entire sequence.
Enhanced to include delta_t in filenames and analyze performance across different delta_t values.
Updated to work with the new direct delta_t approach.

Usage:
    # Test all files in a directory:
    python evaluate_gparc_model_updated.py --model_path best_model.pth --test_dir /path/to/test --output_dir ./evaluation
    
    # Test specific simulation files:
    python evaluate_gparc_model_updated.py --model_path best_model.pth --test_files sim1.pt sim2.pt sim3.pt --output_dir ./evaluation
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

debug_path = os.path.join(os.path.dirname(__file__), '..', '..')
print(f"Script location: {__file__}")
print(f"Adding to path: {os.path.abspath(debug_path)}")
print(f"Files in that directory: {os.listdir(debug_path) if os.path.exists(debug_path) else 'Directory not found'}")
sys.path.insert(0, debug_path)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.ShockChorddt import ShockTubeRolloutDataset, get_simulation_ids
# NEW: Import the refactored model components
from utilities.featureextractor import FeatureExtractorGNN
from utilities.embed import SimulationConditionedLayerNorm, GlobalParameterProcessor, GlobalModulatedGNN
from utilities.trainer import train_and_validate, load_model, plot_loss_curves
from differentiator.differentiator import DerivativeGNN
from integrator.integrator import ExplicitIntegralGNN as IntegralGNN  # Updated to use modified integrator
from models.shocktubev2 import GPARC

################################################################################
# ENHANCED ROLLOUT EVALUATOR WITH DELTA_T ANALYSIS (UPDATED FOR DIRECT DELTA_T)
################################################################################

class GPARCRolloutEvaluator:
    """
    Evaluator for GPARC models using rollout prediction mode.
    Enhanced with delta_t analysis and updated for direct delta_t approach.
    """
    
    def __init__(self, model, device='cpu', denormalization_params=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.denorm_params = denormalization_params
        # Adjust variable names based on number of features (skip meaningless ones)
        if hasattr(model, 'num_dynamic_feats') and model.num_dynamic_feats == 3:
            # Skip the meaningless third variable, so we have density, x_momentum, total_energy
            self.var_names = ['density', 'x_momentum', 'total_energy']
        else:
            self.var_names = ['density', 'x_momentum', 'y_momentum', 'total_energy']
        
        # For delta_t performance tracking
        self.delta_t_metrics = defaultdict(list)
    
    def process_targets(self, target_y, skip_indices):
        """Apply same skipping logic to target data as used in training"""
        if skip_indices:
            target_feat_list = []
            for i in range(target_y.shape[1]):
                if i not in skip_indices:
                    target_feat_list.append(target_y[:, i:i+1])
            return torch.cat(target_feat_list, dim=-1) if target_feat_list else target_y
        return target_y
    
    def load_denormalization_params(self, metadata_file):
        """Load denormalization parameters from normalization metadata."""
        if not Path(metadata_file).exists():
            print(f"Warning: Normalization metadata file not found: {metadata_file}")
            return
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.denorm_params = {}
        norm_params = metadata.get('normalization_params', {})
        
        # Load variable denormalization parameters
        for var in self.var_names:
            if var in norm_params:
                self.denorm_params[var] = norm_params[var]
        
        # Load delta_t denormalization parameters - check multiple possible locations
        delta_t_params = None
        
        # Method 1: Check in global_param_normalization (your normalization script format)
        if 'global_param_normalization' in metadata:
            global_params = metadata['global_param_normalization']
            if 'delta_t' in global_params:
                delta_t_params = global_params['delta_t']
                print(f"Found delta_t in global_param_normalization: min={delta_t_params['min']}, max={delta_t_params['max']}")
        
        # Method 2: Check in normalization_params (alternative format)
        if not delta_t_params:
            if 'delta_t' in norm_params:
                delta_t_params = norm_params['delta_t']
                print(f"Found delta_t in normalization_params: min={delta_t_params['min']}, max={delta_t_params['max']}")
            elif 'global_delta_t' in norm_params:
                delta_t_params = norm_params['global_delta_t']
                print(f"Found global_delta_t in normalization_params: min={delta_t_params['min']}, max={delta_t_params['max']}")
        
        if delta_t_params:
            self.denorm_params['delta_t'] = delta_t_params
        else:
            print("Warning: No delta_t normalization parameters found in metadata")
            print("Available in global_param_normalization:", list(metadata.get('global_param_normalization', {}).keys()))
            print("Available in normalization_params:", list(norm_params.keys()))
        
        print(f"Loaded denormalization parameters for: {list(self.denorm_params.keys())}")
    
    def denormalize_delta_t(self, normalized_delta_t):
        """Convert normalized delta_t back to physical units."""
        if self.denorm_params is None or 'delta_t' not in self.denorm_params:
            print("Warning: No delta_t denormalization parameters available")
            return normalized_delta_t
        
        params = self.denorm_params['delta_t']
        dt_min, dt_max = params['min'], params['max']
        
        # Denormalize: normalized_value * (max - min) + min
        physical_delta_t = normalized_delta_t * (dt_max - dt_min) + dt_min
        return physical_delta_t
    
    def _extract_global_attributes(self, data, sim_idx=None):
        """
        Extract global attributes from data object, handling multiple possible formats.
        """
        # Method 1: Check for already processed global attributes
        if hasattr(data, 'global_pressure') and hasattr(data, 'global_density') and hasattr(data, 'global_delta_t'):
            return data
        
        # Method 2: Check for global_params tensor (as expected by dataset creation)
        if hasattr(data, 'global_params'):
            global_tensor = data.global_params
            if global_tensor.numel() >= 3:
                data.global_pressure = global_tensor[0].unsqueeze(0)
                data.global_density = global_tensor[1].unsqueeze(0) 
                data.global_delta_t = global_tensor[2].unsqueeze(0)
                return data
        
        # Method 3: Check for individual attributes (pressure, density, delta_t)
        if hasattr(data, 'pressure') and hasattr(data, 'density'):
            data.global_pressure = data.pressure.unsqueeze(0) if data.pressure.dim() == 0 else data.pressure
            data.global_density = data.density.unsqueeze(0) if data.density.dim() == 0 else data.density
            
            # Check for delta_t or dt
            if hasattr(data, 'delta_t'):
                data.global_delta_t = data.delta_t.unsqueeze(0) if data.delta_t.dim() == 0 else data.delta_t
            elif hasattr(data, 'dt'):
                data.global_delta_t = data.dt.unsqueeze(0) if data.dt.dim() == 0 else data.dt
            else:
                data.global_delta_t = torch.tensor([0.01], device=data.x.device)
                if sim_idx is not None:
                    print(f"Warning: delta_t not found in simulation {sim_idx}, using default value 0.01")
            
            return data
        
        # Method 4: Extract from filename if available
        if hasattr(data, 'case_name') or sim_idx is not None:
            filename = getattr(data, 'case_name', f'sim_{sim_idx}')
            pressure, density, delta_t = self._extract_from_filename(filename)
            data.global_pressure = torch.tensor([pressure], device=data.x.device)
            data.global_density = torch.tensor([density], device=data.x.device)
            data.global_delta_t = torch.tensor([delta_t], device=data.x.device)
            if sim_idx is not None:
                print(f"Extracted global attributes from filename for simulation {sim_idx}: P={pressure}, ρ={density}, Δt={delta_t}")
            return data
        
        # Method 5: Default values as fallback
        data.global_pressure = torch.tensor([1.0], device=data.x.device)
        data.global_density = torch.tensor([1.0], device=data.x.device) 
        data.global_delta_t = torch.tensor([0.01], device=data.x.device)
        if sim_idx is not None:
            print(f"Warning: No global attributes found in simulation {sim_idx}, using default values")
        
        return data
    
    def _extract_from_filename(self, filename):
        """Extract pressure, density, and delta_t from filename"""
        import re
        
        pressure_match = re.search(r'p_L_(\d+)', filename)
        density_match = re.search(r'rho_L_([\d.]+)', filename)
        # Look for delta_t in various formats: dt_0.01, delta_t_0.01, deltaT_0.01
        delta_t_match = re.search(r'd(?:elta_?)?[tT]_?([\d.]+)', filename)
        
        pressure = float(pressure_match.group(1)) if pressure_match else 1.0
        density = float(density_match.group(1)) if density_match else 1.0
        delta_t = float(delta_t_match.group(1)) if delta_t_match else 0.01
        
        return pressure, density, delta_t
    
    def _format_delta_t_string(self, delta_t_value):
        """Format delta_t value for use in filenames (scientific notation for very small values)"""
        # Handle very small values with scientific notation
        if abs(delta_t_value) < 0.0001:  # Less than 0.1 milliseconds
            # Format as scientific notation: 1.028e-05
            formatted = f"{delta_t_value:.3e}"
            # Replace 'e-0' with 'e-' and 'e+0' with 'e+' for cleaner filenames
            formatted = formatted.replace('e-0', 'e-').replace('e+0', 'e+')
        else:
            # For larger values, use decimal format
            formatted = f"{delta_t_value:.6f}".rstrip('0').rstrip('.')
        
        return formatted
    
    def denormalize_predictions(self, normalized_data, var_idx):
        """Convert normalized predictions back to physical units."""
        if self.denorm_params is None:
            return normalized_data
        
        var_name = self.var_names[var_idx]
        if var_name not in self.denorm_params:
            return normalized_data
        
        params = self.denorm_params[var_name]
        var_min, var_max = params['min'], params['max']
        
        return normalized_data * (var_max - var_min) + var_min
    
    def generate_rollout(self, initial_data, rollout_steps):
        """
        Generate a rollout prediction from initial conditions.
        Updated to work with the direct delta_t approach.
        """
        predictions = []
        F_prev = None
    
        # Extract global parameters based on model's delta_t usage
        global_pressure = initial_data.global_pressure.flatten()[0]
        global_density = initial_data.global_density.flatten()[0]
        delta_t = initial_data.global_delta_t.flatten()[0]
        
        # Handle global parameters based on model configuration
        if hasattr(self.model, 'use_direct_delta_t') and self.model.use_direct_delta_t:
            # Only process pressure and density
            global_attrs = torch.stack([global_pressure, global_density])
        else:
            # Original approach: process all three parameters
            global_attrs = torch.stack([global_pressure, global_density, delta_t])
        
        # Process global parameters once
        global_embed = self.model.global_processor(global_attrs)
    
        # Compute static features once
        static_feats_0 = initial_data.x[:, :self.model.num_static_feats]
        edge_index_0 = initial_data.edge_index
        learned_static_features = self.model.feature_extractor(static_feats_0, edge_index_0)
        learned_static_features = self.model.feature_norm(learned_static_features, global_attrs)
    
        # Rollout loop
        for step in range(rollout_steps):
            if step == 0:
                # First step: use ground truth dynamic features
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
            else:
                # Use previous prediction
                dynamic_feats_t = F_prev
    
            # Normalize dynamic features
            F_prev_used = dynamic_feats_t if F_prev is None else F_prev
            F_prev_used = self.model.derivative_norm(F_prev_used, global_attrs)
    
            # Broadcast global embedding to all nodes
            global_context = global_embed.unsqueeze(0).repeat(initial_data.num_nodes, 1)
    
            # Concatenate all features
            Fdot_input = torch.cat([learned_static_features, F_prev_used, global_context], dim=-1)
    
            # Forward through derivative solver
            Fdot = self.model.derivative_solver(Fdot_input, edge_index_0)
            
            # Forward through integral solver with proper delta_t handling
            if hasattr(self.model, 'use_direct_delta_t') and self.model.use_direct_delta_t:
                # Pass delta_t directly to the integrator
                Fint = self.model.integral_solver(Fdot, edge_index_0, delta_t=delta_t)
            else:
                # Original approach: integrator learns the scaling
                Fint = self.model.integral_solver(Fdot, edge_index_0)
            
            F_pred = F_prev_used + Fint
    
            predictions.append(F_pred)
            F_prev = F_pred
    
        return predictions

    # ... [Rest of the methods remain the same - they don't need changes for delta_t handling] ...
    
    def evaluate_rollout_predictions(self, simulations, rollout_steps=10):
        """
        Generate rollout predictions from loaded simulation files.
        Enhanced to track delta_t specific performance with denormalization.
        """
        all_predictions = []
        all_targets = []
        metadata = []
        
        # Debug: Track delta_t values we encounter
        delta_t_debug_info = []
        
        with torch.no_grad():
            for sim_idx, simulation in enumerate(tqdm(simulations, desc="Generating rollout predictions")):
                
                # Move simulation to device and extract global attributes
                for data in simulation:
                    data.x = data.x.to(self.device)
                    data.y = data.y.to(self.device)
                    data.edge_index = data.edge_index.to(self.device)
                    
                    # Extract global attributes using the helper function
                    data = self._extract_global_attributes(data, sim_idx)
                    
                    # Move global attributes to device
                    data.global_pressure = data.global_pressure.to(self.device)
                    data.global_density = data.global_density.to(self.device)
                    data.global_delta_t = data.global_delta_t.to(self.device)
                    
                    if getattr(data, 'edge_attr', None) is not None:
                        data.edge_attr = data.edge_attr.to(self.device)
                
                # Use only the first timestep as initial condition
                initial_data = simulation[0]
                
                # Extract normalized and physical delta_t
                normalized_delta_t = float(initial_data.global_delta_t[0])
                physical_delta_t = self.denormalize_delta_t(normalized_delta_t)
                case_name = getattr(initial_data, 'case_name', f'sim_{sim_idx}')
                
                # Debug: Collect delta_t information
                delta_t_debug_info.append({
                    'sim_idx': sim_idx,
                    'case_name': case_name,
                    'normalized_delta_t': normalized_delta_t,
                    'physical_delta_t': physical_delta_t,
                    'source': 'extracted_from_data'
                })
                
                # Determine how many steps to predict
                max_available_steps = len(simulation)
                actual_rollout_steps = min(rollout_steps, max_available_steps)
                
                # Generate rollout predictions
                rollout_predictions = self.generate_rollout(
                    initial_data, 
                    rollout_steps=actual_rollout_steps
                )
                
                # Collect ground truth targets for comparison (with variable skipping)
                rollout_targets = []
                for i in range(actual_rollout_steps):
                    target_y = simulation[i].y.cpu()
                    # Apply same variable skipping to targets as used in training
                    target_y = self.process_targets(target_y, self.model.skip_dynamic_indices)
                    rollout_targets.append(target_y)
                
                all_predictions.append([pred.cpu() for pred in rollout_predictions])
                all_targets.append(rollout_targets)
                
                # Extract metadata with physical delta_t information
                delta_t_str = self._format_delta_t_string(physical_delta_t)
                
                sim_metadata = {
                    'simulation_idx': sim_idx,
                    'case_name': case_name,
                    'pressure': float(initial_data.global_pressure[0]),
                    'density': float(initial_data.global_density[0]),
                    'delta_t_normalized': normalized_delta_t,
                    'delta_t_physical': physical_delta_t,
                    'delta_t': physical_delta_t,  # Use physical for compatibility
                    'delta_t_str': delta_t_str,
                    'rollout_length': len(rollout_predictions),
                    'available_targets': len(simulation),
                    'skip_dynamic_indices': self.model.skip_dynamic_indices,
                    'num_dynamic_feats': self.model.num_dynamic_feats
                }
                metadata.append(sim_metadata)
                
                # Track delta_t specific performance using physical delta_t
                self._track_delta_t_performance(rollout_predictions, rollout_targets, delta_t_str)
        
        # Debug output: Show delta_t value distribution
        print(f"\nDelta_t Debug Information:")
        print("="*60)
        unique_normalized = {}
        unique_physical = {}
        
        for info in delta_t_debug_info:
            norm_dt = info['normalized_delta_t']
            phys_dt = info['physical_delta_t']
            
            if norm_dt not in unique_normalized:
                unique_normalized[norm_dt] = []
            unique_normalized[norm_dt].append(info['case_name'])
            
            if phys_dt not in unique_physical:
                unique_physical[phys_dt] = []
            unique_physical[phys_dt].append(info['case_name'])
        
        print(f"Found {len(unique_normalized)} unique normalized delta_t values:")
        for norm_dt, case_names in sorted(unique_normalized.items()):
            print(f"  Normalized Δt = {norm_dt:.6f}: {len(case_names)} simulations")
            if len(case_names) <= 2:
                print(f"    Examples: {', '.join(case_names)}")
        
        print(f"\nCorresponding physical delta_t values:")
        for phys_dt, case_names in sorted(unique_physical.items()):
            print(f"  Physical Δt = {phys_dt:.8f} s: {len(case_names)} simulations")
            if len(case_names) <= 2:
                print(f"    Examples: {', '.join(case_names)}")
        
        # Show conversion verification
        norm_values = list(unique_normalized.keys())
        phys_values = list(unique_physical.keys())
        print(f"\nDelta_t Conversion Verification:")
        print(f"Normalized range: {min(norm_values):.6f} to {max(norm_values):.6f}")
        print(f"Physical range: {min(phys_values):.8f} to {max(phys_values):.8f} seconds")
        
        print(f"\nGenerated rollout predictions for {len(all_predictions)} simulations")
        return all_predictions, all_targets, metadata
    
    def _track_delta_t_performance(self, predictions, targets, delta_t_str):
        """Track performance metrics for specific delta_t values."""
        # Compute overall performance for this simulation
        all_preds = []
        all_targs = []
        
        for step_pred, step_targ in zip(predictions, targets):
            # Ensure tensors are on CPU before converting to numpy
            all_preds.append(step_pred.cpu().numpy())
            all_targs.append(step_targ.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targs = np.concatenate(all_targs, axis=0)
        
        # Overall metrics
        pred_flat = all_preds.flatten()
        target_flat = all_targs.flatten()
        
        overall_metrics = {
            'mse': float(mean_squared_error(target_flat, pred_flat)),
            'mae': float(mean_absolute_error(target_flat, pred_flat)),
            'rmse': float(np.sqrt(mean_squared_error(target_flat, pred_flat))),
            'r2': float(r2_score(target_flat, pred_flat))
        }
        
        # Per-variable metrics
        var_metrics = {}
        for i, var_name in enumerate(self.var_names):
            pred_var = all_preds[:, i]
            target_var = all_targs[:, i]
            
            var_metrics[var_name] = {
            'mse': float(mean_squared_error(target_var, pred_var)),
            'mae': float(mean_absolute_error(target_var, pred_var)),
            'rmse': float(np.sqrt(mean_squared_error(target_var, pred_var))),
            'r2': float(r2_score(target_var, pred_var))
        }
        
        # Store metrics for this delta_t
        self.delta_t_metrics[delta_t_str].append({
            'overall': overall_metrics,
            'variables': var_metrics
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
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targs = np.concatenate(all_targs, axis=0)
        
        # Per-variable metrics
        for i, var_name in enumerate(self.var_names):
            pred_var = all_preds[:, i]
            target_var = all_targs[:, i]
            
            pred_var_phys = self.denormalize_predictions(pred_var, i)
            target_var_phys = self.denormalize_predictions(target_var, i)
            
            metrics[var_name] = {
                'mse': float(mean_squared_error(target_var, pred_var)),
                'mae': float(mean_absolute_error(target_var, pred_var)),
                'rmse': float(np.sqrt(mean_squared_error(target_var, pred_var))),
                'r2': float(r2_score(target_var, pred_var)),
                'mse_physical': float(mean_squared_error(target_var_phys, pred_var_phys)),
                'mae_physical': float(mean_absolute_error(target_var_phys, pred_var_phys)),
                'rmse_physical': float(np.sqrt(mean_squared_error(target_var_phys, pred_var_phys)))
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
    
    def analyze_delta_t_performance(self):
        """Analyze performance across different delta_t values."""
        if not self.delta_t_metrics:
            return {}
        
        delta_t_analysis = {}
        
        for delta_t_str, metrics_list in self.delta_t_metrics.items():
            if not metrics_list:
                continue
                
            # Aggregate metrics across all simulations for this delta_t
            overall_metrics = []
            var_metrics = {var_name: [] for var_name in self.var_names}
            
            for sim_metrics in metrics_list:
                overall_metrics.append(sim_metrics['overall'])
                for var_name in self.var_names:
                    if var_name in sim_metrics['variables']:
                        var_metrics[var_name].append(sim_metrics['variables'][var_name])
            
            # Compute statistics (mean, std, min, max)
            analysis = {
                'num_simulations': len(metrics_list),
                'delta_t_value': float(delta_t_str),
                'overall': self._compute_metric_statistics(overall_metrics),
                'variables': {}
            }
            
            for var_name in self.var_names:
                if var_metrics[var_name]:
                    analysis['variables'][var_name] = self._compute_metric_statistics(var_metrics[var_name])
            
            delta_t_analysis[delta_t_str] = analysis
        
        return delta_t_analysis

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
    
    def create_delta_t_performance_table(self):
        """Create a summary table of performance across delta_t values."""
        delta_t_analysis = self.analyze_delta_t_performance()
        
        if not delta_t_analysis:
            return "No delta_t analysis data available."
        
        # Create table data
        table_data = []
        headers = ['Delta_t', 'Num_Sims', 'Overall_R²', 'Overall_RMSE']
        
        # Add variable-specific headers
        for var_name in self.var_names:
            headers.extend([f'{var_name.title()}_R²', f'{var_name.title()}_RMSE'])
        
        # Sort by delta_t value
        sorted_items = sorted(delta_t_analysis.items(), key=lambda x: x[1]['delta_t_value'])
        
        for delta_t_str, analysis in sorted_items:
            row = [
                f"{analysis['delta_t_value']:.4f}",
                str(analysis['num_simulations']),
                f"{analysis['overall']['r2']['mean']:.4f} ± {analysis['overall']['r2']['std']:.4f}",
                f"{analysis['overall']['rmse']['mean']:.6f} ± {analysis['overall']['rmse']['std']:.6f}"
            ]
            
            # Add variable-specific data
            for var_name in self.var_names:
                if var_name in analysis['variables']:
                    var_data = analysis['variables'][var_name]
                    row.extend([
                        f"{var_data['r2']['mean']:.4f} ± {var_data['r2']['std']:.4f}",
                        f"{var_data['rmse']['mean']:.6f} ± {var_data['rmse']['std']:.6f}"
                    ])
                else:
                    row.extend(['N/A', 'N/A'])
            
            table_data.append(row)
        
        # Format as string table
        table_str = "Delta_t Performance Analysis\n"
        table_str += "=" * 80 + "\n"
        
        # Print headers
        header_line = " | ".join(f"{h:>12}" for h in headers)
        table_str += header_line + "\n"
        table_str += "-" * len(header_line) + "\n"
        
        # Print data rows
        for row in table_data:
            row_line = " | ".join(f"{cell:>12}" for cell in row)
            table_str += row_line + "\n"
        
        return table_str
    
    def plot_rollout_evolution(self, predictions, targets, metadata, seq_idx=0, figsize=(20, 10)):
        """Plot how prediction accuracy evolves over rollout timesteps."""
        if seq_idx >= len(predictions):
            return None
        
        seq_pred = predictions[seq_idx]
        seq_targ = targets[seq_idx]
        seq_meta = metadata[seq_idx]
        
        case_name = seq_meta['case_name']
        delta_t_str = seq_meta['delta_t_str']
        max_steps = min(len(seq_pred), len(seq_targ))
        
        # Calculate errors over time
        timestep_errors = []
        for t in range(max_steps):
            pred_t = seq_pred[t].numpy()
            targ_t = seq_targ[t].numpy()
            
            var_errors = []
            for i in range(len(self.var_names)):
                mse = mean_squared_error(targ_t[:, i], pred_t[:, i])
                var_errors.append(mse)
            timestep_errors.append(var_errors)
        
        timestep_errors = np.array(timestep_errors)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Rollout Evolution: {case_name} (Δt={delta_t_str})', fontsize=16)
        
        # Error over time
        ax = axes[0]
        for i, var_name in enumerate(self.var_names):
            ax.plot(range(max_steps), timestep_errors[:, i], 'o-', label=var_name, linewidth=2)
        ax.set_xlabel('Rollout Timestep')
        ax.set_ylabel('MSE')
        ax.set_title('Prediction Error vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Spatial error at final timestep
        ax = axes[1]
        grid_size = int(np.sqrt(seq_pred[0].shape[0]))
        final_pred = seq_pred[-1][:, 0].numpy().reshape(grid_size, grid_size)
        final_targ = seq_targ[-1][:, 0].numpy().reshape(grid_size, grid_size)
        error_2d = np.abs(final_pred - final_targ)
        
        im = ax.imshow(error_2d, cmap='Reds', aspect='auto')
        ax.set_title(f'Final Density Error (t={max_steps-1})')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        return fig

    def create_rollout_gifs(self, predictions, targets, metadata, seq_idx, output_dir, performance_category=None):
        """Create animated GIFs showing rollout evolution for all variables with delta_t in filename."""
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
        
        case_name = seq_meta['case_name']
        delta_t_str = seq_meta['delta_t_str']
        max_steps = min(len(seq_pred), len(seq_targ))
        
        if max_steps < 2:
            print(f"Skipping GIF for {case_name}: insufficient timesteps")
            return
        
        # Determine grid size
        n_nodes = seq_pred[0].shape[0]
        grid_size = int(np.sqrt(n_nodes))
        
        if grid_size * grid_size != n_nodes:
            print(f"Warning: Non-square grid detected for {case_name}. Using closest square.")
            grid_size = int(np.sqrt(n_nodes))
        
        # Adjust subplot layout based on number of variables
        if len(self.var_names) == 3:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        else:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
        title_suffix = f" ({performance_category.replace('_', ' ').title()})" if performance_category else ""
        base_title = f'Rollout Evolution: {case_name} (Δt={delta_t_str}){title_suffix}'
        fig.suptitle(base_title, fontsize=16)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Initialize plots
        ims = []
        cbars = []
        
        for i, var_name in enumerate(self.var_names):
            # Ground truth subplot
            ax_gt = axes_flat[i]
            ax_gt.set_title(f'{var_name.title()} (Ground Truth)')
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            
            # Prediction subplot  
            ax_pred = axes_flat[i + len(self.var_names)]
            ax_pred.set_title(f'{var_name.title()} (Prediction)')
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            
            # Get data range for consistent colormap
            all_gt_data = np.concatenate([
                self.denormalize_predictions(seq_targ[t][:, i].numpy(), i) 
                for t in range(max_steps)
            ])
            all_pred_data = np.concatenate([
                self.denormalize_predictions(seq_pred[t][:, i].numpy(), i) 
                for t in range(max_steps)
            ])
            
            vmin = min(all_gt_data.min(), all_pred_data.min())
            vmax = max(all_gt_data.max(), all_pred_data.max())
            
            # Initial plots
            gt_data = self.denormalize_predictions(seq_targ[0][:, i].numpy(), i)
            pred_data = self.denormalize_predictions(seq_pred[0][:, i].numpy(), i)
            
            gt_2d = gt_data.reshape(grid_size, grid_size)
            pred_2d = pred_data.reshape(grid_size, grid_size)
            
            im_gt = ax_gt.imshow(gt_2d, cmap='coolwarm', vmin=vmin, vmax=vmax, aspect='auto')
            im_pred = ax_pred.imshow(pred_2d, cmap='coolwarm', vmin=vmin, vmax=vmax, aspect='auto')
            
            ims.append((im_gt, im_pred))
            
            # Add colorbars
            cbar_gt = plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
            cbar_pred = plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
            cbars.append((cbar_gt, cbar_pred))
        
        # Hide unused subplots if we have 3 variables
        if len(self.var_names) == 3 and len(axes_flat) > 6:
            for idx in range(6, len(axes_flat)):
                axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        
        def animate(frame):
            """Animation function for updating the plots."""
            for i, var_name in enumerate(self.var_names):
                # Get data for current frame
                gt_data = self.denormalize_predictions(seq_targ[frame][:, i].numpy(), i)
                pred_data = self.denormalize_predictions(seq_pred[frame][:, i].numpy(), i)
                
                gt_2d = gt_data.reshape(grid_size, grid_size)
                pred_2d = pred_data.reshape(grid_size, grid_size)
                
                # Update images
                ims[i][0].set_array(gt_2d)
                ims[i][1].set_array(pred_2d)
            
            # Update title with timestep
            title_with_timestep = f'{base_title} (Timestep {frame}/{max_steps-1})'
            fig.suptitle(title_with_timestep, fontsize=16)
            
            return [im for im_pair in ims for im in im_pair]
        
        # Create animation
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, animate, frames=max_steps, interval=500, blit=False)
        
        # Save as GIF with delta_t in filename
        gif_filename = f'rollout_evolution_{case_name}_dt{delta_t_str}_all_vars'
        if performance_category:
            gif_filename = f'rollout_evolution_{performance_category}_{case_name}_dt{delta_t_str}_all_vars'
        gif_path = output_dir / f'{gif_filename}.gif'
        writer = PillowWriter(fps=2)
        anim.save(gif_path, writer=writer)
        plt.close(fig)
        
        print(f"Saved rollout GIF: {gif_path}")

    def create_error_evolution_gif(self, predictions, targets, metadata, seq_idx, output_dir, performance_category=None):
        """Create GIF showing how prediction errors evolve over time with delta_t in filename."""
        try:
            from matplotlib.animation import PillowWriter
        except ImportError:
            return
        
        if seq_idx >= len(predictions):
            return
        
        seq_pred = predictions[seq_idx]
        seq_targ = targets[seq_idx]
        seq_meta = metadata[seq_idx]
        
        case_name = seq_meta['case_name']
        delta_t_str = seq_meta['delta_t_str']
        max_steps = min(len(seq_pred), len(seq_targ))
        
        if max_steps < 2:
            return
        
        # Create title with performance category and delta_t
        title_suffix = f" ({performance_category.replace('_', ' ').title()})" if performance_category else ""
        base_title = f'Prediction Error Evolution: {case_name} (Δt={delta_t_str}){title_suffix}'
        
        # Determine grid size
        n_nodes = seq_pred[0].shape[0]
        grid_size = int(np.sqrt(n_nodes))
        
        # Create figure for error evolution
        if len(self.var_names) == 3:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        fig.suptitle(base_title, fontsize=14)
        axes_flat = axes.flatten() if len(self.var_names) > 1 else [axes]
        
        # Calculate error ranges for consistent colormaps
        max_errors = []
        for i, var_name in enumerate(self.var_names):
            errors = []
            for t in range(max_steps):
                gt_data = self.denormalize_predictions(seq_targ[t][:, i].numpy(), i)
                pred_data = self.denormalize_predictions(seq_pred[t][:, i].numpy(), i)
                error = np.abs(pred_data - gt_data)
                errors.append(error)
            max_error = np.max(errors)
            max_errors.append(max_error)
        
        # Initialize plots
        ims = []
        for i, var_name in enumerate(self.var_names):
            ax = axes_flat[i]
            ax.set_title(f'{var_name.title()} Absolute Error')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Initial error
            gt_data = self.denormalize_predictions(seq_targ[0][:, i].numpy(), i)
            pred_data = self.denormalize_predictions(seq_pred[0][:, i].numpy(), i)
            error_2d = np.abs(pred_data - gt_data).reshape(grid_size, grid_size)
            
            im = ax.imshow(error_2d, cmap='Reds', vmin=0, vmax=max_errors[i], aspect='auto')
            ims.append(im)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        def animate_error(frame):
            """Animation function for error evolution."""
            for i, var_name in enumerate(self.var_names):
                gt_data = self.denormalize_predictions(seq_targ[frame][:, i].numpy(), i)
                pred_data = self.denormalize_predictions(seq_pred[frame][:, i].numpy(), i)
                error_2d = np.abs(pred_data - gt_data).reshape(grid_size, grid_size)
                
                ims[i].set_array(error_2d)
            
            # Update title with timestep
            title_with_timestep = f'{base_title} (Timestep {frame}/{max_steps-1})'
            fig.suptitle(title_with_timestep, fontsize=14)
            
            return ims
        
        # Create and save animation
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, animate_error, frames=max_steps, interval=500, blit=False)
        
        # Save as GIF with delta_t in filename
        gif_filename = f'rollout_error_evolution_{case_name}_dt{delta_t_str}'
        if performance_category:
            gif_filename = f'rollout_error_evolution_{performance_category}_{case_name}_dt{delta_t_str}'
        gif_path = output_dir / f'{gif_filename}.gif'
        
        writer = PillowWriter(fps=2)
        anim.save(gif_path, writer=writer)
        plt.close(fig)
        
        print(f"Saved error evolution GIF: {gif_path}")


################################################################################
# UPDATED MODEL CREATION FUNCTION
################################################################################

def create_model_for_evaluation(args, use_direct_delta_t=None):
    """Create model with same architecture as training, with delta_t configuration."""
    
    # Auto-detect delta_t usage if not specified
    if use_direct_delta_t is None:
        # Try to infer from args or default to True for new models
        use_direct_delta_t = getattr(args, 'use_direct_delta_t', True)
    
    # Global embedding dimension changes based on delta_t usage
    if use_direct_delta_t:
        global_embed_dim = 64  # Only pressure and density
    else:
        global_embed_dim = 64  # All three parameters (original)

    feature_extractor = FeatureExtractorGNN(
        in_channels=args.num_static_feats,
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
        feature_out_channels=args.feature_out_channels,
        use_direct_delta_t=use_direct_delta_t  # Pass the delta_t configuration
    )
    
    return model


################################################################################
# [Include all utility functions and main evaluation function unchanged]
################################################################################

################################################################################
# UTILITY FUNCTIONS
################################################################################

def select_simulations_by_performance(predictions, targets, metadata, n_samples=3):
    """Select simulations for visualization based on rollout prediction performance."""
    if len(predictions) == 0:
        return []
    
    # Calculate per-simulation rollout performance
    sim_performances = []
    
    for sim_idx, (pred_seq, targ_seq) in enumerate(zip(predictions, targets)):
        if len(pred_seq) == 0 or len(targ_seq) == 0:
            continue
            
        # Calculate cumulative error over the rollout sequence
        total_mse = 0
        total_points = 0
        
        # Weight later timesteps more heavily since rollout error accumulates
        for step_idx, (pred_step, targ_step) in enumerate(zip(pred_seq, targ_seq)):
            pred_array = pred_step.numpy()
            targ_array = targ_step.numpy()
            
            # Calculate MSE for this timestep
            step_mse = np.mean((pred_array - targ_array) ** 2)
            
            # Weight by timestep (later steps weighted more)
            weight = 1.0 + 0.1 * step_idx  # Gradually increasing weight
            total_mse += step_mse * weight
            total_points += weight
        
        # Average weighted MSE for this simulation
        if total_points > 0:
            avg_weighted_mse = total_mse / total_points
            sim_performances.append((sim_idx, avg_weighted_mse))
    
    if len(sim_performances) == 0:
        return list(range(min(n_samples, len(predictions))))
    
    # Sort by performance (lower MSE = better performance)
    sim_performances.sort(key=lambda x: x[1])
    
    # Select best, median, worst (or available subset)
    selected_indices = []
    n_available = len(sim_performances)
    
    if n_available >= 1:
        # Best performing (lowest error)
        selected_indices.append(sim_performances[0][0])
    
    if n_available >= 2 and n_samples >= 2:
        # Median performing
        median_idx = sim_performances[n_available // 2][0]
        selected_indices.append(median_idx)
    
    if n_available >= 3 and n_samples >= 3:
        # Worst performing (highest error)
        selected_indices.append(sim_performances[-1][0])
    
    # Fill remaining slots if needed
    while len(selected_indices) < min(n_samples, n_available):
        for sim_idx, _ in sim_performances:
            if sim_idx not in selected_indices:
                selected_indices.append(sim_idx)
                break
    
    return selected_indices[:n_samples]


def get_performance_category(sim_idx, predictions, targets):
    """Get performance category label for a simulation."""
    if len(predictions) == 0:
        return "unknown"
    
    # Calculate performance for all simulations
    all_performances = []
    for pred_seq, targ_seq in zip(predictions, targets):
        if len(pred_seq) == 0 or len(targ_seq) == 0:
            continue
        total_mse = 0
        total_points = 0
        for step_idx, (pred_step, targ_step) in enumerate(zip(pred_seq, targ_seq)):
            pred_array = pred_step.numpy()
            targ_array = targ_step.numpy()
            step_mse = np.mean((pred_array - targ_array) ** 2)
            weight = 1.0 + 0.1 * step_idx
            total_mse += step_mse * weight
            total_points += weight
        if total_points > 0:
            all_performances.append(total_mse / total_points)
    
    if len(all_performances) == 0:
        return "unknown"
    
    # Calculate current simulation's performance
    pred_seq = predictions[sim_idx]
    targ_seq = targets[sim_idx]
    total_mse = 0
    total_points = 0
    for step_idx, (pred_step, targ_step) in enumerate(zip(pred_seq, targ_seq)):
        pred_array = pred_step.numpy()
        targ_array = targ_step.numpy()
        step_mse = np.mean((pred_array - targ_array) ** 2)
        weight = 1.0 + 0.1 * step_idx
        total_mse += step_mse * weight
        total_points += weight
    
    if total_points == 0:
        return "unknown"
    
    current_performance = total_mse / total_points
    
    # Determine category based on percentiles
    sorted_perfs = sorted(all_performances)
    n_sims = len(sorted_perfs)
    
    # Find percentile rank
    rank = sorted_perfs.index(min(sorted_perfs, key=lambda x: abs(x - current_performance)))
    percentile = rank / n_sims
    
    if percentile <= 0.33:
        return "best_performance"
    elif percentile <= 0.67:
        return "median_performance"
    else:
        return "worst_performance"


def load_test_simulations(test_dir=None, test_files=None, file_pattern="*.pt", max_files=None):
    """Load test simulation files for rollout evaluation."""
    simulations = []
    
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
                simulation_data = torch.load(file_path, weights_only=False)
                simulations.append(simulation_data)
                print(f"  Loaded {file_path.name}: {len(simulation_data)} timesteps")
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
                simulation_data = torch.load(file_path, weights_only=False)
                simulations.append(simulation_data)
                print(f"  Loaded {file_path.name}: {len(simulation_data)} timesteps")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    else:
        raise ValueError("Either test_dir or test_files must be provided")
    
    if len(simulations) == 0:
        raise ValueError("No simulation files were successfully loaded")
    
    return simulations


################################################################################
# UPDATED MAIN EVALUATION FUNCTION
################################################################################

def evaluate_gparc_rollout(model_path, test_dir, test_files, output_dir, args):
    """Evaluate GPARC model using rollout prediction mode with delta_t analysis."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check for delta_t configuration in checkpoint
    use_direct_delta_t = True  # Default for new models
    if 'model_config' in checkpoint:
        use_direct_delta_t = checkpoint['model_config'].get('use_direct_delta_t', True)
        print(f"Found model config in checkpoint: use_direct_delta_t = {use_direct_delta_t}")
    else:
        print(f"No model config found in checkpoint, using default: use_direct_delta_t = {use_direct_delta_t}")
    
    # Create model with proper configuration
    model = create_model_for_evaluation(args, use_direct_delta_t=use_direct_delta_t)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Model loaded successfully!")
    print(f"Model using direct delta_t: {getattr(model, 'use_direct_delta_t', 'Not specified')}")
    
    # Load test simulations
    simulations = load_test_simulations(
        test_dir=test_dir,
        test_files=test_files,
        file_pattern="*.pt", 
        max_files=args.max_sequences
    )
    
    # Initialize evaluator
    evaluator = GPARCRolloutEvaluator(model, device)
    
    # Load denormalization params
    if test_dir:
        norm_metadata_file = Path(test_dir).parent / 'normalization_metadata.json'
    elif test_files and len(test_files) > 0:
        first_file_dir = Path(test_files[0]).parent
        norm_metadata_file = first_file_dir / 'normalization_metadata.json'
        if not norm_metadata_file.exists():
            norm_metadata_file = first_file_dir.parent / 'normalization_metadata.json'
    else:
        norm_metadata_file = None
    
    if norm_metadata_file and norm_metadata_file.exists():
        evaluator.load_denormalization_params(norm_metadata_file)
    
    # Generate rollout predictions
    print(f"\nGenerating rollout predictions ({args.rollout_steps} steps)...")
    predictions, targets, metadata = evaluator.evaluate_rollout_predictions(
        simulations, rollout_steps=args.rollout_steps
    )
    
    # Compute metrics
    print("Computing metrics...")
    metrics = evaluator.compute_rollout_metrics(predictions, targets)
    
    # Analyze delta_t performance
    print("Analyzing performance across different delta_t values...")
    delta_t_analysis = evaluator.analyze_delta_t_performance()
    
    # Print delta_t performance table
    print("\n" + evaluator.create_delta_t_performance_table())
    
    # [Continue with rest of evaluation as before...]
    
    # Select diverse simulations for visualization based on performance
    print("Selecting representative simulations for visualization...")
    selected_indices = select_simulations_by_performance(predictions, targets, metadata, n_samples=3)
    
    print(f"Selected simulations for visualization:")
    for idx in selected_indices:
        case_name = metadata[idx]['case_name']
        delta_t_str = metadata[idx]['delta_t_str']
        performance_category = get_performance_category(idx, predictions, targets)
        print(f"  - Simulation {idx} ({case_name}, Δt={delta_t_str}): {performance_category}")
    
    # Generate visualizations (basic plots without complex 3D analysis)
    print("Creating basic visualizations...")
    
    # Select diverse simulations for visualization based on performance
    print("Selecting representative simulations for visualization...")
    selected_indices = select_simulations_by_performance(predictions, targets, metadata, n_samples=3)
    
    print(f"Selected simulations for visualization:")
    for idx in selected_indices:
        case_name = metadata[idx]['case_name']
        delta_t_str = metadata[idx]['delta_t_str']
        performance_category = get_performance_category(idx, predictions, targets)
        print(f"  - Simulation {idx} ({case_name}, Δt={delta_t_str}): {performance_category}")
    
    # Create static plots for selected simulations
    print("Creating static plots...")
    for i, sim_idx in enumerate(selected_indices):
        fig = evaluator.plot_rollout_evolution(predictions, targets, metadata, sim_idx)
        if fig:
            case_name = metadata[sim_idx]['case_name']
            delta_t_str = metadata[sim_idx]['delta_t_str']
            performance_cat = get_performance_category(sim_idx, predictions, targets).replace(' ', '_')
            fig.savefig(output_path / f'rollout_evolution_{performance_cat}_{case_name}_dt{delta_t_str}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Saved rollout evolution plot for {case_name}")
    
    # Create animated GIFs for selected simulations
    print("Creating animated GIFs...")
    for sim_idx in selected_indices:
        case_name = metadata[sim_idx]['case_name']
        print(f"  - Creating GIFs for simulation {sim_idx} ({case_name})...")
        performance_cat = get_performance_category(sim_idx, predictions, targets)
        
        # 2D GIFs
        evaluator.create_rollout_gifs(predictions, targets, metadata, sim_idx, output_path, performance_cat)
        evaluator.create_error_evolution_gif(predictions, targets, metadata, sim_idx, output_path, performance_cat)
    
    # Create simple performance summary
    print("Creating performance summary...")
    
    # Simple scatter plot for prediction vs target
    def plot_prediction_vs_target_scatter(evaluator, predictions, targets, figsize=(12, 8)):
        """Create scatter plots comparing predictions vs targets."""
        all_preds = []
        all_targs = []
        
        for seq_pred, seq_targ in zip(predictions, targets):
            for step_pred, step_targ in zip(seq_pred, seq_targ):
                all_preds.append(step_pred.numpy())
                all_targs.append(step_targ.numpy())
        
        if len(all_preds) == 0:
            return None
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targs = np.concatenate(all_targs, axis=0)
        
        # Adjust subplot layout based on number of variables
        if len(evaluator.var_names) == 3:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()[:3]
        else:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()
        
        fig.suptitle('Rollout Prediction vs Target', fontsize=16)
        
        for i, var_name in enumerate(evaluator.var_names):
            ax = axes[i]
            pred_var = all_preds[:, i]
            target_var = all_targs[:, i]
            
            # Sample for visualization
            n_points = min(3000, len(pred_var))
            indices = np.random.choice(len(pred_var), n_points, replace=False)
            
            pred_sample = pred_var[indices]
            target_sample = target_var[indices]
            
            pred_phys = evaluator.denormalize_predictions(pred_sample, i)
            target_phys = evaluator.denormalize_predictions(target_sample, i)
            
            ax.scatter(target_phys, pred_phys, alpha=0.5, s=1)
            
            min_val = min(target_phys.min(), pred_phys.min())
            max_val = max(target_phys.max(), pred_phys.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            r2 = r2_score(target_phys, pred_phys)
            ax.set_title(f'{var_name.title()} (R² = {r2:.3f})')
            ax.set_xlabel('Target')
            ax.set_ylabel('Prediction')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplot if we have 3 variables
        if len(evaluator.var_names) == 3 and len(axes) > 3:
            axes[3].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    # Create scatter plot
    fig_scatter = plot_prediction_vs_target_scatter(evaluator, predictions, targets)
    if fig_scatter:
        fig_scatter.savefig(output_path / 'rollout_scatter.png', dpi=300, bbox_inches='tight')
        plt.close(fig_scatter)
        print("  - Saved rollout scatter plot")
    
    # Create delta_t performance summary
    def create_delta_t_summary_plot(evaluator, figsize=(12, 6)):
        """Create a simple delta_t performance plot."""
        delta_t_analysis = evaluator.analyze_delta_t_performance()
        
        if not delta_t_analysis:
            return None
        
        # Extract data for plotting
        delta_t_values = []
        overall_r2_means = []
        overall_rmse_means = []
        
        # Sort by delta_t value
        sorted_items = sorted(delta_t_analysis.items(), key=lambda x: x[1]['delta_t_value'])
        
        for delta_t_str, analysis in sorted_items:
            delta_t_values.append(analysis['delta_t_value'])
            overall_r2_means.append(analysis['overall']['r2']['mean'])
            overall_rmse_means.append(analysis['overall']['rmse']['mean'])
        
        if len(delta_t_values) == 0:
            return None
        
        # Convert to numpy arrays
        delta_t_values = np.array(delta_t_values)
        overall_r2_means = np.array(overall_r2_means)
        overall_rmse_means = np.array(overall_rmse_means)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Model Performance vs Physical Delta_t', fontsize=14)
        
        # R² vs delta_t
        ax1.scatter(delta_t_values, overall_r2_means, s=50, alpha=0.7, color='blue')
        ax1.set_xlabel('Physical Delta_t (seconds)')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Overall R² Performance')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2e}'))
        
        # RMSE vs delta_t
        ax2.scatter(delta_t_values, overall_rmse_means, s=50, alpha=0.7, color='red')
        ax2.set_xlabel('Physical Delta_t (seconds)')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Overall RMSE')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2e}'))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2e}'))
        
        plt.tight_layout()
        return fig
    
    # Create delta_t summary plot
    fig_delta_summary = create_delta_t_summary_plot(evaluator)
    if fig_delta_summary:
        fig_delta_summary.savefig(output_path / 'delta_t_performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close(fig_delta_summary)
        print("  - Saved delta_t performance summary")
    
    print("Basic visualizations completed.")

    
    # Save results with model configuration
    results = {
        'metrics': metrics,
        'delta_t_analysis': delta_t_analysis,
        'metadata': metadata,
        'model_info': {
            'model_path': str(model_path),
            'use_direct_delta_t': use_direct_delta_t,
            'test_simulations': len(predictions),
            'rollout_steps': args.rollout_steps,
            'device': str(device),
            'skip_dynamic_indices': model.skip_dynamic_indices,
            'num_dynamic_feats': model.num_dynamic_feats,
            'test_source': 'specific_files' if test_files else 'directory',
            'test_files': test_files if test_files else None,
            'test_dir': str(test_dir) if test_dir else None,
            'architecture': {
                'feature_out_channels': args.feature_out_channels,
                'deriv_hidden_channels': args.deriv_hidden_channels,
                'deriv_num_layers': args.deriv_num_layers,
                'integral_hidden_channels': args.integral_hidden_channels,
                'integral_num_layers': args.integral_num_layers
            }
        }
    }
    
    with open(output_path / 'rollout_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRollout evaluation complete! Results saved to: {output_path}")
    return metrics, evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPARC model with rollout prediction and delta_t analysis")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--test_dir", type=str,
                           help="Directory containing test dataset files")
    input_group.add_argument("--test_files", type=str, nargs='+',
                           help="Specific test simulation files to evaluate")
    
    parser.add_argument("--output_dir", type=str, default="./rollout_evaluation",
                       help="Output directory for evaluation results")
    
    # Model architecture - Feature extractor (MUST MATCH TRAINING SCRIPT)
    parser.add_argument("--num_static_feats", type=int, default=2,
                       help="Number of static features")
    parser.add_argument("--num_dynamic_feats", type=int, default=3,
                       help="Number of dynamic features to use (after skipping)")
    parser.add_argument("--skip_dynamic_indices", type=int, nargs='+', default=[2],
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
    
    # Delta_t configuration (for compatibility)
    parser.add_argument("--use_direct_delta_t", action="store_true", default=None,
                       help="Override delta_t usage (if not auto-detected from checkpoint)")
    
    # Evaluation settings
    parser.add_argument("--max_sequences", type=int, default=30,
                       help="Maximum number of simulations to evaluate")
    parser.add_argument("--rollout_steps", type=int, default=10,
                       help="Number of timesteps to predict in rollout")
    
    args = parser.parse_args()
    
    evaluate_gparc_rollout(args.model_path, args.test_dir, args.test_files, args.output_dir, args)


if __name__ == "__main__":
    main()