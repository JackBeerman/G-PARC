#!/usr/bin/env python3
"""
GPARC Model Rollout Snapshot Generator for Publication Figures
====================================================================

This script generates high-quality still snapshots from trained GPARC model rollout
predictions at specific timesteps AND saves predictions as numpy files.

Usage:
    # Generate snapshots for specific simulations and timesteps:
    python create_rollout_snapshots_v2.py --model_path best_model.pth \
        --test_files sim1.pt sim2.pt \
        --output_dir ./paper_figures \
        --timesteps 0 5 10 15 \
        --dpi 300 \
        --figure_width 12

    # Generate snapshots for all test files with automatic timestep selection:
    python create_rollout_snapshots_v2.py --model_path best_model.pth \
        --test_dir /path/to/test \
        --output_dir ./paper_figures \
        --auto_timesteps 4 \
        --dpi 300
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error, r2_score

# Add parent directory to path for imports
debug_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, debug_path)

from data.ShockChorddt import ShockTubeRolloutDataset, get_simulation_ids
from utilities.featureextractor import FeatureExtractorGNN
from utilities.embed import SimulationConditionedLayerNorm, GlobalParameterProcessor, GlobalModulatedGNN
from differentiator.differentiator import DerivativeGNN
from integrator.integrator import IntegralGNN
from models.shocktube import GPARC


class SnapshotGenerator:
    """Generate publication-quality snapshot visualizations of rollout predictions."""

    def __init__(self, model, device='cpu', denormalization_params=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.denorm_params = denormalization_params

        # Variable names
        if hasattr(model, 'num_dynamic_feats') and model.num_dynamic_feats == 3:
            self.var_names = ['density', 'x_momentum', 'total_energy']
        else:
            self.var_names = ['density', 'x_momentum', 'y_momentum', 'total_energy']

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

        # Load delta_t denormalization
        delta_t_params = None
        if 'global_param_normalization' in metadata:
            global_params = metadata['global_param_normalization']
            if 'delta_t' in global_params:
                delta_t_params = global_params['delta_t']

        if not delta_t_params and 'delta_t' in norm_params:
            delta_t_params = norm_params['delta_t']
        elif not delta_t_params and 'global_delta_t' in norm_params:
            delta_t_params = norm_params['global_delta_t']

        if delta_t_params:
            self.denorm_params['delta_t'] = delta_t_params

        print(f"Loaded denormalization parameters for: {list(self.denorm_params.keys())}")

    def denormalize_delta_t(self, normalized_delta_t):
        """Convert normalized delta_t back to physical units."""
        if self.denorm_params is None or 'delta_t' not in self.denorm_params:
            return normalized_delta_t

        params = self.denorm_params['delta_t']
        dt_min, dt_max = params['min'], params['max']
        physical_delta_t = normalized_delta_t * (dt_max - dt_min) + dt_min
        return physical_delta_t

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

    def process_targets(self, target_y, skip_indices):
        """Apply same skipping logic to target data as used in training."""
        if skip_indices:
            target_feat_list = []
            for i in range(target_y.shape[1]):
                if i not in skip_indices:
                    target_feat_list.append(target_y[:, i:i+1])
            return torch.cat(target_feat_list, dim=-1) if target_feat_list else target_y
        return target_y

    def _extract_global_attributes(self, data, sim_idx=None):
        """Extract global attributes from data object."""
        if hasattr(data, 'global_pressure') and hasattr(data, 'global_density') and hasattr(data, 'global_delta_t'):
            return data

        if hasattr(data, 'global_params'):
            global_tensor = data.global_params
            if global_tensor.numel() >= 3:
                data.global_pressure = global_tensor[0].unsqueeze(0)
                data.global_density = global_tensor[1].unsqueeze(0)
                data.global_delta_t = global_tensor[2].unsqueeze(0)
                return data

        if hasattr(data, 'pressure') and hasattr(data, 'density'):
            data.global_pressure = data.pressure.unsqueeze(0) if data.pressure.dim() == 0 else data.pressure
            data.global_density = data.density.unsqueeze(0) if data.density.dim() == 0 else data.density

            if hasattr(data, 'delta_t'):
                data.global_delta_t = data.delta_t.unsqueeze(0) if data.delta_t.dim() == 0 else data.delta_t
            elif hasattr(data, 'dt'):
                data.global_delta_t = data.dt.unsqueeze(0) if data.dt.dim() == 0 else data.dt
            else:
                data.global_delta_t = torch.tensor([0.01], device=data.x.device)
            return data

        # Fallback defaults
        data.global_pressure = torch.tensor([1.0], device=data.x.device)
        data.global_density = torch.tensor([1.0], device=data.x.device)
        data.global_delta_t = torch.tensor([0.01], device=data.x.device)
        return data

    def generate_rollout(self, initial_data, rollout_steps):
        """Generate rollout prediction from initial conditions."""
        predictions = []
        F_prev = None

        # Extract global parameters
        global_attrs = torch.stack([
            initial_data.global_pressure.flatten()[0],
            initial_data.global_density.flatten()[0],
            initial_data.global_delta_t.flatten()[0]
        ])

        # Process global parameters
        global_embed = self.model.global_processor(global_attrs)

        # Compute static features once
        static_feats_0 = initial_data.x[:, :self.model.num_static_feats]
        edge_index_0 = initial_data.edge_index
        learned_static_features = self.model.feature_extractor(static_feats_0, edge_index_0)
        learned_static_features = self.model.feature_norm(learned_static_features, global_attrs)

        # Rollout loop
        for step in range(rollout_steps):
            if step == 0:
                # First step: use ground truth
                all_dynamic_feats = initial_data.x[:,
                    self.model.num_static_feats:
                    self.model.num_static_feats + self.model.num_dynamic_feats + len(self.model.skip_dynamic_indices)
                ]
                keep_indices = [i for i in range(all_dynamic_feats.shape[1]) if i not in self.model.skip_dynamic_indices]
                dynamic_feats_t = all_dynamic_feats[:, keep_indices]
            else:
                # Use previous prediction
                dynamic_feats_t = F_prev

            # Normalize dynamic features
            F_prev_used = dynamic_feats_t if F_prev is None else F_prev
            F_prev_used = self.model.derivative_norm(F_prev_used, global_attrs)

            # Broadcast global embedding
            global_context = global_embed.unsqueeze(0).repeat(initial_data.num_nodes, 1)

            # Concatenate features
            Fdot_input = torch.cat([learned_static_features, F_prev_used, global_context], dim=-1)

            # Forward through derivative and integral solvers
            Fdot = self.model.derivative_solver(Fdot_input, edge_index_0)
            Fint = self.model.integral_solver(Fdot, edge_index_0)
            F_pred = F_prev_used + Fint

            predictions.append(F_pred)
            F_prev = F_pred

        return predictions

    def create_snapshot_comparison(self, simulation, timesteps, output_path,
                                   case_name="simulation", figure_width=12, dpi=300):
        """
        Create a comprehensive snapshot comparison figure for publication.
        Layout: Ground truth in top row(s), predictions in bottom row(s).

        Args:
            simulation: List of Data objects for one simulation
            timesteps: List of timestep indices to visualize
            output_path: Path object for saving figure
            case_name: Name of the simulation case
            figure_width: Total width of the figure in inches
            dpi: Resolution for saved figure
        """
        # Move simulation to device and extract global attributes
        for data in simulation:
            data.x = data.x.to(self.device)
            data.y = data.y.to(self.device)
            data.edge_index = data.edge_index.to(self.device)
            data = self._extract_global_attributes(data)
            data.global_pressure = data.global_pressure.to(self.device)
            data.global_density = data.global_density.to(self.device)
            data.global_delta_t = data.global_delta_t.to(self.device)
            if getattr(data, 'edge_attr', None) is not None:
                data.edge_attr = data.edge_attr.to(self.device)

        initial_data = simulation[0]
        max_rollout = max(timesteps) + 1

        # Generate predictions
        with torch.no_grad():
            predictions = self.generate_rollout(initial_data, rollout_steps=max_rollout)

        # Get ground truth
        targets = []
        for i in range(max_rollout):
            target_y = simulation[i].y.cpu()
            target_y = self.process_targets(target_y, self.model.skip_dynamic_indices)
            targets.append(target_y)

        # Extract metadata
        normalized_delta_t = float(initial_data.global_delta_t[0])
        physical_delta_t = self.denormalize_delta_t(normalized_delta_t)
        pressure = float(initial_data.global_pressure[0])
        density = float(initial_data.global_density[0])

        # Determine grid size
        n_nodes = predictions[0].shape[0]
        grid_size = int(np.sqrt(n_nodes))

        # Create figure
        n_timesteps = len(timesteps)
        n_vars = len(self.var_names)
        n_rows = n_vars * 2 # GT and Pred for each variable

        # Calculate dynamic figure height
        # Let's aim for each plot row to be ~2.2 inches tall, plus 1.5 for the title
        figure_height = (n_rows * 2.2) + 1.5
        figsize = (figure_width, figure_height)

        # Layout: 2 * n_vars rows, n_timesteps columns + colorbar
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_rows, n_timesteps + 1, figure=fig,
                      width_ratios=[1]*n_timesteps + [0.1], # Wider colorbar
                      hspace=0.3, wspace=0.05) # Adjusted spacing

        fig.suptitle(f'{case_name}\nP={pressure:.2f}, ρ={density:.2f}, Δt={physical_delta_t:.2e}s',
                     fontsize=14, fontweight='bold', y=0.93)

        # Store data ranges for consistent colormaps
        data_ranges = {}
        for var_idx, var_name in enumerate(self.var_names):
            all_gt = []
            all_pred = []
            for t in timesteps:
                gt = self.denormalize_predictions(targets[t][:, var_idx].numpy(), var_idx)
                pred = self.denormalize_predictions(predictions[t][:, var_idx].cpu().numpy(), var_idx)
                all_gt.append(gt)
                all_pred.append(pred)

            all_data = np.concatenate(all_gt + all_pred)
            data_ranges[var_name] = (all_data.min(), all_data.max())

        # Create subplots - organize by variable, then GT/Pred rows
        for var_idx, var_name in enumerate(self.var_names):
            vmin, vmax = data_ranges[var_name]

            # Calculate row indices for this variable
            gt_row = var_idx * 2
            pred_row = var_idx * 2 + 1

            for t_idx, timestep in enumerate(timesteps):
                # Ground truth row
                ax_gt = fig.add_subplot(gs[gt_row, t_idx])
                gt_data = self.denormalize_predictions(
                    targets[timestep][:, var_idx].numpy(), var_idx
                ).reshape(grid_size, grid_size)

                im_gt = ax_gt.imshow(gt_data, cmap='coolwarm', vmin=vmin, vmax=vmax,
                                     aspect='equal', interpolation='bilinear')
                ax_gt.set_xticks([])
                ax_gt.set_yticks([])

                # Add column titles on first variable's ground truth row
                if var_idx == 0:
                    ax_gt.set_title(f't={timestep}', fontsize=11, fontweight='bold')

                # Add row labels (variable name + GT) on left column
                if t_idx == 0:
                    ax_gt.set_ylabel(f'{var_name.replace("_", " ").title()}\nGround Truth',
                                     fontsize=11, fontweight='bold')

                # Prediction row
                ax_pred = fig.add_subplot(gs[pred_row, t_idx])
                pred_data = self.denormalize_predictions(
                    predictions[timestep][:, var_idx].cpu().numpy(), var_idx
                ).reshape(grid_size, grid_size)

                im_pred = ax_pred.imshow(pred_data, cmap='coolwarm', vmin=vmin, vmax=vmax,
                                         aspect='equal', interpolation='bilinear')
                ax_pred.set_xticks([])
                ax_pred.set_yticks([])

                # Add row label (Prediction only) on left column
                if t_idx == 0:
                    ax_pred.set_ylabel(f'Prediction',
                                       fontsize=11, fontweight='bold')

            # Add colorbar spanning both GT and Pred rows for this variable
            cbar_ax = fig.add_subplot(gs[gt_row:pred_row+1, -1])
            cbar = plt.colorbar(im_pred, cax=cbar_ax)
            cbar.set_label(var_name.replace('_', ' ').title(), fontsize=10)

        # Save figure
        # Note: We removed plt.subplots_adjust and rely on GridSpec spacing
        # and bbox_inches='tight' for a clean layout.
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved snapshot: {output_path}")

    def create_single_variable_snapshot(self, simulation, var_idx, timesteps,
                                        output_path, case_name="simulation",
                                        figure_width=12, dpi=300):
        """
        Create a detailed snapshot comparison for a single variable.
        Shows ground truth, prediction, and absolute error.
        """
        # Process simulation
        for data in simulation:
            data.x = data.x.to(self.device)
            data.y = data.y.to(self.device)
            data.edge_index = data.edge_index.to(self.device)
            data = self._extract_global_attributes(data)
            data.global_pressure = data.global_pressure.to(self.device)
            data.global_density = data.global_density.to(self.device)
            data.global_delta_t = data.global_delta_t.to(self.device)
            if getattr(data, 'edge_attr', None) is not None:
                data.edge_attr = data.edge_attr.to(self.device)

        initial_data = simulation[0]
        max_rollout = max(timesteps) + 1

        # Generate predictions
        with torch.no_grad():
            predictions = self.generate_rollout(initial_data, rollout_steps=max_rollout)

        # Get ground truth
        targets = []
        for i in range(max_rollout):
            target_y = simulation[i].y.cpu()
            target_y = self.process_targets(target_y, self.model.skip_dynamic_indices)
            targets.append(target_y)

        # Extract metadata
        normalized_delta_t = float(initial_data.global_delta_t[0])
        physical_delta_t = self.denormalize_delta_t(normalized_delta_t)
        pressure = float(initial_data.global_pressure[0])
        density = float(initial_data.global_density[0])

        # Grid size
        n_nodes = predictions[0].shape[0]
        grid_size = int(np.sqrt(n_nodes))

        var_name = self.var_names[var_idx]
        n_timesteps = len(timesteps)
        
        # Use fixed height for this 3-row layout, width is controlled
        figsize = (figure_width, 10) 

        # Create figure with 3 rows (GT, Pred, Error)
        fig, axes = plt.subplots(3, n_timesteps, figsize=figsize)
        if n_timesteps == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'{var_name.replace("_", " ").title()} - {case_name}\n' +
                     f'P={pressure:.2f}, ρ={density:.2f}, Δt={physical_delta_t:.2e}s',
                     fontsize=14, fontweight='bold')

        # Determine data ranges
        all_gt = []
        all_pred = []
        for t in timesteps:
            gt = self.denormalize_predictions(targets[t][:, var_idx].numpy(), var_idx)
            pred = self.denormalize_predictions(predictions[t][:, var_idx].cpu().numpy(), var_idx)
            all_gt.append(gt)
            all_pred.append(pred)

        all_data = np.concatenate(all_gt + all_pred)
        vmin, vmax = all_data.min(), all_data.max()

        # Calculate error range
        errors = [np.abs(pred - gt) for pred, gt in zip(all_pred, all_gt)]
        error_max = max([e.max() for e in errors])

        # Plot each timestep
        for t_idx, timestep in enumerate(timesteps):
            gt_data = self.denormalize_predictions(
                targets[timestep][:, var_idx].numpy(), var_idx
            ).reshape(grid_size, grid_size)

            pred_data = self.denormalize_predictions(
                predictions[timestep][:, var_idx].cpu().numpy(), var_idx
            ).reshape(grid_size, grid_size)

            error_data = np.abs(pred_data - gt_data)

            # Ground truth
            ax_gt = axes[0, t_idx]
            im_gt = ax_gt.imshow(gt_data, cmap='coolwarm', vmin=vmin, vmax=vmax,
                                 aspect='equal', interpolation='bilinear')
            ax_gt.set_title(f't={timestep}\nGround Truth', fontsize=11, fontweight='bold')
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            if t_idx == 0:
                ax_gt.set_ylabel('Ground Truth', fontsize=11, fontweight='bold')

            # Prediction
            ax_pred = axes[1, t_idx]
            im_pred = ax_pred.imshow(pred_data, cmap='coolwarm', vmin=vmin, vmax=vmax,
                                     aspect='equal', interpolation='bilinear')
            r2 = r2_score(gt_data.flatten(), pred_data.flatten())
            rmse = np.sqrt(mean_squared_error(gt_data.flatten(), pred_data.flatten()))
            ax_pred.set_title(f'Prediction\nR²={r2:.3f}, RMSE={rmse:.2e}',
                              fontsize=10, fontweight='bold')
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            if t_idx == 0:
                ax_pred.set_ylabel('Prediction', fontsize=11, fontweight='bold')

            # Error
            ax_err = axes[2, t_idx]
            im_err = ax_err.imshow(error_data, cmap='Reds', vmin=0, vmax=error_max,
                                   aspect='equal', interpolation='bilinear')
            ax_err.set_title(f'Absolute Error\nMax={error_data.max():.2e}',
                             fontsize=10, fontweight='bold')
            ax_err.set_xticks([])
            ax_err.set_yticks([])
            if t_idx == 0:
                ax_err.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')

        # Add colorbars
        fig.colorbar(im_gt, ax=axes[0, :], orientation='horizontal',
                     pad=0.05, label=var_name.replace('_', ' ').title())
        fig.colorbar(im_pred, ax=axes[1, :], orientation='horizontal',
                     pad=0.05, label=var_name.replace('_', ' ').title())
        fig.colorbar(im_err, ax=axes[2, :], orientation='horizontal',
                     pad=0.05, label='Absolute Error')

        # Call tight_layout AFTER adding all elements (like colorbars)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent suptitle overlap
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved single-variable snapshot: {output_path}")


def save_predictions(predictions_list, targets_list, case_name, output_path,
                    grid_size, denorm_params=None, var_names=None,
                    save_denormalized=True):
    """
    Save GPARC predictions and ground truth as numpy files.
    
    Args:
        predictions_list: List of prediction tensors [num_nodes, num_feats] for each timestep
        targets_list: List of target tensors [num_nodes, num_feats] for each timestep
        case_name: Name of the simulation case
        output_path: Path object for output directory
        grid_size: Size of the spatial grid (assuming square grid)
        denorm_params: Denormalization parameters (optional)
        var_names: List of variable names
        save_denormalized: If True and denorm_params provided, save denormalized versions
    """
    if var_names is None:
        var_names = ['density', 'x_momentum', 'total_energy']
    
    # Stack predictions and targets into [T, num_nodes, num_feats]
    predictions_stacked = torch.stack([p.cpu() for p in predictions_list], dim=0)
    targets_stacked = torch.stack([t for t in targets_list], dim=0)
    
    # Reshape to [T, C, H, W]
    n_timesteps = predictions_stacked.shape[0]
    n_feats = predictions_stacked.shape[2]
    
    pred_reshaped = predictions_stacked.reshape(n_timesteps, grid_size, grid_size, n_feats)
    pred_reshaped = pred_reshaped.permute(0, 3, 1, 2).numpy()  # [T, C, H, W]
    
    gt_reshaped = targets_stacked.reshape(n_timesteps, grid_size, grid_size, n_feats)
    gt_reshaped = gt_reshaped.permute(0, 3, 1, 2).numpy()  # [T, C, H, W]
    
    # Save normalized versions
    np.save(output_path / f"{case_name}_predictions_normalized.npy", pred_reshaped)
    np.save(output_path / f"{case_name}_ground_truth_normalized.npy", gt_reshaped)
    print(f"Saved normalized predictions: {case_name}_predictions_normalized.npy")
    print(f"Saved normalized ground truth: {case_name}_ground_truth_normalized.npy")
    
    # Save denormalized versions if requested
    if save_denormalized and denorm_params is not None:
        pred_denorm = np.zeros_like(pred_reshaped)
        gt_denorm = np.zeros_like(gt_reshaped)
        
        for ch, var_name in enumerate(var_names):
            if var_name not in denorm_params:
                # If no denorm params, just copy normalized
                pred_denorm[:, ch, :, :] = pred_reshaped[:, ch, :, :]
                gt_denorm[:, ch, :, :] = gt_reshaped[:, ch, :, :]
                continue
            
            params = denorm_params[var_name]
            var_min, var_max = params['min'], params['max']
            
            pred_denorm[:, ch, :, :] = pred_reshaped[:, ch, :, :] * (var_max - var_min) + var_min
            gt_denorm[:, ch, :, :] = gt_reshaped[:, ch, :, :] * (var_max - var_min) + var_min
        
        np.save(output_path / f"{case_name}_predictions_denormalized.npy", pred_denorm)
        np.save(output_path / f"{case_name}_ground_truth_denormalized.npy", gt_denorm)
        print(f"Saved denormalized predictions: {case_name}_predictions_denormalized.npy")
        print(f"Saved denormalized ground truth: {case_name}_ground_truth_denormalized.npy")
    
    # Save metadata
    metadata = {
        'case_name': case_name,
        'shape': pred_reshaped.shape,
        'channels': var_names,
        'n_timesteps': pred_reshaped.shape[0],
        'spatial_dims': (pred_reshaped.shape[2], pred_reshaped.shape[3]),
        'grid_size': grid_size
    }
    
    with open(output_path / f"{case_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {case_name}_metadata.json")


def load_test_simulation(filepath):
    """Load a single test simulation file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    simulation_data = torch.load(filepath, weights_only=False)
    print(f"Loaded {filepath.name}: {len(simulation_data)} timesteps")
    return simulation_data


def auto_select_timesteps(simulation_length, n_timesteps=4):
    """Automatically select evenly-spaced timesteps."""
    if simulation_length <= n_timesteps:
        return list(range(simulation_length))

    # Include first, last, and evenly spaced intermediate timesteps
    indices = [0]  # Always include initial condition

    if n_timesteps > 2:
        step = (simulation_length - 1) / (n_timesteps - 1)
        for i in range(1, n_timesteps - 1):
            indices.append(int(i * step))

    indices.append(simulation_length - 1)  # Always include final timestep

    return sorted(list(set(indices)))  # Remove duplicates and sort


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality snapshots from GPARC rollout predictions"
    )

    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--test_dir", type=str,
                             help="Directory containing test dataset files")
    input_group.add_argument("--test_files", type=str, nargs='+',
                             help="Specific test simulation files")

    parser.add_argument("--output_dir", type=str, default="./paper_figures",
                        help="Output directory for snapshots")

    # Timestep selection
    timestep_group = parser.add_mutually_exclusive_group()
    timestep_group.add_argument("--timesteps", type=int, nargs='+',
                                  help="Specific timesteps to visualize")
    timestep_group.add_argument("--auto_timesteps", type=int, default=4,
                                  help="Number of timesteps to auto-select")

    # Model architecture arguments (must match training)
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=3)
    parser.add_argument("--skip_dynamic_indices", type=int, nargs='+', default=[2])
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--pool_ratios", type=float, default=0.1)
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

    # Output options
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution for saved figures")
    parser.add_argument("--figure_width", type=float, default=12,
                        help="Total figure width in inches")
    parser.add_argument("--single_variable", type=str, choices=['density', 'x_momentum', 'total_energy'],
                        help="Create detailed snapshots for single variable only")
    parser.add_argument("--max_files", type=int,
                        help="Maximum number of files to process from directory")
    parser.add_argument("--skip_visualization", action='store_true',
                        help="Skip generating visualization images (only save numpy files)")
    parser.add_argument("--skip_numpy", action='store_true',
                        help="Skip saving numpy files (only generate visualizations)")

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_path}")

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)

    global_embed_dim = 64

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
    print("Model loaded successfully!")

    # Initialize snapshot generator
    generator = SnapshotGenerator(model, device)

    # Load denormalization parameters
    if args.test_dir:
        norm_metadata = Path(args.test_dir).parent / 'normalization_metadata.json'
    elif args.test_files:
        norm_metadata = Path(args.test_files[0]).parent / 'normalization_metadata.json'
        if not norm_metadata.exists():
            norm_metadata = Path(args.test_files[0]).parent.parent / 'normalization_metadata.json'

    if norm_metadata.exists():
        generator.load_denormalization_params(norm_metadata)

    # Get list of files to process
    if args.test_files:
        test_file_paths = [Path(f) for f in args.test_files]
    else:
        test_file_paths = list(Path(args.test_dir).glob("*.pt"))
        if args.max_files:
            test_file_paths = test_file_paths[:args.max_files]

    print(f"\nProcessing {len(test_file_paths)} simulation(s)...")

    # Process each file
    for file_path in test_file_paths:
        print(f"\nProcessing: {file_path.name}")

        # Load simulation
        simulation = load_test_simulation(file_path)
        case_name = file_path.stem

        # Move simulation to device and extract global attributes
        for data in simulation:
            data.x = data.x.to(device)
            data.y = data.y.to(device)
            data.edge_index = data.edge_index.to(device)
            data = generator._extract_global_attributes(data)
            data.global_pressure = data.global_pressure.to(device)
            data.global_density = data.global_density.to(device)
            data.global_delta_t = data.global_delta_t.to(device)
            if getattr(data, 'edge_attr', None) is not None:
                data.edge_attr = data.edge_attr.to(device)

        initial_data = simulation[0]
        
        # Generate predictions for full simulation
        with torch.no_grad():
            predictions = generator.generate_rollout(initial_data, rollout_steps=len(simulation))

        # Get ground truth for full simulation
        targets = []
        for i in range(len(simulation)):
            target_y = simulation[i].y.cpu()
            target_y = generator.process_targets(target_y, model.skip_dynamic_indices)
            targets.append(target_y)

        # Determine grid size
        n_nodes = predictions[0].shape[0]
        grid_size = int(np.sqrt(n_nodes))

        # Save predictions as numpy files
        if not args.skip_numpy:
            print("\nSaving predictions to numpy files...")
            save_predictions(
                predictions,
                targets,
                case_name,
                output_path,
                grid_size=grid_size,
                denorm_params=generator.denorm_params,
                var_names=generator.var_names,
                save_denormalized=True
            )

        # Generate visualizations
        if not args.skip_visualization:
            # Determine timesteps
            if args.timesteps:
                timesteps = args.timesteps
                # Filter out timesteps beyond simulation length
                timesteps = [t for t in timesteps if t < len(simulation)]
            else:
                timesteps = auto_select_timesteps(len(simulation), args.auto_timesteps)

            if not timesteps:
                print("No valid timesteps found to plot. Skipping visualization.")
            else:
                print(f"Selected timesteps: {timesteps}")

                # Generate snapshots
                if args.single_variable:
                    # Single variable detailed view
                    var_idx = generator.var_names.index(args.single_variable)
                    output_file = output_path / f"snapshot_{case_name}_{args.single_variable}.png"
                    generator.create_single_variable_snapshot(
                        simulation, var_idx, timesteps, output_file,
                        case_name=case_name, figure_width=args.figure_width, dpi=args.dpi
                    )
                else:
                    # Full comparison view
                    output_file = output_path / f"snapshot_{case_name}_all_vars.png"
                    generator.create_snapshot_comparison(
                        simulation, timesteps, output_file,
                        case_name=case_name, figure_width=args.figure_width, dpi=args.dpi
                    )

    print(f"\n✓ All outputs saved to: {output_path}")


if __name__ == "__main__":
    main()