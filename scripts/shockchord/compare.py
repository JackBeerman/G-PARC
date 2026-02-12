#!/usr/bin/env python3
"""
Multi-Model Comparison Visualization
====================================

This script loads predictions from multiple models (saved as numpy files) and creates
comparison visualizations showing ground truth vs multiple model predictions.

Usage:
    python compare_models.py \
        --parcv2_dir ./parcv2_predictions \
        --gparc_dir ./gparc_predictions \
        --output_dir ./model_comparisons \
        --timesteps 0 10 20 30 \
        --dpi 300
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, r2_score


# Channel configurations
CHANNEL_MAP = {
    'Density (ρ)': 0,
    'x-Momentum (ρu)': 1,
    'Energy (E)': 2,
}

CMAP_CONFIG = {
    'Density (ρ)': {'cmap': 'coolwarm', 'label': 'kg/m³'},
    'x-Momentum (ρu)': {'cmap': 'coolwarm', 'label': 'kg/(m²·s)'},
    'Energy (E)': {'cmap': 'coolwarm', 'label': 'J/m³'},
}


class MultiModelComparisonVisualizer:
    """Create comparison visualizations from multiple model predictions."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.var_names = list(CHANNEL_MAP.keys())
        
    def load_predictions(self, pred_dir, case_name, use_denormalized=True):
        """
        Load predictions and ground truth from a directory.
        
        Args:
            pred_dir: Directory containing prediction numpy files
            case_name: Name of the simulation case
            use_denormalized: If True, load denormalized files, else normalized
            
        Returns:
            tuple: (predictions, ground_truth) as numpy arrays [T, C, H, W]
        """
        pred_dir = Path(pred_dir)
        
        suffix = "denormalized" if use_denormalized else "normalized"
        
        pred_file = pred_dir / f"{case_name}_predictions_{suffix}.npy"
        gt_file = pred_dir / f"{case_name}_ground_truth_{suffix}.npy"
        
        if not pred_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
        
        predictions = np.load(pred_file)
        ground_truth = np.load(gt_file)
        
        print(f"Loaded from {pred_dir.name}:")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Ground truth shape: {ground_truth.shape}")
        
        return predictions, ground_truth
    
    def load_metadata(self, pred_dir, case_name):
        """Load metadata for the simulation case."""
        metadata_file = Path(pred_dir) / f"{case_name}_metadata.json"
        
        if not metadata_file.exists():
            return None
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def compute_metrics(self, ground_truth, prediction):
        """Compute error metrics between ground truth and prediction."""
        gt_flat = ground_truth.flatten()
        pred_flat = prediction.flatten()
        
        rmse = np.sqrt(mean_squared_error(gt_flat, pred_flat))
        r2 = r2_score(gt_flat, pred_flat)
        mae = np.mean(np.abs(gt_flat - pred_flat))
        max_error = np.max(np.abs(gt_flat - pred_flat))
        
        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'max_error': max_error
        }
    
    def create_single_feature_comparison(self, case_name, feature_idx, timesteps,
                                        ground_truth, model_predictions,
                                        model_names, figure_width=16, dpi=300,
                                        output_format='png'):
        """
        Create a comparison figure for a single feature showing ground truth and all models.
        
        Args:
            case_name: Name of the simulation case
            feature_idx: Index of the feature to visualize (0, 1, or 2)
            timesteps: List of timestep indices to visualize
            ground_truth: Ground truth array [T, C, H, W]
            model_predictions: Dict mapping model_name -> predictions array [T, C, H, W]
            model_names: List of model names in desired order
            figure_width: Total figure width in inches
            dpi: Resolution for saved figure
            output_format: 'png' or 'pdf'
        """
        feature_name = self.var_names[feature_idx]
        cmap_info = CMAP_CONFIG[feature_name]
        
        n_timesteps = len(timesteps)
        n_models = len(model_names)
        n_rows = n_models + 1  # Ground truth + each model
        
        # Calculate figure height dynamically
        plot_width = (figure_width * 0.9) / n_timesteps
        plot_height = plot_width
        fig_height = (n_rows * plot_height) + 1.5
        
        fig = plt.figure(figsize=(figure_width, fig_height))
        
        # Layout: n_rows rows, n_timesteps columns + colorbar column
        gs = GridSpec(n_rows, n_timesteps + 1, figure=fig,
                      width_ratios=[1]*n_timesteps + [0.05],
                      hspace=0.25, wspace=0.05)
        
        fig.suptitle(f'{feature_name} Comparison: {case_name}',
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Determine global data range across all timesteps and models
        all_data = []
        for t in timesteps:
            if t >= ground_truth.shape[0]:
                continue
            all_data.append(ground_truth[t, feature_idx, ...].flatten())
            for model_name in model_names:
                all_data.append(model_predictions[model_name][t, feature_idx, ...].flatten())
        
        all_data = np.concatenate(all_data)
        vmin, vmax = all_data.min(), all_data.max()
        
        # Row 0: Ground Truth
        for t_idx, timestep in enumerate(timesteps):
            if timestep >= ground_truth.shape[0]:
                fig.add_subplot(gs[0, t_idx]).axis('off')
                continue
            
            ax = fig.add_subplot(gs[0, t_idx])
            gt_data = ground_truth[timestep, feature_idx, ...]
            
            im = ax.imshow(gt_data, cmap=cmap_info['cmap'], vmin=vmin, vmax=vmax,
                          aspect='auto', interpolation='bilinear', origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Column title
            if t_idx == 0:
                ax.set_ylabel('Ground Truth\n', fontsize=11, fontweight='bold')
            
            # Timestep label at top
            ax.set_title(f't={timestep}', fontsize=11, fontweight='bold')
        
        # Rows 1+: Model Predictions
        for model_idx, model_name in enumerate(model_names):
            row_idx = model_idx + 1
            
            for t_idx, timestep in enumerate(timesteps):
                if timestep >= model_predictions[model_name].shape[0]:
                    fig.add_subplot(gs[row_idx, t_idx]).axis('off')
                    continue
                
                ax = fig.add_subplot(gs[row_idx, t_idx])
                pred_data = model_predictions[model_name][timestep, feature_idx, ...]
                gt_data = ground_truth[timestep, feature_idx, ...]
                
                im = ax.imshow(pred_data, cmap=cmap_info['cmap'], vmin=vmin, vmax=vmax,
                              aspect='auto', interpolation='bilinear', origin='lower')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Row label (model name) on first column
                if t_idx == 0:
                    ax.set_ylabel(f'{model_name}\n', fontsize=11, fontweight='bold')
                
                # Compute and display metrics
                metrics = self.compute_metrics(gt_data, pred_data)
                ax.text(0.95, 0.05, f"R²={metrics['r2']:.3f}\nRMSE={metrics['rmse']:.2e}",
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add colorbar spanning all rows
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label(cmap_info['label'], fontsize=10, rotation=0, labelpad=20)
        cbar.ax.tick_params(labelsize=9)
        
        # Save figure
        output_file = self.output_dir / f"comparison_{case_name}_{feature_name.replace(' ', '_').replace('(', '').replace(')', '')}.{output_format}"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved comparison: {output_file}")
    
    def create_error_comparison(self, case_name, feature_idx, timesteps,
                               ground_truth, model_predictions, model_names,
                               figure_width=16, dpi=300, output_format='png'):
        """
        Create an error comparison figure showing absolute error for each model.
        
        Args:
            case_name: Name of the simulation case
            feature_idx: Index of the feature to visualize
            timesteps: List of timestep indices to visualize
            ground_truth: Ground truth array [T, C, H, W]
            model_predictions: Dict mapping model_name -> predictions array [T, C, H, W]
            model_names: List of model names in desired order
            figure_width: Total figure width in inches
            dpi: Resolution for saved figure
            output_format: 'png' or 'pdf'
        """
        feature_name = self.var_names[feature_idx]
        
        n_timesteps = len(timesteps)
        n_models = len(model_names)
        
        # Calculate figure dimensions
        plot_width = (figure_width * 0.9) / n_timesteps
        plot_height = plot_width
        fig_height = (n_models * plot_height) + 1.5
        
        fig = plt.figure(figsize=(figure_width, fig_height))
        
        gs = GridSpec(n_models, n_timesteps + 1, figure=fig,
                      width_ratios=[1]*n_timesteps + [0.05],
                      hspace=0.25, wspace=0.05)
        
        fig.suptitle(f'{feature_name} Absolute Error Comparison: {case_name}',
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Compute global error range
        max_error = 0
        for model_name in model_names:
            for t in timesteps:
                if t >= ground_truth.shape[0]:
                    continue
                error = np.abs(model_predictions[model_name][t, feature_idx, ...] - 
                              ground_truth[t, feature_idx, ...])
                max_error = max(max_error, error.max())
        
        # Create error plots for each model
        for model_idx, model_name in enumerate(model_names):
            for t_idx, timestep in enumerate(timesteps):
                if timestep >= ground_truth.shape[0]:
                    fig.add_subplot(gs[model_idx, t_idx]).axis('off')
                    continue
                
                ax = fig.add_subplot(gs[model_idx, t_idx])
                
                pred_data = model_predictions[model_name][timestep, feature_idx, ...]
                gt_data = ground_truth[timestep, feature_idx, ...]
                error_data = np.abs(pred_data - gt_data)
                
                im = ax.imshow(error_data, cmap='Reds', vmin=0, vmax=max_error,
                              aspect='auto', interpolation='bilinear', origin='lower')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Row label on first column
                if t_idx == 0:
                    ax.set_ylabel(f'{model_name}\nError', fontsize=11, fontweight='bold')
                
                # Timestep label on first row
                if model_idx == 0:
                    ax.set_title(f't={timestep}', fontsize=11, fontweight='bold')
                
                # Display max error
                ax.text(0.95, 0.05, f"Max={error_data.max():.2e}",
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add colorbar
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Absolute Error', fontsize=10, rotation=90, labelpad=15)
        cbar.ax.tick_params(labelsize=9)
        
        # Save figure
        output_file = self.output_dir / f"error_{case_name}_{feature_name.replace(' ', '_').replace('(', '').replace(')', '')}.{output_format}"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved error comparison: {output_file}")
    
    def create_metrics_table(self, case_name, ground_truth, model_predictions,
                            model_names, timesteps):
        """
        Create and save a table of quantitative metrics for all models.
        """
        metrics_file = self.output_dir / f"metrics_{case_name}.txt"
        
        with open(metrics_file, 'w') as f:
            f.write(f"Quantitative Metrics for {case_name}\n")
            f.write("=" * 80 + "\n\n")
            
            for feature_idx, feature_name in enumerate(self.var_names):
                f.write(f"\n{feature_name}\n")
                f.write("-" * 80 + "\n")
                
                for model_name in model_names:
                    f.write(f"\n{model_name}:\n")
                    
                    for timestep in timesteps:
                        if timestep >= ground_truth.shape[0]:
                            continue
                        
                        gt_data = ground_truth[timestep, feature_idx, ...]
                        pred_data = model_predictions[model_name][timestep, feature_idx, ...]
                        
                        metrics = self.compute_metrics(gt_data, pred_data)
                        
                        f.write(f"  t={timestep:3d}: ")
                        f.write(f"R²={metrics['r2']:7.4f}, ")
                        f.write(f"RMSE={metrics['rmse']:.4e}, ")
                        f.write(f"MAE={metrics['mae']:.4e}, ")
                        f.write(f"Max Error={metrics['max_error']:.4e}\n")
                
                f.write("\n")
        
        print(f"Saved metrics table: {metrics_file}")


def auto_select_timesteps(simulation_length, n_timesteps=4):
    """Automatically select evenly-spaced timesteps."""
    if simulation_length <= n_timesteps:
        return list(range(simulation_length))
    
    indices = [0]  # Always include initial condition
    
    if n_timesteps > 2:
        step = (simulation_length - 1) / (n_timesteps - 1)
        for i in range(1, n_timesteps - 1):
            indices.append(int(i * step))
    
    indices.append(simulation_length - 1)  # Always include final timestep
    
    return sorted(list(set(indices)))


def find_common_cases(parcv2_dir, gparc_dir):
    """Find simulation cases that exist in both model directories, handling name differences."""
    parcv2_path = Path(parcv2_dir)
    gparc_path = Path(gparc_dir)
    
    # Get PARCv2 cases
    parcv2_files = list(parcv2_path.glob("*_predictions_*.npy"))
    parcv2_cases = set()
    for pred_file in parcv2_files:
        name = pred_file.stem
        if '_predictions_denormalized' in name:
            case_name = name.replace('_predictions_denormalized', '')
        elif '_predictions_normalized' in name:
            case_name = name.replace('_predictions_normalized', '')
        else:
            continue
        parcv2_cases.add(case_name)
    
    # Get GPARC cases and try to match them to PARCv2
    gparc_files = list(gparc_path.glob("*_predictions_*.npy"))
    gparc_cases = {}
    for pred_file in gparc_files:
        name = pred_file.stem
        if '_predictions_denormalized' in name:
            gparc_case = name.replace('_predictions_denormalized', '')
        elif '_predictions_normalized' in name:
            gparc_case = name.replace('_predictions_normalized', '')
        else:
            continue
        
        # Try to match to PARCv2 case by removing common GPARC suffixes
        # GPARC has: p_L_162500_rho_L_1.75_test_with_pos_normalized
        # PARCv2 has: p_L_162500_rho_L_1.75
        parcv2_match = gparc_case.replace('_test_with_pos_normalized', '').replace('_test_with_pos', '')
        
        if parcv2_match in parcv2_cases:
            gparc_cases[parcv2_match] = gparc_case
    
    return gparc_cases


def main():
    parser = argparse.ArgumentParser(
        description="Compare predictions from multiple models"
    )
    
    # Model directories
    parser.add_argument("--parcv2_dir", type=str, required=True,
                        help="Directory containing PARCv2 predictions")
    parser.add_argument("--gparc_dir", type=str, required=True,
                        help="Directory containing GPARC predictions")
    parser.add_argument("--output_dir", type=str, default="./model_comparisons",
                        help="Output directory for comparison figures")
    
    # Case selection
    parser.add_argument("--cases", type=str, nargs='+',
                        help="Specific cases to compare (default: all common cases)")
    
    # Timestep selection
    timestep_group = parser.add_mutually_exclusive_group()
    timestep_group.add_argument("--timesteps", type=int, nargs='+',
                                help="Specific timesteps to visualize")
    timestep_group.add_argument("--auto_timesteps", type=int, default=4,
                                help="Number of timesteps to auto-select")
    
    # Data options
    parser.add_argument("--use_normalized", action='store_true',
                        help="Use normalized data instead of denormalized")
    
    # Output options
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution for saved figures")
    parser.add_argument("--figure_width", type=float, default=16,
                        help="Total figure width in inches")
    parser.add_argument("--output_format", type=str, default='png',
                        choices=['png', 'pdf'],
                        help="Output format for figures")
    parser.add_argument("--include_error_plots", action='store_true',
                        help="Generate separate error comparison plots")
    parser.add_argument("--include_metrics_table", action='store_true',
                        help="Generate quantitative metrics tables")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = MultiModelComparisonVisualizer(args.output_dir)
    
    model_names = ['PARCv2', 'GPARC']  # Order for display
    
    # Find cases to process - this returns {parcv2_name: gparc_name}
    case_mappings = find_common_cases(args.parcv2_dir, args.gparc_dir)
    
    if not case_mappings:
        print("ERROR: No common cases found in both model directories")
        print(f"\nPARCv2 directory: {args.parcv2_dir}")
        print(f"GPARC directory: {args.gparc_dir}")
        return
    
    print(f"Found {len(case_mappings)} common case(s)")
    for parcv2_name, gparc_name in case_mappings.items():
        print(f"  {parcv2_name} <-> {gparc_name}")
    
    use_denormalized = not args.use_normalized
    
    # Process each case
    for parcv2_case, gparc_case in case_mappings.items():
        print(f"\n{'='*60}")
        print(f"Processing case: {parcv2_case}")
        print(f"  PARCv2 file: {parcv2_case}")
        print(f"  GPARC file: {gparc_case}")
        print(f"{'='*60}")
        
        # Load predictions from both models with their specific names
        try:
            # Load PARCv2
            parcv2_pred, ground_truth = visualizer.load_predictions(
                args.parcv2_dir, parcv2_case, use_denormalized=use_denormalized
            )
            
            # Load GPARC with its different name
            gparc_pred, gparc_gt = visualizer.load_predictions(
                args.gparc_dir, gparc_case, use_denormalized=use_denormalized
            )
            
            # Handle different sequence lengths - use the shorter one
            min_length = min(ground_truth.shape[0], gparc_gt.shape[0], 
                           parcv2_pred.shape[0], gparc_pred.shape[0])
            
            print(f"  Sequence lengths: PARCv2={parcv2_pred.shape[0]}, GPARC={gparc_pred.shape[0]}")
            print(f"  Using first {min_length} timesteps")
            
            # Trim all arrays to the same length
            ground_truth = ground_truth[:min_length]
            gparc_gt = gparc_gt[:min_length]
            parcv2_pred = parcv2_pred[:min_length]
            gparc_pred = gparc_pred[:min_length]
            
            # Now verify ground truths match (on the overlapping portion)
            if not np.allclose(ground_truth, gparc_gt, rtol=1e-5, atol=1e-8):
                print(f"WARNING: Ground truth mismatch between models for {parcv2_case}")
                print(f"  Max difference: {np.abs(ground_truth - gparc_gt).max()}")
            
            model_predictions = {
                'PARCv2': parcv2_pred,
                'GPARC': gparc_pred
            }
        
        except FileNotFoundError as e:
            print(f"Skipping {parcv2_case}: {e}")
            continue
        
        # Determine timesteps
        simulation_length = ground_truth.shape[0]
        if args.timesteps:
            timesteps = [t for t in args.timesteps if t < simulation_length]
        else:
            timesteps = auto_select_timesteps(simulation_length, args.auto_timesteps)
        
        print(f"Selected timesteps: {timesteps}")
        
        # Create comparison for each feature
        for feature_idx in range(len(visualizer.var_names)):
            feature_name = visualizer.var_names[feature_idx]
            print(f"\nGenerating comparison for {feature_name}...")
            
            visualizer.create_single_feature_comparison(
                case_name=parcv2_case,  # Use PARCv2 name for output files
                feature_idx=feature_idx,
                timesteps=timesteps,
                ground_truth=ground_truth,
                model_predictions=model_predictions,
                model_names=model_names,
                figure_width=args.figure_width,
                dpi=args.dpi,
                output_format=args.output_format
            )
            
            if args.include_error_plots:
                visualizer.create_error_comparison(
                    case_name=parcv2_case,
                    feature_idx=feature_idx,
                    timesteps=timesteps,
                    ground_truth=ground_truth,
                    model_predictions=model_predictions,
                    model_names=model_names,
                    figure_width=args.figure_width,
                    dpi=args.dpi,
                    output_format=args.output_format
                )
        
        # Generate metrics table
        if args.include_metrics_table:
            visualizer.create_metrics_table(
                case_name=parcv2_case,
                ground_truth=ground_truth,
                model_predictions=model_predictions,
                model_names=model_names,
                timesteps=timesteps
            )
    
    print(f"\n{'='*60}")
    print(f"✓ All comparisons saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()