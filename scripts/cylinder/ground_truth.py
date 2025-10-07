#!/usr/bin/env python3
"""
Visualize Ground Truth Cylinder Flow Data
==========================================

Standalone script to visualize the 4 key variables from cylinder flow simulations:
- Pressure
- X-velocity
- Y-velocity
- Z-vorticity

Usage:
    python visualize_ground_truth.py --file /path/to/simulation.pt --output_dir ./viz
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_normalization_params(metadata_file):
    """Load denormalization parameters."""
    if not Path(metadata_file).exists():
        print(f"Warning: Metadata file not found: {metadata_file}")
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    denorm_params = {}
    norm_params = metadata.get('normalization_params', {})
    
    # Variable parameters
    for var in ['pressure', 'x_velocity', 'y_velocity', 'z_vorticity']:
        if var in norm_params:
            denorm_params[var] = norm_params[var]
    
    # Reynolds parameters
    if 'global_param_normalization' in metadata:
        if 'reynolds' in metadata['global_param_normalization']:
            denorm_params['reynolds'] = metadata['global_param_normalization']['reynolds']
    elif 'reynolds' in norm_params:
        denorm_params['reynolds'] = norm_params['reynolds']
    
    return denorm_params


def denormalize_value(normalized_val, param_name, denorm_params):
    """Denormalize a value."""
    if denorm_params is None or param_name not in denorm_params:
        return normalized_val
    
    params = denorm_params[param_name]
    return normalized_val * (params['max'] - params['min']) + params['min']


def extract_data(simulation, num_static_feats=2, skip_dynamic_indices=[]):
    """Extract relevant data from simulation."""
    # Get positions from first timestep (static features)
    first_step = simulation[0]
    positions = first_step.x[:, :num_static_feats].cpu().numpy()
    
    # Extract Reynolds number
    if hasattr(first_step, 'global_params'):
        reynolds_norm = float(first_step.global_params[0])
    elif hasattr(first_step, 'reynolds'):
        reynolds_norm = float(first_step.reynolds[0] if first_step.reynolds.dim() > 0 else first_step.reynolds)
    else:
        reynolds_norm = None
    
    # FIXED: These are the indices AFTER skipping has been applied
    # The y attribute should already have skipped indices removed
    var_indices = {
        'pressure': 0,
        'x_velocity': 1,
        'y_velocity': 2,
        'z_vorticity': 3  # This is the 4th variable after skipping 3,4,5
    }
    
    # Extract data for each timestep
    timesteps = []
    for step_data in simulation:
        # Get target values (y attribute contains the dynamic features AFTER skipping)
        if hasattr(step_data, 'y'):
            step_values = step_data.y.cpu().numpy()
        else:
            # Fallback: extract from x and apply skipping manually
            all_dynamic = step_data.x[:, num_static_feats:].cpu().numpy()
            keep_indices = [i for i in range(all_dynamic.shape[1]) if i not in skip_dynamic_indices]
            step_values = all_dynamic[:, keep_indices]
        
        timesteps.append(step_values)
    
    return positions, timesteps, var_indices, reynolds_norm


def create_static_plot(positions, timesteps, var_indices, denorm_params, 
                       timestep_idx=0, figsize=(16, 10)):
    """Create static plot of all 4 variables at a specific timestep."""
    var_names = ['pressure', 'x_velocity', 'y_velocity', 'z_vorticity']
    var_titles = ['Pressure', 'X-Velocity', 'Y-Velocity', 'Z-Vorticity']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    step_data = timesteps[timestep_idx]
    
    for i, (var_name, var_title) in enumerate(zip(var_names, var_titles)):
        ax = axes[i]
        
        var_idx = var_indices[var_name]
        data = step_data[:, var_idx]
        
        # Denormalize
        if denorm_params:
            data = denormalize_value(data, var_name, denorm_params)
        
        # Determine color limits
        vmin, vmax = np.percentile(data, [2, 98])
        
        # Scatter plot
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=data,
                           cmap='viridis', s=1.0, vmin=vmin, vmax=vmax, alpha=0.8)
        
        ax.set_title(f'{var_title} (Timestep {timestep_idx})', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    return fig


def create_animation(positions, timesteps, var_indices, denorm_params, 
                     output_path, fps=2):
    """Create animated GIF showing evolution of all 4 variables."""
    var_names = ['pressure', 'x_velocity', 'y_velocity', 'z_vorticity']
    var_titles = ['Pressure', 'X-Velocity', 'Y-Velocity', 'Z-Vorticity']
    
    num_steps = len(timesteps)
    
    # Calculate global color limits for each variable
    var_ranges = {}
    for var_name in var_names:
        var_idx = var_indices[var_name]
        all_data = np.concatenate([step[:, var_idx] for step in timesteps])
        if denorm_params:
            all_data = denormalize_value(all_data, var_name, denorm_params)
        var_ranges[var_name] = np.percentile(all_data, [2, 98])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    scatters = []
    
    # Initialize plots
    for i, (var_name, var_title) in enumerate(zip(var_names, var_titles)):
        ax = axes[i]
        var_idx = var_indices[var_name]
        
        data = timesteps[0][:, var_idx]
        if denorm_params:
            data = denormalize_value(data, var_name, denorm_params)
        
        vmin, vmax = var_ranges[var_name]
        
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=data,
                           cmap='viridis', s=1.0, vmin=vmin, vmax=vmax, alpha=0.8)
        
        ax.set_title(f'{var_title}', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, fraction=0.046)
        
        scatters.append(scatter)
    
    plt.tight_layout()
    
    def animate(frame):
        for i, var_name in enumerate(var_names):
            var_idx = var_indices[var_name]
            data = timesteps[frame][:, var_idx]
            if denorm_params:
                data = denormalize_value(data, var_name, denorm_params)
            scatters[i].set_array(data)
        
        fig.suptitle(f'Timestep {frame}/{num_steps-1}', fontsize=14, y=0.995)
        return scatters
    
    anim = FuncAnimation(fig, animate, frames=num_steps, interval=1000//fps, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved animation: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize ground truth cylinder flow data'
    )
    parser.add_argument('--file', required=True, help='Path to simulation .pt file')
    parser.add_argument('--output_dir', default='./ground_truth_viz', 
                       help='Output directory for visualizations')
    parser.add_argument('--num_static_feats', type=int, default=2,
                       help='Number of static features (position dimensions)')
    parser.add_argument('--skip_dynamic_indices', type=int, nargs='+', default=[],
                       help='Indices of dynamic features that were skipped')
    parser.add_argument('--timestep', type=int, default=0,
                       help='Timestep to visualize in static plot')
    parser.add_argument('--fps', type=int, default=2,
                       help='Frames per second for animation')
    parser.add_argument('--no_animation', action='store_true',
                       help='Skip creating animation (only create static plot)')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    sim_file = Path(args.file)
    if not sim_file.exists():
        raise FileNotFoundError(f"Simulation file not found: {sim_file}")
    
    print(f"Loading simulation: {sim_file.name}")
    
    # Load simulation
    simulation = torch.load(sim_file, weights_only=False)
    print(f"  Timesteps: {len(simulation)}")
    print(f"  Nodes: {simulation[0].x.shape[0]}")
    
    # Load denormalization parameters
    metadata_file = sim_file.parent / 'normalization_metadata.json'
    if not metadata_file.exists():
        metadata_file = sim_file.parent.parent / 'normalization_metadata.json'
    
    denorm_params = load_normalization_params(metadata_file)
    if denorm_params:
        print(f"  Loaded denormalization parameters")
    else:
        print(f"  Warning: Using normalized values (no metadata found)")
    
    # Extract data
    positions, timesteps, var_indices, reynolds_norm = extract_data(
        simulation, 
        args.num_static_feats,
        args.skip_dynamic_indices
    )
    
    # Get Reynolds number
    if reynolds_norm is not None and denorm_params:
        reynolds = denormalize_value(reynolds_norm, 'reynolds', denorm_params)
    else:
        reynolds = reynolds_norm if reynolds_norm else 'unknown'
    
    print(f"  Reynolds number: {reynolds}")
    print(f"  Position range: X=[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}], "
          f"Y=[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    
    # Create static plot
    print(f"\nCreating static plot for timestep {args.timestep}...")
    fig = create_static_plot(positions, timesteps, var_indices, denorm_params,
                            timestep_idx=args.timestep)
    
    static_path = output_dir / f'{sim_file.stem}_timestep_{args.timestep}.png'
    fig.savefig(static_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {static_path}")
    
    # Create animation
    if not args.no_animation:
        print(f"\nCreating animation ({len(timesteps)} timesteps)...")
        anim_path = output_dir / f'{sim_file.stem}_animation.gif'
        create_animation(positions, timesteps, var_indices, denorm_params,
                        anim_path, fps=args.fps)
    
    print(f"\nVisualization complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()