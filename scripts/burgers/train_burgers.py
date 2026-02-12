#!/usr/bin/env python3
"""
Training Script for G-PARC Burgers' Equation
============================================

Replicates the logic of the pixel-based PARC model on unstructured graphs.
- Static Features: [pos_x, pos_y, Reynolds]
- Dynamic Features: [u, v]
- Physics: Advection + Diffusion on (u, v)
- Integrator: Numerical (Euler/Heun/RK4)

Usage:
    python train_burgers.py --train_dir ... --val_dir ... --integrator euler
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Imports
from utilities.featureextractor import FeatureExtractorGNN
from utilities.trainer import train_and_validate_weighted, load_model, plot_loss_curves
from differentiator.fast_differential_operators import SolveGradientsLST, SolveWeightLST2d
from differentiator.burgers_differentiator import BurgersDifferentiator
from models.burgers import GPARC_Burgers_Numerical
from data.BurgersDataset import BurgersDataset

def create_burgers_model(args, sample_data):
    """
    Create the G-PARC Burgers model.
    """
    print("\nInitializing MLS operators...")
    # These solvers are shared by Advection and Diffusion modules
    gradient_solver = SolveGradientsLST(boundary_margin=0.1, precompute_mesh=sample_data)
    laplacian_solver = SolveWeightLST2d(boundary_margin=0.1, precompute_mesh=sample_data)
    
    # 1. Feature Extractor (UNet equivalent)
    # Inputs: [x, y, Re] -> 3 channels
    print(f"Feature Extractor Input Channels: {args.num_static_feats}")
    
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
    
    # 2. Differentiator (ADRD equivalent)
    # Calculates dU/dt using Advection/Diffusion operators + SPADE
    derivative_solver = BurgersDifferentiator(
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=args.feature_out_channels,
        spade_heads=args.spade_heads,
        spade_dropout=args.spade_dropout,
        zero_init=args.zero_init
    )
    
    # 3. Model Wrapper (Integrator)
    # Wraps the differentiator with a Numerical Integrator
    print(f"Initializing Integrator: {args.integrator.upper()}")
    
    model = GPARC_Burgers_Numerical(
        derivative_solver=derivative_solver,
        integrator_type=args.integrator, # 'euler', 'heun', or 'rk4'
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats
    )
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train G-PARC for Burgers Equation")
    
    # --- Data Arguments ---
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, default="*.pt")
    parser.add_argument("--output_dir", type=str, default="./outputs_burgers")
    parser.add_argument("--resume", type=str, default=None)
    
    # --- Architecture Inputs ---
    # Static = 3 (x, y, Re) -> Matches pixel input channels
    # Dynamic = 2 (u, v)    -> Matches pixel state vars
    parser.add_argument("--num_static_feats", type=int, default=3) 
    parser.add_argument("--num_dynamic_feats", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=1)
    
    # --- Feature Extractor (GraphUNet) ---
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--feature_out_channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--pool_ratios", type=float, default=0.5)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # --- Differentiator (SPADE) ---
    parser.add_argument("--spade_heads", type=int, default=2)
    parser.add_argument("--spade_dropout", type=float, default=0.1)
    parser.add_argument("--zero_init", action="store_true", default=True,
                        help="Initialize SPADE gamma/beta to zero (Stability)")

    # --- Integrator ---
    # Defaulting to Euler as requested, but Heun/RK4 are available
    parser.add_argument("--integrator", type=str, default="euler", 
                        choices=["euler", "heun", "rk4"])
    
    # --- Training ---
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Data Loading
    train_dataset = BurgersDataset(args.train_dir, file_pattern=args.file_pattern, seq_len=args.seq_len)
    val_dataset = BurgersDataset(args.val_dir, file_pattern=args.file_pattern, seq_len=args.seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=args.num_workers)
    
    print(f"Train size: ~{len(train_dataset)} sequences")
    
    # Init Model
    sample_sequence = next(iter(train_loader))
    sample_data = sample_sequence[0] # Get first frame
    
    model = create_burgers_model(args, sample_data).to(device)
    
    # Initialize weights using sample data (Important for MLS)
    model.derivative_solver.initialize_weights(sample_data.to(device))
    
    #if args.resume:
    #    model = load_model(model, args.resume, device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Try strict first (for new checkpoints)
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("✓ Model resumed from checkpoint (all keys matched)")
        except RuntimeError:
            # Fall back to non-strict for old checkpoints
            missing_keys = model.load_state_dict(
                checkpoint['model_state_dict'], 
                strict=False
            )
            print("✓ Model resumed from old checkpoint (static buffers recomputed)")
            
            if missing_keys.missing_keys:
                static_missing = [k for k in missing_keys.missing_keys if 'static_' in k]
                other_missing = [k for k in missing_keys.missing_keys if 'static_' not in k]
                
                if static_missing:
                    print(f"  ✓ Recomputed {len(static_missing)} static buffers from mesh")
                
                if other_missing:
                    print(f"  ⚠️ WARNING: Unexpected missing keys: {other_missing}")

        
    # Training
    best_model_path = output_dir / "burgers_best.pth"
    latest_model_path = output_dir / "burgers_latest.pth"
    
    # Component weighting: u and v usually have similar scales in Burgers
    # Setting to 1.0/1.0 implies equal importance.
    component_stds = torch.tensor([1.0, 1.0]) 
    
    # Use Float32 (use_amp=False) for physics stability
    train_losses, val_losses = train_and_validate_weighted(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        component_stds=component_stds,
        num_epochs=args.epochs,
        lr=args.lr,
        best_model_path=str(best_model_path),
        latest_model_path=str(latest_model_path),
        grad_clip_norm=args.grad_clip_norm,
        use_amp=False 
    )
    
    # Plot
    plot_path = output_dir / "burgers_loss_curve.png"
    plot_loss_curves(train_losses, val_losses, "Burgers Training Loss", str(plot_path))

if __name__ == "__main__":
    main()