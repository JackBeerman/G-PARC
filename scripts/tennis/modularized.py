#!/usr/bin/env python3
"""
HPC Training Script for GPARC Tennis Model
==============================================

This script trains the integrated GPARC model on tennis serve data
using the TennisRolloutDataset class. Optimized for HPC environments with
proper argument parsing, logging, and checkpointing.

Usage:
    python modularized.py --train_dir /path/to/train --val_dir /path/to/val --test_dir /path/to/test --epochs 50 --lr 1e-4
    
    # With multi-step context (recommended for pose prediction)
    python modularized.py --train_dir /path/to/train --val_dir /path/to/val --test_dir /path/to/test --epochs 50 --lr 1e-4 --num_context_steps 3
    
For SLURM submission:
    sbatch --gres=gpu:1 --time=24:00:00 python hpc_gparc_training.py
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Union, Dict, Any
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GraphUNet
from tqdm import tqdm
import matplotlib.pyplot as plt

debug_path = os.path.join(os.path.dirname(__file__), '..', '..')
print(f"Script location: {__file__}")
print(f"Adding to path: {os.path.abspath(debug_path)}")
print(f"Files in that directory: {os.listdir(debug_path) if os.path.exists(debug_path) else 'Directory not found'}")
sys.path.insert(0, debug_path)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.tennisDataset import TennisServeRolloutDataset, get_simulation_ids
# NEW: Import the refactored model components
from utilities.featureextractor import FeatureExtractorGNN
from utilities.embed import SimulationConditionedLayerNorm, GlobalParameterProcessor, GlobalModulatedGNN
from utilities.trainer import train_and_validate, train_and_validate_with_context, load_model, plot_loss_curves
from differentiator.differentiator import DerivativeGNN
from integrator.integrator import IntegralGNN
from models.tennisv2 import GPARC


def add_solver_arguments(parser):
    """Add arguments for configurable derivative and integral solvers."""
    
    # Derivative solver arguments
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
    
    # Integral solver arguments  
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
    
    return parser

# ==================== CORRECTED FUNCTION ====================
def create_model_with_configurable_solvers(args):
    """Create model with configurable solver architectures optimized for physics learning."""
    
    # The global embedding dimension from GlobalParameterProcessor is 64
    global_embed_dim = 64

    feature_extractor = FeatureExtractorGNN(
        in_channels=args.num_dynamic_feats, # PARCv2 setup 
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        depth=args.depth,
        pool_ratios=args.pool_ratios,
        heads=args.heads,
        concat=True,
        dropout=args.dropout
    )
    
    # Dynamically calculate the input channels for the DerivativeGNN
    # based on the concatenated features.
    deriv_in_channels = args.feature_out_channels + args.num_dynamic_feats + global_embed_dim
    
    derivative_solver = DerivativeGNN(
        in_channels=deriv_in_channels,
        hidden_channels=args.deriv_hidden_channels,
        out_channels=args.num_dynamic_feats,  # Match num_dynamic_feats
        num_layers=args.deriv_num_layers,
        heads=args.deriv_heads,
        concat=True,
        dropout=args.deriv_dropout,
        use_residual=args.deriv_use_residual
    )
    
    integral_solver = IntegralGNN(
        in_channels=args.num_dynamic_feats,  # Match derivative solver output
        hidden_channels=args.integral_hidden_channels,
        out_channels=args.num_dynamic_feats,  # Output the physics state update
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
    
    return model

# ============================================================


################################################################################
# MAIN FUNCTION
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Train GPARC model on tennis serve data")
    
    # Data arguments
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Directory containing training dataset files")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Directory containing validation dataset files") 
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test dataset files")
    parser.add_argument("--file_pattern", type=str, default="*.pt",
                        help="Pattern to match dataset files")
    parser.add_argument("--seq_len", type=int, default=1,
                        help="Sequence length for rollout windows")
    parser.add_argument("--num_static_feats", type=int, default=0,
                        help="Number of static features (e.g., positions)")
    parser.add_argument("--num_dynamic_feats", type=int, default=6,
                        help="Number of dynamic features to use (after excluding meaningless ones)")
    parser.add_argument("--skip_dynamic_indices", type=int, nargs='+', default=[],
                        help="Indices of dynamic features to skip (0-based, e.g., --skip_dynamic_indices 2)")
    
    # Feature extractor model arguments
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
    
    # Add solver-specific arguments
    parser = add_solver_arguments(parser)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_context_steps", type=int, default=0,
                        help="Number of initial context steps (0 = single-step, 3+ = multi-step context)")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data loading workers")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose training progress")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "training.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log arguments
    logger.info("Training Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Device setup
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f} GB")
    
    # Data setup
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    test_dir = Path(args.test_dir)
    
    logger.info(f"Loading data from:")
    logger.info(f"  Train: {train_dir}")
    logger.info(f"  Val: {val_dir}")
    logger.info(f"  Test: {test_dir}")
    
    # Get simulation IDs for each split
    train_ids = get_simulation_ids(train_dir, pattern=args.file_pattern)
    val_ids = get_simulation_ids(val_dir, pattern=args.file_pattern)
    test_ids = get_simulation_ids(test_dir, pattern=args.file_pattern)
    
    logger.info(f"Found simulation files:")
    logger.info(f"  Train: {len(train_ids)} files")
    logger.info(f"  Val: {len(val_ids)} files")
    logger.info(f"  Test: {len(test_ids)} files")
    
    if len(train_ids) == 0:
        logger.error(f"No training files found in {train_dir} matching pattern: {args.file_pattern}")
        sys.exit(1)
    if len(val_ids) == 0:
        logger.error(f"No validation files found in {val_dir} matching pattern: {args.file_pattern}")
        sys.exit(1)
    
    
    # Create datasets for each split
    train_dataset = TennisServeRolloutDataset(
        directory=train_dir,
        simulation_ids=train_ids,
        seq_len=args.seq_len,
        stride=1,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats
    )
    
    val_dataset = TennisServeRolloutDataset(
        directory=val_dir,
        simulation_ids=val_ids,
        seq_len=args.seq_len,
        stride=1,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats
    )
    
    test_dataset = TennisServeRolloutDataset(
        directory=test_dir,
        simulation_ids=test_ids,
        seq_len=args.seq_len,
        stride=1,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    logger.info(f"Dataset created - sequence length: {args.seq_len}")
    
    # Model setup with new configurable architecture
    logger.info("Initializing model...")
    
    # ==================== CORRECTED CALL ====================
    model = create_model_with_configurable_solvers(args).to(device)
    # ========================================================
    
    # Log model configuration
    logger.info("Model Configuration:")
    logger.info(f"  Feature Extractor: {args.depth} layers, {args.hidden_channels} hidden, {args.feature_out_channels} out")
    logger.info(f"  Derivative Solver: {args.deriv_num_layers} layers, {args.deriv_hidden_channels} hidden, {args.deriv_heads} heads")
    logger.info(f"  Integral Solver: {args.integral_num_layers} layers, {args.integral_hidden_channels} hidden, {args.integral_heads} heads")
    logger.info(f"  Skipping dynamic indices: {args.skip_dynamic_indices}")
    logger.info(f"  Using {args.num_dynamic_feats} dynamic features after skipping")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = load_model(model, args.resume, device)
    
    # Save model architecture
    with open(output_dir / "model_summary.txt", "w") as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
        f.write(f"\n\nModel Configuration:")
        f.write(f"\n  Feature Extractor: {args.depth} layers, {args.hidden_channels} hidden, {args.feature_out_channels} out")
        f.write(f"\n  Derivative Solver: {args.deriv_num_layers} layers, {args.deriv_hidden_channels} hidden, {args.deriv_heads} heads")
        f.write(f"\n  Integral Solver: {args.integral_num_layers} layers, {args.integral_hidden_channels} hidden, {args.integral_heads} heads")
        f.write(f"\n  Skipping dynamic indices: {args.skip_dynamic_indices}")
        f.write(f"\n  Using {args.num_dynamic_feats} dynamic features after skipping")
    
    # Training - choose method based on num_context_steps
    logger.info("Starting training...")
    best_model_path = output_dir / "tennis_serve_best_model.pth"
    
    if args.num_context_steps > 0:
        # Use multi-step context training (recommended for pose prediction)
        logger.info(f"Using multi-step context training with {args.num_context_steps} initial timesteps")
        train_losses, val_losses = train_and_validate_with_context(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            best_model_path=str(best_model_path),
            num_context_steps=args.num_context_steps
        )
    else:
        # Use single-step training (original method)
        logger.info("Using single-step training (give t=0, predict full sequence)")
        train_losses, val_losses = train_and_validate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            best_model_path=str(best_model_path)
        )
    
    # Plot results using your original function
    plot_path = output_dir / "tennis_serve_loss_curves.png"
    plot_loss_curves(
        train_losses, 
        val_losses, 
        title="Tennis Serve Training & Validation Loss",
        save_path=str(plot_path)
    )
    logger.info(f"Training curves saved to: {plot_path}")
    
    # Save final results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': min(val_losses),
        'num_context_steps': args.num_context_steps,
        'args': args
    }
    
    with open(output_dir / "training_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    logger.info("Training completed successfully!")
    logger.info(f"Best validation loss: {min(val_losses):.6f}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()