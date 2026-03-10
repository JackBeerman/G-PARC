#!/usr/bin/env python3
"""
Training Script for G-PARCv2 Cylinder Flow (NoSPADE)
=====================================================
MLS advection-diffusion + concat fusion + numerical integration.
FiLM conditioning on Reynolds number.

Designed for 60k-100k node meshes with VRAM efficiency:
  - GraphConvFeatureExtractorV2 (O(N), no attention)
  - Concat fusion MLPs (not SPADE/GAT in differentiator)
  - Numerical integrator (no learnable parameters)
  - Gradient checkpointing option

Usage:
  python train_cylinder_v2.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --epochs 100 \
    --seq_len 5 \
    --skip_dynamic_indices 3 4 5
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import time
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.featureextractor import GraphConvFeatureExtractorV2
from differentiator.cylinder_nospade import CylinderDifferentiator
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from data.Cylinder import StreamingKarmanDataset, get_simulation_ids
from models.cylinder_gparcv2 import GPARC_Cylinder_V2


# =========================================================================
# SCHEDULED SAMPLING
# =========================================================================

def get_teacher_forcing_ratio(epoch, total_epochs, schedule='linear',
                               initial_ratio=1.0, final_ratio=0.0):
    """Compute teacher forcing ratio for current epoch."""
    if schedule == 'linear':
        ratio = initial_ratio - (initial_ratio - final_ratio) * (epoch / total_epochs)
    elif schedule == 'exponential':
        if initial_ratio > 0:
            decay = (final_ratio / max(initial_ratio, 1e-8)) ** (1 / max(total_epochs, 1))
            ratio = initial_ratio * (decay ** epoch)
        else:
            ratio = 0.0
    elif schedule == 'sigmoid':
        x = (epoch - total_epochs / 2) / (total_epochs / 10)
        sigmoid = 1 / (1 + torch.exp(torch.tensor(x)).item())
        ratio = final_ratio + (initial_ratio - final_ratio) * sigmoid
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return max(final_ratio, min(initial_ratio, ratio))


# =========================================================================
# MODEL CREATION
# =========================================================================

def create_model(args, sample_data):
    """Create G-PARCv2 cylinder model with CylinderDifferentiator."""
    
    # Effective num_dynamic after skipping
    num_dynamic_effective = args.num_dynamic_feats
    
    # MLS operators — positions are normalized, need denorm for physical gradients
    # For cylinder: positions may or may not be normalized
    # We use default (no denorm) and let MLS work in normalized space
    print("\nInitializing MLS Operators...")
    gradient_solver = SolveGradientsLST(
        pos_mean=[0.0, 0.0],
        pos_std=[1.0, 1.0],
        norm_method='z_score',
    )
    laplacian_solver = SolveWeightLST2d(
        pos_mean=[0.0, 0.0],
        pos_std=[1.0, 1.0],
        norm_method='z_score',
        min_neighbors=5,
    )
    
    # GraphConv Feature Extractor (VRAM efficient — no attention heads)
    fe_in_channels = args.num_static_feats + num_dynamic_effective
    print(f"\nCreating GraphConv Feature Extractor...")
    print(f"  Input channels: {fe_in_channels} ({args.num_static_feats} static + {num_dynamic_effective} dynamic)")
    print(f"  Layers: {args.num_layers}")
    print(f"  Hidden: {args.hidden_channels}")
    print(f"  Output: {args.feature_out_channels}")
    
    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=fe_in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layer_norm=True,
        use_relative_pos=True,
    )
    
    # Velocity indices in POST-SKIP dynamic space
    # Default raw: [0]=pressure, [1]=vx, [2]=vy, [3]=vz, [4]=ωx, [5]=ωy, [6]=ωz
    # If skip=[3,4,5]: post-skip [0]=p, [1]=vx, [2]=vy, [3]=ωz → velocity at [1,2]
    velocity_indices = args.velocity_indices
    print(f"  Velocity indices (post-skip): {velocity_indices}")
    
    # CylinderDifferentiator (NoSPADE concat fusion)
    derivative_solver = CylinderDifferentiator(
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=num_dynamic_effective,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=args.feature_out_channels,
        global_embed_dim=args.global_embed_dim,
        global_param_dim=args.global_param_dim,
        velocity_indices=velocity_indices,
        diffusion_type=args.diffusion_type,
        fusion_hidden_dim=args.fusion_hidden_dim,
        zero_init=args.zero_init,
        pos_dims=2,  # Use x,y from 3D positions for MLS
    )
    
    # Initialize MLS weights
    print("Initializing MLS weights...")
    # Ensure sample_data has pos
    if not hasattr(sample_data, 'pos') or sample_data.pos is None:
        sample_data.pos = sample_data.x[:, :args.num_static_feats]
    derivative_solver.initialize_weights(sample_data)
    
    # Full model
    model = GPARC_Cylinder_V2(
        derivative_solver_physics=derivative_solver,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=num_dynamic_effective,
        skip_dynamic_indices=args.skip_dynamic_indices,
        global_param_dim=args.global_param_dim,
        global_embed_dim=args.global_embed_dim,
        clamp_output=not args.no_clamp_output,
        clamp_max=args.clamp_max,
    )
    
    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'~'*50}")
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"{'~'*50}\n")
    return trainable


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, path)


# =========================================================================
# TRAINING LOOP
# =========================================================================

def train_epoch(model, train_loader, optimizer, device, epoch, total_epochs,
                args):
    """Single training epoch with scheduled sampling."""
    model.train()
    total_loss = 0.0
    n_sequences = 0
    
    # Compute teacher forcing ratio
    tf_ratio = get_teacher_forcing_ratio(
        epoch, total_epochs,
        schedule=args.ss_schedule,
        initial_ratio=args.ss_initial_ratio,
        final_ratio=args.ss_final_ratio,
    )
    
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
    
    for seq_idx, sequence in enumerate(pbar):
        if not isinstance(sequence, list) or len(sequence) < 2:
            continue
        
        # Move to device and stamp mesh_id for MLS cache
        # Each simulation has a unique mesh — use num_nodes as identifier
        mesh_id = sequence[0].num_nodes  # unique per mesh topology
        for data in sequence:
            data.x = data.x.to(device)
            data.y = data.y.to(device)
            data.edge_index = data.edge_index.to(device)
            if hasattr(data, 'pos') and data.pos is not None:
                data.pos = data.pos.to(device)
            # Stamp mesh_id for MLS operator cache invalidation
            data.mesh_id = torch.tensor([mesh_id], device=device)
        
        # Initialize MLS on first sequence
        if seq_idx == 0:
            deriv = model.derivative_solver
            if hasattr(deriv, '_weights_initialized') and not deriv._weights_initialized:
                sample = sequence[0]
                if not hasattr(sample, 'pos') or sample.pos is None:
                    sample.pos = sample.x[:, :model.num_static_feats]
                deriv.initialize_weights(sample)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(sequence, teacher_forcing_ratio=tf_ratio)
        
        # Compute loss over all timesteps
        loss = 0.0
        n_steps = 0
        for t, pred in enumerate(predictions):
            target = model.process_targets(sequence[t].y)
            step_loss = F.mse_loss(pred, target)
            
            # Optional exponential decay weighting
            if args.use_loss_decay:
                weight = args.loss_decay_gamma ** t
                step_loss = step_loss * weight
            
            loss += step_loss
            n_steps += 1
        
        loss = loss / max(n_steps, 1)
        
        # Backward
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        n_sequences += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'avg': f'{total_loss/n_sequences:.6f}',
            'tf': f'{tf_ratio:.3f}',
        })
        
        # Periodic memory cleanup for large meshes
        if seq_idx % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    avg_loss = total_loss / max(n_sequences, 1)
    return {
        'loss': avg_loss,
        'teacher_forcing_ratio': tf_ratio,
        'n_sequences': n_sequences,
    }


def validate_epoch(model, val_loader, device, args):
    """Validation epoch (free-running, no teacher forcing)."""
    model.eval()
    total_loss = 0.0
    n_sequences = 0
    
    with torch.no_grad():
        for sequence in tqdm(val_loader, desc="Validation"):
            if not isinstance(sequence, list) or len(sequence) < 2:
                continue
            
            mesh_id = sequence[0].num_nodes
            for data in sequence:
                data.x = data.x.to(device)
                data.y = data.y.to(device)
                data.edge_index = data.edge_index.to(device)
                if hasattr(data, 'pos') and data.pos is not None:
                    data.pos = data.pos.to(device)
                data.mesh_id = torch.tensor([mesh_id], device=device)
            
            # Free-running (teacher_forcing_ratio=0.0)
            predictions = model(sequence, teacher_forcing_ratio=0.0)
            
            loss = 0.0
            n_steps = 0
            for t, pred in enumerate(predictions):
                target = model.process_targets(sequence[t].y)
                loss += F.mse_loss(pred, target)
                n_steps += 1
            
            loss = loss / max(n_steps, 1)
            total_loss += loss.item()
            n_sequences += 1
    
    avg_loss = total_loss / max(n_sequences, 1)
    return {'loss': avg_loss}


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train G-PARCv2 Cylinder Flow (NoSPADE)"
    )
    
    # Data
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, default="*.pt")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--num_static_feats", type=int, default=3,
                        help="Number of static features (x, y, z positions)")
    parser.add_argument("--num_dynamic_feats", type=int, default=7,
                        help="Number of dynamic features AFTER skipping")
    parser.add_argument("--skip_dynamic_indices", type=int, nargs='+', default=[],
                        help="Raw dynamic indices to skip (e.g., 3 4 5 for vz, ωx, ωy)")
    parser.add_argument("--velocity_indices", type=int, nargs='+', default=[1, 2],
                        help="Post-skip indices of velocity [vx, vy] for advection")
    parser.add_argument("--max_timesteps", type=int, default=None,
                        help="Max timesteps per simulation (for VRAM control)")
    parser.add_argument("--temporal_subsample", type=int, default=None,
                        help="Use every Nth timestep (reduces data volume)")
    
    # Feature extractor
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_channels", type=int, default=64,
                        help="Hidden channels (keep small for 100k node meshes)")
    parser.add_argument("--feature_out_channels", type=int, default=64,
                        help="Feature extractor output dim (keep small for VRAM)")
    parser.add_argument("--dropout", type=float, default=0.0)
    
    # Differentiator
    parser.add_argument("--diffusion_type", type=str, default='fd',
                        choices=['fd', 'mls', 'none'])
    parser.add_argument("--fusion_hidden_dim", type=int, default=64,
                        help="Hidden dim for per-variable fusion MLPs")
    parser.add_argument("--zero_init", action="store_true", default=True)
    
    # Global conditioning
    parser.add_argument("--global_param_dim", type=int, default=1,
                        help="Dimension of global params (1 = Reynolds only)")
    parser.add_argument("--global_embed_dim", type=int, default=64)
    
    # Model
    parser.add_argument("--integrator", type=str, default="euler",
                        choices=["euler", "heun", "rk4", "implicit"])
    parser.add_argument("--no_clamp_output", action="store_true", default=False)
    parser.add_argument("--clamp_max", type=float, default=10.0)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Scheduled sampling
    parser.add_argument("--ss_schedule", type=str, default="linear",
                        choices=["linear", "exponential", "sigmoid"])
    parser.add_argument("--ss_initial_ratio", type=float, default=1.0)
    parser.add_argument("--ss_final_ratio", type=float, default=0.0)
    
    # Loss
    parser.add_argument("--use_loss_decay", action="store_true", default=False)
    parser.add_argument("--loss_decay_gamma", type=float, default=0.95)
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./outputs_cylinder_v2")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset_best", action="store_true", default=False)
    parser.add_argument("--fresh_scheduler", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Auto-compute num_dynamic_feats if skip indices provided
    raw_dynamic = 7  # pressure + 3 velocity + 3 vorticity
    if args.skip_dynamic_indices:
        args.num_dynamic_feats = raw_dynamic - len(args.skip_dynamic_indices)
        print(f"Auto-computed num_dynamic_feats = {args.num_dynamic_feats} "
              f"(7 raw - {len(args.skip_dynamic_indices)} skipped)")
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("G-PARCv2 CYLINDER FLOW (NoSPADE) TRAINING")
    print("="*70)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Output: {output_dir}")
    print(f"")
    print(f"Data:")
    print(f"  Static feats: {args.num_static_feats} (x,y,z)")
    print(f"  Dynamic feats: {args.num_dynamic_feats} (after skipping {args.skip_dynamic_indices})")
    print(f"  Velocity indices: {args.velocity_indices}")
    print(f"  Seq length: {args.seq_len}, Stride: {args.stride}")
    print(f"")
    print(f"Architecture:")
    print(f"  Feature Extractor: GraphConv V2, {args.num_layers} layers, "
          f"{args.hidden_channels} hidden, {args.feature_out_channels} out")
    print(f"  Differentiator: CylinderDifferentiator (NoSPADE)")
    print(f"  Diffusion: {args.diffusion_type}")
    print(f"  Fusion MLP hidden: {args.fusion_hidden_dim}")
    print(f"  Integrator: {args.integrator}")
    print(f"  Global: Reynolds → embed({args.global_embed_dim})")
    print(f"")
    print(f"Training:")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Scheduled Sampling: {args.ss_schedule} ({args.ss_initial_ratio} → {args.ss_final_ratio})")
    print("="*70)
    
    # Load dataset
    train_ids = get_simulation_ids(Path(args.train_dir), pattern=args.file_pattern)
    val_ids = get_simulation_ids(Path(args.val_dir), pattern=args.file_pattern)
    
    print(f"\nFound {len(train_ids)} training simulations")
    print(f"Found {len(val_ids)} validation simulations")
    
    if len(train_ids) == 0:
        print(f"ERROR: No training files in {args.train_dir}")
        sys.exit(1)
    
    # Dataset kwargs for memory efficiency
    ds_kwargs = dict(
        seq_len=args.seq_len,
        stride=args.stride,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=raw_dynamic,  # Raw count for dataset
        shuffle_simulations=True,
    )
    
    if args.max_timesteps:
        ds_kwargs['max_timesteps_per_sim'] = args.max_timesteps
    if args.temporal_subsample:
        ds_kwargs['timestep_sampling'] = 'every_nth'
        ds_kwargs['temporal_subsample_rate'] = args.temporal_subsample
    
    train_dataset = StreamingKarmanDataset(
        directory=Path(args.train_dir),
        simulation_ids=train_ids,
        **ds_kwargs,
    )
    val_dataset = StreamingKarmanDataset(
        directory=Path(args.val_dir),
        simulation_ids=val_ids,
        **{**ds_kwargs, 'shuffle_simulations': False},
    )
    
    loader_kwargs = {
        'batch_size': None,
        'num_workers': args.num_workers,
        'pin_memory': True if device.type == 'cuda' else False,
    }
    
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    
    # Get sample data for initialization
    print("\nGetting sample for initialization...")
    sample_data = next(iter(train_loader))[0]
    print(f"  Nodes: {sample_data.num_nodes:,}")
    print(f"  Edges: {sample_data.edge_index.shape[1]:,}")
    print(f"  x shape: {sample_data.x.shape}")
    print(f"  y shape: {sample_data.y.shape}")
    if hasattr(sample_data, 'global_params'):
        print(f"  global_params: {sample_data.global_params}")
    
    # VRAM estimate
    nodes = sample_data.num_nodes
    edges = sample_data.edge_index.shape[1]
    approx_mb = (nodes * (args.feature_out_channels + args.num_dynamic_feats) * 4 * 3) / 1e6
    print(f"\n  Approx feature VRAM per step: ~{approx_mb:.0f} MB")
    print(f"  Sequence of {args.seq_len} steps: ~{approx_mb * args.seq_len:.0f} MB")
    
    # Create model
    print("\nCreating model...")
    model = create_model(args, sample_data).to(device)
    count_parameters(model)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if args.fresh_scheduler:
            start_epoch = 0
            print(f"  Fresh optimizer + scheduler (lr={args.lr})")
            optimizer = AdamW(model.parameters(), lr=args.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        else:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if args.reset_best:
            best_val_loss = float('inf')
            print(f"  Reset best_val_loss")
        else:
            best_val_loss = checkpoint.get('metrics', {}).get('val_loss', float('inf'))
        
        print(f"  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.8f}")
    
    # Save config
    config = vars(args)
    config['feature_extractor'] = 'GraphConv V2'
    config['differentiator'] = 'CylinderDifferentiator (NoSPADE)'
    config['sample_nodes'] = int(sample_data.num_nodes)
    config['sample_edges'] = int(sample_data.edge_index.shape[1])
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save model summary
    with open(output_dir / "model_summary.txt", 'w') as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'teacher_forcing_ratio': [],
        'epoch_time': [],
    }
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch,
            total_epochs=args.epochs,
            args=args,
        )
        
        val_metrics = validate_epoch(model, val_loader, device, args)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nTrain Loss: {train_metrics['loss']:.6f} "
              f"(TF ratio: {train_metrics['teacher_forcing_ratio']:.3f}, "
              f"{train_metrics['n_sequences']} sequences)")
        print(f"Val Loss:   {val_metrics['loss']:.6f} (Free running)")
        print(f"Epoch time: {epoch_time:.1f}s")
        print(f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['teacher_forcing_ratio'].append(train_metrics['teacher_forcing_ratio'])
        history['epoch_time'].append(epoch_time)
        
        # Checkpointing
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': best_val_loss,
                 'teacher_forcing_ratio': train_metrics['teacher_forcing_ratio']},
                output_dir / "best_model.pth"
            )
            print(f"✓ Saved best model (val_loss: {best_val_loss:.6f})")
        
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'val_loss': val_metrics['loss'],
             'teacher_forcing_ratio': train_metrics['teacher_forcing_ratio']},
            output_dir / "latest_model.pth"
        )
        
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        # Memory cleanup between epochs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best model: {output_dir / 'best_model.pth'}")
    print(f"Latest model: {output_dir / 'latest_model.pth'}")
    print("="*70)


if __name__ == "__main__":
    main()