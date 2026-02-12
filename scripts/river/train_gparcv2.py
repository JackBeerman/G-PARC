#!/usr/bin/env python3
"""
Training Script for G-PARCv2 River Model
========================================
Aligned with Elastoplastic training standards:
- AdamW + CosineAnnealingLR
- Scheduled Sampling (Teacher Forcing decay)
- JSON Logging & Config
- TQDM Progress Bars

Usage:
    python train_river_v2.py --train_dir /path/to/train --val_dir /path/to/val --output_dir /path/to/out
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.featureextractor import GraphConvFeatureExtractorV2
from differentiator.riverdifferentiator import RiverDifferentiator
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from models.riverV2 import GPARC_River_V2
from data.Riverdataset import RiverDataset


def load_normalization_stats(data_dir):
    """Load normalization statistics from the data directory."""
    stats_file = Path(data_dir).parent / "normalization_stats.json"
    
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"\n✓ Loaded normalization stats from: {stats_file}")
        return stats
    else:
        print(f"\n⚠️  No normalization_stats.json found at {stats_file}")
        return {'normalization_method': 'unknown'}


def count_parameters(model):
    """Counts the total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{' MODEL PARAMETERS ':~^50}")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"{'~'*50}\n")
    return trainable_params


# =========================================================================
# SCHEDULED SAMPLING UTILITIES
# =========================================================================

def get_teacher_forcing_ratio(epoch, total_epochs, schedule='linear', initial_ratio=1.0, final_ratio=0.0):
    """Compute teacher forcing ratio for current epoch."""
    if schedule == 'linear':
        ratio = initial_ratio - (initial_ratio - final_ratio) * (epoch / total_epochs)
    elif schedule == 'exponential':
        if initial_ratio > 0:
            decay = (final_ratio / initial_ratio) ** (1 / total_epochs)
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

def create_model(args, sample_data, norm_stats):
    """Create G-PARCv2 River model."""
    
    print("\nInitializing MLS Operators...")
    gradient_solver = SolveGradientsLST()
    laplacian_solver = SolveWeightLST2d(use_2hop_extension=False)
    
    # GraphConv Feature Extractor
    print(f"\nCreating GraphConv Feature Extractor...")
    print(f"  Layers: {args.num_layers}")
    print(f"  Hidden: {args.hidden_channels}")
    print(f"  Output: {args.feature_out_channels}")
    
    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=args.num_static_feats,
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layer_norm=args.use_layer_norm,
        use_relative_pos=args.use_relative_pos
    )
    
    print(f"\nPhysics feature configuration:")
    print(f"  Advection indices: {list(range(args.num_dynamic_feats))}")
    print(f"  Velocity indices: {args.velocity_indices}")
    print(f"  Integrator: {args.integrator}")

    derivative_solver = RiverDifferentiator(
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=args.feature_out_channels,
        list_adv_idx=list(range(args.num_dynamic_feats)),
        list_dif_idx=list(range(args.num_dynamic_feats)),
        velocity_indices=args.velocity_indices,
        spade_random_noise=args.spade_random_noise,
        heads=args.spade_heads,
        concat=args.spade_concat,
        dropout=args.spade_dropout,
        zero_init=args.zero_init
    )
    
    print("Initializing MLS weights...")
    derivative_solver.initialize_weights(sample_data)
    
    model = GPARC_River_V2(
        derivative_solver_physics=derivative_solver,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
    )
    
    return model


# =========================================================================
# TRAINING WITH SCHEDULED SAMPLING
# =========================================================================

def train_epoch(model, train_loader, optimizer, device, epoch, total_epochs, args):
    """Train for one epoch with scheduled sampling."""
    model.train()
    
    teacher_forcing_ratio = get_teacher_forcing_ratio(
        epoch=epoch,
        total_epochs=total_epochs,
        schedule=args.ss_schedule,
        initial_ratio=args.ss_initial_ratio,
        final_ratio=args.ss_final_ratio
    )
    
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Training (TF={teacher_forcing_ratio:.3f})")
    
    for sequence in pbar:
        # Handle list wrapping if necessary (common in some PyG Loaders)
        if isinstance(sequence, list) and len(sequence) > 0 and isinstance(sequence[0], list):
            sequence = sequence[0]
            
        # Move to device
        sequence = [d.to(device) for d in sequence]
        
        # Ensure pos exists (River data sometimes lacks it in the batch)
        for data in sequence:
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :2] # Assuming first 2 static are x,y

        optimizer.zero_grad()
        
        # Forward pass with Scheduled Sampling
        predictions = model(sequence, dt=1.0, teacher_forcing_ratio=teacher_forcing_ratio)
        
        loss = 0.0
        
        # Calculate loss over the sequence
        # Note: sequence[t].y contains the Ground Truth for step t+1
        for t, pred in enumerate(predictions):
            target = sequence[t].y[:, :args.num_dynamic_feats]
            step_loss = F.mse_loss(pred, target)
            loss += step_loss
            
        loss = loss / len(predictions)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠️ Skipping batch: NaN/Inf loss")
            continue
            
        loss.backward()
        
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.6f}"})
    
    return {
        'loss': total_loss / max(n_batches, 1),
        'teacher_forcing_ratio': teacher_forcing_ratio
    }


@torch.no_grad()
def validate_epoch(model, val_loader, device, args):
    """Validate for one epoch (Always pure autoregressive)."""
    model.eval()
    
    total_loss = 0.0
    n_batches = 0
    
    for sequence in tqdm(val_loader, desc="Validating"):
        if isinstance(sequence, list) and len(sequence) > 0 and isinstance(sequence[0], list):
            sequence = sequence[0]
            
        sequence = [d.to(device) for d in sequence]
        
        for data in sequence:
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :2]

        # Validation always uses TF=0.0 (rollout)
        predictions = model(sequence, dt=1.0, teacher_forcing_ratio=0.0)
        
        loss = 0.0
        for t, pred in enumerate(predictions):
            target = sequence[t].y[:, :args.num_dynamic_feats]
            loss += F.mse_loss(pred, target)
            
        loss = loss / len(predictions)
        total_loss += loss.item()
        n_batches += 1
    
    return {'loss': total_loss / max(n_batches, 1)}


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def main():
    parser = argparse.ArgumentParser(description="Train G-PARCv2 River with Scheduled Sampling")
    
    # Dataset paths
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, default="*.pt")
    
    # Dataset configuration
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_static_feats", type=int, default=9)
    parser.add_argument("--num_dynamic_feats", type=int, default=4)
    parser.add_argument("--velocity_indices", type=int, nargs='+', default=[2, 3])
    
    # Feature Extractor V2
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_layer_norm", action="store_true", default=True)
    parser.add_argument("--use_relative_pos", action="store_true", default=True)
    
    # Model / Integrator
    parser.add_argument("--integrator", type=str, default="euler", choices=["euler", "heun", "rk4"])
    
    # Differentiator (SPADE)
    parser.add_argument("--spade_random_noise", action="store_true", default=False)
    parser.add_argument("--spade_heads", type=int, default=4)
    parser.add_argument("--spade_concat", action="store_true", default=True)
    parser.add_argument("--spade_dropout", type=float, default=0.1)
    parser.add_argument("--zero_init", action="store_true", default=False)
    
    # Scheduled Sampling
    parser.add_argument("--ss_schedule", type=str, default="linear",
                        choices=['linear', 'exponential', 'sigmoid'],
                        help="Schedule for decaying teacher forcing ratio")
    parser.add_argument("--ss_initial_ratio", type=float, default=1.0)
    parser.add_argument("--ss_final_ratio", type=float, default=0.0)
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./outputs_river")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh_scheduler", action="store_true", default=False,
                        help="Fresh optimizer + scheduler when resuming")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load normalization statistics (if available)
    norm_stats = load_normalization_stats(args.train_dir)
    
    print("\n" + "="*70)
    print("G-PARCv2 RIVER TRAINING - SCHEDULED SAMPLING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"")
    print(f"Scheduled Sampling:")
    print(f"  Schedule: {args.ss_schedule}")
    print(f"  Initial TF ratio: {args.ss_initial_ratio}")
    print(f"  Final TF ratio: {args.ss_final_ratio}")
    print("="*70)
    
    # Load dataset
    train_dataset = RiverDataset(
        directory=args.train_dir, seq_len=args.seq_len, stride=args.stride,
        num_static_feats=args.num_static_feats, num_dynamic_feats=args.num_dynamic_feats,
        file_pattern=args.file_pattern, shuffle=True
    )
    val_dataset = RiverDataset(
        directory=args.val_dir, seq_len=args.seq_len, stride=args.stride,
        num_static_feats=args.num_static_feats, num_dynamic_feats=args.num_dynamic_feats,
        file_pattern=args.file_pattern, shuffle=False
    )
    
    loader_kwargs = {'batch_size': None, 'num_workers': args.num_workers, 'pin_memory': True}
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    
    # Get sample data
    print("\nGetting sample for initialization...")
    init_seq = next(iter(train_loader))
    if isinstance(init_seq, list) and len(init_seq) > 0 and isinstance(init_seq[0], list):
        init_seq = init_seq[0]
    sample_data = init_seq[0].to(device)

    # Ensure pos exists for initialization (using logic from Elastoplastic manual fix)
    if not hasattr(sample_data, 'pos') or sample_data.pos is None:
         sample_data.pos = sample_data.x[:, :2]
    
    # Create model
    print("\nCreating model...")
    model = create_model(args, sample_data, norm_stats).to(device)
    count_parameters(model)
    
    # Optimizer and scheduler (AdamW + Cosine)
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
            start_epoch = checkpoint.get('epoch', 0) + 1
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        best_val_loss = checkpoint.get('metrics', {}).get('val_loss', float('inf'))
        if best_val_loss == float('inf'): # Try legacy key
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
        print(f"  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    
    # Save config
    config = vars(args)
    config['scheduled_sampling'] = True
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'teacher_forcing_ratio': []
    }
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch, total_epochs=args.epochs, args=args
        )
        
        val_metrics = validate_epoch(model, val_loader, device, args)
        
        scheduler.step()
        
        print(f"\nTrain Loss: {train_metrics['loss']:.6f} (TF: {train_metrics['teacher_forcing_ratio']:.3f})")
        print(f"Val Loss:   {val_metrics['loss']:.6f}")
        
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['teacher_forcing_ratio'].append(train_metrics['teacher_forcing_ratio'])
        
        # Save Best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': best_val_loss, 'teacher_forcing_ratio': train_metrics['teacher_forcing_ratio']},
                output_dir / "best_model.pth"
            )
            print(f"✓ Saved best model (val_loss: {best_val_loss:.6f})")
        
        # Save Latest
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'val_loss': val_metrics['loss'], 'teacher_forcing_ratio': train_metrics['teacher_forcing_ratio']},
            output_dir / "latest_model.pth"
        )
        
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()