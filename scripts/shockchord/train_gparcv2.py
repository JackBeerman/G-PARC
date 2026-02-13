#!/usr/bin/env python3
"""
Training Script for G-PARCv2 Shock Tube Model
==============================================
Matches the river/elastoplastic training pattern:
  - AdamW + CosineAnnealingLR
  - Scheduled Sampling (teacher forcing decay)
  - JSON logging + config
  - Save best + latest checkpoints
  - TQDM progress bars

Shock-tube specifics:
  - Global parameter conditioning (pressure, density, delta_t)
  - skip_dynamic_indices = [2] (skip y_momentum)
  - process_targets() for correct loss computation
  - 2D mesh, MLS 2-hop extension disabled (structured mesh)
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
from differentiator.shocktubedifferentiator import ShockTubeDifferentiator
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from models.shocktube_gparcv2 import GPARC_ShockTube_V2
from data.ShockChorddt import ShockTubeRolloutDataset, get_simulation_ids


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
            decay = (final_ratio / max(initial_ratio, 1e-8)) ** (1 / total_epochs)
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


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{' MODEL PARAMETERS ':~^50}")
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"{'~'*50}\n")
    return trainable


# =========================================================================
# MODEL CREATION
# =========================================================================

def create_model(args, sample_data):
    """Create G-PARCv2 Shock Tube model."""
    
    print("\nInitializing MLS Operators...")
    gradient_solver = SolveGradientsLST()
    laplacian_solver = SolveWeightLST2d(use_2hop_extension=False)
    
    print(f"\nCreating Feature Extractor...")
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
        use_relative_pos=args.use_relative_pos,
    )
    
    print(f"\nPhysics configuration:")
    print(f"  Static feats: {args.num_static_feats}")
    print(f"  Dynamic feats (raw): {args.num_dynamic_feats + len(args.skip_dynamic_indices)}")
    print(f"  Dynamic feats (used): {args.num_dynamic_feats} (skip {args.skip_dynamic_indices})")
    print(f"  Advection: all {args.num_dynamic_feats} features")
    print(f"  Velocity index: {args.velocity_index} (x_momentum)")
    print(f"  Global embed dim: {args.global_embed_dim}")
    print(f"  Integrator: {args.integrator}")
    print(f"  2-hop extension: DISABLED")
    
    derivative_solver = ShockTubeDifferentiator(
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=args.feature_out_channels,
        global_embed_dim=args.global_embed_dim,
        list_adv_idx=list(range(args.num_dynamic_feats)),
        list_dif_idx=list(range(args.num_dynamic_feats)),
        velocity_indices=[args.velocity_index],
        spade_random_noise=args.spade_random_noise,
        heads=args.spade_heads,
        concat=args.spade_concat,
        dropout=args.spade_dropout,
        zero_init=args.zero_init,
    )
    
    print("Initializing MLS weights...")
    derivative_solver.initialize_weights(sample_data)
    
    model = GPARC_ShockTube_V2(
        derivative_solver_physics=derivative_solver,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        skip_dynamic_indices=args.skip_dynamic_indices,
        global_param_dim=args.global_param_dim,
        global_embed_dim=args.global_embed_dim,
    )
    
    return model


# =========================================================================
# TRAINING / VALIDATION
# =========================================================================

def train_epoch(model, train_loader, optimizer, device, epoch, total_epochs, args):
    """Train one epoch with scheduled sampling."""
    model.train()
    
    tf_ratio = get_teacher_forcing_ratio(
        epoch=epoch,
        total_epochs=total_epochs,
        schedule=args.ss_schedule,
        initial_ratio=args.ss_initial_ratio,
        final_ratio=args.ss_final_ratio,
    )
    
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Training (TF={tf_ratio:.3f})")
    
    for sequence in pbar:
        if isinstance(sequence, list) and len(sequence) > 0 and isinstance(sequence[0], list):
            sequence = sequence[0]
        
        sequence = [d.to(device) for d in sequence]
        
        # Ensure pos exists
        for data in sequence:
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :args.num_static_feats]
        
        optimizer.zero_grad()
        
        predictions = model(sequence, dt=None, teacher_forcing_ratio=tf_ratio)
        
        loss = 0.0
        for t, pred in enumerate(predictions):
            # Use process_targets to apply skip_dynamic_indices to y
            target = model.process_targets(sequence[t].y)
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
        'teacher_forcing_ratio': tf_ratio,
    }


@torch.no_grad()
def validate_epoch(model, val_loader, device, args):
    """Validate one epoch (always pure rollout, TF=0)."""
    model.eval()
    
    total_loss = 0.0
    n_batches = 0
    
    for sequence in tqdm(val_loader, desc="Validating"):
        if isinstance(sequence, list) and len(sequence) > 0 and isinstance(sequence[0], list):
            sequence = sequence[0]
        
        sequence = [d.to(device) for d in sequence]
        
        for data in sequence:
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :args.num_static_feats]
        
        predictions = model(sequence, dt=None, teacher_forcing_ratio=0.0)
        
        loss = 0.0
        for t, pred in enumerate(predictions):
            target = model.process_targets(sequence[t].y)
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
        'metrics': metrics,
    }
    torch.save(checkpoint, filepath)


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train G-PARCv2 Shock Tube")
    
    # Data paths
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, default="*.pt")
    
    # Data config
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=3,
                        help="Dynamic features AFTER skipping (density, x_mom, energy)")
    parser.add_argument("--skip_dynamic_indices", type=int, nargs='+', default=[2],
                        help="Raw dynamic indices to skip (2 = y_momentum)")
    parser.add_argument("--velocity_index", type=int, default=1,
                        help="Index of x_momentum in USED dynamic features")
    
    # Global params
    parser.add_argument("--global_param_dim", type=int, default=3,
                        help="Dimension of global params (pressure, density, delta_t)")
    parser.add_argument("--global_embed_dim", type=int, default=64,
                        help="Embedding dimension for global params")
    
    # Feature Extractor
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_layer_norm", action="store_true", default=True)
    parser.add_argument("--use_relative_pos", action="store_true", default=True)
    
    # Integrator
    parser.add_argument("--integrator", type=str, default="euler",
                        choices=["euler", "heun", "rk4"])
    
    # Differentiator (SPADE)
    parser.add_argument("--spade_random_noise", action="store_true", default=False)
    parser.add_argument("--spade_heads", type=int, default=4)
    parser.add_argument("--spade_concat", action="store_true", default=True)
    parser.add_argument("--spade_dropout", type=float, default=0.1)
    parser.add_argument("--zero_init", action="store_true", default=False)
    
    # Scheduled Sampling
    parser.add_argument("--ss_schedule", type=str, default="linear",
                        choices=['linear', 'exponential', 'sigmoid'])
    parser.add_argument("--ss_initial_ratio", type=float, default=0.0)
    parser.add_argument("--ss_final_ratio", type=float, default=0.0)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./outputs_shocktube_v2")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh_scheduler", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("G-PARCv2 SHOCK TUBE TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Scheduled Sampling: {args.ss_schedule} ({args.ss_initial_ratio} → {args.ss_final_ratio})")
    print(f"Skip dynamic indices: {args.skip_dynamic_indices}")
    print(f"Global params: pressure, density, delta_t → embed_dim={args.global_embed_dim}")
    print("="*70)
    
    # Dataset — note: ShockTubeRolloutDataset uses num_dynamic_feats=4 (raw count)
    # The v1 dataset yields raw features; skipping happens in the model
    raw_dynamic = args.num_dynamic_feats + len(args.skip_dynamic_indices)
    
    train_dataset = ShockTubeRolloutDataset(
        directory=args.train_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=raw_dynamic,
        file_pattern=args.file_pattern,
    )
    val_dataset = ShockTubeRolloutDataset(
        directory=args.val_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=raw_dynamic,
        file_pattern=args.file_pattern,
    )
    
    loader_kwargs = {'batch_size': None, 'num_workers': args.num_workers, 'pin_memory': True}
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    
    # Sample data for initialization
    print("\nGetting sample for initialization...")
    init_seq = next(iter(train_loader))
    if isinstance(init_seq, list) and len(init_seq) > 0 and isinstance(init_seq[0], list):
        init_seq = init_seq[0]
    sample_data = init_seq[0].to(device)
    
    if not hasattr(sample_data, 'pos') or sample_data.pos is None:
        sample_data.pos = sample_data.x[:, :args.num_static_feats]
    
    print(f"  Sample: x={sample_data.x.shape}, y={sample_data.y.shape}, "
          f"edges={sample_data.edge_index.shape}")
    if hasattr(sample_data, 'global_pressure'):
        print(f"  Global: pressure={sample_data.global_pressure.item():.4f}, "
              f"density={sample_data.global_density.item():.4f}, "
              f"delta_t={sample_data.global_delta_t.item():.4f}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(args, sample_data).to(device)
    count_parameters(model)
    
    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        
        if args.fresh_scheduler:
            start_epoch = 0
            optimizer = AdamW(model.parameters(), lr=args.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
            print(f"  Fresh optimizer + scheduler")
        else:
            start_epoch = ckpt.get('epoch', 0) + 1
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if ckpt.get('scheduler_state_dict'):
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        best_val_loss = ckpt.get('metrics', {}).get('val_loss', float('inf'))
        print(f"  Resuming from epoch {start_epoch}, best_val={best_val_loss:.6f}")
    
    # Save config
    config = vars(args)
    config['raw_dynamic_feats'] = raw_dynamic
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'teacher_forcing_ratio': []}
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch, total_epochs=args.epochs, args=args,
        )
        
        val_metrics = validate_epoch(model, val_loader, device, args)
        
        scheduler.step()
        
        print(f"\nTrain Loss: {train_metrics['loss']:.6f} "
              f"(TF: {train_metrics['teacher_forcing_ratio']:.3f})")
        print(f"Val Loss:   {val_metrics['loss']:.6f}")
        
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['teacher_forcing_ratio'].append(train_metrics['teacher_forcing_ratio'])
        
        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': best_val_loss, 'tf': train_metrics['teacher_forcing_ratio']},
                output_dir / "best_model.pth",
            )
            print(f"✓ Saved best model (val_loss: {best_val_loss:.6f})")
        
        # Save latest
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'val_loss': val_metrics['loss'], 'tf': train_metrics['teacher_forcing_ratio']},
            output_dir / "latest_model.pth",
        )
        
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()