#!/usr/bin/env python3
"""
Training Script for G-PARC Elastoplastic Model - DISPLACEMENT ONLY
==================================================================
NEW: Scheduled Sampling for robust rollout training
UPDATED: Support for global max normalization (no hardcoded z-score params)
UPDATED: Passes norm_method to MLS operators for correct boundary damping
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

from utilities.featureextractor import FeatureExtractorGNN
from differentiator.differentiator import DerivativeGNN
from integrator.integrator import IntegralGNN
from data.ElastoPlasticDataset import ElastoPlasticDataset, get_simulation_ids
from models.parcv1_elasto import GPARC


def load_normalization_stats(data_dir):
    """
    Load normalization statistics from the data directory.
    
    Returns dictionary with normalization info or defaults if not found.
    """
    stats_file = Path(data_dir).parent / "normalization_stats.json"
    
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"\n✓ Loaded normalization stats from: {stats_file}")
        print(f"  Method: {stats.get('normalization_method', 'unknown')}")
        
        if 'position' in stats and 'displacement' in stats:
            print(f"  max_position: {stats['position']['max_position']:.2f} mm")
            print(f"  max_displacement: {stats['displacement']['max_displacement']:.2f} mm")
        
        return stats
    else:
        print(f"\n⚠️  No normalization_stats.json found at {stats_file}")
        print("   Using default z-score parameters (may be incorrect!)")
        
        # Default z-score params (for backward compatibility)
        return {
            'normalization_method': 'z_score',
            'position': {
                'x_pos': {'mean': 97.2165, 'std': 59.3803},
                'y_pos': {'mean': 50.2759, 'std': 28.4965}
            }
        }


def get_pos_normalization_params(norm_stats):
    """
    Extract position normalization parameters.
    
    For MLS operators, we need mean/std even with global max normalization
    to define the boundary margin in physical units.
    """
    pos_stats = norm_stats['position']
    pos_mean = [
        pos_stats['x_pos']['mean'],
        pos_stats['y_pos']['mean']
    ]
    pos_std = [
        pos_stats['x_pos']['std'],
        pos_stats['y_pos']['std']
    ]
    return pos_mean, pos_std


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
    """
    Compute teacher forcing ratio for current epoch.
    """
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


def create_model(args, sample_data, norm_stats):
    """Create PARC v1 model (simple GNN baseline)."""
    
    # Parse skip_dynamic_indices from string/list
    skip_indices = []
    if hasattr(args, 'skip_dynamic_indices') and args.skip_dynamic_indices:
        if isinstance(args.skip_dynamic_indices, str):
            skip_indices = [int(x) for x in args.skip_dynamic_indices.split(',')]
        elif isinstance(args.skip_dynamic_indices, list):
            skip_indices = [int(x) for x in args.skip_dynamic_indices]
    
    # Feature extractor output feeds into derivative solver along with dynamic feats
    deriv_in_channels = args.feature_out_channels + args.num_dynamic_feats
    
    print(f"\nCreating Feature Extractor (GNN)...")
    print(f"  Depth: {args.depth}")
    print(f"  Hidden: {args.hidden_channels}")
    print(f"  Output: {args.feature_out_channels}")
    print(f"  Heads: {args.heads}")
    
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
    
    print(f"\nCreating Derivative Solver (GNN)...")
    print(f"  In channels: {deriv_in_channels} (features={args.feature_out_channels} + dynamic={args.num_dynamic_feats})")
    print(f"  Hidden: {args.deriv_hidden_channels}")
    print(f"  Layers: {args.deriv_num_layers}")
    
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
    
    print(f"\nCreating Integral Solver (GNN)...")
    print(f"  In channels: {args.num_dynamic_feats}")
    print(f"  Hidden: {args.integral_hidden_channels}")
    print(f"  Layers: {args.integral_num_layers}")
    
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
    
    print(f"\nCreating PARC v1 model...")
    print(f"  Static feats: {args.num_static_feats}")
    print(f"  Dynamic feats: {args.num_dynamic_feats}")
    print(f"  Skip indices: {skip_indices}")
    
    model = GPARC(
        feature_extractor=feature_extractor,
        derivative_solver=derivative_solver,
        integral_solver=integral_solver,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        skip_dynamic_indices=skip_indices,
        feature_out_channels=args.feature_out_channels
    )
    
    return model


def get_valid_node_mask(elements, current_erosion, next_erosion=None, device='cpu'):
    """
    Get mask of valid nodes for loss computation.
    """
    num_nodes = elements.max().item() + 1
    valid_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    
    if current_erosion is not None:
        eroded_elements = current_erosion.squeeze() < 0.5
        if eroded_elements.any():
            eroded_nodes = elements[eroded_elements].flatten().unique()
            valid_mask[eroded_nodes] = False
    
    if next_erosion is not None:
        will_erode = next_erosion.squeeze() < 0.5
        if will_erode.any():
            eroding_nodes = elements[will_erode].flatten().unique()
            valid_mask[eroding_nodes] = False
    
    return valid_mask


def compute_masked_loss(pred, target, elements, current_erosion, next_erosion=None):
    """Compute MSE loss only on valid (non-eroding) nodes."""
    device = pred.device
    valid_mask = get_valid_node_mask(elements, current_erosion, next_erosion, device)
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device), 0, pred.shape[0]
    
    loss = F.mse_loss(pred[valid_mask], target[valid_mask])
    return loss, valid_mask.sum().item(), (~valid_mask).sum().item()


# =========================================================================
# TRAINING WITH SCHEDULED SAMPLING
# =========================================================================

def train_epoch(model, train_loader, optimizer, device, epoch, total_epochs, args):
    """
    Train for one epoch with scheduled sampling.
    """
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
    total_valid_nodes = 0
    total_eroded_nodes = 0
    
    pbar = tqdm(train_loader, desc=f"Training (TF={teacher_forcing_ratio:.3f})")
    
    for seq in pbar:
        for data in seq:
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :model.num_static_feats]
        
        optimizer.zero_grad()
        
        # PARC v1 forward only takes data_list (no dt or teacher_forcing_ratio)
        predictions = model(seq)
        
        loss = 0.0
        total_weight = 0.0
        
        for t, (pred, data) in enumerate(zip(predictions, seq)):
            # Process targets to match model output (skip indices)
            target = model.process_targets(data.y)
            
            if args.mask_eroding and hasattr(data, 'elements') and hasattr(data, 'x_element'):
                elements = data.elements
                current_erosion = data.x_element
                next_erosion = data.y_element if hasattr(data, 'y_element') else None
                
                step_loss, n_valid, n_masked = compute_masked_loss(
                    pred, target, elements, current_erosion, next_erosion
                )
                total_valid_nodes += n_valid
                total_eroded_nodes += n_masked
            else:
                step_loss = F.mse_loss(pred, target)
            
            if args.use_loss_decay:
                weight = args.loss_decay_gamma ** t
            else:
                weight = 1.0
            
            loss += weight * step_loss
            total_weight += weight
        
        if args.use_loss_decay:
            loss = loss / total_weight
        else:
            loss = loss / len(predictions)
        
        loss.backward()
        
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.6f}",
            'TF': f"{teacher_forcing_ratio:.3f}",
            'masked': f"{100*total_eroded_nodes/(total_valid_nodes+total_eroded_nodes+1e-6):.1f}%"
        })
    
    return {
        'loss': total_loss / n_batches,
        'teacher_forcing_ratio': teacher_forcing_ratio,
        'valid_nodes': total_valid_nodes,
        'eroded_nodes': total_eroded_nodes,
    }


@torch.no_grad()
def validate_epoch(model, val_loader, device, mask_eroding=True):
    """
    Validate for one epoch.
    NOTE: Validation ALWAYS uses teacher_forcing_ratio=0.0 (pure autoregressive)
    """
    model.eval()
    
    total_loss = 0.0
    n_batches = 0
    
    for seq in tqdm(val_loader, desc="Validating"):
        for data in seq:
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :model.num_static_feats]
        
        # PARC v1 forward only takes data_list
        predictions = model(seq)
        
        loss = 0.0
        for t, (pred, data) in enumerate(zip(predictions, seq)):
            # Process targets to match model output (skip indices)
            target = model.process_targets(data.y)
            
            if mask_eroding and hasattr(data, 'elements') and hasattr(data, 'x_element'):
                elements = data.elements
                current_erosion = data.x_element
                next_erosion = data.y_element if hasattr(data, 'y_element') else None
                step_loss, _, _ = compute_masked_loss(
                    pred, target, elements, current_erosion, next_erosion
                )
            else:
                step_loss = F.mse_loss(pred, target)
            
            loss += step_loss
        
        loss = loss / len(predictions)
        total_loss += loss.item()
        n_batches += 1
    
    return {'loss': total_loss / n_batches}


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
    parser = argparse.ArgumentParser(description="Train G-PARC with Scheduled Sampling")
    
    # Dataset paths
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, default="*.pt")
    
    # Dataset configuration
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=2)
    parser.add_argument("--n_state_var", type=int, default=0)
    
    # Physics features
    parser.add_argument("--use_von_mises", action="store_true", default=True)
    parser.add_argument("--use_volumetric", action="store_true", default=True)
    
    # Feature Extractor (GNN)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--feature_out_channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--pool_ratios", type=float, nargs='+', default=[0.2])
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # Derivative Solver (GNN)
    parser.add_argument("--deriv_hidden_channels", type=int, default=128)
    parser.add_argument("--deriv_num_layers", type=int, default=3)
    parser.add_argument("--deriv_heads", type=int, default=4)
    parser.add_argument("--deriv_dropout", type=float, default=0.2)
    parser.add_argument("--deriv_use_residual", action="store_true", default=False)
    
    # Integral Solver (GNN)
    parser.add_argument("--integral_hidden_channels", type=int, default=128)
    parser.add_argument("--integral_num_layers", type=int, default=3)
    parser.add_argument("--integral_heads", type=int, default=4)
    parser.add_argument("--integral_dropout", type=float, default=0.2)
    parser.add_argument("--integral_use_residual", action="store_true", default=False)
    
    # Dynamic feature skipping
    parser.add_argument("--skip_dynamic_indices", type=str, default=None,
                        help="Comma-separated indices of dynamic features to skip (e.g., '2')")
    
    # Loss configuration
    parser.add_argument("--mask_eroding", action="store_true", default=True)
    parser.add_argument("--use_loss_decay", action="store_true", default=False,
                        help="Use exponential weight decay in loss (gamma^t)")
    parser.add_argument("--loss_decay_gamma", type=float, default=0.9,
                        help="Decay factor for multi-step loss")
    
    # Scheduled Sampling
    parser.add_argument("--ss_schedule", type=str, default="linear",
                        choices=['linear', 'exponential', 'sigmoid'],
                        help="Schedule for decaying teacher forcing ratio")
    parser.add_argument("--ss_initial_ratio", type=float, default=1.0,
                        help="Initial teacher forcing ratio (1.0 = full supervision)")
    parser.add_argument("--ss_final_ratio", type=float, default=0.0,
                        help="Final teacher forcing ratio (0.0 = no teacher forcing)")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./outputs_scheduled_sampling")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset_best", action="store_true", default=False,
                        help="Reset best_val_loss when resuming (use when changing seq_len or loss config)")
    parser.add_argument("--fresh_scheduler", action="store_true", default=False,
                        help="Fresh optimizer + scheduler when resuming (use when changing lr or training config)")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load normalization statistics
    norm_stats = load_normalization_stats(args.train_dir)
    
    print("\n" + "="*70)
    print("PARC v1 TRAINING - BASELINE")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Normalization: {norm_stats.get('normalization_method', 'unknown')}")
    print(f"")
    print(f"Feature Extractor: GNN")
    print(f"  Depth: {args.depth}")
    print(f"  Hidden: {args.hidden_channels}")
    print(f"  Output: {args.feature_out_channels}")
    print(f"")
    print(f"Derivative Solver: GNN")
    print(f"  Layers: {args.deriv_num_layers}")
    print(f"  Hidden: {args.deriv_hidden_channels}")
    print(f"")
    print(f"Integral Solver: GNN")
    print(f"  Layers: {args.integral_num_layers}")
    print(f"  Hidden: {args.integral_hidden_channels}")
    print(f"")
    print(f"Dynamic Features: {args.num_dynamic_feats}")
    print(f"Skip Indices: {args.skip_dynamic_indices}")
    print(f"")
    print(f"Scheduled Sampling:")
    print(f"  Schedule: {args.ss_schedule}")
    print(f"  Initial TF ratio: {args.ss_initial_ratio}")
    print(f"  Final TF ratio: {args.ss_final_ratio}")
    print(f"  Epochs: {args.epochs}")
    print(f"")
    print(f"Loss Configuration:")
    print(f"  Mask eroding: {args.mask_eroding}")
    print(f"  Exponential decay: {args.use_loss_decay}")
    if args.use_loss_decay:
        print(f"  Decay gamma: {args.loss_decay_gamma}")
    print("="*70)
    
    # Load dataset
    train_ids = get_simulation_ids(Path(args.train_dir), pattern=args.file_pattern)
    val_ids = get_simulation_ids(Path(args.val_dir), pattern=args.file_pattern)
    
    print(f"\nFound {len(train_ids)} training simulations")
    print(f"Found {len(val_ids)} validation simulations")
    
    train_dataset = ElastoPlasticDataset(
        directory=Path(args.train_dir),
        simulation_ids=train_ids,
        seq_len=args.seq_len,
        stride=args.stride,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats
    )
    val_dataset = ElastoPlasticDataset(
        directory=Path(args.val_dir),
        simulation_ids=val_ids,
        seq_len=args.seq_len,
        stride=args.stride,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats
    )
    
    loader_kwargs = {'batch_size': None, 'num_workers': args.num_workers, 'pin_memory': True}
    
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    
    # Get sample data
    print("\nGetting sample for initialization...")
    sample_data = next(iter(train_loader))[0]
    print(f"  Nodes: {sample_data.num_nodes}")
    print(f"  Edges: {sample_data.edge_index.shape[1]}")
    
    # Create model (now with loaded norm stats)
    print("\nCreating model...")
    model = create_model(args, sample_data, norm_stats).to(device)
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
            start_epoch = 0  # ← ADD THIS LINE
            print(f"  Fresh optimizer + scheduler (lr={args.lr}, T_max={args.epochs})")
            optimizer = AdamW(model.parameters(), lr=args.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        else:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        #start_epoch = checkpoint['epoch'] + 1
        
        if args.reset_best:
            best_val_loss = float('inf')
            print(f"  Reset best_val_loss (new seq_len or loss config)")
        else:
            best_val_loss = checkpoint['metrics'].get('val_loss', float('inf'))
        
        print(f"  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.8f}")
    
    # Save config (including normalization info)
    config = vars(args)
    config['model'] = 'PARC v1 (GNN baseline)'
    config['scheduled_sampling'] = True
    config['normalization'] = norm_stats.get('normalization_method', 'unknown')
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Copy normalization stats to output for reference
    if (Path(args.train_dir).parent / "normalization_stats.json").exists():
        import shutil
        shutil.copy2(
            Path(args.train_dir).parent / "normalization_stats.json",
            output_dir / "normalization_stats.json"
        )
        print(f"✓ Copied normalization_stats.json to output directory")
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING WITH SCHEDULED SAMPLING")
    print("="*70)
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  TF schedule: {args.ss_schedule} ({args.ss_initial_ratio} → {args.ss_final_ratio})")
    print("="*70)
    
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
            epoch=epoch,
            total_epochs=args.epochs,
            args=args
        )
        
        val_metrics = validate_epoch(model, val_loader, device, args.mask_eroding)
        
        scheduler.step()
        
        print(f"\nTrain Loss: {train_metrics['loss']:.6f} "
              f"(TF ratio: {train_metrics['teacher_forcing_ratio']:.3f})")
        print(f"Val Loss:   {val_metrics['loss']:.6f} (Free running rollout)")
        print(f"Valid nodes: {train_metrics['valid_nodes']:,}, "
              f"Masked: {train_metrics['eroded_nodes']:,}")
        
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['teacher_forcing_ratio'].append(train_metrics['teacher_forcing_ratio'])
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': best_val_loss, 'teacher_forcing_ratio': train_metrics['teacher_forcing_ratio']},
                output_dir / "best_model.pth"
            )
            print(f"✓ Saved best model (val_loss: {best_val_loss:.6f})")
        
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'val_loss': val_metrics['loss'], 'teacher_forcing_ratio': train_metrics['teacher_forcing_ratio']},
            output_dir / "latest_model.pth"
        )
        
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best model: {output_dir / 'best_model.pth'}")
    print(f"Latest model: {output_dir / 'latest_model.pth'}")
    print("="*70)


if __name__ == "__main__":
    main()