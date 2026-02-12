"""
Training script for MeshGraphNets on elastoplastic dynamics.

This script:
- Loads data from your preprocessed .pt files
- Trains MeshGraphNets with proper normalization
- Handles sequences and autoregressive rollout
- Saves checkpoints and logs metrics
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse

from meshgraphnet import MeshGraphNet, compute_stats, normalize, unnormalize
from dataset import ElastoPlasticDataset, create_datasets_from_folders


def collate_sequences(batch):
    """
    Collate function for DataLoader that handles sequences.
    
    Args:
        batch: List of sequences, where each sequence is a list of Data objects
        
    Returns:
        Flattened list of Data objects (removes sequence structure)
    """
    # Flatten the batch: each item in batch is a sequence
    flattened = []
    for sequence in batch:
        flattened.extend(sequence)
    return flattened


def train_epoch(model, loader, optimizer, device, stats):
    """
    Train for one epoch.
    
    Args:
        model: MeshGraphNet model
        loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
        stats: Dictionary with normalization statistics
        
    Returns:
        avg_loss: Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Move stats to device
    mean_vec_x = stats['mean_vec_x'].to(device)
    std_vec_x = stats['std_vec_x'].to(device)
    mean_vec_edge = stats['mean_vec_edge'].to(device)
    std_vec_edge = stats['std_vec_edge'].to(device)
    mean_vec_y = stats['mean_vec_y'].to(device)
    std_vec_y = stats['std_vec_y'].to(device)
    
    for batch in loader:
        # Batch is a list of Data objects (sequence)
        # Zero gradients once per sequence
        optimizer.zero_grad()
        batch_loss = 0
        
        # Accumulate gradients across all timesteps in sequence
        for data in batch:
            data = data.to(device)
            
            # Forward pass
            pred = model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            
            # Compute loss
            loss = model.loss(pred, data.y, mean_vec_y, std_vec_y)
            
            # Backward pass - accumulates gradients
            loss.backward()
            
            batch_loss += loss.item()
        
        # Update weights once per sequence (after all timesteps)
        optimizer.step()
        
        total_loss += batch_loss / len(batch)
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, loader, device, stats):
    """
    Validate the model.
    
    Args:
        model: MeshGraphNet model
        loader: DataLoader for validation data
        device: Device to validate on
        stats: Dictionary with normalization statistics
        
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Move stats to device
    mean_vec_x = stats['mean_vec_x'].to(device)
    std_vec_x = stats['std_vec_x'].to(device)
    mean_vec_edge = stats['mean_vec_edge'].to(device)
    std_vec_edge = stats['std_vec_edge'].to(device)
    mean_vec_y = stats['mean_vec_y'].to(device)
    std_vec_y = stats['std_vec_y'].to(device)
    
    with torch.no_grad():
        for batch in loader:
            batch_loss = 0
            
            for data in batch:
                data = data.to(device)
                
                # Forward pass
                pred = model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
                
                # Compute loss
                loss = model.loss(pred, data.y, mean_vec_y, std_vec_y)
                
                batch_loss += loss.item()
            
            total_loss += batch_loss / len(batch)
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def autoregressive_rollout(model, sequence, device, stats, num_steps=None):
    """
    Perform autoregressive rollout on a sequence.
    
    Args:
        model: MeshGraphNet model
        sequence: Initial sequence of Data objects
        device: Device to run on
        stats: Normalization statistics
        num_steps: Number of steps to rollout (default: len(sequence) - 1)
        
    Returns:
        predictions: List of predicted states
        errors: List of errors at each timestep
    """
    model.eval()
    
    if num_steps is None:
        num_steps = len(sequence) - 1
    
    # Move stats to device
    mean_vec_x = stats['mean_vec_x'].to(device)
    std_vec_x = stats['std_vec_x'].to(device)
    mean_vec_edge = stats['mean_vec_edge'].to(device)
    std_vec_edge = stats['std_vec_edge'].to(device)
    mean_vec_y = stats['mean_vec_y'].to(device)
    std_vec_y = stats['std_vec_y'].to(device)
    
    # Start with initial state
    current_state = sequence[0].clone().to(device)
    predictions = [current_state]
    errors = []
    
    with torch.no_grad():
        for t in range(num_steps):
            # Predict next state
            pred = model(current_state, mean_vec_x, std_vec_x, 
                        mean_vec_edge, std_vec_edge)
            
            # Unnormalize prediction
            pred_unnorm = unnormalize(pred, mean_vec_y, std_vec_y)
            
            # Update state (positions stay fixed, displacements update)
            next_state = current_state.clone()
            # Assuming x = [x_pos, y_pos, U_x, U_y]
            next_state.x[:, 2:] = current_state.x[:, 2:] + pred_unnorm
            
            predictions.append(next_state)
            
            # Compute error if ground truth is available
            if t + 1 < len(sequence):
                ground_truth = sequence[t + 1].to(device)
                error = torch.mean((next_state.x[:, 2:] - ground_truth.x[:, 2:]) ** 2).item()
                errors.append(error)
            
            current_state = next_state
    
    return predictions, errors


def main(args):
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    datasets = create_datasets_from_folders(
        base_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        num_static_feats=2,
        num_dynamic_feats=2,
        use_element_features=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        datasets['train'],
        batch_size=None,  # Dataset yields sequences
        num_workers=args.num_workers,
        shuffle=False
    )
    
    val_loader = DataLoader(
        datasets['val'],
        batch_size=None,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    # Compute normalization statistics
    print("\nComputing normalization statistics...")
    stats = compute_stats(datasets['train'], max_samples=args.max_stats_samples)
    
    # Save statistics
    stats_path = Path(args.checkpoint_dir) / 'normalization_stats.pt'
    torch.save(stats, stats_path)
    print(f"Saved normalization stats to {stats_path}")
    
    # Create model
    print("\nCreating model...")
    model = MeshGraphNet(
        input_dim_node=4,  # [x_pos, y_pos, U_x, U_y]
        input_dim_edge=3,  # [dx, dy, distance]
        hidden_dim=args.hidden_dim,
        output_dim=2,  # [ΔU_x, ΔU_y]
        num_layers=args.num_layers
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, 
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        scheduler = None
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, stats)
        train_losses.append(train_loss)
        
        # Validate every N epochs
        if (epoch + 1) % args.val_every == 0:
            val_loss = validate(model, val_loader, device, stats)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = checkpoint_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'args': vars(args)
                }, checkpoint_path)
                print(f"  Saved best model (val_loss: {val_loss:.6f})")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}")
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'args': vars(args)
            }, checkpoint_path)
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'args': vars(args)
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MeshGraphNets on elastoplastic data')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                       default='/scratch/jtb3sud/processed_elasto_plastic/zscore/normalized',
                       help='Base directory containing train/val/test folders')
    parser.add_argument('--seq_len', type=int, default=10,
                       help='Sequence length for training')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for sequence windows')
    parser.add_argument('--max_stats_samples', type=int, default=1000,
                       help='Maximum samples for computing normalization stats')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for MLPs')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of message passing layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='step',
                       choices=['none', 'step', 'cosine'],
                       help='Learning rate scheduler')
    parser.add_argument('--scheduler_step', type=int, default=100,
                       help='Step size for StepLR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.9,
                       help='Gamma for StepLR scheduler')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Logging arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--val_every', type=int, default=10,
                       help='Validate every N epochs')
    parser.add_argument('--save_every', type=int, default=100,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)