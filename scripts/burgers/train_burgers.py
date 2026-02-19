#!/usr/bin/env python3
"""
Training Script for G-PARCv2 Burgers' Equation
===============================================
Matches the shocktube/river/elastoplastic training pattern:
  - AdamW + CosineAnnealingLR
  - Scheduled Sampling (teacher forcing decay)
  - JSON logging + config
  - Save best + latest checkpoints
  - TQDM progress bars

Burgers specifics:
  - Static Features:  [pos_x, pos_y, Re] → 3 channels
  - Dynamic Features: [u, v]             → 2 channels
  - Physics: Advection(MLS) + Diffusion(FD)
  - FiLM conditioning on Reynolds number
  - dt=1.0 (FiLM handles Re variation)
  - Integrator: Euler/Heun/RK4
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
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from differentiator.burgers_differentiator import BurgersDifferentiator
from models.burgers import GPARC_Burgers_Numerical
from data.bbs import BurgersDataset


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
    """Create G-PARCv2 Burgers model."""

    print("\nInitializing MLS Gradient Solver...")
    gradient_solver = SolveGradientsLST()

    laplacian_solver = None
    if args.diffusion_type == 'mls':
        print("Initializing MLS Laplacian Solver...")
        laplacian_solver = SolveWeightLST2d(use_2hop_extension=True)

    print(f"\nCreating Feature Extractor...")
    print(f"  Layers: {args.num_fe_layers}")
    print(f"  Hidden: {args.hidden_channels}")
    print(f"  Output: {args.feature_out_channels}")

    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=args.num_static_feats,
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        num_layers=args.num_fe_layers,
        dropout=args.dropout,
        use_layer_norm=args.use_layer_norm,
        use_relative_pos=args.use_relative_pos,
    )

    print(f"\nPhysics configuration:")
    print(f"  Static feats:  {args.num_static_feats} (pos_x, pos_y, Re)")
    print(f"  Dynamic feats: {args.num_dynamic_feats} (u, v)")
    print(f"  Diffusion:     {args.diffusion_type}")
    print(f"  FiLM on Re:    {args.use_film}")
    print(f"  Integrator:    {args.integrator}")

    derivative_solver = BurgersDifferentiator(
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=args.feature_out_channels,
        spade_heads=args.spade_heads,
        spade_dropout=args.spade_dropout,
        zero_init=args.zero_init,
        diffusion_type=args.diffusion_type,
        use_film=args.use_film,
    )

    print("Initializing MLS weights...")
    derivative_solver.initialize_weights(sample_data)

    model = GPARC_Burgers_Numerical(
        derivative_solver=derivative_solver,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
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

        # Move to device
        for data in sequence:
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :2]

        optimizer.zero_grad()

        # Forward with teacher forcing
        predictions = []
        F_prev = None

        for i, data in enumerate(sequence):
            x = data.x
            edge_index = data.edge_index
            static_feats = x[:, :model.num_static_feats]

            if i == 0:
                current_dynamic = x[:, model.num_static_feats:]
            else:
                if tf_ratio > 0 and torch.rand(1).item() < tf_ratio:
                    current_dynamic = data.x[:, model.num_static_feats:]
                else:
                    current_dynamic = F_prev.detach()

            F_next = model.integrator(
                derivative_fn=model.derivative_solver,
                static_feats=static_feats,
                dynamic_state=current_dynamic,
                edge_index=edge_index,
                dt=1.0,
            )

            predictions.append(F_next)
            F_prev = F_next

        # Loss
        loss = 0.0
        for t, pred in enumerate(predictions):
            target = model.process_targets(sequence[t].y)
            loss += F.mse_loss(pred, target)
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

        for data in sequence:
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :2]

        # Pure autoregressive (no teacher forcing)
        predictions = []
        F_prev = None

        for i, data in enumerate(sequence):
            x = data.x
            edge_index = data.edge_index
            static_feats = x[:, :model.num_static_feats]

            if i == 0:
                current_dynamic = x[:, model.num_static_feats:]
            else:
                current_dynamic = F_prev

            F_next = model.integrator(
                derivative_fn=model.derivative_solver,
                static_feats=static_feats,
                dynamic_state=current_dynamic,
                edge_index=edge_index,
                dt=1.0,
            )

            predictions.append(F_next)
            F_prev = F_next

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
    parser = argparse.ArgumentParser(description="Train G-PARCv2 Burgers")

    # Data paths
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--file_pattern", type=str, default="*.pt")

    # Data config
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--num_static_feats", type=int, default=3,
                        help="Static features: pos_x, pos_y, Re")
    parser.add_argument("--num_dynamic_feats", type=int, default=2,
                        help="Dynamic features: u, v")

    # Feature Extractor
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--num_fe_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_layer_norm", action="store_true", default=True)
    parser.add_argument("--use_relative_pos", action="store_true", default=True)

    # Integrator
    parser.add_argument("--integrator", type=str, default="euler",
                        choices=["euler", "heun", "rk4"])

    # Diffusion operator
    parser.add_argument("--diffusion_type", type=str, default="fd",
                        choices=["fd", "mls", "none"])

    # Differentiator (SPADE)
    parser.add_argument("--spade_heads", type=int, default=2)
    parser.add_argument("--spade_dropout", type=float, default=0.1)
    parser.add_argument("--zero_init", action="store_true", default=True)
    parser.add_argument("--use_film", action="store_true", default=True)
    parser.add_argument("--no_film", dest="use_film", action="store_false")

    # Scheduled Sampling
    parser.add_argument("--ss_schedule", type=str, default="linear",
                        choices=['linear', 'exponential', 'sigmoid'])
    parser.add_argument("--ss_initial_ratio", type=float, default=0.0)
    parser.add_argument("--ss_final_ratio", type=float, default=0.0)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./outputs_burgers")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh_scheduler", action="store_true", default=False)

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("\n" + "=" * 70)
    print("G-PARCv2 BURGERS TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Scheduled Sampling: {args.ss_schedule} "
          f"({args.ss_initial_ratio} → {args.ss_final_ratio})")
    print(f"Diffusion: {args.diffusion_type}")
    print(f"FiLM on Re: {args.use_film}")
    print("=" * 70)

    # Dataset
    train_dataset = BurgersDataset(
        args.train_dir,
        file_pattern=args.file_pattern,
        seq_len=args.seq_len,
    )
    val_dataset = BurgersDataset(
        args.val_dir,
        file_pattern=args.file_pattern,
        seq_len=args.seq_len,
    )

    loader_kwargs = {
        'batch_size': None,
        'num_workers': args.num_workers,
        'pin_memory': True,
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    # Sample data for initialization
    print("\nGetting sample for initialization...")
    init_seq = next(iter(train_loader))
    if isinstance(init_seq, list) and len(init_seq) > 0 and isinstance(init_seq[0], list):
        init_seq = init_seq[0]
    sample_data = init_seq[0].to(device)

    if not hasattr(sample_data, 'pos') or sample_data.pos is None:
        sample_data.pos = sample_data.x[:, :2]

    print(f"  Sample: x={sample_data.x.shape}, y={sample_data.y.shape}, "
          f"edges={sample_data.edge_index.shape}")
    print(f"  Re (sample): {sample_data.x[0, 2].item():.4f}")

    # Create model
    print("\nCreating model...")
    model = create_model(args, sample_data).to(device)
    count_parameters(model)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and Path(args.resume).exists():
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

        if args.fresh_scheduler:
            start_epoch = 0
            optimizer = AdamW(model.parameters(), lr=args.lr)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
            )
            print(f"  Fresh optimizer + scheduler")
        else:
            start_epoch = ckpt.get('epoch', 0) + 1
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if ckpt.get('scheduler_state_dict'):
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        best_val_loss = ckpt.get('metrics', {}).get('val_loss', float('inf'))
        print(f"  Resuming from epoch {start_epoch}, best_val={best_val_loss:.6f}")

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'teacher_forcing_ratio': [],
    }

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print(f"{'=' * 70}")

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
        history['teacher_forcing_ratio'].append(
            train_metrics['teacher_forcing_ratio']
        )

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': best_val_loss,
                 'tf': train_metrics['teacher_forcing_ratio']},
                output_dir / "best_model.pth",
            )
            print(f"✓ Saved best model (val_loss: {best_val_loss:.6f})")

        # Save latest
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'val_loss': val_metrics['loss'],
             'tf': train_metrics['teacher_forcing_ratio']},
            output_dir / "latest_model.pth",
        )

        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()