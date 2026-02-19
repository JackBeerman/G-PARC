#!/usr/bin/env python3
"""
MeshGraphNet training for Elastoplastic dynamics.

Delta-based: model predicts change in displacement per timestep.
Data is pre-normalized (global-max). No runtime z-score normalization.
Independent single-step training (no autoregressive rollout during training).

Data format: x = [x_pos(1), y_pos(1), U_x(1), U_y(1)]  (sf=2, df=2)
             y = full next-step displacement [U_x_next, U_y_next] (2 features)
             delta = y - x[:, 2:4]
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import math

from meshgraphnet import MeshGraphNet
from dataset import ElastoPlasticDataset, create_datasets_from_folders


# =========================================================================
# TRAINING / VALIDATION
# =========================================================================

def train_epoch(model, loader, optimizer, device, sf, df, grad_clip=0.0):
    model.train()
    total_loss, n = 0.0, 0

    for seq in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        batch_loss = 0.0

        for data in seq:
            data = data.to(device)
            pos = data.pos if hasattr(data, 'pos') and data.pos is not None else data.x[:, :sf]

            node_features = data.x  # [N, sf+df] already normalized
            edge_features = model.compute_edge_features(pos, data.edge_index)

            pred_delta = model(node_features, edge_features, data.edge_index)

            # Delta target: how much displacement changes
            current_dynamic = data.x[:, sf:sf + df]
            target_delta = data.y - current_dynamic

            loss = F.mse_loss(pred_delta, target_delta)
            loss.backward()
            batch_loss += loss.item()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += batch_loss / len(seq)
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, sf, df):
    model.eval()
    total_loss, n = 0.0, 0

    for seq in tqdm(loader, desc="Validating"):
        batch_loss = 0.0
        for data in seq:
            data = data.to(device)
            pos = data.pos if hasattr(data, 'pos') and data.pos is not None else data.x[:, :sf]

            node_features = data.x
            edge_features = model.compute_edge_features(pos, data.edge_index)
            pred_delta = model(node_features, edge_features, data.edge_index)

            current_dynamic = data.x[:, sf:sf + df]
            target_delta = data.y - current_dynamic
            loss = F.mse_loss(pred_delta, target_delta)
            batch_loss += loss.item()

        total_loss += batch_loss / len(seq)
        n += 1

    return total_loss / max(n, 1)


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MeshGraphNet - Elastoplastic')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base dir with train/val/test folders of .pt files')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--num_static_feats', type=int, default=2)
    parser.add_argument('--num_dynamic_feats', type=int, default=2)

    # Model
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=10)

    # Training
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'step', 'cosine'])
    parser.add_argument('--scheduler_step', type=int, default=100)
    parser.add_argument('--scheduler_gamma', type=float, default=0.9)

    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_elasto_mgn')
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=100)

    args = parser.parse_args()
    sf = args.num_static_feats
    df = args.num_dynamic_feats
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Device: {device}")

    # Datasets
    datasets = create_datasets_from_folders(
        base_dir=args.data_dir, seq_len=args.seq_len, stride=args.stride,
        num_static_feats=sf, num_dynamic_feats=df, use_element_features=False
    )
    train_loader = DataLoader(datasets['train'], batch_size=None, num_workers=args.num_workers)
    val_loader = DataLoader(datasets['val'], batch_size=None, num_workers=args.num_workers)

    # Model: input = sf + df node features, 3 edge features, output = df (delta)
    model = MeshGraphNet(
        input_dim_node=sf + df,
        input_dim_edge=3,
        hidden_dim=args.hidden_dim,
        output_dim=df,
        num_layers=args.num_layers
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.scheduler_step, args.scheduler_gamma)
    else:
        scheduler = None

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    best_val = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, sf, df, args.grad_clip_norm)

        if (epoch + 1) % args.val_every == 0:
            val_loss = validate(model, val_loader, device, sf, df)
            print(f"Epoch {epoch+1}/{args.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss, 'val_loss': val_loss, 'args': vars(args),
                }, ckpt_dir / 'best_model.pt')
                print(f"  -> Best model saved (val: {val_loss:.6f})")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Train: {train_loss:.6f}")

        if scheduler:
            scheduler.step()

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(), 'args': vars(args),
            }, ckpt_dir / f'checkpoint_epoch_{epoch+1}.pt')

    print(f"\nDone. Best val: {best_val:.6f}")


if __name__ == "__main__":
    main()