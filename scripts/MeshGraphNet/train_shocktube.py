#!/usr/bin/env python3
"""
MeshGraphNet training for Shock Tube simulations.

Delta-based: model predicts change in dynamic state per timestep.
Data is pre-normalized (global-max). No z-score.
Handles skip indices (y_momentum) and global parameter injection.

Data format: x = [x_pos, y_pos, density, x_momentum, y_momentum, energy]
             static = x[:, :2], raw_dynamic = x[:, 2:6]
             skip y_momentum (idx 2 in raw_dynamic) -> used = [density, x_mom, energy] (3)
             y = full next [density, x_mom, y_mom, energy] -> target = y[:, keep] (3)
             delta = target - used_dynamic
             Global params: [pressure, density_param, delta_t] appended as node features
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

# Skip y_momentum
SKIP_INDICES = [2]
NUM_RAW_DYNAMIC = 4
KEEP_INDICES = [i for i in range(NUM_RAW_DYNAMIC) if i not in SKIP_INDICES]
NUM_USED_DYNAMIC = len(KEEP_INDICES)  # 3


def extract_dynamic(x, sf=2):
    raw = x[:, sf:sf + NUM_RAW_DYNAMIC]
    return raw[:, KEEP_INDICES]


def apply_skip(y):
    return y[:, KEEP_INDICES]


def extract_global_params(data):
    parts = []
    for names in [('global_pressure', 'pressure'),
                  ('global_density', 'density_param'),
                  ('global_delta_t', 'delta_t')]:
        val = None
        for attr in names:
            if hasattr(data, attr):
                val = getattr(data, attr)
                break
        if val is not None:
            if isinstance(val, (int, float)):
                parts.append(torch.tensor([val], device=data.x.device, dtype=data.x.dtype))
            else:
                parts.append(val.view(1).to(device=data.x.device, dtype=data.x.dtype))
        else:
            parts.append(torch.zeros(1, device=data.x.device, dtype=data.x.dtype))
    gp = torch.cat(parts)
    return gp.unsqueeze(0).expand(data.x.size(0), -1)


# =========================================================================
# DATASET
# =========================================================================

class ShockTubeDataset(torch.utils.data.IterableDataset):
    def __init__(self, directory, seq_len=10, stride=5, file_pattern="*.pt"):
        super().__init__()
        self.directory = Path(directory)
        self.seq_len = seq_len
        self.stride = stride
        self.sim_files = sorted(self.directory.glob(file_pattern))
        print(f"ShockTubeDataset: {len(self.sim_files)} sims from {directory}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self.sim_files
        else:
            n = len(self.sim_files)
            pw = int(math.ceil(n / worker_info.num_workers))
            s = worker_info.id * pw
            files = self.sim_files[s:min(s + pw, n)]

        for f in files:
            try:
                sim = torch.load(f, weights_only=False)
                if not isinstance(sim, list):
                    continue
                T = len(sim)
                for start in range(0, T - self.seq_len + 1, self.stride):
                    yield [sim[start + i].clone() for i in range(self.seq_len)]
            except Exception as e:
                print(f"Error loading {f}: {e}")

    def __len__(self):
        return len(self.sim_files) * 5


# =========================================================================
# TRAINING / VALIDATION
# =========================================================================

def train_epoch(model, loader, optimizer, device, sf, grad_clip=0.0):
    model.train()
    total_loss, n = 0.0, 0

    for seq in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        batch_loss = 0.0

        for data in seq:
            data = data.to(device)
            pos = data.pos if hasattr(data, 'pos') and data.pos is not None else data.x[:, :sf]

            static = data.x[:, :sf]
            dynamic = extract_dynamic(data.x, sf)
            gp = extract_global_params(data)
            node_features = torch.cat([static, dynamic, gp], dim=-1)

            edge_features = model.compute_edge_features(pos, data.edge_index)
            pred_delta = model(node_features, edge_features, data.edge_index)

            target_full = apply_skip(data.y)
            target_delta = target_full - dynamic
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
def validate(model, loader, device, sf):
    model.eval()
    total_loss, n = 0.0, 0

    for seq in tqdm(loader, desc="Validating"):
        batch_loss = 0.0
        for data in seq:
            data = data.to(device)
            pos = data.pos if hasattr(data, 'pos') and data.pos is not None else data.x[:, :sf]

            static = data.x[:, :sf]
            dynamic = extract_dynamic(data.x, sf)
            gp = extract_global_params(data)
            node_features = torch.cat([static, dynamic, gp], dim=-1)

            edge_features = model.compute_edge_features(pos, data.edge_index)
            pred_delta = model(node_features, edge_features, data.edge_index)

            target_full = apply_skip(data.y)
            target_delta = target_full - dynamic
            loss = F.mse_loss(pred_delta, target_delta)
            batch_loss += loss.item()

        total_loss += batch_loss / len(seq)
        n += 1

    return total_loss / max(n, 1)


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MeshGraphNet - Shock Tube')
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--file_pattern', type=str, default='*.pt')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--num_static_feats', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'step', 'cosine'])
    parser.add_argument('--scheduler_step', type=int, default=100)
    parser.add_argument('--scheduler_gamma', type=float, default=0.9)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_shocktube_mgn')
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=100)

    args = parser.parse_args()
    sf = args.num_static_feats
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    train_ds = ShockTubeDataset(args.train_dir, args.seq_len, args.stride, args.file_pattern)
    val_ds = ShockTubeDataset(args.val_dir, args.seq_len, args.stride, args.file_pattern)
    train_loader = DataLoader(train_ds, batch_size=None, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=args.num_workers)

    # Input: static(2) + used_dynamic(3) + global_params(3) = 8
    input_dim = sf + NUM_USED_DYNAMIC + 3
    model = MeshGraphNet(
        input_dim_node=input_dim, input_dim_edge=3,
        hidden_dim=args.hidden_dim, output_dim=NUM_USED_DYNAMIC,
        num_layers=args.num_layers
    ).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.scheduler == 'cosine' else
                 optim.lr_scheduler.StepLR(optimizer, args.scheduler_step, args.scheduler_gamma)
                 if args.scheduler == 'step' else None)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    best_val = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, sf, args.grad_clip_norm)

        if (epoch + 1) % args.val_every == 0:
            val_loss = validate(model, val_loader, device, sf)
            print(f"Epoch {epoch+1}/{args.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'train_loss': train_loss, 'val_loss': val_loss, 'args': vars(args),
                }, ckpt_dir / 'best_model.pt')
                print(f"  -> Best model (val: {val_loss:.6f})")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Train: {train_loss:.6f}")

        if scheduler:
            scheduler.step()
        if (epoch + 1) % args.save_every == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'args': vars(args)}, ckpt_dir / f'checkpoint_{epoch+1}.pt')

    print(f"\nDone. Best val: {best_val:.6f}")


if __name__ == "__main__":
    main()