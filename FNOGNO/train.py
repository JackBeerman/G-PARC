import os
import math
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Iterator, Dict
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch_geometric.data import Data
from tqdm import tqdm
from neuralop.models import FNOGNO
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import argparse

# ==========================================
# 1) Dataset Definition
# ==========================================

class ElastoPlasticDataset(IterableDataset):
    def __init__(self,
                 directory: Union[str, Path],
                 simulation_ids: List[str] = None,
                 seq_len: int = 10,
                 stride: int = 1,
                 num_static_feats: int = 2,
                 num_dynamic_feats: int = 2,
                 file_pattern: str = "*.pt",
                 use_element_features: bool = False):
        super().__init__()
        self.directory = Path(directory)
        self.seq_len = seq_len
        self.stride = stride
        self.file_pattern = file_pattern
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.use_element_features = use_element_features

        if simulation_ids is None:
            self.simulation_ids = self._discover_simulation_ids()
        else:
            self.simulation_ids = simulation_ids

        self.var_names = ['U_x', 'U_y']
        if self.use_element_features:
            self.element_var_names = ['EROSION_STATUS']

        print(f"Dataset: {self.directory} | {len(self.simulation_ids)} sims")

    def _discover_simulation_ids(self) -> List[str]:
        files = list(self.directory.glob(self.file_pattern))
        return [file.stem for file in files]

    def _extract_id_from_name(self, sim_name: str) -> int:
        import re
        match = re.search(r'\d+', sim_name)
        return int(match.group()) if match else abs(hash(sim_name)) % 100000

    def __iter__(self) -> Iterator[List[Data]]:
        worker_info = get_worker_info()
        if worker_info is None:
            files_to_process = self.simulation_ids
        else:
            total_files = len(self.simulation_ids)
            per_worker = int(math.ceil(total_files / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, total_files)
            files_to_process = self.simulation_ids[start:end]

        for sim_name in files_to_process:
            dataset_file = self.directory / f"{sim_name}.pt"
            sim_id_int = self._extract_id_from_name(sim_name)
            try:
                sim_data = torch.load(dataset_file, weights_only=False)
                if not isinstance(sim_data, list): continue

                for data in sim_data:
                    data.mesh_id = torch.tensor([sim_id_int], dtype=torch.long)

                T = len(sim_data)
                max_start = T - self.seq_len
                if max_start < 0: continue

                for start_idx in range(0, max_start + 1, self.stride):
                    window = []
                    for offset in range(self.seq_len):
                        t = start_idx + offset
                        data_t = sim_data[t].clone()
                        if not hasattr(data_t, 'x') or not hasattr(data_t, 'y'): break
                        window.append(data_t)
                    if len(window) == self.seq_len:
                        yield window
            except Exception as e:
                print(f"Error loading {dataset_file}: {e}")

# ==========================================
# 2) Optimized Helper Utilities
# ==========================================

def collate_windows(batch):
    return batch

@torch.no_grad()
def get_node_coords(data_t: Data, num_static_feats: int = 2):
    if hasattr(data_t, "pos") and data_t.pos is not None:
        return data_t.pos
    return data_t.x[:, :num_static_feats]

def get_dyn_state(data_t: Data, num_static_feats: int = 2, num_dynamic_feats: int = 2):
    return data_t.x[:, num_static_feats:num_static_feats + num_dynamic_feats]

def get_next_dyn_target(data_t: Data, num_dynamic_feats: int = 2):
    return data_t.y[:, :num_dynamic_feats] if data_t.y.shape[-1] >= num_dynamic_feats else data_t.y

def make_regular_grid_points(h: int, w: int, device: str):
    ys = torch.linspace(0.0, 1.0, steps=h, device=device)
    xs = torch.linspace(0.0, 1.0, steps=w, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([X, Y], dim=-1)

class RasterizationBuffers:
    """Reusable buffers for rasterization to avoid memory allocation overhead"""
    def __init__(self, h: int, w: int, num_channels: int, device: str):
        self.h = h
        self.w = w
        self.HW = h * w
        self.f_sum = torch.zeros(self.HW, num_channels, device=device, dtype=torch.float32)
        self.cnt = torch.zeros(self.HW, 1, device=device, dtype=torch.float32)
        self.ones_cache: Dict[int, torch.Tensor] = {}
        
    def rasterize(self, points_xy, values):
        """Fast rasterization using pre-allocated buffers"""
        xy = points_xy.clamp(0.0, 1.0)
        ix = torch.floor(xy[:, 0] * (self.w - 1)).long()
        iy = torch.floor(xy[:, 1] * (self.h - 1)).long()
        idx = iy * self.w + ix
        
        N = values.shape[0]
        
        self.f_sum.zero_()
        self.cnt.zero_()
        
        if N not in self.ones_cache:
            self.ones_cache[N] = torch.ones(N, 1, device=values.device, dtype=values.dtype)
        
        self.f_sum.index_add_(0, idx, values)
        self.cnt.index_add_(0, idx, self.ones_cache[N])
        
        self.cnt.clamp_(min=1.0)
        return (self.f_sum / self.cnt).view(self.h, self.w, -1)

def window_to_samples(window, device, num_static=2, num_dyn=2):
    samples = []
    for data_t in window:
        coords = get_node_coords(data_t, num_static).to(device, non_blocking=True).float()
        dyn = get_dyn_state(data_t, num_static, num_dyn).to(device, non_blocking=True).float()
        tgt = get_next_dyn_target(data_t, num_dyn).to(device, non_blocking=True).float()
        samples.append((coords, dyn, tgt))
    return samples

def count_total_windows(loader):
    """Count total number of windows in dataset"""
    total = 0
    for window in loader:
        total += 1  # Count windows, not samples
    return total

def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{' MODEL PARAMETERS ':~^50}")
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"{'~'*50}\n")
    return trainable

# ==========================================
# 3) Training Logic
# ==========================================

def collate_windows(batch):
    """Collate function for DataLoader - returns first item since batch_size=None"""
    # With batch_size=None, batch is a single window (list of Data objects)
    # Just return it as-is
    return [batch] if not isinstance(batch[0], list) else batch


def train_epoch(loader, model, optimizer, loss_fn, device, grid_h, grid_w, grid_p, 
                num_static, num_dyn, total_samples=None, grad_clip_norm=1.0):
    """Train for one epoch"""
    model.train()
    total_loss, n_samples = 0.0, 0
    
    raster_buffers = RasterizationBuffers(grid_h, grid_w, num_dyn, device)
    
    if total_samples is not None:
        pbar = tqdm(loader, total=total_samples, desc="Training", 
                   bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    else:
        pbar = tqdm(loader, desc="Training", bar_format='{desc}: {n_fmt} [{elapsed}, {rate_fmt}] {postfix}')

    for window in pbar:
        samples = window_to_samples(window, device, num_static, num_dyn)
        
        # CHANGED: Accumulate loss over sequence
        loss = 0.0
        for (coords, dyn, tgt) in samples:
            f_grid = raster_buffers.rasterize(coords, dyn)
            pred = model(grid_p, coords, f_grid)
            if pred.ndim == 3: 
                pred = pred.squeeze(0)
            loss += loss_fn(pred, tgt)
        
        # CHANGED: Average loss
        loss = loss / len(samples)
        
        # CHANGED: Single optimizer step per window
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optimizer.step()

        total_loss += loss.item()
        n_samples += 1
        
        pbar.update(1)
        pbar.set_postfix({"loss": f"{total_loss / n_samples:.6e}"})
    
    pbar.close()
    return {'loss': total_loss / max(n_samples, 1)}

@torch.no_grad()
def validate_epoch(loader, model, loss_fn, device, grid_h, grid_w, grid_p, 
                   num_static, num_dyn, total_samples=None):
    """Validate for one epoch"""
    model.eval()
    total_loss, n_samples = 0.0, 0
    
    raster_buffers = RasterizationBuffers(grid_h, grid_w, num_dyn, device)
    
    if total_samples is not None:
        pbar = tqdm(loader, total=total_samples, desc="Validating", 
                   bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    else:
        pbar = tqdm(loader, desc="Validating", bar_format='{desc}: {n_fmt} [{elapsed}, {rate_fmt}] {postfix}')

    for window in pbar:
        samples = window_to_samples(window, device, num_static, num_dyn)
        
        # CHANGED: Accumulate loss over sequence
        loss = 0.0
        for (coords, dyn, tgt) in samples:
            f_grid = raster_buffers.rasterize(coords, dyn)
            pred = model(grid_p, coords, f_grid)
            if pred.ndim == 3:
                pred = pred.squeeze(0)
            loss += loss_fn(pred, tgt)
        
        # CHANGED: Average loss
        loss = loss / len(samples)
        
        total_loss += loss.item()
        n_samples += 1
        
        pbar.update(1)
        pbar.set_postfix({"loss": f"{total_loss / n_samples:.6e}"})
    
    pbar.close()
    return {'loss': total_loss / max(n_samples, 1)}

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)

# ==========================================
# 4) Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train FNOGNO")
    
    # Dataset paths
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, default="*.pt")
    
    # Dataset configuration
    parser.add_argument("--seq_len", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=2)
    
    # Model configuration
    parser.add_argument("--grid_h", type=int, default=32)
    parser.add_argument("--grid_w", type=int, default=64)
    parser.add_argument("--fno_n_modes", type=int, nargs=2, default=[8, 16])
    parser.add_argument("--fno_hidden_channels", type=int, default=32)
    parser.add_argument("--fno_n_layers", type=int, default=3)
    parser.add_argument("--gno_radius", type=float, default=0.05)
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    
    # Scheduler
    parser.add_argument("--use_scheduler", action="store_true", default=True)
    parser.add_argument("--no_scheduler", action="store_false", dest="use_scheduler")
    
    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./outputs_fnogno")
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    # CUDA Optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("FNOGNO TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"\nModel Configuration:")
    print(f"  Grid: {args.grid_h}x{args.grid_w}")
    print(f"  FNO modes: {args.fno_n_modes}")
    print(f"  FNO hidden: {args.fno_hidden_channels}")
    print(f"  FNO layers: {args.fno_n_layers}")
    print(f"  GNO radius: {args.gno_radius}")
    print("="*70)
    
    # Setup data loaders - FIX HERE
    print(f"\nLoading datasets...")
    print(f"  Train: {args.train_dir}")
    print(f"  Val: {args.val_dir}")
    
    train_dataset = ElastoPlasticDataset(
        Path(args.train_dir), 
        seq_len=args.seq_len, 
        stride=args.stride,
        num_static_feats=args.num_static_feats, 
        num_dynamic_feats=args.num_dynamic_feats,
        file_pattern=args.file_pattern
    )
    
    val_dataset = ElastoPlasticDataset(
        Path(args.val_dir), 
        seq_len=args.seq_len, 
        stride=args.stride,
        num_static_feats=args.num_static_feats, 
        num_dynamic_feats=args.num_dynamic_feats,
        file_pattern=args.file_pattern
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=None,  # Important for IterableDataset
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=None,  # Important for IterableDataset
        num_workers=0,  # Keep 0 for validation
        pin_memory=True
    )
    
    # Count samples
    print("\nCounting total samples...")
    train_total = count_total_windows(train_loader)
    val_total = count_total_windows(val_loader)
    print(f"Train samples: {train_total}")
    print(f"Val samples: {val_total}")


    # ✅ FIX: re-create loaders after exhausting them during counting
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=True
    )

    
    # Create model
    print("\nCreating model...")
    in_p_grid = make_regular_grid_points(args.grid_h, args.grid_w, device)
    model = FNOGNO(
        in_channels=args.num_dynamic_feats,
        out_channels=args.num_dynamic_feats,
        fno_n_modes=tuple(args.fno_n_modes),
        fno_hidden_channels=args.fno_hidden_channels,
        fno_n_layers=args.fno_n_layers,
        gno_coord_dim=2,
        gno_radius=args.gno_radius,
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
    ).to(device)
    
    count_parameters(model)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01) if args.use_scheduler else None
    loss_fn = torch.nn.MSELoss()
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['metrics'].get('val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6e}")
    
    # Save config
    config = vars(args)
    config['model'] = 'FNOGNO'
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gradient clipping: {args.grad_clip_norm}")
    print(f"  Scheduler: {'CosineAnnealing' if args.use_scheduler else 'None'}")
    print("="*70)
    
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        # Train
        train_metrics = train_epoch(
            train_loader, model, optimizer, loss_fn, device,
            args.grid_h, args.grid_w, in_p_grid,
            args.num_static_feats, args.num_dynamic_feats,
            train_total, args.grad_clip_norm
        )
        
        # Validate
        val_metrics = validate_epoch(
            val_loader, model, loss_fn, device,
            args.grid_h, args.grid_w, in_p_grid,
            args.num_static_feats, args.num_dynamic_feats,
            val_total
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nTrain Loss: {train_metrics['loss']:.6e}")
        print(f"Val Loss:   {val_metrics['loss']:.6e}")
        print(f"LR:         {current_lr:.6e}")
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['lr'].append(current_lr)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': best_val_loss},
                output_dir / "best_model.pth"
            )
            print(f"✓ Saved best model (val_loss: {best_val_loss:.6e})")
        
        # Save latest model
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'val_loss': val_metrics['loss']},
            output_dir / "latest_model.pth"
        )
        
        # Save history
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best val loss: {best_val_loss:.6e}")
    print(f"Best model: {output_dir / 'best_model.pth'}")
    print(f"Latest model: {output_dir / 'latest_model.pth'}")
    print("="*70)


if __name__ == "__main__":
    main()