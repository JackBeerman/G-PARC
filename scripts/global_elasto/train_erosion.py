#!/usr/bin/env python3
"""
Training Script for G-PARC Elastoplastic Model - WITH EROSION HEAD
===================================================================
Based on train.py. Additions:
  - ErosionHead created alongside displacement model
  - Separate LR for erosion head parameters
  - Focal loss for erosion alongside MSE for displacement
  - Autoregressive erosion feedback during rollout
  - --erosion_weight controls loss balance
  - --erosion_lr controls erosion head learning rate

Can resume from a displacement-only checkpoint (erosion head trains from scratch).
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
from differentiator.differentiator import ElastoPlasticDifferentiator
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from data.ElastoPlasticDataset import ElastoPlasticDataset, get_simulation_ids
from models.globalelasto import GPARC_ElastoPlastic_Numerical
from models.erosion_head import ErosionHead, FocalLoss, get_gt_erosion_targets


def load_normalization_stats(data_dir):
    stats_file = Path(data_dir).parent / "normalization_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"\n✓ Loaded normalization stats from: {stats_file}")
        return stats
    else:
        print(f"\n⚠️  No normalization_stats.json found, using z-score defaults")
        return {
            'normalization_method': 'z_score',
            'position': {
                'x_pos': {'mean': 97.2165, 'std': 59.3803},
                'y_pos': {'mean': 50.2759, 'std': 28.4965}
            }
        }


def get_pos_normalization_params(norm_stats):
    pos_stats = norm_stats['position']
    pos_mean = [pos_stats['x_pos']['mean'], pos_stats['y_pos']['mean']]
    pos_std = [pos_stats['x_pos']['std'], pos_stats['y_pos']['std']]
    return pos_mean, pos_std


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{' MODEL PARAMETERS ':~^50}")
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    
    # Breakdown if erosion head exists
    if hasattr(model, 'erosion_head') and model.erosion_head is not None:
        eh_params = sum(p.numel() for p in model.erosion_head.parameters())
        disp_params = total - eh_params
        print(f"  Displacement model: {disp_params:,}")
        print(f"  Erosion head:       {eh_params:,}")
    
    print(f"{'~'*50}\n")
    return trainable


def get_teacher_forcing_ratio(epoch, total_epochs, schedule='linear', initial_ratio=1.0, final_ratio=0.0):
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
    """Create G-PARC model with erosion head."""
    pos_mean, pos_std = get_pos_normalization_params(norm_stats)
    norm_method = norm_stats.get('normalization_method', 'z_score')
    max_position = None
    if norm_method == 'global_max' and 'position' in norm_stats:
        max_position = norm_stats['position'].get('max_position', None)

    gradient_solver = SolveGradientsLST(
        pos_mean=pos_mean, pos_std=pos_std,
        norm_method=norm_method, max_position=max_position
    )
    laplacian_solver = SolveWeightLST2d(
        pos_mean=pos_mean, pos_std=pos_std,
        norm_method=norm_method, max_position=max_position,
        min_neighbors=5
    )

    feature_extractor = GraphConvFeatureExtractorV2(
        in_channels=args.num_static_feats,
        hidden_channels=args.hidden_channels,
        out_channels=args.feature_out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layer_norm=args.use_layer_norm,
        use_relative_pos=args.use_relative_pos
    )

    derivative_solver = ElastoPlasticDifferentiator(
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        feature_extractor=feature_extractor,
        gradient_solver=gradient_solver,
        laplacian_solver=laplacian_solver,
        n_fe_features=args.feature_out_channels,
        list_strain_idx=list(range(args.num_dynamic_feats)),
        list_laplacian_idx=list(range(args.num_dynamic_feats)),
        spade_random_noise=args.spade_random_noise,
        heads=args.spade_heads,
        concat=args.spade_concat,
        dropout=args.spade_dropout,
        use_von_mises=args.use_von_mises,
        use_volumetric=args.use_volumetric,
        n_state_var=args.n_state_var,
        zero_init=args.zero_init
    )

    derivative_solver.initialize_weights(sample_data)

    # ---- Erosion Head ----
    # Cached features: resnet_out [N, feat_out] + explicit [N, n_explicit] + prev_erosion [N, 1]
    # Explicit features: 3 strain + von_mises + volumetric + 2 laplacian = 7
    n_explicit = 3 + int(args.use_von_mises) + int(args.use_volumetric)
    n_explicit += args.num_dynamic_feats  # Laplacian per component
    erosion_in_features = args.feature_out_channels + n_explicit + 1  # +1 for prev erosion
    
    print(f"\nCreating Erosion Head:")
    print(f"  Input features: {erosion_in_features} "
          f"({args.feature_out_channels} resnet + {n_explicit} physics + 1 prev_erosion)")
    print(f"  Hidden dim: {args.erosion_hidden_dim}")
    print(f"  MLP layers: {args.erosion_num_layers}")
    
    erosion_head = ErosionHead(
        in_features=erosion_in_features,
        hidden_dim=args.erosion_hidden_dim,
        num_layers=args.erosion_num_layers,
        dropout=args.erosion_dropout,
    )

    model = GPARC_ElastoPlastic_Numerical(
        derivative_solver_physics=derivative_solver,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        pos_mean=pos_mean,
        pos_std=pos_std,
        boundary_threshold=0.5,
        clamp_output=not args.no_clamp_output,
        norm_method=norm_method,
        max_position=max_position,
        erosion_head=erosion_head,
    )

    return model


def get_valid_node_mask(elements, current_erosion, next_erosion=None, device='cpu'):
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
    device = pred.device
    valid_mask = get_valid_node_mask(elements, current_erosion, next_erosion, device)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device), 0, pred.shape[0]
    loss = F.mse_loss(pred[valid_mask], target[valid_mask])
    return loss, valid_mask.sum().item(), (~valid_mask).sum().item()


# =========================================================================
# TRAINING
# =========================================================================

def train_epoch(model, train_loader, optimizer, device, epoch, total_epochs, args, focal_loss_fn):
    model.train()
    
    teacher_forcing_ratio = get_teacher_forcing_ratio(
        epoch=epoch, total_epochs=total_epochs,
        schedule=args.ss_schedule,
        initial_ratio=args.ss_initial_ratio,
        final_ratio=args.ss_final_ratio
    )
    
    total_disp_loss = 0.0
    total_erosion_loss = 0.0
    total_loss = 0.0
    n_batches = 0
    total_valid_nodes = 0
    total_eroded_nodes = 0
    total_erosion_tp = 0
    total_erosion_fp = 0
    total_erosion_fn = 0
    
    pbar = tqdm(train_loader, desc=f"Training (TF={teacher_forcing_ratio:.3f})")
    
    for seq in pbar:
        for data in seq:
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :model.num_static_feats]
        
        optimizer.zero_grad()
        
        # Forward returns (predictions, erosion_logits)
        predictions, erosion_logits_list = model(
            seq, dt=1.0, teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        # ---- Displacement loss ----
        disp_loss = 0.0
        total_weight = 0.0
        
        for t, (pred, data) in enumerate(zip(predictions, seq)):
            target = data.y
            
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
            
            weight = 1.0
            disp_loss += weight * step_loss
            total_weight += weight
        
        disp_loss = disp_loss / len(predictions)
        
        # ---- Erosion loss (focal) ----
        erosion_loss = torch.tensor(0.0, device=device)
        n_erosion_steps = 0
        
        for t, (logits, data) in enumerate(zip(erosion_logits_list, seq)):
            if hasattr(data, 'elements') and hasattr(data, 'x_element'):
                num_elements = data.elements.shape[0]
                
                # GT erosion for THIS timestep's prediction target
                # The model predicts erosion at the NEXT state, so target
                # should be the erosion status at the next timestep
                # But since x_element at time t already reflects the state,
                # and predictions[t] predicts the next displacement,
                # we use x_element from the next timestep if available
                if t + 1 < len(seq) and hasattr(seq[t + 1], 'x_element'):
                    targets = get_gt_erosion_targets(seq[t + 1], num_elements)
                else:
                    targets = get_gt_erosion_targets(data, num_elements)
                
                targets = targets.to(device)
                step_erosion_loss = focal_loss_fn(logits, targets)
                erosion_loss += step_erosion_loss
                n_erosion_steps += 1
                
                # Track erosion metrics
                with torch.no_grad():
                    pred_eroded = (torch.sigmoid(logits) > 0.5).squeeze(-1)
                    gt_eroded = targets.bool()
                    total_erosion_tp += (pred_eroded & gt_eroded).sum().item()
                    total_erosion_fp += (pred_eroded & ~gt_eroded).sum().item()
                    total_erosion_fn += (~pred_eroded & gt_eroded).sum().item()
        
        if n_erosion_steps > 0:
            erosion_loss = erosion_loss / n_erosion_steps
        
        # ---- Combined loss ----
        loss = disp_loss + args.erosion_weight * erosion_loss
        
        loss.backward()
        
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        
        optimizer.step()
        
        total_disp_loss += disp_loss.item()
        total_erosion_loss += erosion_loss.item()
        total_loss += loss.item()
        n_batches += 1
        
        # Erosion F1 for progress bar
        e_prec = total_erosion_tp / max(total_erosion_tp + total_erosion_fp, 1)
        e_rec = total_erosion_tp / max(total_erosion_tp + total_erosion_fn, 1)
        e_f1 = 2 * e_prec * e_rec / max(e_prec + e_rec, 1e-8)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.5f}",
            'disp': f"{disp_loss.item():.5f}",
            'ero': f"{erosion_loss.item():.4f}",
            'F1': f"{e_f1:.3f}",
        })
    
    return {
        'loss': total_loss / n_batches,
        'disp_loss': total_disp_loss / n_batches,
        'erosion_loss': total_erosion_loss / n_batches,
        'teacher_forcing_ratio': teacher_forcing_ratio,
        'valid_nodes': total_valid_nodes,
        'eroded_nodes': total_eroded_nodes,
        'erosion_tp': total_erosion_tp,
        'erosion_fp': total_erosion_fp,
        'erosion_fn': total_erosion_fn,
    }


@torch.no_grad()
def validate_epoch(model, val_loader, device, args, focal_loss_fn):
    model.eval()
    
    total_disp_loss = 0.0
    total_erosion_loss = 0.0
    total_loss = 0.0
    n_batches = 0
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for seq in tqdm(val_loader, desc="Validating"):
        for data in seq:
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :model.num_static_feats]
        
        predictions, erosion_logits_list = model(seq, dt=1.0, teacher_forcing_ratio=0.0)
        
        # Displacement loss
        disp_loss = 0.0
        for t, (pred, data) in enumerate(zip(predictions, seq)):
            target = data.y
            if args.mask_eroding and hasattr(data, 'elements') and hasattr(data, 'x_element'):
                step_loss, _, _ = compute_masked_loss(
                    pred, target, data.elements, data.x_element,
                    data.y_element if hasattr(data, 'y_element') else None
                )
            else:
                step_loss = F.mse_loss(pred, target)
            disp_loss += step_loss
        disp_loss = disp_loss / len(predictions)
        
        # Erosion loss
        erosion_loss = torch.tensor(0.0, device=device)
        n_erosion_steps = 0
        
        for t, (logits, data) in enumerate(zip(erosion_logits_list, seq)):
            if hasattr(data, 'elements') and hasattr(data, 'x_element'):
                num_elements = data.elements.shape[0]
                if t + 1 < len(seq) and hasattr(seq[t + 1], 'x_element'):
                    targets = get_gt_erosion_targets(seq[t + 1], num_elements)
                else:
                    targets = get_gt_erosion_targets(data, num_elements)
                targets = targets.to(device)
                erosion_loss += focal_loss_fn(logits, targets)
                n_erosion_steps += 1
                
                pred_eroded = (torch.sigmoid(logits) > 0.5).squeeze(-1)
                gt_eroded = targets.bool()
                total_tp += (pred_eroded & gt_eroded).sum().item()
                total_fp += (pred_eroded & ~gt_eroded).sum().item()
                total_fn += (~pred_eroded & gt_eroded).sum().item()
        
        if n_erosion_steps > 0:
            erosion_loss = erosion_loss / n_erosion_steps
        
        loss = disp_loss + args.erosion_weight * erosion_loss
        
        total_disp_loss += disp_loss.item()
        total_erosion_loss += erosion_loss.item()
        total_loss += loss.item()
        n_batches += 1
    
    e_prec = total_tp / max(total_tp + total_fp, 1)
    e_rec = total_tp / max(total_tp + total_fn, 1)
    e_f1 = 2 * e_prec * e_rec / max(e_prec + e_rec, 1e-8)
    
    return {
        'loss': total_loss / n_batches,
        'disp_loss': total_disp_loss / n_batches,
        'erosion_loss': total_erosion_loss / n_batches,
        'erosion_f1': e_f1,
        'erosion_precision': e_prec,
        'erosion_recall': e_rec,
    }


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
    parser = argparse.ArgumentParser(description="Train G-PARC with Erosion Head")
    
    # Dataset
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, default="*.pt")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=2)
    parser.add_argument("--n_state_var", type=int, default=0)
    
    # Physics
    parser.add_argument("--use_von_mises", action="store_true", default=True)
    parser.add_argument("--use_volumetric", action="store_true", default=True)
    
    # Feature Extractor
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--feature_out_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_layer_norm", action="store_true", default=True)
    parser.add_argument("--use_relative_pos", action="store_true", default=True)
    
    # Model / Integrator
    parser.add_argument("--integrator", type=str, default="euler")
    parser.add_argument("--no_clamp_output", action="store_true", default=True)
    
    # SPADE
    parser.add_argument("--spade_random_noise", action="store_true", default=False)
    parser.add_argument("--spade_heads", type=int, default=4)
    parser.add_argument("--spade_concat", action="store_true", default=True)
    parser.add_argument("--spade_dropout", type=float, default=0.1)
    parser.add_argument("--zero_init", action="store_true", default=True)
    
    # Loss
    parser.add_argument("--mask_eroding", action="store_true", default=True)
    parser.add_argument("--use_loss_decay", action="store_true", default=False)
    parser.add_argument("--loss_decay_gamma", type=float, default=0.9)
    
    # Scheduled Sampling
    parser.add_argument("--ss_schedule", type=str, default="linear")
    parser.add_argument("--ss_initial_ratio", type=float, default=0.0)
    parser.add_argument("--ss_final_ratio", type=float, default=0.0)
    
    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    
    # ---- EROSION HEAD ----
    parser.add_argument("--erosion_hidden_dim", type=int, default=64)
    parser.add_argument("--erosion_num_layers", type=int, default=2)
    parser.add_argument("--erosion_dropout", type=float, default=0.1)
    parser.add_argument("--erosion_weight", type=float, default=1.0,
                        help="Weight for erosion focal loss relative to displacement MSE")
    parser.add_argument("--erosion_lr", type=float, default=1e-3,
                        help="Learning rate for erosion head (separate from displacement model)")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./outputs_erosion")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (displacement-only or full)")
    parser.add_argument("--resume_displacement_only", action="store_true", default=False,
                        help="Load only displacement weights from checkpoint (erosion head trains from scratch)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    norm_stats = load_normalization_stats(args.train_dir)
    
    print("\n" + "=" * 70)
    print("G-PARC TRAINING WITH EROSION HEAD")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"")
    print(f"Displacement model: lr={args.lr}")
    print(f"Erosion head: lr={args.erosion_lr}, weight={args.erosion_weight}")
    print(f"Focal loss: alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    print("=" * 70)
    
    # Dataset
    train_ids = get_simulation_ids(Path(args.train_dir), pattern=args.file_pattern)
    val_ids = get_simulation_ids(Path(args.val_dir), pattern=args.file_pattern)
    print(f"\n{len(train_ids)} train, {len(val_ids)} val simulations")
    
    train_dataset = ElastoPlasticDataset(
        directory=Path(args.train_dir), simulation_ids=train_ids,
        seq_len=args.seq_len, stride=args.stride,
        num_static_feats=args.num_static_feats, num_dynamic_feats=args.num_dynamic_feats
    )
    val_dataset = ElastoPlasticDataset(
        directory=Path(args.val_dir), simulation_ids=val_ids,
        seq_len=args.seq_len, stride=args.stride,
        num_static_feats=args.num_static_feats, num_dynamic_feats=args.num_dynamic_feats
    )
    
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=args.num_workers, pin_memory=True)
    
    # Sample data for initialization
    sample_data = next(iter(train_loader))[0]
    print(f"  Sample: {sample_data.num_nodes} nodes, {sample_data.edge_index.shape[1]} edges")
    
    # Create model with erosion head
    model = create_model(args, sample_data, norm_stats).to(device)
    count_parameters(model)
    
    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        if args.resume_displacement_only:
            # Load displacement model weights only, skip erosion head
            state_dict = checkpoint['model_state_dict']
            model_dict = model.state_dict()
            
            # Filter out erosion_head keys
            pretrained = {k: v for k, v in state_dict.items() if not k.startswith('erosion_head')}
            
            # Also handle cache_features flag not existing in old checkpoint
            missing, unexpected = model.load_state_dict(pretrained, strict=False)
            print(f"  Loaded displacement weights (skipped erosion head)")
            print(f"  Missing keys (expected): {[k for k in missing if 'erosion_head' in k][:5]}...")
            
            start_epoch = 0  # Train erosion head from scratch
            best_val_loss = float('inf')
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['metrics'].get('val_loss', float('inf'))
            print(f"  Full resume from epoch {start_epoch}")
    
    # ---- Optimizer with separate LR for erosion head ----
    erosion_params = list(model.erosion_head.parameters())
    erosion_param_ids = set(id(p) for p in erosion_params)
    displacement_params = [p for p in model.parameters() if id(p) not in erosion_param_ids]
    
    optimizer = AdamW([
        {'params': displacement_params, 'lr': args.lr},
        {'params': erosion_params, 'lr': args.erosion_lr},
    ])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=min(args.lr, args.erosion_lr) * 0.01)
    
    # Focal loss
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    
    # Save config
    config = vars(args)
    config['feature_extractor'] = 'GraphConv V2'
    config['erosion_head'] = True
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    if (Path(args.train_dir).parent / "normalization_stats.json").exists():
        import shutil
        shutil.copy2(Path(args.train_dir).parent / "normalization_stats.json",
                      output_dir / "normalization_stats.json")
    
    # ---- Training loop ----
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    history = {
        'train_loss': [], 'train_disp_loss': [], 'train_erosion_loss': [],
        'val_loss': [], 'val_disp_loss': [], 'val_erosion_loss': [],
        'val_erosion_f1': [],
    }
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print(f"{'=' * 70}")
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch, total_epochs=args.epochs,
            args=args, focal_loss_fn=focal_loss_fn
        )
        
        val_metrics = validate_epoch(model, val_loader, device, args, focal_loss_fn)
        
        scheduler.step()
        
        # Compute train erosion F1
        tp, fp, fn = train_metrics['erosion_tp'], train_metrics['erosion_fp'], train_metrics['erosion_fn']
        t_prec = tp / max(tp + fp, 1)
        t_rec = tp / max(tp + fn, 1)
        t_f1 = 2 * t_prec * t_rec / max(t_prec + t_rec, 1e-8)
        
        print(f"\nTrain — total: {train_metrics['loss']:.6f}, "
              f"disp: {train_metrics['disp_loss']:.6f}, "
              f"erosion: {train_metrics['erosion_loss']:.5f}, "
              f"F1: {t_f1:.3f}")
        print(f"Val   — total: {val_metrics['loss']:.6f}, "
              f"disp: {val_metrics['disp_loss']:.6f}, "
              f"erosion: {val_metrics['erosion_loss']:.5f}, "
              f"F1: {val_metrics['erosion_f1']:.3f} "
              f"(P={val_metrics['erosion_precision']:.3f}, R={val_metrics['erosion_recall']:.3f})")
        
        # History
        history['train_loss'].append(train_metrics['loss'])
        history['train_disp_loss'].append(train_metrics['disp_loss'])
        history['train_erosion_loss'].append(train_metrics['erosion_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_disp_loss'].append(val_metrics['disp_loss'])
        history['val_erosion_loss'].append(val_metrics['erosion_loss'])
        history['val_erosion_f1'].append(val_metrics['erosion_f1'])
        
        # Save best (based on combined loss)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': best_val_loss, 'erosion_f1': val_metrics['erosion_f1']},
                output_dir / "best_model.pth"
            )
            print(f"✓ Saved best model (val_loss: {best_val_loss:.6f}, "
                  f"erosion_F1: {val_metrics['erosion_f1']:.3f})")
        
        # Also save best erosion F1 separately
        best_f1 = max(history['val_erosion_f1']) if history['val_erosion_f1'] else 0
        if val_metrics['erosion_f1'] >= best_f1 and val_metrics['erosion_f1'] > 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': val_metrics['loss'], 'erosion_f1': val_metrics['erosion_f1']},
                output_dir / "best_erosion_model.pth"
            )
            print(f"✓ Saved best erosion model (F1: {val_metrics['erosion_f1']:.3f})")
        
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'val_loss': val_metrics['loss'], 'erosion_f1': val_metrics['erosion_f1']},
            output_dir / "latest_model.pth"
        )
        
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best erosion F1: {max(history['val_erosion_f1']) if history['val_erosion_f1'] else 0:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()