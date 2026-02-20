#!/usr/bin/env python3
"""
Training Script for G-PARCv3 Elastoplastic — Erosion-Aware
============================================================
Two-phase training:
  Phase 1: Load v2 displacement weights, freeze them.
           Train erosion head + erosion SPADE channel only.
  Phase 2: Unfreeze all. Joint training with separate LR groups.

Key changes from v2:
  - No loss masking — trains on ALL nodes
  - Uses ElastoPlasticDifferentiatorV3 with erosion-masked MLS
  - GPARC_ElastoPlastic_V3 with bidirectional erosion coupling
  - Erosion state fed as input to displacement model
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
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.featureextractor import GraphConvFeatureExtractorV2
from differentiator.differentiator_v3 import ElastoPlasticDifferentiatorV3
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d
from data.ElastoPlasticDataset import ElastoPlasticDataset, get_simulation_ids
from models.globalelasto_v3 import GPARC_ElastoPlastic_V3
from models.erosion_head import ErosionHead, FocalLoss, get_gt_erosion_targets


# ===========================================================================
# UTILITIES
# ===========================================================================

def load_normalization_stats(data_dir):
    stats_file = Path(data_dir).parent / "normalization_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"✓ Loaded normalization stats from: {stats_file}")
        return stats
    print(f"⚠️  normalization_stats.json not found at {stats_file}")
    return {}


def get_pos_normalization_params(norm_stats):
    if not norm_stats:
        return None, None
    pos_stats = norm_stats.get('position', {})
    method = norm_stats.get('normalization_method', 'z_score')
    if method == 'z_score':
        return pos_stats.get('mean'), pos_stats.get('std')
    return None, None


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'MODEL PARAMETERS':~^50}")
    print(f"  Total:     {total:>12,}")
    print(f"  Trainable: {trainable:>12,}")
    if hasattr(model, 'erosion_head') and model.erosion_head is not None:
        eh = sum(p.numel() for p in model.erosion_head.parameters())
        print(f"  Erosion head: {eh:>9,}")
    disp = total - (eh if hasattr(model, 'erosion_head') and model.erosion_head else 0)
    print(f"  Displacement: {disp:>9,}")


def get_teacher_forcing_ratio(epoch, total_epochs, schedule='linear',
                               initial_ratio=1.0, final_ratio=0.0):
    if schedule == 'linear':
        ratio = initial_ratio - (initial_ratio - final_ratio) * (epoch / max(total_epochs, 1))
    elif schedule == 'exponential':
        if initial_ratio > 0:
            decay = (max(final_ratio, 1e-6) / initial_ratio) ** (1 / max(total_epochs, 1))
            ratio = initial_ratio * (decay ** epoch)
        else:
            ratio = 0.0
    else:
        ratio = initial_ratio
    return max(final_ratio, min(initial_ratio, ratio))


# ===========================================================================
# MODEL CREATION
# ===========================================================================

def create_model(args, sample_data, norm_stats):
    """Create G-PARCv3 model with erosion-aware differentiator."""
    pos_mean, pos_std = get_pos_normalization_params(norm_stats)
    norm_method = norm_stats.get('normalization_method', 'z_score')
    max_position = None
    if norm_method == 'global_max' and 'position' in norm_stats:
        max_position = norm_stats['position'].get('max_position')

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

    # V3 differentiator — erosion-aware
    derivative_solver = ElastoPlasticDifferentiatorV3(
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
    # Cached features: resnet_out [N, feat_out] + explicit [N, n_explicit+1] + prev_erosion [N, 1]
    # V3 explicit: strain(5) + laplacian(2) + erosion_state(1) = 8
    n_explicit = 3 + int(args.use_von_mises) + int(args.use_volumetric)
    n_explicit += args.num_dynamic_feats  # Laplacian per component
    n_explicit += 1  # erosion SPADE channel

    # Erosion head input: resnet_out + explicit + prev_erosion
    erosion_in_features = args.feature_out_channels + n_explicit + 1  # +1 for prev erosion fed to head

    print(f"\nCreating Erosion Head:")
    print(f"  Input features: {erosion_in_features} "
          f"({args.feature_out_channels} resnet + {n_explicit} physics+erosion + 1 prev_erosion)")
    print(f"  Hidden dim: {args.erosion_hidden_dim}")
    print(f"  MLP layers: {args.erosion_num_layers}")

    erosion_head = ErosionHead(
        in_features=erosion_in_features,
        hidden_dim=args.erosion_hidden_dim,
        num_layers=args.erosion_num_layers,
        dropout=args.erosion_dropout,
    )

    model = GPARC_ElastoPlastic_V3(
        derivative_solver=derivative_solver,
        erosion_head=erosion_head,
        integrator_type=args.integrator,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        pos_mean=pos_mean,
        pos_std=pos_std,
        boundary_threshold=0.5,
        clamp_output=not args.no_clamp_output,
        norm_method=norm_method,
        max_position=max_position,
        erosion_threshold=args.erosion_threshold,
    )

    return model


def load_v2_weights(model, v2_checkpoint_path, device):
    """
    Load V2 displacement weights into V3 model.
    
    Handles:
      - Missing erosion_head keys (expected — trains from scratch)
      - Resized displacement MAR (V3 has +1 erosion channel)
      - Missing differentiator V3 keys (edge_erosion_cache etc.)
    """
    print(f"\nLoading V2 weights from: {v2_checkpoint_path}")
    ckpt = torch.load(v2_checkpoint_path, map_location=device, weights_only=False)
    v2_sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    # Map v2 keys to v3
    # V2: derivative_solver_physics → V3: derivative_solver
    v3_sd = model.state_dict()
    loaded, skipped = 0, 0

    for v2_key, v2_param in v2_sd.items():
        # Remap v2 key prefix
        v3_key = v2_key.replace('derivative_solver_physics', 'derivative_solver')

        if v3_key in v3_sd:
            if v2_param.shape == v3_sd[v3_key].shape:
                v3_sd[v3_key] = v2_param
                loaded += 1
            else:
                # Shape mismatch — partial load (e.g., resized SPADE linear)
                if len(v2_param.shape) == 2 and len(v3_sd[v3_key].shape) == 2:
                    out_dim = min(v2_param.shape[0], v3_sd[v3_key].shape[0])
                    in_dim = min(v2_param.shape[1], v3_sd[v3_key].shape[1])
                    v3_sd[v3_key][:out_dim, :in_dim] = v2_param[:out_dim, :in_dim]
                    loaded += 1
                    print(f"  Partial load: {v3_key} ({v2_param.shape} → {v3_sd[v3_key].shape})")
                elif len(v2_param.shape) == 1 and len(v3_sd[v3_key].shape) == 1:
                    min_dim = min(v2_param.shape[0], v3_sd[v3_key].shape[0])
                    v3_sd[v3_key][:min_dim] = v2_param[:min_dim]
                    loaded += 1
                else:
                    skipped += 1
        else:
            skipped += 1

    model.load_state_dict(v3_sd)
    print(f"  ✓ Loaded {loaded} parameters, skipped {skipped}")
    if 'epoch' in ckpt:
        print(f"  V2 checkpoint was at epoch {ckpt['epoch']}")
    return ckpt.get('epoch', 0)


# ===========================================================================
# TRAINING
# ===========================================================================

def freeze_displacement(model):
    """Freeze all parameters except erosion head and new SPADE erosion channel."""
    erosion_param_ids = set(id(p) for p in model.erosion_head.parameters())
    # Also keep the displacement MAR trainable (it has the new erosion channel)
    mar_param_ids = set(id(p) for p in model.derivative_solver.list_mar[-1].parameters())

    for p in model.parameters():
        if id(p) not in erosion_param_ids and id(p) not in mar_param_ids:
            p.requires_grad = False

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Phase 1 freeze: {frozen} frozen, {trainable} trainable")


def unfreeze_all(model):
    """Unfreeze everything for Phase 2."""
    for p in model.parameters():
        p.requires_grad = True
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Phase 2 unfreeze: {trainable} trainable")


def build_optimizer(model, args, phase):
    """Build optimizer with appropriate LR groups for the current phase."""
    erosion_params = list(model.erosion_head.parameters())
    erosion_ids = set(id(p) for p in erosion_params)

    mar_params = list(model.derivative_solver.list_mar[-1].parameters())
    mar_ids = set(id(p) for p in mar_params)

    other_params = [p for p in model.parameters()
                    if id(p) not in erosion_ids and id(p) not in mar_ids
                    and p.requires_grad]

    if phase == 1:
        # Only erosion head + MAR (displacement frozen)
        groups = [
            {'params': erosion_params, 'lr': args.erosion_lr},
            {'params': [p for p in mar_params if p.requires_grad], 'lr': args.erosion_lr},
        ]
    else:
        # Phase 2: all trainable with separate LRs
        groups = [
            {'params': other_params, 'lr': args.lr},
            {'params': erosion_params, 'lr': args.erosion_lr},
            {'params': mar_params, 'lr': args.lr},
        ]

    return AdamW(groups, weight_decay=args.weight_decay)


def train_epoch(model, train_loader, optimizer, device, epoch, total_epochs,
                args, focal_loss_fn):
    model.train()

    tf_ratio = get_teacher_forcing_ratio(
        epoch, total_epochs, args.ss_schedule,
        args.ss_initial_ratio, args.ss_final_ratio
    )

    total_disp_loss = 0.0
    total_erosion_loss = 0.0
    total_loss = 0.0
    n_batches = 0
    total_tp, total_fp, total_fn = 0, 0, 0

    pbar = tqdm(train_loader, desc=f"Train (TF={tf_ratio:.2f})")

    for seq in pbar:
        # Move to device and set pos
        for data in seq:
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            if not hasattr(data, 'pos') or data.pos is None:
                data.pos = data.x[:, :model.num_static_feats]

        optimizer.zero_grad()

        # Forward: returns (displacement_preds, erosion_logits)
        predictions, erosion_logits_list = model(
            seq, dt=1.0, teacher_forcing_ratio=tf_ratio
        )

        # ---- Displacement loss (NO MASKING — all nodes) ----
        disp_loss = 0.0
        for t, (pred, data) in enumerate(zip(predictions, seq)):
            disp_loss += F.mse_loss(pred, data.y)
        disp_loss = disp_loss / len(predictions)

        # ---- Erosion loss (focal) ----
        erosion_loss = torch.tensor(0.0, device=device)
        n_erosion_steps = 0

        for t, (logits, data) in enumerate(zip(erosion_logits_list, seq)):
            if hasattr(data, 'elements') and hasattr(data, 'x_element'):
                num_elements = data.elements.shape[0]

                # Target: erosion at NEXT timestep
                if t + 1 < len(seq) and hasattr(seq[t + 1], 'x_element'):
                    targets = get_gt_erosion_targets(seq[t + 1], num_elements)
                else:
                    targets = get_gt_erosion_targets(data, num_elements)

                targets = targets.to(device)
                erosion_loss += focal_loss_fn(logits, targets)
                n_erosion_steps += 1

                with torch.no_grad():
                    pred_eroded = (torch.sigmoid(logits) > 0.5).squeeze(-1)
                    gt_eroded = targets.bool()
                    total_tp += (pred_eroded & gt_eroded).sum().item()
                    total_fp += (pred_eroded & ~gt_eroded).sum().item()
                    total_fn += (~pred_eroded & gt_eroded).sum().item()

        if n_erosion_steps > 0:
            erosion_loss = erosion_loss / n_erosion_steps

        loss = disp_loss + args.erosion_weight * erosion_loss
        loss.backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip_norm
            )

        optimizer.step()

        total_disp_loss += disp_loss.item()
        total_erosion_loss += erosion_loss.item()
        total_loss += loss.item()
        n_batches += 1

        e_prec = total_tp / max(total_tp + total_fp, 1)
        e_rec = total_tp / max(total_tp + total_fn, 1)
        e_f1 = 2 * e_prec * e_rec / max(e_prec + e_rec, 1e-8)

        pbar.set_postfix({
            'loss': f"{loss.item():.5f}",
            'disp': f"{disp_loss.item():.5f}",
            'ero': f"{erosion_loss.item():.4f}",
            'F1': f"{e_f1:.3f}",
        })

    return {
        'loss': total_loss / max(n_batches, 1),
        'disp_loss': total_disp_loss / max(n_batches, 1),
        'erosion_loss': total_erosion_loss / max(n_batches, 1),
        'tf_ratio': tf_ratio,
        'erosion_tp': total_tp, 'erosion_fp': total_fp, 'erosion_fn': total_fn,
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

        # Displacement — no masking
        disp_loss = sum(F.mse_loss(pred, data.y)
                        for pred, data in zip(predictions, seq)) / len(predictions)

        # Erosion
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
        'loss': total_loss / max(n_batches, 1),
        'disp_loss': total_disp_loss / max(n_batches, 1),
        'erosion_loss': total_erosion_loss / max(n_batches, 1),
        'erosion_f1': e_f1,
        'erosion_precision': e_prec,
        'erosion_recall': e_rec,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, filepath)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train G-PARCv3 Elastoplastic")

    # Dataset
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, default="*.pt")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--num_static_feats", type=int, default=2)
    parser.add_argument("--num_dynamic_feats", type=int, default=2)
    parser.add_argument("--n_state_var", type=int, default=0)
    parser.add_argument("--preload", action="store_true", default=True)

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

    # Model
    parser.add_argument("--integrator", type=str, default="euler")
    parser.add_argument("--no_clamp_output", action="store_true", default=True)

    # SPADE
    parser.add_argument("--spade_random_noise", action="store_true", default=False)
    parser.add_argument("--spade_heads", type=int, default=4)
    parser.add_argument("--spade_concat", action="store_true", default=True)
    parser.add_argument("--spade_dropout", type=float, default=0.1)
    parser.add_argument("--zero_init", action="store_true", default=True)

    # Scheduled Sampling
    parser.add_argument("--ss_schedule", type=str, default="linear")
    parser.add_argument("--ss_initial_ratio", type=float, default=0.0)
    parser.add_argument("--ss_final_ratio", type=float, default=0.0)

    # Training
    parser.add_argument("--phase1_epochs", type=int, default=200,
                        help="Epochs for Phase 1 (frozen displacement)")
    parser.add_argument("--phase2_epochs", type=int, default=300,
                        help="Epochs for Phase 2 (joint training)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Displacement model LR (Phase 2)")
    parser.add_argument("--erosion_lr", type=float, default=1e-3,
                        help="Erosion head LR (both phases)")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    # Erosion Head
    parser.add_argument("--erosion_hidden_dim", type=int, default=64)
    parser.add_argument("--erosion_num_layers", type=int, default=2)
    parser.add_argument("--erosion_dropout", type=float, default=0.1)
    parser.add_argument("--erosion_weight", type=float, default=1.0)
    parser.add_argument("--erosion_threshold", type=float, default=0.5)
    parser.add_argument("--focal_alpha", type=float, default=0.75)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./outputs_v3")
    parser.add_argument("--v2_checkpoint", type=str, default=None,
                        help="V2 checkpoint to warm-start from")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume V3 training from checkpoint")
    parser.add_argument("--skip_phase1", action="store_true", default=False,
                        help="Skip Phase 1 (e.g., when resuming into Phase 2)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    norm_stats = load_normalization_stats(args.train_dir)

    total_epochs = args.phase1_epochs + args.phase2_epochs

    print("\n" + "=" * 70)
    print("G-PARCv3 TRAINING — EROSION-AWARE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Phase 1: {args.phase1_epochs} epochs (frozen displacement)")
    print(f"Phase 2: {args.phase2_epochs} epochs (joint training)")
    print(f"Displacement LR: {args.lr}")
    print(f"Erosion LR: {args.erosion_lr}")
    print(f"Focal loss: alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    print(f"No loss masking — training on ALL nodes")
    print("=" * 70)

    # ---- Dataset ----
    train_ids = get_simulation_ids(Path(args.train_dir), pattern=args.file_pattern)
    val_ids = get_simulation_ids(Path(args.val_dir), pattern=args.file_pattern)
    print(f"\n{len(train_ids)} train, {len(val_ids)} val simulations")

    train_dataset = ElastoPlasticDataset(
        directory=Path(args.train_dir), simulation_ids=train_ids,
        seq_len=args.seq_len, stride=args.stride,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        use_element_features=True,
        preload=args.preload,
    )
    val_dataset = ElastoPlasticDataset(
        directory=Path(args.val_dir), simulation_ids=val_ids,
        seq_len=args.seq_len, stride=args.stride,
        num_static_feats=args.num_static_feats,
        num_dynamic_feats=args.num_dynamic_feats,
        use_element_features=True,
        preload=args.preload,
    )

    train_loader = DataLoader(train_dataset, batch_size=None,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=None,
                            num_workers=args.num_workers, pin_memory=True)

    # Sample for initialization
    sample_data = next(iter(train_loader))[0]
    if not hasattr(sample_data, 'pos') or sample_data.pos is None:
        sample_data.pos = sample_data.x[:, :args.num_static_feats]
    print(f"  Sample: {sample_data.num_nodes} nodes, {sample_data.edge_index.shape[1]} edges")

    # ---- Create Model ----
    model = create_model(args, sample_data, norm_stats).to(device)
    count_parameters(model)

    # ---- Load V2 weights ----
    start_epoch = 0
    if args.v2_checkpoint:
        load_v2_weights(model, args.v2_checkpoint, device)
    elif args.resume:
        print(f"\nResuming V3 from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed at epoch {start_epoch}")

    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    # Save config
    config = vars(args)
    config['version'] = 'v3'
    config['erosion_aware_mls'] = True
    config['loss_masking'] = False
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    norm_stats_src = Path(args.train_dir).parent / "normalization_stats.json"
    if norm_stats_src.exists():
        shutil.copy2(norm_stats_src, output_dir / "normalization_stats.json")

    # ---- Training Loop ----
    history = {
        'train_loss': [], 'train_disp_loss': [], 'train_erosion_loss': [],
        'val_loss': [], 'val_disp_loss': [], 'val_erosion_loss': [],
        'val_erosion_f1': [], 'phase': [],
    }
    best_val_loss = float('inf')
    best_erosion_f1 = 0.0

    for epoch in range(start_epoch, total_epochs):
        # Determine phase
        if epoch < args.phase1_epochs and not args.skip_phase1:
            phase = 1
            phase_epoch = epoch
            phase_total = args.phase1_epochs
        else:
            phase = 2
            phase_epoch = epoch - args.phase1_epochs if not args.skip_phase1 else epoch
            phase_total = args.phase2_epochs

        # Setup for new phase
        if epoch == 0 and phase == 1:
            print("\n" + "=" * 70)
            print("PHASE 1: Frozen displacement, training erosion head + SPADE channel")
            print("=" * 70)
            freeze_displacement(model)
            optimizer = build_optimizer(model, args, phase=1)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.phase1_epochs,
                                           eta_min=args.erosion_lr * 0.01)

        elif epoch == args.phase1_epochs and not args.skip_phase1:
            print("\n" + "=" * 70)
            print("PHASE 2: Joint training — all parameters unfrozen")
            print("=" * 70)
            unfreeze_all(model)
            optimizer = build_optimizer(model, args, phase=2)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs,
                                           eta_min=min(args.lr, args.erosion_lr) * 0.01)

        elif epoch == 0 and args.skip_phase1:
            print("\n" + "=" * 70)
            print("PHASE 2 (skip_phase1): Joint training from start")
            print("=" * 70)
            unfreeze_all(model)
            optimizer = build_optimizer(model, args, phase=2)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs,
                                           eta_min=min(args.lr, args.erosion_lr) * 0.01)

        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1}/{total_epochs} (Phase {phase}, {phase_epoch + 1}/{phase_total})")
        print(f"{'=' * 70}")

        train_m = train_epoch(model, train_loader, optimizer, device,
                              phase_epoch, phase_total, args, focal_loss_fn)

        val_m = validate_epoch(model, val_loader, device, args, focal_loss_fn)

        scheduler.step()

        # Print summary
        tp, fp, fn = train_m['erosion_tp'], train_m['erosion_fp'], train_m['erosion_fn']
        t_prec = tp / max(tp + fp, 1)
        t_rec = tp / max(tp + fn, 1)
        t_f1 = 2 * t_prec * t_rec / max(t_prec + t_rec, 1e-8)

        print(f"\nTrain — loss: {train_m['loss']:.6f}, "
              f"disp: {train_m['disp_loss']:.6f}, "
              f"erosion: {train_m['erosion_loss']:.5f}, "
              f"F1: {t_f1:.3f}")
        print(f"Val   — loss: {val_m['loss']:.6f}, "
              f"disp: {val_m['disp_loss']:.6f}, "
              f"erosion: {val_m['erosion_loss']:.5f}, "
              f"F1: {val_m['erosion_f1']:.3f} "
              f"(P={val_m['erosion_precision']:.3f}, R={val_m['erosion_recall']:.3f})")

        # History
        history['train_loss'].append(train_m['loss'])
        history['train_disp_loss'].append(train_m['disp_loss'])
        history['train_erosion_loss'].append(train_m['erosion_loss'])
        history['val_loss'].append(val_m['loss'])
        history['val_disp_loss'].append(val_m['disp_loss'])
        history['val_erosion_loss'].append(val_m['erosion_loss'])
        history['val_erosion_f1'].append(val_m['erosion_f1'])
        history['phase'].append(phase)

        # Save best combined
        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            save_checkpoint(model, optimizer, scheduler, epoch,
                           {'val_loss': best_val_loss, 'erosion_f1': val_m['erosion_f1']},
                           output_dir / "best_model.pth")
            print(f"✓ Best model (loss={best_val_loss:.6f}, F1={val_m['erosion_f1']:.3f})")

        # Save best erosion F1
        if val_m['erosion_f1'] > best_erosion_f1:
            best_erosion_f1 = val_m['erosion_f1']
            save_checkpoint(model, optimizer, scheduler, epoch,
                           {'val_loss': val_m['loss'], 'erosion_f1': best_erosion_f1},
                           output_dir / "best_erosion_model.pth")
            print(f"✓ Best erosion model (F1={best_erosion_f1:.3f})")

        # Latest
        save_checkpoint(model, optimizer, scheduler, epoch,
                       {'val_loss': val_m['loss'], 'erosion_f1': val_m['erosion_f1']},
                       output_dir / "latest_model.pth")

        # Phase transition checkpoint
        if epoch == args.phase1_epochs - 1 and not args.skip_phase1:
            save_checkpoint(model, optimizer, scheduler, epoch,
                           {'val_loss': val_m['loss'], 'erosion_f1': val_m['erosion_f1']},
                           output_dir / "phase1_final.pth")
            print(f"✓ Saved Phase 1 final checkpoint")

        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best erosion F1: {best_erosion_f1:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
