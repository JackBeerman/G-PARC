import sys
from tqdm import tqdm
tqdm.write = lambda msg: sys.stderr.write(str(msg) + "\n")
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp import GradScaler, autocast

# NEW DAT FORMAT####################################################################################

class ComponentWeightedMSELoss(nn.Module):
    """MSE loss with per-component weighting."""
    def __init__(self, component_stds):
        super().__init__()
        weights = 1.0 / (component_stds + 1e-8)
        self.weights = weights / weights.mean()
        
        print("=" * 70)
        print("Component-Weighted MSE Loss Initialized")
        print("=" * 70)
        for i, (std, weight) in enumerate(zip(component_stds, self.weights)):
            print(f"  Component {i}: std={std:.4f}, weight={weight:.4f}")
        print("=" * 70)
    
    def forward(self, pred, target):
        """
        Args:
            pred: [T, N, C] or [N, C]
            target: [T, N, C] or [N, C]
        """
        squared_errors = (pred - target) ** 2
        
        # Broadcast weights to match dimensions
        if pred.ndim == 3:  # [T, N, C]
            weights = self.weights.view(1, 1, -1)
        else:  # [N, C]
            weights = self.weights.view(1, -1)
        
        weighted_errors = squared_errors * weights
        return weighted_errors.mean()


def train_and_validate_weighted(model,
                                train_loader,
                                val_loader,
                                device,
                                component_stds,
                                num_epochs=50,
                                lr=1e-4,
                                best_model_path="best_model.pth",
                                latest_model_path="latest_model.pth",
                                grad_clip_norm=None,
                                use_amp=True,
                                resume_checkpoint=None):
    """
    Trains and validates with COMPONENT-WEIGHTED loss.
    
    Args:
        use_amp (bool): If True, uses Automatic Mixed Precision (Float16). 
                        Set to False for physics models calculating derivatives.
        resume_checkpoint (dict or str): Checkpoint dict or path to resume from (optional).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=50
    )
    
    # ====== RESUME LOGIC ======
    start_epoch = 1
    best_val_loss = float('inf')
    
    if resume_checkpoint is not None:
        print(f"Resuming training from checkpoint...")
        
        # Handle both dict and path inputs
        if isinstance(resume_checkpoint, str):
            checkpoint = torch.load(resume_checkpoint, map_location=device)
        else:
            checkpoint = resume_checkpoint
        
        # Restore optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  ✓ Optimizer state restored")
        
        # Restore scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  ✓ Scheduler state restored")
        
        # Restore epoch counter
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"  ✓ Resuming from epoch {start_epoch}")
        
        # Restore best validation loss
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
            print(f"  ✓ Previous best val loss: {best_val_loss:.9e}")
    # ==========================
    
    # Initialize Scaler only if AMP is enabled
    scaler = GradScaler(device.type) if use_amp else None
    
    criterion = ComponentWeightedMSELoss(component_stds.to(device))

    train_losses = []
    val_losses = []

    print(f"Training Config: AMP={'Enabled' if use_amp else 'Disabled (Float32)'} | Device={device}")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        train_sim_count = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch + num_epochs - 1} [Train]", unit="sim") as pbar:
            for seq in pbar:
                for data in seq:
                    for key, value in data.items():
                        if torch.is_tensor(value):
                            data[key] = value.to(device)

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with autocast(device_type=device.type, dtype=torch.float16):
                        predictions = model(seq)
                        target_losses = []
                        for pred, data_t in zip(predictions, seq):
                            target_y = model.process_targets(data_t.y)
                            target_losses.append(criterion(pred, target_y))
                        sim_loss = torch.stack(target_losses).mean()
                    
                    scaler.scale(sim_loss).backward()
                    
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                
                else:
                    predictions = model(seq)
                    target_losses = []
                    for pred, data_t in zip(predictions, seq):
                        target_y = model.process_targets(data_t.y)
                        target_losses.append(criterion(pred, target_y))
                    sim_loss = torch.stack(target_losses).mean()
                    
                    sim_loss.backward()
                    
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    
                    optimizer.step()

                loss_item = sim_loss.item()
                if not torch.isnan(torch.tensor(loss_item)):
                    total_train_loss += loss_item
                train_sim_count += 1
                pbar.set_postfix({"Loss": f"{loss_item:.6f}"})

        avg_train_loss = total_train_loss / train_sim_count if train_sim_count > 0 else float('inf')
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        val_sim_count = 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch}/{start_epoch + num_epochs - 1} [Val]", unit="sim") as pbar_val:
                for seq in pbar_val:
                    for data in seq:
                        for key, value in data.items():
                            if torch.is_tensor(value):
                                data[key] = value.to(device)
                    
                    if use_amp:
                        with autocast(device_type=device.type, dtype=torch.float16):
                            predictions = model(seq)
                            target_losses = []
                            for pred, data_t in zip(predictions, seq):
                                target_y = model.process_targets(data_t.y)
                                target_losses.append(criterion(pred, target_y))
                            sim_loss = torch.stack(target_losses).mean()
                    else:
                        predictions = model(seq)
                        target_losses = []
                        for pred, data_t in zip(predictions, seq):
                            target_y = model.process_targets(data_t.y)
                            target_losses.append(criterion(pred, target_y))
                        sim_loss = torch.stack(target_losses).mean()

                    loss_item = sim_loss.item()
                    if not torch.isnan(torch.tensor(loss_item)):
                        total_val_loss += loss_item
                    val_sim_count += 1
                    pbar_val.set_postfix({"Val Loss": f"{loss_item:.6f}"})

        avg_val_loss = total_val_loss / val_sim_count if val_sim_count > 0 else float('inf')
        val_losses.append(avg_val_loss)

        print(f"\n[Epoch {epoch}/{start_epoch + num_epochs - 1}] Train Loss: {avg_train_loss:.9e} | Val Loss: {avg_val_loss:.9e}", flush=True)
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            torch.save(checkpoint, best_model_path)
            print(f"  --> Validation improved. Checkpoint saved to: {best_model_path}", flush=True)

        # Save latest model
        latest_checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        torch.save(latest_checkpoint, latest_model_path)
        print(f"  --> Latest model checkpoint overwritten: {latest_model_path}", flush=True)

    return train_losses, val_losses


def load_model(model, checkpoint_path, device):
    """
    Load a saved model checkpoint.
    Returns the loaded checkpoint dictionary for resume functionality.
    """
    ckpt_file = Path(checkpoint_path)
    if ckpt_file.is_file():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model checkpoint.")
        
        # Print context steps if available
        if 'num_context_steps' in checkpoint:
            print(f"Model was trained with {checkpoint['num_context_steps']} context steps")
        
        return checkpoint
    else:
        print("No valid checkpoint found. Starting from scratch.")
        return None


def plot_loss_curves(train_losses, val_losses, title="Training & Validation Loss", save_path=None):
    """
    Plot and save training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'o-', label="Train Loss")
    plt.plot(epochs, val_losses, 's-', label="Val Loss")
    plt.title(title, fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to: {save_path}")
    plt.close()