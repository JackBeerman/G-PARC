from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp import GradScaler, autocast

def train_and_validate(model,
                       train_loader,
                       val_loader,
                       device,
                       num_epochs=50,
                       lr=1e-4,
                       best_model_path="best_model.pth"):
    """
    Trains and validates the integrated GPARC model, updated for debugging.
    """
    #torch.autograd.set_detect_anomaly(True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50)
    
    # [BEST PRACTICE] Updated GradScaler call
    scaler = GradScaler(device.type)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        train_sim_count = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", unit="sim") as pbar:
            for seq in pbar:
                # User's original data-to-device logic
                for data in seq:
                    try:
                        for key, value in data:
                            if torch.is_tensor(value):
                                data[key] = value.to(device)
                    except Exception as e:
                        # [DEBUG UPDATE] Use pbar.write to prevent message from being erased
                        pbar.write(f"  ERROR moving data to device: {e}")

                optimizer.zero_grad(set_to_none=True)
                
                # [BEST PRACTICE] Updated autocast call
                with autocast(device_type=device.type, dtype=torch.float16):
                    # [DEBUG UPDATE] Pass the pbar object to the model
                    predictions = model(seq)#, pbar=pbar)
                    
                    target_losses = []
                    for pred, data_t in zip(predictions, seq):
                        target_y = model.process_targets(data_t.y)
                        target_losses.append(criterion(pred, target_y))
                    sim_loss = torch.stack(target_losses).mean()
                
                scaler.scale(sim_loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
            with tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", unit="sim") as pbar_val:
                for seq in pbar_val:
                    for data in seq:
                        try:
                            for key, value in data:
                                if torch.is_tensor(value):
                                    data[key] = value.to(device)
                        except Exception as e:
                            # [DEBUG UPDATE] Use pbar.write here as well
                            pbar_val.write(f"  ERROR moving validation data to device: {e}")
                            
                    with autocast(device_type=device.type, dtype=torch.float16):
                        # [DEBUG UPDATE] Pass pbar_val to the model
                        predictions = model(seq)#, pbar=pbar_val)
                        
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

        # [DEBUG UPDATE] Add flush=True to ensure this prints immediately
        print(f"\n[Epoch {epoch}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}", flush=True)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6e}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            torch.save(checkpoint, best_model_path)
            print(f"  --> Validation improved. Checkpoint saved to: {best_model_path}", flush=True)

    return train_losses, val_losses


def train_and_validate_with_context(model,
                                    train_loader,
                                    val_loader,
                                    device,
                                    num_epochs=50,
                                    lr=1e-4,
                                    best_model_path="best_model.pth",
                                    num_context_steps=3):
    """
    Trains and validates the GPARC model with multi-step context.
    
    Args:
        model: GPARC model instance
        train_loader: DataLoader for training sequences
        val_loader: DataLoader for validation sequences
        device: torch device (cuda/cpu)
        num_epochs: Number of training epochs
        lr: Learning rate
        best_model_path: Path to save best model checkpoint
        num_context_steps: Number of initial timesteps to use as context (default: 3)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50)
    
    scaler = GradScaler(device.type)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        train_sim_count = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", unit="sim") as pbar:
            for seq in pbar:
                # Skip sequences that are too short
                if len(seq) <= num_context_steps:
                    pbar.write(f"  WARNING: Skipping sequence with length {len(seq)} (need > {num_context_steps})")
                    continue
                
                # Move data to device
                for data in seq:
                    try:
                        for key, value in data:
                            if torch.is_tensor(value):
                                data[key] = value.to(device)
                    except Exception as e:
                        pbar.write(f"  ERROR moving data to device: {e}")

                optimizer.zero_grad(set_to_none=True)
                
                with autocast(device_type=device.type, dtype=torch.float16):
                    # Forward pass with full sequence
                    predictions = model(seq)
                    
                    # Only compute loss on predictions after context steps
                    prediction_outputs = predictions[num_context_steps:]
                    target_sequence = seq[num_context_steps:]
                    
                    target_losses = []
                    for pred, data_t in zip(prediction_outputs, target_sequence):
                        target_y = model.process_targets(data_t.y)
                        target_losses.append(criterion(pred, target_y))
                    
                    sim_loss = torch.stack(target_losses).mean()
                
                scaler.scale(sim_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_item = sim_loss.item()
                if not torch.isnan(torch.tensor(loss_item)):
                    total_train_loss += loss_item
                train_sim_count += 1
                pbar.set_postfix({"Loss": f"{loss_item:.6f}", "Context": num_context_steps})

        avg_train_loss = total_train_loss / train_sim_count if train_sim_count > 0 else float('inf')
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        val_sim_count = 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", unit="sim") as pbar_val:
                for seq in pbar_val:
                    # Skip sequences that are too short
                    if len(seq) <= num_context_steps:
                        continue
                    
                    # Move data to device
                    for data in seq:
                        try:
                            for key, value in data:
                                if torch.is_tensor(value):
                                    data[key] = value.to(device)
                        except Exception as e:
                            pbar_val.write(f"  ERROR moving validation data to device: {e}")
                            
                    with autocast(device_type=device.type, dtype=torch.float16):
                        predictions = model(seq)
                        
                        # Only compute loss on predictions after context steps
                        prediction_outputs = predictions[num_context_steps:]
                        target_sequence = seq[num_context_steps:]
                        
                        target_losses = []
                        for pred, data_t in zip(prediction_outputs, target_sequence):
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

        print(f"\n[Epoch {epoch}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}", flush=True)
        print(f"Context steps: {num_context_steps}", flush=True)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6e}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'num_context_steps': num_context_steps  # Save this for reference
            }
            torch.save(checkpoint, best_model_path)
            print(f"  --> Validation improved. Checkpoint saved to: {best_model_path}", flush=True)

    return train_losses, val_losses


def load_model(model, checkpoint_path, device):
    """
    Load a saved model checkpoint.
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
    else:
        print("No valid checkpoint found. Starting from scratch.")
    return model


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