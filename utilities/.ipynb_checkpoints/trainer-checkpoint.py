#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
#
#def train_and_validate(model,
#                       train_loader,
#                       val_loader,
#                       device,
#                       num_epochs=50,
#                       lr=1e-4,
#                       best_model_path="best_model.pth"):
#    """
#    Trains and validates the integrated GPARC model.
#    """
#    torch.autograd.set_detect_anomaly(True)
#    optimizer = optim.Adam(model.parameters(), lr=lr)
#    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                     factor=0.1, patience=50)
#    scaler = GradScaler()
#    criterion = nn.MSELoss()
#
#    best_val_loss = float('inf')
#    train_losses = []
#    val_losses = []
#
#    for epoch in range(1, num_epochs + 1):
#        #print(f"\n{'='*60}")
#        #print(f"STARTING EPOCH {epoch}/{num_epochs}")
#        #print(f"{'='*60}\n")
#        
#        # --- Training Phase ---
#        model.train()
#        total_train_loss = 0.0
#        train_sim_count = 0
#        
#        #print("About to enter tqdm loop...")
#        #print(f"train_loader type: {type(train_loader)}")
#        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", unit="sim") as pbar:
#            for seq in pbar:
#                #print(f"\n=== DEBUG INFO ===")
#                #print(f"seq type: {type(seq)}")
#                #print(f"seq length: {len(seq)}")
#                #print(f"first element type: {type(seq[0])}")
#                
#                for data in seq:
#                    #print(f"data type: {type(data)}")
#                    #print(f"Attempting to iterate over data object...")
#                    try:
#                        for key, value in data:
#                            #print(f"  key={key}, value type={type(value)}")
#                            if torch.is_tensor(value):
#                                data[key] = value.to(device)
#                    except Exception as e:
#                        print(f"  ERROR: {e}")
#                #print(f"===================\n")
#
#                optimizer.zero_grad()
#                with autocast():
#                    predictions = model(seq)
#                    target_losses = []
#                    for pred, data_t in zip(predictions, seq):
#                        target_y = model.process_targets(data_t.y)
#                        target_losses.append(criterion(pred, target_y))
#                    sim_loss = torch.stack(target_losses).mean()
#                
#                scaler.scale(sim_loss).backward()
#                scaler.step(optimizer)
#                scaler.update()
#
#                total_train_loss += sim_loss.item()
#                train_sim_count += 1
#                pbar.set_postfix({"Loss": f"{sim_loss.item():.6f}"})
#
#        avg_train_loss = total_train_loss / train_sim_count if train_sim_count > 0 else float('inf')
#        train_losses.append(avg_train_loss)
#
#        # --- Validation Phase ---
#        model.eval()
#        total_val_loss = 0.0
#        val_sim_count = 0
#        with torch.no_grad():
#            with tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", unit="sim") as pbar_val:
#                for seq in pbar_val:
#                    for data in seq:
#                        for key, value in data:
#                            if torch.is_tensor(value):
#                                data[key] = value.to(device)
#
#                    with autocast():
#                        predictions = model(seq)
#                        target_losses = []
#                        for pred, data_t in zip(predictions, seq):
#                            target_y = model.process_targets(data_t.y)
#                            target_losses.append(criterion(pred, target_y))
#                        sim_loss = torch.stack(target_losses).mean()
#
#                    total_val_loss += sim_loss.item()
#                    val_sim_count += 1
#                    pbar_val.set_postfix({"Val Loss": f"{sim_loss.item():.6f}"})
#
#        avg_val_loss = total_val_loss / val_sim_count if val_sim_count > 0 else float('inf')
#        val_losses.append(avg_val_loss)
#
#        print(f"[Epoch {epoch}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
#        scheduler.step(avg_val_loss)
#        current_lr = optimizer.param_groups[0]['lr']
#        print(f"Current learning rate: {current_lr:.6e}")
#
#        if avg_val_loss < best_val_loss:
#            best_val_loss = avg_val_loss
#            checkpoint = {
#                'model_state_dict': model.state_dict(),
#                'optimizer_state_dict': optimizer.state_dict(),
#                'epoch': epoch,
#                'train_loss': avg_train_loss,
#                'val_loss': avg_val_loss
#            }
#            torch.save(checkpoint, best_model_path)
#            print(f"  --> Validation improved. Checkpoint saved to: {best_model_path}")
#
#    return train_losses, val_losses

import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp import GradScaler, autocast # [BEST PRACTICE] Use modern torch.amp
from tqdm import tqdm

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
    