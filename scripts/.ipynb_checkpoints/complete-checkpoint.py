#!/usr/bin/env python

import argparse
import pickle
from pathlib import Path
from typing import List, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GraphUNet
from tqdm import tqdm
import matplotlib.pyplot as plt

#####################################################
# 1) DATASET DEFINITION
#####################################################

class IterativeRolloutDataset(IterableDataset):
    """
    Yields consecutive *shifted* sequences of timesteps:
      [ (Data[t].x, Data[t+1].y), (Data[t+1].x, Data[t+2].y), ..., (Data[t+seq_len-1].x, Data[t+seq_len].y) ]
    for each simulation.
    """
    def __init__(self,
                 directory: Path,
                 hydrograph_ids: List[str],
                 seq_len: int,
                 num_static_feats: int,
                 num_dynamic_feats: int):
        """
        Args:
            directory (Path): Directory containing your pickled simulation files.
            hydrograph_ids (List[str]): List of simulation IDs (file stems).
            seq_len (int): Number of consecutive steps in each returned window.
                           NOTE: Because we shift y by +1, we actually need seq_len+1 timesteps.
            num_static_feats (int): Number of static features in data.x.
            num_dynamic_feats (int): Number of dynamic features in data.x.
        """
        super().__init__()
        self.directory = directory
        self.hydrograph_ids = hydrograph_ids
        self.seq_len = seq_len
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats

    def __iter__(self) -> Iterator[List[Data]]:
        for hid in self.hydrograph_ids:
            pickle_file = self.directory / f"{hid}.pkl"
            try:
                with open(pickle_file, 'rb') as file:
                    sim = pickle.load(file)
                    if not isinstance(sim, list):
                        print(f"Unexpected data format in {pickle_file}. Skipping.")
                        continue
                    T = len(sim)
                    max_start = T - self.seq_len
                    if max_start < 1:
                        continue
                    for start_idx in range(max_start):
                        window = []
                        for offset in range(self.seq_len):
                            t = start_idx + offset
                            data_t = sim[t].clone()
                            # Partition x into static and dynamic parts using parameterized indices
                            static_feats = data_t.x[:, :self.num_static_feats]
                            dynamic_feats = data_t.x[:, self.num_static_feats:self.num_static_feats + self.num_dynamic_feats]
                            data_t.x = torch.cat([static_feats, dynamic_feats], dim=1)
                            # Overwrite y with next timestep's ground truth
                            data_tplus1 = sim[t+1]
                            data_t.y = data_tplus1.y.clone()
                            window.append(data_t)
                        yield window
            except FileNotFoundError:
                print(f"Pickle file not found: {pickle_file}")
            except Exception as e:
                print(f"Error loading {pickle_file}: {e}")

def get_hydrograph_ids(directory: Path) -> List[str]:
    return [file.stem for file in directory.glob("*.pkl")]

#####################################################
# 2) MODEL DEFINITION
#####################################################

class FeatureExtractorGNN(nn.Module):
    """
    GraphUNet-based feature extractor for each node with attention.
    """
    def __init__(self, in_channels=9, hidden_channels=64, out_channels=32, depth=3, pool_ratios=0.5, heads=4, concat=True, dropout=0.6):
        super(FeatureExtractorGNN, self).__init__()
        self.unet = GraphUNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            depth=depth,
            pool_ratios=pool_ratios,
            act=F.relu
        )
        self.attention1 = GATConv(out_channels, out_channels, heads=heads, concat=concat, dropout=dropout)
        self.attention2 = GATConv(out_channels * heads if concat else out_channels, out_channels, heads=1, concat=False, dropout=dropout)
        self.residual = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        residual = self.unet(x, edge_index)
        x = F.elu(self.attention1(residual, edge_index))
        x = self.attention2(x, edge_index)
        x += residual
        return x

class DerivativeGNN(nn.Module):
    """
    GNN that approximates Fdot = dF/dt using Graph Attention.
    """
    def __init__(self, in_channels, out_channels=64, heads=4, concat=True, dropout=0.6):
        super(DerivativeGNN, self).__init__()
        self.gat1 = GATConv(in_channels, 64, heads=heads, concat=concat, dropout=dropout)
        self.gat2 = GATConv(64 * heads if concat else 64, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

class IntegralGNN(nn.Module):
    """
    GNN that learns the integral operator: ΔF = ∫ Fdot dt using Graph Attention.
    """
    def __init__(self, in_channels=64, out_channels=4, heads=4, concat=True, dropout=0.6):
        super(IntegralGNN, self).__init__()
        self.gat1 = GATConv(in_channels, 64, heads=heads, concat=concat, dropout=dropout)
        self.gat2 = GATConv(64 * heads if concat else 64, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

#####################################################
# 3) RECURRENT GPARC
#####################################################

class GPARCRecurrent(nn.Module):
    """
    Recurrent GPARC that processes a sequence of Data objects.
    Uses parameterized indices to extract static and dynamic features.
    """
    def __init__(self, derivative_solver, integral_solver, num_static_feats=9, num_dynamic_feats=4):
        super().__init__()
        self.derivative_solver = derivative_solver
        self.integral_solver = integral_solver
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats

    def forward(self, data_list, static_feature_map):
        predictions = []
        F_prev = None
        for data in data_list:
            x = data.x
            edge_index = data.edge_index
            # Use parameterized slicing instead of hard-coded indices
            dynamic_feats_t = x[:, self.num_static_feats:self.num_static_feats + self.num_dynamic_feats]
            F_prev_used = dynamic_feats_t if F_prev is None else F_prev
            Fdot_input = torch.cat([F_prev_used, static_feature_map], dim=-1)
            Fdot = self.derivative_solver(Fdot_input, edge_index)
            Fint = self.integral_solver(Fdot, edge_index)
            F_pred = F_prev_used + Fint
            predictions.append(F_pred)
            F_prev = F_pred
        return predictions

#####################################################
# 4) TRAIN/VALIDATION LOGIC
#####################################################

def train_and_validate(feature_extractor, model, train_loader, val_loader, device, num_static_feats, num_epochs=4, lr=1e-4, best_model_path="best_model.pth"):
    """
    Trains and validates the model.
    Both the feature extractor and model are updated using one optimizer.
    A learning rate scheduler is used to adjust the learning rate based on validation loss.
    """
    optimizer = optim.Adam(list(model.parameters()) + list(feature_extractor.parameters()), lr=lr)
    # Initialize the scheduler to reduce LR on plateau of validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    scaler = GradScaler()
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        feature_extractor.train()
        total_train_loss = 0.0
        train_sim_count = 0

        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", unit="sim") as pbar:
            for seq in pbar:
                for data in seq:
                    data.x = data.x.to(device)
                    data.y = data.y.to(device)
                    data.edge_index = data.edge_index.to(device)
                    if getattr(data, 'edge_attr', None) is not None:
                        data.edge_attr = data.edge_attr.to(device)

                static_feats_0 = seq[0].x[:, :num_static_feats]
                edge_index_0 = seq[0].edge_index

                optimizer.zero_grad()
                with autocast():
                    static_feature_map = feature_extractor(static_feats_0, edge_index_0)
                    predictions = model(seq, static_feature_map)
                    sim_loss = sum(criterion(pred, data_t.y) for pred, data_t in zip(predictions, seq)) / len(seq)
                scaler.scale(sim_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += sim_loss.item()
                train_sim_count += 1
                pbar.set_postfix({"Loss": f"{sim_loss.item():.6f}"})

        avg_train_loss = total_train_loss / train_sim_count if train_sim_count > 0 else float('inf')
        train_losses.append(avg_train_loss)

        model.eval()
        feature_extractor.eval()
        total_val_loss = 0.0
        val_sim_count = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", unit="sim") as pbar_val:
                for seq in pbar_val:
                    for data in seq:
                        data.x = data.x.to(device)
                        data.y = data.y.to(device)
                        data.edge_index = data.edge_index.to(device)
                        if getattr(data, 'edge_attr', None) is not None:
                            data.edge_attr = data.edge_attr.to(device)
                    with autocast():
                        static_feats_0 = seq[0].x[:, :num_static_feats]
                        edge_index_0 = seq[0].edge_index
                        static_feature_map = feature_extractor(static_feats_0, edge_index_0)
                        predictions = model(seq, static_feature_map)
                        sim_loss = sum(criterion(pred, data_t.y) for pred, data_t in zip(predictions, seq)) / len(seq)
                    total_val_loss += sim_loss.item()
                    val_sim_count += 1
                    pbar_val.set_postfix({"Val Loss": f"{sim_loss.item():.6f}"})

        avg_val_loss = total_val_loss / val_sim_count if val_sim_count > 0 else float('inf')
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Step the scheduler with the validation loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'feature_extractor_state_dict': feature_extractor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, best_model_path)
            print(f"  --> Validation improved. Checkpoint saved to: {best_model_path}")

    return train_losses, val_losses

#####################################################
# 5) PLOTTING
#####################################################

def plot_loss_curves(train_losses, val_losses, title="Training & Validation Loss", save_path=None):
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", marker="s")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss plot saved to: {save_path}")
    plt.show()

#####################################################
# 6) MAIN LOGIC WITH ARGPARSE
#####################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training data directory.")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation data directory.")
    parser.add_argument("--test_dir", type=str, default=None, help="Path to test data directory (optional).")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--seq_len", type=int, default=4, help="Sequence length for the IterativeRolloutDataset.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (if collating multiple simulations).")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save checkpoints and plots.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a combined model checkpoint (optional).")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    test_dir = Path(args.test_dir) if args.test_dir else None

    train_ids = get_hydrograph_ids(train_dir)
    val_ids = get_hydrograph_ids(val_dir)
    print(f"[Data] #Train IDs = {len(train_ids)}, #Val IDs = {len(val_ids)}")

    num_static_feats = 9
    num_dynamic_feats = 4

    train_dataset = IterativeRolloutDataset(
        directory=train_dir,
        hydrograph_ids=train_ids,
        seq_len=args.seq_len,
        num_static_feats=num_static_feats,
        num_dynamic_feats=num_dynamic_feats
    )
    val_dataset = IterativeRolloutDataset(
        directory=val_dir,
        hydrograph_ids=val_ids,
        seq_len=args.seq_len,
        num_static_feats=num_static_feats,
        num_dynamic_feats=num_dynamic_feats
    )

    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=0)

    feature_extractor = FeatureExtractorGNN(
        in_channels=num_static_feats,
        hidden_channels=64,
        out_channels=128,
        depth=2,
        pool_ratios=0.1,
        heads=4,
        concat=True,
        dropout=0.2
    ).to(device)

    derivative_solver = DerivativeGNN(
        in_channels=128 + num_dynamic_feats,  # 128 from feature_extractor + dynamic features
        out_channels=256,
        heads=4,
        concat=True,
        dropout=0.2
    ).to(device)

    integral_solver = IntegralGNN(
        in_channels=256,
        out_channels=num_dynamic_feats,
        heads=4,
        concat=True,
        dropout=0.2
    ).to(device)

    model = GPARCRecurrent(
        derivative_solver=derivative_solver,
        integral_solver=integral_solver,
        num_static_feats=num_static_feats,
        num_dynamic_feats=num_dynamic_feats
    ).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
            print("Loaded combined checkpoint for model and feature extractor.")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model checkpoint (only model).")
    else:
        print("No checkpoint provided. Training from scratch.")

    best_model_path = str(Path(args.save_dir) / "complete_modelseq1.pth")
    train_losses, val_losses = train_and_validate(
        feature_extractor=feature_extractor,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_static_feats=num_static_feats,
        num_epochs=args.num_epochs,
        lr=args.lr,
        best_model_path=best_model_path
    )

    loss_plot_path = str(Path(args.save_dir) / "loss_plot.png")
    plot_loss_curves(train_losses, val_losses, title="GPARC Recurrent Training", save_path=loss_plot_path)

    print("All done!")

if __name__ == "__main__":
    main()
