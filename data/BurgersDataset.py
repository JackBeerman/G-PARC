import torch
from torch.utils.data import IterableDataset
from pathlib import Path
from typing import List, Iterator
from torch_geometric.data import Data
import numpy as np

class BurgersDataset(IterableDataset):
    """
    Dataset for 2D Burgers' Equation on Graph.
    Fixed: Exact length calculation to prevent DataLoader warnings.
    """
    
    def __init__(self, directory, simulation_ids=None, seq_len=1, stride=1, file_pattern="*.pt"):
        super().__init__()
        self.directory = Path(directory)
        self.seq_len = seq_len
        self.stride = stride
        self.file_pattern = file_pattern
        
        self.static_indices = [0, 1]    # pos_x, pos_y
        self.dynamic_indices = [2, 3]   # u, v
        self.param_indices = [4]        # Re
        
        if simulation_ids is None:
            self.simulation_ids = sorted([f.stem for f in self.directory.glob(file_pattern)])
        else:
            self.simulation_ids = sorted(simulation_ids)

        # --- FIX: Pre-calculate EXACT length ---
        self.total_samples = 0
        self.sim_lengths = {}
        
        print(f"Scanning {len(self.simulation_ids)} files to calculate dataset length...")
        
        for sim_id in self.simulation_ids:
            try:
                # We load just the list structure, not the tensors (fast load)
                # Note: torch.load on a list of Data objects usually loads everything.
                # To be fast, we assume standard length if possible, or load once.
                # For robustness, we load. If too slow, we can hardcode if known (e.g. 101 steps).
                
                # OPTIMIZATION: If you know all sims have exactly 100 steps, set fixed_T = 100
                # Otherwise, we check one file or all files.
                
                # Let's try loading the first file to gauge length, assuming all are same
                # If your sims vary in length, you must loop all (slower startup).
                # Assuming homogenous dataset for speed:
                path = self.directory / f"{sim_id}.pt"
                # Just read the file size or assume consistent length for speed
                # Or perform full scan:
                sim_data = torch.load(path, weights_only=False)
                T = len(sim_data)
                
                max_start = T - self.seq_len
                if max_start >= 0:
                    # Integer division for stride
                    n_samples = (max_start // self.stride) + 1
                    self.total_samples += n_samples
                    self.sim_lengths[sim_id] = n_samples
            except Exception as e:
                print(f"Error scanning {sim_id}: {e}")
                
        print(f"Total samples calculated: {self.total_samples}")

    def __iter__(self) -> Iterator[List[Data]]:
        worker_info = torch.utils.data.get_worker_info()
        
        # Multi-worker splitting logic
        if worker_info is None:
            # Single process
            sims_to_process = self.simulation_ids
        else:
            # Split simulations among workers
            per_worker = int(np.ceil(len(self.simulation_ids) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.simulation_ids))
            sims_to_process = self.simulation_ids[iter_start:iter_end]

        for sim_id in sims_to_process:
            try:
                sim_data = torch.load(self.directory / f"{sim_id}.pt", weights_only=False)
                T = len(sim_data)
                max_start = T - self.seq_len
                
                if max_start < 0: continue
                
                for start in range(0, max_start + 1, self.stride):
                    window = []
                    for offset in range(self.seq_len):
                        data = sim_data[start + offset].clone()
                        
                        pos = data.x[:, self.static_indices]
                        vel = data.x[:, self.dynamic_indices]
                        re = data.x[:, self.param_indices]
                        
                        data.x = torch.cat([pos, re, vel], dim=1)
                        window.append(data)
                        
                    yield window
                    
            except Exception as e:
                print(f"Error loading {sim_id}: {e}")
                continue

    def __len__(self):
        return self.total_samples