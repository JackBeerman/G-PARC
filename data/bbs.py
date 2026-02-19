import torch
import math
from torch.utils.data import IterableDataset
from pathlib import Path
from typing import List, Iterator
from torch_geometric.data import Data
import numpy as np

class BurgersDataset(IterableDataset):
    """
    Dataset for 2D Burgers' Equation on Graph.
    
    Raw .pt files: x = [pos_x, pos_y, u, v, Re]
    Output:        x = [pos_x, pos_y, Re_norm, u, v]
    
    Re normalization: log10(Re) / log10(Re_max)
      - Puts Re in ~[0.48, 1.0] range, matching pos/velocity scale
      - Log-scale is natural for Reynolds numbers (fluid dynamics convention)
      - Re_max=15000 covers full dataset range (train + val + test)
    """
    
    # Global Re range across ALL splits (train + val + test)
    RE_MAX = 15000.0
    
    def __init__(self, directory, simulation_ids=None, seq_len=1, stride=1, 
                 file_pattern="*.pt"):
        super().__init__()
        self.directory = Path(directory)
        self.seq_len = seq_len
        self.stride = stride
        self.file_pattern = file_pattern
        
        self.static_indices = [0, 1]    # pos_x, pos_y
        self.dynamic_indices = [2, 3]   # u, v
        self.param_indices = [4]        # Re
        
        # Log10 normalization constant
        self.log_re_max = math.log10(self.RE_MAX)
        
        if simulation_ids is None:
            self.simulation_ids = sorted([f.stem for f in self.directory.glob(file_pattern)])
        else:
            self.simulation_ids = sorted(simulation_ids)

        # Pre-calculate exact length
        self.total_samples = 0
        self.sim_lengths = {}
        
        print(f"Scanning {len(self.simulation_ids)} files to calculate dataset length...")
        
        for sim_id in self.simulation_ids:
            try:
                path = self.directory / f"{sim_id}.pt"
                sim_data = torch.load(path, weights_only=False)
                T = len(sim_data)
                
                max_start = T - self.seq_len
                if max_start >= 0:
                    n_samples = (max_start // self.stride) + 1
                    self.total_samples += n_samples
                    self.sim_lengths[sim_id] = n_samples
            except Exception as e:
                print(f"Error scanning {sim_id}: {e}")
                
        print(f"Total samples calculated: {self.total_samples}")
    
    def _normalize_re(self, re_tensor):
        """Normalize Reynolds number: log10(Re) / log10(Re_max).
        
        Mapping:
            Re=100   → 0.479
            Re=500   → 0.646
            Re=1000  → 0.718
            Re=2500  → 0.813
            Re=5000  → 0.885
            Re=7500  → 0.928
            Re=10000 → 0.957
            Re=12500 → 0.980
            Re=15000 → 1.000
        """
        return torch.log10(re_tensor.clamp(min=1.0)) / self.log_re_max

    def __iter__(self) -> Iterator[List[Data]]:
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            sims_to_process = self.simulation_ids
        else:
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
                        
                        # Normalize Re: log10(Re) / log10(15000)
                        re_norm = self._normalize_re(re)
                        
                        # Output: [pos_x, pos_y, Re_norm, u, v]
                        data.x = torch.cat([pos, re_norm, vel], dim=1)
                        window.append(data)
                        
                    yield window
                    
            except Exception as e:
                print(f"Error loading {sim_id}: {e}")
                continue

    def __len__(self):
        return self.total_samples