"""
RiverDataset.py
===============
Dataset for HEC-RAS River Simulations (White River / Iowa River).

Adapted from ElastoPlasticDataset to maintain consistent G-PARC pipeline.

CRITICAL FIX (Feb 2026):
  The pkl→pt conversion script set mesh_id = simulation number (e.g., 30, 348).
  This means each simulation gets a unique MLS cache entry, which WASTES memory
  AND causes a subtle bug: when the cache falls back to pos.data_ptr() (if mesh_id
  propagation fails anywhere in the pipeline), stale cached dX from a different
  mesh can be returned, causing the tensor size mismatch:
    "RuntimeError: The size of tensor a (19148) must match the size of tensor b (9868)"
  
  FIX: ALWAYS override mesh_id with the LOCATION-BASED id:
    - 0 = White River (Muncie mesh)
    - 1 = Iowa River (Flood GNN mesh)
  This ensures all sims sharing the same physical mesh share the same MLS cache.
"""

import math
import re
import random
import torch
import numpy as np
from pathlib import Path
from typing import List, Iterator, Union, Optional
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data


class RiverDataset(IterableDataset):
    """
    Dataset for HEC-RAS 2D River Simulations.
    """
    
    def __init__(self,
                 directory: Union[str, Path],
                 simulation_ids: List[str] = None,
                 seq_len: int = 4,
                 stride: int = 1,
                 num_static_feats: int = 9,
                 num_dynamic_feats: int = 4,
                 file_pattern: str = "*.pt",
                 shuffle: bool = False,
                 shuffle_buffer_size: int = 100):
        super().__init__()
        self.directory = Path(directory)
        self.seq_len = seq_len
        self.stride = stride
        self.file_pattern = file_pattern
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.shuffle = shuffle
        self.buffer_size = shuffle_buffer_size
        
        if simulation_ids is None:
            self.simulation_ids = self._discover_simulation_ids()
        else:
            self.simulation_ids = simulation_ids
        
        self.mesh_groups = self._group_by_mesh()
        
        print(f"RiverDataset initialized:")
        print(f"  Directory: {self.directory}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Stride: {stride}")
        print(f"  Static features: {num_static_feats}")
        print(f"  Dynamic features: {num_dynamic_feats}")
        print(f"  Shuffle: {shuffle}")
        print(f"  Found {len(self.simulation_ids)} simulation files")
        for mesh_name, sims in self.mesh_groups.items():
            print(f"    {mesh_name}: {len(sims)} simulations")
    
    def _discover_simulation_ids(self) -> List[str]:
        files = sorted(list(self.directory.glob(self.file_pattern)))
        return [file.stem for file in files]
    
    def _group_by_mesh(self) -> dict:
        groups = {"iowa_river": [], "white_river": []}
        for sim_id in self.simulation_ids:
            if "iw" in sim_id.lower():
                groups["iowa_river"].append(sim_id)
            else:
                groups["white_river"].append(sim_id)
        return groups
    
    def _get_location_mesh_id(self, sim_name: str) -> int:
        """
        Get LOCATION-BASED mesh_id for MLS operator caching.
        
        All sims from the same river share the same physical mesh,
        so they MUST share the same MLS cache entry.
        
        Returns:
            0 = White River (Muncie)
            1 = Iowa River (Flood GNN)
        """
        if "iw" in sim_name.lower():
            return 1  # Iowa River mesh
        else:
            return 0  # White River mesh
    
    def _extract_sim_id(self, sim_name: str) -> int:
        """Extract unique simulation ID (for tracking, not MLS caching)."""
        match = re.search(r'\d+', sim_name)
        if match:
            return int(match.group())
        return abs(hash(sim_name)) % 100000

    def __iter__(self) -> Iterator[List[Data]]:
        worker_info = get_worker_info()
        if worker_info is None:
            files_to_process = list(self.simulation_ids)
        else:
            total_files = len(self.simulation_ids)
            per_worker = int(math.ceil(total_files / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, total_files)
            files_to_process = self.simulation_ids[start:end]
        
        if self.shuffle:
            files_to_process = list(files_to_process)
            random.shuffle(files_to_process)
        
        buffer = []

        for sim_name in files_to_process:
            dataset_file = self.directory / f"{sim_name}.pt"
            
            # LOCATION-BASED mesh_id for MLS caching (0=White, 1=Iowa)
            location_mesh_id = self._get_location_mesh_id(sim_name)
            sim_id_int = self._extract_sim_id(sim_name)
            
            try:
                sim_data = torch.load(dataset_file, weights_only=False)
                
                if not isinstance(sim_data, list):
                    continue
                
                # ALWAYS override mesh_id with location-based ID.
                # The pkl→pt conversion set mesh_id = sim number (e.g., 30, 348),
                # which breaks MLS caching when different sims share the same mesh.
                for data in sim_data:
                    data.mesh_id = torch.tensor([location_mesh_id], dtype=torch.long)
                    if not hasattr(data, 'sim_id') or data.sim_id is None:
                        data.sim_id = torch.tensor([sim_id_int], dtype=torch.long)

                T = len(sim_data)
                max_start = T - self.seq_len
                
                if max_start < 0:
                    continue
                
                for start_idx in range(0, max_start + 1, self.stride):
                    window = []
                    valid = True
                    for offset in range(self.seq_len):
                        t = start_idx + offset
                        data_t = sim_data[t].clone()
                        
                        if not hasattr(data_t, 'x') or not hasattr(data_t, 'y'):
                            valid = False
                            break
                            
                        window.append(data_t)
                    
                    if valid and len(window) == self.seq_len:
                        if self.shuffle:
                            buffer.append(window)
                            if len(buffer) >= self.buffer_size:
                                random.shuffle(buffer)
                                while buffer:
                                    yield buffer.pop()
                        else:
                            yield window
            
            except Exception as e:
                print(f"Error loading {dataset_file}: {e}")
                continue
        
        if self.shuffle and buffer:
            random.shuffle(buffer)
            while buffer:
                yield buffer.pop()
    
    def get_simulation_info(self, sim_id: str = None) -> dict:
        if sim_id is None:
            return {
                "num_simulations": len(self.simulation_ids),
                "seq_len": self.seq_len,
                "stride": self.stride,
                "num_static_feats": self.num_static_feats,
                "num_dynamic_feats": self.num_dynamic_feats,
                "mesh_groups": {k: len(v) for k, v in self.mesh_groups.items()},
            }
        
        path = self.directory / f"{sim_id}.pt"
        if not path.exists():
            return {"error": f"File not found: {path}"}
        
        sim_data = torch.load(path, weights_only=False)
        data_0 = sim_data[0] if isinstance(sim_data, list) else sim_data
        
        return {
            "sim_id": sim_id,
            "num_timesteps": len(sim_data) if isinstance(sim_data, list) else 1,
            "num_nodes": data_0.x.shape[0],
            "num_edges": data_0.edge_index.shape[1],
            "x_shape": list(data_0.x.shape),
            "y_shape": list(data_0.y.shape),
            "pos_shape": list(data_0.pos.shape) if hasattr(data_0, 'pos') else None,
            "river_location": "iowa_river" if "iw" in sim_id.lower() else "white_river",
        }

    def estimate_length(self) -> int:
        total = 0
        for sim_name in self.simulation_ids:
            path = self.directory / f"{sim_name}.pt"
            try:
                sim_data = torch.load(path, weights_only=False)
                T = len(sim_data) if isinstance(sim_data, list) else 1
                num_windows = max(0, (T - self.seq_len) // self.stride + 1)
                total += num_windows
            except:
                continue
        return total


def get_simulation_ids(directory, file_pattern="*.pt", river_filter=None):
    directory = Path(directory)
    files = sorted(list(directory.glob(file_pattern)))
    sim_ids = [f.stem for f in files]
    if river_filter == 'iowa':
        sim_ids = [s for s in sim_ids if 'iw' in s.lower()]
    elif river_filter == 'white':
        sim_ids = [s for s in sim_ids if 'iw' not in s.lower()]
    return sim_ids


def collate_windows(batch):
    return batch


@torch.no_grad()
def get_node_coords(data_t, num_static_feats=9):
    if hasattr(data_t, "pos") and data_t.pos is not None:
        return data_t.pos
    return data_t.x[:, :2]


def get_static_feats(data_t, num_static_feats=9):
    return data_t.x[:, :num_static_feats]


def get_dynamic_feats(data_t, num_static_feats=9, num_dynamic_feats=4):
    return data_t.x[:, num_static_feats:num_static_feats + num_dynamic_feats]


def get_target(data_t, num_dynamic_feats=4):
    return data_t.y[:, :num_dynamic_feats]