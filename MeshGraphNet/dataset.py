"""
Dataset for PLAID 2D Elastoplastic Dynamics simulations.

Adapted from MeshGraphNets implementation to work with preprocessed .pt files.
"""

import math
import torch
import numpy as np
import re
from pathlib import Path
from typing import List, Iterator, Union
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data
from tqdm import tqdm


class ElastoPlasticDataset(IterableDataset):
    """
    Dataset for PLAID 2D Elasto-Plasto-Dynamics simulations.
    
    Features:
    - Loads sequences from .pt files containing timestep data
    - Injects 'mesh_id' into Data objects for operator caching
    - Handles multi-processing correctly via worker sharding
    - Compatible with both G-PARC and MeshGraphNets
    """
    
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
        
        print(f"ElastoPlasticDataset initialized:")
        print(f"  Directory: {self.directory}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Stride: {stride}")
        print(f"  Found {len(self.simulation_ids)} simulation files")
    
    def _discover_simulation_ids(self) -> List[str]:
        """Discover all simulation files in directory."""
        files = list(self.directory.glob(self.file_pattern))
        return [file.stem for file in files]
    
    def _extract_id_from_name(self, sim_name: str) -> int:
        """
        Extract integer ID from filenames like 'simulation_123' or '123'.
        
        Args:
            sim_name: Simulation filename stem
            
        Returns:
            Unique integer ID for this simulation
        """
        match = re.search(r'\d+', sim_name)
        if match:
            return int(match.group())
        # Fallback hash if no number found
        return abs(hash(sim_name)) % 100000

    def __iter__(self) -> Iterator[List[Data]]:
        """
        Iterate through all simulation files and yield sequences.
        
        Yields:
            window: List of Data objects representing a temporal sequence
        """
        
        # Worker sharding for multi-process data loading
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

        # Iteration loop
        for sim_name in files_to_process:
            dataset_file = self.directory / f"{sim_name}.pt"
            
            # Extract unique ID for operator caching
            sim_id_int = self._extract_id_from_name(sim_name)
            
            try:
                sim_data = torch.load(dataset_file, weights_only=False)
                
                if not isinstance(sim_data, list):
                    continue
                
                # INJECT MESH_ID for operator caching
                # This prevents cache collisions between different meshes!
                for data in sim_data:
                    data.mesh_id = torch.tensor([sim_id_int], dtype=torch.long)

                T = len(sim_data)
                max_start = T - self.seq_len
                
                if max_start < 0:
                    continue
                
                # Create sliding windows
                for start_idx in range(0, max_start + 1, self.stride):
                    window = []
                    for offset in range(self.seq_len):
                        t = start_idx + offset
                        data_t = sim_data[t].clone()
                        
                        # Basic validation
                        if not hasattr(data_t, 'x') or not hasattr(data_t, 'y'):
                            break
                            
                        window.append(data_t)
                    
                    if len(window) == self.seq_len:
                        yield window
            
            except Exception as e:
                print(f"Error loading {dataset_file}: {e}")
                continue
    
    def get_simulation_info(self, sim_id: str = None) -> dict:
        """
        Get information about a simulation.
        
        Args:
            sim_id: Simulation ID to inspect (default: first simulation)
            
        Returns:
            Dictionary with simulation metadata
        """
        if sim_id is None and self.simulation_ids:
            sim_id = self.simulation_ids[0]
        elif sim_id not in self.simulation_ids:
            return {}
        
        dataset_file = self.directory / f"{sim_id}.pt"
        try:
            sim_data = torch.load(dataset_file, weights_only=False)
            if not isinstance(sim_data, list) or len(sim_data) == 0:
                return {"error": "Invalid or empty simulation data"}
            
            sample_data = sim_data[0]
            info = {
                "simulation_id": sim_id,
                "total_timesteps": len(sim_data),
                "num_nodes": sample_data.num_nodes,
                "num_features": sample_data.num_features,
                "feature_shape": sample_data.x.shape,
                "target_shape": sample_data.y.shape,
            }
            
            if hasattr(sample_data, 'edge_index'):
                info['num_edges'] = sample_data.edge_index.shape[1]
            if hasattr(sample_data, 'elements'):
                info['num_elements'] = sample_data.elements.shape[0]
                
            return info
        except Exception as e:
            return {"error": f"Could not load simulation info: {e}"}

    def __len__(self):
        """
        Estimate total number of sequences.
        
        Returns:
            Estimated length based on sampling first few simulations
        """
        if not hasattr(self, '_estimated_length'):
            sample_count = min(5, len(self.simulation_ids))
            if sample_count == 0:
                return 0
            
            est_total = 0
            valid_samples = 0
            
            for sim_id in self.simulation_ids[:sample_count]:
                try:
                    f = self.directory / f"{sim_id}.pt"
                    data = torch.load(f, weights_only=False)
                    max_start = len(data) - self.seq_len
                    if max_start >= 0:
                        est_total += (max_start // self.stride) + 1
                        valid_samples += 1
                except:
                    continue
            
            if valid_samples > 0:
                avg = est_total / valid_samples
                self._estimated_length = int(avg * len(self.simulation_ids))
            else:
                self._estimated_length = len(self.simulation_ids) * 10
        
        return self._estimated_length


def create_datasets_from_folders(base_dir: Union[str, Path], 
                                seq_len: int = 10, 
                                stride: int = 1,
                                num_static_feats: int = 2, 
                                num_dynamic_feats: int = 2,
                                use_element_features: bool = False):
    """
    Create datasets from train/val/test folder structure.
    
    Args:
        base_dir: Base directory containing train/val/test folders
        seq_len: Sequence length for temporal windows
        stride: Stride for sliding window
        num_static_feats: Number of static features (positions)
        num_dynamic_feats: Number of dynamic features (displacements)
        use_element_features: Whether to use element-level features
        
    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    base_dir = Path(base_dir)
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = base_dir / split
        if split_dir.exists():
            datasets[split] = ElastoPlasticDataset(
                directory=split_dir,
                seq_len=seq_len,
                stride=stride,
                num_static_feats=num_static_feats,
                num_dynamic_feats=num_dynamic_feats,
                use_element_features=use_element_features
            )
            print(f"✅ Created {split} dataset with {len(datasets[split].simulation_ids)} simulations")
        else:
            print(f"⚠️  {split_dir} not found, skipping {split} dataset")
    
    return datasets


if __name__ == "__main__":
    # Test the dataset
    print("Testing ElastoPlasticDataset...")
    
    test_dir = Path("/scratch/jtb3sud/processed_elasto_plastic/zscore/normalized/train")
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
    else:
        dataset = ElastoPlasticDataset(
            directory=test_dir,
            seq_len=5,
            stride=1,
            num_static_feats=2,
            num_dynamic_feats=2,
            use_element_features=False
        )
        
        # Get simulation info
        print("\n📋 Simulation Information:")
        info = dataset.get_simulation_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test iteration
        print("\n🧪 Testing dataset iteration...")
        for i, sequence in enumerate(dataset):
            if i >= 2:  # Just test first 2 sequences
                break
            print(f"\nSequence {i}: {len(sequence)} timesteps")
            for j, data in enumerate(sequence[:3]):  # First 3 timesteps
                print(f"  Step {j}:")
                print(f"    x.shape: {data.x.shape}")
                print(f"    y.shape: {data.y.shape}")
                print(f"    pos.shape: {data.pos.shape}")
                print(f"    edge_index.shape: {data.edge_index.shape}")
                if hasattr(data, 'mesh_id'):
                    print(f"    mesh_id: {data.mesh_id.item()}")
        
        print("\n✅ Dataset test complete!")