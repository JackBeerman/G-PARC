import math
import torch
import numpy as np
import random
import re
from pathlib import Path
from typing import List, Iterator, Union, Dict
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data
from tqdm import tqdm

class ElastoPlasticDataset(IterableDataset):
    """
    Dataset for PLAID 2D Elasto-Plasto-Dynamics simulations.
    
    Updated:
    - Injects 'mesh_id' into Data objects for Operator Caching.
    - Handles Multi-Processing correctly.
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
        
        self.var_names = ['U_x', 'U_y'] 
        if self.use_element_features:
            self.element_var_names = ['EROSION_STATUS']
        
        print(f"ElastoPlasticDataset initialized:")
        print(f"  Directory: {self.directory}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Stride: {stride}")
        print(f"  Found {len(self.simulation_ids)} simulation files")
    
    def _discover_simulation_ids(self) -> List[str]:
        files = list(self.directory.glob(self.file_pattern))
        return [file.stem for file in files]
    
    def _extract_id_from_name(self, sim_name: str) -> int:
        """Extracts integer ID from filenames like 'simulation_123' or '123'."""
        match = re.search(r'\d+', sim_name)
        if match:
            return int(match.group())
        # Fallback hash if no number found
        return abs(hash(sim_name)) % 100000

    def __iter__(self) -> Iterator[List[Data]]:
        """Iterate through all simulation files and yield sequences."""
        
        # --- 1. WORKER SHARDING ---
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
    
        # --- 2. ITERATION LOOP ---
        for sim_name in files_to_process:
            dataset_file = self.directory / f"{sim_name}.pt"
            
            try:
                sim_data = torch.load(dataset_file, weights_only=False)
                
                if not isinstance(sim_data, list):
                    continue
                
                # --- INJECT MESH ID BASED ON TOPOLOGY ---
                # Same mesh topology = same mesh_id (for operator caching)
                first_snapshot = sim_data[0]
                mesh_signature = (first_snapshot.num_nodes, first_snapshot.num_edges)
                mesh_id_int = abs(hash(mesh_signature)) % 1000000
                
                for data in sim_data:
                    data.mesh_id = torch.tensor([mesh_id_int], dtype=torch.long)
    
                T = len(sim_data)
                max_start = T - self.seq_len
                
                if max_start < 0:
                    continue
                
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


    def get_feature_stats(self, max_simulations: int = None, analyze_static_dynamic: bool = True) -> dict:
        """Fixed version that handles different mesh sizes."""
        if not self.simulation_ids: 
            return {}
        
        sims_to_analyze = self.simulation_ids[:max_simulations] if max_simulations else self.simulation_ids
        
        all_features = []
        all_targets = []
        
        print(f"Calculating feature statistics from {len(sims_to_analyze)} simulations...")
        for sim_id in tqdm(sims_to_analyze, desc="Loading simulations"):
            dataset_file = self.directory / f"{sim_id}.pt"
            try:
                sim_data = torch.load(dataset_file, weights_only=False)
                if isinstance(sim_data, list) and len(sim_data) > 0:
                    # Flatten each simulation: [T, N, F] -> [T*N, F]
                    sim_features = torch.cat([data.x for data in sim_data], dim=0)
                    sim_targets = torch.cat([data.y for data in sim_data], dim=0)
                    all_features.append(sim_features)
                    all_targets.append(sim_targets)
            except: 
                continue
        
        if not all_features: 
            return {"error": "No valid simulation data found"}
        
        # Concatenate all flattened data
        all_features = torch.cat(all_features, dim=0)  # [total_node_timesteps, F]
        all_targets = torch.cat(all_targets, dim=0)    # [total_node_timesteps, 2]
        
        stats = {
            'feature_means': all_features.mean(dim=0),
            'feature_stds': all_features.std(dim=0),
            'target_means': all_targets.mean(dim=0),
            'target_stds': all_targets.std(dim=0),
            'total_samples': all_features.shape[0]
        }
        return stats
    
    def get_simulation_info(self, sim_id: str = None) -> dict:
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
            return info
        except Exception as e:
            return {"error": f"Could not load simulation info: {e}"}


    def __len__(self):
        if not hasattr(self, '_estimated_length'):
            sample_count = min(5, len(self.simulation_ids))
            if sample_count == 0: return 0
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
                except: continue
            
            if valid_samples > 0:
                avg = est_total / valid_samples
                self._estimated_length = int(avg * len(self.simulation_ids))
            else:
                self._estimated_length = len(self.simulation_ids) * 10
        return self._estimated_length
    
    def __len__(self):
        """Estimate total number of sequences."""
        if not hasattr(self, '_estimated_length'):
            sample_count = min(5, len(self.simulation_ids))
            if sample_count == 0: return 0
            
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


def get_simulation_ids(directory: Path, pattern: str = "*.pt") -> List[str]:
    files = list(directory.glob(pattern))
    return [file.stem for file in files]

def create_datasets_from_folders(base_dir: Union[str, Path], seq_len: int = 10, stride: int = 1,
                                num_static_feats: int = 2, num_dynamic_feats: int = 2,
                                use_element_features: bool = False):
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


def test_elastoplastic_dataset():
    """Test function to verify the dataset works correctly."""
    test_dir = Path("/scratch/jtb3sud/processed_elasto_plastic/zscore/normalized")
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    # Test with train split
    train_dir = test_dir / "train"
    
    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return
    
    dataset = ElastoPlasticDataset(
        directory=train_dir,
        seq_len=5,
        stride=1,
        num_static_feats=2,  # x_pos, y_pos
        num_dynamic_feats=2,  # U_x, U_y
        use_element_features=False
    )
    
    if len(dataset.simulation_ids) == 0:
        print("❌ No simulation files found for testing")
        return
    
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
        for j, data in enumerate(sequence):
            print(f"  Step {j}:")
            print(f"    x.shape: {data.x.shape}")
            print(f"    y.shape: {data.y.shape}")
            print(f"    pos.shape: {data.pos.shape}")
            print(f"    edge_index.shape: {data.edge_index.shape}")
            
            # Check static/dynamic feature separation
            static_feats = data.x[:, :dataset.num_static_feats]
            dynamic_feats = data.x[:, dataset.num_static_feats:dataset.num_static_feats + dataset.num_dynamic_feats]
            print(f"    Static feats (pos): {static_feats.shape}")
            print(f"    Dynamic feats (U): {dynamic_feats.shape}")
            
            # Check if element features exist
            if hasattr(data, 'x_element'):
                print(f"    x_element.shape: {data.x_element.shape}")
            if hasattr(data, 'y_element'):
                print(f"    y_element.shape: {data.y_element.shape}")
    
    # Get feature statistics
    print("\n📊 Feature statistics:")
    stats = dataset.get_feature_stats(max_simulations=10, analyze_static_dynamic=True)
    
    print("\n✅ Dataset test complete!")


def inspect_dataset_features(dataset_path: Union[str, Path], num_static_feats: int = 2, num_dynamic_feats: int = 2):
    """
    Utility function to inspect the feature structure of a dataset file.
    
    Args:
        dataset_path: Path to a single .pt dataset file
        num_static_feats: Expected number of static features
        num_dynamic_feats: Expected number of dynamic features
    """
    try:
        data = torch.load(dataset_path, weights_only=False)
        
        if not isinstance(data, list) or len(data) == 0:
            print("❌ Invalid dataset format")
            return
        
        sample = data[0]
        print(f"🔍 Dataset Feature Inspection: {dataset_path}")
        print(f"  Total timesteps: {len(data)}")
        print(f"  Sample data structure:")
        print(f"    x.shape: {sample.x.shape}")
        print(f"    y.shape: {sample.y.shape}")
        print(f"    pos.shape: {sample.pos.shape if hasattr(sample, 'pos') else 'N/A'}")
        print(f"    num_nodes: {sample.num_nodes}")
        print(f"    num_edges: {sample.num_edges}")
        
        if hasattr(sample, 'elements'):
            print(f"    elements.shape: {sample.elements.shape}")
        if hasattr(sample, 'x_element'):
            print(f"    x_element.shape: {sample.x_element.shape}")
        if hasattr(sample, 'y_element'):
            print(f"    y_element.shape: {sample.y_element.shape}")
        
        # Feature analysis
        total_feats = sample.x.shape[1]
        print(f"\n📊 Feature Analysis:")
        print(f"  Total features: {total_feats}")
        print(f"  Expected static: {num_static_feats}")
        print(f"  Expected dynamic: {num_dynamic_feats}")
        
        if total_feats >= num_static_feats + num_dynamic_feats:
            static_sample = sample.x[:5, :num_static_feats]  # First 5 nodes
            dynamic_sample = sample.x[:5, num_static_feats:num_static_feats + num_dynamic_feats]
            
            print(f"\n  Static features (positions, first 5 nodes):")
            print(f"    {static_sample}")
            print(f"\n  Dynamic features (displacements, first 5 nodes):")
            print(f"    {dynamic_sample}")
        else:
            print(f"  ⚠️ Warning: Total features ({total_feats}) < expected ({num_static_feats + num_dynamic_feats})")
            
    except Exception as e:
        print(f"❌ Error inspecting dataset: {e}")


if __name__ == "__main__":
    # Run test
    test_elastoplastic_dataset()