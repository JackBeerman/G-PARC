import pickle
from pathlib import Path
from typing import List, Iterator, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GraphUNet
from tqdm import tqdm
import matplotlib.pyplot as plt


class ShockTubeRolloutDataset(IterableDataset):
    """
    Dataset for shock tube simulations with multiple PyG dataset files.
    Yields consecutive *shifted* sequences of timesteps from multiple simulations:
      [ (Data[t].x, Data[t+1].y), (Data[t+1].x, Data[t+2].y), ..., (Data[t+seq_len-1].x, Data[t+seq_len].y) ]
    for each simulation file.
    
    Now extracts global attributes (pressure, density, delta_t) and attaches them to each Data object.
    """
    def __init__(self,
                 directory: Union[str, Path],
                 simulation_ids: List[str] = None,
                 seq_len: int = 10,
                 stride: int = 1,
                 num_static_feats: int = 2,
                 num_dynamic_feats: int = 4,
                 file_pattern: str = "*.pt"):
        """
        Args:
            directory (Union[str, Path]): Directory containing PyG dataset files (.pt files).
            simulation_ids (List[str]): List of simulation IDs (file stems). If None, auto-discover.
            seq_len (int): Number of consecutive steps in each returned window.
            stride (int): Step size between sequence starts (default: 1 for overlapping sequences).
            num_static_feats (int): Number of static features (e.g., node positions, material properties).
            num_dynamic_feats (int): Number of dynamic features (e.g., density, momentum, energy).
            file_pattern (str): Pattern to match simulation files (default: "*.pt").
        """
        super().__init__()
        self.directory = Path(directory)
        self.seq_len = seq_len
        self.stride = stride
        self.file_pattern = file_pattern
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        
        # Auto-discover simulation IDs if not provided
        if simulation_ids is None:
            self.simulation_ids = self._discover_simulation_ids()
        else:
            self.simulation_ids = simulation_ids
        
        # Variables in shock tube data
        self.var_names = ['density', 'x_momentum', 'y_momentum', 'total_energy']
        
        print(f"ShockTubeRolloutDataset initialized:")
        print(f"  Directory: {self.directory}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Stride: {stride}")
        print(f"  Static features: {num_static_feats}")
        print(f"  Dynamic features: {num_dynamic_feats}")
        print(f"  Found {len(self.simulation_ids)} simulation files")
        print(f"  Variables: {self.var_names}")
        if len(self.simulation_ids) > 0:
            print(f"  Example files: {self.simulation_ids[:3]}")
    
    def _discover_simulation_ids(self) -> List[str]:
        """Auto-discover simulation files in the directory."""
        files = list(self.directory.glob(self.file_pattern))
        return [file.stem for file in files]
    
    def __iter__(self) -> Iterator[List[Data]]:
        """
        Iterate through all simulation files and yield sequences.
        Now extracts global attributes (pressure, density, delta_t) from each timestep.
        """
        for sim_id in self.simulation_ids:
            dataset_file = self.directory / f"{sim_id}.pt"
            
            try:
                # Load the simulation data
                sim_data = torch.load(dataset_file, weights_only=False)
                
                if not isinstance(sim_data, list):
                    print(f"Unexpected data format in {dataset_file}. Expected list, got {type(sim_data)}. Skipping.")
                    continue
                
                T = len(sim_data)
                max_start = T - self.seq_len
                
                if max_start < 0:
                    print(f"Simulation {sim_id} has only {T} timesteps, need at least {self.seq_len}. Skipping.")
                    continue
                
                # Generate sequences with specified stride
                for start_idx in range(0, max_start + 1, self.stride):
                    window = []
                    for offset in range(self.seq_len):
                        t = start_idx + offset
                        data_t = sim_data[t].clone()
                        
                        # Ensure the data has the expected structure
                        if not hasattr(data_t, 'x') or not hasattr(data_t, 'y'):
                            print(f"Warning: Data at timestep {t} in {sim_id} missing x or y. Skipping sequence.")
                            break
                        
                        # --- START OF MODIFICATION ---
                        # Correctly extract and name the global attributes
                        # Assuming 'global_params' is the name of the tensor containing [pressure, density, delta_t]
                        if hasattr(data_t, 'global_params'):
                            global_tensor = data_t.global_params
                            if global_tensor.numel() >= 3:
                                data_t.global_pressure = global_tensor[0].unsqueeze(0)
                                data_t.global_density = global_tensor[1].unsqueeze(0)
                                data_t.global_delta_t = global_tensor[2].unsqueeze(0)
                            else:
                                print(f"Warning: 'global_params' tensor in {sim_id} at step {t} has unexpected size. Assigning zeros.")
                                data_t.global_pressure = torch.zeros(1, dtype=torch.float32)
                                data_t.global_density = torch.zeros(1, dtype=torch.float32)
                                data_t.global_delta_t = torch.zeros(1, dtype=torch.float32)
                        else:
                            # Fallback if global_params does not exist
                            print(f"Warning: 'global_params' not found in {sim_id} at step {t}. Assigning zeros.")
                            data_t.global_pressure = torch.zeros(1, dtype=torch.float32)
                            data_t.global_density = torch.zeros(1, dtype=torch.float32)
                            data_t.global_delta_t = torch.zeros(1, dtype=torch.float32)
                        # --- END OF MODIFICATION ---

                        # The rest of your code to partition x and clone y remains unchanged.
                        
                        window.append(data_t)
                    
                    # Only yield complete windows
                    if len(window) == self.seq_len:
                        yield window
            
            except FileNotFoundError:
                print(f"Dataset file not found: {dataset_file}")
            except Exception as e:
                print(f"Error loading {dataset_file}: {e}")
    
    def get_simulation_info(self, sim_id: str = None) -> dict:
        """Get information about a specific simulation or the first available one."""
        if sim_id is None and self.simulation_ids:
            sim_id = self.simulation_ids[0]
        elif sim_id not in self.simulation_ids:
            print(f"Simulation {sim_id} not found in available simulations.")
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
                "num_edges": sample_data.num_edges,
                "feature_shape": sample_data.x.shape,
                "target_shape": sample_data.y.shape,
                "has_positions": hasattr(sample_data, 'pos'),
                "is_directed": sample_data.is_directed() if hasattr(sample_data, 'is_directed') else "Unknown",
                # Check for global attributes
                "has_global_pressure": hasattr(sample_data, 'pressure'),
                "has_global_density": hasattr(sample_data, 'density'),
                "has_global_delta_t": hasattr(sample_data, 'delta_t')
            }
            
            # Add sample global attribute values if they exist
            if hasattr(sample_data, 'pressure'):
                info["sample_pressure"] = sample_data.pressure.item() if hasattr(sample_data.pressure, 'item') else sample_data.pressure
            if hasattr(sample_data, 'density'):
                info["sample_density"] = sample_data.density.item() if hasattr(sample_data.density, 'item') else sample_data.density
            if hasattr(sample_data, 'delta_t'):
                info["sample_delta_t"] = sample_data.delta_t.item() if hasattr(sample_data.delta_t, 'item') else sample_data.delta_t
            
            return info
            
        except Exception as e:
            return {"error": f"Could not load simulation info: {e}"}
    
    def get_feature_stats(self, max_simulations: int = None, analyze_static_dynamic: bool = True, 
                         analyze_global_attrs: bool = True) -> dict:
        """
        Calculate feature statistics across simulations for normalization.
        
        Args:
            max_simulations (int): Limit number of simulations to analyze (for speed).
            analyze_static_dynamic (bool): Whether to separate static and dynamic feature statistics.
            analyze_global_attrs (bool): Whether to analyze global attribute statistics.
        """
        if not self.simulation_ids:
            return {}
        
        sims_to_analyze = self.simulation_ids[:max_simulations] if max_simulations else self.simulation_ids
        
        all_features = []
        all_targets = []
        global_attrs = {'pressure': [], 'density': [], 'delta_t': []}
        
        print(f"Calculating feature statistics from {len(sims_to_analyze)} simulations...")
        
        for sim_id in tqdm(sims_to_analyze, desc="Loading simulations"):
            dataset_file = self.directory / f"{sim_id}.pt"
            
            try:
                sim_data = torch.load(dataset_file, weights_only=False)
                
                if isinstance(sim_data, list) and len(sim_data) > 0:
                    # Stack features and targets from this simulation
                    sim_features = torch.stack([data.x for data in sim_data])  # [T, N, F]
                    sim_targets = torch.stack([data.y for data in sim_data])   # [T, N, F]
                    
                    all_features.append(sim_features)
                    all_targets.append(sim_targets)
                    
                    # Collect global attributes if analyzing them
                    if analyze_global_attrs:
                        for data in sim_data:
                            if hasattr(data, 'pressure'):
                                global_attrs['pressure'].append(data.pressure)
                            if hasattr(data, 'density'):
                                global_attrs['density'].append(data.density)
                            if hasattr(data, 'delta_t'):
                                global_attrs['delta_t'].append(data.delta_t)
                    
            except Exception as e:
                print(f"Error processing {sim_id}: {e}")
                continue
        
        if not all_features:
            return {"error": "No valid simulation data found"}
        
        # Concatenate all data along time dimension
        all_features = torch.cat(all_features, dim=0)  # [Total_T, N, F]
        all_targets = torch.cat(all_targets, dim=0)    # [Total_T, N, F]
        
        # Calculate overall statistics
        feature_means = all_features.mean(dim=[0, 1])  # [F]
        feature_stds = all_features.std(dim=[0, 1])    # [F]
        target_means = all_targets.mean(dim=[0, 1])    # [F]
        target_stds = all_targets.std(dim=[0, 1])      # [F]
        
        stats = {
            'feature_means': feature_means,
            'feature_stds': feature_stds,
            'target_means': target_means,
            'target_stds': target_stds,
            'variable_names': self.var_names,
            'total_samples': all_features.shape[0],
            'num_nodes': all_features.shape[1],
            'num_features': all_features.shape[2],
            'num_static_feats': self.num_static_feats,
            'num_dynamic_feats': self.num_dynamic_feats
        }
        
        # Calculate global attribute statistics
        if analyze_global_attrs:
            global_stats = {}
            for attr_name, attr_values in global_attrs.items():
                if attr_values:
                    attr_tensor = torch.stack(attr_values)
                    global_stats[f'global_{attr_name}_mean'] = attr_tensor.mean()
                    global_stats[f'global_{attr_name}_std'] = attr_tensor.std()
                    global_stats[f'global_{attr_name}_min'] = attr_tensor.min()
                    global_stats[f'global_{attr_name}_max'] = attr_tensor.max()
                    global_stats[f'global_{attr_name}_count'] = len(attr_values)
            
            stats.update(global_stats)
        
        # Separate static and dynamic feature statistics if requested
        if analyze_static_dynamic:
            static_feats = all_features[:, :, :self.num_static_feats]
            dynamic_feats = all_features[:, :, self.num_static_feats:self.num_static_feats + self.num_dynamic_feats]
            
            static_means = static_feats.mean(dim=[0, 1])
            static_stds = static_feats.std(dim=[0, 1])
            dynamic_means = dynamic_feats.mean(dim=[0, 1])
            dynamic_stds = dynamic_feats.std(dim=[0, 1])
            
            stats.update({
                'static_feature_means': static_means,
                'static_feature_stds': static_stds,
                'dynamic_feature_means': dynamic_means,
                'dynamic_feature_stds': dynamic_stds
            })
        
        print(f"\nFeature Statistics (from {len(sims_to_analyze)} simulations):")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Static features: {self.num_static_feats}, Dynamic features: {self.num_dynamic_feats}")
        
        if analyze_static_dynamic:
            print(f"\nStatic Feature Statistics:")
            for i in range(self.num_static_feats):
                print(f"  Static feature {i}: mean={static_means[i]:.6f}, std={static_stds[i]:.6f}")
            
            print(f"\nDynamic Feature Statistics:")
            for i, var in enumerate(self.var_names[:self.num_dynamic_feats]):
                idx = i  # Index within dynamic features
                print(f"  {var}: mean={dynamic_means[idx]:.6f}, std={dynamic_stds[idx]:.6f}")
        
        # Print global attribute statistics
        if analyze_global_attrs:
            print(f"\nGlobal Attribute Statistics:")
            for attr_name in ['pressure', 'density', 'delta_t']:
                if f'global_{attr_name}_count' in stats:
                    count = stats[f'global_{attr_name}_count']
                    mean_val = stats[f'global_{attr_name}_mean']
                    std_val = stats[f'global_{attr_name}_std']
                    min_val = stats[f'global_{attr_name}_min']
                    max_val = stats[f'global_{attr_name}_max']
                    print(f"  {attr_name}: count={count}, mean={mean_val:.6f}, std={std_val:.6f}, range=[{min_val:.6f}, {max_val:.6f}]")
        
        return stats
    
    def __len__(self):
        """Estimate total number of sequences across all simulations."""
        if not hasattr(self, '_estimated_length'):
            self._estimated_length = 0
            for sim_id in self.simulation_ids[:5]:  # Sample a few to estimate
                try:
                    dataset_file = self.directory / f"{sim_id}.pt"
                    sim_data = torch.load(dataset_file, weights_only=False)
                    if isinstance(sim_data, list):
                        T = len(sim_data)
                        max_start = T - self.seq_len
                        if max_start >= 0:
                            sequences_per_sim = (max_start // self.stride) + 1
                            self._estimated_length += sequences_per_sim
                except:
                    continue
            
            # Extrapolate to all simulations
            if len(self.simulation_ids) > 5:
                avg_per_sim = self._estimated_length / min(5, len(self.simulation_ids))
                self._estimated_length = int(avg_per_sim * len(self.simulation_ids))
        
        return self._estimated_length


def get_simulation_ids(directory: Path, pattern: str = "*.pt") -> List[str]:
    """
    Get list of simulation IDs from a directory (equivalent to get_hydrograph_ids).
    
    Args:
        directory (Path): Directory containing simulation files.
        pattern (str): File pattern to match.
    
    Returns:
        List[str]: List of simulation file stems.
    """
    files = list(directory.glob(pattern))
    return [file.stem for file in files]


# Example usage and testing functions
def create_datasets_from_folders(base_dir: Union[str, Path], seq_len: int = 10, stride: int = 1, 
                                num_static_feats: int = 2, num_dynamic_feats: int = 4):
    """
    Create train/val/test datasets from organized folder structure.
    
    Args:
        base_dir: Base directory containing train/val/test folders
        seq_len: Sequence length for rollout windows  
        stride: Stride between sequence starts
        num_static_feats: Number of static features (e.g., node positions)
        num_dynamic_feats: Number of dynamic features (e.g., physical variables)
    
    Expected structure:
    base_dir/
    ├── train/
    │   ├── simulation_001.pt
    │   ├── simulation_002.pt
    │   └── ...
    ├── val/
    │   ├── simulation_101.pt
    │   └── ...
    └── test/
        ├── simulation_201.pt
        └── ...
    """
    base_dir = Path(base_dir)
    
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = base_dir / split
        if split_dir.exists():
            datasets[split] = ShockTubeRolloutDataset(
                directory=split_dir,
                seq_len=seq_len,
                stride=stride,
                num_static_feats=num_static_feats,
                num_dynamic_feats=num_dynamic_feats
            )
            print(f"✅ Created {split} dataset with {len(datasets[split].simulation_ids)} simulations")
        else:
            print(f"⚠️  {split_dir} not found, skipping {split} dataset")
    
    return datasets


def test_shock_tube_dataset():
    """Test function to verify the dataset works correctly with global attributes."""
    # Test with current single file setup
    current_dir = Path(".")  # Assumes shock_tube_pyg_dataset.pt is in current directory
    
    dataset = ShockTubeRolloutDataset(
        directory=current_dir,
        seq_len=5,
        stride=1,
        num_static_feats=2,  # Assuming x,y coordinates as static features
        num_dynamic_feats=4  # density, x_momentum, y_momentum, total_energy
    )
    
    if len(dataset.simulation_ids) == 0:
        print("❌ No simulation files found for testing")
        return
    
    # Get simulation info
    print("\n📋 Simulation Information:")
    info = dataset.get_simulation_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test iteration and check for global attributes
    print("\n🧪 Testing dataset iteration...")
    for i, sequence in enumerate(dataset):
        if i >= 2:  # Just test first 2 sequences
            break
        print(f"Sequence {i}: {len(sequence)} timesteps")
        for j, data in enumerate(sequence):
            print(f"  Step {j}: x.shape={data.x.shape}, y.shape={data.y.shape}")
            
            # Check static/dynamic feature separation
            static_feats = data.x[:, :dataset.num_static_feats]
            dynamic_feats = data.x[:, dataset.num_static_feats:dataset.num_static_feats + dataset.num_dynamic_feats]
            print(f"    Static feats: {static_feats.shape}, Dynamic feats: {dynamic_feats.shape}")
            
            # Check global attributes
            global_attrs = []
            if hasattr(data, 'global_pressure'):
                global_attrs.append(f"pressure={data.global_pressure}")
            if hasattr(data, 'global_density'):
                global_attrs.append(f"density={data.global_density}")
            if hasattr(data, 'global_delta_t'):
                global_attrs.append(f"delta_t={data.global_delta_t}")
            
            if global_attrs:
                print(f"    Global attrs: {', '.join(global_attrs)}")
            else:
                print(f"    Global attrs: None found")
    
    # Get feature statistics including global attributes
    print("\n📊 Feature statistics:")
    stats = dataset.get_feature_stats(analyze_static_dynamic=True, analyze_global_attrs=True)
    
    # Test with DataLoader
    print("\n🔄 Testing with DataLoader...")
    train_loader = DataLoader(dataset, batch_size=2, num_workers=0)
    
    for i, batch in enumerate(train_loader):
        if i >= 1:  # Just test first batch
            break
        print(f"Batch {i}: {len(batch)} sequences")
        for seq_idx, sequence in enumerate(batch):
            print(f"  Sequence {seq_idx} length: {len(sequence)}")
            print(f"    First timestep x.shape: {sequence[0].x.shape}")
            print(f"    First timestep y.shape: {sequence[0].y.shape}")
            
            # Check if global attributes are preserved through DataLoader
            first_data = sequence[0]
            global_preserved = []
            if hasattr(first_data, 'global_pressure'):
                global_preserved.append("pressure")
            if hasattr(first_data, 'global_density'):
                global_preserved.append("density")
            if hasattr(first_data, 'global_delta_t'):
                global_preserved.append("delta_t")
            print(f"    Global attrs preserved: {global_preserved}")


# Additional utility function for feature inspection with global attributes
def inspect_dataset_features(dataset_path: Union[str, Path], num_static_feats: int = 2, num_dynamic_feats: int = 4):
    """
    Utility function to inspect the feature structure of a dataset file including global attributes.
    
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
        print(f"    num_nodes: {sample.num_nodes}")
        print(f"    num_edges: {sample.num_edges}")
        
        if hasattr(sample, 'pos'):
            print(f"    pos.shape: {sample.pos.shape}")
        
        # Check for global attributes
        print(f"\n🌍 Global Attributes:")
        global_attrs_found = []
        if hasattr(sample, 'pressure'):
            global_attrs_found.append(f"pressure: {sample.pressure}")
        if hasattr(sample, 'density'):
            global_attrs_found.append(f"density: {sample.density}")
        if hasattr(sample, 'delta_t'):
            global_attrs_found.append(f"delta_t: {sample.delta_t}")
        
        if global_attrs_found:
            for attr in global_attrs_found:
                print(f"    {attr}")
        else:
            print(f"    No global attributes found")
        
        # Feature analysis
        total_feats = sample.x.shape[1]
        print(f"\n📊 Feature Analysis:")
        print(f"  Total features: {total_feats}")
        print(f"  Expected static: {num_static_feats}")
        print(f"  Expected dynamic: {num_dynamic_feats}")
        
        if total_feats >= num_static_feats + num_dynamic_feats:
            static_sample = sample.x[:5, :num_static_feats]  # First 5 nodes
            dynamic_sample = sample.x[:5, num_static_feats:num_static_feats + num_dynamic_feats]
            
            print(f"\n  Static features (first 5 nodes):")
            print(f"    {static_sample}")
            print(f"\n  Dynamic features (first 5 nodes):")
            print(f"    {dynamic_sample}")
        else:
            print(f"  ⚠️ Warning: Total features ({total_feats}) < expected ({num_static_feats + num_dynamic_feats})")
            
    except Exception as e:
        print(f"❌ Error inspecting dataset: {e}")