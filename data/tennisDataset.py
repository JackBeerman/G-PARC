import pickle
from pathlib import Path
from typing import List, Iterator, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np


class TennisServeRolloutDataset(IterableDataset):
    """
    Dataset for tennis serve motion with multiple PyG dataset files.
    Yields consecutive sequences of timesteps from multiple serves:
      [ (Data[t].x, Data[t+1].y), (Data[t+1].x, Data[t+2].y), ..., (Data[t+seq_len-1].x, Data[t+seq_len].y) ]
    for each serve file.
    
    Extracts global attributes (server_id, serve_number, etc.) and attaches them to each Data object.
    """
    def __init__(self,
                 directory: Union[str, Path],
                 simulation_ids: List[str] = None,
                 seq_len: int = 10,
                 stride: int = 1,
                 num_static_feats: int = 3,
                 num_dynamic_feats: int = 3,
                 file_pattern: str = "*.pt",
                 normalization_stats: Dict = None):
        """
        Args:
            directory (Union[str, Path]): Directory containing PyG dataset files (.pt files).
            simulation_ids (List[str]): List of serve IDs (file stems). If None, auto-discover.
            seq_len (int): Number of consecutive steps in each returned window.
            stride (int): Step size between sequence starts (default: 1 for overlapping sequences).
            num_static_feats (int): Number of position features per joint (default: 3 for x,y,z).
            num_dynamic_feats (int): Number of velocity features per joint (default: 3 for x,y,z).
            file_pattern (str): Pattern to match serve files (default: "*.pt").
            normalization_stats (Dict): Normalization statistics for denormalization if needed.
        """
        super().__init__()
        self.directory = Path(directory)
        self.seq_len = seq_len
        self.stride = stride
        self.file_pattern = file_pattern
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.normalization_stats = normalization_stats
        
        # Auto-discover serve IDs if not provided
        if simulation_ids is None:
            self.simulation_ids = self._discover_simulation_ids()
        else:
            self.simulation_ids = simulation_ids
        
        # Tennis joint names
        self.joint_names = [
            "Pelvis", "R Hip", "R Knee", "R Ankle", "L Hip", "L Knee", "L Ankle", 
            "Spine", "Thorax", "Neck", "Head", "L Shoulder", "L Elbow", "L Wrist", 
            "R Shoulder", "R Elbow", "R Wrist"
        ]
        
        # Load player mapping if available
        self.player_mapping = self._load_player_mapping()
        
        print(f"TennisServeRolloutDataset initialized:")
        print(f"  Directory: {self.directory}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Stride: {stride}")
        print(f"  Position features: {num_static_feats}")
        print(f"  Velocity features: {num_dynamic_feats}")
        print(f"  Found {len(self.simulation_ids)} serve files")
        print(f"  Joints: {len(self.joint_names)}")
        if len(self.simulation_ids) > 0:
            print(f"  Example files: {self.simulation_ids[:3]}")
    
    def _discover_simulation_ids(self) -> List[str]:
        """Auto-discover serve files in the directory."""
        files = list(self.directory.glob(self.file_pattern))
        return [file.stem for file in files]
    
    def _load_player_mapping(self) -> Dict:
        """Load player mapping from parent directory if available."""
        try:
            player_mapping_path = self.directory.parent / 'player_mapping.json'
            if player_mapping_path.exists():
                with open(player_mapping_path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def __iter__(self) -> Iterator[List[Data]]:
        """
        Iterate through all serve files and yield sequences.
        Extracts global attributes (server_id, serve_number, etc.) from each timestep.
        """
        for simulation_id in self.simulation_ids:
            dataset_file = self.directory / f"{simulation_id}.pt"
            
            try:
                # Load the serve data
                serve_data = torch.load(dataset_file, weights_only=False)
                
                if not isinstance(serve_data, list):
                    print(f"Unexpected data format in {dataset_file}. Expected list, got {type(serve_data)}. Skipping.")
                    continue
                
                T = len(serve_data)
                max_start = T - self.seq_len
                
                if max_start < 0:
                    print(f"Serve {simulation_id} has only {T} timesteps, need at least {self.seq_len}. Skipping.")
                    continue
                
                # Generate sequences with specified stride
                for start_idx in range(0, max_start + 1, self.stride):
                    window = []
                    for offset in range(self.seq_len):
                        t = start_idx + offset
                        data_t = serve_data[t].clone()
                        
                        # Ensure the data has the expected structure
                        if not hasattr(data_t, 'x') or not hasattr(data_t, 'y'):
                            print(f"Warning: Data at timestep {t} in {simulation_id} missing x or y. Skipping sequence.")
                            break
                        
                        # Extract and properly name the global attributes
                        # These come from the tennis serve context
                        if hasattr(data_t, 'server_id'):
                            data_t.global_server_id = data_t.server_id.clone()
                        else:
                            data_t.global_server_id = torch.tensor([-1], dtype=torch.long)
                        
                        if hasattr(data_t, 'serve_number'):
                            data_t.global_serve_number = data_t.serve_number.clone()
                        else:
                            data_t.global_serve_number = torch.tensor([0.0], dtype=torch.float32)
                        
                        if hasattr(data_t, 'set_number'):
                            data_t.global_set_number = data_t.set_number.clone()
                        else:
                            data_t.global_set_number = torch.tensor([0.0], dtype=torch.float32)
                        
                        if hasattr(data_t, 'game_number'):
                            data_t.global_game_number = data_t.game_number.clone()
                        else:
                            data_t.global_game_number = torch.tensor([0.0], dtype=torch.float32)
                        
                        if hasattr(data_t, 'point_number'):
                            data_t.global_point_number = data_t.point_number.clone()
                        else:
                            data_t.global_point_number = torch.tensor([0.0], dtype=torch.float32)
                        
                        # Add simulation_id as a global attribute for tracking
                        data_t.global_simulation_id = simulation_id
                        
                        window.append(data_t)
                    
                    # Only yield complete windows
                    if len(window) == self.seq_len:
                        yield window
            
            except FileNotFoundError:
                print(f"Dataset file not found: {dataset_file}")
            except Exception as e:
                print(f"Error loading {dataset_file}: {e}")
    
    def get_serve_info(self, simulation_id: str = None) -> dict:
        """Get information about a specific serve or the first available one."""
        if simulation_id is None and self.simulation_ids:
            simulation_id = self.simulation_ids[0]
        elif simulation_id not in self.simulation_ids:
            print(f"Serve {simulation_id} not found in available serves.")
            return {}
        
        dataset_file = self.directory / f"{simulation_id}.pt"
        
        try:
            serve_data = torch.load(dataset_file, weights_only=False)
            
            if not isinstance(serve_data, list) or len(serve_data) == 0:
                return {"error": "Invalid or empty serve data"}
            
            sample_data = serve_data[0]
            
            info = {
                "simulation_id": simulation_id,
                "total_timesteps": len(serve_data),
                "num_joints": sample_data.num_nodes,
                "num_features": sample_data.num_features,
                "num_edges": sample_data.num_edges,
                "feature_shape": sample_data.x.shape,
                "target_shape": sample_data.y.shape,
                "has_positions": hasattr(sample_data, 'pos'),
                "is_directed": sample_data.is_directed() if hasattr(sample_data, 'is_directed') else "Unknown",
                # Check for global attributes
                "has_server_id": hasattr(sample_data, 'server_id'),
                "has_serve_number": hasattr(sample_data, 'serve_number'),
                "has_set_number": hasattr(sample_data, 'set_number'),
                "has_game_number": hasattr(sample_data, 'game_number'),
                "has_point_number": hasattr(sample_data, 'point_number')
            }
            
            # Add sample global attribute values if they exist
            if hasattr(sample_data, 'server_id'):
                server_id = sample_data.server_id.item() if hasattr(sample_data.server_id, 'item') else sample_data.server_id
                info["sample_server_id"] = server_id
                # Convert to player name if mapping available
                if self.player_mapping:
                    id_to_player = {v: k for k, v in self.player_mapping.items()}
                    info["sample_player_name"] = id_to_player.get(server_id, f"Unknown_{server_id}")
            
            if hasattr(sample_data, 'serve_number'):
                info["sample_serve_number"] = sample_data.serve_number.item() if hasattr(sample_data.serve_number, 'item') else sample_data.serve_number
            if hasattr(sample_data, 'set_number'):
                info["sample_set_number"] = sample_data.set_number.item() if hasattr(sample_data.set_number, 'item') else sample_data.set_number
            if hasattr(sample_data, 'game_number'):
                info["sample_game_number"] = sample_data.game_number.item() if hasattr(sample_data.game_number, 'item') else sample_data.game_number
            if hasattr(sample_data, 'point_number'):
                info["sample_point_number"] = sample_data.point_number.item() if hasattr(sample_data.point_number, 'item') else sample_data.point_number
            
            return info
            
        except Exception as e:
            return {"error": f"Could not load serve info: {e}"}
    
    def get_feature_stats(self, max_serves: int = None, analyze_position_velocity: bool = True, 
                         analyze_global_attrs: bool = True) -> dict:
        """
        Calculate feature statistics across serves for normalization.
        
        Args:
            max_serves (int): Limit number of serves to analyze (for speed).
            analyze_position_velocity (bool): Whether to separate position and velocity feature statistics.
            analyze_global_attrs (bool): Whether to analyze global attribute statistics.
        """
        if not self.simulation_ids:
            return {}
        
        serves_to_analyze = self.simulation_ids[:max_serves] if max_serves else self.simulation_ids
        
        all_features = []
        all_targets = []
        global_attrs = {
            'server_id': [], 'serve_number': [], 'set_number': [], 
            'game_number': [], 'point_number': []
        }
        
        print(f"Calculating feature statistics from {len(serves_to_analyze)} serves...")
        
        for simulation_id in tqdm(serves_to_analyze, desc="Loading serves"):
            dataset_file = self.directory / f"{simulation_id}.pt"
            
            try:
                serve_data = torch.load(dataset_file, weights_only=False)
                
                if isinstance(serve_data, list) and len(serve_data) > 0:
                    # Stack features and targets from this serve
                    serve_features = torch.stack([data.x for data in serve_data])  # [T, N, F]
                    serve_targets = torch.stack([data.y for data in serve_data])   # [T, N, F]
                    
                    all_features.append(serve_features)
                    all_targets.append(serve_targets)
                    
                    # Collect global attributes if analyzing them
                    if analyze_global_attrs:
                        for data in serve_data:
                            if hasattr(data, 'server_id'):
                                global_attrs['server_id'].append(data.server_id)
                            if hasattr(data, 'serve_number'):
                                global_attrs['serve_number'].append(data.serve_number)
                            if hasattr(data, 'set_number'):
                                global_attrs['set_number'].append(data.set_number)
                            if hasattr(data, 'game_number'):
                                global_attrs['game_number'].append(data.game_number)
                            if hasattr(data, 'point_number'):
                                global_attrs['point_number'].append(data.point_number)
                    
            except Exception as e:
                print(f"Error processing {simulation_id}: {e}")
                continue
        
        if not all_features:
            return {"error": "No valid serve data found"}
        
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
            'joint_names': self.joint_names,
            'total_samples': all_features.shape[0],
            'num_joints': all_features.shape[1],
            'num_features': all_features.shape[2],
            'num_static_feats': self.num_static_feats,
            'num_dynamic_feats': self.num_dynamic_feats
        }
        
        # Calculate global attribute statistics
        if analyze_global_attrs:
            global_stats = {}
            for attr_name, attr_values in global_attrs.items():
                if attr_values:
                    if attr_name == 'server_id':
                        # Handle integer IDs differently
                        attr_tensor = torch.stack(attr_values).flatten()
                        unique_ids = torch.unique(attr_tensor)
                        global_stats[f'global_{attr_name}_unique_count'] = len(unique_ids)
                        global_stats[f'global_{attr_name}_min'] = attr_tensor.min()
                        global_stats[f'global_{attr_name}_max'] = attr_tensor.max()
                    else:
                        # Handle float attributes
                        attr_tensor = torch.stack(attr_values).flatten()
                        global_stats[f'global_{attr_name}_mean'] = attr_tensor.mean()
                        global_stats[f'global_{attr_name}_std'] = attr_tensor.std()
                        global_stats[f'global_{attr_name}_min'] = attr_tensor.min()
                        global_stats[f'global_{attr_name}_max'] = attr_tensor.max()
                    
                    global_stats[f'global_{attr_name}_count'] = len(attr_values)
            
            stats.update(global_stats)
        
        # Separate position and velocity feature statistics if requested
        if analyze_position_velocity and all_features.shape[2] >= self.num_static_feats + self.num_dynamic_feats:
            position_feats = all_features[:, :, :self.num_static_feats]
            velocity_feats = all_features[:, :, self.num_static_feats:self.num_static_feats + self.num_dynamic_feats]
            
            position_means = position_feats.mean(dim=[0, 1])
            position_stds = position_feats.std(dim=[0, 1])
            velocity_means = velocity_feats.mean(dim=[0, 1])
            velocity_stds = velocity_feats.std(dim=[0, 1])
            
            stats.update({
                'position_feature_means': position_means,
                'position_feature_stds': position_stds,
                'velocity_feature_means': velocity_means,
                'velocity_feature_stds': velocity_stds
            })
        
        print(f"\nFeature Statistics (from {len(serves_to_analyze)} serves):")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Position features: {self.num_static_feats}, Velocity features: {self.num_dynamic_feats}")
        
        if analyze_position_velocity and 'position_feature_means' in stats:
            print(f"\nPosition Feature Statistics (X, Y, Z):")
            for i in range(self.num_static_feats):
                coord = ['X', 'Y', 'Z'][i] if i < 3 else f'Pos_{i}'
                print(f"  {coord}: mean={stats['position_feature_means'][i]:.6f}, std={stats['position_feature_stds'][i]:.6f}")
            
            print(f"\nVelocity Feature Statistics (dX, dY, dZ):")
            for i in range(self.num_dynamic_feats):
                coord = ['dX', 'dY', 'dZ'][i] if i < 3 else f'Vel_{i}'
                print(f"  {coord}: mean={stats['velocity_feature_means'][i]:.6f}, std={stats['velocity_feature_stds'][i]:.6f}")
        
        # Print global attribute statistics
        if analyze_global_attrs:
            print(f"\nGlobal Attribute Statistics:")
            for attr_name in ['server_id', 'serve_number', 'set_number', 'game_number', 'point_number']:
                if f'global_{attr_name}_count' in stats:
                    count = stats[f'global_{attr_name}_count']
                    if attr_name == 'server_id':
                        unique_count = stats[f'global_{attr_name}_unique_count']
                        min_val = stats[f'global_{attr_name}_min']
                        max_val = stats[f'global_{attr_name}_max']
                        print(f"  {attr_name}: count={count}, unique={unique_count}, range=[{min_val}, {max_val}]")
                    else:
                        mean_val = stats[f'global_{attr_name}_mean']
                        std_val = stats[f'global_{attr_name}_std']
                        min_val = stats[f'global_{attr_name}_min']
                        max_val = stats[f'global_{attr_name}_max']
                        print(f"  {attr_name}: count={count}, mean={mean_val:.2f}, std={std_val:.2f}, range=[{min_val}, {max_val}]")
        
        return stats
    
    def __len__(self):
        """Estimate total number of sequences across all serves."""
        if not hasattr(self, '_estimated_length'):
            self._estimated_length = 0
            for simulation_id in self.simulation_ids[:5]:  # Sample a few to estimate
                try:
                    dataset_file = self.directory / f"{simulation_id}.pt"
                    serve_data = torch.load(dataset_file, weights_only=False)
                    if isinstance(serve_data, list):
                        T = len(serve_data)
                        max_start = T - self.seq_len
                        if max_start >= 0:
                            sequences_per_serve = (max_start // self.stride) + 1
                            self._estimated_length += sequences_per_serve
                except:
                    continue
            
            # Extrapolate to all serves
            if len(self.simulation_ids) > 5:
                avg_per_serve = self._estimated_length / min(5, len(self.simulation_ids))
                self._estimated_length = int(avg_per_serve * len(self.simulation_ids))
        
        return self._estimated_length


def get_simulation_ids(directory: Path, pattern: str = "*.pt") -> List[str]:
    """
    Get list of serve IDs from a directory.
    
    Args:
        directory (Path): Directory containing serve files.
        pattern (str): File pattern to match.
    
    Returns:
        List[str]: List of serve file stems.
    """
    files = list(directory.glob(pattern))
    return [file.stem for file in files]


def create_datasets_from_folders(base_dir: Union[str, Path], seq_len: int = 10, stride: int = 1, 
                                num_static_feats: int = 3, num_dynamic_feats: int = 3,
                                load_normalization_stats: bool = True):
    """
    Create train/val/test datasets from organized folder structure.
    
    Args:
        base_dir: Base directory containing train/val/test folders
        seq_len: Sequence length for rollout windows  
        stride: Stride between sequence starts
        num_static_feats: Number of position features per joint (x,y,z)
        num_dynamic_feats: Number of velocity features per joint (dx,dy,dz)
        load_normalization_stats: Whether to load normalization statistics
    
    Expected structure:
    base_dir/
    ├── train/
    │   ├── serve_001.pt
    │   ├── serve_002.pt
    │   └── ...
    ├── val/
    │   ├── serve_101.pt
    │   └── ...
    ├── test/
    │   ├── serve_201.pt
    │   └── ...
    ├── normalization_stats.pkl
    ├── player_mapping.json
    └── dataset_info.json
    """
    base_dir = Path(base_dir)
    
    # Load normalization stats if available
    normalization_stats = None
    if load_normalization_stats:
        norm_stats_path = base_dir / 'normalization_stats.pkl'
        if norm_stats_path.exists():
            with open(norm_stats_path, 'rb') as f:
                normalization_stats = pickle.load(f)
            print(f"✅ Loaded normalization statistics")
        else:
            print(f"⚠️  Normalization statistics not found at {norm_stats_path}")
    
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = base_dir / split
        if split_dir.exists():
            datasets[split] = TennisServeRolloutDataset(
                directory=split_dir,
                seq_len=seq_len,
                stride=stride,
                num_static_feats=num_static_feats,
                num_dynamic_feats=num_dynamic_feats,
                normalization_stats=normalization_stats
            )
            print(f"✅ Created {split} dataset with {len(datasets[split].simulation_ids)} serves")
        else:
            print(f"⚠️  {split_dir} not found, skipping {split} dataset")
    
    return datasets


def test_tennis_dataset():
    """Test function to verify the dataset works correctly with global attributes."""
    # Test with split data structure
    processed_dir = Path("/project/vil_baek/psaap/tennis/seq_tennis_data_normalized")
    
    if not processed_dir.exists():
        print("❌ Processed data directory not found")
        return
    
    # Test with train split
    train_dir = processed_dir / "train"
    if not train_dir.exists():
        print("❌ Train directory not found")
        return
    
    dataset = TennisServeRolloutDataset(
        directory=train_dir,
        seq_len=5,
        stride=1,
        num_static_feats=3,  # x, y, z coordinates
        num_dynamic_feats=3   # dx, dy, dz velocities
    )
    
    if len(dataset.simulation_ids) == 0:
        print("❌ No serve files found for testing")
        return
    
    # Get serve info
    print("\n📋 Serve Information:")
    info = dataset.get_serve_info()
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
            
            # Check position/velocity feature separation
            if data.x.shape[1] >= dataset.num_static_feats + dataset.num_dynamic_feats:
                position_feats = data.x[:, :dataset.num_static_feats]
                velocity_feats = data.x[:, dataset.num_static_feats:dataset.num_static_feats + dataset.num_dynamic_feats]
                print(f"    Position feats: {position_feats.shape}, Velocity feats: {velocity_feats.shape}")
            
            # Check global attributes
            global_attrs = []
            if hasattr(data, 'global_server_id'):
                global_attrs.append(f"server_id={data.global_server_id}")
            if hasattr(data, 'global_serve_number'):
                global_attrs.append(f"serve_number={data.global_serve_number}")
            if hasattr(data, 'global_set_number'):
                global_attrs.append(f"set_number={data.global_set_number}")
            if hasattr(data, 'global_game_number'):
                global_attrs.append(f"game_number={data.global_game_number}")
            if hasattr(data, 'global_point_number'):
                global_attrs.append(f"point_number={data.global_point_number}")
            if hasattr(data, 'global_simulation_id'):
                global_attrs.append(f"simulation_id={data.global_simulation_id}")
            
            if global_attrs:
                print(f"    Global attrs: {', '.join(global_attrs)}")
            else:
                print(f"    Global attrs: None found")
    
    # Get feature statistics including global attributes
    print("\n📊 Feature statistics:")
    stats = dataset.get_feature_stats(max_serves=5, analyze_position_velocity=True, analyze_global_attrs=True)
    
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
            if hasattr(first_data, 'global_server_id'):
                global_preserved.append("server_id")
            if hasattr(first_data, 'global_serve_number'):
                global_preserved.append("serve_number")
            if hasattr(first_data, 'global_set_number'):
                global_preserved.append("set_number")
            if hasattr(first_data, 'global_game_number'):
                global_preserved.append("game_number")
            if hasattr(first_data, 'global_point_number'):
                global_preserved.append("point_number")
            print(f"    Global attrs preserved: {global_preserved}")


def inspect_serve_features(serve_path: Union[str, Path], num_static_feats: int = 3, num_dynamic_feats: int = 3):
    """
    Utility function to inspect the feature structure of a serve file.
    
    Args:
        serve_path: Path to a single .pt serve file
        num_static_feats: Expected number of position features per joint
        num_dynamic_feats: Expected number of velocity features per joint
    """
    try:
        data = torch.load(serve_path, weights_only=False)
        
        if not isinstance(data, list) or len(data) == 0:
            print("❌ Invalid serve format")
            return
        
        sample = data[0]
        print(f"🔍 Serve Feature Inspection: {serve_path}")
        print(f"  Total timesteps: {len(data)}")
        print(f"  Sample data structure:")
        print(f"    x.shape: {sample.x.shape}")
        print(f"    y.shape: {sample.y.shape}")
        print(f"    num_nodes (joints): {sample.num_nodes}")
        print(f"    num_edges: {sample.num_edges}")
        
        if hasattr(sample, 'pos'):
            print(f"    pos.shape: {sample.pos.shape}")
        
        # Check for global attributes
        print(f"\n🌍 Global Attributes:")
        global_attrs_found = []
        if hasattr(sample, 'server_id'):
            global_attrs_found.append(f"server_id: {sample.server_id}")
        if hasattr(sample, 'serve_number'):
            global_attrs_found.append(f"serve_number: {sample.serve_number}")
        if hasattr(sample, 'set_number'):
            global_attrs_found.append(f"set_number: {sample.set_number}")
        if hasattr(sample, 'game_number'):
            global_attrs_found.append(f"game_number: {sample.game_number}")
        if hasattr(sample, 'point_number'):
            global_attrs_found.append(f"point_number: {sample.point_number}")
        
        if global_attrs_found:
            for attr in global_attrs_found:
                print(f"    {attr}")
        else:
            print(f"    No global attributes found")
        
        # Feature analysis
        total_feats = sample.x.shape[1]
        print(f"\n📊 Feature Analysis:")
        print(f"  Total features per joint: {total_feats}")
        print(f"  Expected position feats: {num_static_feats}")
        print(f"  Expected velocity feats: {num_dynamic_feats}")
        
        if total_feats >= num_static_feats + num_dynamic_feats:
            # Sample from first few joints
            position_sample = sample.x[:5, :num_static_feats]  # First 5 joints
            velocity_sample = sample.x[:5, num_static_feats:num_static_feats + num_dynamic_feats]
            
            print(f"\n  Position features (first 5 joints):")
            print(f"    {position_sample}")
            print(f"\n  Velocity features (first 5 joints):")
            print(f"    {velocity_sample}")
        else:
            print(f"  ⚠️ Warning: Total features ({total_feats}) < expected ({num_static_feats + num_dynamic_feats})")
        
        # Check targets
        if hasattr(sample, 'y'):
            print(f"\n🎯 Target Analysis:")
            print(f"  Target shape: {sample.y.shape}")
            print(f"  Target sample (first 3 joints):")
            print(f"    {sample.y[:3]}")
            
    except Exception as e:
        print(f"❌ Error inspecting serve: {e}")


# Example usage
if __name__ == "__main__":
    # Test the tennis dataset
    print("🎾 Testing Tennis Serve Rollout Dataset")
    test_tennis_dataset()
    
    # Example of creating datasets from folders
    print("\n📁 Creating datasets from organized folders:")
    try:
        base_dir = Path("/project/vil_baek/psaap/tennis/seq_tennis_data_normalized")
        datasets = create_datasets_from_folders(
            base_dir=base_dir,
            seq_len=8,
            stride=1,
            num_static_feats=3,
            num_dynamic_feats=3
        )
        
        print(f"\n✅ Successfully created datasets:")
        for split_name, dataset in datasets.items():
            print(f"  {split_name}: {len(dataset.simulation_ids)} serves, estimated {len(dataset)} sequences")
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
    
    print("\n🎉 Tennis dataset setup complete!")