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


class BaseballPitchRolloutDataset(IterableDataset):
    """
    Dataset for baseball pitching motion with multiple PyG dataset files.
    Yields consecutive sequences of timesteps from multiple pitches:
      [ (Data[t].x, Data[t+1].y), (Data[t+1].x, Data[t+2].y), ..., (Data[t+seq_len-1].x, Data[t+seq_len].y) ]
    for each pitch file.
    
    Extracts global attributes (pitch_speed) and attaches them to each Data object.
    """
    def __init__(self,
                 directory: Union[str, Path],
                 simulation_ids: List[str] = None,
                 seq_len: int = 10,
                 stride: int = 1,
                 num_static_feats: int = 0,
                 num_dynamic_feats: int = 9,
                 file_pattern: str = "*.pt",
                 normalization_stats: Dict = None):
        """
        Args:
            directory (Union[str, Path]): Directory containing PyG dataset files (.pt files).
            simulation_ids (List[str]): List of pitch IDs (file stems). If None, auto-discover.
            seq_len (int): Number of consecutive steps in each returned window.
            stride (int): Step size between sequence starts (default: 1 for overlapping sequences).
            num_static_feats (int): Number of static features per joint (default: 0 for baseball).
            num_dynamic_feats (int): Number of dynamic features per joint (default: 9 for pos+vel+angles).
            file_pattern (str): Pattern to match pitch files (default: "*.pt").
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
        
        # Auto-discover pitch IDs if not provided
        if simulation_ids is None:
            self.simulation_ids = self._discover_simulation_ids()
        else:
            self.simulation_ids = simulation_ids
        
        # Baseball joint names (18 joints)
        self.joint_names = [
            "centerofmass", "elbow_jc", "glove_elbow_jc", "glove_hand_jc",
            "glove_shoulder_jc", "glove_wrist_jc", "hand_jc", "lead_ankle_jc",
            "lead_hip", "lead_knee_jc", "rear_ankle_jc", "rear_hip",
            "rear_knee_jc", "shoulder_jc", "thorax_ap", "thorax_dist",
            "thorax_prox", "wrist_jc"
        ]
        
        # Load player mapping if available
        self.player_mapping = self._load_player_mapping()
        
        print(f"BaseballPitchRolloutDataset initialized:")
        print(f"  Directory: {self.directory}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Stride: {stride}")
        print(f"  Static features: {num_static_feats}")
        print(f"  Dynamic features: {num_dynamic_feats} (3 pos + 3 vel + 3 angles)")
        print(f"  Found {len(self.simulation_ids)} pitch files")
        print(f"  Joints: {len(self.joint_names)}")
        if len(self.simulation_ids) > 0:
            print(f"  Example files: {self.simulation_ids[:3]}")
    
    def _discover_simulation_ids(self) -> List[str]:
        """Auto-discover pitch files in the directory."""
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
        Iterate through all pitch files and yield sequences.
        Extracts global attributes (pitch_speed) from each timestep.
        """
        for simulation_id in self.simulation_ids:
            dataset_file = self.directory / f"{simulation_id}.pt"
            
            try:
                # Load the pitch data
                pitch_data = torch.load(dataset_file, weights_only=False)
                
                if not isinstance(pitch_data, list):
                    print(f"Unexpected data format in {dataset_file}. Expected list, got {type(pitch_data)}. Skipping.")
                    continue
                
                T = len(pitch_data)
                max_start = T - self.seq_len
                
                if max_start < 0:
                    print(f"Pitch {simulation_id} has only {T} timesteps, need at least {self.seq_len}. Skipping.")
                    continue
                
                # Generate sequences with specified stride
                for start_idx in range(0, max_start + 1, self.stride):
                    window = []
                    for offset in range(self.seq_len):
                        t = start_idx + offset
                        data_t = pitch_data[t].clone()
                        
                        # Ensure the data has the expected structure
                        if not hasattr(data_t, 'x') or not hasattr(data_t, 'y'):
                            print(f"Warning: Data at timestep {t} in {simulation_id} missing x or y. Skipping sequence.")
                            break
                        
                        # Extract and properly name the global attributes
                        # Baseball only has pitch_speed as global feature
                        if hasattr(data_t, 'pitch_speed'):
                            data_t.global_pitch_speed = data_t.pitch_speed.clone()
                        else:
                            data_t.global_pitch_speed = torch.tensor([85.0], dtype=torch.float32)  # Default mph
                        
                        # Create global_feats tensor for model compatibility
                        # Model expects data.global_feats with shape [num_global_feats]
                        data_t.global_feats = data_t.global_pitch_speed.clone()
                        
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
    
    def get_pitch_info(self, simulation_id: str = None) -> dict:
        """Get information about a specific pitch or the first available one."""
        if simulation_id is None and self.simulation_ids:
            simulation_id = self.simulation_ids[0]
        elif simulation_id not in self.simulation_ids:
            print(f"Pitch {simulation_id} not found in available pitches.")
            return {}
        
        dataset_file = self.directory / f"{simulation_id}.pt"
        
        try:
            pitch_data = torch.load(dataset_file, weights_only=False)
            
            if not isinstance(pitch_data, list) or len(pitch_data) == 0:
                return {"error": "Invalid or empty pitch data"}
            
            sample_data = pitch_data[0]
            
            info = {
                "simulation_id": simulation_id,
                "total_timesteps": len(pitch_data),
                "num_joints": sample_data.num_nodes,
                "num_features": sample_data.num_features,
                "num_edges": sample_data.num_edges,
                "feature_shape": sample_data.x.shape,
                "target_shape": sample_data.y.shape,
                "has_positions": hasattr(sample_data, 'pos'),
                "is_directed": sample_data.is_directed() if hasattr(sample_data, 'is_directed') else "Unknown",
                # Check for global attributes
                "has_pitch_speed": hasattr(sample_data, 'pitch_speed'),
                "has_pitcher_id": hasattr(sample_data, 'pitcher_id'),
                "has_playing_level": hasattr(sample_data, 'playing_level')
            }
            
            # Add sample global attribute values if they exist
            if hasattr(sample_data, 'pitch_speed'):
                pitch_speed = sample_data.pitch_speed.item() if hasattr(sample_data.pitch_speed, 'item') else sample_data.pitch_speed
                info["sample_pitch_speed"] = pitch_speed
            
            if hasattr(sample_data, 'pitcher_id'):
                pitcher_id = sample_data.pitcher_id.item() if hasattr(sample_data.pitcher_id, 'item') else sample_data.pitcher_id
                info["sample_pitcher_id"] = pitcher_id
                # Convert to player name if mapping available
                if self.player_mapping:
                    pitcher_name = None
                    for name, pid in self.player_mapping.items():
                        if pid == pitcher_id:
                            pitcher_name = name
                            break
                    info["sample_pitcher_name"] = pitcher_name if pitcher_name else f"Unknown_{pitcher_id}"
            
            if hasattr(sample_data, 'playing_level'):
                info["sample_playing_level"] = sample_data.playing_level.item() if hasattr(sample_data.playing_level, 'item') else sample_data.playing_level
            
            return info
            
        except Exception as e:
            return {"error": f"Could not load pitch info: {e}"}
    
    def get_feature_stats(self, max_pitches: int = None, analyze_feature_types: bool = True, 
                         analyze_global_attrs: bool = True) -> dict:
        """
        Calculate feature statistics across pitches for normalization.
        
        Args:
            max_pitches (int): Limit number of pitches to analyze (for speed).
            analyze_feature_types (bool): Whether to separate position, velocity, and angle statistics.
            analyze_global_attrs (bool): Whether to analyze global attribute statistics.
        """
        if not self.simulation_ids:
            return {}
        
        pitches_to_analyze = self.simulation_ids[:max_pitches] if max_pitches else self.simulation_ids
        
        all_features = []
        all_targets = []
        global_attrs = {
            'pitch_speed': [],
            'pitcher_id': [],
            'playing_level': []
        }
        
        print(f"Calculating feature statistics from {len(pitches_to_analyze)} pitches...")
        
        for simulation_id in tqdm(pitches_to_analyze, desc="Loading pitches"):
            dataset_file = self.directory / f"{simulation_id}.pt"
            
            try:
                pitch_data = torch.load(dataset_file, weights_only=False)
                
                if isinstance(pitch_data, list) and len(pitch_data) > 0:
                    # Stack features and targets from this pitch
                    pitch_features = torch.stack([data.x for data in pitch_data])  # [T, N, F]
                    pitch_targets = torch.stack([data.y for data in pitch_data])   # [T, N, F]
                    
                    all_features.append(pitch_features)
                    all_targets.append(pitch_targets)
                    
                    # Collect global attributes if analyzing them
                    if analyze_global_attrs:
                        for data in pitch_data:
                            if hasattr(data, 'pitch_speed'):
                                global_attrs['pitch_speed'].append(data.pitch_speed)
                            if hasattr(data, 'pitcher_id'):
                                global_attrs['pitcher_id'].append(data.pitcher_id)
                            if hasattr(data, 'playing_level'):
                                global_attrs['playing_level'].append(data.playing_level)
                    
            except Exception as e:
                print(f"Error processing {simulation_id}: {e}")
                continue
        
        if not all_features:
            return {"error": "No valid pitch data found"}
        
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
                    if attr_name in ['pitcher_id', 'playing_level']:
                        # Handle integer IDs differently
                        attr_tensor = torch.stack(attr_values).flatten()
                        unique_vals = torch.unique(attr_tensor)
                        global_stats[f'global_{attr_name}_unique_count'] = len(unique_vals)
                        global_stats[f'global_{attr_name}_min'] = attr_tensor.min()
                        global_stats[f'global_{attr_name}_max'] = attr_tensor.max()
                    else:
                        # Handle float attributes (pitch_speed)
                        attr_tensor = torch.stack(attr_values).flatten()
                        global_stats[f'global_{attr_name}_mean'] = attr_tensor.mean()
                        global_stats[f'global_{attr_name}_std'] = attr_tensor.std()
                        global_stats[f'global_{attr_name}_min'] = attr_tensor.min()
                        global_stats[f'global_{attr_name}_max'] = attr_tensor.max()
                    
                    global_stats[f'global_{attr_name}_count'] = len(attr_values)
            
            stats.update(global_stats)
        
        # Separate position, velocity, and angle feature statistics if requested
        # Baseball: 9 features = 3 pos + 3 vel + 3 angles
        if analyze_feature_types and all_features.shape[2] >= 9:
            position_feats = all_features[:, :, :3]  # First 3: positions
            velocity_feats = all_features[:, :, 3:6]  # Next 3: velocities
            angle_feats = all_features[:, :, 6:9]  # Last 3: angles
            
            position_means = position_feats.mean(dim=[0, 1])
            position_stds = position_feats.std(dim=[0, 1])
            velocity_means = velocity_feats.mean(dim=[0, 1])
            velocity_stds = velocity_feats.std(dim=[0, 1])
            angle_means = angle_feats.mean(dim=[0, 1])
            angle_stds = angle_feats.std(dim=[0, 1])
            
            stats.update({
                'position_feature_means': position_means,
                'position_feature_stds': position_stds,
                'velocity_feature_means': velocity_means,
                'velocity_feature_stds': velocity_stds,
                'angle_feature_means': angle_means,
                'angle_feature_stds': angle_stds
            })
        
        print(f"\nFeature Statistics (from {len(pitches_to_analyze)} pitches):")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Dynamic features: {self.num_dynamic_feats} (3 pos + 3 vel + 3 angles)")
        
        if analyze_feature_types and 'position_feature_means' in stats:
            print(f"\nPosition Feature Statistics (X, Y, Z):")
            for i in range(3):
                coord = ['X', 'Y', 'Z'][i]
                print(f"  {coord}: mean={stats['position_feature_means'][i]:.6f}, std={stats['position_feature_stds'][i]:.6f}")
            
            print(f"\nVelocity Feature Statistics (dX, dY, dZ):")
            for i in range(3):
                coord = ['dX', 'dY', 'dZ'][i]
                print(f"  {coord}: mean={stats['velocity_feature_means'][i]:.6f}, std={stats['velocity_feature_stds'][i]:.6f}")
            
            print(f"\nAngle Feature Statistics (angleX, angleY, angleZ):")
            for i in range(3):
                coord = ['angleX', 'angleY', 'angleZ'][i]
                print(f"  {coord}: mean={stats['angle_feature_means'][i]:.6f}, std={stats['angle_feature_stds'][i]:.6f}")
        
        # Print global attribute statistics
        if analyze_global_attrs:
            print(f"\nGlobal Attribute Statistics:")
            if 'global_pitch_speed_count' in stats:
                count = stats['global_pitch_speed_count']
                mean_val = stats['global_pitch_speed_mean']
                std_val = stats['global_pitch_speed_std']
                min_val = stats['global_pitch_speed_min']
                max_val = stats['global_pitch_speed_max']
                print(f"  pitch_speed: count={count}, mean={mean_val:.2f} mph, std={std_val:.2f}, range=[{min_val:.1f}, {max_val:.1f}]")
            
            for attr_name in ['pitcher_id', 'playing_level']:
                if f'global_{attr_name}_count' in stats:
                    count = stats[f'global_{attr_name}_count']
                    unique_count = stats[f'global_{attr_name}_unique_count']
                    min_val = stats[f'global_{attr_name}_min']
                    max_val = stats[f'global_{attr_name}_max']
                    print(f"  {attr_name}: count={count}, unique={unique_count}, range=[{min_val}, {max_val}]")
        
        return stats
    
    def __len__(self):
        """Estimate total number of sequences across all pitches."""
        if not hasattr(self, '_estimated_length'):
            self._estimated_length = 0
            for simulation_id in self.simulation_ids[:5]:  # Sample a few to estimate
                try:
                    dataset_file = self.directory / f"{simulation_id}.pt"
                    pitch_data = torch.load(dataset_file, weights_only=False)
                    if isinstance(pitch_data, list):
                        T = len(pitch_data)
                        max_start = T - self.seq_len
                        if max_start >= 0:
                            sequences_per_pitch = (max_start // self.stride) + 1
                            self._estimated_length += sequences_per_pitch
                except:
                    continue
            
            # Extrapolate to all pitches
            if len(self.simulation_ids) > 5:
                avg_per_pitch = self._estimated_length / min(5, len(self.simulation_ids))
                self._estimated_length = int(avg_per_pitch * len(self.simulation_ids))
        
        return self._estimated_length


def get_simulation_ids(directory: Path, pattern: str = "*.pt") -> List[str]:
    """
    Get list of pitch IDs from a directory.
    
    Args:
        directory (Path): Directory containing pitch files.
        pattern (str): File pattern to match.
    
    Returns:
        List[str]: List of pitch file stems.
    """
    files = list(directory.glob(pattern))
    return [file.stem for file in files]


def create_datasets_from_folders(base_dir: Union[str, Path], seq_len: int = 10, stride: int = 1, 
                                num_static_feats: int = 0, num_dynamic_feats: int = 9,
                                load_normalization_stats: bool = True):
    """
    Create train/val/test datasets from organized folder structure.
    
    Args:
        base_dir: Base directory containing train/val/test folders
        seq_len: Sequence length for rollout windows  
        stride: Stride between sequence starts
        num_static_feats: Number of static features per joint (0 for baseball)
        num_dynamic_feats: Number of dynamic features per joint (9 for baseball)
        load_normalization_stats: Whether to load normalization statistics
    
    Expected structure:
    base_dir/
    ├── train/
    │   ├── pitch_001.pt
    │   ├── pitch_002.pt
    │   └── ...
    ├── val/
    │   ├── pitch_101.pt
    │   └── ...
    ├── test/
    │   ├── pitch_201.pt
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
            datasets[split] = BaseballPitchRolloutDataset(
                directory=split_dir,
                seq_len=seq_len,
                stride=stride,
                num_static_feats=num_static_feats,
                num_dynamic_feats=num_dynamic_feats,
                normalization_stats=normalization_stats
            )
            print(f"✅ Created {split} dataset with {len(datasets[split].simulation_ids)} pitches")
        else:
            print(f"⚠️  {split_dir} not found, skipping {split} dataset")
    
    return datasets


def test_baseball_dataset():
    """Test function to verify the dataset works correctly with global attributes."""
    # Test with split data structure
    processed_dir = Path("/project/vil_baek/psaap/baseball/seq_baseball_data_normalized")
    
    if not processed_dir.exists():
        print("❌ Processed data directory not found")
        return
    
    # Test with train split
    train_dir = processed_dir / "train"
    if not train_dir.exists():
        print("❌ Train directory not found")
        return
    
    dataset = BaseballPitchRolloutDataset(
        directory=train_dir,
        seq_len=5,
        stride=1,
        num_static_feats=0,  # All features are dynamic
        num_dynamic_feats=9   # 3 pos + 3 vel + 3 angles
    )
    
    if len(dataset.simulation_ids) == 0:
        print("❌ No pitch files found for testing")
        return
    
    # Get pitch info
    print("\n📋 Pitch Information:")
    info = dataset.get_pitch_info()
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
            
            # Check feature separation (pos, vel, angles)
            if data.x.shape[1] >= 9:
                position_feats = data.x[:, :3]
                velocity_feats = data.x[:, 3:6]
                angle_feats = data.x[:, 6:9]
                print(f"    Position feats: {position_feats.shape}, Velocity feats: {velocity_feats.shape}, Angle feats: {angle_feats.shape}")
            
            # Check global attributes
            global_attrs = []
            if hasattr(data, 'global_pitch_speed'):
                global_attrs.append(f"pitch_speed={data.global_pitch_speed}")
            if hasattr(data, 'global_simulation_id'):
                global_attrs.append(f"simulation_id={data.global_simulation_id}")
            
            if global_attrs:
                print(f"    Global attrs: {', '.join(global_attrs)}")
            else:
                print(f"    Global attrs: None found")
    
    # Get feature statistics including global attributes
    print("\n📊 Feature statistics:")
    stats = dataset.get_feature_stats(max_pitches=5, analyze_feature_types=True, analyze_global_attrs=True)
    
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
            if hasattr(first_data, 'global_pitch_speed'):
                global_preserved.append("pitch_speed")
            print(f"    Global attrs preserved: {global_preserved}")


def inspect_pitch_features(pitch_path: Union[str, Path], num_static_feats: int = 0, num_dynamic_feats: int = 9):
    """
    Utility function to inspect the feature structure of a pitch file.
    
    Args:
        pitch_path: Path to a single .pt pitch file
        num_static_feats: Expected number of static features per joint (0 for baseball)
        num_dynamic_feats: Expected number of dynamic features per joint (9 for baseball)
    """
    try:
        data = torch.load(pitch_path, weights_only=False)
        
        if not isinstance(data, list) or len(data) == 0:
            print("❌ Invalid pitch format")
            return
        
        sample = data[0]
        print(f"🔍 Pitch Feature Inspection: {pitch_path}")
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
        if hasattr(sample, 'pitch_speed'):
            global_attrs_found.append(f"pitch_speed: {sample.pitch_speed}")
        if hasattr(sample, 'pitcher_id'):
            global_attrs_found.append(f"pitcher_id: {sample.pitcher_id}")
        if hasattr(sample, 'playing_level'):
            global_attrs_found.append(f"playing_level: {sample.playing_level}")
        
        if global_attrs_found:
            for attr in global_attrs_found:
                print(f"    {attr}")
        else:
            print(f"    No global attributes found")
        
        # Feature analysis
        total_feats = sample.x.shape[1]
        print(f"\n📊 Feature Analysis:")
        print(f"  Total features per joint: {total_feats}")
        print(f"  Expected dynamic feats: {num_dynamic_feats} (3 pos + 3 vel + 3 angles)")
        
        if total_feats >= 9:
            # Sample from first few joints
            position_sample = sample.x[:5, :3]  # First 5 joints
            velocity_sample = sample.x[:5, 3:6]
            angle_sample = sample.x[:5, 6:9]
            
            print(f"\n  Position features (first 5 joints):")
            print(f"    {position_sample}")
            print(f"\n  Velocity features (first 5 joints):")
            print(f"    {velocity_sample}")
            print(f"\n  Angle features (first 5 joints):")
            print(f"    {angle_sample}")
        else:
            print(f"  ⚠️ Warning: Total features ({total_feats}) < expected ({num_dynamic_feats})")
        
        # Check targets
        if hasattr(sample, 'y'):
            print(f"\n🎯 Target Analysis:")
            print(f"  Target shape: {sample.y.shape}")
            print(f"  Target sample (first 3 joints):")
            print(f"    {sample.y[:3]}")
        
        # Edge information
        if hasattr(sample, 'edge_index'):
            print(f"\n🔗 Edge Information:")
            print(f"  Edge index shape: {sample.edge_index.shape}")
            print(f"  Number of edges: {sample.num_edges}")
            if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
                print(f"  Edge attributes shape: {sample.edge_attr.shape}")
        
        return sample
        
    except Exception as e:
        print(f"❌ Error inspecting pitch: {e}")
        import traceback
        traceback.print_exc()
        return None


def collate_sequences(batch):
    """
    Custom collate function for batching sequences of PyG Data objects.
    
    Args:
        batch: List of sequences, where each sequence is a list of Data objects
    
    Returns:
        List of sequences (preserves structure for easy iteration)
    """
    return batch


def create_dataloaders(datasets: dict, batch_size: int = 32, num_workers: int = 4,
                       prefetch_factor: int = 2):
    """
    Create DataLoaders from train/val/test datasets.
    
    Args:
        datasets: Dictionary with 'train', 'val', 'test' datasets
        batch_size: Batch size for training
        num_workers: Number of worker processes
        prefetch_factor: Number of batches to prefetch per worker
    
    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {}
    
    for split, dataset in datasets.items():
        shuffle = (split == 'train')  # Only shuffle training data
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # IterableDataset doesn't support shuffle in DataLoader
            num_workers=num_workers,
            collate_fn=collate_sequences,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        print(f"✅ Created {split} DataLoader (batch_size={batch_size}, num_workers={num_workers})")
    
    return dataloaders


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Baseball Pitch Dataset')
    parser.add_argument('--data_dir', type=str, 
                       default='/project/vil_baek/psaap/baseball/seq_baseball_data_normalized',
                       help='Base directory containing train/val/test splits')
    parser.add_argument('--inspect_file', type=str, default=None,
                       help='Path to a specific pitch file to inspect')
    parser.add_argument('--seq_len', type=int, default=5,
                       help='Sequence length for rollout windows')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride between sequence starts')
    parser.add_argument('--create_datasets', action='store_true',
                       help='Create train/val/test datasets')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for DataLoader')
    
    args = parser.parse_args()
    
    if args.inspect_file:
        # Inspect a specific pitch file
        print("=" * 80)
        print("INSPECTING PITCH FILE")
        print("=" * 80)
        inspect_pitch_features(args.inspect_file, num_static_feats=0, num_dynamic_feats=9)
    
    elif args.create_datasets:
        # Create datasets and dataloaders
        print("=" * 80)
        print("CREATING DATASETS FROM FOLDERS")
        print("=" * 80)
        try:
            base_dir = Path(args.data_dir)
            datasets = create_datasets_from_folders(
                base_dir=base_dir,
                seq_len=args.seq_len,
                stride=args.stride,
                num_static_feats=0,
                num_dynamic_feats=9
            )
            
            print(f"\n✅ Successfully created datasets:")
            for split_name, dataset in datasets.items():
                print(f"  {split_name}: {len(dataset.simulation_ids)} pitches, estimated {len(dataset)} sequences")
            
            # Create dataloaders
            print(f"\n📦 Creating DataLoaders...")
            dataloaders = create_dataloaders(
                datasets,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            # Test iteration
            print(f"\n🔄 Testing DataLoader iteration...")
            for split_name, dataloader in dataloaders.items():
                print(f"\n{split_name.upper()} DataLoader:")
                for i, batch in enumerate(dataloader):
                    if i >= 1:  # Just test first batch
                        break
                    print(f"  Batch {i}: {len(batch)} sequences")
                    for seq_idx, sequence in enumerate(batch[:2]):  # Show first 2 sequences
                        print(f"    Sequence {seq_idx}: {len(sequence)} timesteps")
                        first_step = sequence[0]
                        print(f"      x.shape: {first_step.x.shape}, y.shape: {first_step.y.shape}")
                        if hasattr(first_step, 'global_pitch_speed'):
                            print(f"      pitch_speed: {first_step.global_pitch_speed.item():.1f} mph")
            
            print(f"\n🎉 Baseball dataset setup complete!")
            
        except Exception as e:
            print(f"❌ Error creating datasets: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # Run full dataset tests
        print("=" * 80)
        print("⚾ TESTING BASEBALL PITCH ROLLOUT DATASET")
        print("=" * 80)
        test_baseball_dataset()
        
        print("\n" + "=" * 80)
        print("CREATING DATASETS FROM FOLDERS")
        print("=" * 80)
        try:
            base_dir = Path(args.data_dir)
            datasets = create_datasets_from_folders(
                base_dir=base_dir,
                seq_len=args.seq_len,
                stride=args.stride,
                num_static_feats=0,
                num_dynamic_feats=9
            )
            
            print(f"\n✅ Successfully created datasets:")
            for split_name, dataset in datasets.items():
                print(f"  {split_name}: {len(dataset.simulation_ids)} pitches, estimated {len(dataset)} sequences")
            
        except Exception as e:
            print(f"❌ Error creating datasets: {e}")
        
        print("\n🎉 Baseball dataset testing complete!")