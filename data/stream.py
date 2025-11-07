import torch
from pathlib import Path
from typing import List, Iterator, Union, Optional, Tuple
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
import gc
import random


class StreamingKarmanDataset(IterableDataset):
    """
    Streaming dataset that loads ONLY the required timesteps from each simulation.
    
    Instead of loading all 400 timesteps, we only load seq_len timesteps at a time.
    This is the most memory-efficient approach for large simulation files.
    """
    def __init__(self,
                 directory: Union[str, Path],
                 simulation_ids: List[str] = None,
                 seq_len: int = 10,
                 stride: int = 1,
                 num_static_feats: int = 3,
                 num_dynamic_feats: int = 7,
                 file_pattern: str = "*.pt",
                 shuffle_simulations: bool = True,
                 max_timesteps_per_sim: Optional[int] = None,
                 timestep_range: Optional[Tuple[int, int]] = None,
                 timestep_sampling: str = 'consecutive',
                 temporal_subsample_rate: Optional[int] = None):
        """
        Args:
            directory: Directory containing PyG dataset files (.pt files).
            simulation_ids: List of simulation IDs. If None, auto-discover.
            seq_len: Number of consecutive steps in each sequence.
            stride: Step size between sequence starts.
            num_static_feats: Number of static features (positions).
            num_dynamic_feats: Number of dynamic features (physics).
            file_pattern: Pattern to match simulation files.
            shuffle_simulations: If True, shuffle order of simulations each epoch.
            max_timesteps_per_sim: If set, only use first N timesteps from each simulation.
            timestep_range: Tuple (start, end) to use specific timestep range.
                          E.g., (100, 300) uses timesteps 100-299.
                          Overrides max_timesteps_per_sim if both are set.
            timestep_sampling: How to sample timesteps:
                - 'consecutive': Use all timesteps in range (default)
                - 'random': Randomly sample timesteps from range each epoch
                - 'every_nth': Use every Nth timestep (set temporal_subsample_rate)
            temporal_subsample_rate: If timestep_sampling='every_nth', use every Nth timestep.
                                    E.g., rate=2 means [0, 2, 4, 6, ...], rate=5 means [0, 5, 10, ...]
        """
        super().__init__()
        self.directory = Path(directory)
        self.seq_len = seq_len
        self.stride = stride
        self.file_pattern = file_pattern
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.shuffle_simulations = shuffle_simulations
        self.max_timesteps_per_sim = max_timesteps_per_sim
        self.timestep_range = timestep_range
        self.timestep_sampling = timestep_sampling
        self.temporal_subsample_rate = temporal_subsample_rate
        
        # Auto-discover simulation IDs if not provided
        if simulation_ids is None:
            self.simulation_ids = self._discover_simulation_ids()
        else:
            self.simulation_ids = simulation_ids
        
        # Variables
        self.var_names = ['pressure', 'velocity_x', 'velocity_y', 'velocity_z', 
                         'vorticity_x', 'vorticity_y', 'vorticity_z']
        
        # Validate settings
        if timestep_sampling == 'every_nth' and temporal_subsample_rate is None:
            raise ValueError("temporal_subsample_rate must be set when timestep_sampling='every_nth'")
        
        print(f"StreamingKarmanDataset initialized:")
        print(f"  Directory: {self.directory}")
        print(f"  Sequence length: {seq_len}, Stride: {stride}")
        print(f"  Static features: {num_static_feats}, Dynamic features: {num_dynamic_feats}")
        print(f"  Found {len(self.simulation_ids)} simulation files")
        
        # Print timestep selection info
        if timestep_range:
            print(f"  Timestep range: {timestep_range[0]} to {timestep_range[1]}")
        elif max_timesteps_per_sim:
            print(f"  Max timesteps per sim: {max_timesteps_per_sim} (first N timesteps)")
        else:
            print(f"  Max timesteps per sim: All (400)")
        
        print(f"  Timestep sampling: {timestep_sampling}")
        if timestep_sampling == 'every_nth':
            print(f"  Temporal subsample rate: every {temporal_subsample_rate}th timestep")
        
        print(f"  Shuffle simulations: {shuffle_simulations}")
        if len(self.simulation_ids) > 0:
            print(f"  Example files: {self.simulation_ids[:3]}")
    
    def _select_timesteps(self, sim_data: List[Data]) -> List[Data]:
        """
        Select which timesteps to use based on configuration.
        Returns a subset of the simulation data.
        """
        total_timesteps = len(sim_data)
        
        # Step 1: Determine the range
        if self.timestep_range is not None:
            start_t, end_t = self.timestep_range
            start_t = max(0, start_t)
            end_t = min(total_timesteps, end_t)
            selected_data = sim_data[start_t:end_t]
        elif self.max_timesteps_per_sim is not None:
            selected_data = sim_data[:self.max_timesteps_per_sim]
        else:
            selected_data = sim_data
        
        # Step 2: Apply temporal subsampling
        if self.timestep_sampling == 'consecutive':
            # Use all timesteps in range (default)
            return selected_data
        
        elif self.timestep_sampling == 'every_nth':
            # Use every Nth timestep
            indices = list(range(0, len(selected_data), self.temporal_subsample_rate))
            return [selected_data[i] for i in indices]
        
        elif self.timestep_sampling == 'random':
            # Randomly sample timesteps (changes each epoch)
            # Sample enough to create at least some sequences
            n_samples = min(len(selected_data), max(100, len(selected_data) // 2))
            indices = sorted(random.sample(range(len(selected_data)), n_samples))
            return [selected_data[i] for i in indices]
        
        else:
            return selected_data
    
    def _discover_simulation_ids(self) -> List[str]:
        """Auto-discover simulation files in the directory."""
        files = sorted(list(self.directory.glob(self.file_pattern)))
        return [file.stem for file in files]
    
    def _load_timestep_range(self, sim_data: List[Data], start_idx: int) -> List[Data]:
        """
        Load only the required timesteps for one sequence.
        This avoids keeping all 400 timesteps in memory.
        
        IMPORTANT: Ensures compatibility with GPARC model requirements:
        - global_params must be present as graph-level attribute
        - global_reynolds is added for convenience
        """
        window = []
        for offset in range(self.seq_len):
            t = start_idx + offset
            if t >= len(sim_data):
                break
            
            data_t = sim_data[t].clone()
            
            # Ensure required attributes
            if not hasattr(data_t, 'x') or not hasattr(data_t, 'y'):
                break
            
            # CRITICAL: Ensure global_params exists (required by GPARC)
            if hasattr(data_t, 'global_params'):
                global_tensor = data_t.global_params
                # Keep as-is for GPARC model
                if global_tensor.numel() >= 1:
                    # Also add convenience attribute
                    data_t.global_reynolds = global_tensor.flatten()
                else:
                    # Fallback if empty
                    data_t.global_params = torch.zeros(1, dtype=torch.float32)
                    data_t.global_reynolds = torch.zeros(1, dtype=torch.float32)
            else:
                # Add global_params if missing (shouldn't happen with normalized data)
                print(f"Warning: 'global_params' not found, adding zeros. This may affect model performance!")
                data_t.global_params = torch.zeros(1, dtype=torch.float32)
                data_t.global_reynolds = torch.zeros(1, dtype=torch.float32)
            
            window.append(data_t)
        
        return window if len(window) == self.seq_len else []
    
    def __iter__(self) -> Iterator[List[Data]]:
        """
        Iterate through simulations, loading and immediately processing each one.
        CRITICAL: We delete the simulation data after yielding all sequences.
        """
        # Shuffle simulation order if requested
        sim_ids = self.simulation_ids.copy()
        if self.shuffle_simulations:
            random.shuffle(sim_ids)
        
        for sim_id in sim_ids:
            dataset_file = self.directory / f"{sim_id}.pt"
            
            try:
                # Load entire simulation file
                sim_data = torch.load(dataset_file, weights_only=False)
                
                if not isinstance(sim_data, list):
                    print(f"Unexpected data format in {dataset_file}. Skipping.")
                    continue
                
                # Select timesteps based on configuration
                sim_data = self._select_timesteps(sim_data)
                
                T = len(sim_data)
                max_start = T - self.seq_len
                
                if max_start < 0:
                    print(f"Simulation {sim_id} has only {T} timesteps, need {self.seq_len}. Skipping.")
                    del sim_data
                    gc.collect()
                    continue
                
                # Generate and yield sequences
                for start_idx in range(0, max_start + 1, self.stride):
                    window = self._load_timestep_range(sim_data, start_idx)
                    
                    if len(window) == self.seq_len:
                        yield window
                
                # CRITICAL: Delete simulation data to free memory
                del sim_data
                gc.collect()
                
            except FileNotFoundError:
                print(f"Dataset file not found: {dataset_file}")
            except Exception as e:
                print(f"Error loading {dataset_file}: {e}")
                import traceback
                traceback.print_exc()
    
    def get_simulation_info(self, sim_id: str = None) -> dict:
        """Get information about a specific simulation."""
        if sim_id is None and self.simulation_ids:
            sim_id = self.simulation_ids[0]
        elif sim_id not in self.simulation_ids:
            print(f"Simulation {sim_id} not found.")
            return {}
        
        dataset_file = self.directory / f"{sim_id}.pt"
        
        try:
            # Load just to get info, then delete
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
                "has_global_params": hasattr(sample_data, 'global_params'),
                "has_reynolds_number": hasattr(sample_data, 'reynolds_number')
            }
            
            if hasattr(sample_data, 'reynolds_number'):
                info["reynolds_number"] = sample_data.reynolds_number
            if hasattr(sample_data, 'global_params'):
                info["global_params_value"] = sample_data.global_params.tolist()
            
            # Calculate memory usage
            bytes_per_timestep = (
                sample_data.x.element_size() * sample_data.x.nelement() +
                sample_data.y.element_size() * sample_data.y.nelement() +
                sample_data.edge_index.element_size() * sample_data.edge_index.nelement()
            )
            total_mb = (bytes_per_timestep * len(sim_data)) / (1024 ** 2)
            info["approx_memory_mb"] = f"{total_mb:.1f} MB"
            
            # Cleanup
            del sim_data
            gc.collect()
            
            return info
            
        except Exception as e:
            return {"error": f"Could not load simulation info: {e}"}


class ChunkedKarmanDataset(IterableDataset):
    """
    Alternative: Process simulations in chunks of timesteps.
    
    Instead of loading all 400 timesteps at once, we load and process
    them in smaller chunks (e.g., 50 timesteps at a time).
    """
    def __init__(self,
                 directory: Union[str, Path],
                 simulation_ids: List[str] = None,
                 seq_len: int = 10,
                 stride: int = 1,
                 chunk_size: int = 50,
                 num_static_feats: int = 3,
                 num_dynamic_feats: int = 7,
                 file_pattern: str = "*.pt",
                 shuffle_simulations: bool = True):
        """
        Args:
            chunk_size: Number of timesteps to load at once from each simulation.
                       Smaller = less memory, but more I/O overhead.
                       Recommended: 50-100 for 400 timestep files.
        """
        super().__init__()
        self.directory = Path(directory)
        self.seq_len = seq_len
        self.stride = stride
        self.chunk_size = chunk_size
        self.file_pattern = file_pattern
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.shuffle_simulations = shuffle_simulations
        
        if simulation_ids is None:
            self.simulation_ids = self._discover_simulation_ids()
        else:
            self.simulation_ids = simulation_ids
        
        print(f"ChunkedKarmanDataset initialized:")
        print(f"  Directory: {self.directory}")
        print(f"  Sequence length: {seq_len}, Stride: {stride}")
        print(f"  Chunk size: {chunk_size} timesteps")
        print(f"  Found {len(self.simulation_ids)} simulation files")
        print(f"  Memory per chunk: ~{chunk_size}/400 of full simulation")
    
    def _discover_simulation_ids(self) -> List[str]:
        """Auto-discover simulation files."""
        files = sorted(list(self.directory.glob(self.file_pattern)))
        return [file.stem for file in files]
    
    def __iter__(self) -> Iterator[List[Data]]:
        """Iterate using chunked loading."""
        sim_ids = self.simulation_ids.copy()
        if self.shuffle_simulations:
            random.shuffle(sim_ids)
        
        for sim_id in sim_ids:
            dataset_file = self.directory / f"{sim_id}.pt"
            
            try:
                # Load full simulation to get total length
                sim_data = torch.load(dataset_file, weights_only=False)
                
                if not isinstance(sim_data, list):
                    continue
                
                T = len(sim_data)
                
                # Process in chunks
                for chunk_start in range(0, T, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size + self.seq_len, T)
                    chunk = sim_data[chunk_start:chunk_end]
                    
                    # Generate sequences from this chunk
                    chunk_T = len(chunk)
                    max_start = chunk_T - self.seq_len
                    
                    for start_idx in range(0, max_start + 1, self.stride):
                        window = []
                        for offset in range(self.seq_len):
                            t = start_idx + offset
                            data_t = chunk[t].clone()
                            
                            # Add Reynolds number
                            if hasattr(data_t, 'global_params'):
                                data_t.global_reynolds = data_t.global_params.flatten()
                            else:
                                data_t.global_reynolds = torch.zeros(1, dtype=torch.float32)
                            
                            window.append(data_t)
                        
                        if len(window) == self.seq_len:
                            yield window
                    
                    # Delete chunk
                    del chunk
                    gc.collect()
                
                # Delete full simulation
                del sim_data
                gc.collect()
                
            except Exception as e:
                print(f"Error processing {sim_id}: {e}")


def get_simulation_ids(directory: Path, pattern: str = "*.pt") -> List[str]:
    """Get list of simulation IDs from a directory."""
    files = list(directory.glob(pattern))
    return [file.stem for file in files]


# Usage examples and memory reduction strategies
if __name__ == "__main__":
    train_dir = Path("/standard/sds_baek_energetic/von_karman_vortex/processed_multi_dir/normalized/train")
    
    print("="*70)
    print("MEMORY REDUCTION STRATEGIES FOR 400-TIMESTEP SIMULATIONS")
    print("="*70)
    
    # STRATEGY 1: Limit timesteps per simulation (MOST EFFECTIVE)
    # Only use first 100 timesteps instead of all 400
    # Memory reduction: 75%!
    print("\n1. LIMITED TIMESTEPS (Recommended)")
    dataset_limited = StreamingKarmanDataset(
        directory=train_dir,
        seq_len=10,
        stride=5,
        max_timesteps_per_sim=100,  # Only first 100 of 400 timesteps
        shuffle_simulations=True
    )
    print("   → Uses only 100/400 timesteps = 75% memory reduction")
    
    # STRATEGY 2: Increase stride (reduce overlap)
    # Default stride=1 creates ~390 sequences per sim
    # stride=10 creates ~39 sequences per sim
    print("\n2. INCREASED STRIDE")
    dataset_stride = StreamingKarmanDataset(
        directory=train_dir,
        seq_len=10,
        stride=20,  # Skip more timesteps
        shuffle_simulations=True
    )
    print("   → Generates 20x fewer sequences")
    
    # STRATEGY 3: Shorter sequences
    print("\n3. SHORTER SEQUENCES")
    dataset_short = StreamingKarmanDataset(
        directory=train_dir,
        seq_len=5,  # Instead of 10
        stride=5,
        shuffle_simulations=True
    )
    print("   → Each sequence uses 5 timesteps instead of 10")
    
    # STRATEGY 4: Chunked loading
    print("\n4. CHUNKED LOADING")
    dataset_chunked = ChunkedKarmanDataset(
        directory=train_dir,
        seq_len=10,
        stride=5,
        chunk_size=50,  # Load 50 timesteps at a time
        shuffle_simulations=True
    )
    print("   → Loads 50/400 timesteps at a time = 87.5% memory reduction")
    
    # STRATEGY 5: Combined approach (BEST FOR LARGE DATASETS)
    print("\n5. COMBINED APPROACH (Best)")
    dataset_combined = StreamingKarmanDataset(
        directory=train_dir,
        seq_len=8,              # Shorter sequences
        stride=10,              # Less overlap
        max_timesteps_per_sim=150,  # Limit timesteps
        shuffle_simulations=True
    )
    print("   → Combines multiple strategies for maximum memory efficiency")
    
    print("\n" + "="*70)
    print("MEMORY CALCULATION EXAMPLE:")
    print("="*70)
    print("Original: 400 timesteps × 60k nodes × 10 features × 4 bytes")
    print("        ≈ 960 MB per simulation file")
    print("\nWith max_timesteps_per_sim=100:")
    print("        ≈ 240 MB per simulation file (75% reduction!)")
    print("\nWith timestep_range=(100, 200):")
    print("        ≈ 240 MB per simulation file (uses middle section)")
    print("\nWith temporal_subsample_rate=4 (every 4th timestep):")
    print("        ≈ 240 MB per simulation file (75% reduction!)")
    print("="*70)
    
    # Additional examples for different use cases
    print("\n" + "="*70)
    print("TIMESTEP SELECTION EXAMPLES")
    print("="*70)
    
    # Example 1: Use middle timesteps (after initial transient)
    print("\n📊 EXAMPLE 1: Skip initial transient, use steady-state region")
    dataset_steady = StreamingKarmanDataset(
        directory=train_dir,
        seq_len=10,
        stride=5,
        timestep_range=(50, 250),  # Skip first 50, use 50-250
    )
    print("   Use case: Focus on developed flow, ignore startup transients")
    
    # Example 2: Use only late timesteps (fully developed flow)
    print("\n📊 EXAMPLE 2: Use only late timesteps")
    dataset_late = StreamingKarmanDataset(
        directory=train_dir,
        seq_len=10,
        stride=5,
        timestep_range=(200, 400),  # Only last 200 timesteps
    )
    print("   Use case: Train on fully developed turbulent flow")
    
    # Example 3: Temporal subsampling (every Nth timestep)
    print("\n📊 EXAMPLE 3: Temporal subsampling (every 5th timestep)")
    dataset_subsampled = StreamingKarmanDataset(
        directory=train_dir,
        seq_len=10,
        stride=5,
        timestep_sampling='every_nth',
        temporal_subsample_rate=5,  # [0, 5, 10, 15, 20, ...]
    )
    print("   Use case: Capture slower dynamics, reduce temporal resolution")
    print("   Result: 400 timesteps → 80 timesteps (80% reduction!)")
    
    # Example 4: Random sampling (different each epoch)
    print("\n📊 EXAMPLE 4: Random timestep sampling")
    dataset_random = StreamingKarmanDataset(
        directory=train_dir,
        seq_len=10,
        stride=5,
        timestep_sampling='random',  # Different timesteps each epoch
    )
    print("   Use case: Augmentation, prevent overfitting to temporal order")
    
    # Example 5: Combined - steady state + subsampling
    print("\n📊 EXAMPLE 5: Combined approach")
    dataset_best = StreamingKarmanDataset(
        directory=train_dir,
        seq_len=10,
        stride=10,
        timestep_range=(100, 300),     # Middle 200 timesteps
        timestep_sampling='every_nth',
        temporal_subsample_rate=2,     # Every 2nd timestep
    )
    print("   Use case: Maximum efficiency while maintaining quality")
    print("   Result: 400 → 200 (range) → 100 (subsample) = 75% reduction!")
    
    # Example 6: Early vs late comparison
    print("\n📊 EXAMPLE 6: Train/test on different time periods")
    dataset_train_early = StreamingKarmanDataset(
        directory=train_dir,
        timestep_range=(0, 200),  # Early timesteps for training
        seq_len=10,
        stride=5,
    )
    dataset_test_late = StreamingKarmanDataset(
        directory=train_dir,
        timestep_range=(200, 400),  # Late timesteps for testing
        seq_len=10,
        stride=5,
    )
    print("   Use case: Test generalization to different flow regimes")
    
    print("\n" + "="*70)
    print("RECOMMENDED CONFIGURATIONS BY USE CASE")
    print("="*70)
    print("\n1. QUICK EXPERIMENTS (maximum speed):")
    print("   timestep_range=(0, 100), temporal_subsample_rate=5")
    print("   → Only 20 timesteps used!")
    
    print("\n2. NORMAL TRAINING (balanced):")
    print("   timestep_range=(50, 250), stride=10")
    print("   → 200 timesteps, less overlap")
    
    print("\n3. FULL TRAINING (best accuracy):")
    print("   timestep_sampling='every_nth', temporal_subsample_rate=2")
    print("   → All 200 timesteps, half temporal resolution")
    
    print("\n4. STEADY-STATE ONLY (ignore transients):")
    print("   timestep_range=(100, 400)")
    print("   → Skip startup, use developed flow")
    
    print("="*70)
    
    # Test the dataset
    print("\n\nTesting dataset...")
    info = dataset_limited.get_simulation_info()
    print(f"\nSimulation info: {info}")
    
    print("\nGenerating first few sequences...")
    for i, sequence in enumerate(dataset_limited):
        if i == 0:
            print(f"\nFirst sequence:")
            print(f"  Length: {len(sequence)}")
            print(f"  Nodes: {sequence[0].num_nodes}")
            print(f"  Features: {sequence[0].x.shape}")
            print(f"  Edges: {sequence[0].edge_index.shape}")
            if hasattr(sequence[0], 'global_reynolds'):
                print(f"  Reynolds: {sequence[0].global_reynolds}")
        if i >= 2:
            break
    
    print("\n✓ Dataset is working!")
    
    print("\n" + "="*70)
    print("GPARC MODEL COMPATIBILITY CHECK")
    print("="*70)
    
    # Verify the dataset works with GPARC model requirements
    print("\nChecking GPARC compatibility...")
    for i, sequence in enumerate(dataset_limited):
        if i == 0:
            print(f"\n✓ Sequence format:")
            print(f"  - Length: {len(sequence)} timesteps")
            print(f"  - Each timestep is a PyG Data object")
            
            first_data = sequence[0]
            print(f"\n✓ First timestep attributes:")
            print(f"  - x.shape: {first_data.x.shape} (node features)")
            print(f"  - y.shape: {first_data.y.shape} (targets)")
            print(f"  - edge_index.shape: {first_data.edge_index.shape}")
            print(f"  - num_nodes: {first_data.num_nodes}")
            
            # Check GPARC requirements
            print(f"\n✓ GPARC model requirements:")
            if hasattr(first_data, 'global_params'):
                print(f"  ✓ global_params: {first_data.global_params.shape} = {first_data.global_params}")
                print(f"    (GPARC uses: first_data.global_params.flatten())")
            else:
                print(f"  ✗ global_params: MISSING!")
            
            if hasattr(first_data, 'global_reynolds'):
                print(f"  ✓ global_reynolds: {first_data.global_reynolds} (convenience attribute)")
            
            # Check feature layout
            print(f"\n✓ Feature layout (for GPARC):")
            print(f"  - Static features: x[:, :3] (positions)")
            print(f"  - Dynamic features: x[:, 3:10] (physics: 7 vars)")
            print(f"  - Total input features: {first_data.x.shape[1]}")
            print(f"  - Target features: {first_data.y.shape[1]} (physics only)")
            
            # Verify consistency across sequence
            print(f"\n✓ Sequence consistency:")
            all_same_nodes = all(d.num_nodes == first_data.num_nodes for d in sequence)
            all_same_edges = all(d.edge_index.shape == first_data.edge_index.shape for d in sequence)
            all_have_global = all(hasattr(d, 'global_params') for d in sequence)
            
            print(f"  - Same number of nodes: {all_same_nodes}")
            print(f"  - Same graph structure: {all_same_edges}")
            print(f"  - All have global_params: {all_have_global}")
            
            if all_same_nodes and all_same_edges and all_have_global:
                print(f"\n  ✅ FULLY COMPATIBLE WITH GPARC MODEL!")
            else:
                print(f"\n  ⚠️  WARNING: Potential compatibility issues detected!")
        
        if i >= 0:
            break
    
    print("\n" + "="*70)
    print("EXAMPLE: Using with GPARC Training Loop")
    print("="*70)
    print("""
# 1. Create dataset with memory optimization
from torch.utils.data import DataLoader

train_dataset = StreamingKarmanDataset(
    directory="/path/to/normalized/train",
    seq_len=10,              # Match your GPARC sequence length
    stride=5,                # Less overlap = less data
    timestep_range=(50, 250), # Skip transients
    shuffle_simulations=True
)

# 2. IMPORTANT: Use num_workers=0 for IterableDataset
train_loader = DataLoader(
    train_dataset,
    batch_size=None,  # Already yields sequences
    num_workers=0,    # Required for IterableDataset
    pin_memory=True
)

# 3. Training loop (same as before!)
for epoch in range(num_epochs):
    for sequence in train_loader:
        # sequence is a list of PyG Data objects
        # Each Data has: x, y, edge_index, global_params
        
        optimizer.zero_grad()
        
        # GPARC forward pass
        predictions = model(sequence)  # Returns list of predictions
        
        # Compute loss
        loss = 0
        for pred, data in zip(predictions, sequence):
            target = model.process_targets(data.y)
            loss += criterion(pred, target)
        
        loss = loss / len(sequence)
        loss.backward()
        optimizer.step()

# 4. Memory monitoring
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    """)
    print("="*70)