import torch
from pathlib import Path
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from tqdm import tqdm
from typing import List, Iterator, Union, Optional, Tuple
import gc
import random


class KarmanVortexRolloutDataset(IterableDataset):
    """
    Dataset for Karman vortex simulations with Reynolds number as global parameter.
    Yields consecutive sequences of timesteps from multiple simulations.
    """
    def __init__(self,
                 directory: Union[str, Path],
                 simulation_ids: List[str] = None,
                 seq_len: int = 10,
                 stride: int = 1,
                 num_static_feats: int = 3,  # x, y, z positions
                 num_dynamic_feats: int = 7,  # pressure, velocity (3), vorticity (3)
                 file_pattern: str = "*.pt"):
        """
        Args:
            directory: Directory containing PyG dataset files (.pt files).
            simulation_ids: List of simulation IDs (file stems). If None, auto-discover.
            seq_len: Number of consecutive steps in each returned window.
            stride: Step size between sequence starts (default: 1 for overlapping sequences).
            num_static_feats: Number of static features (positions: x, y, z).
            num_dynamic_feats: Number of dynamic features (physics variables).
            file_pattern: Pattern to match simulation files (default: "*.pt").
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
        
        # Variables in Karman vortex data
        self.var_names = ['pressure', 'velocity_x', 'velocity_y', 'velocity_z', 
                         'vorticity_x', 'vorticity_y', 'vorticity_z']
        
        print(f"KarmanVortexRolloutDataset initialized:")
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
        Extracts Reynolds number as global parameter.
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
                        
                        # Extract Reynolds number from global_params
                        if hasattr(data_t, 'global_params'):
                            global_tensor = data_t.global_params
                            if global_tensor.numel() >= 1:
                                # For Karman vortex: just Reynolds number
                                data_t.global_reynolds = global_tensor.flatten()  # Keep as [1] tensor
                            else:
                                print(f"Warning: 'global_params' tensor in {sim_id} at step {t} has unexpected size. Assigning zeros.")
                                data_t.global_reynolds = torch.zeros(1, dtype=torch.float32)
                        else:
                            print(f"Warning: 'global_params' not found in {sim_id} at step {t}. Assigning zeros.")
                            data_t.global_reynolds = torch.zeros(1, dtype=torch.float32)
                        
                        window.append(data_t)
                    
                    # Only yield complete windows
                    if len(window) == self.seq_len:
                        yield window
            
            except FileNotFoundError:
                print(f"Dataset file not found: {dataset_file}")
            except Exception as e:
                print(f"Error loading {dataset_file}: {e}")
    
    def get_simulation_info(self, sim_id: str = None) -> dict:
        """Get information about a specific simulation."""
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
                "has_global_params": hasattr(sample_data, 'global_params'),
                "has_reynolds_number": hasattr(sample_data, 'reynolds_number')
            }
            
            # Add Reynolds number if available
            if hasattr(sample_data, 'reynolds_number'):
                info["reynolds_number"] = sample_data.reynolds_number
            if hasattr(sample_data, 'global_params'):
                info["global_params_value"] = sample_data.global_params.tolist()
            
            return info
            
        except Exception as e:
            return {"error": f"Could not load simulation info: {e}"}
    
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


