import torch
from pathlib import Path
from typing import List, Iterator, Union
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from tqdm import tqdm


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