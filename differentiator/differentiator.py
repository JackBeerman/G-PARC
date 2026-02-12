# differentiator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from .hop import AdvectionMLS, DiffusionMLS, StrainMLS
from .mappingandrecon import MappingAndRecon

class DerivativeGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=3,
                 num_layers=4, heads=8, concat=True, dropout=0.3, use_residual=True):
        super(DerivativeGNN, self).__init__()
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.out_channels = out_channels

        if self.use_residual:
            self.residual_proj = nn.Linear(in_channels, out_channels)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            current_in = in_channels if i == 0 else hidden_channels * (heads if concat else 1)
            current_out = hidden_channels if i < num_layers - 1 else out_channels
            is_last_layer = (i == num_layers - 1)
            
            self.layers.append(nn.LayerNorm(current_in, eps=1e-6))# adding in , eps=1e-6
            self.layers.append(
                GATConv(current_in, current_out, 
                        heads=1 if is_last_layer else heads,
                        concat=False if is_last_layer else concat, 
                        dropout=dropout)
            )

    def forward(self, x, edge_index):
        residual = self.residual_proj(x) if self.use_residual else None

        for i in range(self.num_layers):
            ln = self.layers[2*i]
            gnn = self.layers[2*i + 1]
            
            x_res = x
            x = ln(x)
            x = gnn(x, edge_index)

            if i < self.num_layers - 1:
                x = F.gelu(x)
                if x.shape == x_res.shape: # Add skip connections between layers
                     x = x + x_res
        
        if self.use_residual and residual is not None:
            x = x + residual
            
        return x

class ADRDifferentiator(nn.Module):
    """
    Graph-based Advection-Diffusion-Reaction Differentiator.
    
    Direct analog to pixel ADRDifferentiator but for graph data.
    Fully configurable for different variable orderings and dimensions.
    
    Parameters
    ----------
    n_state_var: int
        Number of state variables
    n_fe_features: int
        Number of features from feature_extraction
    list_adv_idx: list of int
        List of channel indices to calculate advection on
    list_dif_idx: list of int
        List of channel indices to calculate diffusion on
    feature_extraction: torch.nn.Module
        Feature extraction network. Expected to take PyG Data and output 
        [num_nodes, n_fe_features]
    gradient_solver: SolveGradientsLST
        MLS gradient solver (replaces finite_difference_method)
    laplacian_solver: SolveWeightLST2d
        MLS Laplacian solver (replaces finite_difference_method)
    spade_random_noise: bool, optional
        Whether to add noise in mapping and reconstruction modules (default: True)
    heads: int, optional
        Number of attention heads for GAT layers (default: 4)
    concat: bool, optional
        Whether to concatenate attention heads (default: True)
    dropout: float, optional
        Dropout rate (default: 0.0)
    velocity_start_idx: int, optional
        Starting index of velocity field in data.x (default: n_state_var)
        For standard ordering [state_vars, velocity], use default.
        For custom ordering, specify explicitly.
    n_velocity_components: int, optional
        Number of velocity components (default: 2 for 2D)
        Use 3 for 3D problems, 1 for 1D problems.
    **kwarg: other arguments to be passed to torch.nn.Module
    
    Examples
    --------
    Standard 2D case (Burgers, Navier-Stokes):
        data.x = [state_vars (n_state_var), u, v]
        >>> ADRDifferentiator(n_state_var=1, ...) # velocity at indices [1, 2]
    
    3D case:
        data.x = [state_vars, u, v, w]
        >>> ADRDifferentiator(n_state_var=1, n_velocity_components=3, ...)
    
    Custom ordering (velocity first):
        data.x = [u, v, state_vars]
        >>> ADRDifferentiator(n_state_var=1, velocity_start_idx=0, ...)
    
    Multiple state variables:
        data.x = [temperature, pressure, concentration, u, v]
        >>> ADRDifferentiator(n_state_var=3, ...) # velocity at indices [3, 4]
    """
    
    def __init__(
        self,
        n_state_var,
        n_fe_features,
        list_adv_idx,
        list_dif_idx,
        feature_extraction,
        gradient_solver,
        laplacian_solver,
        spade_random_noise=True,
        heads=4,
        concat=True,
        dropout=0.0,
        velocity_start_idx=None,
        n_velocity_components=2,
        **kwarg,
    ):
        super(ADRDifferentiator, self).__init__(**kwarg)
        
        # Store configuration
        self.n_state_var = n_state_var
        self.n_velocity_components = n_velocity_components
        self.feature_extraction = feature_extraction
        
        # Determine velocity indices
        if velocity_start_idx is None:
            # Default: velocity comes after state variables
            self.velocity_start_idx = n_state_var
        else:
            self.velocity_start_idx = velocity_start_idx
        
        self.velocity_end_idx = self.velocity_start_idx + n_velocity_components
        
        # Total number of variables
        self.n_total_vars = n_state_var + n_velocity_components
        
        # Module lists
        self.list_adv = nn.ModuleList()
        self.list_dif = nn.ModuleList()
        self.list_mar = nn.ModuleList()
        n_explicit_features = [0 for _ in range(self.n_total_vars)]
        
        # Initializing advections
        for i in range(self.n_total_vars):
            if i in list_adv_idx:
                self.list_adv.append(AdvectionMLS(gradient_solver))
                n_explicit_features[i] += 1
            else:
                self.list_adv.append(None)
        
        # Initializing diffusions
        for i in range(self.n_total_vars):
            if i in list_dif_idx:
                self.list_dif.append(DiffusionMLS(laplacian_solver))
                n_explicit_features[i] += 1
            else:
                self.list_dif.append(None)
        
        # Initializing mapping and reconstruction
        # State variables first
        for i in range(n_state_var):
            if n_explicit_features[i] == 0:
                # No explicit features
                self.list_mar.append(None)
            else:
                # One or more explicit feature
                self.list_mar.append(
                    MappingAndRecon(
                        n_base_features=n_fe_features,
                        n_mask_channel=n_explicit_features[i],
                        output_channel=1,
                        heads=heads,
                        concat=concat,
                        dropout=dropout,
                        add_noise=spade_random_noise,
                    )
                )
        
        # Velocity variables second
        # Count explicit features for all velocity components
        n_velocity_explicit = sum([
            n_explicit_features[self.velocity_start_idx + j] 
            for j in range(n_velocity_components)
        ])
        
        if n_velocity_explicit == 0:
            self.list_mar.append(None)
        else:
            self.list_mar.append(
                MappingAndRecon(
                    n_base_features=n_fe_features,
                    n_mask_channel=n_velocity_explicit,
                    output_channel=n_velocity_components,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    add_noise=spade_random_noise,
                )
            )
        
        self._weights_initialized = False
    
    def initialize_weights(self, sample_data):
        """
        Pre-compute MLS weights once before training.
        
        Parameters
        ----------
        sample_data: Data
            A sample PyG Data object from the dataset
        """
        if not self._weights_initialized:
            print("Initializing MLS operator weights...")
            
            dummy_u = torch.zeros(sample_data.num_nodes, self.n_velocity_components, 
                                device=sample_data.pos.device)
            
            # Get gradient solver from first advection module
            grad_solver = None
            for adv in self.list_adv:
                if adv is not None:
                    grad_solver = adv.gradient_solver
                    break
            
            # Get laplacian solver from first diffusion module
            lap_solver = None
            for dif in self.list_dif:
                if dif is not None:
                    lap_solver = dif.laplacian_solver
                    break
            
            if grad_solver:
                _ = grad_solver(sample_data, dummy_u)
            if lap_solver:
                _ = lap_solver(sample_data)
            
            self._weights_initialized = True
            print(f"✓ MLS weights initialized (n_state_var={self.n_state_var}, "
                  f"velocity_indices=[{self.velocity_start_idx}:{self.velocity_end_idx}])")
    
    def forward(self, t, current):
        """
        Forward of differentiator. Advection and diffusion will be calculated 
        per channel for those necessary and combined with dynamic features.
        Those that do not have explicit advection and diffusion calculation 
        will have zero as output. This design choice was made because certain 
        integrators (e.g. those in torchdiffeq) require differentiator to have 
        the same output and input shape.
        
        Parameters
        ----------
        t: float
            Float scalar for current time
        current: Data
            PyG Data object with x [num_nodes, n_total_vars] containing the 
            current state and velocity variables. Also contains edge_index and pos.
        
        Returns
        -------
        t_dot: torch.Tensor
            Tensor of shape [num_nodes, n_total_vars], the predicted time 
            derivatives on current state and velocity variables
        """
        # Extract graph data
        if isinstance(current, Data):
            data = current
            current_x = data.x
        else:
            raise TypeError("current must be a PyG Data object for graph version")
        
        # Extract velocity field
        velocity_field = current_x[:, self.velocity_start_idx:self.velocity_end_idx]
        
        # Feature extraction
        dynamic_features = self.feature_extraction(data)
        
        t_dot = []
        
        # State variables
        for i in range(self.n_state_var):
            if self.list_mar[i] is not None:
                explicit_features = []
                
                if self.list_adv[i] is not None:
                    explicit_features.append(
                        self.list_adv[i](
                            current_x[:, i:i+1], 
                            velocity_field,
                            data
                        )
                    )
                
                if self.list_dif[i] is not None:
                    explicit_features.append(
                        self.list_dif[i](current_x[:, i:i+1], data)
                    )
                
                t_dot.append(
                    self.list_mar[i](
                        dynamic_features,
                        torch.cat(explicit_features, 1),
                        data.edge_index
                    )
                )
            else:
                t_dot.append(torch.zeros(current_x.shape[0], 1, device=current_x.device))
        
        # Velocity variables
        if self.list_mar[-1] is not None:
            explicit_features = []
            
            for j in range(self.n_velocity_components):
                i = self.velocity_start_idx + j
                
                if self.list_adv[i] is not None:
                    explicit_features.append(
                        self.list_adv[i](
                            current_x[:, i:i+1],
                            velocity_field,
                            data
                        )
                    )
                
                if self.list_dif[i] is not None:
                    explicit_features.append(
                        self.list_dif[i](current_x[:, i:i+1], data)
                    )
            
            t_dot.append(
                self.list_mar[-1](
                    dynamic_features,
                    torch.cat(explicit_features, 1),
                    data.edge_index
                )
            )
        else:
            t_dot.append(torch.zeros(current_x.shape[0], self.n_velocity_components, 
                                    device=current_x.device))
        
        t_dot = torch.cat(t_dot, 1)
        return t_dot



class ElastoPlasticDifferentiator(nn.Module):
    """
    Self-contained ElastoPlastic Differentiator.
    
    Update: 
    - NO boundary masking - learns full physics field
    - BC enforcement handled by integrator only
    - Cleaner gradient flow for learning
    """
    
    def __init__(
        self,
        num_static_feats,
        num_dynamic_feats,
        feature_extractor,
        gradient_solver,
        laplacian_solver,
        n_fe_features,
        list_strain_idx,
        list_laplacian_idx,
        spade_random_noise=True,
        heads=4,
        concat=True,
        dropout=0.0,
        use_von_mises=True,
        use_volumetric=True,
        n_state_var=0,
        zero_init=True,
        **kwarg,
    ):
        super(ElastoPlasticDifferentiator, self).__init__(**kwarg)
        
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.n_state_var = n_state_var
        self.n_displacement_components = num_dynamic_feats
        self.n_fe_features = n_fe_features
        
        self.feature_extractor = feature_extractor
        
        self.displacement_start_idx = n_state_var
        self.displacement_end_idx = n_state_var + num_dynamic_feats
        self.n_total_vars = n_state_var + num_dynamic_feats
        
        self.list_strain = nn.ModuleList()
        self.list_laplacian = nn.ModuleList()
        self.list_mar = nn.ModuleList()
        n_explicit_features = [0 for _ in range(self.n_total_vars)]

        # Operators Setup
        for i in range(self.n_total_vars):
            if i in list_strain_idx:
                self.list_strain.append(StrainMLS(
                    gradient_solver, 
                    use_von_mises=use_von_mises, 
                    use_volumetric=use_volumetric, 
                    n_dimensions=num_dynamic_feats
                ))
                if i < n_state_var:
                    if num_dynamic_feats == 2:
                        n_explicit_features[i] += 3 + int(use_von_mises) + int(use_volumetric)
            else:
                self.list_strain.append(None)
        
        for i in range(self.n_total_vars):
            if i in list_laplacian_idx:
                self.list_laplacian.append(DiffusionMLS(laplacian_solver))
                n_explicit_features[i] += 1
            else:
                self.list_laplacian.append(None)
        
        # MAR Blocks for State Variables
        for i in range(n_state_var):
            if n_explicit_features[i] == 0:
                self.list_mar.append(None)
            else:
                self.list_mar.append(MappingAndRecon(
                    n_base_features=n_fe_features, 
                    n_mask_channel=n_explicit_features[i], 
                    output_channel=1, 
                    heads=heads, 
                    concat=concat, 
                    dropout=dropout, 
                    add_noise=spade_random_noise, 
                    zero_init=zero_init
                ))
        
        # Displacement MAR
        n_displacement_explicit = 0
        displacement_has_strain = any([
            self.list_strain[self.displacement_start_idx + j] is not None 
            for j in range(num_dynamic_feats)
        ])
        
        if displacement_has_strain:
            if num_dynamic_feats == 2:
                n_displacement_explicit += 3 + int(use_von_mises) + int(use_volumetric)
            elif num_dynamic_feats == 3:
                n_displacement_explicit += 6 + int(use_von_mises) + int(use_volumetric)
        
        for j in range(num_dynamic_feats):
            i = self.displacement_start_idx + j
            if self.list_laplacian[i] is not None:
                n_displacement_explicit += 1
        
        if n_displacement_explicit == 0:
            self.list_mar.append(None)
        else:
            self.list_mar.append(MappingAndRecon(
                n_base_features=n_fe_features, 
                n_mask_channel=n_displacement_explicit, 
                output_channel=num_dynamic_feats, 
                heads=heads, 
                concat=concat, 
                dropout=dropout, 
                add_noise=spade_random_noise, 
                zero_init=zero_init
            ))
        
        self._weights_initialized = False

    def initialize_weights(self, sample_data):
        """Pre-compute MLS weights once before training."""
        if not self._weights_initialized:
            print("Initializing MLS operator weights...")
            dummy_u = torch.zeros(
                sample_data.num_nodes, 
                self.n_displacement_components, 
                device=sample_data.pos.device
            )
            
            # Initialize gradient solver
            grad_solver = None
            for strain in self.list_strain:
                if strain is not None:
                    grad_solver = strain.gradient_solver
                    break
            
            # Initialize laplacian solver
            lap_solver = None
            for lap in self.list_laplacian:
                if lap is not None:
                    lap_solver = lap.laplacian_solver
                    break
            
            if grad_solver: 
                _ = grad_solver(sample_data, dummy_u)
            if lap_solver: 
                _ = lap_solver(sample_data)
            
            self._weights_initialized = True
            print(f"✓ MLS weights initialized")

    def forward(self, full_state, edge_index):
        # DEBUG: Check inputs
        #print(f"[DEBUG] full_state shape: {full_state.shape}, range: [{full_state.min():.4f}, {full_state.max():.4f}]")
       # print(f"[DEBUG] full_state has NaN: {torch.isnan(full_state).any()}, Inf: {torch.isinf(full_state).any()}")
        
        if not self._weights_initialized:
            raise RuntimeError("initialize_weights() must be called before forward()")
        
        static_features = full_state[:, :self.num_static_feats]
        dynamic_state = full_state[:, self.num_static_feats:]
        
        # Check after split
        #print(f"[DEBUG] static range: [{static_features.min():.4f}, {static_features.max():.4f}]")
        #print(f"[DEBUG] dynamic range: [{dynamic_state.min():.4f}, {dynamic_state.max():.4f}]")
        
        #learned_features = self.feature_extractor(static_features, edge_index)
        learned_features = self.feature_extractor(static_features, edge_index, pos=static_features)
        #print(f"[DEBUG] learned_features range: [{learned_features.min():.4f}, {learned_features.max():.4f}]")
        #print(f"[DEBUG] learned_features has NaN: {torch.isnan(learned_features).any()}")
    
        
        # Create Data object for MLS operators
        data = Data(x=dynamic_state, edge_index=edge_index, pos=static_features)
        if hasattr(edge_index, 'mesh_id'):
            data.mesh_id = edge_index.mesh_id
        
        displacement_field = dynamic_state[:, self.displacement_start_idx:self.displacement_end_idx]
        
        t_dot = []
        
        # State Variables Loop
        for i in range(self.n_state_var):
            if self.list_mar[i] is not None:
                explicit_features = []
                
                if self.list_strain[i] is not None:
                    feat = self.list_strain[i](dynamic_state[:, i:i+1], data)
                    explicit_features.append(feat)
                
                if self.list_laplacian[i] is not None:
                    feat = self.list_laplacian[i](dynamic_state[:, i:i+1], data)
                    explicit_features.append(feat)
                
                t_dot.append(self.list_mar[i](
                    learned_features, 
                    torch.cat(explicit_features, 1), 
                    data.edge_index
                ))
            else:
                t_dot.append(torch.zeros(
                    dynamic_state.shape[0], 1, 
                    device=dynamic_state.device
                ))
        
        # Displacement Variables Loop
        if self.list_mar[-1] is not None:
            explicit_features = []
            
            # Check if strain features needed
            displacement_has_strain = any([
                self.list_strain[self.displacement_start_idx + j] is not None 
                for j in range(self.n_displacement_components)
            ])
            
            if displacement_has_strain:
                strain_op = None
                for j in range(self.n_displacement_components):
                    i = self.displacement_start_idx + j
                    if self.list_strain[i] is not None:
                        strain_op = self.list_strain[i]
                        break
                
                if strain_op is not None:
                    strain_feats = strain_op(displacement_field, data)
                    explicit_features.append(strain_feats)
            
            # Laplacian features
            for j in range(self.n_displacement_components):
                i = self.displacement_start_idx + j
                if self.list_laplacian[i] is not None:
                    lap_feat = self.list_laplacian[i](dynamic_state[:, i:i+1], data)
                    explicit_features.append(lap_feat)
            
            # Compute displacement derivative
            # NO BOUNDARY MASKING - let network learn full physics field
            # BC enforcement happens in the integrator
            disp_derivative = self.list_mar[-1](
                learned_features, 
                torch.cat(explicit_features, 1), 
                data.edge_index
            )
            
            t_dot.append(disp_derivative)
        else:
            t_dot.append(torch.zeros(
                dynamic_state.shape[0], 
                self.n_displacement_components, 
                device=dynamic_state.device
            ))
        
        return torch.cat(t_dot, 1)
    