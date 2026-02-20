"""
ElastoPlastic Differentiator V3 — Erosion-Aware
=================================================
Extends ElastoPlasticDifferentiator with:
  1. Erosion-masked MLS: edge weights zero out contributions through eroded material
  2. Erosion as explicit SPADE channel: the SPADE block sees erosion status alongside
     strain and Laplacian features
  3. Feature caching for the erosion head (inherited from v2)

The mesh topology stays static. Only the edge weights change per timestep.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .differentiator import ElastoPlasticDifferentiator
from .hop import StrainMLS, DiffusionMLS, apply_laplacian
from .mappingandrecon import MappingAndRecon
from .erosion_utils import EdgeErosionCache


class ElastoPlasticDifferentiatorV3(ElastoPlasticDifferentiator):
    """
    Erosion-aware differentiator.
    
    Differences from V2:
      - MLS gradient and Laplacian computations are masked by edge erosion weights
      - Erosion state is added as an extra explicit feature to the displacement SPADE block
      - The MappingAndRecon for displacement has +1 n_mask_channel for erosion
    """
    
    def __init__(self, *args, **kwargs):
        # Pop v3-specific args before passing to parent
        # (parent doesn't know about erosion SPADE channel)
        self._v3_init = True
        super().__init__(*args, **kwargs)
        self._v3_init = False
        
        # Edge erosion cache (initialized per mesh)
        self.edge_erosion_cache = EdgeErosionCache()
        
        # Always cache features (for erosion head)
        self.cache_features = True
        
        # Rebuild the displacement MappingAndRecon with +1 channel for erosion
        # The parent already built list_mar[-1] with n_displacement_explicit channels.
        # We need n_displacement_explicit + 1.
        self._rebuild_displacement_mar_with_erosion(**kwargs)
    
    def _rebuild_displacement_mar_with_erosion(self, **kwargs):
        """
        Replace the displacement MappingAndRecon with one that has +1 input
        channel for the node-level erosion mask.
        """
        # Count original explicit features (same logic as parent __init__)
        n_displacement_explicit = 0
        displacement_has_strain = any([
            self.list_strain[self.displacement_start_idx + j] is not None
            for j in range(self.n_displacement_components)
        ])
        
        if displacement_has_strain:
            if self.n_displacement_components == 2:
                n_displacement_explicit += 3 + int(
                    any(s is not None and s.use_von_mises for s in self.list_strain)
                ) + int(
                    any(s is not None and s.use_volumetric for s in self.list_strain)
                )
        
        for j in range(self.n_displacement_components):
            i = self.displacement_start_idx + j
            if self.list_laplacian[i] is not None:
                n_displacement_explicit += 1
        
        self.n_displacement_explicit_base = n_displacement_explicit
        
        # +1 for erosion channel
        n_with_erosion = n_displacement_explicit + 1
        
        # Get MAR construction params from kwargs or existing MAR
        old_mar = self.list_mar[-1]
        if old_mar is not None:
            # Extract params from existing MAR
            heads = kwargs.get('heads', 4)
            concat = kwargs.get('concat', True)
            dropout = kwargs.get('dropout', 0.0)
            spade_random_noise = kwargs.get('spade_random_noise', False)
            zero_init = kwargs.get('zero_init', True)
            
            new_mar = MappingAndRecon(
                n_base_features=self.n_fe_features,
                n_mask_channel=n_with_erosion,
                output_channel=self.n_displacement_components,
                heads=heads,
                concat=concat,
                dropout=dropout,
                add_noise=spade_random_noise,
                zero_init=zero_init,
            )
            
            # Copy weights from old MAR for the original channels
            # The SPADE linear layers go from n_mask_channel → hidden,
            # so we can partially load weights
            self._partial_load_mar(old_mar, new_mar, n_displacement_explicit, n_with_erosion)
            
            self.list_mar[-1] = new_mar
    
    def _partial_load_mar(self, old_mar, new_mar, old_channels, new_channels):
        """
        Copy matching weights from old MAR to new MAR.
        The SPADE block's first linear layer changes from [old_channels, H] to [new_channels, H].
        We copy the [old_channels, H] portion and leave the new erosion channel at zero-init.
        """
        old_sd = old_mar.state_dict()
        new_sd = new_mar.state_dict()
        
        for key in new_sd:
            if key in old_sd:
                old_w = old_sd[key]
                new_w = new_sd[key]
                
                if old_w.shape == new_w.shape:
                    # Same shape — direct copy
                    new_sd[key] = old_w
                elif len(old_w.shape) == 2 and len(new_w.shape) == 2:
                    # Weight matrix with different input dim (the SPADE linear)
                    # old: [out, old_in], new: [out, new_in]
                    if old_w.shape[0] == new_w.shape[0] and old_w.shape[1] < new_w.shape[1]:
                        new_sd[key][:, :old_w.shape[1]] = old_w
                        # New channels stay at zero (from zero_init)
                elif len(old_w.shape) == 1 and len(new_w.shape) == 1:
                    if old_w.shape[0] == new_w.shape[0]:
                        new_sd[key] = old_w
        
        new_mar.load_state_dict(new_sd)
    
    def initialize_weights(self, sample_data):
        """Initialize MLS weights AND edge-element mapping."""
        super().initialize_weights(sample_data)
        
        # Build edge→element mapping for erosion masking
        if hasattr(sample_data, 'elements') and sample_data.elements is not None:
            if not self.edge_erosion_cache.initialized:
                print("  Building edge→element mapping for erosion-aware MLS...")
                self.edge_erosion_cache.initialize(
                    sample_data.edge_index, sample_data.elements
                )
                print(f"  ✓ Edge erosion cache ready")
    
    def forward(self, full_state, edge_index, erosion_node_mask=None, 
                erosion_elem=None):
        """
        Erosion-aware forward pass.
        
        Args:
            full_state: [N, S+D] concatenated static + dynamic features
            edge_index: edge connectivity
            erosion_node_mask: [N, 1] float — 1.0 if node touches eroded element
                              (None = no erosion, all valid)
            erosion_elem: [M] float — element erosion status (1.0=valid, 0.0=eroded)
                          Used to compute edge weights for MLS masking.
                          (None = all elements valid)
        
        Returns:
            t_dot: [N, D] time derivatives
        """
        if not self._weights_initialized:
            raise RuntimeError("initialize_weights() must be called before forward()")
        
        static_features = full_state[:, :self.num_static_feats]
        dynamic_state = full_state[:, self.num_static_feats:]
        
        learned_features = self.feature_extractor(
            static_features, edge_index, pos=static_features
        )
        
        # Create Data object for MLS operators
        data = Data(x=dynamic_state, edge_index=edge_index, pos=static_features)
        if hasattr(edge_index, 'mesh_id'):
            data.mesh_id = edge_index.mesh_id
        
        # Compute edge erosion weights
        edge_mask = None
        if erosion_elem is not None and self.edge_erosion_cache.initialized:
            edge_mask = self.edge_erosion_cache.get_weights(erosion_elem)
        
        displacement_field = dynamic_state[:, 
            self.displacement_start_idx:self.displacement_end_idx]
        
        t_dot = []
        
        # State variables loop (same as parent, no erosion needed here)
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
                    dynamic_state.shape[0], 1, device=dynamic_state.device
                ))
        
        # Displacement variables — erosion-aware
        if self.list_mar[-1] is not None:
            explicit_features = []
            
            # Strain with erosion-masked gradients
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
                    strain_feats = self._compute_strain_erosion_aware(
                        strain_op, displacement_field, data, edge_mask
                    )
                    explicit_features.append(strain_feats)
            
            # Laplacian with erosion-masked weights
            for j in range(self.n_displacement_components):
                i = self.displacement_start_idx + j
                if self.list_laplacian[i] is not None:
                    lap_feat = self._compute_laplacian_erosion_aware(
                        self.list_laplacian[i], dynamic_state[:, i:i+1],
                        data, edge_mask
                    )
                    explicit_features.append(lap_feat)
            
            # Append erosion node mask as explicit feature
            if erosion_node_mask is not None:
                explicit_features.append(erosion_node_mask)
            else:
                # No erosion — all zeros (valid everywhere)
                explicit_features.append(
                    torch.zeros(dynamic_state.shape[0], 1, 
                               device=dynamic_state.device)
                )
            
            explicit_cat = torch.cat(explicit_features, 1)
            
            # Always cache features for erosion head
            disp_derivative, resnet_out = self.list_mar[-1](
                learned_features,
                explicit_cat,
                data.edge_index,
                return_features=True
            )
            self._cached_features = torch.cat([
                resnet_out,     # [N, 128] learned
                explicit_cat,   # [N, 7+1] physics + erosion
            ], dim=1)
            
            t_dot.append(disp_derivative)
        else:
            t_dot.append(torch.zeros(
                dynamic_state.shape[0],
                self.n_displacement_components,
                device=dynamic_state.device
            ))
        
        return torch.cat(t_dot, 1)
    
    def _compute_strain_erosion_aware(self, strain_op, displacement, data, edge_mask):
        """
        Compute strain using erosion-masked gradients.
        
        If edge_mask is None, falls back to standard computation.
        Otherwise, masks the gradient contributions from eroded material.
        """
        if edge_mask is None:
            return strain_op(displacement, data)
        
        # We need to intercept the gradient computation
        # The gradient solver uses: du = u[col] - u[row], then scatter_add
        # We mask by multiplying edge contributions by edge_mask
        grad_solver = strain_op.gradient_solver
        pos, edge_index = data.pos, data.edge_index
        key = grad_solver._get_cache_key(data)
        
        features = []
        if strain_op.n_dimensions == 2:
            # Compute masked gradients for each displacement component
            grads = []
            for c in range(displacement.shape[1]):
                g = self._solve_gradient_masked(
                    grad_solver, pos, edge_index, 
                    displacement[:, c:c+1], key, edge_mask
                )
                grads.append(g)
            
            dUx_dx = torch.clamp(grads[0][:, 0:1], -strain_op.sanity_limit, strain_op.sanity_limit)
            dUx_dy = torch.clamp(grads[0][:, 1:2], -strain_op.sanity_limit, strain_op.sanity_limit)
            dUy_dx = torch.clamp(grads[1][:, 0:1], -strain_op.sanity_limit, strain_op.sanity_limit)
            dUy_dy = torch.clamp(grads[1][:, 1:2], -strain_op.sanity_limit, strain_op.sanity_limit)
            
            epsilon_xx = dUx_dx
            epsilon_yy = dUy_dy
            epsilon_xy = 0.5 * (dUx_dy + dUy_dx)
            features.extend([epsilon_xx, epsilon_yy, epsilon_xy])
            
            if strain_op.use_von_mises:
                vm_sq = epsilon_xx**2 + epsilon_yy**2 + epsilon_xx * epsilon_yy + 3 * epsilon_xy**2
                features.append(torch.sqrt(torch.clamp(vm_sq, min=1e-8)))
            if strain_op.use_volumetric:
                features.append(epsilon_xx + epsilon_yy)
        
        return torch.cat(features, dim=1)
    
    def _solve_gradient_masked(self, grad_solver, pos, edge_index, u, cache_key, edge_mask):
        """
        MLS gradient solve with edge erosion masking.
        
        The key idea: multiply each edge's contribution to the RHS by the
        edge erosion weight. Edges through eroded material contribute nothing.
        """
        row, col = edge_index
        N = pos.size(0)
        
        # Get cached geometry
        if cache_key is not None and cache_key in grad_solver.geo_cache:
            M_inv, dX = grad_solver.geo_cache[cache_key]
            if M_inv.device != pos.device:
                M_inv = M_inv.to(pos.device)
                dX = dX.to(pos.device)
                grad_solver.geo_cache[cache_key] = (M_inv, dX)
        else:
            M_inv, dX = grad_solver._precompute_geometry(pos, edge_index, pos.device)
            if cache_key is not None:
                grad_solver.geo_cache[cache_key] = (M_inv, dX)
        
        # Compute RHS with erosion masking
        du = u[col] - u[row]
        V_edge = dX.unsqueeze(2) * du.unsqueeze(1)
        
        # Apply edge erosion mask: zero out contributions from eroded material
        V_edge = V_edge * edge_mask.unsqueeze(1).unsqueeze(2)
        
        V_node = torch.zeros(N, 2, 1, device=pos.device, dtype=torch.float32)
        V_node.index_add_(0, row, V_edge.float())
        
        grads = torch.bmm(M_inv, V_node).squeeze(2)
        grads = torch.clamp(grads, -grad_solver.grad_limit, grad_solver.grad_limit)
        
        return grads.to(dtype=pos.dtype)
    
    def _compute_laplacian_erosion_aware(self, lap_op, state_var, data, edge_mask):
        """
        Compute Laplacian with erosion-masked edge weights.
        """
        if edge_mask is None:
            return lap_op(state_var, data)
        
        weights = lap_op.laplacian_solver(data)
        
        # Apply erosion mask to the Laplacian application
        row, col = data.edge_index
        N = data.pos.shape[0] if hasattr(data, 'pos') else state_var.shape[0]
        diff = state_var[col] - state_var[row]
        
        # Mask: eroded edges contribute nothing to Laplacian
        masked_weights = weights * edge_mask
        weighted_diff = masked_weights.unsqueeze(1) * diff
        
        laplacian = torch.zeros(N, state_var.shape[1], 
                               device=state_var.device, dtype=state_var.dtype)
        laplacian.index_add_(0, row, weighted_diff)
        
        return laplacian