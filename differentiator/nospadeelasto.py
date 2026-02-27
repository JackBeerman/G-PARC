"""
ElastoPlasticDifferentiator — NoSPADE (Concat + MLP Fusion)
============================================================
Drop-in replacement for ElastoPlasticDifferentiator that replaces SPADE
(MappingAndRecon) with simple concat + MLP, matching the approach that
improved shock tube and river performance.

Physics operators are IDENTICAL:
  - StrainMLS: eps_xx, eps_yy, eps_xy, von_mises, volumetric (5 features)
  - DiffusionMLS (Laplacian): per-component (2 features)
  Total: 7 physics features for 2D displacement

Fusion change:
  SPADE version:  learned_features → SPADE(physics) → ResNet → output
  This version:   concat[learned, physics_normed] → MLP → output

The MLP is per-variable-group (one for displacement), with:
  - LayerNorm on physics features (scale to O(1))
  - Two hidden layers with GELU + residual
  - Zero-initialized output for stable training start

Keeps identical:
  - Feature extractor (GraphConvFeatureExtractorV2)
  - MLS operators (StrainMLS, DiffusionMLS)
  - initialize_weights() interface
  - forward() signature: (full_state, edge_index) → dφ/dt
  - All constructor args (SPADE args accepted but ignored for compat)
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .hop import StrainMLS, DiffusionMLS


class PhysicsFusionMLP(nn.Module):
    """
    MLP that fuses learned features with physics features via concatenation.

    Input:  [learned_features (n_fe) | physics_features (n_phys)]
    Output: [n_output]

    Two hidden layers with GELU + residual. Zero-initialized output.
    """
    def __init__(self, n_learned, n_physics, n_output=2, hidden_dim=128, zero_init=True):
        super().__init__()
        input_dim = n_learned + n_physics

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden_dim, n_output)
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, learned_features, physics_features):
        x = torch.cat([learned_features, physics_features], dim=-1)
        h = self.net(x) + self.residual(x)
        return self.out(h)


class ElastoPlasticDifferentiatorNoSPADE(nn.Module):
    """
    ElastoPlastic differentiator with concat+MLP fusion instead of SPADE.

    Constructor signature matches ElastoPlasticDifferentiator exactly,
    so training scripts only need to change the import.
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
        use_von_mises=True,
        use_volumetric=True,
        n_state_var=0,
        zero_init=True,
        fusion_hidden_dim=128,
        # Legacy SPADE args — accepted but ignored
        spade_random_noise=True,
        heads=4,
        concat=True,
        dropout=0.0,
        **kwarg,
    ):
        super().__init__(**kwarg)

        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.n_state_var = n_state_var
        self.n_displacement_components = num_dynamic_feats
        self.n_fe_features = n_fe_features

        self.feature_extractor = feature_extractor

        self.displacement_start_idx = n_state_var
        self.displacement_end_idx = n_state_var + num_dynamic_feats
        self.n_total_vars = n_state_var + num_dynamic_feats

        # --- Physics operators (identical to SPADE version) ---
        self.list_strain = nn.ModuleList()
        self.list_laplacian = nn.ModuleList()
        n_explicit_features = [0 for _ in range(self.n_total_vars)]

        for i in range(self.n_total_vars):
            if i in list_strain_idx:
                self.list_strain.append(StrainMLS(
                    gradient_solver,
                    use_von_mises=use_von_mises,
                    use_volumetric=use_volumetric,
                    n_dimensions=num_dynamic_feats,
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

        # --- Count displacement physics features ---
        n_displacement_explicit = 0
        displacement_has_strain = any(
            self.list_strain[self.displacement_start_idx + j] is not None
            for j in range(num_dynamic_feats)
        )
        if displacement_has_strain:
            if num_dynamic_feats == 2:
                n_displacement_explicit += 3 + int(use_von_mises) + int(use_volumetric)
            elif num_dynamic_feats == 3:
                n_displacement_explicit += 6 + int(use_von_mises) + int(use_volumetric)

        for j in range(num_dynamic_feats):
            i = self.displacement_start_idx + j
            if self.list_laplacian[i] is not None:
                n_displacement_explicit += 1

        self.n_displacement_explicit = n_displacement_explicit

        # --- State variable fusion MLPs (if any) ---
        self.list_state_fusion = nn.ModuleList()
        self.list_state_phys_norm = nn.ModuleList()
        for i in range(n_state_var):
            if n_explicit_features[i] > 0:
                self.list_state_phys_norm.append(nn.LayerNorm(n_explicit_features[i]))
                self.list_state_fusion.append(PhysicsFusionMLP(
                    n_learned=n_fe_features,
                    n_physics=n_explicit_features[i],
                    n_output=1,
                    hidden_dim=fusion_hidden_dim,
                    zero_init=zero_init,
                ))
            else:
                self.list_state_phys_norm.append(None)
                self.list_state_fusion.append(None)

        # --- Displacement fusion MLP ---
        if n_displacement_explicit > 0:
            self.disp_phys_norm = nn.LayerNorm(n_displacement_explicit)
            self.disp_fusion = PhysicsFusionMLP(
                n_learned=n_fe_features,
                n_physics=n_displacement_explicit,
                n_output=num_dynamic_feats,
                hidden_dim=fusion_hidden_dim,
                zero_init=zero_init,
            )
        else:
            self.disp_phys_norm = None
            self.disp_fusion = None

        self._weights_initialized = False

        print(f"  ElastoPlasticDifferentiator (NoSPADE)")
        print(f"    Physics features: {n_displacement_explicit} "
              f"(strain={displacement_has_strain}, "
              f"von_mises={use_von_mises}, volumetric={use_volumetric})")
        print(f"    Fusion: concat[{n_fe_features} learned + "
              f"{n_displacement_explicit} physics] → MLP({fusion_hidden_dim}) → {num_dynamic_feats}")

    def initialize_weights(self, sample_data):
        """Pre-compute MLS weights once before training."""
        if not self._weights_initialized:
            print("Initializing MLS operator weights...")
            dummy_u = torch.zeros(
                sample_data.num_nodes,
                self.n_displacement_components,
                device=sample_data.pos.device,
            )

            grad_solver = None
            for strain in self.list_strain:
                if strain is not None:
                    grad_solver = strain.gradient_solver
                    break

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
            print("✓ MLS weights initialized")

    def forward(self, full_state, edge_index):
        if not self._weights_initialized:
            raise RuntimeError("initialize_weights() must be called before forward()")

        static_features = full_state[:, :self.num_static_feats]
        dynamic_state = full_state[:, self.num_static_feats:]

        # Learned features (static + dynamic → feature extractor)
        fe_input = torch.cat([static_features, dynamic_state], dim=-1)
        learned_features = self.feature_extractor(fe_input, edge_index, pos=static_features)

        # Mesh data for MLS
        data = Data(x=dynamic_state, edge_index=edge_index, pos=static_features)
        if hasattr(edge_index, 'mesh_id'):
            data.mesh_id = edge_index.mesh_id

        displacement_field = dynamic_state[:, self.displacement_start_idx:self.displacement_end_idx]

        t_dot = []

        # --- State variables (if any) ---
        for i in range(self.n_state_var):
            if self.list_state_fusion[i] is not None:
                explicit_features = []
                if self.list_strain[i] is not None:
                    explicit_features.append(self.list_strain[i](dynamic_state[:, i:i+1], data))
                if self.list_laplacian[i] is not None:
                    explicit_features.append(self.list_laplacian[i](dynamic_state[:, i:i+1], data))

                phys_cat = torch.cat(explicit_features, dim=1)
                phys_cat = self.list_state_phys_norm[i](phys_cat)
                t_dot.append(self.list_state_fusion[i](learned_features, phys_cat))
            else:
                t_dot.append(torch.zeros(
                    dynamic_state.shape[0], 1, device=dynamic_state.device
                ))

        # --- Displacement ---
        if self.disp_fusion is not None:
            explicit_features = []

            # Strain features
            displacement_has_strain = any(
                self.list_strain[self.displacement_start_idx + j] is not None
                for j in range(self.n_displacement_components)
            )
            if displacement_has_strain:
                strain_op = None
                for j in range(self.n_displacement_components):
                    i = self.displacement_start_idx + j
                    if self.list_strain[i] is not None:
                        strain_op = self.list_strain[i]
                        break
                if strain_op is not None:
                    explicit_features.append(strain_op(displacement_field, data))

            # Laplacian features
            for j in range(self.n_displacement_components):
                i = self.displacement_start_idx + j
                if self.list_laplacian[i] is not None:
                    explicit_features.append(
                        self.list_laplacian[i](dynamic_state[:, i:i+1], data)
                    )

            phys_cat = torch.cat(explicit_features, dim=1)
            phys_cat = self.disp_phys_norm(phys_cat)
            t_dot.append(self.disp_fusion(learned_features, phys_cat))
        else:
            t_dot.append(torch.zeros(
                dynamic_state.shape[0], self.n_displacement_components,
                device=dynamic_state.device,
            ))

        return torch.cat(t_dot, dim=1)