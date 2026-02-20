"""
G-PARCv3 Elastoplastic Model — Erosion-Aware
=============================================
Key changes from v2:
  1. Erosion feeds INTO displacement (bidirectional coupling)
  2. MLS operators masked by edge erosion weights (physics-correct)
  3. No loss masking — train on ALL nodes
  4. Erosion state is an explicit SPADE feature

The autoregressive loop:
  t=0: erosion_state from GT (x_element) or zeros
  For each step:
    1. Compute edge erosion weights from erosion_state
    2. Convert element erosion → node erosion mask
    3. Run differentiator with erosion-masked MLS + erosion as SPADE input
    4. Integrate → displacement prediction
    5. Run erosion head on cached features → element erosion prediction
    6. Apply irreversibility (OR with previous erosion)
    7. Feed erosion back to next step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from integrator.numerical import Euler, Heun, RK4, ImplicitEuler
from models.erosion_head import ErosionHead, element_erosion_to_node


class GPARC_ElastoPlastic_V3(nn.Module):
    """
    Erosion-aware G-PARC for elastoplastic dynamics.
    
    The displacement model receives erosion state as input, and the
    erosion head receives displacement-derived features as input.
    Full bidirectional coupling.
    """
    
    def __init__(
        self,
        derivative_solver,
        erosion_head,
        integrator_type="euler",
        num_static_feats=2,
        num_dynamic_feats=2,
        pos_mean=None,
        pos_std=None,
        boundary_threshold=0.5,
        clamp_output=True,
        clamp_max=10.0,
        norm_method="z_score",
        max_position=None,
        erosion_threshold=0.5,
    ):
        super().__init__()
        
        self.derivative_solver = derivative_solver
        self.erosion_head = erosion_head
        self.num_static_feats = num_static_feats
        self.num_dynamic_feats = num_dynamic_feats
        self.boundary_threshold = boundary_threshold
        self.clamp_output = clamp_output
        self.clamp_max = clamp_max
        self.norm_method = norm_method
        self.max_position = max_position
        self.erosion_threshold = erosion_threshold
        
        if pos_mean is not None and pos_std is not None:
            self.register_buffer("pos_mean", torch.tensor(pos_mean, dtype=torch.float32))
            self.register_buffer("pos_std", torch.tensor(pos_std, dtype=torch.float32))
        else:
            self.pos_mean = None
            self.pos_std = None
        
        # Integrator
        it = integrator_type.lower()
        integrators = {
            "euler": Euler, "heun": Heun, "rk4": RK4,
        }
        if it in ("implicit", "impliciteuler", "implicit_euler"):
            self.integrator = ImplicitEuler(max_iters=3, damping=0.9)
        elif it in integrators:
            self.integrator = integrators[it]()
        else:
            raise ValueError(f"Unknown integrator: {integrator_type}")
    
    # =========================================================================
    # BOUNDARY CONDITIONS
    # =========================================================================
    
    def _denormalize_positions(self, static_feats):
        if self.norm_method == 'global_max' and self.max_position is not None:
            return static_feats * self.max_position
        elif self.pos_mean is not None and self.pos_std is not None:
            return static_feats * self.pos_std.to(static_feats.device) + \
                   self.pos_mean.to(static_feats.device)
        return static_feats
    
    def _enforce_dirichlet_bc(self, F_next, F_current, static_feats):
        if self.pos_mean is None and self.pos_std is None and self.max_position is None:
            return F_next
        pos_phys = self._denormalize_positions(static_feats)
        is_boundary = (pos_phys[:, 0] < self.boundary_threshold).unsqueeze(1)
        return torch.where(is_boundary, F_current, F_next)
    
    # =========================================================================
    # SINGLE STEP — erosion-aware
    # =========================================================================
    
    def step(self, static_feats, dynamic_state, edge_index, dt=1.0,
             erosion_node_mask=None, erosion_elem=None):
        """
        Single integration step with erosion-aware physics.
        
        Args:
            static_feats: [N, S] positions
            dynamic_state: [N, D] displacement
            edge_index: graph connectivity
            dt: timestep
            erosion_node_mask: [N, 1] node-level erosion (1.0 = eroded node)
            erosion_elem: [M] element erosion status (1.0 = valid, 0.0 = eroded)
        
        Returns:
            F_next: [N, D] predicted displacement
        """
        full_state = torch.cat([static_feats, dynamic_state], dim=1)
        
        # The derivative solver is V3 — it handles erosion internally
        def deriv_fn(full_state, edge_index):
            return self.derivative_solver(
                full_state, edge_index,
                erosion_node_mask=erosion_node_mask,
                erosion_elem=erosion_elem,
            )
        
        F_next = self.integrator(
            derivative_fn=deriv_fn,
            static_feats=static_feats,
            dynamic_state=dynamic_state,
            edge_index=edge_index,
            dt=dt,
        )
        
        if self.clamp_output:
            F_next = torch.clamp(F_next, -self.clamp_max, self.clamp_max)
        
        F_next = self._enforce_dirichlet_bc(F_next, dynamic_state, static_feats)
        return F_next
    
    def predict_erosion(self, elements, prev_erosion_nodes):
        """
        Run erosion head on cached differentiator features.
        
        Args:
            elements: [M, 3] element connectivity
            prev_erosion_nodes: [N, 1] node-level erosion from prev step
            
        Returns:
            erosion_logits: [M, 1] raw logits
            erosion_elem: [M] float — updated element erosion (1=valid, 0=eroded)
            erosion_nodes: [N, 1] float — updated node erosion mask
        """
        cached = self.derivative_solver._cached_features
        if cached is None:
            raise RuntimeError("No cached features — run step() first")
        
        logits = self.erosion_head(cached, elements, prev_erosion_nodes)
        
        # Predict: sigmoid > threshold → eroded
        probs = torch.sigmoid(logits)
        pred_eroded = (probs > self.erosion_threshold).float().squeeze(1)  # [M]
        
        # Element status: 1.0 = valid, 0.0 = eroded
        erosion_elem = 1.0 - pred_eroded
        
        # Convert to node level
        erosion_nodes = element_erosion_to_node(
            pred_eroded.bool(), elements, cached.shape[0]
        )
        
        return logits, erosion_elem, erosion_nodes
    
    # =========================================================================
    # FORWARD — autoregressive training with erosion coupling
    # =========================================================================
    
    def forward(self, data_list, dt=1.0, teacher_forcing_ratio=0.0):
        """
        Autoregressive rollout with bidirectional erosion-displacement coupling.
        
        Args:
            data_list: sequence of PyG Data objects
            dt: timestep size
            teacher_forcing_ratio: probability of using GT displacement (not erosion)
        
        Returns:
            predictions: list of [N, D] displacement predictions
            erosion_logits_list: list of [M, 1] erosion logits per step
        """
        predictions = []
        erosion_logits_list = []
        
        F_prev = None
        erosion_elem = None      # element-level: 1.0=valid, 0.0=eroded
        erosion_nodes = None     # node-level: 1.0=eroded, 0.0=valid
        
        for i, data in enumerate(data_list):
            x = data.x
            edge_index = data.edge_index
            elements = data.elements
            
            if hasattr(data, 'mesh_id'):
                edge_index.mesh_id = data.mesh_id
            
            static_feats = x[:, :self.num_static_feats]
            num_nodes = x.shape[0]
            num_elements = elements.shape[0]
            
            # ---- Initialize erosion state ----
            if i == 0:
                # Rebuild edge→element cache if mesh changed
                if hasattr(data, 'elements') and data.elements is not None:
                    cache = self.derivative_solver.edge_erosion_cache
                    needs_rebuild = (not cache.initialized or 
                                    getattr(cache, '_num_elements', None) != num_elements)
                    if needs_rebuild:
                        cache.clear()
                        cache.initialize(edge_index, elements)
                        cache._num_elements = num_elements
                
                # First step: use GT erosion if available
                if hasattr(data, 'x_element') and data.x_element is not None:
                    erosion_elem = data.x_element.flatten()  # 1=valid, 0=eroded
                    eroded_mask = (erosion_elem < 0.5)
                    erosion_nodes = element_erosion_to_node(
                        eroded_mask, elements, num_nodes
                    )
                else:
                    erosion_elem = torch.ones(num_elements, device=x.device)
                    erosion_nodes = torch.zeros(num_nodes, 1, device=x.device)
            
            # ---- Dynamic state (scheduled sampling for displacement) ----
            if i == 0:
                current_dynamic = x[:, self.num_static_feats:
                                     self.num_static_feats + self.num_dynamic_feats]
            else:
                if self.training and teacher_forcing_ratio > 0:
                    if torch.rand(1).item() < teacher_forcing_ratio:
                        current_dynamic = x[:, self.num_static_feats:
                                             self.num_static_feats + self.num_dynamic_feats]
                    else:
                        current_dynamic = F_prev.detach()
                else:
                    current_dynamic = F_prev.detach()
            
            # ---- Displacement step (erosion-aware) ----
            F_next = self.step(
                static_feats=static_feats,
                dynamic_state=current_dynamic,
                edge_index=edge_index,
                dt=dt,
                erosion_node_mask=erosion_nodes,
                erosion_elem=erosion_elem,
            )
            
            predictions.append(F_next)
            F_prev = F_next
            
            # ---- Erosion prediction ----
            logits, new_erosion_elem, new_erosion_nodes = self.predict_erosion(
                elements, erosion_nodes
            )
            erosion_logits_list.append(logits)
            
            # Irreversibility: once eroded, always eroded
            # erosion_elem: 1=valid, 0=eroded. min() preserves erasure.
            erosion_elem = torch.min(erosion_elem, new_erosion_elem)
            
            # Update node mask (OR: if any element says eroded, node is eroded)
            erosion_nodes = torch.max(erosion_nodes, new_erosion_nodes)
        
        return predictions, erosion_logits_list
    
    # =========================================================================
    # ROLLOUT — inference mode
    # =========================================================================
    
    @torch.no_grad()
    def rollout(self, simulation, num_steps, device=None):
        """
        Full autoregressive rollout for evaluation.
        
        Returns:
            states: list of [N, D] numpy arrays (displacement per step)
            erosion_preds: list of [M] numpy bool arrays (element erosion per step)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Move data to device
        for data in simulation:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            if hasattr(data, 'pos') and data.pos is not None:
                data.pos = data.pos.to(device)
            if hasattr(data, 'elements'):
                data.elements = data.elements.to(device)
            if hasattr(data, 'x_element') and data.x_element is not None:
                data.x_element = data.x_element.to(device)
        
        # Initialize MLS
        self.derivative_solver.initialize_weights(simulation[0])
        
        sf = self.num_static_feats
        df = self.num_dynamic_feats
        static = simulation[0].x[:, :sf]
        edge_index = simulation[0].edge_index
        elements = simulation[0].elements
        num_nodes = simulation[0].x.shape[0]
        num_elements = elements.shape[0]
        
        if hasattr(simulation[0], 'mesh_id'):
            edge_index.mesh_id = simulation[0].mesh_id
        
        # Initial state
        current = simulation[0].x[:, sf:sf + df].clone()
        
        # Initial erosion
        if hasattr(simulation[0], 'x_element') and simulation[0].x_element is not None:
            erosion_elem = simulation[0].x_element.flatten()
            eroded_mask = (erosion_elem < 0.5)
            erosion_nodes = element_erosion_to_node(eroded_mask, elements, num_nodes)
        else:
            erosion_elem = torch.ones(num_elements, device=device)
            erosion_nodes = torch.zeros(num_nodes, 1, device=device)
        
        states = [current.cpu().numpy()]
        erosion_preds = [(erosion_elem < 0.5).cpu().numpy()]
        
        for step in range(num_steps):
            # Displacement step
            F_next = self.step(
                static_feats=static,
                dynamic_state=current,
                edge_index=edge_index,
                dt=1.0,
                erosion_node_mask=erosion_nodes,
                erosion_elem=erosion_elem,
            )
            
            # Erosion prediction
            _, new_erosion_elem, new_erosion_nodes = self.predict_erosion(
                elements, erosion_nodes
            )
            
            # Irreversibility
            erosion_elem = torch.min(erosion_elem, new_erosion_elem)
            erosion_nodes = torch.max(erosion_nodes, new_erosion_nodes)
            
            current = F_next
            states.append(current.cpu().numpy())
            erosion_preds.append((erosion_elem < 0.5).cpu().numpy())
        
        return states, erosion_preds