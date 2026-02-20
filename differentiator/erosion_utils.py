"""
Erosion-Aware Utilities for G-PARCv3
=====================================
Computes the mapping from edges to elements, and uses it to derive
per-edge erosion weights that mask out contributions through eroded material.

The mesh is static, so the edge→element mapping is precomputed once per mesh.
Only the erosion weights change per timestep.
"""

import torch
import numpy as np


def build_edge_to_element_map(edge_index, elements):
    """
    Build a mapping from each edge to the elements that contain it.
    
    For triangular meshes, each interior edge is shared by exactly 2 elements,
    and boundary edges belong to 1 element.
    
    Args:
        edge_index: [2, E] long tensor — directed edges
        elements: [M, 3] long tensor — triangle connectivity
        
    Returns:
        edge_elem_indices: [E] list of lists — for each directed edge,
                           the element indices that contain both endpoints.
                           Stored as a tensor-friendly format.
        edge_elem_ptr: CSR-style pointer for variable-length lists
    """
    device = edge_index.device
    E = edge_index.shape[1]
    M = elements.shape[0]
    
    # Build node→element adjacency (which elements touch each node)
    # For each element m with nodes (a, b, c), record m for nodes a, b, c
    node_to_elems = {}
    elements_np = elements.cpu().numpy()
    for m in range(M):
        for node in elements_np[m]:
            if node not in node_to_elems:
                node_to_elems[node] = set()
            node_to_elems[node].add(m)
    
    # For each directed edge (i→j), find elements containing BOTH i and j
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    
    indices = []  # flat list of element indices
    counts = []   # number of elements per edge
    
    for e in range(E):
        i, j = int(row[e]), int(col[e])
        elems_i = node_to_elems.get(i, set())
        elems_j = node_to_elems.get(j, set())
        shared = elems_i & elems_j
        indices.extend(shared)
        counts.append(len(shared))
    
    # Convert to tensors
    edge_elem_indices = torch.tensor(indices, dtype=torch.long, device=device)
    edge_elem_counts = torch.tensor(counts, dtype=torch.long, device=device)
    
    # CSR-style pointer
    edge_elem_ptr = torch.zeros(E + 1, dtype=torch.long, device=device)
    edge_elem_ptr[1:] = torch.cumsum(edge_elem_counts, 0)
    
    return edge_elem_indices, edge_elem_ptr


def compute_edge_erosion_weights(edge_elem_indices, edge_elem_ptr, 
                                  erosion_elem, num_edges):
    """
    Compute per-edge weight based on erosion status of adjacent elements.
    
    For each edge, weight = mean erosion_status of elements sharing that edge.
    erosion_elem uses convention: 1.0 = valid, 0.0 = eroded.
    
    Edges with no elements (shouldn't happen in valid mesh) get weight 1.0.
    
    Args:
        edge_elem_indices: [sum(counts)] flat element indices
        edge_elem_ptr: [E+1] CSR pointer
        erosion_elem: [M] float — 1.0 = valid, 0.0 = eroded
        num_edges: E
        
    Returns:
        edge_weights: [E] float — 1.0 for fully valid, 0.0 for fully eroded,
                      0.5 for boundary (one valid, one eroded element)
    """
    device = erosion_elem.device
    edge_weights = torch.ones(num_edges, device=device)
    
    for e in range(num_edges):
        start = edge_elem_ptr[e].item()
        end = edge_elem_ptr[e + 1].item()
        if end > start:
            elem_ids = edge_elem_indices[start:end]
            edge_weights[e] = erosion_elem[elem_ids].mean()
    
    return edge_weights


def compute_edge_erosion_weights_fast(edge_elem_indices, edge_elem_ptr,
                                       erosion_elem, num_edges):
    """
    Vectorized version using segment_reduce.
    Falls back to loop if segment_reduce unavailable.
    
    Args:
        edge_elem_indices: [sum(counts)] flat element indices  
        edge_elem_ptr: [E+1] CSR pointer
        erosion_elem: [M] float — 1.0 = valid, 0.0 = eroded
        num_edges: E
        
    Returns:
        edge_weights: [E] float
    """
    device = erosion_elem.device
    
    if len(edge_elem_indices) == 0:
        return torch.ones(num_edges, device=device)
    
    # Gather erosion status for each (edge, element) pair
    elem_status = erosion_elem[edge_elem_indices]  # [sum(counts)]
    
    # Create edge assignment: which edge each entry belongs to
    counts = edge_elem_ptr[1:] - edge_elem_ptr[:-1]  # [E]
    edge_ids = torch.repeat_interleave(
        torch.arange(num_edges, device=device), counts
    )  # [sum(counts)]
    
    # Scatter mean: sum per edge, then divide by count
    edge_sum = torch.zeros(num_edges, device=device)
    edge_sum.scatter_add_(0, edge_ids, elem_status)
    
    # Avoid division by zero for edges with no elements
    counts_float = counts.float().clamp(min=1.0)
    edge_weights = edge_sum / counts_float
    
    # Edges with 0 elements get weight 1.0 (shouldn't happen)
    edge_weights[counts == 0] = 1.0
    
    return edge_weights


class EdgeErosionCache:
    """
    Manages the edge-to-element mapping (static per mesh) and 
    computes per-timestep edge erosion weights.
    
    Usage:
        cache = EdgeErosionCache()
        cache.initialize(edge_index, elements)  # once per mesh
        weights = cache.get_weights(erosion_elem)  # each timestep
    """
    
    def __init__(self):
        self._edge_elem_indices = None
        self._edge_elem_ptr = None
        self._num_edges = None
        self._initialized = False
    
    def initialize(self, edge_index, elements):
        """Build edge→element mapping. Call once per mesh."""
        self._edge_elem_indices, self._edge_elem_ptr = \
            build_edge_to_element_map(edge_index, elements)
        self._num_edges = edge_index.shape[1]
        self._initialized = True
    
    @property
    def initialized(self):
        return self._initialized
    
    def get_weights(self, erosion_elem):
        """
        Compute edge erosion weights for current erosion state.
        
        Args:
            erosion_elem: [M] float — 1.0 = valid, 0.0 = eroded
            
        Returns:
            [E] float edge weights
        """
        if not self._initialized:
            raise RuntimeError("EdgeErosionCache not initialized. Call initialize() first.")
        
        return compute_edge_erosion_weights_fast(
            self._edge_elem_indices,
            self._edge_elem_ptr,
            erosion_elem,
            self._num_edges
        )
    
    def clear(self):
        self._edge_elem_indices = None
        self._edge_elem_ptr = None
        self._num_edges = None
        self._initialized = False