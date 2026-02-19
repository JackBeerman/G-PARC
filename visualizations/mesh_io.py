"""
visualizations.mesh_io
======================
HEC-RAS mesh loading (h5py), erosion masking, polygon precomputation.
"""

import numpy as np
from pathlib import Path

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

__all__ = [
    'HAS_H5PY',
    'load_hec_ras_mesh', 'get_clean_polygons', 'load_mesh_for_sim',
    'EROSION_THRESHOLD', 'get_erosion_mask', 'get_valid_node_mask',
    'precompute_element_polygons', 'get_node_positions',
]

EROSION_THRESHOLD = 0.5


# ─── HEC-RAS mesh (river) ────────────────────────────────────────────────

def load_hec_ras_mesh(hdf_path, sim_id):
    """Load HEC-RAS mesh (facepoints + cell indices) from .p0x.hdf file."""
    if not HAS_H5PY:
        raise ImportError("h5py required for HEC-RAS mesh loading")

    with h5py.File(hdf_path, "r") as f:
        if "iw" in sim_id.lower():
            facepts_path = "Geometry/2D Flow Areas/Flow Area/FacePoints Coordinate"
            cells_path = "Geometry/2D Flow Areas/Flow Area/Cells FacePoint Indexes"
        else:
            facepts_path = "Geometry/2D Flow Areas/Perimeter 1/FacePoints Coordinate"
            cells_path = "Geometry/2D Flow Areas/Perimeter 1/Cells FacePoint Indexes"
        facepts = f[facepts_path][:]
        cells = f[cells_path][:]

    return facepts, cells


def get_clean_polygons(facepts, cells):
    """Convert HEC-RAS facepts + padded cell indices to polygon vertex arrays."""
    polys = []
    for cell_ids in cells:
        valid_ids = cell_ids[cell_ids >= 0].astype(int)
        polys.append(facepts[valid_ids])
    return polys


def load_mesh_for_sim(sim_id, hec_ras_dir):
    """Load HEC-RAS mesh polygons for a simulation. Returns list of polygons or None."""
    if not HAS_H5PY:
        return None

    hec_ras_dir = Path(hec_ras_dir)
    if "iw" in sim_id.lower():
        hdf_path = hec_ras_dir / "Flood_GNN.p01.hdf"
    else:
        hdf_path = hec_ras_dir / "Muncie2D_SI.p02.hdf"

    if not hdf_path.exists():
        print(f"  ⚠️  HDF file not found: {hdf_path}")
        return None

    try:
        facepts, cells = load_hec_ras_mesh(hdf_path, sim_id)
        polys = get_clean_polygons(facepts, cells)
        print(f"  ✓ Loaded mesh: {len(polys)} cells from {hdf_path.name}")
        return polys
    except Exception as e:
        print(f"  ⚠️  Failed to load mesh from {hdf_path}: {e}")
        return None


# ─── Erosion (elastoplastic) ─────────────────────────────────────────────

def get_erosion_mask(data, num_elements):
    """Get boolean mask of eroded elements from data.x_element."""
    if hasattr(data, 'x_element') and data.x_element is not None:
        erosion_status = data.x_element.cpu().numpy().flatten()
        return erosion_status < EROSION_THRESHOLD
    return np.zeros(num_elements, dtype=bool)


def get_valid_node_mask(elements, eroded_mask):
    """Get boolean mask of nodes that belong to at least one non-eroded element."""
    valid_elements = elements[~eroded_mask]
    if len(valid_elements) == 0:
        return np.zeros(elements.max() + 1, dtype=bool)
    valid_nodes = np.unique(valid_elements.flatten())
    valid_node_mask = np.zeros(elements.max() + 1, dtype=bool)
    valid_node_mask[valid_nodes] = True
    return valid_node_mask


# ─── Generic helpers ──────────────────────────────────────────────────────

def precompute_element_polygons(pos, elements):
    """Convert node positions + element connectivity to polygon vertex coords."""
    return pos[elements]


def get_node_positions(simulation):
    """Extract node positions from first timestep."""
    data0 = simulation[0]
    if hasattr(data0, 'pos') and data0.pos is not None:
        return data0.pos.cpu().numpy()
    return data0.x[:, :2].cpu().numpy()
