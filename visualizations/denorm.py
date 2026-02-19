"""
visualizations.denorm
=====================
Denormalization helpers:
  - River: extrema .pth (y_min, y_max tensors)
  - Elastoplastic: global_max normalization_stats.json
  - Shock tube: per-variable min/max from normalization_metadata.json
  - Generic: min/max denorm for scalar values
"""

import json
import numpy as np
from pathlib import Path

__all__ = [
    'load_denorm_extrema', 'denormalize_array', 'denormalize_all',
    'load_normalization_stats', 'denormalize_global_max',
    'denorm_minmax', 'denorm_all_from_params',
    'load_normalization_metadata',
]


# ─── Generic min/max denorm ──────────────────────────────────────────────

def denorm_minmax(normalized, params):
    """
    Denormalize using min/max: physical = norm * (max - min) + min.
    params: dict with 'min' and 'max' keys.
    """
    if params is None:
        return normalized
    return normalized * (params['max'] - params['min']) + params['min']


def denorm_all_from_params(data_np, var_names, denorm_params):
    """
    Denormalize all variables in [N, D] array using per-variable denorm_params dict.
    denorm_params: {var_name: {'min': ..., 'max': ...}} or empty.
    """
    out = np.copy(data_np)
    for vi in range(min(out.shape[1], len(var_names))):
        vn = var_names[vi]
        if vn in denorm_params:
            out[:, vi] = denorm_minmax(out[:, vi], denorm_params[vn])
    return out


# ─── River-style: extrema .pth ───────────────────────────────────────────

def load_denorm_extrema(extrema_path):
    """
    Load global y extrema dict from .pth file.
    Expected keys: 'y_min', 'y_max' — each a 1D tensor [n_vars].
    """
    extrema_path = Path(extrema_path)
    if not extrema_path.exists():
        print(f"  ⚠️  Extrema file not found: {extrema_path}")
        return None
    import torch
    extrema = torch.load(extrema_path, weights_only=False)
    print(f"  ✓ Loaded extrema: y_min={extrema['y_min'].tolist()}, y_max={extrema['y_max'].tolist()}")
    return extrema


def denormalize_array(normalized, var_idx, extrema):
    """Denormalize a single variable column using extrema dict."""
    if extrema is None:
        return normalized
    y_min = extrema['y_min'][var_idx].item()
    y_max = extrema['y_max'][var_idx].item()
    return normalized * (y_max - y_min) + y_min


def denormalize_all(normalized, extrema):
    """Denormalize all variable columns in an [N, D] array using extrema."""
    if extrema is None:
        return normalized
    physical = np.zeros_like(normalized)
    for v in range(normalized.shape[1]):
        physical[:, v] = denormalize_array(normalized[:, v], v, extrema)
    return physical


# ─── Elastoplastic-style: normalization_stats.json ────────────────────────

def load_normalization_stats(*search_paths):
    """Search for normalization_stats.json. Returns parsed dict or None."""
    for p in search_paths:
        p = Path(p)
        candidates = [p, p / "normalization_stats.json", p.parent / "normalization_stats.json"]
        for c in candidates:
            if c.is_file() and c.suffix == '.json':
                with open(c, 'r') as f:
                    stats = json.load(f)
                print(f"  ✓ Loaded normalization stats from: {c}")
                return stats
    return None


def denormalize_global_max(normalized_data, norm_stats):
    """Denormalize using global_max method: physical = norm * max_displacement."""
    if norm_stats is None:
        return normalized_data
    method = norm_stats.get('normalization_method', 'none')
    if method == 'global_max':
        max_disp = norm_stats.get('displacement', {}).get('max_displacement', 1.0)
        return normalized_data * max_disp
    return normalized_data


# ─── Shock tube / general: normalization_metadata.json ────────────────────

def load_normalization_metadata(*search_paths):
    """
    Search for normalization_metadata.json. Returns parsed dict with
    'normalization_params', 'global_param_normalization', etc.
    """
    for p in search_paths:
        p = Path(p)
        candidates = [
            p,
            p / "normalization_metadata.json",
            p.parent / "normalization_metadata.json",
        ]
        for c in candidates:
            if c.is_file() and c.suffix == '.json':
                with open(c, 'r') as f:
                    metadata = json.load(f)
                print(f"  ✓ Loaded normalization metadata from: {c}")
                return metadata
    return None
