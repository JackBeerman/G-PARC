"""
visualizations.metrics
======================
RRMSE, per-step RMSE, R² — shared across all evaluators.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

__all__ = [
    'compute_rrmse', 'compute_rrmse_per_variable',
    'compute_rrmse_scalar', 'compute_rrmse_scalar_per_variable',
    'compute_per_step_metrics', 'compute_timestep_rmse',
]


def compute_rrmse(predictions, references, valid_masks=None):
    """
    RRMSE = sqrt( (1/N) * sum_i [ MSE_i / ||ref_i||_inf^2 ] )

    Works for both [N, D] arrays (per-timestep) and flat arrays.
    """
    if len(predictions) == 0:
        return float('inf')

    n_samples, ratio_sum = 0, 0.0
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if valid_masks is not None and i < len(valid_masks) and valid_masks[i] is not None:
            pred, ref = pred[valid_masks[i]], ref[valid_masks[i]]
        if len(pred) == 0:
            continue
        ref_inf = np.max(np.abs(ref))
        if ref_inf < 1e-12:
            continue
        mse = np.mean((pred - ref) ** 2)
        ratio_sum += mse / (ref_inf ** 2)
        n_samples += 1

    return float(np.sqrt(ratio_sum / n_samples)) if n_samples > 0 else float('inf')


def compute_rrmse_per_variable(predictions, references, var_names=None, valid_masks=None):
    """RRMSE computed independently for each variable column."""
    if len(predictions) == 0:
        return {}

    n_vars = predictions[0].shape[1] if predictions[0].ndim > 1 else 1
    if var_names is None:
        var_names = [f'var_{i}' for i in range(n_vars)]

    rrmse = {}
    for c in range(min(n_vars, len(var_names))):
        n_samples, ratio_sum = 0, 0.0
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if valid_masks is not None and i < len(valid_masks) and valid_masks[i] is not None:
                p, r = pred[valid_masks[i], c], ref[valid_masks[i], c]
            else:
                p, r = pred[:, c], ref[:, c]
            if len(p) == 0:
                continue
            ref_inf = np.max(np.abs(r))
            if ref_inf < 1e-12:
                continue
            mse = np.mean((p - r) ** 2)
            ratio_sum += mse / (ref_inf ** 2)
            n_samples += 1
        rrmse[var_names[c]] = float(np.sqrt(ratio_sum / n_samples)) if n_samples > 0 else float('inf')

    return rrmse


def compute_rrmse_scalar(predictions, references, valid_masks=None):
    """
    PLAID scalar RRMSE — per-node normalization.

    RRMSEs = sqrt( (1/n*) * sum_i [ |s_ref^i - s_pred^i|^2 / |s_ref^i|^2 ] )

    Each sample i is a single node from a single timestep.
    The denominator is that individual node's reference value.

    Args:
        predictions: list of [N, D] arrays (one per timestep)
        references:  list of [N, D] arrays (one per timestep)
        valid_masks: optional list of boolean masks per timestep
    """
    if len(predictions) == 0:
        return float('inf')

    n_samples, ratio_sum = 0, 0.0
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if valid_masks is not None and i < len(valid_masks) and valid_masks[i] is not None:
            pred, ref = pred[valid_masks[i]], ref[valid_masks[i]]
        if len(pred) == 0:
            continue
        # Flatten all components together
        p_flat = pred.flatten()
        r_flat = ref.flatten()
        # Skip entries where reference is near zero
        nonzero = np.abs(r_flat) > 1e-12
        if nonzero.sum() == 0:
            continue
        ratios = (p_flat[nonzero] - r_flat[nonzero]) ** 2 / r_flat[nonzero] ** 2
        ratio_sum += ratios.sum()
        n_samples += int(nonzero.sum())

    return float(np.sqrt(ratio_sum / n_samples)) if n_samples > 0 else float('inf')


def compute_rrmse_scalar_per_variable(predictions, references, var_names=None, valid_masks=None):
    """
    PLAID scalar RRMSE per variable — per-node normalization for each component.

    Each node's error is normalized by that same node's reference value,
    computed independently per variable column.

    Args:
        predictions: list of [N, D] arrays (one per timestep)
        references:  list of [N, D] arrays (one per timestep)
        var_names:   list of variable names
        valid_masks: optional list of boolean masks per timestep
    """
    if len(predictions) == 0:
        return {}

    n_vars = predictions[0].shape[1] if predictions[0].ndim > 1 else 1
    if var_names is None:
        var_names = [f'var_{i}' for i in range(n_vars)]

    rrmse = {}
    for c in range(min(n_vars, len(var_names))):
        n_samples, ratio_sum = 0, 0.0
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if valid_masks is not None and i < len(valid_masks) and valid_masks[i] is not None:
                p, r = pred[valid_masks[i], c], ref[valid_masks[i], c]
            else:
                p, r = pred[:, c], ref[:, c]
            if len(p) == 0:
                continue
            nonzero = np.abs(r) > 1e-12
            if nonzero.sum() == 0:
                continue
            ratios = (p[nonzero] - r[nonzero]) ** 2 / r[nonzero] ** 2
            ratio_sum += ratios.sum()
            n_samples += int(nonzero.sum())
        rrmse[var_names[c]] = float(np.sqrt(ratio_sum / n_samples)) if n_samples > 0 else float('inf')

    return rrmse


def compute_per_step_metrics(seq_pred, seq_targ):
    """Per-timestep RMSE and R²."""
    steps = min(len(seq_pred), len(seq_targ))
    rmse_per_step, r2_per_step = [], []
    for t in range(steps):
        p, tg = seq_pred[t].flatten(), seq_targ[t].flatten()
        rmse_per_step.append(float(np.sqrt(mean_squared_error(tg, p))))
        r2_per_step.append(float(r2_score(tg, p)))
    return rmse_per_step, r2_per_step


def compute_timestep_rmse(preds, targs, scale=1.0):
    """Per-timestep RMSE, optionally scaled (e.g. for denormalization)."""
    T = min(len(preds), len(targs))
    rmse = np.zeros(T)
    for t in range(T):
        diff = (preds[t] - targs[t]) * scale
        rmse[t] = np.sqrt(np.mean(diff ** 2))
    return rmse