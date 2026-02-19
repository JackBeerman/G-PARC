"""
visualizations.hydrology
========================
NSE, CSI, mass balance, important-node masking, segmented & per-timestep
hydrology metrics for river/flood evaluators.
"""

import numpy as np

__all__ = [
    'nse', 'csi', 'mass_balance_error',
    'get_important_mask', 'compute_mean_domain_volume',
    'compute_overall_hydrology_metrics',
    'compute_per_timestep_hydrology', 'compute_segmented_metrics',
]


# ─── Core metrics ────────────────────────────────────────────────────────

def nse(pred, obs):
    """Nash-Sutcliffe Efficiency: 1 = perfect, <0 = worse than mean."""
    num = np.sum((pred - obs) ** 2)
    den = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - num / den if den > 0 else np.nan


def csi(pred, obs, threshold):
    """Critical Success Index (Threat Score) for threshold exceedance."""
    hits = np.sum((pred > threshold) & (obs > threshold))
    misses = np.sum((pred <= threshold) & (obs > threshold))
    false_alarms = np.sum((pred > threshold) & (obs <= threshold))
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom > 0 else np.nan


def mass_balance_error(vol_pred, vol_prev, inflow, dt):
    """Squared mass balance error: (V_pred - V_prev - dt*inflow)²."""
    return (vol_pred - vol_prev - dt * inflow) ** 2


# ─── Important node mask ─────────────────────────────────────────────────

def get_important_mask(simulation, num_static_feats, pred_idx, threshold, extrema):
    """
    Identify nodes where the variable exceeds *threshold* at ANY timestep.
    Returns [N] boolean mask (True = "important").
    """
    n = simulation[0].x.size(0)
    mask = np.zeros(n, dtype=bool)

    y_min = extrema['y_min'][pred_idx].item() if extrema else 0.0
    y_max = extrema['y_max'][pred_idx].item() if extrema else 1.0

    for g in simulation:
        norm_vals = g.x[:, num_static_feats + pred_idx].cpu().numpy()
        phys_vals = norm_vals * (y_max - y_min) + y_min
        mask |= (phys_vals > threshold)

        if hasattr(g, 'y') and g.y is not None and g.y.shape[1] > pred_idx:
            norm_y = g.y[:, pred_idx].cpu().numpy()
            phys_y = norm_y * (y_max - y_min) + y_min
            mask |= (phys_y > threshold)

    return mask


def compute_mean_domain_volume(simulations, extrema):
    """Mean total domain volume (m³) across all sims & timesteps."""
    if extrema is None:
        return None
    volumes = []
    y_min_vol = extrema['y_min'][1].item()
    y_max_vol = extrema['y_max'][1].item()
    for simulation in simulations:
        for g in simulation:
            if hasattr(g, 'y') and g.y is not None and g.y.shape[1] > 1:
                norm_vol = g.y[:, 1].cpu().numpy()
                phys_vol = norm_vol * (y_max_vol - y_min_vol) + y_min_vol
                volumes.append(phys_vol.sum())
    return float(np.mean(volumes)) if volumes else None


# ─── Overall hydrology ───────────────────────────────────────────────────

def compute_overall_hydrology_metrics(preds_phys, targs_phys, depth_threshold=0.3):
    """RMSE, NSE, CSI on depth across all timesteps/nodes."""
    dp = np.concatenate([p[:, 0] for p in preds_phys])
    dg = np.concatenate([t[:, 0] for t in targs_phys])

    hits = np.sum((dp > depth_threshold) & (dg > depth_threshold))
    misses = np.sum((dp <= depth_threshold) & (dg > depth_threshold))
    fa = np.sum((dp > depth_threshold) & (dg <= depth_threshold))
    csi_denom = hits + misses + fa

    return {
        'Depth_RMSE': float(np.sqrt(np.mean((dp - dg) ** 2))),
        'Depth_NSE': float(nse(dp, dg)),
        'Depth_CSI': float(hits / csi_denom) if csi_denom > 0 else np.nan,
    }


# ─── Per-timestep metrics ────────────────────────────────────────────────

def compute_per_timestep_hydrology(preds_phys, targs_phys, important_mask=None,
                                    depth_threshold=0.3, inflow_series=None,
                                    dt=1.0, mean_vol=None):
    """
    Per-timestep depth RMSE, NSE, CSI, important/non-important split, mass balance.
    Returns list of T dicts.
    """
    T = len(preds_phys)
    n_nodes = preds_phys[0].shape[0]
    if important_mask is None:
        important_mask = np.ones(n_nodes, dtype=bool)
    imp = important_mask
    non_imp = ~important_mask

    per_t = []
    for t in range(T):
        p, g = preds_phys[t], targs_phys[t]
        dp_all, dg_all = p[:, 0], g[:, 0]

        depth_rmse = float(np.sqrt(np.mean((dp_all - dg_all) ** 2)))

        denom_nse = np.sum((dg_all - np.mean(dg_all)) ** 2)
        depth_nse = float(1.0 - np.sum((dp_all - dg_all) ** 2) / denom_nse) if denom_nse > 0 else np.nan

        hits = np.sum((dp_all > depth_threshold) & (dg_all > depth_threshold))
        misses = np.sum((dp_all <= depth_threshold) & (dg_all > depth_threshold))
        fa = np.sum((dp_all > depth_threshold) & (dg_all <= depth_threshold))
        csi_denom = hits + misses + fa
        depth_csi = float(hits / csi_denom) if csi_denom > 0 else np.nan

        rmse_imp = float(np.sqrt(np.mean((p[imp, 0] - g[imp, 0]) ** 2))) if imp.sum() > 0 else np.nan
        rmse_non = float(np.sqrt(np.mean((p[non_imp, 0] - g[non_imp, 0]) ** 2))) if non_imp.sum() > 0 else np.nan

        mb_rmse_m3, mb_pct = np.nan, np.nan
        if p.shape[1] > 1:
            vol_pred = p[:, 1].sum()
            vol_prev = preds_phys[t - 1][:, 1].sum() if t > 0 else targs_phys[0][:, 1].sum()
            inflow_t = inflow_series[t] if (inflow_series is not None and t < len(inflow_series)) else 0.0
            mb_sq = mass_balance_error(vol_pred, vol_prev, inflow_t, dt)
            mb_rmse_m3 = float(np.sqrt(mb_sq))
            if mean_vol is not None and mean_vol > 0:
                mb_pct = float((mb_rmse_m3 / mean_vol) * 100)

        per_t.append({
            'timestep': t,
            'depth_rmse': depth_rmse,
            'depth_nse': depth_nse,
            'depth_csi': depth_csi,
            'depth_rmse_important': rmse_imp,
            'depth_rmse_non_important': rmse_non,
            'mass_balance_rmse_m3': mb_rmse_m3,
            'mass_balance_pct': mb_pct,
        })
    return per_t


# ─── Segmented metrics ───────────────────────────────────────────────────

def compute_segmented_metrics(preds_phys, targs_phys, segments,
                               important_mask=None, depth_threshold=0.3,
                               inflow_series=None, dt=1.0, mean_vol=None):
    """
    Per-segment, per-group (Important / Non-Important / All) metrics.
    Returns nested dict: {group: {Segment_i: {RMSE, NSE, CSI, MassBal, ...}}}.
    """
    T = len(preds_phys)
    n_nodes = preds_phys[0].shape[0]
    if important_mask is None:
        important_mask = np.ones(n_nodes, dtype=bool)

    groups = {
        'Important': important_mask,
        'Non_Important': ~important_mask,
        'All_Nodes': np.ones(n_nodes, dtype=bool),
    }

    result = {}
    for grp_name, grp_mask in groups.items():
        seg_metrics = {}
        for seg_idx, (s, e) in enumerate(segments, start=1):
            key = f"Segment_{seg_idx}"
            e_actual = min(e, T)
            if s >= T or e_actual <= s:
                seg_metrics[key] = 'No Data'
                continue

            depth_pred_list, depth_targ_list = [], []
            hits = misses = false_alarms = 0
            mass_errors = []

            for t in range(s, e_actual):
                p, g = preds_phys[t], targs_phys[t]
                dp, dg = p[grp_mask, 0], g[grp_mask, 0]
                depth_pred_list.append(dp)
                depth_targ_list.append(dg)

                hits += np.sum((dp > depth_threshold) & (dg > depth_threshold))
                misses += np.sum((dp <= depth_threshold) & (dg > depth_threshold))
                false_alarms += np.sum((dp > depth_threshold) & (dg <= depth_threshold))

                if grp_name == 'All_Nodes' and p.shape[1] > 1:
                    vol_pred = p[:, 1].sum()
                    vol_prev = (preds_phys[t - 1][:, 1].sum() if t > s
                                else targs_phys[max(0, t - 1)][:, 1].sum())
                    inflow_t = (inflow_series[t]
                                if inflow_series is not None and t < len(inflow_series) else 0.0)
                    mass_errors.append(mass_balance_error(vol_pred, vol_prev, inflow_t, dt))

            dp_cat = np.concatenate(depth_pred_list)
            dg_cat = np.concatenate(depth_targ_list)
            csi_denom = hits + misses + false_alarms

            mb_rmse = float(np.sqrt(np.mean(mass_errors))) if mass_errors else np.nan
            mb_pct = (float((mb_rmse / mean_vol) * 100)
                      if mass_errors and mean_vol is not None and mean_vol > 0
                      else np.nan)

            seg_metrics[key] = {
                'Time_Range': (s, e_actual - 1),
                'RMSE': float(np.sqrt(np.mean((dp_cat - dg_cat) ** 2))),
                'NSE': float(nse(dp_cat, dg_cat)),
                'CSI': float(hits / csi_denom) if csi_denom > 0 else np.nan,
                'MassBalance_RMSE_m3': mb_rmse,
                'MassBalance_pct': mb_pct,
            }
        result[grp_name] = seg_metrics

    return result
