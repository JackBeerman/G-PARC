"""
visualizations.dashboard
========================
Multi-panel performance dashboards for river and elastoplastic evaluators.
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

__all__ = [
    'plot_river_dashboard', 'plot_elasto_dashboard',
    'plot_shocktube_dashboard', 'plot_global_parameter_analysis',
    'create_delta_t_performance_table',
]


def _get_overall(m):
    """Resolve metric dict key: accept 'overall', 'overall_physical', or top-level."""
    if 'overall' in m:
        return m['overall']
    if 'overall_physical' in m:
        return m['overall_physical']
    # Fallback: keys at top level (river evaluator style)
    return m


def plot_river_dashboard(sim_metrics_list, output_path, model_name='Model',
                          eval_mode='rollout'):
    """
    River evaluation dashboard: R²/RMSE distributions, mesh-colored scatter,
    per-sim R² bar chart, summary stats with hydrology.
    
    Args:
        sim_metrics_list: list of dicts with keys 'overall' (rmse, r2),
                          'metadata' (mesh_id, simulation_idx), 'hydrology' (optional)
    """
    if not sim_metrics_list:
        return
    r2s = [_get_overall(m)['r2'] for m in sim_metrics_list]
    rmses = [_get_overall(m)['rmse'] for m in sim_metrics_list]
    mids = [m['metadata'].get('mesh_id', 0) for m in sim_metrics_list]
    ml = eval_mode.capitalize()

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # R² histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(r2s, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(r2s), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(r2s):.4f}')
    ax1.set_xlabel('R²'); ax1.set_title('R² Distribution')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # RMSE histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(rmses, bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(rmses), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(rmses):.4e}')
    ax2.set_xlabel('RMSE'); ax2.set_title('RMSE Distribution')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # Stats text
    ax3 = fig.add_subplot(gs[0, 2]); ax3.axis('off')
    nses = [m.get('hydrology', {}).get('Depth_NSE', np.nan) for m in sim_metrics_list]
    csis = [m.get('hydrology', {}).get('Depth_CSI', np.nan) for m in sim_metrics_list]
    vn = [v for v in nses if not np.isnan(v)]
    vc = [v for v in csis if not np.isnan(v)]
    txt = (f"{model_name} {ml.upper()}\n{'=' * 28}\n\n"
           f"R²:  Mean={np.mean(r2s):.4f}  Med={np.median(r2s):.4f}\n"
           f"     Min={np.min(r2s):.4f}  Max={np.max(r2s):.4f}\n\n"
           f"RMSE: Mean={np.mean(rmses):.4e}\n")
    if vn: txt += f"\nDepth NSE: {np.mean(vn):.4f}"
    if vc: txt += f"\nDepth CSI: {np.mean(vc):.4f}"
    txt += f"\n\nSimulations: {len(sim_metrics_list)}\nMeshes: {len(set(mids))}"
    ax3.text(0.05, 0.5, txt, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # R² vs RMSE by mesh
    ax4 = fig.add_subplot(gs[1, 0])
    colors = ['steelblue', 'coral', 'green', 'purple']
    mesh_labels = {0: 'White River', 1: 'Iowa River'}
    for mi, mid in enumerate(sorted(set(mids))):
        mask = [i for i, m in enumerate(mids) if m == mid]
        ax4.scatter([r2s[i] for i in mask], [rmses[i] for i in mask],
                    c=colors[mi % len(colors)], s=80, alpha=0.7, edgecolors='black',
                    label=mesh_labels.get(mid, f'Mesh {mid}'))
    ax4.set_xlabel('R²'); ax4.set_ylabel('RMSE')
    ax4.set_title('R² vs RMSE by Mesh')
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    # Per-sim bar chart
    ax5 = fig.add_subplot(gs[1, 1:])
    sim_labels = [f"S{m['metadata']['simulation_idx']}" for m in sim_metrics_list]
    bc = ['green' if r > 0.8 else 'orange' if r > 0.5 else 'red' for r in r2s]
    ax5.bar(range(len(r2s)), r2s, color=bc, edgecolor='black', alpha=0.7)
    ax5.set_xticks(range(len(sim_labels)))
    ax5.set_xticklabels(sim_labels, rotation=45, fontsize=8)
    ax5.set_ylabel('R²'); ax5.set_title('R² by Simulation')
    ax5.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
    ax5.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (0.5)')
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3, axis='y')

    fig.suptitle(f'{model_name} River Performance ({ml})',
                 fontsize=14, fontweight='bold')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Dashboard saved: {output_path}")


def plot_elasto_dashboard(sim_metrics_list, output_path, model_name='Model',
                           norm_stats=None, eval_mode='rollout'):
    """
    Elastoplastic dashboard: R²/RMSE distributions, R²-vs-RMSE scatter
    colored by erosion, per-sim bars, best/worst table.
    
    Args:
        sim_metrics_list: list of dicts with 'overall' (rmse, r2),
                          'metadata' (simulation_idx, max_eroded)
    """
    if not sim_metrics_list:
        return

    r2s = [_get_overall(m)['r2'] for m in sim_metrics_list]
    rmses = [_get_overall(m)['rmse'] for m in sim_metrics_list]
    sim_idx = [m['metadata']['simulation_idx'] for m in sim_metrics_list]
    max_eroded = [m['metadata'].get('max_eroded', 0) for m in sim_metrics_list]
    ml = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # R² histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(r2s, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(r2s), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(r2s):.3f}')
    ax1.set_xlabel('R² Score'); ax1.set_ylabel('Frequency')
    ax1.set_title('R² Score Distribution', fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # RMSE histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(rmses, bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(rmses), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(rmses):.3e}')
    ax2.set_xlabel('RMSE'); ax2.set_ylabel('Frequency')
    ax2.set_title('RMSE Distribution', fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # Stats panel
    ax3 = fig.add_subplot(gs[0, 2]); ax3.axis('off')
    nm = norm_stats.get('normalization_method', 'unknown') if norm_stats else 'unknown'
    txt = (f"{model_name} {ml.upper()}\n{'=' * 30}\n"
           f"Normalization: {nm}\n\n"
           f"R² Score:\n  Mean:   {np.mean(r2s):.4f}\n  Median: {np.median(r2s):.4f}\n"
           f"  Min:    {np.min(r2s):.4f}\n  Max:    {np.max(r2s):.4f}\n\n"
           f"RMSE:\n  Mean:   {np.mean(rmses):.3e}\n  Median: {np.median(rmses):.3e}\n\n"
           f"Simulations: {len(sim_metrics_list)}")
    ax3.text(0.1, 0.5, txt, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # R² vs RMSE colored by erosion
    ax4 = fig.add_subplot(gs[1, :2])
    scatter = ax4.scatter(r2s, rmses, c=max_eroded, cmap='YlOrRd',
                          s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('R² Score'); ax4.set_ylabel('RMSE')
    ax4.set_title('R² vs RMSE (colored by max eroded elements)', fontweight='bold')
    ax4.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax4).set_label('Max Eroded Elements')

    # Per-sim R² bars
    ax5 = fig.add_subplot(gs[1, 2])
    bc = ['green' if r > 0.8 else 'orange' if r > 0.5 else 'red' for r in r2s]
    ax5.barh(range(len(sim_idx)), r2s, color=bc, edgecolor='black', alpha=0.7)
    ax5.set_xlabel('R² Score'); ax5.set_ylabel('Simulation Index')
    ax5.set_title('Performance by Simulation', fontweight='bold')
    ax5.axvline(0.8, color='green', linestyle='--', alpha=0.5)
    ax5.axvline(0.5, color='orange', linestyle='--', alpha=0.5)
    ax5.grid(alpha=0.3, axis='x')

    # Erosion vs R²
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.scatter(max_eroded, r2s, c='steelblue', s=80, alpha=0.7, edgecolors='black')
    ax6.set_xlabel('Max Eroded Elements'); ax6.set_ylabel('R² Score')
    ax6.set_title('R² vs Erosion Level', fontweight='bold'); ax6.grid(alpha=0.3)

    # RMSE over sims
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(range(len(rmses)), rmses, marker='o', linestyle='-',
             linewidth=1.5, markersize=6, color='coral', alpha=0.7)
    ax7.set_xlabel('Simulation Index'); ax7.set_ylabel('RMSE')
    ax7.set_title('RMSE by Simulation', fontweight='bold'); ax7.grid(alpha=0.3)
    ax7.axhline(np.mean(rmses), color='red', linestyle='--', alpha=0.5, label='Mean')
    ax7.legend(fontsize=9)

    # Best/worst
    ax8 = fig.add_subplot(gs[2, 2]); ax8.axis('off')
    si = np.argsort(r2s)
    best_3, worst_3 = si[-3:][::-1], si[:3]
    ct = f"BEST PERFORMERS\n{'=' * 25}\n"
    for i, idx in enumerate(best_3, 1):
        ct += f"{i}. Sim {sim_idx[idx]}: R²={r2s[idx]:.4f}\n"
    ct += f"\nWORST PERFORMERS\n{'=' * 25}\n"
    for i, idx in enumerate(worst_3, 1):
        ct += f"{i}. Sim {sim_idx[idx]}: R²={r2s[idx]:.4f}\n"
    ax8.text(0.1, 0.5, ct, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    fig.suptitle(f'{model_name} Performance Analysis ({ml})',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Dashboard saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# SHOCK TUBE
# ═══════════════════════════════════════════════════════════════════════════

def plot_shocktube_dashboard(sim_metrics_list, output_path,
                              var_names=None, model_name='Model',
                              eval_mode='rollout'):
    """
    Shock tube performance dashboard (3×3 GridSpec).

    Args:
        sim_metrics_list: list of dicts with keys:
            'overall_physical': {'rmse', 'r2'}
            'metadata': {'simulation_idx', 'delta_t', 'pressure', 'density', ...}
            'per_variable': {var_name: {'rmse', 'r2', 'mae'}}
    """
    if not sim_metrics_list:
        return

    if var_names is None:
        var_names = ['density', 'x_momentum', 'total_energy']

    r2_vals = [_get_overall(m)['r2'] for m in sim_metrics_list]
    rmse_vals = [_get_overall(m)['rmse'] for m in sim_metrics_list]
    sim_ids = [m['metadata']['simulation_idx'] for m in sim_metrics_list]
    delta_ts = [m['metadata'].get('delta_t', 0) for m in sim_metrics_list]
    ml = 'Snapshot' if eval_mode == 'snapshot' else 'Rollout'

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # (0,0) R² histogram
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(r2_vals, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(r2_vals), color='red', ls='--', lw=2,
               label=f'Mean: {np.mean(r2_vals):.3f}')
    ax.set_xlabel('R²'); ax.set_ylabel('Count')
    ax.set_title('R² Distribution', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (0,1) RMSE histogram
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(rmse_vals, bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(rmse_vals), color='red', ls='--', lw=2,
               label=f'Mean: {np.mean(rmse_vals):.3e}')
    ax.set_xlabel('RMSE'); ax.set_ylabel('Count')
    ax.set_title('RMSE Distribution', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (0,2) Summary text
    ax = fig.add_subplot(gs[0, 2]); ax.axis('off')
    txt = (
        f"{model_name} {ml.upper()}\n{'=' * 28}\n"
        f"Simulations: {len(sim_metrics_list)}\n\n"
        f"R²:\n  Mean:   {np.mean(r2_vals):.4f}\n  Median: {np.median(r2_vals):.4f}\n"
        f"  Min:    {np.min(r2_vals):.4f}\n  Max:    {np.max(r2_vals):.4f}\n\n"
        f"RMSE:\n  Mean:   {np.mean(rmse_vals):.3e}\n  Median: {np.median(rmse_vals):.3e}"
    )
    ax.text(0.05, 0.5, txt, fontsize=10, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # (1, :2) R² vs RMSE scatter colored by Δt
    ax = fig.add_subplot(gs[1, :2])
    sc = ax.scatter(r2_vals, rmse_vals, c=delta_ts, cmap='viridis',
                    s=80, alpha=0.7, edgecolors='k', lw=0.5)
    ax.set_xlabel('R²'); ax.set_ylabel('RMSE')
    ax.set_title('R² vs RMSE (colored by Δt)', fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax).set_label('Δt (physical)')

    # (1, 2) Per-simulation bar chart
    ax = fig.add_subplot(gs[1, 2])
    colors = ['green' if r > 0.8 else 'orange' if r > 0.5 else 'red' for r in r2_vals]
    ax.barh(range(len(sim_ids)), r2_vals, color=colors, edgecolor='k', alpha=0.7)
    ax.set_xlabel('R²'); ax.set_ylabel('Sim Index')
    ax.set_title('Performance by Sim', fontweight='bold')
    ax.axvline(0.8, color='green', ls='--', alpha=0.5)
    ax.axvline(0.5, color='orange', ls='--', alpha=0.5)
    ax.grid(alpha=0.3, axis='x')

    # (2, 0) Per-variable R² boxplot
    ax = fig.add_subplot(gs[2, 0])
    var_r2_data = {vn: [] for vn in var_names}
    for m in sim_metrics_list:
        pv = m.get('per_variable', {})
        for vn in var_names:
            if vn in pv:
                var_r2_data[vn].append(pv[vn]['r2'])
    bp_data = [var_r2_data[vn] for vn in var_names if var_r2_data[vn]]
    bp_labels = [vn for vn in var_names if var_r2_data[vn]]
    if bp_data:
        ax.boxplot(bp_data, labels=bp_labels)
    ax.set_ylabel('R²')
    ax.set_title('Per-Variable R²', fontweight='bold')
    ax.grid(alpha=0.3)

    # (2, 1) RMSE timeline
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(range(len(rmse_vals)), rmse_vals, 'o-', color='coral', ms=5, lw=1.5)
    ax.axhline(np.mean(rmse_vals), color='red', ls='--', alpha=0.5, label='Mean')
    ax.set_xlabel('Sim Index'); ax.set_ylabel('RMSE')
    ax.set_title('RMSE by Simulation', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (2, 2) Best / Worst table
    ax = fig.add_subplot(gs[2, 2]); ax.axis('off')
    sorted_idx = np.argsort(r2_vals)
    best_3 = sorted_idx[-3:][::-1]
    worst_3 = sorted_idx[:3]
    ct = f"BEST\n{'=' * 20}\n"
    for rank, i in enumerate(best_3, 1):
        ct += f"{rank}. Sim {sim_ids[i]}: R²={r2_vals[i]:.4f}\n"
    ct += f"\nWORST\n{'=' * 20}\n"
    for rank, i in enumerate(worst_3, 1):
        ct += f"{rank}. Sim {sim_ids[i]}: R²={r2_vals[i]:.4f}\n"
    ax.text(0.05, 0.5, ct, fontsize=10, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    fig.suptitle(f'{model_name} Shock Tube — {ml} Performance',
                 fontsize=16, fontweight='bold', y=0.99)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Dashboard saved: {output_path}")


def plot_global_parameter_analysis(sim_metrics_list, output_path,
                                    var_names=None, figsize=(22, 18)):
    """
    Comprehensive 4×3 analysis of performance vs global parameters
    (Δt, pressure, density). Includes 3D scatter, correlation heatmap,
    grouped bar charts.

    Args:
        sim_metrics_list: list of dicts with keys:
            'overall_physical': {'rmse', 'r2'}
            'metadata': {'delta_t', 'pressure', 'density', ...}
            'per_variable': {var_name: {'rmse', 'r2', 'mae'}}
    """
    if not sim_metrics_list:
        return

    if var_names is None:
        var_names = ['density', 'x_momentum', 'total_energy']

    # Build dataframe
    rows = []
    for m in sim_metrics_list:
        row = {
            'delta_t': m['metadata'].get('delta_t', 0),
            'pressure': m['metadata'].get('pressure', 0),
            'density': m['metadata'].get('density', 0),
            'overall_r2': _get_overall(m)['r2'],
            'overall_rmse': _get_overall(m)['rmse'],
        }
        pv = m.get('per_variable', {})
        for vn in var_names:
            if vn in pv:
                row[f'{vn}_r2'] = pv[vn]['r2']
                row[f'{vn}_rmse'] = pv[vn]['rmse']
        rows.append(row)

    # Use numpy structured approach to avoid pandas dependency
    # (pandas is optional but preferred)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        _has_pandas = True
    except ImportError:
        _has_pandas = False
        # Fallback: plain dict-of-lists
        df = {k: [r.get(k, 0) for r in rows] for k in rows[0].keys()}

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle('Performance vs Global Parameters (Δt, Pressure, Density)',
                 fontsize=16, fontweight='bold')

    params = ['delta_t', 'pressure', 'density']
    param_labels = ['Δt (s)', 'Pressure', 'Density']

    def _col(name):
        return df[name].values if _has_pandas else np.array(df[name])

    # ── Row 0: Overall R² vs each parameter ──
    for col, (param, plabel) in enumerate(zip(params, param_labels)):
        ax = fig.add_subplot(gs[0, col])
        sc = ax.scatter(_col(param), _col('overall_r2'), c=_col('overall_rmse'),
                        cmap='plasma_r', s=60, alpha=0.8, edgecolors='k', lw=0.3)
        ax.set_xlabel(plabel); ax.set_ylabel('R²')
        ax.set_title(f'R² vs {plabel}', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label='RMSE')
        if param == 'delta_t':
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))

    # ── Row 1: Overall RMSE vs each parameter ──
    for col, (param, plabel) in enumerate(zip(params, param_labels)):
        ax = fig.add_subplot(gs[1, col])
        ax.scatter(_col(param), _col('overall_rmse'), c='coral',
                   s=60, alpha=0.7, edgecolors='k', lw=0.3)
        ax.set_xlabel(plabel); ax.set_ylabel('RMSE')
        ax.set_title(f'RMSE vs {plabel}', fontweight='bold')
        ax.grid(alpha=0.3)
        if param == 'delta_t':
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))

    # ── Row 2, col 0: Per-variable R² vs delta_t ──
    ax = fig.add_subplot(gs[2, 0])
    markers = ['o', 's', '^', 'D', 'v']
    colors_var = plt.cm.Set1(np.linspace(0, 0.6, len(var_names)))
    for vi, vn in enumerate(var_names):
        col_name = f'{vn}_r2'
        vals = _col(col_name) if col_name in (df.columns if _has_pandas else df) else None
        if vals is not None:
            ax.scatter(_col('delta_t'), vals, marker=markers[vi % 5],
                       color=colors_var[vi], label=vn, s=50, alpha=0.7)
    ax.set_xlabel('Δt (s)'); ax.set_ylabel('R²')
    ax.set_title('Per-Variable R² vs Δt', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))

    # ── Row 2, col 1: Pressure vs Density colored by R² ──
    ax = fig.add_subplot(gs[2, 1])
    sc = ax.scatter(_col('pressure'), _col('density'), c=_col('overall_r2'),
                    cmap='viridis', s=60, alpha=0.8, edgecolors='k', lw=0.3)
    ax.set_xlabel('Pressure'); ax.set_ylabel('Density')
    ax.set_title('R² in Pressure–Density Space', fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label='R²')

    # ── Row 2, col 2: Correlation heatmap ──
    ax = fig.add_subplot(gs[2, 2])
    corr_cols = params + ['overall_r2', 'overall_rmse']
    if _has_pandas:
        valid_cols = [c for c in corr_cols if c in df.columns]
        corr = df[valid_cols].corr()
        corr_vals = corr.values
        corr_labels = list(corr.columns)
    else:
        valid_cols = [c for c in corr_cols if c in df]
        mat = np.column_stack([np.array(df[c], dtype=float) for c in valid_cols])
        corr_vals = np.corrcoef(mat.T)
        corr_labels = valid_cols
    im = ax.imshow(corr_vals, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr_labels)))
    ax.set_yticks(range(len(corr_labels)))
    ax.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr_labels, fontsize=8)
    ax.set_title('Correlation Matrix', fontweight='bold')
    for i in range(len(corr_labels)):
        for j in range(len(corr_labels)):
            ax.text(j, i, f'{corr_vals[i, j]:.2f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax)

    # ── Row 3, col 0: 3D scatter ──
    try:
        ax3d = fig.add_subplot(gs[3, 0], projection='3d')
        sc3d = ax3d.scatter(_col('delta_t'), _col('pressure'), _col('density'),
                            c=_col('overall_r2'), cmap='viridis', s=50, alpha=0.8)
        ax3d.set_xlabel('Δt'); ax3d.set_ylabel('Pressure'); ax3d.set_zlabel('Density')
        ax3d.set_title('R² in 3D Param Space', fontweight='bold')
        plt.colorbar(sc3d, ax=ax3d, shrink=0.6, label='R²')
    except Exception:
        ax3d = fig.add_subplot(gs[3, 0])
        ax3d.axis('off')
        ax3d.text(0.5, 0.5, '3D plot unavailable', ha='center', va='center')

    # ── Row 3, col 1-2: Grouped bar charts by Δt ──
    dt_arr = np.array(_col('delta_t'))
    r2_arr = np.array(_col('overall_r2'))
    rmse_arr = np.array(_col('overall_rmse'))
    dt_unique = np.unique(np.round(dt_arr, 8))

    dt_r2_means, dt_r2_stds, dt_rmse_means, dt_rmse_stds, dt_counts = [], [], [], [], []
    for dt_val in dt_unique:
        mask = np.abs(dt_arr - dt_val) < 1e-9
        dt_r2_means.append(r2_arr[mask].mean())
        dt_r2_stds.append(r2_arr[mask].std() if mask.sum() > 1 else 0)
        dt_rmse_means.append(rmse_arr[mask].mean())
        dt_rmse_stds.append(rmse_arr[mask].std() if mask.sum() > 1 else 0)
        dt_counts.append(mask.sum())

    x_pos = np.arange(len(dt_unique))
    xlabels = [f'{v:.2e}\n(n={int(c)})' for v, c in zip(dt_unique, dt_counts)]

    ax = fig.add_subplot(gs[3, 1])
    ax.bar(x_pos, dt_r2_means, yerr=dt_r2_stds, capsize=4,
           color='steelblue', edgecolor='k', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha='right')
    ax.set_xlabel('Δt'); ax.set_ylabel('Mean R²')
    ax.set_title('Mean R² by Δt Group', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[3, 2])
    ax.bar(x_pos, dt_rmse_means, yerr=dt_rmse_stds, capsize=4,
           color='coral', edgecolor='k', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha='right')
    ax.set_xlabel('Δt'); ax.set_ylabel('Mean RMSE')
    ax.set_title('Mean RMSE by Δt Group', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Global parameter analysis saved: {output_path}")


def create_delta_t_performance_table(sim_metrics_list):
    """
    Generate a formatted text table summarizing performance grouped by Δt.
    Returns a string.
    """
    if not sim_metrics_list:
        return "No data."

    dt_arr = np.array([m['metadata'].get('delta_t', 0) for m in sim_metrics_list])
    r2_arr = np.array([_get_overall(m)['r2'] for m in sim_metrics_list])
    rmse_arr = np.array([_get_overall(m)['rmse'] for m in sim_metrics_list])

    # Round both the unique bins AND the values used for matching
    dt_rounded = np.round(dt_arr, 8)
    dt_unique = np.unique(dt_rounded)

    lines = [
        f"\nDelta_t Performance Summary",
        "=" * 70,
        f"{'Δt':>14} {'Count':>6} {'R² mean':>10} {'R² std':>10} {'RMSE mean':>12} {'RMSE std':>12}",
        "-" * 70,
    ]
    for dt_val in sorted(dt_unique):
        mask = dt_rounded == dt_val
        count = int(mask.sum())
        if count == 0:
            continue
        r2_g = r2_arr[mask]
        rmse_g = rmse_arr[mask]
        lines.append(
            f"{dt_val:>14.6e} {count:>6} "
            f"{r2_g.mean():>10.4f} {r2_g.std():>10.4f} "
            f"{rmse_g.mean():>12.6e} {rmse_g.std():>12.6e}"
        )
    lines.append("=" * 70)
    return "\n".join(lines)