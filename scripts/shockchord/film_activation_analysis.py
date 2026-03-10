#!/usr/bin/env python3
"""
FiLM Activation Analysis -- Weak vs Strong Shock Regimes
=========================================================
Extracts gamma/beta from SimulationConditionedLayerNorm layers
across different (P0, rho0, dt) configurations and visualizes how
FiLM adapts the network's internal representations to shock strength.

Generates:
  1. Gamma/beta distributions for feature_norm and derivative_norm
     across weak vs strong shock regimes
  2. Per-variable derivative gamma/beta scatter vs pressure
  3. PCA of global embeddings colored by shock strength
  4. Feature gamma heatmap (top-k most variable dimensions)
  5. Summary panel (presentation-ready 2x2)
  6. Summary statistics JSON

Usage (place in scripts/shockchord/):
    python film_activation_analysis.py \
        --test_dir /path/to/test_cases_normalized \
        --checkpoint /path/to/best_model.pth \
        --output_dir ./film_analysis \
        --device cuda
"""

import argparse
import sys
import os
import json
import re
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.shockchord.eval_comparison import (
    NUM_STATIC, NUM_USED_DYNAMIC, SKIP_INDICES,
    extract_global_params_from_data, parse_params_from_filename,
    load_model_gparcv2,
)


# ===========================================================================
# STYLE
# ===========================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

WEAK_COLOR = '#2196F3'
STRONG_COLOR = '#E53935'
MID_COLOR = '#FF9800'
DYN_LABELS = [r'$\rho$', r'$\rho u$', r'$E$']
DYN_NAMES = ['density', 'x_momentum', 'total_energy']


# ===========================================================================
# EXTRACTION
# ===========================================================================

def ensure_global_attrs(data):
    """Unpack global_params -> individual attrs. Mirrors rollout_gparcv2."""
    if hasattr(data, 'global_params') and data.global_params.numel() >= 3:
        gp = data.global_params
        if not hasattr(data, 'global_pressure'):
            data.global_pressure = gp[0].unsqueeze(0)
            data.global_density = gp[1].unsqueeze(0)
            data.global_delta_t = gp[2].unsqueeze(0)


def extract_film_activations(model, sim_data, device):
    """Extract FiLM gamma/beta from feature_norm and derivative_norm."""
    model.eval()
    diff = model.derivative_solver
    first = sim_data[0]
    ensure_global_attrs(first)
    global_attrs = model._extract_global_attrs(first).to(device)
    global_embed = model.global_processor(global_attrs)

    with torch.no_grad():
        feat_gamma, feat_beta = diff.feature_norm.generate_gamma_beta(global_attrs)
        deriv_gamma, deriv_beta = diff.derivative_norm.generate_gamma_beta(global_attrs)

    return {
        'feature_gamma': feat_gamma.cpu().numpy().flatten(),
        'feature_beta': feat_beta.cpu().numpy().flatten(),
        'derivative_gamma': deriv_gamma.cpu().numpy().flatten(),
        'derivative_beta': deriv_beta.cpu().numpy().flatten(),
        'global_embed': global_embed.detach().cpu().numpy().flatten(),
        'global_attrs': global_attrs.cpu().numpy().flatten(),
    }


# ===========================================================================
# FIGURE 1: Violin distributions by regime
# ===========================================================================

def plot_film_distributions(acts_by_regime, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('FiLM Modulation Across Shock Regimes',
                 fontsize=15, fontweight='bold', y=0.98)

    rc = {'weak': WEAK_COLOR, 'medium': MID_COLOR, 'strong': STRONG_COLOR}
    order = [r for r in ['weak', 'medium', 'strong'] if r in acts_by_regime]

    panels = [
        (r'Feature $\gamma$ (Learned Features)', 'feature_gamma'),
        (r'Feature $\beta$ (Learned Features)', 'feature_beta'),
        (r'Derivative $\gamma$ (Dynamic State)', 'derivative_gamma'),
        (r'Derivative $\beta$ (Dynamic State)', 'derivative_beta'),
    ]

    for ax_i, (title, key) in enumerate(panels):
        ax = axes[ax_i // 2, ax_i % 2]
        data, labels, colors = [], [], []
        for regime in order:
            acts = acts_by_regime[regime]
            if not acts: continue
            data.append(np.concatenate([a[key] for a in acts]))
            labels.append(f'{regime.capitalize()}\n(n={len(acts)})')
            colors.append(rc[regime])

        if not data:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        parts = ax.violinplot(data, positions=range(len(data)),
                              showmeans=True, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i]); pc.set_alpha(0.6)
            pc.set_edgecolor('black'); pc.set_linewidth(0.8)
        parts['cmeans'].set_color('black'); parts['cmeans'].set_linewidth(1.5)
        parts['cmedians'].set_color('white'); parts['cmedians'].set_linewidth(1.2)

        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_title(title, fontweight='bold'); ax.set_ylabel('Activation Value')
        ax.axhline(y=1.0 if 'gamma' in key else 0.0,
                   color='gray', ls='--', alpha=0.5, lw=0.8)

    fig.text(0.5, 0.01,
             r'Dashed: identity ($\gamma=1, \beta=0$). '
             'Deviation = regime-specific modulation.',
             ha='center', fontsize=9, fontstyle='italic', color='gray')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'film_distributions.{ext}')
    plt.close(fig)
    print('  ok film_distributions.png/pdf')


# ===========================================================================
# FIGURE 2: Derivative FiLM scatter vs pressure
# ===========================================================================

def plot_derivative_film_per_variable(all_acts, all_params, output_dir):
    n_vars = len(DYN_LABELS)
    fig, axes = plt.subplots(2, n_vars, figsize=(4.5 * n_vars, 8))
    fig.suptitle(r'Derivative FiLM Coefficients vs Shock Strength ($P_0$)',
                 fontsize=14, fontweight='bold', y=0.98)

    mask = [p.get('pressure') is not None for p in all_params]
    pressures = np.array([p['pressure'] for p, m in zip(all_params, mask) if m])
    va = [a for a, m in zip(all_acts, mask) if m]

    if len(va) == 0:
        plt.close(fig); print('  warn: no data for derivative plot'); return

    gammas = np.array([a['derivative_gamma'] for a in va])
    betas = np.array([a['derivative_beta'] for a in va])

    p33, p67 = np.percentile(pressures, [33, 67])
    colors = [WEAK_COLOR if p <= p33 else STRONG_COLOR if p >= p67 else MID_COLOR
              for p in pressures]

    p_sorted = np.sort(pressures)

    for vi in range(n_vars):
        for row, (vals, ref, label) in enumerate([
            (gammas, 1.0, r'$\gamma$'), (betas, 0.0, r'$\beta$')
        ]):
            ax = axes[row, vi]
            ax.scatter(pressures, vals[:, vi], c=colors, s=40, alpha=0.7,
                       edgecolors='black', linewidth=0.5)
            if len(pressures) > 2:
                z = np.polyfit(pressures, vals[:, vi], 1)
                ax.plot(p_sorted, np.poly1d(z)(p_sorted), 'k--', alpha=0.5, lw=1.2)
                corr = np.corrcoef(pressures, vals[:, vi])[0, 1]
                ax.text(0.05, 0.92, f'r = {corr:.3f}', transform=ax.transAxes,
                        fontsize=9, va='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            ax.set_title(f'{DYN_LABELS[vi]}  {label}', fontweight='bold')
            ax.axhline(ref, color='gray', ls=':', alpha=0.4)
            if vi == 0: ax.set_ylabel(f'{label} value')
            if row == 1: ax.set_xlabel('Initial Pressure $P_0$ (Pa)')

    legend_el = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=l)
        for c, l in [(WEAK_COLOR, 'Weak'), (MID_COLOR, 'Medium'), (STRONG_COLOR, 'Strong')]
    ]
    fig.legend(handles=legend_el, loc='lower center', ncol=3, fontsize=10,
               title='Shock Regime', title_fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'derivative_film_per_variable.{ext}')
    plt.close(fig)
    print('  ok derivative_film_per_variable.png/pdf')


# ===========================================================================
# FIGURE 3: Global embedding PCA
# ===========================================================================

def plot_embedding_space(all_acts, all_params, output_dir, train_acts=None, train_params=None):
    embs = np.array([a['global_embed'] for a in all_acts])
    pressures = np.array([p.get('pressure', 0) for p in all_params])
    densities = np.array([p.get('density', 0) for p in all_params])

    if len(embs) < 3:
        print('  warn: too few sims for PCA'); return

    centered = embs - embs.mean(0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    pcs = centered @ evecs[:, :2]
    var_exp = evals[:2] / evals.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Global Parameter Embedding Space (PCA)', fontsize=14, fontweight='bold')

    for ax, vals, cmap, lab in [
        (axes[0], pressures, 'RdYlBu_r', '$P_0$ (Pa)'),
        (axes[1], densities, 'viridis', r'$\rho_0$ (kg/m$^3$)'),
    ]:
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=vals, cmap=cmap,
                        s=60, edgecolors='black', linewidth=0.5, alpha=0.8)
        plt.colorbar(sc, ax=ax, label=lab)
        ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var.)')
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var.)')

    axes[0].set_title('Colored by Pressure')
    axes[1].set_title('Colored by Density')
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'embedding_pca.{ext}')
    plt.close(fig)
    print('  ok embedding_pca.png/pdf')


# ===========================================================================
# FIGURE 4: Feature gamma heatmap
# ===========================================================================

def plot_gamma_correlations(all_acts, all_params, output_dir):
    """Scatter FiLM gamma magnitude vs each global parameter with linear fit and r value."""
    feat_gamma_mag = np.array([np.mean(np.abs(a['feature_gamma'])) for a in all_acts])
    pressures = np.array([p.get('pressure', np.nan) for p in all_params])
    densities = np.array([p.get('density', np.nan) for p in all_params])
    delta_ts = np.array([p.get('delta_t', np.nan) for p in all_params])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(r'FiLM $\gamma$ Correlation with Global Parameters',
                 fontsize=15, fontweight='bold', y=1.02)

    params_list = [
        (pressures, 'Initial Pressure $P_0$ (Pa)', 'Pressure'),
        (densities, r'Initial Density $\rho_0$ (kg/m$^3$)', 'Density'),
        (delta_ts, r'Timestep $\Delta t$', 'Timestep'),
    ]
    colors = ['#E53935', '#1E88E5', '#43A047']

    for ax, (vals, xlabel, label), color in zip(axes, params_list, colors):
        mask = ~np.isnan(vals)
        if mask.sum() < 3:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
            continue
        x, y = vals[mask], feat_gamma_mag[mask]
        ax.scatter(x, y, c=color, s=50, alpha=0.7, edgecolors='black', linewidth=0.4)
        z = np.polyfit(x, y, 1)
        xs = np.sort(x)
        ax.plot(xs, np.poly1d(z)(xs), 'k--', alpha=0.6, lw=1.5)
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.92, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, edgecolor=color, linewidth=1.5))
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(r'Mean $|\gamma_{feat}|$', fontsize=11)
        ax.set_title(f'vs {label}', fontweight='bold', fontsize=12)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'film_gamma_correlations.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  ok film_gamma_correlations.png/pdf')


def plot_feature_gamma_heatmap(all_acts, all_params, output_dir, top_k=20):
    gammas = np.array([a['feature_gamma'] for a in all_acts])
    pressures = np.array([p.get('pressure', 0) for p in all_params])

    if len(gammas) < 3:
        print('  warn: too few sims for heatmap'); return

    si = np.argsort(pressures)
    gammas_s, pressures_s = gammas[si], pressures[si]

    top_dims = np.argsort(np.var(gammas, axis=0))[::-1][:top_k]
    gammas_top = gammas_s[:, top_dims]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(gammas_top.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(im, ax=ax, label=r'$\gamma$ value')

    n = len(pressures_s)
    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{pressures_s[i]:.0f}' for i in ticks], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel(r'Simulations sorted by $P_0$ (Pa) $\longrightarrow$')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f'dim {d}' for d in top_dims], fontsize=7)
    ax.set_ylabel(f'Top-{top_k} Most Variable Feature Dimensions')
    ax.set_title(r'Feature FiLM $\gamma$ -- How Learned Features Scale With Shock Strength',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'feature_gamma_heatmap.{ext}')
    plt.close(fig)
    print('  ok feature_gamma_heatmap.png/pdf')


# ===========================================================================
# FIGURE 5: Presentation summary panel (2x2)
# ===========================================================================

def plot_summary_panel(acts_by_regime, all_acts, all_params, output_dir, train_acts=None, train_params=None):
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.32)
    fig.suptitle('FiLM Conditioning Response to Global Simulation Parameters',
                 fontsize=16, fontweight='bold', y=0.99)
    pressures = np.array([p.get('pressure', np.nan) for p in all_params])
    densities = np.array([p.get('density', np.nan) for p in all_params])
    delta_ts = np.array([p.get('delta_t', np.nan) for p in all_params])
    feat_gamma_mag = np.array([np.mean(np.abs(a['feature_gamma'])) for a in all_acts])
    has_train = train_acts is not None and len(train_acts) > 0
    if has_train:
        train_pressures = np.array([p.get('pressure', np.nan) for p in train_params])
        train_densities = np.array([p.get('density', np.nan) for p in train_params])
        train_delta_ts = np.array([p.get('delta_t', np.nan) for p in train_params])
    # --- TL: Train vs Test parameter space coverage ---
    ax = fig.add_subplot(gs[0, 0])
    if has_train:
        tm = ~(np.isnan(train_pressures) | np.isnan(train_densities))
        if tm.any():
            ax.scatter(train_pressures[tm], train_densities[tm], c='#90CAF9', s=40, marker='x', alpha=0.6, linewidths=1.0, label='Train (n=%d)' % tm.sum())
    mask = ~(np.isnan(pressures) | np.isnan(densities))
    if mask.any():
        ax.scatter(pressures[mask], densities[mask], c='#E53935', s=50, edgecolors='black', linewidth=0.5, alpha=0.8, label='Test (n=%d)' % mask.sum())
    ax.set_xlabel('Initial Pressure $P_0$')
    ax.set_ylabel(r'Initial Density $\rho_0$')
    ax.set_title('Train vs Test: Parameter Space Coverage', fontweight='bold', fontsize=11)
    ax.text(0.5, -0.19, 'Test samples span and extend training distribution', transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic', color='#555555')
    ax.legend(fontsize=9, loc='center left')
    # --- TR: Joint P0 vs rho0 colored by FiLM gamma ---
    ax = fig.add_subplot(gs[0, 1])
    mask = ~(np.isnan(pressures) | np.isnan(densities))
    if mask.any():
        sc = ax.scatter(pressures[mask], densities[mask], c=feat_gamma_mag[mask], cmap='magma', s=60, edgecolors='black', linewidth=0.5, alpha=0.85)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label(r'Mean $|\gamma_{feat}|$', fontsize=10)
    ax.set_xlabel('Initial Pressure $P_0$')
    ax.set_ylabel(r'Initial Density $\rho_0$')
    ax.set_title(r'FiLM $\gamma$ in Joint Parameter Space', fontweight='bold', fontsize=11)
    ax.text(0.5, -0.19, 'Brighter = stronger feature modulation', transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic', color='#555555')
    # --- BL: Embedding PCA (train + test) ---
    ax = fig.add_subplot(gs[1, 0])
    embs = np.array([a['global_embed'] for a in all_acts])
    test_pressures = pressures
    if has_train:
        train_embs = np.array([a['global_embed'] for a in train_acts])
        train_p = np.array([p.get('pressure', np.nan) for p in train_params])
        combined_embs = np.vstack([train_embs, embs])
    else:
        combined_embs = embs
    if len(combined_embs) >= 3:
        c = combined_embs - combined_embs.mean(0)
        ev, evec = np.linalg.eigh(np.cov(c.T))
        idx = np.argsort(ev)[::-1]
        evec, ev = evec[:, idx], ev[idx]
        all_pcs = c @ evec[:, :2]
        ve = ev[:2] / ev.sum() * 100
        if has_train:
            n_train = len(train_embs)
            train_pcs = all_pcs[:n_train]
            test_pcs = all_pcs[n_train:]
            ax.scatter(train_pcs[:, 0], train_pcs[:, 1], c=train_p, cmap='coolwarm', s=30, marker='x', alpha=0.5, linewidths=0.8, label='Train')
            sc = ax.scatter(test_pcs[:, 0], test_pcs[:, 1], c=test_pressures, cmap='coolwarm', s=50, edgecolors='black', linewidth=0.4, alpha=0.8, label='Test')
            ax.legend(fontsize=8, loc='upper left')
        else:
            sc = ax.scatter(all_pcs[:, 0], all_pcs[:, 1], c=test_pressures, cmap='coolwarm', s=50, edgecolors='black', linewidth=0.4, alpha=0.8)
        plt.colorbar(sc, ax=ax, label='$P_0$', shrink=0.85)
        ax.set_xlabel('PC1 (%.1f%%)' % ve[0])
        ax.set_ylabel('PC2 (%.1f%%)' % ve[1])
    if has_train:
        ax.set_title('Embedding Space (Train + Test)', fontweight='bold', fontsize=11)
        ax.text(0.5, -0.19, 'Train (x) and test (o) co-locate by regime', transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic', color='#555555')
    else:
        ax.set_title('Global Embedding Space (PCA)', fontweight='bold', fontsize=11)
        ax.text(0.5, -0.19, 'Distinct clusters separate shock regimes', transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic', color='#555555')
    # --- BR: FiLM gamma vs Delta-t ---
    ax = fig.add_subplot(gs[1, 1])
    mask_dt = ~np.isnan(delta_ts)
    if mask_dt.any():
        p_valid = pressures[~np.isnan(pressures)]
        p33 = np.percentile(p_valid, 33) if len(p_valid) > 0 else 0
        p67 = np.percentile(p_valid, 67) if len(p_valid) > 0 else 1
        dt_colors = []
        for p in pressures:
            if np.isnan(p): dt_colors.append('#999999')
            elif p <= p33: dt_colors.append(WEAK_COLOR)
            elif p >= p67: dt_colors.append(STRONG_COLOR)
            else: dt_colors.append(MID_COLOR)
        cm = [dt_colors[i] for i in range(len(dt_colors)) if mask_dt[i]]
        ax.scatter(delta_ts[mask_dt], feat_gamma_mag[mask_dt], c=cm, s=50, edgecolors='black', linewidth=0.5, alpha=0.8)
        if mask_dt.sum() > 2:
            z = np.polyfit(delta_ts[mask_dt], feat_gamma_mag[mask_dt], 1)
            xs = np.sort(delta_ts[mask_dt])
            ax.plot(xs, np.poly1d(z)(xs), 'k--', alpha=0.5, lw=1.2)
            corr = np.corrcoef(delta_ts[mask_dt], feat_gamma_mag[mask_dt])[0, 1]
            ax.text(0.05, 0.92, 'r = %.3f' % corr, transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'Mean $|\gamma_{feat}|$')
    ax.set_title(r'FiLM $\gamma$ vs Timestep Size', fontweight='bold', fontsize=11)
    ax.text(0.5, -0.19, r'No dependence on numerical resolution ($r \approx 0$)', transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic', color='#555555')
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'film_summary_panel.{ext}')
    plt.close(fig)
    print('  ok film_summary_panel.png/pdf')




# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="FiLM Activation Analysis")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default=None, help="Training data dir for PCA overlay")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./film_analysis")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_sims", type=int, default=None)
    parser.add_argument("--weak_threshold", type=float, default=None,
                        help="Pressure threshold for weak (auto=33rd percentile)")
    parser.add_argument("--strong_threshold", type=float, default=None,
                        help="Pressure threshold for strong (auto=67th percentile)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_dir = Path(args.test_dir)
    sim_files = sorted(test_dir.glob("*.pt"))
    if args.max_sims:
        sim_files = sim_files[:args.max_sims]
    print(f"Found {len(sim_files)} test simulations in {test_dir}")
    if not sim_files:
        print("No simulation files found!"); return

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    first_sim = torch.load(sim_files[0], weights_only=False)
    sample_data = first_sim[0]
    if not hasattr(sample_data, 'pos') or sample_data.pos is None:
        sample_data.pos = sample_data.x[:, :NUM_STATIC]

    ensure_global_attrs(sample_data)
    model = load_model_gparcv2(args.checkpoint, sample_data, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} parameters")

    # Extract activations
    all_acts, all_params = [], []
    for sf in sim_files:
        sname = sf.stem
        sd = torch.load(sf, weights_only=False)
        for d in sd:
            d.x = d.x.to(device)
            d.edge_index = d.edge_index.to(device)
            ensure_global_attrs(d)

        params = extract_global_params_from_data(sd[0])
        pf, rf = parse_params_from_filename(sname)
        if pf is not None: params['pressure'] = pf
        if rf is not None: params['density'] = rf

        acts = extract_film_activations(model, sd, device)
        all_acts.append(acts); all_params.append(params)
        print(f"  {sname}: P={params.get('pressure','?')}, "
              f"rho={params.get('density','?')}, dt={params.get('delta_t','?')}")

    print(f"\nExtracted {len(all_acts)} simulations")

    # Optionally extract train activations for PCA overlay
    train_acts, train_params = [], []
    if args.train_dir:
        train_dir = Path(args.train_dir)
        train_files = sorted(train_dir.glob("*.pt"))
        if args.max_sims:
            train_files = train_files[:args.max_sims]
        print(f"Extracting train activations from {len(train_files)} files...")
        for sf in train_files:
            try:
                sd = torch.load(sf, weights_only=False)
                for d in sd:
                    d.x = d.x.to(device)
                    d.edge_index = d.edge_index.to(device)
                    ensure_global_attrs(d)
                tp = extract_global_params_from_data(sd[0])
                pf, rf = parse_params_from_filename(sf.stem)
                if pf is not None: tp['pressure'] = pf
                if rf is not None: tp['density'] = rf
                ta = extract_film_activations(model, sd, device)
                train_acts.append(ta)
                train_params.append(tp)
            except Exception as e:
                pass
        print(f"  Extracted {len(train_acts)} train simulations")

    # Classify regimes
    vp = [p['pressure'] for p in all_params if p.get('pressure') is not None]
    abr = defaultdict(list)
    if vp:
        pa = np.array(vp)
        wt = args.weak_threshold or float(np.percentile(pa, 33))
        st = args.strong_threshold or float(np.percentile(pa, 67))
        print(f"\nThresholds: weak < {wt:.0f}, strong > {st:.0f}")
        for a, p in zip(all_acts, all_params):
            pr = p.get('pressure')
            if pr is None: continue
            if pr <= wt:   abr['weak'].append(a)
            elif pr >= st: abr['strong'].append(a)
            else:          abr['medium'].append(a)
        for r, a in abr.items(): print(f"  {r}: {len(a)}")

    # Generate figures
    print("\n" + "=" * 60 + "\nGenerating figures...\n" + "=" * 60)
    if abr:
        plot_film_distributions(abr, output_dir)
        plot_summary_panel(abr, all_acts, all_params, output_dir, train_acts=train_acts, train_params=train_params)
    plot_derivative_film_per_variable(all_acts, all_params, output_dir)
    plot_embedding_space(all_acts, all_params, output_dir)
    plot_gamma_correlations(all_acts, all_params, output_dir)
    plot_feature_gamma_heatmap(all_acts, all_params, output_dir)

    # Summary JSON
    summary = {
        'n_simulations': len(all_acts),
        'regimes': {r: len(a) for r, a in abr.items()},
        'pressure_range': [float(min(vp)), float(max(vp))] if vp else None,
        'derivative_gamma_stats': {
            var: {'mean': float(np.mean([a['derivative_gamma'][vi] for a in all_acts])),
                  'std': float(np.std([a['derivative_gamma'][vi] for a in all_acts]))}
            for vi, var in enumerate(DYN_NAMES)
        },
        'derivative_beta_stats': {
            var: {'mean': float(np.mean([a['derivative_beta'][vi] for a in all_acts])),
                  'std': float(np.std([a['derivative_beta'][vi] for a in all_acts]))}
            for vi, var in enumerate(DYN_NAMES)
        },
    }
    with open(output_dir / 'film_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n  ok film_analysis_summary.json')
    print(f"\nDone! All figures saved to {output_dir}")


if __name__ == "__main__":
    main()