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
  6. Each of the 4 summary sub-panels as standalone publication PDFs
     saved to <output_dir>/standalone_panels/
  7. Summary statistics JSON

Usage (place in scripts/shockchord/):
    python film_activation_analysis.py \
        --test_dir /path/to/test_cases_normalized \
        --checkpoint /path/to/best_model.pth \
        --output_dir ./film_analysis \
        --device cuda
"""

import json
import torch
from pathlib import Path

# In a notebook __file__ is undefined — point directly to your source root
SRC_ROOT = "/path/to/your/G-PARC"   # e.g. "/home/jtb3sud/G-PARC" or wherever models/ lives
import sys, os
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from models.shocktube_gparcv2 import GPARC_ShockTube_V2
from differentiator.nospade import ShockTubeDifferentiator
from utilities.featureextractor import GraphConvFeatureExtractorV2
from differentiator.hop import SolveGradientsLST, SolveWeightLST2d

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

# Publication-quality rcParams for standalone single-panel figures
PAPER_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

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
# FIGURE 4: Feature gamma heatmap + correlations
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
# SHARED HELPERS FOR STANDALONE PANELS
# ===========================================================================

def _compute_pca(all_acts, all_params, train_acts=None, train_params=None):
    """Shared PCA computation reused by the summary panel and standalone PCA figure."""
    embs = np.array([a['global_embed'] for a in all_acts])
    pressures = np.array([p.get('pressure', np.nan) for p in all_params])
    has_train = train_acts is not None and len(train_acts) > 0

    if has_train:
        train_embs = np.array([a['global_embed'] for a in train_acts])
        train_p = np.array([p.get('pressure', np.nan) for p in train_params])
        combined_embs = np.vstack([train_embs, embs])
    else:
        train_embs, train_p = None, None
        combined_embs = embs

    if len(combined_embs) < 3:
        return None

    c = combined_embs - combined_embs.mean(0)
    ev, evec = np.linalg.eigh(np.cov(c.T))
    idx = np.argsort(ev)[::-1]
    evec, ev = evec[:, idx], ev[idx]
    all_pcs = c @ evec[:, :2]
    ve = ev[:2] / ev.sum() * 100

    result = {
        've': ve,
        'pressures': pressures,
        'has_train': has_train,
    }
    if has_train:
        n_train = len(train_embs)
        result['train_pcs'] = all_pcs[:n_train]
        result['test_pcs'] = all_pcs[n_train:]
        result['train_p'] = train_p
    else:
        result['test_pcs'] = all_pcs
    return result


def _save_standalone(fig, output_dir, stem):
    """Save a standalone figure as both PDF and PNG."""
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'{stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'    ok {stem}.pdf/.png')


# ===========================================================================
# STANDALONE PANEL A — Parameter Space Coverage
# ===========================================================================

def plot_standalone_param_coverage(all_acts, all_params, output_dir,
                                   train_acts=None, train_params=None):
    """Standalone publication figure: train vs test parameter space coverage."""
    pressures = np.array([p.get('pressure', np.nan) for p in all_params])
    densities = np.array([p.get('density', np.nan) for p in all_params])
    has_train = train_acts is not None and len(train_acts) > 0

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        if has_train:
            tp = np.array([p.get('pressure', np.nan) for p in train_params])
            td = np.array([p.get('density', np.nan) for p in train_params])
            tm = ~(np.isnan(tp) | np.isnan(td))
            if tm.any():
                ax.scatter(tp[tm], td[tm], c='#90CAF9', s=50, marker='x',
                           alpha=0.65, linewidths=1.3,
                           label=f'Train ($n={tm.sum()}$)', zorder=2)

        mask = ~(np.isnan(pressures) | np.isnan(densities))
        if mask.any():
            ax.scatter(pressures[mask], densities[mask], c='#E53935', s=60,
                       edgecolors='black', linewidth=0.5, alpha=0.85,
                       label=f'Test ($n={mask.sum()}$)', zorder=3)

        ax.set_xlabel('Initial Pressure $P_0$ (Pa)')
        ax.set_ylabel(r'Initial Density $\rho_0$ (kg/m$^3$)')
        ax.set_title('Train vs. Test: Parameter Space Coverage', fontweight='bold')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        plt.tight_layout()

    _save_standalone(fig, output_dir, 'panel_A_param_coverage')


# ===========================================================================
# STANDALONE PANEL B — FiLM gamma in joint parameter space
# ===========================================================================

def plot_standalone_gamma_joint(all_acts, all_params, output_dir):
    """Standalone publication figure: mean |gamma_feat| over (P0, rho0) space."""
    pressures = np.array([p.get('pressure', np.nan) for p in all_params])
    densities = np.array([p.get('density', np.nan) for p in all_params])
    feat_gamma_mag = np.array([np.mean(np.abs(a['feature_gamma'])) for a in all_acts])

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        mask = ~(np.isnan(pressures) | np.isnan(densities))
        if mask.any():
            sc = ax.scatter(
                pressures[mask], densities[mask],
                c=feat_gamma_mag[mask], cmap='magma',
                s=70, edgecolors='black', linewidth=0.5, alpha=0.9,
                vmin=feat_gamma_mag[mask].min(),
                vmax=feat_gamma_mag[mask].max(),
            )
            cbar = plt.colorbar(sc, ax=ax, shrink=0.92, pad=0.02)
            cbar.set_label(r'Mean $|\gamma_{\mathrm{feat}}|$', fontsize=13)
            cbar.ax.tick_params(labelsize=11)

        ax.set_xlabel('Initial Pressure $P_0$ (Pa)')
        ax.set_ylabel(r'Initial Density $\rho_0$ (kg/m$^3$)')
        ax.set_title(r'FiLM Feature Modulation Strength', fontweight='bold')
        plt.tight_layout()

    _save_standalone(fig, output_dir, 'panel_B_gamma_joint')


# ===========================================================================
# STANDALONE PANEL C — Global embedding PCA
# ===========================================================================

def plot_standalone_embedding_pca(all_acts, all_params, output_dir,
                                  train_acts=None, train_params=None):
    """Standalone publication figure: PCA of global embeddings colored by P0."""
    pca = _compute_pca(all_acts, all_params, train_acts, train_params)
    if pca is None:
        print('  warn: too few sims for standalone PCA'); return

    ve = pca['ve']
    pressures = pca['pressures']

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.8))

        if pca['has_train']:
            ax.scatter(
                pca['train_pcs'][:, 0], pca['train_pcs'][:, 1],
                c=pca['train_p'], cmap='coolwarm',
                s=35, marker='x', alpha=0.5, linewidths=0.9,
                label='Train', zorder=2,
            )
            sc = ax.scatter(
                pca['test_pcs'][:, 0], pca['test_pcs'][:, 1],
                c=pressures, cmap='coolwarm',
                s=60, edgecolors='black', linewidth=0.4, alpha=0.85,
                label='Test', zorder=3,
            )
            ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
            title = 'Global Embedding PCA (Train + Test)'
        else:
            sc = ax.scatter(
                pca['test_pcs'][:, 0], pca['test_pcs'][:, 1],
                c=pressures, cmap='coolwarm',
                s=65, edgecolors='black', linewidth=0.4, alpha=0.85,
            )
            title = 'Global Embedding Space (PCA)'

        cbar = plt.colorbar(sc, ax=ax, shrink=0.92, pad=0.02)
        cbar.set_label('Initial Pressure $P_0$ (Pa)', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        ax.set_xlabel('PC$_1$ (%.1f%% var.)' % ve[0])
        ax.set_ylabel('PC$_2$ (%.1f%% var.)' % ve[1])
        ax.set_title(title, fontweight='bold')
        plt.tight_layout()

    _save_standalone(fig, output_dir, 'panel_C_embedding_pca')


# ===========================================================================
# STANDALONE PANEL C2 — Extended PCA: what drives each PC axis?
# ===========================================================================

def _format_pca_axes(ax, ve, xlabel=None, ylabel=None):
    """Clean tick formatting for PCA scatter axes — prevents label crowding."""
    ax.set_xlabel(xlabel or ('PC$_1$ (%.1f%% var.)' % ve[0]), labelpad=6)
    ax.set_ylabel(ylabel or ('PC$_2$ (%.1f%% var.)' % ve[1]), labelpad=6)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4, prune='both'))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune='both'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '%.2f' % x))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '%.2f' % x))
    ax.tick_params(axis='x', labelrotation=30, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)


def plot_standalone_pca_decomposed(all_acts, all_params, output_dir,
                                   train_acts=None, train_params=None):
    """
    Three-panel PCA figure coloring the same embedding by P0, rho0, and delta_t.
    Axis tick crowding is fixed via MaxNLocator + rotation.
    r values show which parameter explains each PC.

    Interpretation note: delta_t is CFL-derived from (P0, rho0), so r(PC1, dt)~1
    means the encoder learned the CFL wave-speed summary as its primary axis.
    """
    pca = _compute_pca(all_acts, all_params, train_acts, train_params)
    if pca is None:
        print('  warn: too few sims for decomposed PCA'); return

    ve = pca['ve']
    test_pcs = pca['test_pcs']
    pressures = np.array([p.get('pressure', np.nan) for p in all_params])
    densities = np.array([p.get('density', np.nan) for p in all_params])
    delta_ts  = np.array([p.get('delta_t',  np.nan) for p in all_params])

    params_cfg = [
        (pressures, 'RdYlBu_r', 'Initial Pressure $P_0$ (Pa)',          'Pressure $P_0$'),
        (densities, 'viridis',  r'Density $\rho_0$ (kg/m$^3$)',         r'Density $\rho_0$'),
        (delta_ts,  'plasma',   r'Timestep $\Delta t$ (s)',              r'Timestep $\Delta t$'),
    ]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
        fig.suptitle('Global Embedding PCA: What Drives Each Component?\n'
                     r'($\Delta t$ is CFL-derived from $P_0$, $\rho_0$ — not independent)',
                     fontsize=13, fontweight='bold', y=1.04)

        for ax, (vals, cmap, cbar_label, title) in zip(axes, params_cfg):
            mask = ~np.isnan(vals)

            if pca['has_train']:
                ax.scatter(pca['train_pcs'][:, 0], pca['train_pcs'][:, 1],
                           c='#dddddd', s=18, marker='x', alpha=0.35,
                           linewidths=0.6, zorder=1)

            sc = ax.scatter(
                test_pcs[mask, 0], test_pcs[mask, 1],
                c=vals[mask], cmap=cmap,
                s=65, edgecolors='black', linewidth=0.4, alpha=0.9, zorder=3,
            )
            cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
            cbar.set_label(cbar_label, fontsize=10)
            cbar.ax.tick_params(labelsize=9)

            # r values — right-aligned so they don't crowd the data
            for pc_idx, pc_tag in [(0, 'PC_1'), (1, 'PC_2')]:
                v = test_pcs[mask, pc_idx]
                r = np.corrcoef(vals[mask], v)[0, 1] if mask.sum() > 2 else np.nan
                ax.text(0.97, 0.97 - pc_idx * 0.11,
                        '$r(\\mathrm{%s})=%+.2f$' % (pc_tag, r),
                        transform=ax.transAxes, fontsize=10, va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                  alpha=0.88, edgecolor='#aaaaaa', linewidth=0.8))

            _format_pca_axes(ax, ve)
            ax.set_title('Colored by %s' % title, fontweight='bold', fontsize=12)

        plt.tight_layout()

    _save_standalone(fig, output_dir, 'panel_C2_pca_decomposed')


def plot_standalone_pca_derived(all_acts, all_params, output_dir,
                                train_acts=None, train_params=None):
    """
    Panel C3 (final): single panel colored by log(P0/rho0) ~ log(c^2).
    All four r-values reported in two annotation boxes.
    - log(P0/rho0) box: upper-left (dominant for PC1)
    - log(P0*rho0) box: upper-right (dominant for PC2)
    Title removed (goes in figure caption).
    """
    pca = _compute_pca(all_acts, all_params, train_acts, train_params)
    if pca is None:
        print('  warn: too few sims for derived PCA'); return

    ve       = pca['ve']
    test_pcs = pca['test_pcs']
    pressures = np.array([p.get('pressure', np.nan) for p in all_params])
    densities = np.array([p.get('density',  np.nan) for p in all_params])

    with np.errstate(divide='ignore', invalid='ignore'):
        p_over_rho  = np.where((pressures > 0) & (densities > 0),
                               pressures / densities, np.nan)
        p_times_rho = np.where((pressures > 0) & (densities > 0),
                               pressures * densities, np.nan)
        log_c2 = np.where(p_over_rho  > 0, np.log10(p_over_rho),  np.nan)
        log_Z2 = np.where(p_times_rho > 0, np.log10(p_times_rho), np.nan)

    PANEL_RC = {
        **PAPER_RC,
        'font.size':         17,
        'axes.labelsize':    18,
        'axes.titlesize':    18,
        'xtick.labelsize':   15,
        'ytick.labelsize':   15,
        'legend.fontsize':   15,
        'axes.spines.top':   False,
        'axes.spines.right': False,
    }

    mask   = ~np.isnan(log_c2)
    mask_Z = ~np.isnan(log_Z2)

    with plt.rc_context(PANEL_RC):
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))

        # ── Training samples ─────────────────────────────────────────────────
        if pca['has_train']:
            ax.scatter(
                pca['train_pcs'][:, 0], pca['train_pcs'][:, 1],
                c='#b0b0b0', s=65, marker='o', alpha=0.55,
                linewidths=0, zorder=1, label='Train',
            )

        # ── Test samples colored by log(P0/rho0) ─────────────────────────────
        sc = ax.scatter(
            test_pcs[mask, 0], test_pcs[mask, 1],
            c=log_c2[mask], cmap='RdBu_r',
            s=100, edgecolors='black', linewidth=0.5, alpha=0.95, zorder=3,
            label='Test' if pca['has_train'] else None,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.86, pad=0.03)
        cbar.set_label(r'$\log_{10}(P_0/\rho_0) \propto \log(c^2)$', fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # ── r-value annotation blocks ─────────────────────────────────────────
        # Compute all four r values
        r_c2_pc1 = np.corrcoef(log_c2[mask],   test_pcs[mask,   0])[0, 1] if mask.sum()   > 2 else np.nan
        r_c2_pc2 = np.corrcoef(log_c2[mask],   test_pcs[mask,   1])[0, 1] if mask.sum()   > 2 else np.nan
        r_Z2_pc1 = np.corrcoef(log_Z2[mask_Z], test_pcs[mask_Z, 0])[0, 1] if mask_Z.sum() > 2 else np.nan
        r_Z2_pc2 = np.corrcoef(log_Z2[mask_Z], test_pcs[mask_Z, 1])[0, 1] if mask_Z.sum() > 2 else np.nan

        # Upper-left: log(P0/rho0) — dominant for PC1
        block_c2 = '\n'.join([
            r'$\log(P_0/\rho_0)$',
            r'  $|r(\mathrm{PC_1})| = %.2f$' % abs(r_c2_pc1),
            r'  $|r(\mathrm{PC_2})| = %.2f$' % abs(r_c2_pc2),
        ])
        ax.text(
            0.03, 0.98, block_c2,
            transform=ax.transAxes, fontsize=15,
            va='top', ha='left', linespacing=1.7,
            bbox=dict(boxstyle='round,pad=0.55', fc='white', alpha=0.95,
                      edgecolor='#C62828', linewidth=2.2),
        )

        # Upper-right: log(P0*rho0) — dominant for PC2
        block_Z2 = '\n'.join([
            r'$\log(P_0\!\cdot\!\rho_0)$',
            r'  $|r(\mathrm{PC_1})| = %.2f$' % abs(r_Z2_pc1),
            r'  $|r(\mathrm{PC_2})| = %.2f$' % abs(r_Z2_pc2),
        ])
        ax.text(
            0.97, 0.98, block_Z2,
            transform=ax.transAxes, fontsize=15,
            va='top', ha='right', linespacing=1.7,
            bbox=dict(boxstyle='round,pad=0.55', fc='white', alpha=0.95,
                      edgecolor='#C62828', linewidth=2.2),
        )

        # ── Directional arrows outside axes ──────────────────────────────────
        # Horizontal arrow + label — pushed well below the x-axis title
        ax.annotate(
            '', xy=(0.92, -0.22), xytext=(0.08, -0.22),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='#606060', lw=1.8),
            annotation_clip=False,
        )
        ax.text(
            0.50, -0.27, r'increasing wave speed $c$',
            ha='center', va='top', fontsize=14, color='#555555',
            fontstyle='italic', transform=ax.transAxes,
        )

        # Vertical arrow + label to the left of the y-axis
        ax.annotate(
            '', xy=(-0.20, 0.92), xytext=(-0.20, 0.08),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='#606060', lw=1.8),
            annotation_clip=False,
        )
        ax.text(
            -0.26, 0.50, r'increasing impedance $Z$',
            ha='center', va='center', fontsize=14, color='#555555',
            fontstyle='italic', rotation=90, transform=ax.transAxes,
        )

        # ── Axes labels + title ───────────────────────────────────────────────
        _format_pca_axes(ax, ve)
        ax.set_title(
            r'PC$_1$: Speed-of-Sound Axis  |  PC$_2$: Acoustic Impedance Axis',
            fontweight='bold', fontsize=18, pad=14,
        )

        if pca['has_train']:
            ax.legend(fontsize=14, loc='lower right', framealpha=0.92,
                      markerscale=1.3, handletextpad=0.5)

        plt.subplots_adjust(left=0.18, right=0.96, bottom=0.26, top=0.93)

    _save_standalone(fig, output_dir, 'panel_C3_pca_physical')

    # Also save the full 4-panel exploratory version (unchanged)
    _plot_pca_derived_exploratory(
        pca, test_pcs, pressures, densities,
        log_c2, log_Z2,
        np.array([p.get('delta_t', np.nan) for p in all_params]),
        ve, output_dir,
    )


def _plot_pca_derived_exploratory(pca, test_pcs, pressures, densities,
                                  log_c2, log_Z2, delta_ts, ve, output_dir):
    """4-panel exploratory version kept for reference."""
    with np.errstate(divide='ignore', invalid='ignore'):
        p_over_rho = np.where((pressures > 0) & (densities > 0),
                              pressures / densities, np.nan)

    derived_cfg = [
        (log_c2,     'RdBu_r', r'$\log_{10}(P_0/\rho_0)$',
         r'$\log(P_0/\rho_0) \propto \log(c^2)$' + '\n[PC$_1$ hypothesis]'),
        (log_Z2,     'PuOr',   r'$\log_{10}(P_0 \cdot \rho_0)$',
         r'$\log(P_0 \cdot \rho_0) \propto \log(Z^2)$' + '\n[PC$_2$ hypothesis]'),
        (p_over_rho, 'magma',  r'$P_0/\rho_0$ (prop. to $c^2$)',
         r'$P_0/\rho_0$ (linear scale)'),
        (delta_ts,   'plasma', r'$\Delta t$ (s)',
         r'$\Delta t$ (CFL proxy for $c$)'),
    ]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
        fig.suptitle(
            r'Probing the Encoder: PC$_1$ $\approx$ Speed of Sound, '
            r'PC$_2$ $\approx$ Acoustic Impedance?',
            fontsize=13, fontweight='bold', y=1.04)

        for ax, (vals, cmap, cbar_label, title) in zip(axes, derived_cfg):
            mask = ~np.isnan(vals)
            if mask.sum() < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
                continue
            if pca['has_train']:
                ax.scatter(pca['train_pcs'][:, 0], pca['train_pcs'][:, 1],
                           c='#dddddd', s=18, marker='x', alpha=0.35,
                           linewidths=0.6, zorder=1)
            sc = ax.scatter(test_pcs[mask, 0], test_pcs[mask, 1],
                            c=vals[mask], cmap=cmap,
                            s=60, edgecolors='black', linewidth=0.4,
                            alpha=0.9, zorder=3)
            cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
            cbar.set_label(cbar_label, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            for pc_idx, pc_tag in [(0, 'PC_1'), (1, 'PC_2')]:
                v = test_pcs[mask, pc_idx]
                r = np.corrcoef(vals[mask], v)[0, 1] if mask.sum() > 2 else np.nan
                ax.text(0.97, 0.97 - pc_idx * 0.11,
                        '$r(\\mathrm{%s})=%+.2f$' % (pc_tag, r),
                        transform=ax.transAxes, fontsize=10, va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                  alpha=0.88, edgecolor='#aaaaaa', linewidth=0.8))
            _format_pca_axes(ax, ve)
            ax.set_title(title, fontweight='bold', fontsize=10, pad=6)
        plt.tight_layout()

    _save_standalone(fig, output_dir, 'panel_C3_pca_derived')


def plot_standalone_pca_r2_summary(all_acts, all_params, output_dir,
                                   train_acts=None, train_params=None):
    """
    Panel C4: R² bar chart + signed r heatmap.
    Includes log(P0/rho0) [speed-of-sound] and log(P0*rho0) [impedance]
    as derived candidates. Physical interpretation annotations mark the
    dominant explanator for PC1 and PC2.
    """
    pca = _compute_pca(all_acts, all_params, train_acts, train_params)
    if pca is None:
        print('  warn: too few sims for R2 summary'); return

    test_pcs = pca['test_pcs']
    pressures = np.array([p.get('pressure', np.nan) for p in all_params])
    densities = np.array([p.get('density', np.nan) for p in all_params])
    delta_ts  = np.array([p.get('delta_t',  np.nan) for p in all_params])

    with np.errstate(divide='ignore', invalid='ignore'):
        p_over_rho  = np.where((pressures > 0) & (densities > 0),
                               pressures / densities, np.nan)
        p_times_rho = np.where((pressures > 0) & (densities > 0),
                               pressures * densities, np.nan)
        log_p_rho   = np.where(p_over_rho  > 0, np.log10(p_over_rho),  np.nan)
        log_p_x_rho = np.where(p_times_rho > 0, np.log10(p_times_rho), np.nan)

    named_params = [
        (pressures,   '$P_0$'),
        (densities,   r'$\rho_0$'),
        (delta_ts,    r'$\Delta t$'),
        (log_p_rho,   r'$\log(P_0/\rho_0)$' + '\n' + r'[$\propto\log c^2$]'),
        (log_p_x_rho, r'$\log(P_0\!\cdot\!\rho_0)$' + '\n' + r'[$\propto\log Z^2$]'),
    ]
    # Short labels for heatmap x-axis
    short_labels = [
        '$P_0$', r'$\rho_0$', r'$\Delta t$',
        r'$\log(P_0/\rho_0)$', r'$\log(P_0{\cdot}\rho_0)$',
    ]

    n_pcs_show = min(4, test_pcs.shape[1])
    r2_matrix = np.full((len(named_params), n_pcs_show), np.nan)
    r_matrix  = np.full((len(named_params), n_pcs_show), np.nan)

    for pi, (vals, _) in enumerate(named_params):
        mask = ~np.isnan(vals)
        if mask.sum() < 3: continue
        for ci in range(n_pcs_show):
            r = np.corrcoef(vals[mask], test_pcs[mask, ci])[0, 1]
            r2_matrix[pi, ci] = r ** 2
            r_matrix[pi, ci]  = r

    ve = pca['ve']
    pc_labels = ['PC$_%d$ (%.1f%%)' % (i + 1, ve[i]) for i in range(n_pcs_show)]
    colors_pc = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']

    # Find best explanator per PC (for annotation)
    best_idx = [int(np.nanargmax(r2_matrix[:, ci])) for ci in range(2)]
    phys_labels = {
        best_idx[0]: r'$\leftarrow$ PC$_1$: speed of sound $c$',
        best_idx[1]: r'$\leftarrow$ PC$_2$: impedance $Z$?',
    }

    param_labels = [lbl for _, lbl in named_params]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                 gridspec_kw={'width_ratios': [1.7, 1]})
        fig.suptitle(
            r'FiLM Encoder $R^2$: Physical Coordinates Emerge from Training' + '\n' +
            r'PC$_1 \approx \log(c^2)$,  PC$_2 \approx \log(Z^2)$  '
            r'(speed-of-sound and acoustic impedance)',
            fontsize=12, fontweight='bold', y=1.04)

        # --- Left: grouped bar chart ---
        ax = axes[0]
        x = np.arange(len(named_params))
        bar_w = 0.18
        for ci in range(n_pcs_show):
            offset = (ci - (n_pcs_show - 1) / 2) * bar_w
            ax.bar(x + offset, r2_matrix[:, ci], bar_w,
                   label=pc_labels[ci], color=colors_pc[ci],
                   edgecolor='black', linewidth=0.5, alpha=0.85)
            for bi, (bval, rv) in enumerate(zip(r2_matrix[:, ci], r_matrix[:, ci])):
                if not np.isnan(bval) and bval > 0.05:
                    sign = '+' if rv > 0 else '−'
                    ax.text(x[bi] + offset, bval + 0.012, sign,
                            ha='center', va='bottom', fontsize=8, color='#333333')

        # Highlight the best-explaining parameter for PC1 and PC2
        highlight_colors = {best_idx[0]: '#E53935', best_idx[1]: '#1E88E5'}
        for pi, color in highlight_colors.items():
            ax.axvspan(pi - 0.45, pi + 0.45, alpha=0.07, color=color, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, fontsize=11)
        ax.set_ylabel('$R^2$ (fraction of PC variance explained)')
        ax.set_ylim(0, 1.18)
        ax.axhline(1.0, color='gray', lw=0.7, ls='--', alpha=0.4)
        ax.legend(fontsize=10, loc='upper left', framealpha=0.9, ncol=2)
        ax.set_title('Which Parameter Best Explains Each PC?',
                     fontweight='bold', fontsize=11)

        # Physical interpretation text box
        interp_text = (
            r'$\mathbf{PC_1}$ ($72\%$ var.): $r^2\!=\!0.98$ with $\Delta t$, '
            r'$r^2\!=\!0.98$ with $\log(P_0/\rho_0)$' + '\n'
            r'  $\Rightarrow$ Encoder axis $\approx$ speed of sound $c = \sqrt{\gamma P/\rho}$' + '\n\n'
            r'$\mathbf{PC_2}$ ($22\%$ var.): low $r^2$ with $\Delta t$ and $\log(P_0/\rho_0)$' + '\n'
            r'  $\Rightarrow$ Orthogonal residual; likely $\log(P_0\!\cdot\!\rho_0) \propto \log Z^2$' + '\n'
            r'     where $Z = \rho c$ is acoustic impedance'
        )
        ax.text(0.01, 0.01, interp_text,
                transform=ax.transAxes, fontsize=8.5, va='bottom',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', fc='#f9f9f9',
                          edgecolor='#cccccc', linewidth=1.0, alpha=0.95))

        # --- Right: signed r heatmap ---
        ax = axes[1]
        r_plot = np.where(np.isnan(r_matrix[:, :2]), 0, r_matrix[:, :2])
        im = ax.imshow(r_plot.T, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, label='Pearson $r$', shrink=0.75)
        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, fontsize=10, rotation=30, ha='right')
        ax.set_yticks(range(2))
        ax.set_yticklabels(pc_labels[:2], fontsize=11)

        for pi in range(len(short_labels)):
            for ci in range(2):
                rv = r_matrix[pi, ci]
                if not np.isnan(rv):
                    ax.text(pi, ci, '%+.2f' % rv, ha='center', va='center',
                            fontsize=10,
                            color='white' if abs(rv) > 0.6 else 'black',
                            fontweight='bold' if abs(rv) > 0.8 else 'normal')

        # Box the dominant cell per PC
        from matplotlib.patches import Rectangle
        for ci in range(2):
            pi_best = int(np.nanargmax(np.abs(r_matrix[:, ci])))
            rect = Rectangle((pi_best - 0.5, ci - 0.5), 1, 1,
                              linewidth=2.5, edgecolor='gold', facecolor='none', zorder=5)
            ax.add_patch(rect)

        ax.set_title('Signed $r$ (PC$_1$ and PC$_2$ only)',
                     fontweight='bold', fontsize=11)

        plt.tight_layout()

    _save_standalone(fig, output_dir, 'panel_C4_pca_r2_summary')


# ===========================================================================
# STANDALONE PANEL D — FiLM gamma vs timestep size
# ===========================================================================

def plot_standalone_gamma_vs_dt(all_acts, all_params, output_dir):
    """Standalone publication figure: mean |gamma_feat| vs delta-t, colored by shock regime."""
    pressures = np.array([p.get('pressure', np.nan) for p in all_params])
    delta_ts = np.array([p.get('delta_t', np.nan) for p in all_params])
    feat_gamma_mag = np.array([np.mean(np.abs(a['feature_gamma'])) for a in all_acts])

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        mask_dt = ~np.isnan(delta_ts)
        if mask_dt.any():
            p_valid = pressures[~np.isnan(pressures)]
            p33 = np.percentile(p_valid, 33) if len(p_valid) > 0 else 0
            p67 = np.percentile(p_valid, 67) if len(p_valid) > 0 else 1

            dt_colors = []
            for p in pressures:
                if np.isnan(p):    dt_colors.append('#999999')
                elif p <= p33:     dt_colors.append(WEAK_COLOR)
                elif p >= p67:     dt_colors.append(STRONG_COLOR)
                else:              dt_colors.append(MID_COLOR)

            cm = [dt_colors[i] for i in range(len(dt_colors)) if mask_dt[i]]
            ax.scatter(delta_ts[mask_dt], feat_gamma_mag[mask_dt],
                       c=cm, s=65, edgecolors='black', linewidth=0.5,
                       alpha=0.85, zorder=3)

            if mask_dt.sum() > 2:
                z = np.polyfit(delta_ts[mask_dt], feat_gamma_mag[mask_dt], 1)
                xs = np.sort(delta_ts[mask_dt])
                ax.plot(xs, np.poly1d(z)(xs), 'k--', alpha=0.55, lw=1.5, zorder=2)
                corr = np.corrcoef(delta_ts[mask_dt], feat_gamma_mag[mask_dt])[0, 1]
                ax.text(0.06, 0.93, f'$r = {corr:.3f}$',
                        transform=ax.transAxes, fontsize=13, va='top',
                        bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.85,
                                  edgecolor='#888888', linewidth=1.0))

            legend_el = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                       markeredgecolor='black', markersize=10, label=l)
                for c, l in [(WEAK_COLOR, 'Weak'), (MID_COLOR, 'Medium'), (STRONG_COLOR, 'Strong')]
            ]
            ax.legend(handles=legend_el, fontsize=10, loc='best', framealpha=0.9,
                      title='Shock Regime', title_fontsize=10)

        ax.set_xlabel(r'Timestep Size $\Delta t$ (s)')
        ax.set_ylabel(r'Mean $|\gamma_{\mathrm{feat}}|$')
        ax.set_title('FiLM Modulation vs. Timestep Size', fontweight='bold')
        plt.tight_layout()

    _save_standalone(fig, output_dir, 'panel_D_gamma_vs_dt')




# ===========================================================================
# GAMMA / BETA PCA  — separate analysis of scaling vs shifting
# ===========================================================================

def _run_pca_on_vectors(vecs):
    """
    Run PCA on a matrix of row vectors (n_sims x n_dims).
    Returns dict with test_pcs, ve (variance explained), evecs.
    """
    if len(vecs) < 3:
        return None
    centered = vecs - vecs.mean(0)
    cov = np.cov(centered.T)
    ev, evec = np.linalg.eigh(cov)
    idx = np.argsort(ev)[::-1]
    ev, evec = ev[idx], evec[:, idx]
    pcs = centered @ evec[:, :4]          # keep first 4 PCs
    ve  = ev[:4] / ev.sum() * 100
    return {'test_pcs': pcs, 've': ve, 'evecs': evec}


def _pca_r_block(pcs, ve, vals_dict, ax_r, title):
    """
    Heatmap of Pearson r between named parameter arrays and first 4 PCs.
    vals_dict: {label: array}
    """
    n_pcs = min(4, pcs.shape[1])
    labels = list(vals_dict.keys())
    r_mat  = np.full((len(labels), n_pcs), np.nan)

    for pi, lbl in enumerate(labels):
        v = vals_dict[lbl]
        mask = ~np.isnan(v)
        if mask.sum() < 3: continue
        for ci in range(n_pcs):
            r_mat[pi, ci] = np.corrcoef(v[mask], pcs[mask, ci])[0, 1]

    pc_lbls = ['PC$_%d$\n(%.1f%%)' % (i+1, ve[i]) for i in range(n_pcs)]
    im = ax_r.imshow(r_mat.T, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax_r.set_xticks(range(len(labels)))
    ax_r.set_xticklabels(labels, fontsize=10, rotation=35, ha='right')
    ax_r.set_yticks(range(n_pcs))
    ax_r.set_yticklabels(pc_lbls, fontsize=9)
    ax_r.set_title(title, fontweight='bold', fontsize=11)

    for pi in range(len(labels)):
        for ci in range(n_pcs):
            rv = r_mat[pi, ci]
            if not np.isnan(rv):
                ax_r.text(pi, ci, '%+.2f' % rv, ha='center', va='center',
                          fontsize=8.5,
                          color='white' if abs(rv) > 0.6 else 'black',
                          fontweight='bold' if abs(rv) > 0.85 else 'normal')

    from matplotlib.patches import Rectangle
    for ci in range(min(2, n_pcs)):
        best = int(np.nanargmax(np.abs(r_mat[:, ci])))
        ax_r.add_patch(Rectangle((best-0.5, ci-0.5), 1, 1,
                                  lw=2.5, edgecolor='gold',
                                  facecolor='none', zorder=5))
    return im, r_mat


def _scatter_pca_colored(ax, pcs, vals, cmap, train_pcs=None):
    """Scatter test PCs colored by vals, with optional train backdrop."""
    if train_pcs is not None:
        ax.scatter(train_pcs[:, 0], train_pcs[:, 1],
                   c='#e0e0e0', s=15, marker='x', alpha=0.3,
                   linewidths=0.5, zorder=1)
    mask = ~np.isnan(vals)
    sc = ax.scatter(pcs[mask, 0], pcs[mask, 1],
                    c=vals[mask], cmap=cmap,
                    s=65, edgecolors='black', linewidth=0.4,
                    alpha=0.9, zorder=3)
    return sc


def plot_standalone_gamma_beta_pca(all_acts, all_params, output_dir,
                                   train_acts=None, train_params=None):
    """
    Separate PCA on feature_gamma vectors and feature_beta vectors.
    2x3 grid:
      Row 1: gamma PCA colored by log(P/rho)=c², log(P*rho)=Z², delta_t
      Row 2: beta  PCA colored by the same three quantities
    Answers: does gamma track c, does beta track Z, or vice versa?
    """
    pressures = np.array([p.get('pressure', np.nan) for p in all_params])
    densities = np.array([p.get('density',  np.nan) for p in all_params])
    delta_ts  = np.array([p.get('delta_t',  np.nan) for p in all_params])

    with np.errstate(divide='ignore', invalid='ignore'):
        log_c2 = np.where((pressures>0)&(densities>0),
                          np.log10(pressures/densities), np.nan)
        log_Z2 = np.where((pressures>0)&(densities>0),
                          np.log10(pressures*densities), np.nan)

    color_cfg = [
        (log_c2,   'RdBu_r', r'$\log(P_0/\rho_0) \propto \log c^2$'),
        (log_Z2,   'PuOr',   r'$\log(P_0{\cdot}\rho_0) \propto \log Z^2$'),
        (delta_ts, 'plasma', r'$\Delta t$  (CFL proxy for $c$)'),
    ]

    gamma_vecs = np.array([a['feature_gamma'] for a in all_acts])
    beta_vecs  = np.array([a['feature_beta']  for a in all_acts])

    pca_g = _run_pca_on_vectors(gamma_vecs)
    pca_b = _run_pca_on_vectors(beta_vecs)
    if pca_g is None or pca_b is None:
        print('  warn: too few sims for gamma/beta PCA'); return

    train_g = train_b = None
    if train_acts:
        tg = np.array([a['feature_gamma'] for a in train_acts])
        tb = np.array([a['feature_beta']  for a in train_acts])
        cg = tg - gamma_vecs.mean(0)
        cb = tb - beta_vecs.mean(0)
        train_g = cg @ pca_g['evecs'][:, :2]
        train_b = cb @ pca_b['evecs'][:, :2]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(
            r'Separate PCA on FiLM $\gamma$ (scaling) and $\beta$ (shifting) Vectors' + '\n'
            r'Do $\gamma$ and $\beta$ track different physical quantities?',
            fontsize=13, fontweight='bold', y=1.02)

        row_cfg = [
            (pca_g, train_g, r'$\gamma$ (feature scaling)'),
            (pca_b, train_b, r'$\beta$ (feature shifting)'),
        ]

        for row, (pca, train_pcs, row_label) in enumerate(row_cfg):
            ve = pca['ve']
            pcs = pca['test_pcs']
            for col, (vals, cmap, col_label) in enumerate(color_cfg):
                ax = axes[row, col]
                sc = _scatter_pca_colored(ax, pcs, vals, cmap, train_pcs)
                cbar = plt.colorbar(sc, ax=ax, shrink=0.88, pad=0.02)
                cbar.set_label(col_label, fontsize=9)
                cbar.ax.tick_params(labelsize=8)

                # r values for PC1 and PC2
                mask = ~np.isnan(vals)
                for pc_idx, pc_tag in [(0, 'PC_1'), (1, 'PC_2')]:
                    v = pcs[mask, pc_idx]
                    r = np.corrcoef(vals[mask], v)[0, 1] if mask.sum() > 2 else np.nan
                    weight = 'bold' if abs(r) > 0.85 else 'normal'
                    color  = '#C62828' if abs(r) > 0.85 else '#444444'
                    ax.text(0.97, 0.97 - pc_idx * 0.12,
                            '$r=%+.2f$' % r,
                            transform=ax.transAxes, fontsize=10,
                            va='top', ha='right', fontweight=weight, color=color,
                            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                      alpha=0.9,
                                      edgecolor='#C62828' if abs(r) > 0.85 else '#cccccc',
                                      linewidth=1.4 if abs(r) > 0.85 else 0.7))

                _format_pca_axes(ax, ve)
                if row == 0:
                    ax.set_title(col_label, fontweight='bold', fontsize=11)
                # Row label on leftmost column
                if col == 0:
                    ax.set_ylabel(row_label + '\nPC$_2$ (%.1f%% var.)' % ve[1],
                                  fontsize=11)

        plt.tight_layout()

    _save_standalone(fig, output_dir, 'panel_E1_gamma_beta_pca')


def plot_standalone_gamma_beta_r2(all_acts, all_params, output_dir):
    """
    Side-by-side R² heatmaps for gamma vs beta vectors.
    Cleanly shows which physical quantity each FiLM component tracks.
    """
    pressures = np.array([p.get('pressure', np.nan) for p in all_params])
    densities = np.array([p.get('density',  np.nan) for p in all_params])
    delta_ts  = np.array([p.get('delta_t',  np.nan) for p in all_params])

    with np.errstate(divide='ignore', invalid='ignore'):
        log_c2 = np.where((pressures>0)&(densities>0),
                          np.log10(pressures/densities), np.nan)
        log_Z2 = np.where((pressures>0)&(densities>0),
                          np.log10(pressures*densities), np.nan)

    vals_dict = {
        '$P_0$':                   pressures,
        r'$\rho_0$':               densities,
        r'$\Delta t$':             delta_ts,
        r'$\log(P_0/\rho_0)$':    log_c2,
        r'$\log(P_0{\cdot}\rho_0)$': log_Z2,
    }

    gamma_vecs = np.array([a['feature_gamma'] for a in all_acts])
    beta_vecs  = np.array([a['feature_beta']  for a in all_acts])
    pca_g = _run_pca_on_vectors(gamma_vecs)
    pca_b = _run_pca_on_vectors(beta_vecs)
    if pca_g is None or pca_b is None:
        print('  warn: too few sims for gamma/beta R2'); return

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                                 gridspec_kw={'width_ratios': [1, 1, 0.08]})
        fig.suptitle(
            r'FiLM $\gamma$ vs $\beta$: Which Physical Quantity Does Each Track?' + '\n'
            r'Highlighted cell = strongest correlation per PC',
            fontsize=13, fontweight='bold', y=1.04)

        im_g, _ = _pca_r_block(pca_g['test_pcs'], pca_g['ve'], vals_dict,
                                axes[0], r'$\gamma$ (Feature Scaling)')
        im_b, _ = _pca_r_block(pca_b['test_pcs'], pca_b['ve'], vals_dict,
                                axes[1], r'$\beta$ (Feature Shifting)')
        axes[1].set_ylabel('')   # suppress duplicate y-label

        # Shared colorbar
        plt.colorbar(im_g, cax=axes[2], label='Pearson $r$')

        plt.tight_layout()

    _save_standalone(fig, output_dir, 'panel_E2_gamma_beta_r2')


def plot_standalone_division_of_labor(all_acts, all_params, output_dir):
    """
    Summary figure: direct scatter of PC1 coordinate vs log(c²) and
    PC1 coordinate vs log(Z²) for BOTH gamma and beta, in a 2x2 grid.
    This is the clearest way to show division of labor — one plot per
    (gamma/beta) x (c/Z) combination with r annotated prominently.
    """
    pressures = np.array([p.get('pressure', np.nan) for p in all_params])
    densities = np.array([p.get('density',  np.nan) for p in all_params])

    with np.errstate(divide='ignore', invalid='ignore'):
        log_c2 = np.where((pressures>0)&(densities>0),
                          np.log10(pressures/densities), np.nan)
        log_Z2 = np.where((pressures>0)&(densities>0),
                          np.log10(pressures*densities), np.nan)

    gamma_vecs = np.array([a['feature_gamma'] for a in all_acts])
    beta_vecs  = np.array([a['feature_beta']  for a in all_acts])
    pca_g = _run_pca_on_vectors(gamma_vecs)
    pca_b = _run_pca_on_vectors(beta_vecs)
    if pca_g is None or pca_b is None:
        print('  warn: too few sims for division of labor'); return

    # 2x2: rows = gamma/beta, cols = log_c2/log_Z2
    row_cfg  = [(pca_g, r'$\gamma$ PC'), (pca_b, r'$\beta$ PC')]
    col_cfg  = [
        (log_c2, r'$\log(P_0/\rho_0) \propto \log c^2$',   '#E53935', r'PC$_1$'),
        (log_Z2, r'$\log(P_0{\cdot}\rho_0) \propto \log Z^2$', '#1E88E5', r'PC$_2$'),
    ]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        fig.suptitle(
            r'Division of Labor: Does $\gamma$ Scale with $c$, $\beta$ Shift with $Z$?' + '\n'
            r'Each panel: scatter of FiLM vector PC coordinate vs physical quantity',
            fontsize=13, fontweight='bold', y=1.02)

        for row, (pca, row_label) in enumerate(row_cfg):
            for col, (vals, xlabel, color, pc_label) in enumerate(col_cfg):
                ax = axes[row, col]
                # Use the PC index matching the hypothesis
                pc_idx = col   # PC1 for c (col 0), PC2 for Z (col 1)
                mask = ~np.isnan(vals)
                x = vals[mask]
                y = pca['test_pcs'][mask, pc_idx]

                ax.scatter(x, y, c=color, s=55, alpha=0.75,
                           edgecolors='black', linewidth=0.4)

                if mask.sum() > 2:
                    z = np.polyfit(x, y, 1)
                    xs = np.sort(x)
                    ax.plot(xs, np.poly1d(z)(xs), 'k--', alpha=0.5, lw=1.5)
                    r = np.corrcoef(x, y)[0, 1]
                    weight = 'bold' if abs(r) > 0.85 else 'normal'
                    ax.text(0.05, 0.93, '$r = %+.3f$' % r,
                            transform=ax.transAxes, fontsize=13,
                            va='top', fontweight=weight,
                            color='#C62828' if abs(r) > 0.85 else '#333333',
                            bbox=dict(boxstyle='round,pad=0.35', fc='white',
                                      alpha=0.92,
                                      edgecolor='#C62828' if abs(r) > 0.85 else '#aaaaaa',
                                      linewidth=1.5 if abs(r) > 0.85 else 0.8))

                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel('%s$_%d$ coordinate' % (row_label, pc_idx+1), fontsize=11)
                ax.set_title(
                    '%s  vs  %s' % (row_label + ('$_%d$' % (pc_idx+1)), pc_label),
                    fontweight='bold', fontsize=11)
                ax.xaxis.set_major_locator(plt.MaxNLocator(5, prune='both'))
                ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune='both'))
                ax.tick_params(axis='x', labelrotation=20)

        plt.tight_layout()

    _save_standalone(fig, output_dir, 'panel_E3_division_of_labor')


# ===========================================================================
# FIGURE 5: Presentation summary panel (2x2)  +  standalone exports
# ===========================================================================

def plot_summary_panel(acts_by_regime, all_acts, all_params, output_dir,
                       train_acts=None, train_params=None):
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

    # --- TL: Train vs Test parameter space coverage ---
    ax = fig.add_subplot(gs[0, 0])
    if has_train:
        tm = ~(np.isnan(train_pressures) | np.isnan(train_densities))
        if tm.any():
            ax.scatter(train_pressures[tm], train_densities[tm], c='#90CAF9', s=40,
                       marker='x', alpha=0.6, linewidths=1.0,
                       label='Train (n=%d)' % tm.sum())
    mask = ~(np.isnan(pressures) | np.isnan(densities))
    if mask.any():
        ax.scatter(pressures[mask], densities[mask], c='#E53935', s=50,
                   edgecolors='black', linewidth=0.5, alpha=0.8,
                   label='Test (n=%d)' % mask.sum())
    ax.set_xlabel('Initial Pressure $P_0$')
    ax.set_ylabel(r'Initial Density $\rho_0$')
    ax.set_title('Train vs Test: Parameter Space Coverage', fontweight='bold', fontsize=11)
    ax.text(0.5, -0.19, 'Test samples span and extend training distribution',
            transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic', color='#555555')
    ax.legend(fontsize=9, loc='center left')

    # --- TR: Joint P0 vs rho0 colored by FiLM gamma ---
    ax = fig.add_subplot(gs[0, 1])
    mask = ~(np.isnan(pressures) | np.isnan(densities))
    if mask.any():
        sc = ax.scatter(pressures[mask], densities[mask], c=feat_gamma_mag[mask],
                        cmap='magma', s=60, edgecolors='black', linewidth=0.5, alpha=0.85)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label(r'Mean $|\gamma_{feat}|$', fontsize=10)
    ax.set_xlabel('Initial Pressure $P_0$')
    ax.set_ylabel(r'Initial Density $\rho_0$')
    ax.set_title(r'FiLM $\gamma$ in Joint Parameter Space', fontweight='bold', fontsize=11)
    ax.text(0.5, -0.19, 'Brighter = stronger feature modulation',
            transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic', color='#555555')

    # --- BL: Embedding PCA (train + test) ---
    ax = fig.add_subplot(gs[1, 0])
    pca = _compute_pca(all_acts, all_params, train_acts, train_params)
    test_pressures = pressures
    if pca is not None:
        ve = pca['ve']
        if pca['has_train']:
            ax.scatter(pca['train_pcs'][:, 0], pca['train_pcs'][:, 1],
                       c=pca['train_p'], cmap='coolwarm', s=30, marker='x',
                       alpha=0.5, linewidths=0.8, label='Train')
            sc = ax.scatter(pca['test_pcs'][:, 0], pca['test_pcs'][:, 1],
                            c=test_pressures, cmap='coolwarm', s=50,
                            edgecolors='black', linewidth=0.4, alpha=0.8, label='Test')
            ax.legend(fontsize=8, loc='upper left')
        else:
            sc = ax.scatter(pca['test_pcs'][:, 0], pca['test_pcs'][:, 1],
                            c=test_pressures, cmap='coolwarm', s=50,
                            edgecolors='black', linewidth=0.4, alpha=0.8)
        plt.colorbar(sc, ax=ax, label='$P_0$', shrink=0.85)
        ax.set_xlabel('PC1 (%.1f%%)' % ve[0])
        ax.set_ylabel('PC2 (%.1f%%)' % ve[1])
    if has_train:
        ax.set_title('Embedding Space (Train + Test)', fontweight='bold', fontsize=11)
        ax.text(0.5, -0.19, 'Train (x) and test (o) co-locate by regime',
                transform=ax.transAxes, ha='center', fontsize=9,
                fontstyle='italic', color='#555555')
    else:
        ax.set_title('Global Embedding Space (PCA)', fontweight='bold', fontsize=11)
        ax.text(0.5, -0.19, 'Distinct clusters separate shock regimes',
                transform=ax.transAxes, ha='center', fontsize=9,
                fontstyle='italic', color='#555555')

    # --- BR: FiLM gamma vs Delta-t ---
    ax = fig.add_subplot(gs[1, 1])
    mask_dt = ~np.isnan(delta_ts)
    if mask_dt.any():
        p_valid = pressures[~np.isnan(pressures)]
        p33 = np.percentile(p_valid, 33) if len(p_valid) > 0 else 0
        p67 = np.percentile(p_valid, 67) if len(p_valid) > 0 else 1
        dt_colors = []
        for p in pressures:
            if np.isnan(p):    dt_colors.append('#999999')
            elif p <= p33:     dt_colors.append(WEAK_COLOR)
            elif p >= p67:     dt_colors.append(STRONG_COLOR)
            else:              dt_colors.append(MID_COLOR)
        cm = [dt_colors[i] for i in range(len(dt_colors)) if mask_dt[i]]
        ax.scatter(delta_ts[mask_dt], feat_gamma_mag[mask_dt], c=cm, s=50,
                   edgecolors='black', linewidth=0.5, alpha=0.8)
        if mask_dt.sum() > 2:
            z = np.polyfit(delta_ts[mask_dt], feat_gamma_mag[mask_dt], 1)
            xs = np.sort(delta_ts[mask_dt])
            ax.plot(xs, np.poly1d(z)(xs), 'k--', alpha=0.5, lw=1.2)
            corr = np.corrcoef(delta_ts[mask_dt], feat_gamma_mag[mask_dt])[0, 1]
            ax.text(0.05, 0.92, 'r = %.3f' % corr, transform=ax.transAxes,
                    fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'Mean $|\gamma_{feat}|$')
    ax.set_title(r'FiLM $\gamma$ vs Timestep Size', fontweight='bold', fontsize=11)
    ax.text(0.5, -0.19, r'No dependence on numerical resolution ($r \approx 0$)',
            transform=ax.transAxes, ha='center', fontsize=9,
            fontstyle='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'film_summary_panel.{ext}')
    plt.close(fig)
    print('  ok film_summary_panel.png/pdf')

    # ------------------------------------------------------------------
    # Standalone publication PDFs for each of the four sub-panels
    # ------------------------------------------------------------------
    standalone_dir = output_dir / 'standalone_panels'
    standalone_dir.mkdir(exist_ok=True)
    print('\n  Generating standalone panel PDFs -> standalone_panels/')
    plot_standalone_param_coverage(all_acts, all_params, standalone_dir,
                                   train_acts=train_acts, train_params=train_params)
    plot_standalone_gamma_joint(all_acts, all_params, standalone_dir)
    plot_standalone_embedding_pca(all_acts, all_params, standalone_dir,
                                  train_acts=train_acts, train_params=train_params)
    plot_standalone_pca_decomposed(all_acts, all_params, standalone_dir,
                                   train_acts=train_acts, train_params=train_params)
    plot_standalone_pca_derived(all_acts, all_params, standalone_dir,
                                train_acts=train_acts, train_params=train_params)
    plot_standalone_pca_r2_summary(all_acts, all_params, standalone_dir,
                                   train_acts=train_acts, train_params=train_params)
    plot_standalone_gamma_vs_dt(all_acts, all_params, standalone_dir)
    plot_standalone_gamma_beta_pca(all_acts, all_params, standalone_dir)
    plot_standalone_gamma_beta_r2(all_acts, all_params, standalone_dir)
    plot_standalone_division_of_labor(all_acts, all_params, standalone_dir)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="FiLM Activation Analysis")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Training data dir for PCA overlay")
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
            except Exception:
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
        plot_summary_panel(abr, all_acts, all_params, output_dir,
                           train_acts=train_acts, train_params=train_params)
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
            var: {
                'mean': float(np.mean([a['derivative_gamma'][vi] for a in all_acts])),
                'std':  float(np.std( [a['derivative_gamma'][vi] for a in all_acts])),
            }
            for vi, var in enumerate(DYN_NAMES)
        },
        'derivative_beta_stats': {
            var: {
                'mean': float(np.mean([a['derivative_beta'][vi] for a in all_acts])),
                'std':  float(np.std( [a['derivative_beta'][vi] for a in all_acts])),
            }
            for vi, var in enumerate(DYN_NAMES)
        },
    }
    with open(output_dir / 'film_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n  ok film_analysis_summary.json')
    print(f"\nDone! All figures saved to {output_dir}")
    print(f"  Standalone panel PDFs: {output_dir}/standalone_panels/")


if __name__ == "__main__":
    main()