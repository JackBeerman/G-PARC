#!/usr/bin/env python3
"""
Compare how G-PARC (FiLM) vs MeshGraphKAN use global simulation parameters.

G-PARC: Dedicated GlobalParameterProcessor MLP -> 64-dim embedding -> FiLM gamma/beta
MeshGraphKAN: Concatenate [static, dynamic, global_params] -> KAN encoder -> 128-dim latent

Usage:
    python mgkan_compare.py \
        --test_dir /path/to/test_cases_normalized \
        --train_dir /path/to/train_cases_normalized \
        --gparc_ckpt /path/to/gparc/best_model.pth \
        --mgkan_ckpt /path/to/mgkan/best_model.pth \
        --output_dir ./film_vs_mgkan --device cuda
"""
import argparse, sys, os, json, re
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.shockchord.eval_comparison import (
    NUM_STATIC, NUM_USED_DYNAMIC, SKIP_INDICES, RAW_DYNAMIC, KEEP_INDICES,
    extract_global_params_from_data, parse_params_from_filename,
    load_model_gparcv2,
)

def ensure_global_attrs(data):
    if hasattr(data, 'global_params') and data.global_params.numel() >= 3:
        gp = data.global_params
        if not hasattr(data, 'global_pressure'):
            data.global_pressure = gp[0].unsqueeze(0)
            data.global_density = gp[1].unsqueeze(0)
            data.global_delta_t = gp[2].unsqueeze(0)

def get_params(data, fname):
    params = extract_global_params_from_data(data)
    pf, rf = parse_params_from_filename(fname)
    if pf is not None: params['pressure'] = pf
    if rf is not None: params['density'] = rf
    return params

# =====================================================================
# G-PARC loader + embedding extractor
# =====================================================================
def load_gparc(ckpt_path, sample_data, device):
    return load_model_gparcv2(ckpt_path, sample_data, device)

def extract_gparc(model, sim_data, device):
    d = sim_data[0]
    with torch.no_grad():
        global_attrs = model._extract_global_attrs(d).unsqueeze(0).to(device)
        emb = model.global_processor(global_attrs).detach().cpu().numpy().flatten()
        diff = model.derivative_solver
        gamma, beta = diff.feature_norm.generate_gamma_beta(global_attrs)
        gamma = gamma.detach().cpu().numpy().flatten()
    return {'embedding': emb, 'gamma_mag': float(np.mean(np.abs(gamma)))}

# =====================================================================
# MeshGraphKAN loader + encoder extractor
# =====================================================================
def load_mgkan(ckpt_path, sample_data, device):
    from models.meshgraphkan import MeshGraphKAN, MeshGraphKANShocktubeRollout
    ckpt_dir = Path(ckpt_path).parent
    config = json.load(open(ckpt_dir/"config.json")) if (ckpt_dir/"config.json").exists() else {}
    base = MeshGraphKAN(input_dim_nodes=8, input_dim_edges=3, output_dim=NUM_USED_DYNAMIC,
        processor_size=config.get('processor_size',15),
        hidden_dim_processor=config.get('hidden_dim_processor',128),
        num_harmonics=config.get('num_harmonics',5))
    wrapper = MeshGraphKANShocktubeRollout(model=base, num_static_feats=NUM_STATIC,
        num_dynamic_feats=NUM_USED_DYNAMIC, skip_dynamic_indices=SKIP_INDICES, global_param_dim=3)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    wrapper.load_state_dict(ckpt.get('model_state_dict', ckpt)); wrapper.to(device).eval()
    return wrapper

def extract_mgkan(wrapper, sim_data, device):
    d = sim_data[0]
    sf = wrapper.num_static_feats
    current = wrapper._extract_dynamic(d.x.to(device))
    gf = wrapper._extract_global_params(d)
    nf = torch.cat([d.x[:, :sf].to(device), current, gf.to(device)], -1)
    with torch.no_grad():
        encoded = wrapper.model.node_encoder(nf)  # [N, 128]
        mean_enc = encoded.mean(dim=0).cpu().numpy()  # [128]
    return {'encoder_output': mean_enc, 'encoder_std': float(np.std(mean_enc))}

# =====================================================================
# PLOTTING
# =====================================================================
def do_pca(embs):
    c = embs - embs.mean(0)
    ev, evec = np.linalg.eigh(np.cov(c.T))
    idx = np.argsort(ev)[::-1]; evec, ev = evec[:, idx], ev[idx]
    return c @ evec[:, :2], ev[:2] / ev.sum() * 100

def plot_comparison(gp_test, mk_test, test_params, output_dir,
                    gp_train=None, mk_train=None, train_params=None):
    fig = plt.figure(figsize=(16, 6.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.32)
    fig.suptitle('How Models Use Global Simulation Parameters',
                 fontsize=16, fontweight='bold', y=1.01)
    test_p = np.array([p.get('pressure', np.nan) for p in test_params])
    has_train = gp_train is not None and len(gp_train) > 0
    if has_train:
        tr_p = np.array([p.get('pressure', np.nan) for p in train_params])

    # --- Left: G-PARC embedding PCA ---
    ax = fig.add_subplot(gs[0, 0])
    embs = np.array([d['embedding'] for d in gp_test])
    if has_train:
        tr_embs = np.array([d['embedding'] for d in gp_train])
        comb = np.vstack([tr_embs, embs])
    else:
        comb = embs
    pcs, ve = do_pca(comb)
    if has_train:
        nt = len(tr_embs)
        ax.scatter(pcs[:nt, 0], pcs[:nt, 1], c=tr_p, cmap='coolwarm',
                   s=30, marker='x', alpha=0.4, linewidths=0.8, label='Train')
        sc = ax.scatter(pcs[nt:, 0], pcs[nt:, 1], c=test_p, cmap='coolwarm',
                        s=50, edgecolors='black', linewidth=0.4, alpha=0.8, label='Test')
        ax.legend(fontsize=8, loc='upper left')
    else:
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=test_p, cmap='coolwarm',
                        s=50, edgecolors='black', linewidth=0.4, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='$P_0$ (Pa)', shrink=0.85)
    ax.set_xlabel('PC1 (%.1f%%)' % ve[0]); ax.set_ylabel('PC2 (%.1f%%)' % ve[1])
    ax.set_title('G-PARC: FiLM Embedding', fontweight='bold', fontsize=12)
    ax.text(0.5, -0.18, 'Dedicated 64-dim parameter processor\nClear regime separation',
            transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic', color='#555555')

    # --- Center: MeshGraphKAN encoder PCA ---
    ax = fig.add_subplot(gs[0, 1])
    encs = np.array([d['encoder_output'] for d in mk_test])
    if has_train:
        tr_encs = np.array([d['encoder_output'] for d in mk_train])
        comb = np.vstack([tr_encs, encs])
    else:
        comb = encs
    pcs, ve = do_pca(comb)
    if has_train:
        nt = len(tr_encs)
        ax.scatter(pcs[:nt, 0], pcs[:nt, 1], c=tr_p, cmap='coolwarm',
                   s=30, marker='x', alpha=0.4, linewidths=0.8, label='Train')
        sc = ax.scatter(pcs[nt:, 0], pcs[nt:, 1], c=test_p, cmap='coolwarm',
                        s=50, edgecolors='black', linewidth=0.4, alpha=0.8, label='Test')
        ax.legend(fontsize=8, loc='upper left')
    else:
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=test_p, cmap='coolwarm',
                        s=50, edgecolors='black', linewidth=0.4, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='$P_0$ (Pa)', shrink=0.85)
    ax.set_xlabel('PC1 (%.1f%%)' % ve[0]); ax.set_ylabel('PC2 (%.1f%%)' % ve[1])
    ax.set_title('MeshGraphKAN: Node Encoder', fontweight='bold', fontsize=12)
    ax.text(0.5, -0.18, 'Global params concatenated with node features\n128-dim KAN encoder output (mean-pooled)',
            transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic', color='#555555')

    # --- Right: Pressure sensitivity comparison ---
    ax = fig.add_subplot(gs[0, 2])
    gp_gamma = np.array([d['gamma_mag'] for d in gp_test])
    mk_std = np.array([d['encoder_std'] for d in mk_test])
    # Normalize both to [0,1]
    def norm01(x): return (x - x.min()) / (x.max() - x.min() + 1e-12)
    gn, mn = norm01(gp_gamma), norm01(mk_std)
    valid = ~np.isnan(test_p)
    ax.scatter(test_p[valid], gn[valid], c='#E53935', s=50, alpha=0.7,
               edgecolors='black', linewidth=0.4, label=r'G-PARC: FiLM $|\gamma|$')
    ax.scatter(test_p[valid], mn[valid], c='#1E88E5', s=50, alpha=0.7,
               edgecolors='black', linewidth=0.4, marker='s', label='MGKAN: encoder variability')
    r_gp = np.corrcoef(test_p[valid], gp_gamma[valid])[0,1]
    r_mk = np.corrcoef(test_p[valid], mk_std[valid])[0,1]
    ax.text(0.05, 0.92, f'G-PARC r = {r_gp:.3f}', transform=ax.transAxes,
            fontsize=9, color='#E53935', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    ax.text(0.05, 0.82, f'MGKAN r = {r_mk:.3f}', transform=ax.transAxes,
            fontsize=9, color='#1E88E5', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    ax.set_xlabel('Initial Pressure $P_0$ (Pa)')
    ax.set_ylabel('Normalized Response')
    ax.set_title('Pressure Sensitivity', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, loc='lower right')
    ax.text(0.5, -0.18, 'FiLM provides explicit regime conditioning\nvs implicit feature mixing',
            transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'film_vs_mgkan.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  saved film_vs_mgkan.png/pdf')

# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--gparc_ckpt", type=str, required=True)
    parser.add_argument("--mgkan_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./film_vs_mgkan")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_sims", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    test_dir = Path(args.test_dir)
    sim_files = sorted(test_dir.glob("*.pt"))
    if args.max_sims: sim_files = sim_files[:args.max_sims]
    print(f"Test sims: {len(sim_files)}")

    sample = torch.load(sim_files[0], weights_only=False)
    sd0 = sample[0]
    if not hasattr(sd0, 'pos') or sd0.pos is None: sd0.pos = sd0.x[:, :NUM_STATIC]
    sd0 = sd0.to(device)
    for d in sample: d.x=d.x.to(device); d.edge_index=d.edge_index.to(device); ensure_global_attrs(d)

    print("Loading G-PARC..."); gparc = load_gparc(args.gparc_ckpt, sd0, device)
    print(f"  {sum(p.numel() for p in gparc.parameters()):,} params")
    print("Loading MeshGraphKAN..."); mgkan = load_mgkan(args.mgkan_ckpt, sd0, device)
    print(f"  {sum(p.numel() for p in mgkan.parameters()):,} params")

    gp_test, mk_test, test_params = [], [], []
    for sf in sim_files:
        sd = torch.load(sf, weights_only=False)
        for d in sd: d.x=d.x.to(device); d.edge_index=d.edge_index.to(device); ensure_global_attrs(d)
        params = get_params(sd[0], sf.stem)
        gp_test.append(extract_gparc(gparc, sd, device))
        mk_test.append(extract_mgkan(mgkan, sd, device))
        test_params.append(params)
    print(f"Extracted {len(test_params)} test sims")

    gp_train, mk_train, train_params = [], [], []
    if args.train_dir:
        tr_files = sorted(Path(args.train_dir).glob("*.pt"))
        if args.max_sims: tr_files = tr_files[:args.max_sims]
        print(f"Extracting {len(tr_files)} train sims...")
        for sf in tr_files:
            try:
                sd = torch.load(sf, weights_only=False)
                for d in sd: d.x=d.x.to(device); d.edge_index=d.edge_index.to(device); ensure_global_attrs(d)
                params = get_params(sd[0], sf.stem)
                gp_train.append(extract_gparc(gparc, sd, device))
                mk_train.append(extract_mgkan(mgkan, sd, device))
                train_params.append(params)
            except: pass
        print(f"  Extracted {len(train_params)} train sims")

    plot_comparison(gp_test, mk_test, test_params, output_dir,
                    gp_train=gp_train, mk_train=mk_train, train_params=train_params)
    print("Done!")

if __name__ == "__main__":
    main()