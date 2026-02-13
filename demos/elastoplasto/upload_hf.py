#!/usr/bin/env python3
"""
upload_to_hf.py — Stage and upload G-PARC artifacts to Hugging Face Hub.

Usage:
    1. pip install huggingface_hub --user
    2. huggingface-cli login   (paste your write token)
    3. python upload_to_hf.py

This script will:
    - Create the repo if it doesn't exist (private by default)
    - Copy files into a staging directory
    - Upload everything to Hugging Face
"""

import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ============================================================
# CONFIGURE THESE
# ============================================================
HF_USERNAME = "jacktbeerman"          # ← Your Hugging Face username
REPO_NAME = "Gparc"             # ← Repo name (change if you want)
PRIVATE = True                         # ← Set to False when ready to go public

REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

# ============================================================
# SOURCE PATHS ON HPC — adjust if needed
# ============================================================

# Model checkpoints
CHECKPOINTS = {
    "checkpoints/gparcv1_best.pth":     "/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/best_model.pth",
    "checkpoints/gparcv2_best.pth":     "/scratch/jtb3sud/elasto_graphconv_V2/2hop/best_model.pth",
    "checkpoints/meshgraphkan_best.pth": "/scratch/jtb3sud/elasto_meshgraphkan/run1/best_model.pth",
}

# Config files
CONFIGS = {
    "configs/gparcv1_config.json":      "/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/config.json",
    "configs/gparcv2_config.json":      "/scratch/jtb3sud/elasto_graphconv_V2/2hop/config.json",
    "configs/meshgraphkan_config.json": "/scratch/jtb3sud/elasto_meshgraphkan/run1/config.json",
}

# Training histories
HISTORIES = {
    "training_histories/gparcv1_history.json":      "/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/training_history.json",
    "training_histories/gparcv2_history.json":      "/scratch/jtb3sud/elasto_graphconv_V2/2hop/training_history.json",
    "training_histories/meshgraphkan_history.json": "/scratch/jtb3sud/elasto_meshgraphkan/run1/training_history.json",
}

# Normalization stats
NORM_STATS = {
    "data/normalization_stats.json": "/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/normalization_stats.json",
}

# Test data directory (all .pt files will be uploaded)
TEST_DIR = Path("/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/test")

# ============================================================
# STAGING
# ============================================================
STAGING_DIR = Path("/scratch/jtb3sud/hf_upload_staging")

def stage_files():
    """Copy all files into staging directory."""
    # Clean previous staging
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    
    all_files = {**CHECKPOINTS, **CONFIGS, **HISTORIES, **NORM_STATS}
    
    print("Staging files...")
    for dest_rel, source in all_files.items():
        source_path = Path(source)
        dest_path = STAGING_DIR / dest_rel
        
        if not source_path.exists():
            print(f"  ✗ MISSING: {source}")
            continue
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        size_mb = source_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {dest_rel} ({size_mb:.1f} MB)")
    
    # Stage test data
    test_dest = STAGING_DIR / "data" / "test"
    test_dest.mkdir(parents=True, exist_ok=True)
    
    test_files = sorted(TEST_DIR.glob("*.pt"))
    print(f"\nStaging {len(test_files)} test simulations...")
    for f in test_files:
        shutil.copy2(f, test_dest / f.name)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  ✓ data/test/{f.name} ({size_mb:.1f} MB)")
    
    # Create a README
    readme = STAGING_DIR / "README.md"
    readme.write_text(f"""---
license: mit
tags:
  - physics-ml
  - graph-neural-networks
  - computational-mechanics
  - elastoplastic
---

# G-PARC: Graph Physics-Aware Recurrent Convolutions

Model weights, test data, and configuration files for the G-PARC elastoplastic simulation paper.

## Contents

```
checkpoints/          # Trained model weights (.pth)
  gparcv1_best.pth    # G-PARCv1 (GNN baseline)
  gparcv2_best.pth    # G-PARCv2 (MLS + numerical integration) 
  meshgraphkan_best.pth  # MeshGraphKAN (NVIDIA architecture)

configs/              # Training configuration files
data/
  normalization_stats.json   # Global max normalization parameters
  test/                      # PLAID test simulations (.pt)

training_histories/   # Loss curves per epoch
```

## Dataset

PLAID 2D Elasto-Plasto-Dynamics benchmark — high-velocity impact on steel plates.

- **Variables**: Displacement field (U_x, U_y)
- **Normalization**: Global max
- **Meshes**: Unstructured quad elements

## Usage

```python
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("{REPO_ID}", "checkpoints/gparcv2_best.pth")
```

## Code

Full training and evaluation code: [GitHub repo link here]
""")
    print(f"\n✓ Created README.md")
    
    # Total size
    total_size = sum(f.stat().st_size for f in STAGING_DIR.rglob("*") if f.is_file())
    print(f"\nTotal staged: {total_size / (1024**2):.1f} MB")
    print(f"Staging directory: {STAGING_DIR}")


def upload():
    """Create repo and upload staged files."""
    api = HfApi()
    
    # Create repo (no-op if it already exists)
    print(f"\nCreating repo: {REPO_ID} (private={PRIVATE})")
    try:
        create_repo(REPO_ID, private=PRIVATE, exist_ok=True)
        print(f"  ✓ Repo ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"  Repo creation: {e}")
    
    # Upload
    print(f"\nUploading from {STAGING_DIR}...")
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(STAGING_DIR),
        commit_message="Upload G-PARC model weights, test data, and configs",
    )
    
    print(f"\n{'='*60}")
    print(f"✅ Upload complete!")
    print(f"   https://huggingface.co/{REPO_ID}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print("G-PARC → Hugging Face Upload")
    print("=" * 60)
    print(f"Repo: {REPO_ID}")
    print(f"Private: {PRIVATE}")
    print()
    
    stage_files()
    
    response = input("\nProceed with upload? [y/N] ")
    if response.lower() == 'y':
        upload()
    else:
        print("Upload cancelled. Files are staged at:", STAGING_DIR)