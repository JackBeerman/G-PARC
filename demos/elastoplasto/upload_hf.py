#!/usr/bin/env python3
"""
upload_to_hf.py — Stage and upload G-PARC artifacts to Hugging Face Hub.
"""

import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "jacktbeerman"
REPO_NAME = "Gparc"
PRIVATE = True
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

CHECKPOINTS = {
    "checkpoints/gparcv1_best.pth":      "/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/best_model.pth",
    "checkpoints/gparcv2_best.pth":      "/scratch/jtb3sud/elasto_graphconv_V2/2hop/best_model.pth",
    "checkpoints/meshgraphkan_best.pth": "/scratch/jtb3sud/delta/elasto/best_model.pth",
    "checkpoints/meshgraphnet_best.pth": "/scratch/jtb3sud/meshgraphnet/elasto/run1/best_model.pt",
}

CONFIGS = {
    "configs/gparcv1_config.json":      "/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/config.json",
    "configs/gparcv2_config.json":      "/scratch/jtb3sud/elasto_graphconv_V2/2hop/config.json",
    "configs/meshgraphkan_config.json": "/scratch/jtb3sud/delta/elasto/config.json",
    "configs/meshgraphnet_config.json": "/scratch/jtb3sud/meshgraphnet/elasto/run1/config.json",
}

# No training history saved for MeshGraphNet
HISTORIES = {
    "training_histories/gparcv1_history.json":      "/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/training_history.json",
    "training_histories/gparcv2_history.json":      "/scratch/jtb3sud/elasto_graphconv_V2/2hop/training_history.json",
    "training_histories/meshgraphkan_history.json": "/scratch/jtb3sud/delta/elasto/training_history.json",
}

NORM_STATS = {
    "data/normalization_stats.json": "/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/normalization_stats.json",
}

TEST_DIR = Path("/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/test")
STAGING_DIR = Path("/scratch/jtb3sud/hf_upload_staging")

def stage_files():
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
    
    test_dest = STAGING_DIR / "data" / "test"
    test_dest.mkdir(parents=True, exist_ok=True)
    test_files = sorted(TEST_DIR.glob("*.pt"))
    print(f"\nStaging {len(test_files)} test simulations...")
    for f in test_files:
        shutil.copy2(f, test_dest / f.name)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  ✓ data/test/{f.name} ({size_mb:.1f} MB)")
    
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

## Models

| Model | Description |
|-------|-------------|
| G-PARCv1 | Graph Physics-Aware Recurrent Convolutions — fully learned GNN operators |
| G-PARCv2 | MLS differential operators + numerical Euler integration |
| MeshGraphKAN | Kolmogorov-Arnold Network message passing with Fourier basis |
| MeshGraphNet | Standard encode-process-decode GNN (Pfaff et al., 2021) |

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
""")
    print(f"\n✓ Created README.md")
    total_size = sum(f.stat().st_size for f in STAGING_DIR.rglob("*") if f.is_file())
    print(f"\nTotal staged: {total_size / (1024**2):.1f} MB")


def upload():
    api = HfApi()
    print(f"\nCreating repo: {REPO_ID} (private={PRIVATE})")
    try:
        create_repo(REPO_ID, private=PRIVATE, exist_ok=True)
        print(f"  ✓ Repo ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"  Repo creation: {e}")
    
    print(f"\nUploading from {STAGING_DIR}...")
    api.upload_folder(
        repo_id=REPO_ID, folder_path=str(STAGING_DIR),
        commit_message="Upload G-PARC model weights, test data, and configs (4 models)",
    )
    print(f"\n{'='*60}\n✅ Upload complete!\n   https://huggingface.co/{REPO_ID}\n{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print("G-PARC → Hugging Face Upload")
    print("=" * 60)
    print(f"Repo: {REPO_ID}, Private: {PRIVATE}\n")
    stage_files()
    response = input("\nProceed with upload? [y/N] ")
    if response.lower() == 'y':
        upload()
    else:
        print("Upload cancelled. Files are staged at:", STAGING_DIR)