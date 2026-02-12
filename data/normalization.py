#!/usr/bin/env python3
"""
Global Max Normalization Preprocessing Script
==============================================
Processes elastoplastic dynamics data with physics-preserving normalization.

Usage:
    python preprocess_global_max.py

Author: Jack
Date: February 2026
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DATA_DIR = Path("/scratch/jtb3sud/processed_elasto_plastic/zscore/raw")
OUTPUT_DIR = Path("/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized")

print("="*80)
print("GLOBAL MAX NORMALIZATION - PREPROCESSING")
print("="*80)
print(f"\nInput:  {RAW_DATA_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("\nThis will:")
print("  1. Scan train/val/test data")
print("  2. Compute max_position and max_displacement")
print("  3. Normalize to preserve physics ratios")
print("="*80)

# ============================================================================
# STEP 1: COMPUTE NORMALIZATION STATISTICS
# ============================================================================

print("\n" + "="*80)
print("STEP 1: COMPUTING NORMALIZATION STATISTICS")
print("="*80)

# We'll compute from training data only (proper ML practice)
train_dir = RAW_DATA_DIR / "train"
train_files = sorted(list(train_dir.glob("*.pt")))

print(f"\nScanning {len(train_files)} training files...")

# Initialize trackers for statistics
global_max_x_pos = 0
global_max_y_pos = 0
global_max_ux = 0
global_max_uy = 0
global_min_ux = 0
global_min_uy = 0

# For computing statistics (we'll sample to avoid memory issues)
x_pos_sample = []
y_pos_sample = []
ux_sample = []
uy_sample = []

SAMPLE_SIZE = 1_000_000  # Sample 1M points for statistics

for pt_file in tqdm(train_files, desc="Scanning training data"):
    try:
        data = torch.load(pt_file, weights_only=False)
        
        if isinstance(data, list):
            for timestep in data:
                if not hasattr(timestep, 'x') or timestep.x is None:
                    continue
                
                # Track global extremes (no memory issue)
                global_max_x_pos = max(global_max_x_pos, timestep.x[:, 0].max().item())
                global_max_y_pos = max(global_max_y_pos, timestep.x[:, 1].max().item())
                
                ux = timestep.x[:, 2]
                uy = timestep.x[:, 3]
                
                global_max_ux = max(global_max_ux, ux.max().item())
                global_max_uy = max(global_max_uy, uy.max().item())
                global_min_ux = min(global_min_ux, ux.min().item())
                global_min_uy = min(global_min_uy, uy.min().item())
                
                # Sample for statistics
                if len(ux_sample) < SAMPLE_SIZE:
                    # Sample every 10th node to reduce memory
                    x_pos_sample.extend(timestep.x[::10, 0].cpu().numpy())
                    y_pos_sample.extend(timestep.x[::10, 1].cpu().numpy())
                    ux_sample.extend(timestep.x[::10, 2].cpu().numpy())
                    uy_sample.extend(timestep.x[::10, 3].cpu().numpy())
        
        # Free memory
        del data
        
    except Exception as e:
        print(f"Error loading {pt_file.name}: {e}")

# Convert samples to arrays for statistics
x_pos_sample = np.array(x_pos_sample[:SAMPLE_SIZE])
y_pos_sample = np.array(y_pos_sample[:SAMPLE_SIZE])
ux_sample = np.array(ux_sample[:SAMPLE_SIZE])
uy_sample = np.array(uy_sample[:SAMPLE_SIZE])

# Remove NaNs
x_pos_sample = x_pos_sample[~np.isnan(x_pos_sample)]
y_pos_sample = y_pos_sample[~np.isnan(y_pos_sample)]
ux_sample = ux_sample[~np.isnan(ux_sample)]
uy_sample = uy_sample[~np.isnan(uy_sample)]

# Compute global maximums
max_position = max(global_max_x_pos, global_max_y_pos)
max_displacement = max(abs(global_min_ux), abs(global_max_ux), 
                      abs(global_min_uy), abs(global_max_uy))

print(f"\n{'='*80}")
print("COMPUTED NORMALIZATION CONSTANTS")
print("="*80)

print(f"\nPositions:")
print(f"  X max: {global_max_x_pos:.2f} mm")
print(f"  Y max: {global_max_y_pos:.2f} mm")
print(f"  max_position = {max_position:.6f} mm")

print(f"\nDisplacements:")
print(f"  U_x range: [{global_min_ux:.2f}, {global_max_ux:.2f}] mm")
print(f"  U_y range: [{global_min_uy:.2f}, {global_max_uy:.2f}] mm")
print(f"  max_displacement = {max_displacement:.6f} mm")

# Compute physics ratios from samples
aspect_ratio = global_max_x_pos / global_max_y_pos
displacement_ratio = ux_sample.std() / uy_sample.std()

print(f"\nPhysics ratios (PRESERVED):")
print(f"  Mesh aspect ratio (X/Y): {aspect_ratio:.2f}:1")
print(f"  Displacement std ratio (U_x/U_y): {displacement_ratio:.2f}:1")

print("="*80)

# Store statistics
stats = {
    'normalization_method': 'global_max',
    'computed_at': datetime.now().isoformat(),
    'num_samples': len(train_files),
    'num_nodes_sampled': len(ux_sample),
    
    'position': {
        'max_position': float(max_position),
        'x_pos': {
            'min': 0.0,  # We know this from domain
            'max': float(global_max_x_pos),
            'mean': float(x_pos_sample.mean()),
            'std': float(x_pos_sample.std()),
        },
        'y_pos': {
            'min': 0.0,  # We know this from domain
            'max': float(global_max_y_pos),
            'mean': float(y_pos_sample.mean()),
            'std': float(y_pos_sample.std()),
        }
    },
    
    'displacement': {
        'max_displacement': float(max_displacement),
        'U_x': {
            'min': float(global_min_ux),
            'max': float(global_max_ux),
            'mean': float(ux_sample.mean()),
            'std': float(ux_sample.std()),
        },
        'U_y': {
            'min': float(global_min_uy),
            'max': float(global_max_uy),
            'mean': float(uy_sample.mean()),
            'std': float(uy_sample.std()),
        }
    },
    
    'physics_ratios': {
        'aspect_ratio_xy': float(aspect_ratio),
        'displacement_ratio_ux_uy': float(displacement_ratio),
    }
}

# Free memory
del x_pos_sample, y_pos_sample, ux_sample, uy_sample
gc.collect()

# ============================================================================
# STEP 2: NORMALIZE FUNCTION
# ============================================================================

def normalize_data(data, max_position, max_displacement):
    """
    Normalize a single Data object with global max scaling.
    
    Args:
        data: PyG Data object
        max_position: Global max for positions
        max_displacement: Global max for displacements
    
    Returns:
        Normalized Data object
    """
    data_norm = data.clone()
    
    # Normalize positions (columns 0-1)
    data_norm.x[:, 0] = data.x[:, 0] / max_position  # x position
    data_norm.x[:, 1] = data.x[:, 1] / max_position  # y position
    
    # Normalize displacements (columns 2-3)
    data_norm.x[:, 2] = data.x[:, 2] / max_displacement  # U_x
    data_norm.x[:, 3] = data.x[:, 3] / max_displacement  # U_y
    
    # Erosion (column 4+) - keep as is (binary 0/1)
    # No normalization needed
    
    # Normalize target if present
    if hasattr(data, 'y') and data.y is not None:
        data_norm.y = data.y / max_displacement
    
    return data_norm

# ============================================================================
# STEP 3: PROCESS EACH SPLIT
# ============================================================================

def process_split(input_dir, output_dir, split_name, max_position, max_displacement):
    """Process a single data split."""
    print(f"\n{'='*80}")
    print(f"PROCESSING {split_name.upper()} SET")
    print("="*80)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pt_files = sorted(list(input_path.glob("*.pt")))
    
    if len(pt_files) == 0:
        print(f"Warning: No files found in {input_dir}")
        return 0, 0
    
    print(f"Found {len(pt_files)} files to process...")
    
    processed_count = 0
    error_count = 0
    
    for pt_file in tqdm(pt_files, desc=f"Processing {split_name}"):
        try:
            # Load raw data
            data_raw = torch.load(pt_file, weights_only=False)
            
            # Normalize
            if isinstance(data_raw, list):
                data_normalized = []
                for d in data_raw:
                    d_norm = normalize_data(d, max_position, max_displacement)
                    data_normalized.append(d_norm)
            else:
                data_normalized = normalize_data(data_raw, max_position, max_displacement)
            
            # Save
            output_file = output_path / pt_file.name
            torch.save(data_normalized, output_file)
            
            processed_count += 1
            
            # Free memory
            del data_raw, data_normalized
            
        except Exception as e:
            print(f"\nError processing {pt_file.name}: {e}")
            error_count += 1
        
        # Periodic garbage collection
        if processed_count % 50 == 0:
            gc.collect()
    
    print(f"\nCompleted {split_name}:")
    print(f"  Processed: {processed_count} files")
    print(f"  Errors:    {error_count} files")
    
    return processed_count, error_count

# Process all splits
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total_processed = 0
total_errors = 0

for split in ['train', 'val', 'test']:
    split_dir = RAW_DATA_DIR / split
    if split_dir.exists():
        n_proc, n_err = process_split(
            split_dir,
            OUTPUT_DIR / split,
            split,
            max_position,
            max_displacement
        )
        total_processed += n_proc
        total_errors += n_err

# ============================================================================
# STEP 4: SAVE METADATA AND CREATE README
# ============================================================================

print(f"\n{'='*80}")
print("SAVING METADATA")
print("="*80)

# Save statistics
stats_file = OUTPUT_DIR / "normalization_stats.json"
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"\n✓ Saved: {stats_file}")

# Create README
readme_path = OUTPUT_DIR / "README.md"
with open(readme_path, 'w') as f:
    f.write(f"""# Elastoplastic Dynamics - Global Max Normalized Dataset

## Normalization Method

**Global Max Normalization** (Physics-Preserving)

### Normalization Constants

- **Positions**: `max_position = {max_position:.6f} mm`
  - Preserves mesh aspect ratio: {aspect_ratio:.2f}:1
  
- **Displacements**: `max_displacement = {max_displacement:.6f} mm`
  - Preserves U_x:U_y variation ratio: {displacement_ratio:.2f}:1

- **Erosion**: Kept as-is (binary 0/1 values)

## Why Global Max?

Z-score normalization destroys physical relationships:
- Forces both U_x and U_y to have std ≈ 1.0
- Network cannot learn that U_x is physically {displacement_ratio:.1f}× harder to predict
- Results in imbalanced component errors (U_x RRMSE 2× worse than U_y)

Global max normalization:
- Preserves the physical {displacement_ratio:.1f}:1 ratio
- Network sees which component is inherently more difficult
- Results in balanced, better predictions

## Denormalization

To convert predictions back to physical units:
```python
# Positions
x_physical = x_normalized * {max_position:.6f}
y_physical = y_normalized * {max_position:.6f}

# Displacements
ux_physical = ux_normalized * {max_displacement:.6f}
uy_physical = uy_normalized * {max_displacement:.6f}
```

## Dataset Statistics

- Training samples: {len(train_files)}
- Validation samples: {len(list((RAW_DATA_DIR / 'val').glob('*.pt')))}
- Test samples: {len(list((RAW_DATA_DIR / 'test').glob('*.pt')))}

See `normalization_stats.json` for complete statistics.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

print(f"✓ Saved: {readme_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("PREPROCESSING COMPLETE!")
print("="*80)

print(f"\nTotal files processed: {total_processed}")
print(f"Total errors:          {total_errors}")

print(f"\nNormalization constants:")
print(f"  max_position:     {max_position:.6f} mm")
print(f"  max_displacement: {max_displacement:.6f} mm")

print(f"\nPhysics preserved:")
print(f"  Aspect ratio:     {aspect_ratio:.2f}:1")
print(f"  U_x/U_y ratio:    {displacement_ratio:.2f}:1")

print(f"\nOutput directory: {OUTPUT_DIR}")
train_count = len(list((OUTPUT_DIR / 'train').glob('*.pt'))) if (OUTPUT_DIR / 'train').exists() else 0
val_count = len(list((OUTPUT_DIR / 'val').glob('*.pt'))) if (OUTPUT_DIR / 'val').exists() else 0
test_count = len(list((OUTPUT_DIR / 'test').glob('*.pt'))) if (OUTPUT_DIR / 'test').exists() else 0

print(f"  - train/ ({train_count} files)")
print(f"  - val/   ({val_count} files)")
print(f"  - test/  ({test_count} files)")

print(f"\nNext steps:")
print(f"  1. Update training config:")
print(f"     DATA_DIR = '{OUTPUT_DIR}'")
print(f"  2. Remove any manual loss weighting (not needed anymore)")
print(f"  3. Train and expect balanced U_x/U_y RRMSE!")

print("="*80)