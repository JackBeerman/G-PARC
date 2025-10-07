#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic       
#SBATCH -J gparc_cylinder       
#SBATCH -o %x.out                   
#SBATCH -e %x.err                   
#SBATCH -p gpu                      
#SBATCH --gres=gpu:a100:1           
#SBATCH -t 36:14:00                 
#SBATCH -c 4                        
#SBATCH --mem=80G                   

# Load modules
module purge
module load apptainer

# Define paths - UPDATED FOR CYLINDER DATA
TRAIN_DIR="/standard/sds_baek_energetic/von_karman_vortex/Reynolds 1~150/processed_parc/normalized/train"
VAL_DIR="/standard/sds_baek_energetic/von_karman_vortex/Reynolds 1~150/processed_parc/normalized/val"
TEST_DIR="/standard/sds_baek_energetic/von_karman_vortex/Reynolds 1~150/processed_parc/normalized/test"
OUTPUT_DIR="/standard/sds_baek_energetic/von_karman_vortex/Reynolds 1~150/parc_model/train_$(date +%Y%m%d_%H%M%S)_run"

# Container path
CONTAINER="/share/resources/containers/apptainer/pytorch-2.4.0.sif"

# Training arguments
NUM_EPOCHS=300
SEQ_LEN=1
LR=1e-5
NUM_STATIC_FEATS=3      # x, y, z positions
NUM_DYNAMIC_FEATS=7     # 7 raw physics variables (pressure, velocity_x/y/z, vorticity_x/y/z)
SKIP_INDICES=""         # No skipping - using all raw variables
FILE_PATTERN="*_normalized.pt"  # Updated file pattern

# Feature extractor parameters
HIDDEN_CHANNELS=64      # Increased - larger mesh (60k nodes vs 4k)
FEATURE_OUT_CHANNELS=64
DEPTH=3                 # Increased depth for larger mesh
HEADS=4                 # More heads for complex flow
DROPOUT=0.2

# Derivative solver parameters
DERIV_HIDDEN_CHANNELS=128
DERIV_NUM_LAYERS=3
DERIV_HEADS=4
DERIV_DROPOUT=0.2

# Integral solver parameters
INTEGRAL_HIDDEN_CHANNELS=128
INTEGRAL_NUM_LAYERS=3
INTEGRAL_HEADS=4
INTEGRAL_DROPOUT=0.2

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting GPARC cylinder training with new data structure..."
echo "Mesh size: ~60,746 nodes per graph"
echo "Reynolds numbers: [1, 20, 40, 50, 100, 150]"
echo "Using ${NUM_DYNAMIC_FEATS} dynamic features (RAW DATA ONLY)"
echo "Features: pressure, velocity_x/y/z, vorticity_x/y/z"
echo "Derivative solver: ${DERIV_NUM_LAYERS} layers, ${DERIV_HIDDEN_CHANNELS} channels, ${DERIV_HEADS} heads"
echo "Integral solver: ${INTEGRAL_NUM_LAYERS} layers, ${INTEGRAL_HIDDEN_CHANNELS} channels, ${INTEGRAL_HEADS} heads"

# Run with container
apptainer run --nv "$CONTAINER" \
  modularized.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --lr "$LR" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --file_pattern "$FILE_PATTERN" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT_CHANNELS" \
    --depth "$DEPTH" \
    --pool_ratios 0.2 \
    --heads "$HEADS" \
    --dropout "$DROPOUT" \
    --resume "/standard/sds_baek_energetic/von_karman_vortex/Reynolds 1~150/parc_model/train_20250929_212524_run/shock_tube_best_model.pth" \
    --deriv_hidden_channels "$DERIV_HIDDEN_CHANNELS" \
    --deriv_num_layers "$DERIV_NUM_LAYERS" \
    --deriv_heads "$DERIV_HEADS" \
    --deriv_dropout "$DERIV_DROPOUT" \
    --deriv_use_residual \
    --integral_hidden_channels "$INTEGRAL_HIDDEN_CHANNELS" \
    --integral_num_layers "$INTEGRAL_NUM_LAYERS" \
    --integral_heads "$INTEGRAL_HEADS" \
    --integral_dropout "$INTEGRAL_DROPOUT" \
    --integral_use_residual \
    --verbose \
    --device "cuda"

echo "Training completed. Results saved to: $OUTPUT_DIR"