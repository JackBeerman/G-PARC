#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic       
#SBATCH -J gparc_cylinder_large     
#SBATCH -o %x.out                   
#SBATCH -e %x.err                   
#SBATCH -p gpu                      
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH -t 72:00:00                 
#SBATCH -c 4                        
#SBATCH --mem=80G                   

# Load modules
module purge
module load apptainer

# Define paths
TRAIN_DIR="/standard/sds_baek_energetic/von_karman_vortex/full_data/split_normalized/train"
VAL_DIR="/standard/sds_baek_energetic/von_karman_vortex/full_data/split_normalized/val"
TEST_DIR="/standard/sds_baek_energetic/von_karman_vortex/full_data/split_normalized/test"
OUTPUT_DIR="/standard/sds_baek_energetic/von_karman_vortex/processed_multi_dir/parc_model/train_$(date +%Y%m%d_%H%M%S)_large_150_6"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# Training arguments
NUM_EPOCHS=50
SEQ_LEN=3                
LR=1e-5                  # CHANGED: Lower LR (was 1e-5)
NUM_STATIC_FEATS=3
NUM_DYNAMIC_FEATS=4
SKIP_INDICES="3 4 5"
FILE_PATTERN="*_normalized.pt"

# Feature extractor parameters
HIDDEN_CHANNELS=256
FEATURE_OUT_CHANNELS=128
DEPTH=5
HEADS=2
DROPOUT=0.1

# Derivative solver parameters
DERIV_HIDDEN_CHANNELS=256
DERIV_NUM_LAYERS=4
DERIV_HEADS=2
DERIV_DROPOUT=0.1

# Integral solver parameters
INTEGRAL_HIDDEN_CHANNELS=256
INTEGRAL_NUM_LAYERS=4
INTEGRAL_HEADS=2
INTEGRAL_DROPOUT=0.1

# Create output directory
# 
mkdir -p "$OUTPUT_DIR"

echo "Starting GPARC cylinder training - SCALED UP MODEL..."
echo "Mesh size: ~60,746 nodes"
echo "Model parameters: ~5-6M (up from 900k)"
echo "Hidden channels: 256, Depth: 4, Heads: 6"

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
    --skip_dynamic_indices $SKIP_INDICES \
    --file_pattern "$FILE_PATTERN" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT_CHANNELS" \
    --depth "$DEPTH" \
    --heads "$HEADS" \
    --dropout "$DROPOUT" \
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

#    --resume "/standard/sds_baek_energetic/von_karman_vortex/Reynolds 1~150/parc_model/train_20251006_230126_large_300_3/shock_tube_best_model.pth" 