#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic       
#SBATCH -J gparc_shock_tube_modv2    
#SBATCH -o %x.out                   
#SBATCH -e %x.err                   
#SBATCH -p gpu                      
#SBATCH --gres=gpu:a100:1           
#SBATCH -t 72:00:00                 
#SBATCH -c 4                        
#SBATCH --mem=80G                   

# Load modules
module purge
module load apptainer

# Define paths
TRAIN_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/train_cases_normalized"
VAL_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/val_cases_normalized"
TEST_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/test_cases_normalized"
OUTPUT_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/shock_tube_$(date +%Y%m%d_%H%M%S)_run10_900"

# Container path
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# Training arguments
NUM_EPOCHS=600
SEQ_LEN=10
LR=1e-5
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=3  # Changed from 4 to 3 (after skipping meaningless variable)
SKIP_INDICES="2"     # Skip the third variable (0-indexed)
FILE_PATTERN="*.pt"

# Feature extractor parameters (can keep these moderate since mesh is fixed)
HIDDEN_CHANNELS=32
FEATURE_OUT_CHANNELS=32
DEPTH=2
HEADS=2
DROPOUT=0.2

# NEW: Derivative solver parameters (focus on physics learning)
DERIV_HIDDEN_CHANNELS=128
DERIV_NUM_LAYERS=3
DERIV_HEADS=4
DERIV_DROPOUT=0.2

# NEW: Integral solver parameters (focus on temporal integration)
INTEGRAL_HIDDEN_CHANNELS=128
INTEGRAL_NUM_LAYERS=3
INTEGRAL_HEADS=4
INTEGRAL_DROPOUT=0.2

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting GPARC shock tube training with deeper physics networks and direct delta_t..."
echo "Using ${NUM_DYNAMIC_FEATS} dynamic features (skipping indices: ${SKIP_INDICES})"
echo "Derivative solver: ${DERIV_NUM_LAYERS} layers, ${DERIV_HIDDEN_CHANNELS} channels, ${DERIV_HEADS} heads"
echo "Integral solver: ${INTEGRAL_NUM_LAYERS} layers, ${INTEGRAL_HIDDEN_CHANNELS} channels, ${INTEGRAL_HEADS} heads"
echo "Using direct delta_t for interpretability"

# Run with container
apptainer run --nv "$CONTAINER" \
  new_mod.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --lr "$LR" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --skip_dynamic_indices "$SKIP_INDICES" \
    --file_pattern "$FILE_PATTERN" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT_CHANNELS" \
    --depth "$DEPTH" \
    --pool_ratios 0.2 \
    --resume "/home/jtb3sud/G-PARC/scripts/shockchord/latest_model.pth" \
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
    --use_direct_delta_t \
    --verbose \
    --device "cuda"

echo "Training completed. Results saved to: $OUTPUT_DIR"