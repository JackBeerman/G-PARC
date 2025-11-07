#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic       
#SBATCH -J gparc_baseball_motion        
#SBATCH -o %x.out                   
#SBATCH -e %x.err                   
#SBATCH -p gpu                      
#SBATCH --gres=gpu:a100:1           
#SBATCH -t 72:00:00                 # Increased time for baseball training
#SBATCH -c 4                        
#SBATCH --mem=80G                   

# Load modules
module purge
module load apptainer

# Define paths
TRAIN_DIR="/project/vil_baek/psaap/baseball/seq_baseball_data_normalized/train"
VAL_DIR="/project/vil_baek/psaap/baseball/seq_baseball_data_normalized/val"
TEST_DIR="/project/vil_baek/psaap/baseball/seq_baseball_data_normalized/test"
OUTPUT_DIR="/project/vil_baek/psaap/baseball/training_files/baseball_training_$(date +%Y%m%d_%H%M%S)"

# Container path
CONTAINER="/share/resources/containers/apptainer/pytorch-2.4.0.sif"

# Training arguments for baseball pitching motion
NUM_EPOCHS=100
SEQ_LEN=4                    # Sequences for motion prediction
LR=1e-4
NUM_STATIC_FEATS=0           # All features are dynamic
NUM_DYNAMIC_FEATS=9          # 3 pos + 3 vel + 3 angles
FILE_PATTERN="*.pt"

# NEW: Multi-step context configuration
# Set to 0 for single-step (original), or 3+ for multi-step context
NUM_CONTEXT_STEPS=3          # Recommended: 3 for baseball pose prediction

# Feature extractor parameters (for joint position processing)
# Baseball has more complex motion, so larger networks
HIDDEN_CHANNELS=128
FEATURE_OUT_CHANNELS=256
DEPTH=3                      # Deeper network for baseball
HEADS=8                      # More attention heads
DROPOUT=0.2

# Derivative solver parameters (for motion dynamics)
DERIV_HIDDEN_CHANNELS=256    # Larger for baseball
DERIV_NUM_LAYERS=4
DERIV_HEADS=8
DERIV_DROPOUT=0.3

# Integral solver parameters (for next frame prediction)
INTEGRAL_HIDDEN_CHANNELS=256 # Larger for baseball
INTEGRAL_NUM_LAYERS=4
INTEGRAL_HEADS=8
INTEGRAL_DROPOUT=0.3

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "GPARC Baseball Pitching Motion Training"
echo "========================================"
echo "Train directory: ${TRAIN_DIR}"
echo "Val directory: ${VAL_DIR}"
echo "Test directory: ${TEST_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Training Configuration:"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Sequence length: ${SEQ_LEN}"
echo "  Learning rate: ${LR}"
echo "  Static features: ${NUM_STATIC_FEATS}"
echo "  Dynamic features: ${NUM_DYNAMIC_FEATS} (3 pos + 3 vel + 3 angles)"
echo "  Context steps: ${NUM_CONTEXT_STEPS}"
if [ "$NUM_CONTEXT_STEPS" -gt 0 ]; then
    echo "  -> Using MULTI-STEP context training"
else
    echo "  -> Using SINGLE-STEP training"
fi
echo ""
echo "Data Info:"
echo "  Data is NORMALIZED (mean=0, std=1)"
echo "  18 joints in skeleton"
echo "  Global feature: pitch_speed (mph)"
echo ""
echo "Model Architecture:"
echo "  Feature Extractor: ${DEPTH} layers, ${HIDDEN_CHANNELS}→${FEATURE_OUT_CHANNELS}, ${HEADS} heads"
echo "  Derivative Solver: ${DERIV_NUM_LAYERS} layers, ${DERIV_HIDDEN_CHANNELS} channels, ${DERIV_HEADS} heads"
echo "  Integral Solver: ${INTEGRAL_NUM_LAYERS} layers, ${INTEGRAL_HIDDEN_CHANNELS} channels, ${INTEGRAL_HEADS} heads"
echo "========================================"
echo ""

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
    --num_context_steps "$NUM_CONTEXT_STEPS" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT_CHANNELS" \
    --depth "$DEPTH" \
    --pool_ratios 0.7 \
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

echo ""
echo "========================================"
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"