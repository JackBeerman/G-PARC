#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic       
#SBATCH -J gparc_tennis_motion        
#SBATCH -o %x.out                   
#SBATCH -e %x.err                   
#SBATCH -p gpu                      
#SBATCH --gres=gpu:a100:1           
#SBATCH -t 72:00:00                 # Increased time for tennis training
#SBATCH -c 4                        
#SBATCH --mem=80G                   

# Load modules
module purge
module load apptainer

# Define paths
TRAIN_DIR="/project/vil_baek/psaap/tennis/seq_tennis_data_normalized/train"
VAL_DIR="/project/vil_baek/psaap/tennis/seq_tennis_data_normalized/val"
TEST_DIR="/project/vil_baek/psaap/tennis/seq_tennis_data_normalized/test"
OUTPUT_DIR="/project/vil_baek/psaap/tennis/training_files/tennis_training_$(date +%Y%m%d_%H%M%S)"

# Container path
CONTAINER="/share/resources/containers/apptainer/pytorch-2.4.0.sif"

# Training arguments for tennis motion
NUM_EPOCHS=15
SEQ_LEN=1                    # Longer sequences for motion prediction
LR=1e-4
NUM_STATIC_FEATS=0
NUM_DYNAMIC_FEATS=6  
FILE_PATTERN="*.pt"

# Feature extractor parameters (for joint position processing)
HIDDEN_CHANNELS=64
FEATURE_OUT_CHANNELS=128
DEPTH=2
HEADS=4
DROPOUT=0.2

# Derivative solver parameters (for motion dynamics)
DERIV_HIDDEN_CHANNELS=128
DERIV_NUM_LAYERS=3
DERIV_HEADS=4
DERIV_DROPOUT=0.3

# Integral solver parameters (for next frame prediction)
INTEGRAL_HIDDEN_CHANNELS=128
INTEGRAL_NUM_LAYERS=3
INTEGRAL_HEADS=4
INTEGRAL_DROPOUT=0.3

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting GPARC tennis motion training..."
echo "Data directory: ${DATA_DIR}"
echo "Position features: ${NUM_POSITION_FEATS}, Velocity features: ${NUM_VELOCITY_FEATS}"
echo "Sequence length: ${SEQ_LEN}"
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

echo "Tennis training completed. Results saved to: $OUTPUT_DIR"