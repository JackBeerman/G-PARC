#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J gparc_erosion
#SBATCH -o gparc_erosion.out
#SBATCH -e gparc_erosion.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 24:10:00
#SBATCH -c 8
#SBATCH --mem=80G

echo "================================================================"
echo "G-PARC: JOINT DISPLACEMENT + EROSION HEAD TRAINING"
echo "================================================================"
echo ""
echo "Strategy:"
echo "  - Resume displacement model from 2hop checkpoint"
echo "  - Train erosion head from scratch on top"
echo "  - Separate LR: displacement=3e-4, erosion=1e-3"
echo "  - Focal loss (alpha=0.25, gamma=2) for class imbalance"
echo "  - Autoregressive erosion feedback during rollout"
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# DATA DIRECTORIES
# ============================================================
TRAIN_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/train"
VAL_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/val"
OUTPUT_DIR="/scratch/jtb3sud/elasto_graphconv_V2/erosion"

# Resume from best displacement-only model
RESUME_FROM="/scratch/jtb3sud/elasto_graphconv_V2/2hop/best_model.pth"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
NUM_EPOCHS=500
SEQ_LEN=16
STRIDE=16
LR=3e-4                  # Displacement model LR (same as 2hop)
EROSION_LR=1e-3           # Erosion head LR (higher — training from scratch)
EROSION_WEIGHT=1.0        # Focal loss weight relative to MSE
NUM_WORKERS=4
GRAD_CLIP_NORM=2.0

# Focal loss
FOCAL_ALPHA=0.25
FOCAL_GAMMA=2.0

# Erosion head architecture
EROSION_HIDDEN=64
EROSION_LAYERS=2
EROSION_DROPOUT=0.1

# No teacher forcing
SS_SCHEDULE="linear"
SS_INITIAL_RATIO=0.0
SS_FINAL_RATIO=0.0

# ============================================================
# ARCHITECTURE (must match 2hop checkpoint)
# ============================================================
NUM_LAYERS=4
HIDDEN_CHANNELS=128
FEATURE_OUT_CHANNELS=128
DROPOUT=0.0
USE_LAYER_NORM="--use_layer_norm"
USE_RELATIVE_POS="--use_relative_pos"
CLAMP_FLAG="--no_clamp_output"

# Physics
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=2
USE_VON_MISES="--use_von_mises"
USE_VOLUMETRIC="--use_volumetric"
SPADE_HEADS=4
SPADE_CONCAT="--spade_concat"
SPADE_DROPOUT=0.1
ZERO_INIT="--zero_init"
MASK_ERODING="--mask_eroding"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Output: $OUTPUT_DIR"
echo "  Resume: $RESUME_FROM (displacement only)"
echo "  Displacement LR: $LR"
echo "  Erosion LR: $EROSION_LR"
echo "  Erosion weight: $EROSION_WEIGHT"
echo "  Focal loss: alpha=$FOCAL_ALPHA, gamma=$FOCAL_GAMMA"
echo "  Erosion head: hidden=$EROSION_HIDDEN, layers=$EROSION_LAYERS"
echo "================================================================"

apptainer run --nv "$CONTAINER" train_erosion.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --lr "$LR" \
    --erosion_lr "$EROSION_LR" \
    --erosion_weight "$EROSION_WEIGHT" \
    --erosion_hidden_dim "$EROSION_HIDDEN" \
    --erosion_num_layers "$EROSION_LAYERS" \
    --erosion_dropout "$EROSION_DROPOUT" \
    --focal_alpha "$FOCAL_ALPHA" \
    --focal_gamma "$FOCAL_GAMMA" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --integrator "euler" \
    --num_layers "$NUM_LAYERS" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT_CHANNELS" \
    --dropout "$DROPOUT" \
    $USE_LAYER_NORM \
    $USE_RELATIVE_POS \
    $USE_VON_MISES \
    $USE_VOLUMETRIC \
    --spade_heads "$SPADE_HEADS" \
    $SPADE_CONCAT \
    --spade_dropout "$SPADE_DROPOUT" \
    $ZERO_INIT \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    $MASK_ERODING \
    --num_workers "$NUM_WORKERS" \
    $CLAMP_FLAG \
    --ss_schedule "$SS_SCHEDULE" \
    --ss_initial_ratio "$SS_INITIAL_RATIO" \
    --ss_final_ratio "$SS_FINAL_RATIO" \
    --resume "$RESUME_FROM" \
    --resume_displacement_only

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training complete!"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE