#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J v3_elasto
#SBATCH -o v3_elasto.out
#SBATCH -e v3_elasto.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 72:00:00
#SBATCH -c 4
#SBATCH --mem=80G

echo "================================================================"
echo "G-PARCv3 ELASTOPLASTIC TRAINING — EROSION-AWARE"
echo "================================================================"
echo "Start: $(date)"

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# DATA
# ============================================================
TRAIN_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/train"
VAL_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/val"
OUTPUT_DIR="/scratch/jtb3sud/elasto_v3"

# ============================================================
# TRAINING — FROM SCRATCH
# ============================================================
PHASE1_EPOCHS=0        # Skip Phase 1 (no frozen displacement)
PHASE2_EPOCHS=1500     # Full joint training

# ============================================================
# LEARNING RATES
# ============================================================
LR=3e-4               # Displacement model
EROSION_LR=1e-3       # Erosion head
WEIGHT_DECAY=1e-5

# ============================================================
# ARCHITECTURE
# ============================================================
NUM_LAYERS=4
HIDDEN_CHANNELS=128
FEATURE_OUT_CHANNELS=128
DROPOUT=0.0
USE_LAYER_NORM="--use_layer_norm"
USE_RELATIVE_POS="--use_relative_pos"
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=2
USE_VON_MISES="--use_von_mises"
USE_VOLUMETRIC="--use_volumetric"
SPADE_HEADS=4
SPADE_CONCAT="--spade_concat"
SPADE_DROPOUT=0.1
ZERO_INIT="--zero_init"
CLAMP_FLAG="--no_clamp_output"

# ============================================================
# EROSION HEAD
# ============================================================
EROSION_HIDDEN=64
EROSION_LAYERS=2
EROSION_DROPOUT=0.1
EROSION_WEIGHT=1.0
EROSION_THRESHOLD=0.5
FOCAL_ALPHA=0.75
FOCAL_GAMMA=2.0

# ============================================================
# TRAINING CONFIG
# ============================================================
SEQ_LEN=16
STRIDE=8
GRAD_CLIP_NORM=2.0
NUM_WORKERS=4
SS_SCHEDULE="linear"
SS_INITIAL_RATIO=0.0
SS_FINAL_RATIO=0.0

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Train data: $TRAIN_DIR"
echo "Val data:   $VAL_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Training:   $PHASE2_EPOCHS epochs from scratch"
echo "Disp LR:    $LR"
echo "Erosion LR: $EROSION_LR"
echo "Focal:      alpha=$FOCAL_ALPHA gamma=$FOCAL_GAMMA"
echo "================================================================"

apptainer run --nv "$CONTAINER" train_v3.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --skip_phase1 \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
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
    $CLAMP_FLAG \
    --phase1_epochs "$PHASE1_EPOCHS" \
    --phase2_epochs "$PHASE2_EPOCHS" \
    --lr "$LR" \
    --erosion_lr "$EROSION_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    --erosion_hidden_dim "$EROSION_HIDDEN" \
    --erosion_num_layers "$EROSION_LAYERS" \
    --erosion_dropout "$EROSION_DROPOUT" \
    --erosion_weight "$EROSION_WEIGHT" \
    --erosion_threshold "$EROSION_THRESHOLD" \
    --focal_alpha "$FOCAL_ALPHA" \
    --focal_gamma "$FOCAL_GAMMA" \
    --ss_schedule "$SS_SCHEDULE" \
    --ss_initial_ratio "$SS_INITIAL_RATIO" \
    --ss_final_ratio "$SS_FINAL_RATIO" \
    --num_workers "$NUM_WORKERS" \
    --preload \
    --device cuda

EXIT_CODE=$?

echo ""
echo "End: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training complete! Results in $OUTPUT_DIR"
else
    echo "❌ Failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE