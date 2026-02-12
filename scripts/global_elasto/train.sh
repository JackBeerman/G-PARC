#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J gparc_2hopcont
#SBATCH -o gparc_2hopcont.out
#SBATCH -e gparc_2hopcont.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH -t 24:00:00
#SBATCH -c 8
#SBATCH --mem=80G

echo "================================================================"
echo "G-PARC v3: FRESH TRAINING WITH LAPLACIAN DAMPING FIX"
echo "================================================================"
echo ""
echo "Changes from previous runs:"
echo "  1. Laplacian 2hop computation (fixes boundary artifacts)"
echo "  2. LR=3e-4 from start (what PLAID authors used)"
echo "  3. TF=0.0 throughout (proven best strategy)"
echo "  4. seq_len=16 from start (no Phase 1/2 split needed)"
echo "  5. No loss decay (wasn't helping)"
echo "  6. 1500 epochs with cosine decay (long enough for convergence)"
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
OUTPUT_DIR="/scratch/jtb3sud/elasto_graphconv_V2/2hop"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
NUM_EPOCHS=1500           # Long cosine schedule — model was still improving at 548
SEQ_LEN=16               # 16-step rollout from the start (no Phase 1/2 needed)
STRIDE=16                # Non-overlapping windows
LR=3e-4                  # PLAID authors' LR, proven effective in Phase 2
NUM_WORKERS=4
GRAD_CLIP_NORM=2.0       # Kept from Phase 2, handles 3e-4 LR spikes

# ============================================================
# NO TEACHER FORCING — proven best strategy
# TF=0 beat TF→0.5 by 33x (0.074 vs 2.43 RRMSE)
# ============================================================
SS_SCHEDULE="linear"
SS_INITIAL_RATIO=0.0
SS_FINAL_RATIO=0.0

# ============================================================
# ARCHITECTURE
# ============================================================
NUM_LAYERS=4
HIDDEN_CHANNELS=128
FEATURE_OUT_CHANNELS=128
DROPOUT=0.0
USE_LAYER_NORM="--use_layer_norm"
USE_RELATIVE_POS="--use_relative_pos"
CLAMP_FLAG="--no_clamp_output"

# ============================================================
# PHYSICS
# ============================================================
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=2
USE_VON_MISES="--use_von_mises"
USE_VOLUMETRIC="--use_volumetric"
SPADE_HEADS=4
SPADE_CONCAT="--spade_concat"
SPADE_DROPOUT=0.1
ZERO_INIT="--zero_init"
MASK_ERODING="--mask_eroding"

# NO loss decay — removed, wasn't helping
# NO boundary_margin — replaced by neighbor-count Laplacian damping in difftest.py

NORM_STATS_FILE="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/normalization_stats.json"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Output: $OUTPUT_DIR"
echo "  LR: $LR (cosine → ~0 over $NUM_EPOCHS epochs)"
echo "  Seq length: $SEQ_LEN"
echo "  Teacher forcing: 0.0 (pure free-running)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Grad clip: $GRAD_CLIP_NORM"
echo "  MLS: Laplacian damping at <5 neighbors, gradients undamped"
echo "================================================================"

if [ -f "$NORM_STATS_FILE" ]; then
    cp "$NORM_STATS_FILE" "$OUTPUT_DIR/normalization_stats.json"
    echo "✓ Copied normalization stats to output directory"
fi

apptainer run --nv "$CONTAINER" 2hop.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --lr "$LR" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --integrator "euler" \
    --num_layers "$NUM_LAYERS" \
    --resume "/scratch/jtb3sud/elasto_graphconv_V2/2hop/latest_model.pth" \
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
    --ss_final_ratio "$SS_FINAL_RATIO"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training complete!"
    echo "  Next: run eval with --eval_mode both"
else
    echo "❌ Training failed"
fi

exit $EXIT_CODE
