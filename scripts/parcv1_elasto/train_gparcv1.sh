#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J gparcv1
#SBATCH -o gparcv1.out
#SBATCH -e gparcv1.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH -t 72:00:00
#SBATCH -c 8
#SBATCH --mem=80G

echo "================================================================"
echo "G-PARC v3: FRESH TRAINING WITH LAPLACIAN DAMPING FIX"
echo "================================================================"
echo ""
echo "Changes from previous runs:"
echo "  3. TF=0.0 throughout (proven best strategy)"
echo "  4. seq_len=16 from start (no Phase 1/2 split needed)"
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
OUTPUT_DIR="/scratch/jtb3sud/elasto_graphconv_V2/gparcv1"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
NUM_EPOCHS=1500           # Long cosine schedule — model was still improving at 
SEQ_LEN=16               # 16-step rollout from the start (no Phase 1/2 needed)
STRIDE=16                 # Full coverage: stride=8 gives 100% with T=40, seq_len=16
LR=3e-4                  # PLAID authors' LR, proven effective in Phase 2
NUM_WORKERS=4

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
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=2  # Changed from 4 to 3 (after skipping meaningless variable)
FILE_PATTERN="*.pt"

# Feature extractor parameters (can keep these moderate since mesh is fixed)
HIDDEN_CHANNELS=128
FEATURE_OUT_CHANNELS=128
DEPTH=3
HEADS=3
DROPOUT=0.1

# NEW: Derivative solver parameters (focus on physics learning)
DERIV_HIDDEN_CHANNELS=128
DERIV_NUM_LAYERS=3
DERIV_HEADS=3
DERIV_DROPOUT=0.1

# NEW: Integral solver parameters (focus on temporal integration)
INTEGRAL_HIDDEN_CHANNELS=128
INTEGRAL_NUM_LAYERS=3
INTEGRAL_HEADS=4
INTEGRAL_DROPOUT=0.1


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

apptainer run --nv "$CONTAINER" train_gparcv1.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
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
    --mask_eroding \
    --num_workers "$NUM_WORKERS" \
    --ss_schedule "$SS_SCHEDULE" \
    --ss_initial_ratio "$SS_INITIAL_RATIO" \
    --ss_final_ratio "$SS_FINAL_RATIO" \
    --device "cuda"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training complete!"
    echo "  Next: run eval with --eval_mode both"
else
    echo "❌ Training failed"
fi

exit $EXIT_CODE