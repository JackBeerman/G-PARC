#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J gparc_river_v2
#SBATCH -o gparc_river_v2.out
#SBATCH -e gparc_river_v2.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 48:00:00
#SBATCH -c 8
#SBATCH --mem=120G

echo "================================================================"
echo "G-PARCv2 RIVER: GRAPH CONV + MLS + SCHEDULED SAMPLING"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Key Updates:"
echo "  1. AdamW + Cosine Annealing (instead of ReduceLROnPlateau)"
echo "  2. Scheduled Sampling enabled (Teacher Forcing control)"
echo "  3. GraphConv V2 Feature Extractor"
echo "================================================================"

module purge
module load apptainer

# Optimization for Graph Operations
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# DATA PATHS
# ============================================================
BASE_DATA="/standard/sds_baek_energetic/HEC_RAS (River)"
TRAIN_DIR="${BASE_DATA}/pt_train_normalized"
VAL_DIR="${BASE_DATA}/pt_val_normalized"
OUTPUT_DIR="/scratch/jtb3sud/river_v2_training_scheduled"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
NUM_EPOCHS=50
SEQ_LEN=1
STRIDE=1
LR=1e-4                 # Kept your original River LR (Elasto uses 3e-4, change if needed)
GRAD_CLIP_NORM=1.0      # Matches your previous River clip
NUM_WORKERS=4

# ============================================================
# SCHEDULED SAMPLING (Teacher Forcing)
# ============================================================
# Options: linear, exponential, sigmoid
# Set INITIAL=0.0 and FINAL=0.0 for pure rollout (like Elasto)
# Set INITIAL=1.0 and FINAL=0.0 to gradually wean off supervision
SS_SCHEDULE="linear"
SS_INITIAL_RATIO=0.0
SS_FINAL_RATIO=0.0

# ============================================================
# ARCHITECTURE (GraphConv V2)
# ============================================================
NUM_LAYERS=2            # Standard for GraphConv V2
HIDDEN_CHANNELS=64
FEATURE_OUT_CHANNELS=128
DROPOUT=0.2
USE_LAYER_NORM="--use_layer_norm"
USE_RELATIVE_POS="--use_relative_pos"

# ============================================================
# PHYSICS & DIFFERENTIATOR
# ============================================================
NUM_STATIC_FEATS=9
NUM_DYNAMIC_FEATS=4
VELOCITY_INDICES="2 3"
INTEGRATOR="euler"
SPADE_HEADS=4
SPADE_CONCAT="--spade_concat"
SPADE_DROPOUT=0.1
ZERO_INIT="--zero_init"

# Load Normalization Stats if they exist (optional metadata)
NORM_STATS_FILE="${BASE_DATA}/normalization_stats.json"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Output: $OUTPUT_DIR"
echo "  LR: $LR | Epochs: $NUM_EPOCHS"
echo "  Seq Len: $SEQ_LEN | Stride: $STRIDE"
echo "  Scheduled Sampling: $SS_SCHEDULE ($SS_INITIAL_RATIO -> $SS_FINAL_RATIO)"
echo "  Integrator: $INTEGRATOR"
echo "================================================================"

if [ -f "$NORM_STATS_FILE" ]; then
    cp "$NORM_STATS_FILE" "$OUTPUT_DIR/normalization_stats.json"
    echo "✓ Copied normalization stats to output directory"
fi

# NOTE: Ensure the python script name matches your file (e.g. train_river_v2.py)
apptainer run --nv "$CONTAINER" train_gparcv2.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --lr "$LR" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --velocity_indices $VELOCITY_INDICES \
    --integrator "$INTEGRATOR" \
    --num_layers "$NUM_LAYERS" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT_CHANNELS" \
    --dropout "$DROPOUT" \
    $USE_LAYER_NORM \
    $USE_RELATIVE_POS \
    --spade_heads "$SPADE_HEADS" \
    $SPADE_CONCAT \
    --spade_dropout "$SPADE_DROPOUT" \
    $ZERO_INIT \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    --num_workers "$NUM_WORKERS" \
    --ss_schedule "$SS_SCHEDULE" \
    --ss_initial_ratio "$SS_INITIAL_RATIO" \
    --ss_final_ratio "$SS_FINAL_RATIO"

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Finished at: $(date) | Exit: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS"
else
    echo "❌ FAILED"
fi
echo "================================================================"