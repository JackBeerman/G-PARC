#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J gparc_shock_v2
#SBATCH -o gparc_shock_v2.out
#SBATCH -e gparc_shock_v2.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 48:10:00
#SBATCH -c 8
#SBATCH --mem=120G

echo "================================================================"
echo "G-PARCv2 SHOCK TUBE: MLS + GLOBAL FiLM + NUMERICAL INTEGRATION"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# DATA PATHS — EDIT THESE
# ============================================================
OUTPUT_DIR="/scratch/jtb3sud/shocktube_v2_training"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

TRAIN_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/train_cases_normalized"
VAL_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/val_cases_normalized"
TEST_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/test_cases_normalized"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
NUM_EPOCHS=100
SEQ_LEN=4
STRIDE=4
LR=1e-4
GRAD_CLIP_NORM=1.0
NUM_WORKERS=0

# ============================================================
# SCHEDULED SAMPLING
# ============================================================
SS_SCHEDULE="linear"
SS_INITIAL_RATIO=0.0
SS_FINAL_RATIO=0.0

# ============================================================
# ARCHITECTURE
# ============================================================
NUM_LAYERS=4
HIDDEN_CHANNELS=64
FEATURE_OUT_CHANNELS=128
DROPOUT=0.2
USE_LAYER_NORM="--use_layer_norm"
USE_RELATIVE_POS="--use_relative_pos"

# ============================================================
# PHYSICS & SHOCK TUBE SPECIFICS
# ============================================================
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=3         # After skipping y_momentum
SKIP_DYNAMIC_INDICES="2"    # Skip y_momentum (raw index 2)
VELOCITY_INDEX=1             # x_momentum in used dynamic features
GLOBAL_PARAM_DIM=3           # pressure, density, delta_t
GLOBAL_EMBED_DIM=64
INTEGRATOR="euler"

# SPADE
SPADE_HEADS=4
SPADE_CONCAT="--spade_concat"
SPADE_DROPOUT=0.1
ZERO_INIT="--zero_init"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Output: $OUTPUT_DIR"
echo "  LR: $LR | Epochs: $NUM_EPOCHS"
echo "  Seq Len: $SEQ_LEN | Stride: $STRIDE"
echo "  Dynamic feats: $NUM_DYNAMIC_FEATS (skip raw indices: $SKIP_DYNAMIC_INDICES)"
echo "  Global: dim=$GLOBAL_PARAM_DIM → embed=$GLOBAL_EMBED_DIM"
echo "  Scheduled Sampling: $SS_SCHEDULE ($SS_INITIAL_RATIO -> $SS_FINAL_RATIO)"
echo "================================================================"

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
    --skip_dynamic_indices $SKIP_DYNAMIC_INDICES \
    --velocity_index "$VELOCITY_INDEX" \
    --global_param_dim "$GLOBAL_PARAM_DIM" \
    --global_embed_dim "$GLOBAL_EMBED_DIM" \
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