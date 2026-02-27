#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J river_concat20
#SBATCH -o river_concat20.out
#SBATCH -e river_concat20.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 72:00:00
#SBATCH -c 8
#SBATCH --mem=120G

echo "================================================================"
echo "G-PARCv2 RIVER — CONCAT+MLP FUSION (no SPADE)"
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
# DATA PATHS
# ============================================================
BASE_DATA="/standard/sds_baek_energetic/HEC_RAS (River)"
TRAIN_DIR="${BASE_DATA}/pt_train_normalized"
VAL_DIR="${BASE_DATA}/pt_val_normalized"
OUTPUT_DIR="/scratch/jtb3sud/gparcv2/river/concat/20"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
# Key changes from your current script:
SEQ_LEN=20
STRIDE=20
NUM_EPOCHS=200
LR=1e-4
GRAD_CLIP_NORM=1.0
NUM_WORKERS=4

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
# PHYSICS & DIFFERENTIATOR — CONCAT+MLP (no SPADE flags needed)
# ============================================================
NUM_STATIC_FEATS=9
NUM_DYNAMIC_FEATS=4
VELOCITY_INDICES="2 3"
INTEGRATOR="euler"
FUSION_HIDDEN_DIM=128
# No zero_init — learned from shocktube that it traps the model

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Output: $OUTPUT_DIR"
echo "  LR: $LR | Epochs: $NUM_EPOCHS"
echo "  Seq Len: $SEQ_LEN | Stride: $STRIDE"
echo "  Num Layers: $NUM_LAYERS"
echo "  Fusion: concat+MLP (hidden=$FUSION_HIDDEN_DIM, no zero_init)"
echo "  Scheduled Sampling: $SS_SCHEDULE ($SS_INITIAL_RATIO -> $SS_FINAL_RATIO)"
echo "================================================================"


apptainer run --nv "$CONTAINER" train_gparcv2.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resume "/scratch/jtb3sud/gparcv2/river/concat/best_model.pth" \
    --fresh_scheduler \
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
    --fusion_hidden_dim "$FUSION_HIDDEN_DIM" \
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