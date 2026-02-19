#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J mgkan_river
#SBATCH -o mgkan_river.out
#SBATCH -e mgkan_river.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH -t 72:00:00
#SBATCH -c 8
#SBATCH --mem=80G

echo "================================================================"
echo "MeshGraphKAN: RIVER TRAINING"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# DATA DIRECTORIES
# ============================================================
BASE_DATA="/standard/sds_baek_energetic/HEC_RAS (River)"
TRAIN_DIR="${BASE_DATA}/pt_train_normalized"
VAL_DIR="${BASE_DATA}/pt_val_normalized"
OUTPUT_DIR="/scratch/jtb3sud/delta/river/run2_51_150"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"
SCRIPT_DIR="$HOME/G-PARC/scripts/MeshGraphKan"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
NUM_EPOCHS=150
SEQ_LEN=1
STRIDE=1
LR=5e-5
NUM_WORKERS=4
GRAD_CLIP_NORM=1.0

# ============================================================
# SCHEDULED SAMPLING
# ============================================================
SS_SCHEDULE="linear"
SS_INITIAL_RATIO=0.0
SS_FINAL_RATIO=0.0

# ============================================================
# MeshGraphKAN ARCHITECTURE
# ============================================================
HIDDEN_DIM=128
PROCESSOR_SIZE=4
NUM_HARMONICS=5
AGGREGATION="sum"
ACTIVATION="relu"

# ============================================================
# DATASET
# ============================================================
NUM_STATIC_FEATS=9
NUM_DYNAMIC_FEATS=4

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Train: $TRAIN_DIR"
echo "  Val:   $VAL_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "  LR: $LR (cosine -> ~0 over $NUM_EPOCHS epochs)"
echo "  Seq length: $SEQ_LEN, stride: $STRIDE"
echo "  Teacher forcing: $SS_INITIAL_RATIO -> $SS_FINAL_RATIO ($SS_SCHEDULE)"
echo "  Epochs: $NUM_EPOCHS"
echo ""
echo "  Hidden dim: $HIDDEN_DIM"
echo "  Processor layers: $PROCESSOR_SIZE"
echo "  KAN harmonics: $NUM_HARMONICS"
echo "  Static feats: $NUM_STATIC_FEATS, Dynamic feats: $NUM_DYNAMIC_FEATS"
echo "  Prediction mode: delta"
echo "================================================================"

apptainer run --nv "$CONTAINER" train_river.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --lr "$LR" \
    --resume "/scratch/jtb3sud/delta/river/best_model.pth" \
    --fresh_scheduler \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --hidden_dim "$HIDDEN_DIM" \
    --processor_size "$PROCESSOR_SIZE" \
    --num_harmonics "$NUM_HARMONICS" \
    --aggregation "$AGGREGATION" \
    --mlp_activation "$ACTIVATION" \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    --num_workers "$NUM_WORKERS" \
    --ss_schedule "$SS_SCHEDULE" \
    --ss_initial_ratio "$SS_INITIAL_RATIO" \
    --ss_final_ratio "$SS_FINAL_RATIO" \
    --shuffle

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training complete!"
    echo "  Best model: $OUTPUT_DIR/best_model.pth"
    echo "  Latest model: $OUTPUT_DIR/latest_model.pth"
else
    echo "Training failed with exit code $EXIT_CODE"
fi
echo "End time: $(date)"

exit $EXIT_CODE