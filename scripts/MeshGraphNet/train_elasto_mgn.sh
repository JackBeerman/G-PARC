#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J elasto_MGN
#SBATCH -o elasto_MGN.out
#SBATCH -e elasto_MGN.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 72:00:00
#SBATCH -c 8
#SBATCH --mem=80G

echo "================================================================"
echo "MeshGraphNet — Elastoplastic (Delta Formulation)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""
echo "CONVENTION: Model predicts DELTA, rollout accumulates current + pred"
echo "DATA: Pre-normalized (global-max), NO z-score"
echo "================================================================"

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# PATHS
# ============================================================
DATA_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small"
OUTPUT_DIR="/scratch/jtb3sud/meshgraphnet/elasto/run1"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

mkdir -p "$OUTPUT_DIR"

# ============================================================
# HYPERPARAMETERS (standardized across baselines)
# ============================================================
NUM_EPOCHS=1500          # Match G-PARCv2 / MeshGraphKAN
SEQ_LEN=16              # Match G-PARCv2 / MeshGraphKAN
STRIDE=16
LR=1e-4                 # NVIDIA canonical MGN LR
WEIGHT_DECAY=5e-4
GRAD_CLIP=1.0
NUM_WORKERS=4

# ============================================================
# MODEL ARCHITECTURE (NVIDIA defaults)
# ============================================================
HIDDEN_DIM=128           # Match NVIDIA / G-PARCv2
NUM_LAYERS=4            # NVIDIA default (Pfaff et al.)

# ============================================================
# SCHEDULE
# ============================================================
SCHEDULER="cosine"       # Match G-PARCv2 / MeshGraphKAN

# ============================================================
# LOGGING
# ============================================================
VAL_EVERY=10
SAVE_EVERY=100

echo ""
echo "================================================================"
echo "CONFIGURATION"
echo "================================================================"
echo "Data:    $DATA_DIR"
echo "Output:  $OUTPUT_DIR"
echo ""
echo "Architecture:"
echo "  Hidden dim:       $HIDDEN_DIM"
echo "  Msg-pass layers:  $NUM_LAYERS (NVIDIA default)"
echo ""
echo "Training:"
echo "  Epochs:     $NUM_EPOCHS"
echo "  LR:         $LR (cosine -> ~0)"
echo "  Seq length: $SEQ_LEN"
echo "  Stride:     $STRIDE"
echo "  Grad clip:  $GRAD_CLIP"
echo ""
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || echo 'N/A')"
echo "================================================================"
echo ""

# ============================================================
# RUN
# ============================================================

apptainer run --nv "$CONTAINER" train_elasto.py \
    --data_dir "$DATA_DIR" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --hidden_dim "$HIDDEN_DIM" \
    --num_layers "$NUM_LAYERS" \
    --epochs "$NUM_EPOCHS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --grad_clip_norm "$GRAD_CLIP" \
    --scheduler "$SCHEDULER" \
    --device cuda \
    --num_workers "$NUM_WORKERS" \
    --checkpoint_dir "$OUTPUT_DIR" \
    --val_every "$VAL_EVERY" \
    --save_every "$SAVE_EVERY"

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "End time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "  Best model:  $OUTPUT_DIR/best_model.pt"
    echo "  Config:      $OUTPUT_DIR/config.json"
else
    echo "Training failed with exit code $EXIT_CODE"
fi
echo "================================================================"

exit $EXIT_CODE
