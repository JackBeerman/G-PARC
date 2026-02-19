#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J shocktube_MGN
#SBATCH -o shocktube_MGN.out
#SBATCH -e shocktube_MGN.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 24:00:00
#SBATCH -c 8
#SBATCH --mem=40G

echo "================================================================"
echo "MeshGraphNet — Shock Tube (Delta Formulation)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""
echo "CONVENTION: Model predicts DELTA, rollout accumulates current + pred"
echo "DATA: Pre-normalized (global-max), NO z-score"
echo "SKIP: y_momentum (noise), INJECT: [pressure, density, delta_t]"
echo "================================================================"

# ============================================================
# ENVIRONMENT
# ============================================================
module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# PATHS
# ============================================================
TRAIN_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/train_cases_normalized"
VAL_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/val_cases_normalized"
OUTPUT_DIR="/scratch/jtb3sud/meshgraphnet/shocktube/run1"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

mkdir -p "$OUTPUT_DIR"

# ============================================================
# HYPERPARAMETERS (standardized)
# ============================================================
NUM_EPOCHS=500           # Match MeshGraphKAN shocktube (was 100, increase for fairness)
SEQ_LEN=2              # Shocktube sims are shorter
STRIDE=1
LR=1e-4                 # NVIDIA canonical MGN LR
WEIGHT_DECAY=5e-4
GRAD_CLIP=1.0
NUM_WORKERS=4

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
HIDDEN_DIM=128
NUM_LAYERS=5            # NVIDIA default
NUM_STATIC_FEATS=2       # x_pos, y_pos

# ============================================================
# SCHEDULE
# ============================================================
SCHEDULER="cosine"

# ============================================================
# LOGGING
# ============================================================
VAL_EVERY=10
SAVE_EVERY=50

echo ""
echo "================================================================"
echo "CONFIGURATION"
echo "================================================================"
echo "Train:   $TRAIN_DIR"
echo "Val:     $VAL_DIR"
echo "Output:  $OUTPUT_DIR"
echo ""
echo "Architecture:"
echo "  Hidden dim:       $HIDDEN_DIM"
echo "  Msg-pass layers:  $NUM_LAYERS"
echo "  Input dim:        8 (static=2 + dynamic=3 + global=3)"
echo "  Output dim:       3 (density, x_mom, energy)"
echo ""
echo "Training:"
echo "  Epochs:     $NUM_EPOCHS"
echo "  LR:         $LR (cosine -> ~0)"
echo "  Seq length: $SEQ_LEN, stride: $STRIDE"
echo "  Grad clip:  $GRAD_CLIP"
echo ""
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || echo 'N/A')"
echo "================================================================"
echo ""

# ============================================================
# RUN
# ============================================================

apptainer run --nv "$CONTAINER" train_shocktube.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --num_static_feats "$NUM_STATIC_FEATS" \
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
