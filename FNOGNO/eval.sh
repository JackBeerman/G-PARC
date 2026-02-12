#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J elasto_FNOGNO
#SBATCH -o elasto_FNOGNO_eval.out
#SBATCH -e elasto_FNOGNO_eval.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -c 8
#SBATCH --mem=80G

echo "================================================================"
echo "FNOGNO ElastoPlastic - Baseline Comparison"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""
echo "CONFIGURATION:"
echo "  - MODEL: FNOGNO (Hybrid FNO-GNO)"
echo "  - FNO: Grid-based operator"
echo "  - GNO: Irregular geometry query"
echo "  - PHYSICS: Euler-like (1-step prediction)"
echo "================================================================"

module purge
module load apptainer

# Performance tuning for A100
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Paths
TRAIN_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/test"
OUTPUT_DIR="/scratch/jtb3sud/elasto_fnogno_results/baseline"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# TRAINING HYPERPARAMETERS (Match G-PARC)
# ============================================================
NUM_EPOCHS=50
SEQ_LEN=2           # Match G-PARC
STRIDE=1            # Match G-PARC
LR=1e-5             # Match G-PARC
NUM_WORKERS=0       # FNOGNO uses IterableDataset, keep 0
GRAD_CLIP_NORM=1.0  # Match G-PARC

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
GRID_H=32
GRID_W=64
FNO_MODES_H=8
FNO_MODES_W=16
FNO_HIDDEN=32
FNO_LAYERS=3
GNO_RADIUS=0.033

# ============================================================
# DATA CONFIGURATION
# ============================================================
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=2

echo ""
echo "Hyperparameters:"
echo "  Epochs: $NUM_EPOCHS"
echo "  Sequence Length: $SEQ_LEN"
echo "  Learning Rate: $LR"
echo "  Grad Clip: $GRAD_CLIP_NORM"
echo ""
echo "Model:"
echo "  Grid: ${GRID_H}x${GRID_W}"
echo "  FNO Modes: ${FNO_MODES_H}x${FNO_MODES_W}"
echo "  FNO Hidden: $FNO_HIDDEN"
echo "  FNO Layers: $FNO_LAYERS"
echo "  GNO Radius: $GNO_RADIUS"
echo "================================================================"

mkdir -p "$OUTPUT_DIR"

apptainer run --nv "$CONTAINER" jesus.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --lr "$LR" \
    --weight_decay 1e-4 \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --grid_h "$GRID_H" \
    --grid_w "$GRID_W" \
    --fno_n_modes $FNO_MODES_H $FNO_MODES_W \
    --fno_hidden_channels "$FNO_HIDDEN" \
    --fno_n_layers "$FNO_LAYERS" \
    --gno_radius "$GNO_RADIUS" \
    --use_scheduler \
    --num_workers "$NUM_WORKERS" \
    --device cuda

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "Output files:"
    echo "  Best model:    $OUTPUT_DIR/best_model.pth"
    echo "  Latest model:  $OUTPUT_DIR/latest_model.pth"
    echo "  Config:        $OUTPUT_DIR/config.json"
    echo "  History:       $OUTPUT_DIR/training_history.json"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo "Check error file: elasto_FNOGNO_${SLURM_JOB_ID}.err"
fi

echo "================================================================"

exit $EXIT_CODE