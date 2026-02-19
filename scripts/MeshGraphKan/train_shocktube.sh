#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J mgkan_shock
#SBATCH -o mgkan_shock.out
#SBATCH -e mgkan_shock.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 24:00:00
#SBATCH -c 8
#SBATCH --mem=80G

echo "================================================================"
echo "MeshGraphKAN: SHOCK TUBE TRAINING"
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
TRAIN_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/train_cases_normalized"
VAL_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/val_cases_normalized"
OUTPUT_DIR="/scratch/jtb3sud/delta/shocktube/run_101_300"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"
SCRIPT_DIR="$HOME/G-PARC/scripts/MeshGraphKan"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
NUM_EPOCHS=200
SEQ_LEN=4
STRIDE=4
LR=1e-5
NUM_WORKERS=0
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
# SHOCK TUBE SPECIFICS
# ============================================================
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=3          # After skipping y_momentum
SKIP_DYNAMIC_INDICES="2"     # Skip y_momentum (raw index 2)
GLOBAL_PARAM_DIM=3           # pressure, density_param, delta_t

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Train: $TRAIN_DIR"
echo "  Val:   $VAL_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "  LR: $LR | Epochs: $NUM_EPOCHS"
echo "  Seq length: $SEQ_LEN, stride: $STRIDE"
echo "  Dynamic feats: $NUM_DYNAMIC_FEATS (skip raw: $SKIP_DYNAMIC_INDICES)"
echo "  Global params: $GLOBAL_PARAM_DIM (concatenated as node features)"
echo "  Input dim: $((NUM_STATIC_FEATS + NUM_DYNAMIC_FEATS + GLOBAL_PARAM_DIM))"
echo ""
echo "  Hidden dim: $HIDDEN_DIM"
echo "  Processor layers: $PROCESSOR_SIZE"
echo "  KAN harmonics: $NUM_HARMONICS"
echo "  Prediction mode: delta"
echo "================================================================"

apptainer run --nv "$CONTAINER" train_shocktube.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --lr "$LR" \
    --resume "/scratch/jtb3sud/delta/shocktube/best_model.pth" \
    --fresh_scheduler \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --skip_dynamic_indices $SKIP_DYNAMIC_INDICES \
    --global_param_dim "$GLOBAL_PARAM_DIM" \
    --hidden_dim "$HIDDEN_DIM" \
    --processor_size "$PROCESSOR_SIZE" \
    --num_harmonics "$NUM_HARMONICS" \
    --aggregation "$AGGREGATION" \
    --mlp_activation "$ACTIVATION" \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    --num_workers "$NUM_WORKERS" \
    --ss_schedule "$SS_SCHEDULE" \
    --ss_initial_ratio "$SS_INITIAL_RATIO" \
    --ss_final_ratio "$SS_FINAL_RATIO"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training complete!"
    echo "  Best model: $OUTPUT_DIR/best_model.pth"
else
    echo "Training failed with exit code $EXIT_CODE"
fi
echo "End time: $(date)"

exit $EXIT_CODE