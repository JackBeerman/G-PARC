#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J meshgraphkan
#SBATCH -o meshgraphkan.out
#SBATCH -e meshgraphkan.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH -t 72:00:00
#SBATCH -c 8
#SBATCH --mem=80G

echo "================================================================"
echo "MeshGraphKAN: NVIDIA-FAITHFUL PyG REIMPLEMENTATION"
echo "================================================================"
echo ""
echo "Architecture (verified against NVIDIA PhysicsNeMo source):"
echo "  - Node encoder: KolmogorovArnoldNetwork (learnable Fourier coeffs)"
echo "  - Edge encoder: MeshGraphMLP (2 hidden layers, SiLU, LayerNorm)"
echo "  - Processor: 15 × (MeshEdgeBlock + MeshNodeBlock) with residuals"
echo "  - Decoder: MeshGraphMLP (2 hidden layers, SiLU, NO LayerNorm)"
echo "  - Reimplemented in PyG due to DGL/PyTorch 2.9.1 incompatibility"
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
OUTPUT_DIR="/scratch/jtb3sud/elasto_meshgraphkan/run1"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# TRAINING HYPERPARAMETERS (matched to G-PARC for fair comparison)
# ============================================================
NUM_EPOCHS=1500
SEQ_LEN=16
STRIDE=16
LR=5e-5                  # MeshGraphKAN default (NVIDIA uses 5e-5)
NUM_WORKERS=4
GRAD_CLIP_NORM=1.0

# ============================================================
# SCHEDULED SAMPLING
# Set TF=0.0 throughout to match G-PARC's best strategy
# ============================================================
SS_SCHEDULE="linear"
SS_INITIAL_RATIO=0.0
SS_FINAL_RATIO=0.0

# ============================================================
# MeshGraphKAN ARCHITECTURE (NVIDIA defaults)
# ============================================================
HIDDEN_DIM=128
PROCESSOR_SIZE=4
NUM_HARMONICS=5
AGGREGATION="sum"

# ============================================================
# DATASET
# ============================================================
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=2
MASK_ERODING="--mask_eroding"

NORM_STATS_FILE="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/normalization_stats.json"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Output: $OUTPUT_DIR"
echo "  LR: $LR (cosine → ~0 over $NUM_EPOCHS epochs)"
echo "  Seq length: $SEQ_LEN"
echo "  Teacher forcing: $SS_INITIAL_RATIO → $SS_FINAL_RATIO ($SS_SCHEDULE)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Grad clip: $GRAD_CLIP_NORM"
echo ""
echo "  Hidden dim: $HIDDEN_DIM"
echo "  Processor layers: $PROCESSOR_SIZE"
echo "  KAN harmonics: $NUM_HARMONICS"
echo "  Aggregation: $AGGREGATION"
echo "================================================================"

if [ -f "$NORM_STATS_FILE" ]; then
    cp "$NORM_STATS_FILE" "$OUTPUT_DIR/normalization_stats.json"
    echo "✓ Copied normalization stats to output directory"
fi

apptainer run --nv "$CONTAINER" train.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --lr "$LR" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --hidden_dim "$HIDDEN_DIM" \
    --processor_size "$PROCESSOR_SIZE" \
    --num_harmonics "$NUM_HARMONICS" \
    --aggregation "$AGGREGATION" \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    $MASK_ERODING \
    --num_workers "$NUM_WORKERS" \
    --ss_schedule "$SS_SCHEDULE" \
    --ss_initial_ratio "$SS_INITIAL_RATIO" \
    --ss_final_ratio "$SS_FINAL_RATIO"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ MeshGraphKAN training complete!"
    echo "  Best model: $OUTPUT_DIR/best_model.pth"
    echo "  Latest model: $OUTPUT_DIR/latest_model.pth"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE