#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J elasto_MGN_baseline
#SBATCH -o elasto_MGN_baseline.out
#SBATCH -e elasto_MGN_baseline.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 72:00:00
#SBATCH -c 8
#SBATCH --mem=80G

echo "================================================================"
echo "MeshGraphNets ElastoPlastic Training"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""
echo "CONFIGURATION:"
echo "  - MODEL: MeshGraphNets (DeepMind 2021)"
echo "  - ARCHITECTURE: Encoder-Processor-Decoder"
echo "  - MESSAGE PASSING: Graph Neural Network"
echo "  - NORMALIZATION: LayerNorm"
echo "  - ACTIVATION: ReLU"
echo "================================================================"

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
module purge
module load apptainer

# Performance tuning for PyG/A100
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# PATHS
# ============================================================
DATA_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small"
OUTPUT_DIR="/scratch/jtb3sud/meshgraphnet_baseline"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
NUM_EPOCHS=1000
SEQ_LEN=10          # Sequence length for temporal windows
STRIDE=1            # Sliding window stride
LR=1e-3             # Learning rate (standard for MeshGraphNets)
WEIGHT_DECAY=5e-4   # Weight decay for regularization
NUM_WORKERS=4       # Parallel CPU workers for data loading
BATCH_SIZE=1        # Not used (dataset yields sequences)

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
HIDDEN_DIM=128              # Hidden dimension (matches original paper)
NUM_LAYERS=4               # Message passing layers (15 in NVIDIA, 10 baseline)
NUM_STATIC_FEATS=2          # x_pos, y_pos
NUM_DYNAMIC_FEATS=2         # U_x, U_y

# ============================================================
# OPTIMIZATION
# ============================================================
OPTIMIZER="adam"            # Optimizer type
SCHEDULER="step"            # Learning rate scheduler
SCHEDULER_STEP=100          # Decay every N epochs
SCHEDULER_GAMMA=0.9         # Decay factor

# ============================================================
# LOGGING & CHECKPOINTING
# ============================================================
VAL_EVERY=10                # Validate every N epochs
SAVE_EVERY=100              # Save checkpoint every N epochs
MAX_STATS_SAMPLES=1000      # Max samples for normalization stats

# ============================================================
# DEVICE
# ============================================================
DEVICE="cuda"               # Use GPU

echo ""
echo "================================================================"
echo "HYPERPARAMETERS"
echo "================================================================"
echo "Data:"
echo "  - Data directory: $DATA_DIR"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Sequence length: $SEQ_LEN"
echo "  - Stride: $STRIDE"
echo ""
echo "Model Architecture:"
echo "  - Hidden dimension: $HIDDEN_DIM"
echo "  - Message passing layers: $NUM_LAYERS"
echo "  - Static features: $NUM_STATIC_FEATS"
echo "  - Dynamic features: $NUM_DYNAMIC_FEATS"
echo ""
echo "Training:"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Learning rate: $LR"
echo "  - Weight decay: $WEIGHT_DECAY"
echo "  - Optimizer: $OPTIMIZER"
echo "  - LR scheduler: $SCHEDULER"
echo ""
echo "System:"
echo "  - Device: $DEVICE"
echo "  - Workers: $NUM_WORKERS"
echo "  - GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || echo 'N/A')"
echo "================================================================"
echo ""

# ============================================================
# RUN TRAINING
# ============================================================
echo "Starting training..."
echo ""

apptainer run --nv "$CONTAINER" train.py \
    --data_dir "$DATA_DIR" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --max_stats_samples "$MAX_STATS_SAMPLES" \
    --hidden_dim "$HIDDEN_DIM" \
    --num_layers "$NUM_LAYERS" \
    --epochs "$NUM_EPOCHS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --optimizer "$OPTIMIZER" \
    --scheduler "$SCHEDULER" \
    --scheduler_step "$SCHEDULER_STEP" \
    --scheduler_gamma "$SCHEDULER_GAMMA" \
    --device "$DEVICE" \
    --num_workers "$NUM_WORKERS" \
    --checkpoint_dir "$OUTPUT_DIR" \
    --val_every "$VAL_EVERY" \
    --save_every "$SAVE_EVERY"

EXIT_CODE=$?

# ============================================================
# COMPLETION
# ============================================================
echo ""
echo "================================================================"
echo "End time: $(date)"
echo "================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "Output files:"
    echo "  - Best model: $OUTPUT_DIR/best_model.pt"
    echo "  - Final model: $OUTPUT_DIR/final_model.pt"
    echo "  - Normalization stats: $OUTPUT_DIR/normalization_stats.pt"
    echo ""
    echo "To evaluate the model, run:"
    echo "  python evaluate_meshgraphnet.py --checkpoint $OUTPUT_DIR/best_model.pt"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo "Check error file: elasto_MGN_baseline.err"
fi

exit $EXIT_CODE
