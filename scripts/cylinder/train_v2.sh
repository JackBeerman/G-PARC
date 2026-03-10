#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J gparc_cyl_v2
#SBATCH -o gparc_cyl_v2.out
#SBATCH -e gparc_cyl_v2.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH -t 72:00:00
#SBATCH -c 8
#SBATCH --mem=80G

echo "================================================================"
echo "G-PARCv2 CYLINDER FLOW (NoSPADE — Concat Fusion)"
echo "================================================================"
echo ""
echo "Architecture:"
echo "  GraphConvFeatureExtractorV2 + CylinderDifferentiator (NoSPADE)"
echo "  MLS advection (v·∇φ) + FD diffusion (∇²φ)"
echo "  Per-variable PhysicsFusionMLP (concat learned + physics → MLP)"
echo "  FiLM conditioning on Reynolds number"
echo "  Euler numerical integration"
echo ""
echo "Data: 60k-100k node cylinder meshes, ~400 timesteps each"
echo "Skip: vz, ωx, ωy (near-zero in 2D flow) → 4 effective features"
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# DATA DIRECTORIES
# ============================================================
DATA_ROOT="/standard/sds_baek_energetic/von_karman_vortex/full_data/split_normalized"
TRAIN_DIR="${DATA_ROOT}/train"
VAL_DIR="${DATA_ROOT}/val"
OUTPUT_DIR="/scratch/jtb3sud/gparcv2/cylinder"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
NUM_EPOCHS=1000
SEQ_LEN=5                # Start conservative for 60-100k node meshes
STRIDE=4
LR=3e-4                  # PLAID authors' LR
NUM_WORKERS=4
GRAD_CLIP=2.0

# ============================================================
# NO TEACHER FORCING — proven best strategy from elasto experiments
# ============================================================
SS_SCHEDULE="linear"
SS_INITIAL_RATIO=0.0
SS_FINAL_RATIO=0.0

# ============================================================
# ARCHITECTURE — 128 hidden, model is lightweight enough for 100k nodes
# ============================================================
NUM_LAYERS=4
HIDDEN_CHANNELS=128
FEATURE_OUT_CHANNELS=128
FUSION_HIDDEN_DIM=128
DROPOUT=0.0
DIFFUSION="mls"           # MLS Laplacian (same quality as gradient solver)

# ============================================================
# PHYSICS FEATURES
# ============================================================
NUM_STATIC_FEATS=3        # x, y, z positions
# Raw dynamic: [0]=p, [1]=vx, [2]=vy, [3]=vz, [4]=ωx, [5]=ωy, [6]=ωz
# Skip vz(3), ωx(4), ωy(5) → post-skip: [0]=p, [1]=vx, [2]=vy, [3]=ωz
SKIP_INDICES="3 4 5"
VEL_INDICES="1 2"         # Post-skip velocity indices for advection

# Global conditioning
GLOBAL_PARAM_DIM=1         # Reynolds number only
GLOBAL_EMBED_DIM=64

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Output: $OUTPUT_DIR"
echo "  LR: $LR (cosine → ~0 over $NUM_EPOCHS epochs)"
echo "  Seq length: $SEQ_LEN"
echo "  Teacher forcing: 0.0 (pure free-running)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Grad clip: $GRAD_CLIP"
echo "  Hidden: $HIDDEN_CHANNELS"
echo "  Skip indices: $SKIP_INDICES"
echo "  Velocity indices: $VEL_INDICES"
echo "================================================================"

apptainer run --nv "$CONTAINER" train_cylinder_v2.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --lr "$LR" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --skip_dynamic_indices $SKIP_INDICES \
    --velocity_indices $VEL_INDICES \
    --global_param_dim "$GLOBAL_PARAM_DIM" \
    --global_embed_dim "$GLOBAL_EMBED_DIM" \
    --integrator "euler" \
    --num_layers "$NUM_LAYERS" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT_CHANNELS" \
    --fusion_hidden_dim "$FUSION_HIDDEN_DIM" \
    --diffusion_type "$DIFFUSION" \
    --dropout "$DROPOUT" \
    --zero_init \
    --grad_clip "$GRAD_CLIP" \
    --num_workers "$NUM_WORKERS" \
    --ss_schedule "$SS_SCHEDULE" \
    --ss_initial_ratio "$SS_INITIAL_RATIO" \
    --ss_final_ratio "$SS_FINAL_RATIO" \
    --device cuda

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training complete!"
else
    echo "❌ Training failed"
fi

exit $EXIT_CODE