#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J shock_compare
#SBATCH -o shock_compare.out
#SBATCH -e shock_compare.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=40G

echo "================================================================"
echo "SHOCK TUBE MODEL COMPARISON"
echo "================================================================"
echo "Start: $(date)"

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# DATA
# ============================================================
TEST_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/test_cases_normalized"
OUTPUT_DIR="/scratch/jtb3sud/shocktube_comparison"

# ============================================================
# CHECKPOINTS
# ============================================================
GPARCV1_CKPT="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/shock_tube_20250927_104720_run_mod10_750/shock_tube_best_model.pth"
GPARCV2_CKPT="/scratch/jtb3sud/shocktube_v2_training/nospadeFAST/best_model.pth"
MGKAN_CKPT="/scratch/jtb3sud/delta/shocktube/run_101_300/best_model.pth"
MGNET_CKPT="/scratch/jtb3sud/meshgraphnet/shocktube/run1/best_model.pt"

# ============================================================
# CONFIG
# ============================================================
ROLLOUT_STEPS=40
NUM_VIZ=5
MAX_SIMS=20

mkdir -p "$OUTPUT_DIR"

# Build model list dynamically (skip missing checkpoints)
MODELS=""
if [ -f "$GPARCV1_CKPT" ]; then
    MODELS="$MODELS gparcv1:$GPARCV1_CKPT"
    echo "  G-PARCv1:     $GPARCV1_CKPT"
else
    echo "  G-PARCv1:     NOT FOUND"
fi

if [ -f "$GPARCV2_CKPT" ]; then
    MODELS="$MODELS gparcv2:$GPARCV2_CKPT"
    echo "  G-PARCv2:     $GPARCV2_CKPT"
else
    echo "  G-PARCv2:     NOT FOUND"
fi

if [ -f "$MGKAN_CKPT" ]; then
    MODELS="$MODELS mgkan:$MGKAN_CKPT"
    echo "  MeshGraphKAN: $MGKAN_CKPT"
else
    echo "  MeshGraphKAN: NOT FOUND"
fi

if [ -f "$MGNET_CKPT" ]; then
    MODELS="$MODELS mgnet:$MGNET_CKPT"
    echo "  MeshGraphNet: $MGNET_CKPT"
else
    echo "  MeshGraphNet: NOT FOUND"
fi

if [ -z "$MODELS" ]; then
    echo "ERROR: No model checkpoints found!"
    exit 1
fi

echo ""
echo "Test data:     $TEST_DIR"
echo "Output:        $OUTPUT_DIR"
echo "Rollout steps: $ROLLOUT_STEPS"
echo "================================================================"

apptainer run --nv "$CONTAINER" eval_comparison.py \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --models $MODELS \
    --rollout_steps "$ROLLOUT_STEPS" \
    --num_viz "$NUM_VIZ" \
    --max_sims "$MAX_SIMS" \
    --device cuda

EXIT_CODE=$?

echo ""
echo "End: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Comparison complete! Results in $OUTPUT_DIR"
else
    echo "❌ Failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE