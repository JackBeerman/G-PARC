#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J river_comparison
#SBATCH -o river_comparison.out
#SBATCH -e river_comparison.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:20:00
#SBATCH -c 4
#SBATCH --mem=40G

echo "================================================================"
echo "RIVER MODEL COMPARISON EVALUATION"
echo "================================================================"
echo "Start: $(date)"

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# DATA
# ============================================================
BASE_DATA="/standard/sds_baek_energetic/HEC_RAS (River)"
TEST_DIR="${BASE_DATA}/pt_test_normalized"
EXTREMA_PATH="${BASE_DATA}/global_y_extrema_test.pth"
OUTPUT_DIR="/scratch/jtb3sud/river_comparison"

# ============================================================
# MODEL CHECKPOINTS — update these paths
# ============================================================
GPARCV1_CKPT="/home/jtb3sud/G-PARC/weights/new_river/modelseq20_ep250.pth"
GPARCV2_CKPT="/scratch/jtb3sud/river_v2_training_scheduled/best_model.pth"
MGKAN_CKPT="/scratch/jtb3sud/delta/river/best_model.pth"
MGNET_CKPT="/scratch/jtb3sud/meshgraphnet/river/run1/best_model.pt"

# ============================================================
# TIME SEGMENTS (matching your event structure)
# ============================================================
SEGMENTS="0:22,22:64,64:79,79:97,97:111"

# ============================================================
# BUILD MODEL LIST (only include models that exist)
# ============================================================
MODELS=""

if [ -f "$GPARCV1_CKPT" ]; then
    MODELS="$MODELS gparcv1:$GPARCV1_CKPT"
    echo "  ✓ G-PARCv1: $GPARCV1_CKPT"
else
    echo "  ✗ G-PARCv1: not found at $GPARCV1_CKPT"
fi

if [ -f "$GPARCV2_CKPT" ]; then
    MODELS="$MODELS gparcv2:$GPARCV2_CKPT"
    echo "  ✓ G-PARCv2: $GPARCV2_CKPT"
else
    echo "  ✗ G-PARCv2: not found at $GPARCV2_CKPT"
fi

if [ -f "$MGKAN_CKPT" ]; then
    MODELS="$MODELS mgkan:$MGKAN_CKPT"
    echo "  ✓ MeshGraphKAN: $MGKAN_CKPT"
else
    echo "  ✗ MeshGraphKAN: not found at $MGKAN_CKPT"
fi

if [ -f "$MGNET_CKPT" ]; then
    MODELS="$MODELS mgnet:$MGNET_CKPT"
    echo "  ✓ MeshGraphNet: $MGNET_CKPT"
else
    echo "  ✗ MeshGraphNet: not found at $MGNET_CKPT"
fi

if [ -z "$MODELS" ]; then
    echo "❌ No model checkpoints found. Exiting."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Test data: $TEST_DIR"
echo "Extrema:   $EXTREMA_PATH"
echo "Output:    $OUTPUT_DIR"
echo "Segments:  $SEGMENTS"
echo "================================================================"

apptainer run --nv "$CONTAINER" eval_comparison.py \
    --test_dir "$TEST_DIR" \
    --extrema_path "$EXTREMA_PATH" \
    --models $MODELS \
    --output_dir "$OUTPUT_DIR" \
    --segments "$SEGMENTS" \
    --depth_threshold 0.3 \
    --num_viz 3 \
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