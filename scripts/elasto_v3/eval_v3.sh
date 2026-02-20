#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J v3_eval
#SBATCH -o v3_eval.out
#SBATCH -e v3_eval.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH -t 2:00:00
#SBATCH -c 4
#SBATCH --mem=40G

echo "================================================================"
echo "G-PARCv3 EVALUATION — EROSION-AWARE"
echo "================================================================"
echo "Start: $(date)"

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# PATHS
# ============================================================
TEST_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/test"
CHECKPOINT="/scratch/jtb3sud/elasto_v3/best_model.pth"
OUTPUT_DIR="/scratch/jtb3sud/elasto_v3/eval"

# ============================================================
# EVAL CONFIG
# ============================================================
NUM_VIZ=10
MAX_SIMS=""   # Leave empty for all, or set e.g. "20"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Test data:   $TEST_DIR"
echo "Checkpoint:  $CHECKPOINT"
echo "Output:      $OUTPUT_DIR"
echo "================================================================"

CMD="eval.py \
    --test_dir $TEST_DIR \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --num_viz $NUM_VIZ \
    --device cuda"

if [ -n "$MAX_SIMS" ]; then
    CMD="$CMD --max_sims $MAX_SIMS"
fi

apptainer run --nv "$CONTAINER" $CMD

EXIT_CODE=$?

echo ""
echo "End: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation complete! Results in $OUTPUT_DIR"
else
    echo "❌ Failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
