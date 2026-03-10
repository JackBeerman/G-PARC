#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J eval_cyl_v21
#SBATCH -o eval_cyl_v21.out
#SBATCH -e eval_cyl_v21.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH -t 01:00:00
#SBATCH -c 4
#SBATCH --mem=64G

echo "================================================================"
echo "G-PARCv2 CYLINDER FLOW EVALUATION"
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# PATHS — ADJUST THESE
# ============================================================
DATA_ROOT="/standard/sds_baek_energetic/von_karman_vortex/full_data/split_normalized"
TEST_DIR="${DATA_ROOT}/train"
CHECKPOINT_DIR="/scratch/jtb3sud/gparcv2/cylinder"
CHECKPOINT="${CHECKPOINT_DIR}/best_model.pth"
OUTPUT_DIR="${CHECKPOINT_DIR}/eval/1"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# EVAL SETTINGS
# ============================================================
FPS=8
FRAME_SKIP=1        # Render every 2nd frame (200 frames for 400 timestep sim)
# MAX_SIMS=5         # Uncomment to limit number of sims evaluated

mkdir -p "$OUTPUT_DIR"

echo "Checkpoint: $CHECKPOINT"
echo "Test dir:   $TEST_DIR"
echo "Output:     $OUTPUT_DIR"
echo "================================================================"

apptainer run --nv "$CONTAINER" eval.py \
    --checkpoint "$CHECKPOINT" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --fps "$FPS" \
    --frame_skip "$FRAME_SKIP" \
    --num_rollout_steps 50 \
    --device cuda
    # --max_sims $MAX_SIMS  # Uncomment to limit

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation complete! Results in $OUTPUT_DIR"
else
    echo "❌ Evaluation failed"
fi

exit $EXIT_CODE