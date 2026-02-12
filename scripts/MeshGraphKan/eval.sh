#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J mgkan_eval
#SBATCH -o mgkan_eval.out
#SBATCH -e mgkan_eval.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=40G

echo "================================================================"
echo "MeshGraphKAN EVALUATION"
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# PATHS
# ============================================================
MODEL_DIR="/scratch/jtb3sud/elasto_meshgraphkan/run1"
MODEL_PATH="${MODEL_DIR}/best_model.pth"
TEST_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/test"
OUTPUT_DIR="${MODEL_DIR}/eval"
NORM_STATS="${MODEL_DIR}/normalization_stats.json"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# EVAL SETTINGS
# ============================================================
EVAL_MODE="both"          # rollout, snapshot, or both
ROLLOUT_STEPS=37
MAX_SEQUENCES=10
CREATE_GIFS="--create_gifs"
NUM_VIZ=3
GIF_FPS=10
GIF_FRAME_SKIP=1

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Test dir: $TEST_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Eval mode: $EVAL_MODE"
echo "  Rollout steps: $ROLLOUT_STEPS"
echo "  Max sequences: $MAX_SEQUENCES"
echo "  Create GIFs: yes"
echo "================================================================"

apptainer run --nv "$CONTAINER" eval.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --norm_stats_file "$NORM_STATS" \
    --eval_mode "$EVAL_MODE" \
    --rollout_steps "$ROLLOUT_STEPS" \
    --max_sequences "$MAX_SEQUENCES" \
    $CREATE_GIFS \
    --num_viz_simulations "$NUM_VIZ" \
    --gif_fps "$GIF_FPS" \
    --gif_frame_skip "$GIF_FRAME_SKIP"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation complete!"
    echo "  Results: $OUTPUT_DIR"
    echo "  Metrics: rollout_metrics.json, snapshot_metrics.json"
    echo "  Dashboard: rollout_dashboard.png, snapshot_dashboard.png"
else
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE