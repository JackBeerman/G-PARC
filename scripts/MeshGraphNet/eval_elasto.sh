#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J eval_elasto_MGN
#SBATCH -o eval_elasto_MGN.out
#SBATCH -e eval_elasto_MGN.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=40G

echo "================================================================"
echo "MeshGraphNet Evaluation — Elastoplastic"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

module purge
module load apptainer

# ============================================================
# PATHS
# ============================================================
MODEL_PATH="/scratch/jtb3sud/meshgraphnet/elasto/run1/best_model.pt"
TEST_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/test"
NORM_STATS="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/normalization_stats.json"
OUTPUT_DIR="/scratch/jtb3sud/meshgraphnet/elasto/run1/eval/test"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# EVAL SETTINGS
# ============================================================
EVAL_MODE="both"
ROLLOUT_STEPS=37
MAX_SEQUENCES=10
NUM_VIZ=3
GIF_FPS=10
GIF_FRAME_SKIP=1

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Model:         $MODEL_PATH"
echo "  Test dir:      $TEST_DIR"
echo "  Norm stats:    $NORM_STATS"
echo "  Output:        $OUTPUT_DIR"
echo "  Eval mode:     $EVAL_MODE"
echo "  Rollout steps: $ROLLOUT_STEPS"
echo "  Max sequences: $MAX_SEQUENCES"
echo "  Create GIFs:   yes"
echo "================================================================"

apptainer run --nv "$CONTAINER" eval_elasto.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --norm_stats_file "$NORM_STATS" \
    --eval_mode "$EVAL_MODE" \
    --rollout_steps "$ROLLOUT_STEPS" \
    --max_sequences "$MAX_SEQUENCES" \
    --create_gifs \
    --num_viz_simulations "$NUM_VIZ" \
    --gif_fps "$GIF_FPS" \
    --gif_frame_skip "$GIF_FRAME_SKIP"

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "End time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation complete!"
    echo "  Metrics: $OUTPUT_DIR/"
else
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
fi
echo "================================================================"
exit $EXIT_CODE