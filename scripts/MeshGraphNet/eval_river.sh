#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J eval_river_MGN
#SBATCH -o eval_river_MGN.out
#SBATCH -e eval_river_MGN.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 0:30:00
#SBATCH -c 4
#SBATCH --mem=40G

echo "================================================================"
echo "MeshGraphNet Evaluation — River / Flood"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

module purge
module load apptainer

# ============================================================
# PATHS
# ============================================================
MODEL_PATH="/scratch/jtb3sud/meshgraphnet/river/run1/best_model.pt"
TEST_DIR="/standard/sds_baek_energetic/HEC_RAS (River)/pt_test_normalized"
OUTPUT_DIR="/scratch/jtb3sud/meshgraphnet/river/run1/eval/test"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"
EXTREMA_PATH="/standard/sds_baek_energetic/HEC_RAS (River)/global_y_extrema_test.pth"
HEC_RAS_DIR="/standard/sds_baek_energetic/HEC_RAS (River)"

# ============================================================
# EVAL SETTINGS
# ============================================================
EVAL_MODE="rollout"
ROLLOUT_STEPS=60
CREATE_GIFS="--create_gifs"
NUM_VIZ=3
GIF_FPS=5
GIF_FRAME_SKIP=1

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Model:         $MODEL_PATH"
echo "  Test dir:      $TEST_DIR"
echo "  Extrema:       $EXTREMA_PATH"
echo "  HEC-RAS dir:   $HEC_RAS_DIR"
echo "  Output:        $OUTPUT_DIR"
echo "  Eval mode:     $EVAL_MODE"
echo "  Rollout steps: $ROLLOUT_STEPS"
echo "  GIFs:          yes ($NUM_VIZ sims, ${GIF_FPS}fps)"
echo "================================================================"

apptainer run --nv "$CONTAINER" eval_river.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --eval_mode "$EVAL_MODE" \
    --rollout_steps "$ROLLOUT_STEPS" \
    --extrema_path "$EXTREMA_PATH" \
    --hec_ras_dir "$HEC_RAS_DIR" \
    $CREATE_GIFS \
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