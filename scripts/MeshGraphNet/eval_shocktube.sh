#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J eval_shocktube_MGN
#SBATCH -o eval_shocktube_MGN.out
#SBATCH -e eval_shocktube_MGN.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=40G

echo "================================================================"
echo "MeshGraphNet Evaluation — Shock Tube"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

module purge
module load apptainer

# ============================================================
# PATHS
# ============================================================
MODEL_PATH="/scratch/jtb3sud/meshgraphnet/shocktube/run1/best_model.pt"
TEST_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/test_cases_normalized"
NORM_METADATA="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/normalization_metadata.json"
OUTPUT_DIR="/scratch/jtb3sud/meshgraphnet/shocktube/run1/eval/test"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# EVAL SETTINGS
# ============================================================
EVAL_MODE="both"
MAX_SEQUENCES=400
ROLLOUT_STEPS=43  # How many timesteps to predict into the future
CREATE_GIFS="--create_gifs"
NUM_VIZ=3
GIF_FPS=4
GIF_FRAME_SKIP=1

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Model:         $MODEL_PATH"
echo "  Test dir:      $TEST_DIR"
echo "  Norm metadata: $NORM_METADATA"
echo "  Output:        $OUTPUT_DIR"
echo "  Eval mode:     $EVAL_MODE"
echo "  Rollout steps: $ROLLOUT_STEPS"
echo "  Max sequences: $MAX_SEQUENCES"
echo "  GIFs:          yes ($NUM_VIZ sims, ${GIF_FPS}fps)"
echo "================================================================"

apptainer run --nv "$CONTAINER" eval_shocktube.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --norm_metadata_file "$NORM_METADATA" \
    --output_dir "$OUTPUT_DIR" \
    --eval_mode "$EVAL_MODE" \
    --rollout_steps "$ROLLOUT_STEPS" \
    --max_sequences "$MAX_SEQUENCES" \
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