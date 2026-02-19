#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J eval_river_MGKAN
#SBATCH -o eval_river_MGKAN.out
#SBATCH -e eval_river_MGKAN.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=40G
echo "================================================================"
echo "MeshGraphKAN Evaluation — River / Flood"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"
module purge
module load apptainer
# ============================================================
# PATHS — update these to match your training output
# ============================================================
MODEL_PATH="/scratch/jtb3sud/delta/river/best_model.pth"
TEST_DIR="/standard/sds_baek_energetic/HEC_RAS (River)/pt_test_normalized"
NORM_METADATA="/scratch/jtb3sud/combined/normalization_metadata.json"
OUTPUT_DIR="/scratch/jtb3sud/delta/river/eval"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# Denormalization — extrema .pth is preferred over JSON metadata
# Denormalization extrema for physical units (optional but recommended)
EXTREMA_PATH="/standard/sds_baek_energetic/HEC_RAS (River)/global_y_extrema_test.pth"

# HEC-RAS geometry for PolyCollection mesh rendering (optional)
# Without this, GIFs will use scatter plots instead of filled mesh cells
HEC_RAS_DIR="/standard/sds_baek_energetic/HEC_RAS (River)"


# ============================================================
# EVALUATION SETTINGS
# ============================================================
ROLLOUT_STEPS=60
EVAL_MODE="rollout"
CREATE_GIFS="--create_gifs"
NUM_VIZ=3
GIF_FPS=5
GIF_FRAME_SKIP=1

mkdir -p "$OUTPUT_DIR"
echo ""
echo "Configuration:"
echo "  Model:         $MODEL_PATH"
echo "  Test dir:      $TEST_DIR"
echo "  Norm metadata: $NORM_METADATA"
echo "  Extrema:       $EXTREMA_PATH"
echo "  HEC-RAS dir:   $HEC_RAS_DIR"
echo "  Output:        $OUTPUT_DIR"
echo "  Rollout steps: $ROLLOUT_STEPS"
echo "  Eval mode:     $EVAL_MODE"
echo "  GIFs:          yes ($NUM_VIZ sims, ${GIF_FPS}fps)"
echo "================================================================"

# Build optional args
EXTREMA_ARG=""
if [ -n "$EXTREMA_PATH" ] && [ -f "$EXTREMA_PATH" ]; then
    EXTREMA_ARG="--extrema_path $EXTREMA_PATH"
fi

HEC_RAS_ARG=""
if [ -n "$HEC_RAS_DIR" ] && [ -d "$HEC_RAS_DIR" ]; then
    HEC_RAS_ARG="--hec_ras_dir $HEC_RAS_DIR"
fi

# ============================================================
# RUN
# ============================================================
apptainer run --nv "$CONTAINER" python eval_river.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --norm_metadata "$NORM_METADATA" \
    --output_dir "$OUTPUT_DIR" \
    --eval_mode "$EVAL_MODE" \
    --rollout_steps "$ROLLOUT_STEPS" \
    --device cuda \
    --create_gifs \
    --num_viz_simulations "$NUM_VIZ" \
    --gif_fps "$GIF_FPS" \
    --gif_frame_skip "$GIF_FRAME_SKIP" \
    --extrema_path "$EXTREMA_PATH" \
    --hec_ras_dir "$HEC_RAS_DIR"

EXIT_CODE=$?
echo ""
echo "================================================================"
echo "End time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation complete!"
    echo "  Metrics:   $OUTPUT_DIR/${EVAL_MODE}_metrics.json"
    echo "  Dashboard: $OUTPUT_DIR/${EVAL_MODE}_dashboard.png"
    echo "  GIFs:      $OUTPUT_DIR/${EVAL_MODE}_*.gif"
else
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
fi
echo "================================================================"
exit $EXIT_CODE
