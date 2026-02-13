#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J eval_river_v2
#SBATCH -o eval_river_v2.out
#SBATCH -e eval_river_v2.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:30:00
#SBATCH -c 8
#SBATCH --mem=64G

echo "================================================================"
echo "EVALUATION: G-PARCv2 River Model"
echo "  Static (9): x, y, Area, Elevation, Slope, Aspect, Curvature, Manning's n, FA"
echo "  Dynamic (4): Depth, Volume, Vel_X, Vel_Y"
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# PATHS  (EDIT THESE)
# ============================================================
MODEL_PATH="/scratch/jtb3sud/river_v2_training_scheduled/best_model.pth"
TEST_DIR="/standard/sds_baek_energetic/HEC_RAS (River)/pt_val_normalized"
OUTPUT_DIR="/scratch/jtb3sud/river_v2_training_scheduled/evaluation"

# HEC-RAS mesh files for PolyCollection rendering (optional but recommended)
HEC_RAS_DIR="/standard/sds_baek_energetic/HEC_RAS (River)"

# Denormalization extrema for physical units (optional but recommended)
EXTREMA_PATH="/standard/sds_baek_energetic/HEC_RAS (River)/global_y_extrema_test.pth"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# ARCHITECTURE (GraphConv V2)
# ============================================================
NUM_LAYERS=2            # Standard for GraphConv V2
HIDDEN_CHANNELS=64
FEATURE_OUT_CHANNELS=128
DROPOUT=0.2
USE_LAYER_NORM="--use_layer_norm"
USE_RELATIVE_POS="--use_relative_pos"

# ============================================================
# PHYSICS & DIFFERENTIATOR
# ============================================================
NUM_STATIC_FEATS=9
NUM_DYNAMIC_FEATS=4
VELOCITY_INDICES="2 3"
INTEGRATOR="euler"
SPADE_HEADS=4
SPADE_CONCAT="--spade_concat"
SPADE_DROPOUT=0.1
ZERO_INIT="--zero_init"

# ============================================================
# EVALUATION SETTINGS
# ============================================================
EVAL_MODE="both"                 # rollout, snapshot, or both
MAX_SEQUENCES=""                 # empty = all, or set e.g. 10
ROLLOUT_STEPS=50                 # number of autoregressive steps

# ============================================================
# VISUALIZATION SETTINGS
# ============================================================
CREATE_GIFS="--create_gifs"      # comment out to skip GIFs
NUM_VIZ_SIMS=3
VIZ_MODE="representative"        # representative, best, worst, all
GIF_FPS=5
GIF_FRAME_SKIP=1

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Model:      $MODEL_PATH"
echo "  Test data:  $TEST_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  HEC-RAS:    $HEC_RAS_DIR"
echo "  Extrema:    $EXTREMA_PATH"
echo ""
echo "Architecture:"
echo "  Layers: $NUM_LAYERS, Hidden: $HIDDEN_CHANNELS, Output: $FEATURE_OUT_CHANNELS"
echo "  Layer norm: enabled, Relative pos: enabled"
echo ""
echo "Physics:"
echo "  Static: $NUM_STATIC_FEATS, Dynamic: $NUM_DYNAMIC_FEATS"
echo "  Velocity indices: $VELOCITY_INDICES"
echo "  Integrator: $INTEGRATOR"
echo ""
echo "Evaluation:"
echo "  Mode: $EVAL_MODE, Rollout steps: $ROLLOUT_STEPS"
echo "  Viz: $NUM_VIZ_SIMS sims ($VIZ_MODE), GIFs: ${CREATE_GIFS:+enabled}"
echo "================================================================"

# --- Validation ---
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$TEST_DIR" ]; then
    echo "❌ Test directory not found: $TEST_DIR"
    exit 1
fi

# --- Build optional flags (Using Arrays) ---
OPTIONAL_ARGS=()

if [ -n "$HEC_RAS_DIR" ] && [ -d "$HEC_RAS_DIR" ]; then
    # We add the flag and the value as separate array elements
    OPTIONAL_ARGS+=(--hec_ras_dir "$HEC_RAS_DIR")
    echo "  ✓ HEC-RAS mesh rendering enabled"
else
    echo "  ⚠ HEC-RAS dir not found — falling back to scatter plots"
fi

if [ -n "$EXTREMA_PATH" ] && [ -f "$EXTREMA_PATH" ]; then
    OPTIONAL_ARGS+=(--extrema_path "$EXTREMA_PATH")
    echo "  ✓ Physical unit denormalization enabled"
else
    echo "  ⚠ Extrema file not found — metrics in normalized units"
fi

if [ -n "$MAX_SEQUENCES" ]; then
    OPTIONAL_ARGS+=(--max_sequences "$MAX_SEQUENCES")
fi

echo ""
echo "Starting evaluation..."
echo ""

# Note: We use "${OPTIONAL_ARGS[@]}" to expand the array safely
apptainer run --nv "$CONTAINER" eval_gparcv2.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --eval_mode "$EVAL_MODE" \
    --num_layers "$NUM_LAYERS" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT_CHANNELS" \
    --dropout "$DROPOUT" \
    $USE_LAYER_NORM \
    $USE_RELATIVE_POS \
    --integrator "$INTEGRATOR" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --velocity_indices $VELOCITY_INDICES \
    --spade_heads "$SPADE_HEADS" \
    $SPADE_CONCAT \
    --spade_dropout "$SPADE_DROPOUT" \
    $ZERO_INIT \
    --rollout_steps "$ROLLOUT_STEPS" \
    $CREATE_GIFS \
    --num_viz_simulations "$NUM_VIZ_SIMS" \
    --viz_selection_mode "$VIZ_MODE" \
    --gif_fps "$GIF_FPS" \
    --gif_frame_skip "$GIF_FRAME_SKIP" \
    "${OPTIONAL_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation complete!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - rollout_metrics.json     (RRMSE per variable)"
    echo "  - snapshot_metrics.json    (single-step metrics)"
    echo "  - comparison_metrics.json  (snapshot vs rollout)"
    echo "  - *_dashboard.png          (performance dashboards)"
    echo "  - depth_*.gif              (water depth animations)"
    echo "  - volume_*.gif             (volume animations)"
    echo "  - vel_x_*.gif / vel_y_*.gif (velocity animations)"
    echo "  - timeseries_*.png         (per-variable RMSE over time)"
    echo "  - scatter_*.png            (pred vs target)"
    echo ""
    echo "Key metrics:"
    echo "  - RRMSE_Depth (primary accuracy measure)"
    echo "  - RRMSE_Vel_X / RRMSE_Vel_Y (velocity field accuracy)"
    echo "  - Snapshot vs Rollout ratio (error accumulation)"
else
    echo "❌ Evaluation failed with exit code: $EXIT_CODE"
fi
echo "================================================================"

exit $EXIT_CODE