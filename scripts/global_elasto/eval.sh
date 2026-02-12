#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J eval_2hop
#SBATCH -o eval_2hop.out
#SBATCH -e eval_2hop.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 00:30:00
#SBATCH -c 8
#SBATCH --mem=64G

echo "================================================================"
echo "EVALUATION: Global Max Normalized Model"
echo "Physics-Preserving Normalization (max_displacement = 542.1 mm)"
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# PATHS
# ============================================================
MODEL_PATH="/scratch/jtb3sud/elasto_graphconv_V2/2hop/best_model.pth"
TEST_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/test"
OUTPUT_DIR="/scratch/jtb3sud/elasto_graphconv_V2/2hop/evaluation/test"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# ARCHITECTURE (MUST MATCH TRAINING!)
# ============================================================
NUM_LAYERS=4
HIDDEN_CHANNELS=128
FEATURE_OUT_CHANNELS=128
DROPOUT=0.0
USE_LAYER_NORM="--use_layer_norm"
USE_RELATIVE_POS="--use_relative_pos"
CLAMP_FLAG="--no_clamp_output"

# ============================================================
# PHYSICS (MUST MATCH TRAINING!)
# ============================================================
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=2
N_STATE_VAR=0
USE_VON_MISES="--use_von_mises"
USE_VOLUMETRIC="--use_volumetric"
SPADE_HEADS=4
SPADE_CONCAT="--spade_concat"
SPADE_DROPOUT=0.1
ZERO_INIT="--zero_init"

# ============================================================
# EVALUATION SETTINGS
# ============================================================
EVAL_MODE="both"              # rollout, snapshot, or both
MAX_SEQUENCES=8                  # Number of test sequences to evaluate
ROLLOUT_STEPS=39                 # Full rollout length (all timesteps)

# ============================================================
# VISUALIZATION SETTINGS
# ============================================================
CREATE_GIFS="--create_gifs"      # Generate visualization GIFs
NUM_VIZ_SIMS=3                   # Number of simulations to visualize
VIZ_MODE="representative"        # representative, best, worst, random, all
GIF_FPS=10                       # Frames per second for GIFs
GIF_FRAME_SKIP=1                 # Frame skip (1 = all frames)

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Test data: $TEST_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Architecture:"
echo "  Layers: $NUM_LAYERS"
echo "  Hidden: $HIDDEN_CHANNELS"
echo "  Feature output: $FEATURE_OUT_CHANNELS"
echo "  Layer norm: enabled"
echo "  Relative pos: enabled"
echo ""
echo "Evaluation:"
echo "  Mode: $EVAL_MODE"
echo "  Sequences: $MAX_SEQUENCES"
echo "  Rollout steps: $ROLLOUT_STEPS"
echo "  Create GIFs: enabled"
echo "  Viz simulations: $NUM_VIZ_SIMS ($VIZ_MODE)"
echo ""
echo "Expected metrics with global max normalization:"
echo "  - Balanced U_x/U_y RRMSE (both ~0.10-0.12)"
echo "  - Better than z-score baseline"
echo "================================================================"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    exit 1
fi

# Check if test data exists
if [ ! -d "$TEST_DIR" ]; then
    echo "❌ Test directory not found: $TEST_DIR"
    exit 1
fi

echo ""
echo "Starting evaluation..."
echo ""

apptainer run --nv "$CONTAINER" eval_elasto.py \
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
    --integrator "euler" \
    $CLAMP_FLAG \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --n_state_var "$N_STATE_VAR" \
    $USE_VON_MISES \
    $USE_VOLUMETRIC \
    --spade_heads "$SPADE_HEADS" \
    $SPADE_CONCAT \
    --spade_dropout "$SPADE_DROPOUT" \
    $ZERO_INIT \
    --max_sequences "$MAX_SEQUENCES" \
    --rollout_steps "$ROLLOUT_STEPS" \
    $CREATE_GIFS \
    --num_viz_simulations "$NUM_VIZ_SIMS" \
    --viz_selection_mode "$VIZ_MODE" \
    --gif_fps "$GIF_FPS" \
    --gif_frame_skip "$GIF_FRAME_SKIP"

EXIT_CODE=$?

echo ""
echo "================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation complete!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - rollout_metrics.json    (PLAID benchmark metrics)"
    echo "  - rollout_dashboard.png   (Performance visualization)"
    echo "  - reference_*.gif         (Reference config animations)"
    echo "  - deformed_*.gif          (Deformed config animations)"
    echo "  - error_*.gif             (Error visualizations)"
    echo ""
    echo "Key metrics to check:"
    echo "  - RRMSE_Ux vs RRMSE_Uy ratio (should be ~1:1 with global max)"
    echo "  - Overall RRMSE (compare with z-score baseline)"
    echo "  - R² scores (higher is better)"
    echo ""
    echo "Compare with z-score normalization to see improvement!"
else
    echo "❌ Evaluation failed with exit code: $EXIT_CODE"
fi
echo "================================================================"

exit $EXIT_CODE