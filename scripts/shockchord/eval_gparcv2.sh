#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J gparc_shock_eval
#SBATCH -o gparc_shock_eval.out
#SBATCH -e gparc_shock_eval.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 0:30:00
#SBATCH -c 8
#SBATCH --mem=120G

echo "================================================================"
echo "G-PARCv2 SHOCK TUBE: EVALUATION"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# PATHS — EDIT THESE
# ============================================================
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

MODEL_PATH="/scratch/jtb3sud/shocktube_v2_training/best_model.pth"
TEST_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/train_cases_normalized"
OUTPUT_DIR="/scratch/jtb3sud/shocktube_v2_training/evaluation/train"

# Optional: explicit normalization metadata (auto-detected if omitted)
# NORM_METADATA="/path/to/normalization_metadata.json"

# ============================================================
# EVALUATION SETTINGS
# ============================================================
EVAL_MODE="rollout"          # rollout | snapshot | both
ROLLOUT_STEPS=40
MAX_SEQUENCES=10
CREATE_GIFS="--create_gifs"  # Comment out to skip GIFs
NUM_VIZ_SIMS=3
VIZ_SELECTION="representative"  # representative | best | worst | all
GIF_FPS=4
GIF_FRAME_SKIP=1

# ============================================================
# ARCHITECTURE (MUST MATCH TRAINING!)
# ============================================================
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=3
SKIP_DYNAMIC_INDICES="2"
VELOCITY_INDEX=1
GLOBAL_PARAM_DIM=3
GLOBAL_EMBED_DIM=64
INTEGRATOR="euler"

NUM_LAYERS=4
HIDDEN_CHANNELS=64
FEATURE_OUT_CHANNELS=128
DROPOUT=0.2
USE_LAYER_NORM="--use_layer_norm"
USE_RELATIVE_POS="--use_relative_pos"

SPADE_HEADS=4
SPADE_CONCAT="--spade_concat"
SPADE_DROPOUT=0.1
ZERO_INIT="--zero_init"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Model:    $MODEL_PATH"
echo "  Test dir: $TEST_DIR"
echo "  Output:   $OUTPUT_DIR"
echo "  Mode:     $EVAL_MODE"
echo "  Rollout:  $ROLLOUT_STEPS steps | Max sims: $MAX_SEQUENCES"
echo "  GIFs:     $CREATE_GIFS ($NUM_VIZ_SIMS sims, $VIZ_SELECTION)"
echo "  Arch:     layers=$NUM_LAYERS hidden=$HIDDEN_CHANNELS fe_out=$FEATURE_OUT_CHANNELS"
echo "  Dynamic:  $NUM_DYNAMIC_FEATS feats (skip raw: $SKIP_DYNAMIC_INDICES)"
echo "  Global:   dim=$GLOBAL_PARAM_DIM → embed=$GLOBAL_EMBED_DIM"
echo "================================================================"

# Build optional args
NORM_ARG=""
if [ -n "${NORM_METADATA:-}" ]; then
    NORM_ARG="--norm_metadata_file $NORM_METADATA"
fi

apptainer run --nv "$CONTAINER" eval_gparcv2.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --eval_mode "$EVAL_MODE" \
    --rollout_steps "$ROLLOUT_STEPS" \
    --max_sequences "$MAX_SEQUENCES" \
    $CREATE_GIFS \
    --num_viz_simulations "$NUM_VIZ_SIMS" \
    --viz_selection_mode "$VIZ_SELECTION" \
    --gif_fps "$GIF_FPS" \
    --gif_frame_skip "$GIF_FRAME_SKIP" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --skip_dynamic_indices $SKIP_DYNAMIC_INDICES \
    --velocity_index "$VELOCITY_INDEX" \
    --global_param_dim "$GLOBAL_PARAM_DIM" \
    --global_embed_dim "$GLOBAL_EMBED_DIM" \
    --integrator "$INTEGRATOR" \
    --num_layers "$NUM_LAYERS" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT_CHANNELS" \
    --dropout "$DROPOUT" \
    $USE_LAYER_NORM \
    $USE_RELATIVE_POS \
    --spade_heads "$SPADE_HEADS" \
    $SPADE_CONCAT \
    --spade_dropout "$SPADE_DROPOUT" \
    $ZERO_INIT \
    $NORM_ARG

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Finished at: $(date) | Exit: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS — Results in: $OUTPUT_DIR"
    echo "  Key outputs:"
    echo "    ${OUTPUT_DIR}/rollout_dashboard.png"
    echo "    ${OUTPUT_DIR}/rollout_global_parameter_analysis.png"
    echo "    ${OUTPUT_DIR}/rollout_metrics.json"
    echo "    ${OUTPUT_DIR}/rollout_per_simulation.json"
else
    echo "❌ FAILED"
fi
echo "================================================================"