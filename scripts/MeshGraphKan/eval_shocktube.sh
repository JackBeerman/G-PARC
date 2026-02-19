#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J eval_mgkan_shock
#SBATCH -o eval_mgkan_shock.out
#SBATCH -e eval_mgkan_shock.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=40G

echo "================================================================"
echo "MeshGraphKAN SHOCK TUBE EVALUATION"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname)"
echo "Start: $(date)"
echo "================================================================"

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"
SCRIPT_DIR="$HOME/G-PARC/scripts/MeshGraphKan"

# ============================================================
# PATHS — EDIT THESE
# ============================================================
MODEL_PATH="/scratch/jtb3sud/delta/shocktube/run_101_300/best_model.pth"
TEST_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/test_cases_normalized"
OUTPUT_DIR="/scratch/jtb3sud/delta/shocktube/eval/test"

# Optional: explicit normalization metadata path
# NORM_META="/path/to/normalization_metadata.json"

# ============================================================
# EVALUATION SETTINGS
# ============================================================
EVAL_MODE="both"          # rollout, snapshot, or both
MAX_SEQUENCES=400
ROLLOUT_STEPS=43  # How many timesteps to predict into the future
CREATE_GIFS="--create_gifs"
NUM_VIZ=3
GIF_FPS=4

mkdir -p "$OUTPUT_DIR"

echo ""
echo "  Model:   $MODEL_PATH"
echo "  Test:    $TEST_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "  Mode:    $EVAL_MODE"
echo "  Steps:   $ROLLOUT_STEPS"
echo "  Max sim: $MAX_SEQUENCES"
echo "================================================================"

NORM_ARG=""
if [ -n "${NORM_META:-}" ] && [ -f "$NORM_META" ]; then
    NORM_ARG="--norm_metadata_file $NORM_META"
    echo "  Norm:    $NORM_META"
fi

apptainer run --nv "$CONTAINER" eval_shocktube.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --eval_mode "$EVAL_MODE" \
    --rollout_steps "$ROLLOUT_STEPS" \
    --max_sequences "$MAX_SEQUENCES" \
    --num_viz_simulations "$NUM_VIZ" \
    --gif_fps "$GIF_FPS" \
    $CREATE_GIFS \
    $NORM_ARG

echo ""
echo "Done at $(date) | Exit: $?"