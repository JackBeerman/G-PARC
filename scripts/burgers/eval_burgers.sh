#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J burg_eval
#SBATCH -o burg_eval.out
#SBATCH -e burg_eval.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 0:30:00
#SBATCH -c 4
#SBATCH --mem=32G

echo "================================================================"
echo "G-PARCv2 BURGERS: EVALUATION (CONCAT+MLP)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "================================================================"

module purge
module load apptainer

# Paths
MODEL_PATH="/scratch/jtb3sud/burgers_v2_concat/best_model.pth"
DATA_ROOT="/scratch/jtb3sud/processed_burgers_graph"
TEST_DIR="$DATA_ROOT/test"
OUTPUT_DIR="/scratch/jtb3sud/burgers_v2_concat/evaluation"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"
SCRIPT_DIR="$HOME/G-PARC/scripts/burgers"

# Optional: metadata for denormalization
METADATA_FILE="$DATA_ROOT/normalization_metadata.json"

# Architecture (must match training)
HIDDEN_CHANNELS=64
FEATURE_OUT=128
NUM_FE_LAYERS=4
FUSION_HIDDEN_DIM=128
DIFFUSION_TYPE="fd"
INTEGRATOR="euler"

# Eval settings
MODE="rollout"
ROLLOUT_STEPS=50
MAX_SIMS=10
NUM_GIFS=5

echo ""
echo "Configuration:"
echo "  Model:      $MODEL_PATH"
echo "  Test dir:   $TEST_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Fusion:     concat+MLP (hidden=$FUSION_HIDDEN_DIM)"
echo "  Mode:       $MODE"
echo "  Steps:      $ROLLOUT_STEPS"
echo "  Max sims:   $MAX_SIMS"
echo "  GIFs:       $NUM_GIFS"
echo "================================================================"

mkdir -p "$OUTPUT_DIR"

METADATA_FLAG=""
if [ -f "$METADATA_FILE" ]; then
    METADATA_FLAG="--metadata_file $METADATA_FILE"
fi

apptainer run --nv "$CONTAINER" eval_burgers.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT" \
    --num_fe_layers "$NUM_FE_LAYERS" \
    --fusion_hidden_dim "$FUSION_HIDDEN_DIM" \
    --diffusion_type "$DIFFUSION_TYPE" \
    --use_film \
    --integrator "$INTEGRATOR" \
    --mode "$MODE" \
    --rollout_steps "$ROLLOUT_STEPS" \
    --max_sequences "$MAX_SIMS" \
    --create_gifs \
    --num_gifs "$NUM_GIFS" \
    --device auto \
    $METADATA_FLAG

echo ""
echo "Finished: $(date)"