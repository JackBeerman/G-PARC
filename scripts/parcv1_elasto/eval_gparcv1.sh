#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J eval_parcv1
#SBATCH -o eval_parcv1.out
#SBATCH -e eval_parcv1.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:30:00
#SBATCH -c 8
#SBATCH --mem=40G

echo "================================================================"
echo "PARC v1 EVALUATION"
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# PATHS
# ============================================================
MODEL_PATH="/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/best_model.pth"
TEST_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/test"
OUTPUT_DIR="/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/eval_results/test"
NORM_STATS="/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/normalization_stats.json"

# ============================================================
# ARCHITECTURE (must match training!)
# ============================================================
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=2

HIDDEN_CHANNELS=128
FEATURE_OUT_CHANNELS=128
DEPTH=3
HEADS=3
DROPOUT=0.1

DERIV_HIDDEN_CHANNELS=128
DERIV_NUM_LAYERS=3
DERIV_HEADS=3
DERIV_DROPOUT=0.1

INTEGRAL_HIDDEN_CHANNELS=128
INTEGRAL_NUM_LAYERS=3
INTEGRAL_HEADS=4
INTEGRAL_DROPOUT=0.1

# ============================================================
# EVAL SETTINGS
# ============================================================
EVAL_MODE="both"
ROLLOUT_STEPS=39
MAX_SEQUENCES=10

mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Test dir: $TEST_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Eval mode: $EVAL_MODE"
echo "  Rollout steps: $ROLLOUT_STEPS"
echo "================================================================"

apptainer run --nv "$CONTAINER" eval_gparcv1.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --norm_stats_file "$NORM_STATS" \
    --eval_mode "$EVAL_MODE" \
    --rollout_steps "$ROLLOUT_STEPS" \
    --max_sequences "$MAX_SEQUENCES" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT_CHANNELS" \
    --depth "$DEPTH" \
    --pool_ratios 0.2 \
    --heads "$HEADS" \
    --dropout "$DROPOUT" \
    --deriv_hidden_channels "$DERIV_HIDDEN_CHANNELS" \
    --deriv_num_layers "$DERIV_NUM_LAYERS" \
    --deriv_heads "$DERIV_HEADS" \
    --deriv_dropout "$DERIV_DROPOUT" \
    --deriv_use_residual \
    --integral_hidden_channels "$INTEGRAL_HIDDEN_CHANNELS" \
    --integral_num_layers "$INTEGRAL_NUM_LAYERS" \
    --integral_heads "$INTEGRAL_HEADS" \
    --integral_dropout "$INTEGRAL_DROPOUT" \
    --integral_use_residual \
    --create_gifs

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation complete!"
    echo "  Results: $OUTPUT_DIR"
else
    echo "❌ Evaluation failed"
fi

exit $EXIT_CODE