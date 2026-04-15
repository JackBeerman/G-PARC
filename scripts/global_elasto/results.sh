#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J elasto_compare
#SBATCH -o elasto_compare.out
#SBATCH -e elasto_compare.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=64G

echo "================================================================"
echo "ELASTOPLASTIC MULTI-MODEL COMPARISON"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "================================================================"

module purge
module load apptainer

# Paths
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"
SCRIPT_DIR="$HOME/G-PARC/scripts/eval_elasto"
DATA_ROOT="/scratch/jtb3sud/processed_elasto_plastic/p99clip/normalized/small"
TEST_DIR="$DATA_ROOT/test"
NORM_STATS="$DATA_ROOT/normalization_stats.json"
OUTPUT_DIR="/scratch/jtb3sud/elasto_comparison99"

# Model checkpoints
# config.json is auto-detected from checkpoint directory
GPARCV2_CKPT="/scratch/jtb3sud/gparcv2/p99/best_model.pth"
GPARCV1_CKPT="/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/best_model.pth"
MGKAN_CKPT="/scratch/jtb3sud/delta/elasto/best_model.pth"
MGN_CKPT="/scratch/jtb3sud/meshgraphnet/elasto/run1/best_model.pt"
GSAGE_CKPT="/scratch/jtb3sud/graphsage/elasto/best_model.pth"

# Which models to compare (space-separated: gparcv2 gparcv1 mgkan mgn)
MODELS="gparcv2 gparcv1"

# Max simulations (set to empty for all)
MAX_SIMS=

echo ""
echo "Configuration:"
echo "  Test dir:   $TEST_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Models:     $MODELS"
echo "  Max sims:   ${MAX_SIMS:-all}"
echo "================================================================"

mkdir -p "$OUTPUT_DIR"

apptainer run --nv "$CONTAINER" results.py \
    --test_dir "$TEST_DIR" \
    --norm_stats "$NORM_STATS" \
    --output_dir "$OUTPUT_DIR" \
    --models $MODELS \
    --gparcv2_ckpt "$GPARCV2_CKPT" \
    --gparcv1_ckpt "$GPARCV1_CKPT" \
    --mgkan_ckpt "$MGKAN_CKPT" \
    --mgn_ckpt "$MGN_CKPT" \
    --graphsage_ckpt "$GSAGE_CKPT" \
    ${MAX_SIMS:+--max_sims $MAX_SIMS} \
    --device cuda

echo ""
echo "Finished: $(date)"