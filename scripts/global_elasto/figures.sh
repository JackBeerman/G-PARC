#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J elasto_figures
#SBATCH -o elasto_figures.out
#SBATCH -e elasto_figures.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=64G

echo "================================================================"
echo "ELASTOPLASTIC PAPER FIGURES"
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
DATA_ROOT="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small"
TEST_DIR="$DATA_ROOT/test"
NORM_STATS="$DATA_ROOT/normalization_stats.json"
OUTPUT_DIR="/scratch/jtb3sud/elasto_comparison/paper_figures"

# Model checkpoints
GPARCV2_CKPT="/scratch/jtb3sud/elasto_graphconv_V2/2hop/best_model.pth"
GPARCV1_CKPT="/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/best_model.pth"
MGKAN_CKPT="/scratch/jtb3sud/delta/elasto/best_model.pth"
MGN_CKPT="/scratch/jtb3sud/meshgraphnet/elasto/run1/best_model.pt"

# Which models to include (space-separated: gparcv2 gparcv1 mgn mgkan)
MODELS="gparcv2 gparcv1 mgkan mgn"

# Simulation index to visualize (run multiple times with different indices)
SIM_INDEX=0

# Number of timestep columns
NUM_TIMESTEPS=4

echo ""
echo "Configuration:"
echo "  Test dir:   $TEST_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Models:     $MODELS"
echo "  Sim index:  $SIM_INDEX"
echo "  Timesteps:  $NUM_TIMESTEPS"
echo "================================================================"

mkdir -p "$OUTPUT_DIR"

apptainer run --nv "$CONTAINER" figures.py \
    --test_dir "$TEST_DIR" \
    --norm_stats "$NORM_STATS" \
    --output_dir "$OUTPUT_DIR" \
    --models $MODELS \
    --gparcv2_ckpt "$GPARCV2_CKPT" \
    --gparcv1_ckpt "$GPARCV1_CKPT" \
    --mgkan_ckpt "$MGKAN_CKPT" \
    --mgn_ckpt "$MGN_CKPT" \
    --sim_index "$SIM_INDEX" \
    --num_timesteps "$NUM_TIMESTEPS" \
    --dpi 300 \
    --device cuda

echo ""
echo "Finished: $(date)"