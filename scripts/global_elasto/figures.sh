#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J elasto_figures
#SBATCH -o elasto_figures.out
#SBATCH -e elasto_figures.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:10:00
#SBATCH -c 4
#SBATCH --mem=64G

echo "================================================================"
echo "ELASTOPLASTIC PAPER FIGURES (dual-run)"
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

# Model checkpoints
GPARCV2_CKPT="/scratch/jtb3sud/elasto_graphconv_V2/2hop/best_model.pth"
GPARCV1_CKPT="/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/best_model.pth"
MGKAN_CKPT="/scratch/jtb3sud/delta/elasto/best_model.pth"
MGN_CKPT="/scratch/jtb3sud/meshgraphnet/elasto/run1/best_model.pt"
GSAGE_CKPT="/scratch/jtb3sud/graphsage/elasto/best_model.pth"

SIM_INDEX=0
NUM_TIMESTEPS=3

echo ""
echo "Configuration:"
echo "  Test dir:   $TEST_DIR"
echo "  Norm stats: $NORM_STATS"
echo "  Sim index:  $SIM_INDEX"
echo "  Timesteps:  $NUM_TIMESTEPS"
echo "================================================================"

# ============================================================
# RUN 1: G-PARC comparison (gparcv2, gparcv1, mgkan)
# ============================================================
OUTPUT_GPARC="/scratch/jtb3sud/elasto_comparison/paper_figures/gparc"
mkdir -p "$OUTPUT_GPARC"

echo ""
echo "RUN 1: G-PARC comparison → $OUTPUT_GPARC"
echo "  Models: G-PARC, G-PARC (w/o MLS), MeshGraphKAN"
echo "================================================================"

apptainer run --nv "$CONTAINER" figures.py \
    --test_dir "$TEST_DIR" \
    --norm_stats "$NORM_STATS" \
    --output_dir "$OUTPUT_GPARC" \
    --deformed \
    --models gparcv2 gparcv1 mgkan \
    --paper_models gparcv2 gparcv1 mgkan \
    --gparcv2_ckpt "$GPARCV2_CKPT" \
    --gparcv1_ckpt "$GPARCV1_CKPT" \
    --mgkan_ckpt "$MGKAN_CKPT" \
    --sim_index "$SIM_INDEX" \
    --num_timesteps "$NUM_TIMESTEPS" \
    --dpi 300 \
    --device cuda

# ============================================================
# RUN 2: Baseline comparison (mgn, graphsage)
# ============================================================
OUTPUT_BASELINES="/scratch/jtb3sud/elasto_comparison/paper_figures/baselines"
mkdir -p "$OUTPUT_BASELINES"

echo ""
echo "RUN 2: Baselines → $OUTPUT_BASELINES"
echo "  Models: MeshGraphNet, GraphSAGE"
echo "================================================================"

apptainer run --nv "$CONTAINER" figures.py \
    --test_dir "$TEST_DIR" \
    --norm_stats "$NORM_STATS" \
    --output_dir "$OUTPUT_BASELINES" \
    --deformed \
    --models mgn graphsage \
    --paper_models mgn graphsage \
    --mgn_ckpt "$MGN_CKPT" \
    --graphsage_ckpt "$GSAGE_CKPT" \
    --sim_index "$SIM_INDEX" \
    --num_timesteps "$NUM_TIMESTEPS" \
    --dpi 300 \
    --device cuda

echo ""
echo "================================================================"
echo "Finished: $(date)"
echo "  G-PARC figures:    $OUTPUT_GPARC"
echo "  Baseline figures:  $OUTPUT_BASELINES"
echo "================================================================"