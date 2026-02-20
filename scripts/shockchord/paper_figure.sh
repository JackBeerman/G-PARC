#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J shocktube_paper_fig
#SBATCH -o shocktube_paper_fig.out
#SBATCH -e shocktube_paper_fig.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:10:00
#SBATCH -c 4
#SBATCH --mem=32G

echo "================================================================"
echo "SHOCK TUBE PAPER FIGURE"
echo "================================================================"
echo "Start: $(date)"

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# DATA
# ============================================================
BASE_DATA="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt"
TEST_DIR="${BASE_DATA}/normalized_datasets/test_cases_normalized"
OUTPUT_DIR="/scratch/jtb3sud/shocktube_comparison/paper_figures"

# ============================================================
# MODEL CHECKPOINTS
# ============================================================
GPARCV1_CKPT="${BASE_DATA}/shock_tube_20250927_104720_run_mod10_750/shock_tube_best_model.pth"
GPARCV2_CKPT="/scratch/jtb3sud/shocktube_v2_training/best_model.pth"
MGKAN_CKPT="/scratch/jtb3sud/delta/shocktube/run_101_300/best_model.pth"
MGNET_CKPT="/scratch/jtb3sud/meshgraphnet/shocktube/run1/best_model.pt"

# ============================================================
# BUILD MODEL LIST (skip missing)
# ============================================================
MODELS=""
for name_ckpt in "gparcv1:$GPARCV1_CKPT" "gparcv2:$GPARCV2_CKPT" "mgkan:$MGKAN_CKPT" "mgnet:$MGNET_CKPT"; do
    mtype="${name_ckpt%%:*}"
    mpath="${name_ckpt#*:}"
    if [ -f "$mpath" ]; then
        MODELS="$MODELS $mtype:$mpath"
        echo "  ✓ $mtype: $mpath"
    else
        echo "  ✗ $mtype: not found"
    fi
done

if [ -z "$MODELS" ]; then
    echo "❌ No checkpoints found."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Test data:  $TEST_DIR"
echo "Output:     $OUTPUT_DIR"
echo "================================================================"

apptainer run --nv "$CONTAINER" paper_figure.py \
    --test_dir "$TEST_DIR" \
    --models $MODELS \
    --output_dir "$OUTPUT_DIR" \
    --sim_index 0 \
    --rollout_steps 40 \
    --dpi 300 \
    --cmap RdBu_r \
    --error_fig \
    --device cuda

echo ""
echo "End: $(date)"
echo "✅ Figures in $OUTPUT_DIR"