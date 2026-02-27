#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J river_paper_fig
#SBATCH -o river_paper_fig.out
#SBATCH -e river_paper_fig.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:20:00
#SBATCH -c 4
#SBATCH --mem=40G

echo "================================================================"
echo "RIVER PAPER FIGURES"
echo "================================================================"
echo "Start: $(date)"

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"
SCRIPT_DIR="$HOME/G-PARC/scripts/river"

# ============================================================
# DATA
# ============================================================
TEST_DIR="/standard/sds_baek_energetic/HEC_RAS (River)/pt_test_normalized"
HEC_RAS_DIR="/standard/sds_baek_energetic/HEC_RAS (River)"
EXTREMA="/standard/sds_baek_energetic/HEC_RAS (River)/global_y_extrema_test.pth"
OUTPUT_DIR="/scratch/jtb3sud/river_comparison/paper_figures"

# ============================================================
# MODEL CHECKPOINTS
# ============================================================
GPARCV1_CKPT="/home/jtb3sud/G-PARC/weights/new_river/modelseq20_ep250.pth"
GPARCV2_CKPT="/scratch/jtb3sud/gparcv2/river/concat/latest_model.pth"
MGKAN_CKPT="/scratch/jtb3sud/delta/river/run2_51_150/best_model.pth"
MGNET_CKPT="/scratch/jtb3sud/meshgraphnet/river/run1/best_model.pt"
GSAGE_CKPT="/scratch/jtb3sud/graphsage/river/best_model.pth"

# ============================================================
# BUILD MODEL LIST
# ============================================================
MODELS=""
for name_ckpt in "gparcv1:$GPARCV1_CKPT" "gparcv2:$GPARCV2_CKPT" "mgkan:$MGKAN_CKPT" "mgnet:$MGNET_CKPT" "graphsage:$GSAGE_CKPT"; do
    mtype="${name_ckpt%%:*}"
    mpath="${name_ckpt#*:}"
    if [ -f "$mpath" ]; then
        MODELS="$MODELS $mtype:$mpath"
        echo "  ✓ $mtype: $mpath"
    else
        echo "  ✗ $mtype: not found ($mpath)"
    fi
done

if [ -z "$MODELS" ]; then
    echo "❌ No checkpoints found."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Test data:    $TEST_DIR"
echo "HEC-RAS dir:  $HEC_RAS_DIR"
echo "Extrema:      $EXTREMA"
echo "Output:       $OUTPUT_DIR"
echo "================================================================"

# Only plot Depth (0) and Velocity_X (2) for paper — adjust as needed
apptainer run --nv "$CONTAINER" paper_figure.py \
    --test_dir "$TEST_DIR" \
    --models $MODELS \
    --hec_ras_dir "$HEC_RAS_DIR" \
    --extrema_path "$EXTREMA" \
    --output_dir "$OUTPUT_DIR" \
    --sim_index_wr 0 \
    --sim_index_iw 0 \
    --rollout_steps 110 \
    --variables 0 1 2 3 \
    --dpi 300 \
    --error_fig \
    --device cuda

echo ""
echo "End: $(date)"
echo "✅ Figures in $OUTPUT_DIR"