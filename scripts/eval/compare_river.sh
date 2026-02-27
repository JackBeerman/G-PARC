#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J river_gifs
#SBATCH -o river_gifs_%j.out
#SBATCH -e river_gifs_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:10:00
#SBATCH -c 4
#SBATCH --mem=192G

echo "================================================================"
echo "RIVER — MULTI-MODEL COMPARISON GIFS"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "================================================================"

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"
SCRIPT="$HOME/G-PARC/scripts/eval/compare_river_gif.py"

# ── Paths ────────────────────────────────────────────────────────
RV_TEST_DIR="/standard/sds_baek_energetic/HEC_RAS (River)/pt_test_normalized"
RV_HEC_RAS_DIR="/standard/sds_baek_energetic/HEC_RAS (River)"
RV_EXTREMA="/standard/sds_baek_energetic/HEC_RAS (River)/global_y_extrema_test.pth"
OUTPUT_DIR="/scratch/jtb3sud/unified_eval_results/river_gifs"

RV_GPARCV2="/scratch/jtb3sud/gparcv2/river/concat/20/best_model.pth"
RV_GPARCV1="/home/jtb3sud/G-PARC/weights/new_river/modelseq20_ep250.pth"
RV_MGKAN="/scratch/jtb3sud/delta/river/run2_51_150/best_model.pth"
RV_MGNET="/scratch/jtb3sud/meshgraphnet/river/run1/best_model.pt"
RV_GSAGE="/scratch/jtb3sud/graphsage/whathe/best_model_river2.pth"

echo ""
echo "Configuration:"
echo "  Test dir:     $RV_TEST_DIR"
echo "  HEC-RAS dir:  $RV_HEC_RAS_DIR"
echo "  Extrema:      $RV_EXTREMA"
echo "  Output:       $OUTPUT_DIR"
echo "================================================================"

mkdir -p "$OUTPUT_DIR"

apptainer run --nv "$CONTAINER" python "$SCRIPT" \
    --test_dir "$RV_TEST_DIR" \
    --hec_ras_dir "$RV_HEC_RAS_DIR" \
    --extrema "$RV_EXTREMA" \
    --models \
        "gparcv2:$RV_GPARCV2" \
        "gparcv1:$RV_GPARCV1" \
        "mgkan:$RV_MGKAN" \
        "mgnet:$RV_MGNET" \
        "graphsage:$RV_GSAGE" \
    --output_dir "$OUTPUT_DIR" \
    --sim_indices 0 1 2 \
    --rollout_steps 50 \
    --fps 6 \
    --frame_skip 1 \
    --dt_minutes 20 \
    --exclude_error MeshGraphNet GraphSAGE

echo ""
echo "================================================================"
echo "Finished: $(date)"
echo "GIFs in:  $OUTPUT_DIR"
echo "================================================================"