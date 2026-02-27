#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J elasto_gifs
#SBATCH -o elasto_gifs_%j.out
#SBATCH -e elasto_gifs_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:10:00
#SBATCH -c 4
#SBATCH --mem=192G

echo "================================================================"
echo "ELASTOPLASTIC — MULTI-MODEL COMPARISON GIFS"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "================================================================"

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"
SCRIPT="$HOME/G-PARC/scripts/eval/compare_elasto_gif.py"

# ── Paths ────────────────────────────────────────────────────────
EL_TEST_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/full_test"
OUTPUT_DIR="/scratch/jtb3sud/unified_eval_results/elasto_gifs"

EL_GPARCV2="/scratch/jtb3sud/elasto_graphconv_V2/2hop/best_model.pth"
EL_GPARCV1="/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/best_model.pth"
EL_MGKAN="/scratch/jtb3sud/delta/elasto/best_model.pth"
EL_MGN="/scratch/jtb3sud/meshgraphnet/elasto/run1/best_model.pt"
EL_GSAGE="/scratch/jtb3sud/graphsage/elasto/best_model.pth"

echo ""
echo "Configuration:"
echo "  Test dir:   $EL_TEST_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "================================================================"

mkdir -p "$OUTPUT_DIR"

apptainer run --nv "$CONTAINER" compare_elasto_gif.py \
    --test_dir "$EL_TEST_DIR" \
    --models \
        "gparcv2:$EL_GPARCV2" \
        "gparcv1:$EL_GPARCV1" \
        "mgkan:$EL_MGKAN" \
        "mgnet:$EL_MGN" \
        "graphsage:$EL_GSAGE" \
    --output_dir "$OUTPUT_DIR" \
    --sim_indices 0 1 2 \
    --rollout_steps 37 \
    --fps 6 \
    --frame_skip 1 \
    --exclude_error MeshGraphNet GraphSAGE

echo ""
echo "================================================================"
echo "Finished: $(date)"
echo "GIFs in:  $OUTPUT_DIR"
echo "================================================================"