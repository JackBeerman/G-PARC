#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J st_gifs
#SBATCH -o st_gifs.out
#SBATCH -e st_gifs.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=64G

echo "================================================================"
echo "SHOCK TUBE — MULTI-MODEL COMPARISON GIFS"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "================================================================"

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"
SCRIPT="$HOME/G-PARC/scripts/eval/compare_shocktube_gif.py"
OUTPUT_DIR="/scratch/jtb3sud/shocktube_gifs"

# Data
ST_TEST_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/test_cases_normalized"

# Model checkpoints
ST_GPARCV2="/scratch/jtb3sud/shocktube_v2_training/nospadeFAST/best_model.pth"
ST_GPARCV1="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/shock_tube_20250927_104720_run_mod10_750/shock_tube_best_model.pth"
ST_MGKAN="/scratch/jtb3sud/delta/shocktube/run_101_300/best_model.pth"
ST_MGNET="/scratch/jtb3sud/meshgraphnet/shocktube/run1/best_model.pt"
ST_GSAGE="/scratch/jtb3sud/graphsage/shocktube/best_model.pth"

# GIF settings
ROLLOUT_STEPS=42
FPS=6
FRAME_SKIP=1
SIM_INDICES="0 1 2 3 4 5 6 7 8 9 10"

echo ""
echo "Configuration:"
echo "  Test dir:  $ST_TEST_DIR"
echo "  Output:    $OUTPUT_DIR"
echo "  Sims:      $SIM_INDICES"
echo "  Steps:     $ROLLOUT_STEPS"
echo "  FPS:       $FPS"
echo "================================================================"

mkdir -p "$OUTPUT_DIR"

apptainer run --nv "$CONTAINER" compare_shocktube_gif.py \
    --test_dir "$ST_TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --rollout_steps "$ROLLOUT_STEPS" \
    --fps "$FPS" \
    --frame_skip "$FRAME_SKIP" \
    --sim_indices $SIM_INDICES \
    --device cuda \
    --models \
        "gparcv2:$ST_GPARCV2" \
        "gparcv1:$ST_GPARCV1" \
        "mgkan:$ST_MGKAN" \
        "mgnet:$ST_MGNET" \
        "gsage:$ST_GSAGE"

echo ""
echo "================================================================"
echo "Finished: $(date)"
echo "GIFs:     $OUTPUT_DIR"
echo "================================================================"