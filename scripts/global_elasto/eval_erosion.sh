#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J eval_erosion
#SBATCH -o eval_erosion.out
#SBATCH -e eval_erosion.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=40G

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# PATHS — update MODEL_PATH to whichever checkpoint you want
# ============================================================
MODEL_PATH="/scratch/jtb3sud/elasto_graphconv_V2/erosion/best_model.pth"
TEST_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/test"
OUTPUT_DIR="/scratch/jtb3sud/elasto_graphconv_V2/erosion25/eval_results"

mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "EROSION MODEL EVALUATION"
echo "  Model: $MODEL_PATH"
echo "  Test:  $TEST_DIR"
echo "  Out:   $OUTPUT_DIR"
echo "================================================================"

apptainer run --nv "$CONTAINER" eval_erosion.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_viz 5 \
    --device cuda

echo "Done! Results in $OUTPUT_DIR"