#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J film_analysis
#SBATCH -o film_analysis.out
#SBATCH -e film_analysis.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:15:00
#SBATCH -c 4
#SBATCH --mem=32G

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

BASE_DATA="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt"
TEST_DIR="${BASE_DATA}/normalized_datasets/train_cases_normalized"
TRAIN_DIR="${BASE_DATA}/normalized_datasets/test_cases_normalized"
CHECKPOINT="/scratch/jtb3sud/shocktube_v2_training/nospadeFAST/best_model.pth"
OUTPUT_DIR="/scratch/jtb3sud/shocktube_comparison/film_analysis/test"

mkdir -p "${OUTPUT_DIR}"

apptainer run --nv "${CONTAINER}" film_activation_analysis.py \
    --test_dir "${TEST_DIR}" \
    --train_dir "${TRAIN_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda

echo "Done! Figures in ${OUTPUT_DIR}"
