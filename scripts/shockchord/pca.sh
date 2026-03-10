#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J film_vs_mgkan
#SBATCH -o film_vs_mgkan.out
#SBATCH -e film_vs_mgkan.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:15:00
#SBATCH -c 4
#SBATCH --mem=32G

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"
BASE_DATA="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt"
TEST_DIR="${BASE_DATA}/normalized_datasets/test_cases_normalized"
TRAIN_DIR="${BASE_DATA}/normalized_datasets/train_cases_normalized"

GPARC_CKPT="/scratch/jtb3sud/shocktube_v2_training/nospadeFAST/best_model.pth"
MGKAN_CKPT="/scratch/jtb3sud/delta/shocktube/run_101_300/best_model.pth"

OUTPUT_DIR="/scratch/jtb3sud/shocktube_comparison/film_vs_mgkan"
mkdir -p "${OUTPUT_DIR}"

apptainer run --nv "${CONTAINER}" mgkan_pca.py \
    --test_dir "${TEST_DIR}" \
    --train_dir "${TRAIN_DIR}" \
    --gparc_ckpt "${GPARC_CKPT}" \
    --mgkan_ckpt "${MGKAN_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda

echo "Done! Figures in ${OUTPUT_DIR}"