#!/bin/bash
#SBATCH -A sds_baek_energetic
#SBATCH -J gno_gino_benchmark
#SBATCH -o gino_benchmark_%j.out
#SBATCH -e gino_benchmark_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 1:00:00
#SBATCH -c 4
#SBATCH --mem=192G

echo "=========================================="
echo "  GNO / GINO MEMORY BENCHMARK"
echo "  $(date)"
echo "  Node: $(hostname)"
echo "=========================================="

module purge
module load apptainer

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ================================================================
# TEST DATA DIRECTORIES (same as main benchmark)
# ================================================================
ST_TEST_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/test_cases_normalized"
EL_TEST_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/full_test"
RV_TEST_DIR="/standard/sds_baek_energetic/HEC_RAS (River)/pt_test_normalized"

echo ""
echo "Configuration:"
echo "  ST: $ST_TEST_DIR"
echo "  EL: $EL_TEST_DIR"
echo "  RV: $RV_TEST_DIR"
echo "=========================================="

# NOTE: neuraloperator must be installed in the container or via pip first
# If not in container, uncomment:
# pip install neuraloperator --break-system-packages

apptainer run --nv "$CONTAINER" benchmark_gino.py \
    --st_test_dir "$ST_TEST_DIR" \
    --el_test_dir "$EL_TEST_DIR" \
    --rv_test_dir "$RV_TEST_DIR" \
    --grid_res 32 \
    --device cuda

echo ""
echo "=========================================="
echo "✓ Done: $(date)"
echo "=========================================="