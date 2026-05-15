#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J unified_eval
#SBATCH -o unified_eval.out
#SBATCH -e unified_eval.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 4:10:00
#SBATCH -c 4
#SBATCH --mem=192G

echo "================================================================"
echo "UNIFIED MULTI-DATASET EVALUATION"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "================================================================"

module purge
module load apptainer

# ---- Container & Script ----
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"
SCRIPT="$HOME/G-PARC/scripts/eval/eval.py"
OUTPUT_ROOT="/scratch/jtb3sud/unified_eval_results99"

# ================================================================
# SHOCK TUBE
# ================================================================
ST_TEST_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/test_cases_normalized"
ST_GPARCV1="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/shock_tube_20250927_104720_run_mod10_750/shock_tube_best_model.pth"
ST_GPARCV2="/scratch/jtb3sud/shocktube_v2_training/nospadeFAST/best_model.pth"
ST_MGKAN="/scratch/jtb3sud/delta/shocktube/run_101_300/best_model.pth"
ST_MGNET="/scratch/jtb3sud/meshgraphnet/shocktube/run1/best_model.pt"
ST_GSAGE="/scratch/jtb3sud/graphsage/shocktube/best_model.pth"
ST_ROLLOUT=40

# ================================================================
# ELASTOPLASTIC
# ================================================================
EL_DATA_ROOT="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small"
EL_TEST_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/full_test"
EL_NORM_STATS="$EL_DATA_ROOT/normalization_stats.json"
EL_GPARCV2="/scratch/jtb3sud/gparcv2/elasto_nospade/best_model.pth"
EL_GPARCV1="/scratch/jtb3sud/elasto_graphconv_V2/gparcv1/best_model.pth"
EL_MGKAN="/scratch/jtb3sud/delta/elasto/best_model.pth"
EL_MGN="/scratch/jtb3sud/meshgraphnet/elasto/run1/best_model.pt"
EL_GSAGE="/scratch/jtb3sud/graphsage/elasto/best_model.pth"

# ================================================================
# RIVER (HEC-RAS)
# ================================================================
RV_TEST_DIR="/standard/sds_baek_energetic/HEC_RAS (River)/pt_test_normalized"
RV_HEC_RAS_DIR="/standard/sds_baek_energetic/HEC_RAS (River)"
RV_EXTREMA="/standard/sds_baek_energetic/HEC_RAS (River)/global_y_extrema_test.pth"
RV_GPARCV1="/home/jtb3sud/G-PARC/weights/new_river/modelseq20_ep250.pth"
RV_GPARCV2="/scratch/jtb3sud/gparcv2/river/concat/20/best_model.pth"
RV_MGKAN="/scratch/jtb3sud/delta/river/run2_51_150/best_model.pth"
RV_MGNET="/scratch/jtb3sud/meshgraphnet/river/run1/best_model.pt"
RV_GSAGE="/scratch/jtb3sud/graphsage/whathe/best_model_river2.pth"

echo ""
echo "Configuration:"
echo "  Shock tube test:  $ST_TEST_DIR"
echo "  Elasto test:      $EL_TEST_DIR"
echo "  River test:       $RV_TEST_DIR"
echo "  Output:           $OUTPUT_ROOT"
echo "================================================================"

mkdir -p "$OUTPUT_ROOT"

apptainer run --nv "$CONTAINER" eval.py \
    --datasets shocktube elasto river \
    --output_dir "$OUTPUT_ROOT" \
    --device cuda \
    --num_viz 3 \
    \
    --st_test_dir "$ST_TEST_DIR" \
    --st_rollout_steps "$ST_ROLLOUT" \
    --st_models \
        "gparcv2:$ST_GPARCV2" \
        "gparcv1:$ST_GPARCV1" \
        "mgkan:$ST_MGKAN" \
        "mgnet:$ST_MGNET" \
        "gsage:$ST_GSAGE" \
    \
    --el_test_dir "$EL_TEST_DIR" \
    --el_norm_stats "$EL_NORM_STATS" \
    --el_models \
        "gparcv2:$EL_GPARCV2" \
        "gparcv1:$EL_GPARCV1" \
        "mgkan:$EL_MGKAN" \
        "mgn:$EL_MGN" \
        "graphsage:$EL_GSAGE" \
        "gparcv2_nospade:/scratch/jtb3sud/gparcv2/p99/best_model.pth" \
    \
    --rv_test_dir "$RV_TEST_DIR" \
    --rv_extrema "$RV_EXTREMA" \
    --rv_models \
        "gparcv2:$RV_GPARCV2" \
        "gparcv1:$RV_GPARCV1" \
        "mgkan:$RV_MGKAN" \
        "mgnet:$RV_MGNET" \
        "gsage:$RV_GSAGE"

echo ""
echo "================================================================"
echo "Finished: $(date)"
echo "Results:  $OUTPUT_ROOT"
echo "  shocktube/      — RRMSE, NMSE, SSIM over-time plots + table.tex + results.json"
echo "  elastoplastic/  — same"
echo "  river/          — same"
echo "  cross_dataset_summary.png (3 datasets)"
echo "================================================================"