#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J eval_MGN
#SBATCH -o eval_MGN.out
#SBATCH -e eval_MGN.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 0:30:00
#SBATCH -c 4
#SBATCH --mem=40G

echo "================================================================"
echo "MeshGraphNet Evaluation"
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

MODEL_PATH="/scratch/jtb3sud/meshgraphnet_baseline/best_model.pt"
STATS_PATH="/scratch/jtb3sud/meshgraphnet_baseline/normalization_stats.pt"
TEST_DIR="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/small/test"
OUTPUT_DIR="/scratch/jtb3sud/meshgraphnet_baseline/eval_rollout"
NORM_JSON="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/normalization_stats.json"

mkdir -p "$OUTPUT_DIR"

apptainer run --nv "$CONTAINER" eval.py \
    --model_path "$MODEL_PATH" \
    --stats_path "$STATS_PATH" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --norm_stats_json "$NORM_JSON" \
    --hidden_dim 128 \
    --num_layers 4 \
    --rollout_steps 37 \
    --max_sequences 10 \
    --denorm_method global_max \
    --create_gifs \
    --num_viz_simulations 3 \
    --gif_fps 10

echo ""
echo "✅ Evaluation complete! Results in: $OUTPUT_DIR"
