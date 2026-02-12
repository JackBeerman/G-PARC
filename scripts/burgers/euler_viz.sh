#!/usr/bin/env bash
#SBATCH -J eval_burgers_euler4
#SBATCH -o eval_burgers_euler4.out
#SBATCH -e eval_burgers_euler4.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:1
#SBATCH -t 00:30:00
#SBATCH --mem=32G

module purge
module load apptainer

MODEL_PATH="/scratch/jtb3sud/burgers_euler_run/201_250seq4/burgers_latest.pth"
TEST_DIR="/scratch/jtb3sud/processed_burgers_graph/train_reversed"
OUTPUT_DIR="./burgers_eval_results/train_reversed/seq4"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# Match Training Config
HIDDEN=64
HEADS=4
DEPTH=3
INTEGRATOR="euler"

apptainer run --nv "$CONTAINER" evaluate_burgers.py \
    --model_path "$MODEL_PATH" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --hidden_channels "$HIDDEN" \
    --feature_out_channels "$HIDDEN" \
    --depth "$DEPTH" \
    --heads "$HEADS" \
    --integrator "$INTEGRATOR" \
    --rollout_steps 50 \
    --max_sequences 100 \
    --create_gifs