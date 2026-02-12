#!/usr/bin/env bash
#SBATCH -J eval_burgers_heun
#SBATCH -o eval_burgers_heun.out
#SBATCH -e eval_burgers_heun.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -t 00:10:00
#SBATCH --mem=32G

module purge
module load apptainer

# 1. UPDATE MODEL PATH to Heun checkpoint
MODEL_PATH="/scratch/jtb3sud/burgers_euler_run/heun/burgers_best.pth"  # Changed!

TEST_DIR="/scratch/jtb3sud/processed_burgers_graph/train"
OUTPUT_DIR="./burgers_eval_heun_results/train"  # Optional: rename output dir

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# Match Training Config
HIDDEN=64
HEADS=4
DEPTH=3

# 2. UPDATE INTEGRATOR to heun
INTEGRATOR="heun"  # Changed!

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
    --max_sequences 3 \
    --create_gifs