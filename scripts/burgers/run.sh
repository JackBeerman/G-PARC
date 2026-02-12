#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J burgers_euler
#SBATCH -o burgers_euler_seq4.out
#SBATCH -e burgers_euler_seq4.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 72:00:00
#SBATCH -c 8
#SBATCH --mem=60G

echo "========================================"
echo "G-PARC Burgers (Euler)"
echo "========================================"

module purge
module load apptainer

# Point to where you saved the processed graphs
DATA_ROOT="/scratch/jtb3sud/processed_burgers_graph"
OUTPUT_DIR="/scratch/jtb3sud/burgers_euler_run/201_250seq4"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# Configuration matches pixel model capacity
INTEGRATOR="euler"
NUM_EPOCHS=50
LR=1e-5

# Architecture 
HIDDEN_CHANNELS=64
FEATURE_OUT=64
DEPTH=3
HEADS=4
SPADE_HEADS=2
ZERO_INIT="--zero_init"

mkdir -p "$OUTPUT_DIR"

apptainer run --nv "$CONTAINER" train_burgers.py \
    --train_dir "$DATA_ROOT/train" \
    --val_dir "$DATA_ROOT/val" \
    --test_dir "$DATA_ROOT/test" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --lr "$LR" \
    --integrator "$INTEGRATOR" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT" \
    --depth "$DEPTH" \
    --heads "$HEADS" \
    --spade_heads "$SPADE_HEADS" \
    --resume "/scratch/jtb3sud/burgers_euler_run/101_200/burgers_latest.pth" \
    --seq_len 4 \
    --grad_clip_norm 1.0 \
    --device auto \
    --num_workers 4


#--resume "/scratch/jtb3sud/burgers_euler_run/burgers_latest.pth" \
#    $ZERO_INIT \