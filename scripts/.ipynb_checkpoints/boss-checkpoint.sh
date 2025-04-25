#!/usr/bin/env bash

#SBATCH -A sds_baek_energetic       # Account name
#SBATCH -J seq1                    # Job name
#SBATCH -o %x.out                   # Standard output
#SBATCH -e %x.err                   # Standard error
#SBATCH -p gpu                      # Partition
#SBATCH --gres=gpu:a100:1           # Request 1 GPU
#SBATCH -t 14:00:00                 # Wall time
#SBATCH -c 4                        # CPU cores
#SBATCH --mem=80G                   # Memory

# Load modules
module purge
module load apptainer
module load pytorch/2.0.1

# Define paths
TRAIN_DIR="/scratch/jtb3sud/Individual_Simulations/train/normalized_data"
VAL_DIR="/scratch/jtb3sud/Individual_Simulations/val/normalized_data"
TEST_DIR="/scratch/jtb3sud/Individual_Simulations/test/normalized_data"
SAVE_DIR="/home/jtb3sud/G-PARC/weights"
#CHECK_DIR="/home/jtb3sud/G-PARC/scripts/complete_model.pth"

# Container path
CONTAINER="/share/resources/containers/apptainer/pytorch-2.0.1.sif"

# Additional arguments
NUM_EPOCHS=25
SEQ_LEN=1
LR=1e-4

# Now run your 'final.py' script inside the container
apptainer run --nv "$CONTAINER" \
  complete.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir   "$VAL_DIR" \
    --test_dir  "$TEST_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --lr "$LR" \
    --save_dir "$SAVE_DIR" \
    --device "cuda"
