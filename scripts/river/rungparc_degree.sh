#!/usr/bin/env bash

#SBATCH -A sds_baek_energetic       # Account name
#SBATCH -J gparc_degree16000               # Job name
#SBATCH -o %x.out                   # Standard output (%x expands to job name)
#SBATCH -e %x.err                   # Standard error
#SBATCH -p gpu                      # Partition
#SBATCH --gres=gpu:a6000:1           # Request 1 A100 GPU
#SBATCH -t 00:24:00                 # Wall time (hh:mm:ss)
#SBATCH -c 4                        # CPU cores
#SBATCH --mem=80G                   # Memory

# 1) Load modules
module purge
module load apptainer
module load pytorch/2.4.0

# 2) Define paths (adjust these as needed)
TRAIN_DIR="/scratch/jtb3sud/hecras/combined_normalized/train"
VAL_DIR="/scratch/jtb3sud/hecras/combined_normalized/val"
TEST_DIR="/scratch/jtb3sud/hecras/combined_normalized/test"
SAVE_DIR="/home/jtb3sud/G-PARC/weights/degree"
#CHECKPOINT_PATH="/home/jtb3sud/G-PARC/weights/new_river/modelseq10_ep200.pth"

# 3) Container path (adjust if needed)
CONTAINER="/share/resources/containers/apptainer/pytorch-2.4.0.sif"

# 4) Training arguments
NUM_EPOCHS=50
SEQ_LEN=1
LR=1e-5

# 5) Run your new integrated GPARC Python script inside the container
apptainer run --nv "$CONTAINER" \
  gparc_new.py \
    --train_dir  "$TRAIN_DIR" \
    --val_dir    "$VAL_DIR" \
    --test_dir   "$TEST_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --seq_len    "$SEQ_LEN" \
    --lr         "$LR" \
    --checkpoint "$CHECKPOINT_PATH" \
    --save_dir   "$SAVE_DIR" \
    --device     "cuda"
