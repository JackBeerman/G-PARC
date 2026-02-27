#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J burg_concat
#SBATCH -o burg_concat.out
#SBATCH -e burg_concat.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 12:00:00
#SBATCH -c 8
#SBATCH --mem=60G

echo "================================================================"
echo "G-PARCv2 BURGERS — CONCAT+MLP FUSION (no SPADE)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "================================================================"

module purge
module load apptainer

# Paths
DATA_ROOT="/scratch/jtb3sud/processed_burgers_graph"
OUTPUT_DIR="/scratch/jtb3sud/burgers_v2_concat"
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# Architecture
HIDDEN_CHANNELS=64
FEATURE_OUT=128
NUM_FE_LAYERS=4
FUSION_HIDDEN_DIM=128
DIFFUSION_TYPE="fd"

# Training
INTEGRATOR="euler"
NUM_EPOCHS=250
LR=1e-4
SEQ_LEN=4
GRAD_CLIP=1.0

# Scheduled Sampling
SS_SCHEDULE="linear"
SS_INITIAL=0.0
SS_FINAL=0.0

echo ""
echo "Configuration:"
echo "  Diffusion:  $DIFFUSION_TYPE"
echo "  FiLM:       enabled"
echo "  Fusion:     concat+MLP (hidden=$FUSION_HIDDEN_DIM, no zero_init)"
echo "  Integrator: $INTEGRATOR"
echo "  LR: $LR | Epochs: $NUM_EPOCHS"
echo "  Seq Len: $SEQ_LEN"
echo "  FE: layers=$NUM_FE_LAYERS hidden=$HIDDEN_CHANNELS out=$FEATURE_OUT"
echo "  Scheduled Sampling: $SS_SCHEDULE ($SS_INITIAL → $SS_FINAL)"
echo "================================================================"

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
    --num_fe_layers "$NUM_FE_LAYERS" \
    --fusion_hidden_dim "$FUSION_HIDDEN_DIM" \
    --diffusion_type "$DIFFUSION_TYPE" \
    --use_film \
    --seq_len "$SEQ_LEN" \
    --grad_clip_norm "$GRAD_CLIP" \
    --ss_schedule "$SS_SCHEDULE" \
    --ss_initial_ratio "$SS_INITIAL" \
    --ss_final_ratio "$SS_FINAL" \
    --device auto \
    --num_workers 4

EXIT_CODE=$?
echo ""
echo "================================================================"
echo "Finished at: $(date) | Exit: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then echo "✅ SUCCESS"; else echo "❌ FAILED"; fi
echo "================================================================"