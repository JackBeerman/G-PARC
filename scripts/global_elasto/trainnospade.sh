#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J elasto_p99true
#SBATCH -o elasto_p99true.out
#SBATCH -e elasto_p99true.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH -t 72:00:00
#SBATCH -c 8
#SBATCH --mem=80G

echo "================================================================"
echo "G-PARC v2 ELASTOPLASTIC — NoSPADE (Concat+MLP Fusion)"
echo "================================================================"
echo ""
echo "Ablation: Replace SPADE (MappingAndRecon) with concat+MLP."
echo "Same approach that improved shock tube and river results."
echo "Everything else identical to the SPADE training run."
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# DATA (same as SPADE run)
# ============================================================
TRAIN_DIR="/scratch/jtb3sud/processed_elasto_plastic/p99clip/normalized/small/train"
VAL_DIR="/scratch/jtb3sud/processed_elasto_plastic/p99clip/normalized/small/val"
OUTPUT_DIR="/scratch/jtb3sud/gparcv2/p99true"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================
# TRAINING HYPERPARAMETERS (identical to SPADE run)
# ============================================================
NUM_EPOCHS=1500
SEQ_LEN=16
STRIDE=8
LR=3e-4
NUM_WORKERS=4
GRAD_CLIP_NORM=2.0

SS_SCHEDULE="linear"
SS_INITIAL_RATIO=0.0
SS_FINAL_RATIO=0.0

# ============================================================
# ARCHITECTURE (identical to SPADE run)
# ============================================================
NUM_LAYERS=4
HIDDEN_CHANNELS=128
FEATURE_OUT_CHANNELS=128
DROPOUT=0.0
USE_LAYER_NORM="--use_layer_norm"
USE_RELATIVE_POS="--use_relative_pos"
CLAMP_FLAG="--no_clamp_output"

# ============================================================
# PHYSICS (identical to SPADE run)
# SPADE args are still passed but ignored by NoSPADE differentiator
# ============================================================
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=2
USE_VON_MISES="--use_von_mises"
USE_VOLUMETRIC="--use_volumetric"
SPADE_HEADS=4
SPADE_CONCAT="--spade_concat"
SPADE_DROPOUT=0.1
ZERO_INIT="--zero_init"
MASK_ERODING="--mask_eroding"

NORM_STATS_FILE="/scratch/jtb3sud/processed_elasto_plastic/global_max/normalized/normalization_stats.json"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Output:     $OUTPUT_DIR"
echo "  Fusion:     Concat + MLP (no SPADE)"
echo "  LR:         $LR (cosine → ~0 over $NUM_EPOCHS epochs)"
echo "  Seq length: $SEQ_LEN"
echo "  Epochs:     $NUM_EPOCHS"
echo "================================================================"

if [ -f "$NORM_STATS_FILE" ]; then
    cp "$NORM_STATS_FILE" "$OUTPUT_DIR/normalization_stats.json"
    echo "✓ Copied normalization stats to output directory"
fi

apptainer run --nv "$CONTAINER" trainnospade.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --stride "$STRIDE" \
    --lr "$LR" \
    --num_static_feats "$NUM_STATIC_FEATS" \
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
    --integrator "euler" \
    --num_layers "$NUM_LAYERS" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --feature_out_channels "$FEATURE_OUT_CHANNELS" \
    --dropout "$DROPOUT" \
    $USE_LAYER_NORM \
    $USE_RELATIVE_POS \
    $USE_VON_MISES \
    $USE_VOLUMETRIC \
    --spade_heads "$SPADE_HEADS" \
    $SPADE_CONCAT \
    --spade_dropout "$SPADE_DROPOUT" \
    $ZERO_INIT \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    $MASK_ERODING \
    --num_workers "$NUM_WORKERS" \
    $CLAMP_FLAG \
    --ss_schedule "$SS_SCHEDULE" \
    --ss_initial_ratio "$SS_INITIAL_RATIO" \
    --ss_final_ratio "$SS_FINAL_RATIO"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training complete!"
    echo "  Compare against SPADE version at: /scratch/jtb3sud/elasto_graphconv_V2/2hop/"
else
    echo "❌ Training failed"
fi

exit $EXIT_CODE