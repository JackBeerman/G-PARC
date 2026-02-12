#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic       
#SBATCH -J gparc_rollout_eval_test         
#SBATCH -o %x.out                   
#SBATCH -e %x.err                   
#SBATCH -p gpu                      
#SBATCH --gres=gpu:a100:1 
#SBATCH --constraint=a100_80gb
#SBATCH -t 03:20:00                  
#SBATCH -c 4                        
#SBATCH --mem=232G                   

# Load modules
module purge
module load apptainer

# Configuration options
# Set EVAL_MODE to either "directory" or "files"
EVAL_MODE="directory"  # Change to "files" to test specific files

# Define paths
MODEL_PATH="/standard/sds_baek_energetic/von_karman_vortex/processed_multi_dir/parc_model/train_large_75_4/shock_tube_best_model.pth"

# For directory mode
TEST_DIR="/standard/sds_baek_energetic/von_karman_vortex/full_data/split_normalized/test"

# For specific files mode - update these paths as needed
SPECIFIC_FILES=(
)

OUTPUT_DIR="/standard/sds_baek_energetic/von_karman_vortex/Reynolds 1~150/parc_model/eval/file_large"

# Container path
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# Model architecture parameters (updated for new configurable solver architecture)
SEQ_LEN=1
NUM_STATIC_FEATS=3
NUM_DYNAMIC_FEATS=4  # Updated: using 3 after skipping meaningless variable
SKIP_INDICES="3 4 5"       # Skip the third variable (0-indexed)

# Feature extractor parameters
HIDDEN_CHANNELS=256
FEATURE_OUT_CHANNELS=128
DEPTH=5
HEADS=2
DROPOUT=0.1

# Derivative solver parameters
DERIV_HIDDEN_CHANNELS=256
DERIV_NUM_LAYERS=4
DERIV_HEADS=2
DERIV_DROPOUT=0.1

# Integral solver parameters
INTEGRAL_HIDDEN_CHANNELS=256
INTEGRAL_NUM_LAYERS=4
INTEGRAL_HEADS=2
INTEGRAL_DROPOUT=0.1

# Rollout evaluation parameters
MAX_SEQUENCES=40
ROLLOUT_STEPS=15  # How many timesteps to predict into the future

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting GPARC rollout evaluation with configurable solvers..."
echo "Model: $MODEL_PATH"
echo "Evaluation mode: $EVAL_MODE"
echo "Output: $OUTPUT_DIR"
echo "Rollout steps: $ROLLOUT_STEPS"
echo "Using ${NUM_DYNAMIC_FEATS} dynamic features (skipping indices: ${SKIP_INDICES})"
echo "Derivative solver: ${DERIV_NUM_LAYERS} layers, ${DERIV_HIDDEN_CHANNELS} channels, ${DERIV_HEADS} heads"
echo "Integral solver: ${INTEGRAL_NUM_LAYERS} layers, ${INTEGRAL_HIDDEN_CHANNELS} channels, ${INTEGRAL_HEADS} heads"

# Build command arguments based on evaluation mode
if [ "$EVAL_MODE" = "directory" ]; then
    echo "Test data directory: $TEST_DIR"
    
    # Run rollout evaluation with directory mode
    apptainer run --nv "$CONTAINER" \
       test_mod.py \
        --model_path "$MODEL_PATH" \
        --test_dir "$TEST_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_static_feats "$NUM_STATIC_FEATS" \
        --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
        --skip_dynamic_indices $SKIP_INDICES \
        --hidden_channels "$HIDDEN_CHANNELS" \
        --feature_out_channels "$FEATURE_OUT_CHANNELS" \
        --depth "$DEPTH" \
        --heads "$HEADS" \
        --dropout "$DROPOUT" \
        --deriv_hidden_channels "$DERIV_HIDDEN_CHANNELS" \
        --deriv_num_layers "$DERIV_NUM_LAYERS" \
        --deriv_heads "$DERIV_HEADS" \
        --deriv_dropout "$DERIV_DROPOUT" \
        --deriv_use_residual \
        --integral_hidden_channels "$INTEGRAL_HIDDEN_CHANNELS" \
        --integral_num_layers "$INTEGRAL_NUM_LAYERS" \
        --integral_heads "$INTEGRAL_HEADS" \
        --integral_dropout "$INTEGRAL_DROPOUT" \
        --integral_use_residual \
        --max_sequences "$MAX_SEQUENCES" \
        --rollout_steps "$ROLLOUT_STEPS"

elif [ "$EVAL_MODE" = "files" ]; then
    echo "Specific test files:"
    for file in "${SPECIFIC_FILES[@]}"; do
        echo "  - $file"
    done
    
    # Run rollout evaluation with specific files mode
    apptainer run --nv "$CONTAINER" \
       test_mod.py \
        --model_path "$MODEL_PATH" \
        --test_files "${SPECIFIC_FILES[@]}" \
        --output_dir "$OUTPUT_DIR" \
        --num_static_feats "$NUM_STATIC_FEATS" \
        --num_dynamic_feats "$NUM_DYNAMIC_FEATS" \
        --skip_dynamic_indices $SKIP_INDICES \
        --hidden_channels "$HIDDEN_CHANNELS" \
        --feature_out_channels "$FEATURE_OUT_CHANNELS" \
        --depth "$DEPTH" \
        --heads "$HEADS" \
        --dropout "$DROPOUT" \
        --deriv_hidden_channels "$DERIV_HIDDEN_CHANNELS" \
        --deriv_num_layers "$DERIV_NUM_LAYERS" \
        --deriv_heads "$DERIV_HEADS" \
        --deriv_dropout "$DERIV_DROPOUT" \
        --deriv_use_residual \
        --integral_hidden_channels "$INTEGRAL_HIDDEN_CHANNELS" \
        --integral_num_layers "$INTEGRAL_NUM_LAYERS" \
        --integral_heads "$INTEGRAL_HEADS" \
        --integral_dropout "$INTEGRAL_DROPOUT" \
        --integral_use_residual \
        --max_sequences "$MAX_SEQUENCES" \
        --rollout_steps "$ROLLOUT_STEPS"

else
    echo "ERROR: EVAL_MODE must be either 'directory' or 'files'"
    exit 1
fi

echo "Rollout evaluation completed. Results saved to: $OUTPUT_DIR"
echo "Check rollout_evolution_*.png for temporal accuracy analysis"
echo "Check rollout_*.gif for animated visualizations"

# Optional: Display a summary of generated files
echo ""
echo "Generated files:"
find "$OUTPUT_DIR" -name "*.png" -o -name "*.gif" -o -name "*.json" | head -10