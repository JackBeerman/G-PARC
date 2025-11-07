#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic       
#SBATCH -J gparc_rollout_tennis         
#SBATCH -o %x.out                   
#SBATCH -e %x.err                   
#SBATCH -p gpu                      
#SBATCH --gres=gpu:a6000:1           
#SBATCH -t 00:10:00                  
#SBATCH -c 4                        
#SBATCH --mem=32G                   

# Load modules
module purge
module load apptainer

# Configuration options
# Set EVAL_MODE to either "directory" or "files"
EVAL_MODE="directory"  # Change to "files" to test specific files

# Define paths
MODEL_PATH="/project/vil_baek/psaap/tennis/training_files/tennis_training_20251008_123120_sinner/tennis_serve_best_model.pth"

# For directory mode
TEST_DIR="/project/vil_baek/psaap/tennis/player_jannik_sinner_data/test"

# For specific files mode - update these paths as needed
SPECIFIC_FILES=(
"/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/train_cases_normalized/p_L_137500_rho_L_2.0_train_with_pos_normalized.pt"
"/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/train_cases_normalized/p_L_112500_rho_L_1.25_train_with_pos_normalized.pt"
)

OUTPUT_DIR="/project/vil_baek/psaap/rollout_evaluation_tennis$(date +%Y%m%d_%H%M%S)"

# Container path
CONTAINER="/share/resources/containers/apptainer/pytorch-2.4.0.sif"

# Model architecture parameters (updated for new configurable solver architecture)
SEQ_LEN=1
NUM_STATIC_FEATS=0
NUM_DYNAMIC_FEATS=6  # Updated: using 3 after skipping meaningless variable

# Feature extractor parameters (for joint position processing)
HIDDEN_CHANNELS=64
FEATURE_OUT_CHANNELS=128
DEPTH=2
HEADS=4
DROPOUT=0.2

# Derivative solver parameters (for motion dynamics)
DERIV_HIDDEN_CHANNELS=128
DERIV_NUM_LAYERS=3
DERIV_HEADS=4
DERIV_DROPOUT=0.3

# Integral solver parameters (for next frame prediction)
INTEGRAL_HIDDEN_CHANNELS=128
INTEGRAL_NUM_LAYERS=3
INTEGRAL_HEADS=4
INTEGRAL_DROPOUT=0.3

# Rollout evaluation parameters
MAX_SEQUENCES=400
ROLLOUT_STEPS=80  # How many timesteps to predict into the future

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
        --hidden_channels "$HIDDEN_CHANNELS" \
        --feature_out_channels "$FEATURE_OUT_CHANNELS" \
        --depth "$DEPTH" \
        --pool_ratios 0.7 \
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
        --hidden_channels "$HIDDEN_CHANNELS" \
        --feature_out_channels "$FEATURE_OUT_CHANNELS" \
        --depth "$DEPTH" \
        --pool_ratios 0.7 \
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