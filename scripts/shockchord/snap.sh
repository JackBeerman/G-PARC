#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic       
#SBATCH -J gparc_snapshots         
#SBATCH -o %x.out                   
#SBATCH -e %x.err                   
#SBATCH -p gpu                      
#SBATCH --gres=gpu:a6000:1           
#SBATCH -t 00:30:00                  
#SBATCH -c 4                        
#SBATCH --mem=32G                   

# Load modules
module purge
module load apptainer

# Configuration options
# Set EVAL_MODE to either "directory" or "files"
EVAL_MODE="files"  # Change to "directory" to process all files in a directory

# Define paths
MODEL_PATH="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/shock_tube_20250927_104720_run_mod10_750/shock_tube_best_model.pth"

# For directory mode
TEST_DIR="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/test_cases_normalized"

# For specific files mode - update these paths to your interesting cases
SPECIFIC_FILES=(
"/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/test_cases_normalized/p_L_162500_rho_L_1.75_test_with_pos_normalized.pt"
"/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/test_cases_normalized/p_L_143750_rho_L_2.0_test_with_pos_normalized.pt"
"/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt/normalized_datasets/test_cases_normalized/p_L_56250_rho_L_0.625_test_with_pos_normalized.pt"
)

OUTPUT_DIR="/project/vil_baek/psaap/paper_figures"

# Container path
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# Model architecture parameters (MUST MATCH TRAINING)
NUM_STATIC_FEATS=2
NUM_DYNAMIC_FEATS=3  # Using 3 after skipping meaningless variable
SKIP_INDICES="2"     # Skip the third variable (0-indexed)

# Feature extractor parameters
HIDDEN_CHANNELS=32
FEATURE_OUT_CHANNELS=32
DEPTH=2
HEADS=2
DROPOUT=0.2

# Derivative solver parameters
DERIV_HIDDEN_CHANNELS=128
DERIV_NUM_LAYERS=3
DERIV_HEADS=4
DERIV_DROPOUT=0.2

# Integral solver parameters
INTEGRAL_HIDDEN_CHANNELS=128
INTEGRAL_NUM_LAYERS=3
INTEGRAL_HEADS=4
INTEGRAL_DROPOUT=0.2

# Snapshot generation parameters
TIMESTEP_MODE="auto"  # Options: "auto" or "manual"
MANUAL_TIMESTEPS="1 9 10 15 20"  # Used if TIMESTEP_MODE="manual"
AUTO_TIMESTEPS=5  # Number of timesteps to auto-select if TIMESTEP_MODE="auto"

# Output options
DPI=300  # Resolution for figures (300 for publication, 150 for preview)
FIGSIZE_WIDTH=12

# Optional: Generate single variable detailed snapshots
# Set to "density", "x_momentum", "total_energy", or leave empty for all variables
SINGLE_VARIABLE=""  # Empty means all variables

# Optional: Limit number of files from directory
MAX_FILES=10  # Only used in directory mode

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "GPARC Snapshot Generator for Publication"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Evaluation mode: $EVAL_MODE"
echo "Output: $OUTPUT_DIR"
echo "Timestep mode: $TIMESTEP_MODE"
if [ "$TIMESTEP_MODE" = "manual" ]; then
    echo "Manual timesteps: $MANUAL_TIMESTEPS"
else
    echo "Auto-selecting $AUTO_TIMESTEPS timesteps"
fi
echo "Figure size: ${FIGSIZE_WIDTH}x${FIGSIZE_HEIGHT}"
echo "DPI: $DPI"
echo ""
echo "Model Configuration:"
echo "  Static features: $NUM_STATIC_FEATS"
echo "  Dynamic features: $NUM_DYNAMIC_FEATS (skipping indices: $SKIP_INDICES)"
echo "  Feature extractor: ${HIDDEN_CHANNELS}ch → ${FEATURE_OUT_CHANNELS}ch (depth=${DEPTH}, heads=${HEADS})"
echo "  Derivative solver: ${DERIV_NUM_LAYERS} layers, ${DERIV_HIDDEN_CHANNELS}ch, ${DERIV_HEADS} heads"
echo "  Integral solver: ${INTEGRAL_NUM_LAYERS} layers, ${INTEGRAL_HIDDEN_CHANNELS}ch, ${INTEGRAL_HEADS} heads"
echo "========================================="
echo ""

# Build base command arguments (common to both modes)
BASE_ARGS=(
    --model_path "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --num_static_feats "$NUM_STATIC_FEATS"
    --num_dynamic_feats "$NUM_DYNAMIC_FEATS"
    --skip_dynamic_indices "$SKIP_INDICES"
    --hidden_channels "$HIDDEN_CHANNELS"
    --feature_out_channels "$FEATURE_OUT_CHANNELS"
    --depth "$DEPTH"
    --pool_ratios 0.2
    --heads "$HEADS"
    --dropout "$DROPOUT"
    --deriv_hidden_channels "$DERIV_HIDDEN_CHANNELS"
    --deriv_num_layers "$DERIV_NUM_LAYERS"
    --deriv_heads "$DERIV_HEADS"
    --deriv_dropout "$DERIV_DROPOUT"
    --deriv_use_residual
    --integral_hidden_channels "$INTEGRAL_HIDDEN_CHANNELS"
    --integral_num_layers "$INTEGRAL_NUM_LAYERS"
    --integral_heads "$INTEGRAL_HEADS"
    --integral_dropout "$INTEGRAL_DROPOUT"
    --integral_use_residual
    --dpi "$DPI"
    --figure_width "$FIGSIZE_WIDTH"
)

# Add timestep selection arguments
if [ "$TIMESTEP_MODE" = "manual" ]; then
    BASE_ARGS+=(--timesteps $MANUAL_TIMESTEPS)
else
    BASE_ARGS+=(--auto_timesteps "$AUTO_TIMESTEPS")
fi

# Add single variable option if specified
if [ -n "$SINGLE_VARIABLE" ]; then
    BASE_ARGS+=(--single_variable "$SINGLE_VARIABLE")
    echo "Generating detailed snapshots for: $SINGLE_VARIABLE"
else
    echo "Generating comparison snapshots for all variables"
fi

# Run based on evaluation mode
if [ "$EVAL_MODE" = "directory" ]; then
    echo "Processing all files in directory: $TEST_DIR"
    echo "Maximum files to process: $MAX_FILES"
    echo ""
    
    # Run snapshot generation with directory mode
    apptainer run --nv "$CONTAINER" \
        snapshot_images.py \
        "${BASE_ARGS[@]}" \
        --test_dir "$TEST_DIR" \
        --max_files "$MAX_FILES"

elif [ "$EVAL_MODE" = "files" ]; then
    echo "Processing specific files:"
    for file in "${SPECIFIC_FILES[@]}"; do
        echo "  - $(basename "$file")"
    done
    echo ""
    
    # Run snapshot generation with specific files mode
    apptainer run --nv "$CONTAINER" \
        snapshot_images.py \
        "${BASE_ARGS[@]}" \
        --test_files "${SPECIFIC_FILES[@]}"

else
    echo "ERROR: EVAL_MODE must be either 'directory' or 'files'"
    exit 1
fi

# Check if snapshot generation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Snapshot generation completed successfully!"
    echo "========================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated snapshot files:"
    find "$OUTPUT_DIR" -name "snapshot_*.png" -exec basename {} \; | sort
    echo ""
    echo "Total snapshots created: $(find "$OUTPUT_DIR" -name "snapshot_*.png" | wc -l)"
else
    echo ""
    echo "========================================="
    echo "✗ ERROR: Snapshot generation failed!"
    echo "========================================="
    echo "Check the error log: ${SLURM_JOB_NAME}.err"
    exit 1
fi

# Optional: Create a summary file
SUMMARY_FILE="$OUTPUT_DIR/snapshot_generation_summary.txt"
cat > "$SUMMARY_FILE" << EOF
GPARC Snapshot Generation Summary
==================================
Date: $(date)
Job ID: ${SLURM_JOB_ID}

Configuration:
--------------
Model: $MODEL_PATH
Mode: $EVAL_MODE
Timestep mode: $TIMESTEP_MODE
Output directory: $OUTPUT_DIR

Model Architecture:
-------------------
Static features: $NUM_STATIC_FEATS
Dynamic features: $NUM_DYNAMIC_FEATS
Skipped indices: $SKIP_INDICES
Feature extractor: ${HIDDEN_CHANNELS}→${FEATURE_OUT_CHANNELS} (depth=${DEPTH})
Derivative solver: ${DERIV_NUM_LAYERS} layers, ${DERIV_HIDDEN_CHANNELS} channels
Integral solver: ${INTEGRAL_NUM_LAYERS} layers, ${INTEGRAL_HIDDEN_CHANNELS} channels

Output Settings:
----------------
DPI: $DPI
Figure size: ${FIGSIZE_WIDTH}x${FIGSIZE_HEIGHT}
Single variable mode: ${SINGLE_VARIABLE:-"All variables"}

Generated Files:
----------------
EOF

# Add list of generated files to summary
find "$OUTPUT_DIR" -name "snapshot_*.png" -exec basename {} \; | sort >> "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "========================================="
echo "You can now use these snapshots in your paper!"
echo "========================================="