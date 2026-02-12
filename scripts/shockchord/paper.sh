#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic       
#SBATCH -J model_comparison         
#SBATCH -o %x.out                   
#SBATCH -e %x.err                   
#SBATCH -p standard                 
#SBATCH -t 01:00:00                  
#SBATCH -c 4                        
#SBATCH --mem=16G                   

# Load modules
module purge
module load apptainer

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model prediction directories
PARCV2_DIR="/home/jtb3sud/PARCtorch/PARCtorch/scripts/paper_figures_parcv2"
GPARC_DIR="/project/vil_baek/psaap/paper_figures"

# Output directory for comparison figures
OUTPUT_DIR="/project/vil_baek/psaap/model_comparisons"

# Container path
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

# ============================================================================
# CASE SELECTION
# ============================================================================

# Mode: "all" to compare all common cases, or "specific" for listed cases
COMPARISON_MODE="all"  # Options: "all" or "specific"

# Specific cases to compare (only used if COMPARISON_MODE="specific")
# These should be the case names without any suffix (e.g., "p_L_162500_rho_L_1.75_test_with_pos")
SPECIFIC_CASES=(
    "p_L_162500_rho_L_1.75_test_with_pos"
    "p_L_143750_rho_L_2.0_test_with_pos"
    "p_L_56250_rho_L_0.625_test_with_pos"
)

# ============================================================================
# VISUALIZATION OPTIONS
# ============================================================================

# Timestep selection
TIMESTEP_MODE="auto"  # Options: "auto" or "manual"
MANUAL_TIMESTEPS="0 10 20 30 39"  # Used if TIMESTEP_MODE="manual"
AUTO_TIMESTEPS=5  # Number of timesteps to auto-select if TIMESTEP_MODE="auto"

# Data normalization
USE_NORMALIZED=false  # Set to true to use normalized data instead of denormalized

# Output options
OUTPUT_FORMAT="png"  # Options: "png" or "pdf"
DPI=300  # Resolution (300 for publication, 150 for preview)
FIGURE_WIDTH=16  # Width of comparison figures in inches

# Additional comparison features
INCLUDE_ERROR_PLOTS=true  # Generate separate error comparison plots
INCLUDE_METRICS_TABLE=true  # Generate quantitative metrics tables

# ============================================================================
# SETUP
# ============================================================================

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "Multi-Model Comparison Visualization"
echo "========================================="
echo "PARCv2 predictions: $PARCV2_DIR"
echo "GPARC predictions: $GPARC_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Comparison Configuration:"
echo "  Mode: $COMPARISON_MODE"
if [ "$COMPARISON_MODE" = "specific" ]; then
    echo "  Specific cases: ${#SPECIFIC_CASES[@]}"
fi
echo "  Timestep mode: $TIMESTEP_MODE"
if [ "$TIMESTEP_MODE" = "manual" ]; then
    echo "  Manual timesteps: $MANUAL_TIMESTEPS"
else
    echo "  Auto-selecting $AUTO_TIMESTEPS timesteps"
fi
echo "  Data type: $([ "$USE_NORMALIZED" = true ] && echo "Normalized" || echo "Denormalized")"
echo "  Output format: $OUTPUT_FORMAT"
echo "  Figure width: ${FIGURE_WIDTH} inches"
echo "  DPI: $DPI"
echo "  Error plots: $([ "$INCLUDE_ERROR_PLOTS" = true ] && echo "Yes" || echo "No")"
echo "  Metrics tables: $([ "$INCLUDE_METRICS_TABLE" = true ] && echo "Yes" || echo "No")"
echo "========================================="
echo ""

# ============================================================================
# VERIFY DIRECTORIES EXIST
# ============================================================================

if [ ! -d "$PARCV2_DIR" ]; then
    echo "ERROR: PARCv2 prediction directory not found: $PARCV2_DIR"
    exit 1
fi

if [ ! -d "$GPARC_DIR" ]; then
    echo "ERROR: GPARC prediction directory not found: $GPARC_DIR"
    exit 1
fi

echo "Checking for prediction files..."
PARCV2_FILES=$(find "$PARCV2_DIR" -name "*_predictions_*.npy" | wc -l)
GPARC_FILES=$(find "$GPARC_DIR" -name "*_predictions_*.npy" | wc -l)

echo "  PARCv2 prediction files: $PARCV2_FILES"
echo "  GPARC prediction files: $GPARC_FILES"

if [ "$PARCV2_FILES" -eq 0 ] || [ "$GPARC_FILES" -eq 0 ]; then
    echo "ERROR: No prediction files found in one or both directories"
    exit 1
fi
echo ""

# ============================================================================
# BUILD COMMAND ARGUMENTS
# ============================================================================

CMD_ARGS=(
    --parcv2_dir "$PARCV2_DIR"
    --gparc_dir "$GPARC_DIR"
    --output_dir "$OUTPUT_DIR"
    --output_format "$OUTPUT_FORMAT"
    --dpi "$DPI"
    --figure_width "$FIGURE_WIDTH"
)

# Add case selection
if [ "$COMPARISON_MODE" = "specific" ]; then
    CMD_ARGS+=(--cases "${SPECIFIC_CASES[@]}")
    echo "Comparing specific cases:"
    for case in "${SPECIFIC_CASES[@]}"; do
        echo "  - $case"
    done
else
    echo "Comparing all common cases found in both directories"
fi
echo ""

# Add timestep selection
if [ "$TIMESTEP_MODE" = "manual" ]; then
    CMD_ARGS+=(--timesteps $MANUAL_TIMESTEPS)
else
    CMD_ARGS+=(--auto_timesteps "$AUTO_TIMESTEPS")
fi

# Add data normalization option
if [ "$USE_NORMALIZED" = true ]; then
    CMD_ARGS+=(--use_normalized)
fi

# Add error plots option
if [ "$INCLUDE_ERROR_PLOTS" = true ]; then
    CMD_ARGS+=(--include_error_plots)
fi

# Add metrics table option
if [ "$INCLUDE_METRICS_TABLE" = true ]; then
    CMD_ARGS+=(--include_metrics_table)
fi

# ============================================================================
# RUN COMPARISON
# ============================================================================

echo "Starting model comparison..."
echo ""

apptainer run "$CONTAINER" \
    compare.py \
    "${CMD_ARGS[@]}"

COMPARISON_STATUS=$?

# ============================================================================
# POST-PROCESSING AND SUMMARY
# ============================================================================

if [ $COMPARISON_STATUS -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Model comparison completed successfully!"
    echo "========================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    
    # Count generated files
    COMPARISON_IMAGES=$(find "$OUTPUT_DIR" -name "comparison_*.${OUTPUT_FORMAT}" | wc -l)
    ERROR_IMAGES=$(find "$OUTPUT_DIR" -name "error_*.${OUTPUT_FORMAT}" | wc -l)
    METRICS_FILES=$(find "$OUTPUT_DIR" -name "metrics_*.txt" | wc -l)
    
    echo "Generated files:"
    echo "  Comparison figures: $COMPARISON_IMAGES"
    if [ "$INCLUDE_ERROR_PLOTS" = true ]; then
        echo "  Error plots: $ERROR_IMAGES"
    fi
    if [ "$INCLUDE_METRICS_TABLE" = true ]; then
        echo "  Metrics tables: $METRICS_FILES"
    fi
    echo ""
    
    # List comparison files by feature
    echo "Comparison figures by feature:"
    for feature in "Density_ρ" "x-Momentum_ρu" "Energy_E"; do
        feature_files=$(find "$OUTPUT_DIR" -name "comparison_*_${feature}.${OUTPUT_FORMAT}" | wc -l)
        if [ "$feature_files" -gt 0 ]; then
            echo "  ${feature}: $feature_files case(s)"
        fi
    done
    echo ""
    
    # Create summary file
    SUMMARY_FILE="$OUTPUT_DIR/comparison_summary.txt"
    cat > "$SUMMARY_FILE" << EOF
Multi-Model Comparison Summary
==============================
Date: $(date)
Job ID: ${SLURM_JOB_ID:-N/A}

Directories:
------------
PARCv2: $PARCV2_DIR
GPARC: $GPARC_DIR
Output: $OUTPUT_DIR

Configuration:
--------------
Comparison mode: $COMPARISON_MODE
Timestep mode: $TIMESTEP_MODE
$([ "$TIMESTEP_MODE" = "manual" ] && echo "Timesteps: $MANUAL_TIMESTEPS" || echo "Auto timesteps: $AUTO_TIMESTEPS")
Data type: $([ "$USE_NORMALIZED" = true ] && echo "Normalized" || echo "Denormalized")
Output format: $OUTPUT_FORMAT
DPI: $DPI
Figure width: ${FIGURE_WIDTH} inches
Error plots: $([ "$INCLUDE_ERROR_PLOTS" = true ] && echo "Yes" || echo "No")
Metrics tables: $([ "$INCLUDE_METRICS_TABLE" = true ] && echo "Yes" || echo "No")

Results:
--------
Comparison figures: $COMPARISON_IMAGES
$([ "$INCLUDE_ERROR_PLOTS" = true ] && echo "Error plots: $ERROR_IMAGES")
$([ "$INCLUDE_METRICS_TABLE" = true ] && echo "Metrics tables: $METRICS_FILES")

Cases Compared:
---------------
EOF
    
    # List all cases that were compared
    find "$OUTPUT_DIR" -name "comparison_*_Density_ρ.${OUTPUT_FORMAT}" -exec basename {} \; | \
        sed 's/comparison_//;s/_Density_ρ.'${OUTPUT_FORMAT}'//' | sort >> "$SUMMARY_FILE"
    
    echo "Summary saved to: $SUMMARY_FILE"
    echo ""
    
    # Print sample commands for viewing results
    echo "Quick view commands:"
    echo "  # View all comparison figures:"
    echo "  ls -lh $OUTPUT_DIR/comparison_*.${OUTPUT_FORMAT}"
    echo ""
    if [ "$INCLUDE_ERROR_PLOTS" = true ]; then
        echo "  # View all error plots:"
        echo "  ls -lh $OUTPUT_DIR/error_*.${OUTPUT_FORMAT}"
        echo ""
    fi
    if [ "$INCLUDE_METRICS_TABLE" = true ]; then
        echo "  # View metrics for a specific case:"
        echo "  cat $OUTPUT_DIR/metrics_*.txt"
        echo ""
    fi
    
    echo "========================================="
    echo "Comparison figures are ready for your paper!"
    echo "========================================="
    
else
    echo ""
    echo "========================================="
    echo "✗ ERROR: Model comparison failed!"
    echo "========================================="
    echo "Check the error log: ${SLURM_JOB_NAME}.err"
    echo ""
    echo "Common issues:"
    echo "  1. Missing prediction files in one or both directories"
    echo "  2. Mismatched case names between directories"
    echo "  3. Incorrect data format (normalized vs denormalized)"
    echo "  4. Invalid timestep indices"
    echo ""
    echo "Debug commands:"
    echo "  # Check PARCv2 files:"
    echo "  ls -lh $PARCV2_DIR/*.npy"
    echo ""
    echo "  # Check GPARC files:"
    echo "  ls -lh $GPARC_DIR/*.npy"
    echo ""
    exit 1
fi

# ============================================================================
# OPTIONAL: GENERATE COMBINED REPORT
# ============================================================================

if [ $COMPARISON_STATUS -eq 0 ] && [ "$INCLUDE_METRICS_TABLE" = true ]; then
    echo "Generating combined metrics report..."
    
    COMBINED_REPORT="$OUTPUT_DIR/combined_metrics_report.txt"
    
    echo "Combined Model Comparison Metrics" > "$COMBINED_REPORT"
    echo "=================================" >> "$COMBINED_REPORT"
    echo "Date: $(date)" >> "$COMBINED_REPORT"
    echo "" >> "$COMBINED_REPORT"
    
    # Combine all individual metrics files
    for metrics_file in "$OUTPUT_DIR"/metrics_*.txt; do
        if [ -f "$metrics_file" ]; then
            echo "" >> "$COMBINED_REPORT"
            echo "----------------------------------------" >> "$COMBINED_REPORT"
            cat "$metrics_file" >> "$COMBINED_REPORT"
        fi
    done
    
    echo "Combined metrics report saved to: $COMBINED_REPORT"
fi

echo ""
echo "Job completed at: $(date)"