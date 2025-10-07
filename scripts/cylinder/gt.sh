#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --account=sds_baek_energetic
#SBATCH --job-name=gparc_gt_viz
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=16G

# Load modules
module purge
module load apptainer

# Configuration
SIMULATION_FILE="/standard/sds_baek_energetic/von_karman_vortex/Reynolds 1~150/processed_parc/normalized/train/Reynolds_150_raw_data_normalized.pt"

OUTPUT_DIR="/standard/sds_baek_energetic/von_karman_vortex/Reynolds 1~150/ground_truth_viz/$(date +%Y%m%d_%H%M%S)"

# Container path
CONTAINER="/share/resources/containers/apptainer/pytorch-2.4.0.sif"

# Visualization parameters
NUM_STATIC_FEATS=3
SKIP_INDICES="3 4 5"  # Skip z_velocity, x_vorticity, y_vorticity
TIMESTEP=0            # Which timestep to visualize in static plot
FPS=2                 # Frames per second for animation

# Options
CREATE_ANIMATION=true  # Set to false to skip animation (faster)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting ground truth visualization..."
echo "Simulation file: $SIMULATION_FILE"
echo "Output: $OUTPUT_DIR"
echo "Static features: $NUM_STATIC_FEATS"
echo "Skipped dynamic indices: $SKIP_INDICES"
echo "Timestep for static plot: $TIMESTEP"
echo "Animation: $CREATE_ANIMATION"

# Build command arguments
CMD_ARGS=(
    --file "$SIMULATION_FILE"
    --output_dir "$OUTPUT_DIR"
    --num_static_feats "$NUM_STATIC_FEATS"
    --skip_dynamic_indices $SKIP_INDICES
    --timestep "$TIMESTEP"
    --fps "$FPS"
)

# Add flag to skip animation if disabled
if [ "$CREATE_ANIMATION" = false ]; then
    CMD_ARGS+=(--no_animation)
fi

# Run visualization (CPU only - no --nv flag)
apptainer run "$CONTAINER" ground_truth.py "${CMD_ARGS[@]}"

echo ""
echo "Visualization completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"