"""
visualizations — shared evaluation utilities
=============================================
Reusable metrics, rendering, GIF creation, dashboards, and hydrology
functions for G-PARCv2, MeshGraphNet, and MeshGraphKAN evaluators
across river, elastoplastic, and shock tube datasets.

Usage:
    from visualizations.metrics import compute_rrmse, compute_rrmse_per_variable
    from visualizations.hydrology import nse, csi, compute_overall_hydrology_metrics
    from visualizations.river_viz import create_scalar_field_gif, create_river_visualizations
    from visualizations.elasto_viz import create_reference_gif, create_elasto_visualizations
    from visualizations.shocktube_viz import create_shocktube_visualizations, plot_rollout_error_growth
    from visualizations.mesh_io import load_mesh_for_sim
    from visualizations.denorm import denormalize_all, load_denorm_extrema, load_normalization_metadata
    from visualizations.dashboard import plot_river_dashboard, plot_shocktube_dashboard
"""

from visualizations.metrics import *
from visualizations.hydrology import *
from visualizations.denorm import *
from visualizations.mesh_io import *
from visualizations.river_viz import *
from visualizations.elasto_viz import *
from visualizations.shocktube_viz import *
from visualizations.dashboard import *
from visualizations.selection import *
