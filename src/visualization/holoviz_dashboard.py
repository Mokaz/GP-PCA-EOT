# File: dashboard.py
import panel as pn
import plotly.graph_objects as go
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

import sys

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_ROOT))
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result
from src.global_project_paths import SIMDATA_PATH
from src.utils.geometry_utils import compute_estimated_shape_from_params, compute_exact_vessel_shape_global
from src.senfuslib.plotting import show_consistency
from src.senfuslib.analysis import ConsistencyAnalysis
from src.states.states import State_PCA # <-- Import State_PCA

# --- 1. Setup and Widget Definition ---
pn.extension('plotly', 'tabulator')

# Find all pickle files in the simulation data directory
pickle_files = sorted([f.name for f in Path(SIMDATA_PATH).glob("*.pkl")], reverse=True)
file_selector = pn.widgets.Select(name='Select Simulation File', options=pickle_files)

# --- 2. Cached Data Loading Function ---
@pn.cache(max_items=5) # Cache up to 5 loaded simulation files
def load_data(filename):
    """Loads simulation result and performs initial analysis. Cached for performance."""
    if not filename:
        return None
    
    print(f"Loading and processing {filename}...")
    with open(Path(SIMDATA_PATH) / filename, "rb") as f:
        sim_result = pickle.load(f)
    
    # Pre-calculate the full consistency analysis
    consistency_analyzer = create_consistency_analysis_from_sim_result(sim_result)
    
    # Extract config and PCA params for plotting functions
    config = sim_result.config
    pca_params = np.load(Path(config.tracker.PCA_parameters_path))
    
    return {
        "sim_result": sim_result,
        "consistency_analyzer": consistency_analyzer,
        "config": config,
        "pca_params": pca_params,
    }

# --- 3. Dynamic Dashboard Generation ---
@pn.depends(file_selector.param.value)
def create_dashboard_view(filename):
    """Creates the main dashboard view depending on the selected file."""
    loaded_data = load_data(filename)
    if not loaded_data:
        return pn.pane.Markdown("### Please select a simulation file to begin.")

    # Unpack loaded data
    sim_result = loaded_data["sim_result"]
    consistency_analyzer = loaded_data["consistency_analyzer"]
    config = loaded_data["config"]
    pca_params = loaded_data["pca_params"]
    
    PCA_eigenvectors_M = pca_params['eigenvectors'][:, :config.tracker.N_pca].real
    fourier_coeff_mean = pca_params['mean']
    ground_truth_states = list(sim_result.ground_truth_ts.values)

    # --- Define Widgets for the selected data ---
    frame_slider = pn.widgets.IntSlider(name='Frame', start=0, end=sim_result.config.sim.num_frames - 1, step=1)
    
    # Get state names by creating a dummy instance with placeholder values
    state_names = ['x', 'y', 'yaw', 'vel_x', 'vel_y', 'yaw_rate', 'length', 'width', 'pca_coeffs']
    nees_states_checklist = pn.widgets.CheckBoxGroup(name='NEES States', options=state_names)

    # --- Define Interactive Plotting Functions for this specific data ---
    @pn.depends(frame_slider.param.value)
    def update_plotly_view(frame_idx):
        gt_state = ground_truth_states[frame_idx + 1]
        tracker_result = sim_result.tracker_results_ts.values[frame_idx]
        est_state = tracker_result.state_posterior.mean
        z_lidar_cart = tracker_result.measurements.reshape((-1, 2))
        locationx = [s.x for s in ground_truth_states[:frame_idx + 2]]
        locationy = [s.y for s in ground_truth_states[:frame_idx + 2]]
        shape_x, shape_y = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)
        est_shape_x, est_shape_y = compute_estimated_shape_from_params(
            est_state.x, est_state.y, est_state.yaw, est_state.length, est_state.width, est_state.pca_coeffs,
            fourier_coeff_mean, PCA_eigenvectors_M, config.extent.angles, config.extent.N_fourier
        )
        lidar_ray_x, lidar_ray_y = [], []
        for z_pos in z_lidar_cart:
            dist = np.linalg.norm(z_pos - np.array(config.lidar.lidar_position))
            if dist < config.lidar.max_distance:
                lidar_ray_x.extend([config.lidar.lidar_position[0], z_pos[0], None])
                lidar_ray_y.extend([config.lidar.lidar_position[1], z_pos[1], None])
        est_pos = np.array([est_state.x, est_state.y])
        est_heading = est_state.yaw
        arrow_length = 5.0
        arrow_end = est_pos + arrow_length * np.array([np.cos(est_heading), np.sin(est_heading)])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=locationy, y=locationx, mode='lines', name='Vessel Path', line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=shape_y, y=shape_x, mode='lines', name='Vessel Extent (GT)', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=est_shape_y, y=est_shape_x, mode='lines', name='Estimated Extent', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=lidar_ray_y, y=lidar_ray_x, mode='lines+markers', name='LiDAR Rays', line=dict(color='red', width=1)))
        fig.add_trace(go.Scatter(x=[est_pos[1], arrow_end[1]], y=[est_pos[0], arrow_end[0]], mode='lines', name='Estimated Heading', line=dict(color='purple', width=2)))
        fig.update_layout(
            title=f"Frame: {frame_idx}", plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(range=[-60, 60], constrain='domain', gridcolor='rgb(200, 200, 200)', zerolinecolor='rgb(200, 200, 200)', title='East [m]'),
            yaxis=dict(range=[-15, 35], scaleanchor="x", scaleratio=1, gridcolor='rgb(200, 200, 200)', zerolinecolor='rgb(200, 200, 200)', title='North [m]'),
            legend=dict(x=1.05, y=1)
        )
        return pn.pane.Plotly(fig, config={'responsive': True})

    @pn.depends(nees_states_checklist.param.value)
    def update_nees_view(selected_states):
        if not selected_states:
            return pn.pane.Markdown("### Select states to show NEES plot.")
        fig, axs = plt.subplots(len(selected_states), 1, sharex=True, figsize=(8, 2 * len(selected_states)))
        if len(selected_states) == 1:
            axs = [axs]
        show_consistency(analysis=consistency_analyzer, fields_nees=selected_states, axs_nees=axs)
        plt.tight_layout()
        return pn.pane.Matplotlib(fig, tight=True)

    # --- Return the layout for the selected file ---
    return pn.Column(
        pn.Row(
            pn.Column(
                "### 2D Simulation View",
                update_plotly_view,
                sizing_mode='stretch_width'
            ),
            pn.Column(
                "### NEES Consistency",
                update_nees_view,
                sizing_mode='stretch_width'
            )
        ),
        pn.Column(frame_slider, nees_states_checklist, sizing_mode='stretch_width')
    )

# --- 4. Assemble the Final Dashboard Layout ---
dashboard = pn.template.FastListTemplate(
    site="Filter Analysis Dashboard",
    title="GP-PCA-EOT Simulation Analysis",
    sidebar=[
        pn.pane.Markdown("## Controls"),
        file_selector,
    ],
    main=[create_dashboard_view]
)

# To run this, save it as a .py file and run from the terminal:
# panel serve holoviz_dashboard.py --show
dashboard.servable()
