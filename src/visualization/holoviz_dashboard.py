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

# --- 1. Setup ---
pn.extension('plotly', 'tabulator')

# --- 2. Cached Data Loading Function ---
@pn.cache(max_items=5)
def load_data(filename):
    """Loads simulation result and performs initial analysis. Cached for performance."""
    if not filename:
        return None
    
    print(f"Loading and processing {filename}...")
    with open(Path(SIMDATA_PATH) / filename, "rb") as f:
        sim_result = pickle.load(f)
    
    consistency_analyzer = create_consistency_analysis_from_sim_result(sim_result)
    config = sim_result.config
    pca_params = np.load(Path(config.tracker.PCA_parameters_path))
    
    return {
        "sim_result": sim_result,
        "consistency_analyzer": consistency_analyzer,
        "config": config,
        "pca_params": pca_params,
    }

# --- 3. Create Widgets in a shared scope ---
pickle_files = sorted([f.name for f in Path(SIMDATA_PATH).glob("*.pkl")], reverse=True)
file_selector = pn.widgets.Select(name='Select Simulation File', options=pickle_files)

# These widgets are created once and will be updated
frame_slider = pn.widgets.IntSlider(name='Frame', start=0, end=0, step=1, visible=False)
nees_states_checklist = pn.widgets.CheckBoxGroup(name='NEES States', options=[], visible=False)


# --- 4. Define Interactive Plotting Functions ---

# This function now ONLY updates the widgets when the file changes
@pn.depends(file_selector.param.value, watch=True)
def update_widgets(filename):
    loaded_data = load_data(filename)
    if not loaded_data:
        frame_slider.visible = False
        nees_states_checklist.visible = False
        return

    sim_result = loaded_data["sim_result"]
    frame_slider.end = sim_result.config.sim.num_frames - 1
    frame_slider.value = 0 # Reset slider to start
    frame_slider.visible = True
    
    state_names = ['x', 'y', 'yaw', 'vel_x', 'vel_y', 'yaw_rate', 'length', 'width', 'pca_coeffs']
    nees_states_checklist.options = state_names
    nees_states_checklist.value = [] # Reset checklist
    nees_states_checklist.visible = True


@pn.depends(frame_slider.param.value, file_selector.param.value)
def get_plotly_view(frame_idx, filename):
    loaded_data = load_data(filename)
    if not loaded_data: return pn.pane.Markdown("### Select a file to begin.")
    
    # Unpack data needed for the plot
    sim_result = loaded_data["sim_result"]
    config = loaded_data["config"]
    pca_params = loaded_data["pca_params"]
    PCA_eigenvectors_M = pca_params['eigenvectors'][:, :config.tracker.N_pca].real
    fourier_coeff_mean = pca_params['mean']
    ground_truth_states = list(sim_result.ground_truth_ts.values)

    # --- Plotting logic ---
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

@pn.depends(nees_states_checklist.param.value, file_selector.param.value)
def get_nees_view(selected_states, filename):
    loaded_data = load_data(filename)
    if not loaded_data or not selected_states:
        return pn.pane.Markdown("### Select states to show NEES plot.")
    
    consistency_analyzer = loaded_data["consistency_analyzer"]
    fig, axs = plt.subplots(len(selected_states), 1, sharex=True, figsize=(8, 2 * len(selected_states)))
    if len(selected_states) == 1: axs = [axs]
    show_consistency(analysis=consistency_analyzer, fields_nees=selected_states, axs_nees=axs)
    plt.tight_layout()
    return pn.pane.Matplotlib(fig, tight=True)


# --- 5. Assemble the Final Dashboard Layout ---
dashboard = pn.template.FastListTemplate(
    site="Filter Analysis Dashboard",
    title="GP-PCA-EOT Simulation Analysis",
    sidebar=[
        pn.pane.Markdown("## Controls"),
        file_selector,
        frame_slider,
        nees_states_checklist,
    ],
    main=[
        pn.Row(
            pn.Column("### 2D Simulation View", get_plotly_view, sizing_mode='stretch_width'),
            pn.Column("### NEES Consistency", get_nees_view, sizing_mode='stretch_width')
        )
    ]
)

# To run this, save it as a .py file and run from the terminal:
# panel serve holoviz_dashboard.py --show
dashboard.servable()
