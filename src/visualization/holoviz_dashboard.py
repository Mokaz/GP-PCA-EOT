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
from src.visualization.plotly_offline_generator import generate_plotly_fig_for_frame, generate_initial_plotly_fig

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
# Add a placeholder to prompt user selection. The default value will be None.
file_selector = pn.widgets.Select(
    name='Select Simulation File', 
    options=[None] + pickle_files, 
    sizing_mode='stretch_width'
)

# Replace IntSlider with Player for playback controls
frame_player = pn.widgets.Player(name='Frame', start=0, end=0, step=1, visible=False, loop_policy='loop', interval=50, sizing_mode='stretch_width')
nees_states_checklist = pn.widgets.CheckBoxGroup(name='NEES States', options=[], visible=False, sizing_mode='stretch_width')


# --- 4. Define Interactive Plotting Functions ---
@pn.depends(file_selector.param.value, watch=True)
def update_widgets(filename):
    loaded_data = load_data(filename)
    if not loaded_data:
        frame_player.visible = False
        nees_states_checklist.visible = False
        return

    sim_result = loaded_data["sim_result"]
    # The player now represents all states, including the initial one (frame 0)
    frame_player.end = sim_result.config.sim.num_frames
    frame_player.value = 0 # Reset player to start
    frame_player.visible = True
    
    state_names = ['x', 'y', 'yaw', 'vel_x', 'vel_y', 'yaw_rate', 'length', 'width', 'pca_coeffs']
    nees_states_checklist.options = state_names
    nees_states_checklist.value = [] # Reset checklist
    nees_states_checklist.visible = True


@pn.depends(frame_player.param.value, file_selector.param.value)
def get_plotly_view(frame_idx, filename):
    loaded_data = load_data(filename)
    if not loaded_data: return pn.pane.Markdown("### Select a file to begin.")
    
    # Unpack data needed for the plot
    sim_result = loaded_data["sim_result"]
    config = loaded_data["config"]
    pca_params = loaded_data["pca_params"]
    ground_truth_states = list(sim_result.ground_truth_ts.values)

    fig = go.Figure()

    if frame_idx == 0:
        # --- Initial State (Frame 0) ---
        gt_state = ground_truth_states[0]
        est_state = sim_result.tracker_results_ts.values[0].state_prior.mean
        fig = generate_initial_plotly_fig(gt_state, est_state, config, pca_params)

    else:
        # --- Regular Update Step (Frames 1 and onwards) ---
        tracker_frame_idx = frame_idx - 1 
        gt_state = ground_truth_states[frame_idx]
        tracker_result = sim_result.tracker_results_ts.values[tracker_frame_idx]
        
        fig = generate_plotly_fig_for_frame(
            frame_idx=frame_idx,
            gt_state=gt_state,
            est_state=tracker_result.state_posterior.mean,
            z_lidar_cart=tracker_result.measurements.reshape((-1, 2)),
            ground_truth_history=ground_truth_states[:frame_idx + 1],
            config=config,
            pca_params=pca_params
        )

    # Common layout for all frames
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
        frame_player,
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
