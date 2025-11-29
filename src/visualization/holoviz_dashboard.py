# holoviz_dashboard.py
import panel as pn
import plotly.graph_objects as go
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import hvplot.pandas
import pandas as pd

from bokeh.util.serialization import make_globally_unique_id

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_ROOT))
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

# now import everything else that may be referenced by the pickle:
from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result
from src.global_project_paths import SIMDATA_PATH
from src.senfuslib.plotting import interactive_show_consistency, show_consistency
from src.visualization.plotly_offline_generator import (
    generate_plotly_fig_for_frame,
    generate_initial_plotly_fig,
)
from src.states.states import State_PCA, State_GP 

js_files = {
    'jquery': 'https://code.jquery.com/jquery-1.11.1.min.js',
    'goldenlayout': 'https://golden-layout.com/files/latest/js/goldenlayout.min.js',
}
css_files = [
    'https://golden-layout.com/files/latest/css/goldenlayout-base.css',
    'https://golden-layout.com/files/latest/css/goldenlayout-light-theme.css',
]
pn.extension('plotly', 'tabulator', js_files=js_files, css_files=css_files)

# --- Data loading and caching ---
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

# --- Widgets ---
pickle_files = sorted([f.name for f in Path(SIMDATA_PATH).glob("*.pkl")], reverse=True)

file_selector = pn.widgets.Select(
    name='Select Simulation File', 
    options=[None] + pickle_files, 
    sizing_mode='stretch_width'
)

frame_player = pn.widgets.Player(
    name='Frame', start=0, end=0, step=1,
    visible=False, loop_policy='loop', interval=50,
    sizing_mode='stretch_width'
)

# --- NEES Widgets ---
nees_group_selector = pn.widgets.MultiSelect(
    name='NEES Consistency Groups', 
    visible=False, 
    sizing_mode='stretch_width'
)

custom_nees_selector = pn.widgets.MultiSelect(
    name='Custom NEES States',
    visible=False,
    sizing_mode='stretch_width'
)

# --- Covariance Widgets ---
COV_MATRIX_MAPPING = {
    'Posterior Covariance (P_k|k)': 'state_posterior',
    'Prior Covariance (P_k|k-1)': 'state_prior',
    'Innovation Covariance (S_k)': 'predicted_measurement',
}
cov_matrix_selector = pn.widgets.Select(
    name='Covariance Matrix to Display',
    options=list(COV_MATRIX_MAPPING.keys()),
    visible=False,
    sizing_mode='stretch_width'
)

# --- NIS Widgets ---
nis_field_selector = pn.widgets.Select(
    name='NIS Consistency',
    options=['all'],
    value='all',
    visible=False,
    sizing_mode='stretch_width'
)


# --- Interactive functions ---
@pn.depends(file_selector.param.value, watch=True)
def update_widgets(filename):
    loaded_data = load_data(filename)
    if not loaded_data:
        frame_player.visible = False
        nees_group_selector.visible = False
        cov_matrix_selector.visible = False
        nis_field_selector.visible = False
        custom_nees_selector.visible = False
        return

    sim_result = loaded_data["sim_result"]
    config = sim_result.config
    
    # 1. Inspect the first posterior state to determine type
    first_state_posterior = sim_result.tracker_results_ts.values[0].state_posterior.mean

    # 2. Define Groups and Labels dynamically based on State Type
    if isinstance(first_state_posterior, State_GP):
        # --- GP Configuration ---
        n_radii = len(first_state_posterior.radii)
        
        # NEES Groups
        dynamic_nees_mapping = {
            'Total NEES': 'all',
            'Position (x, y)': ['x', 'y'],
            'Heading (yaw)': ['yaw'],
            'Velocity (vx, vy)': ['vel_x', 'vel_y'],
            'Yaw Rate': ['yaw_rate'],
            'Shape (All Radii)': ['radii'] # Uses the 'radii' slice from State_GP
        }
        
        # Individual State Labels (for Custom Selector)
        # We map display names to actual lookup keys (Strings for named fields, Ints for array indices)
        # State_GP structure: 0:x, 1:y, 2:yaw, 3:vx, 4:vy, 5:rate, 6+:radii
        custom_state_options = {
            'x': 'x', 'y': 'y', 'yaw': 'yaw', 
            'vel_x': 'vel_x', 'vel_y': 'vel_y', 'yaw_rate': 'yaw_rate'
        }
        # Add radii indices dynamically
        for i in range(n_radii):
            custom_state_options[f'radius_{i}'] = 6 + i

    elif isinstance(first_state_posterior, State_PCA):
        # --- PCA Configuration ---
        n_pca = config.tracker.N_pca
        
        # NEES Groups
        dynamic_nees_mapping = {
            'Total NEES': 'all',
            'Position (x, y)': ['x', 'y'],
            'Heading (yaw)': ['yaw'],
            'Velocity (vx, vy)': ['vel_x', 'vel_y'],
            'Extent (L, W)': ['length', 'width'],
            'PCA Components': ['pca_coeffs'] # Uses the 'pca_coeffs' slice
        }
        
        # Individual State Labels
        custom_state_options = {
            'x': 'x', 'y': 'y', 'yaw': 'yaw', 
            'vel_x': 'vel_x', 'vel_y': 'vel_y', 'yaw_rate': 'yaw_rate',
            'length': 'length', 'width': 'width'
        }
        # Add PCA indices dynamically (assuming they map to the slice start)
        # Note: If State_PCA has explicit fields 'pca_coeff_0', use strings, else use indices
        base_pca_idx = 8
        for i in range(n_pca):
            custom_state_options[f'pca_coeff_{i}'] = base_pca_idx + i

    else:
        # Fallback / Error safety
        dynamic_nees_mapping = {'Total NEES': 'all'}
        custom_state_options = {}

    # 3. Update Widgets
    frame_player.end = sim_result.config.sim.num_frames
    frame_player.value = 0
    frame_player.visible = True
    
    # Update NEES Group Selector
    nees_group_selector.options = dynamic_nees_mapping
    nees_group_selector.value = []
    nees_group_selector.visible = True
    nees_group_selector.size = len(dynamic_nees_mapping)
    nees_group_selector.height = min(len(dynamic_nees_mapping) * 20, 400)

    # Update Custom NEES Selector
    custom_nees_selector.options = custom_state_options
    custom_nees_selector.value = []
    custom_nees_selector.visible = True
    
    n_items = len(custom_state_options)
    custom_nees_selector.size = min(n_items, 15)
    custom_nees_selector.height = min(n_items * 18, 400) 

    cov_matrix_selector.visible = True
    nis_field_selector.visible = True

def create_empty_figure(message="Select a file to begin"):
    """Helper to create a blank figure with a centered message."""
    empty_fig = go.Figure()
    empty_fig.add_annotation(
        text=message, 
        xref="paper", yref="paper", 
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="gray")
    )
    empty_fig.update_layout(
        xaxis={'visible': False}, yaxis={'visible': False},
        plot_bgcolor='white', paper_bgcolor='white'
    )
    return empty_fig

# 1. Create the pane ONCE, initializing it with the empty figure
persistent_plotly_pane = pn.pane.Plotly(
    object=create_empty_figure(),  # <--- Initialize with text
    sizing_mode='stretch_both', 
    config={'responsive': True}
)

# 2. Define the update logic
def update_plotly_view(frame_idx, filename):
    loaded_data = load_data(filename)
    
    # Handle "No Data" without destroying the Plotly pane
    if not loaded_data:
        # Use the helper to reset the view to the message
        persistent_plotly_pane.object = create_empty_figure()
        return
    
    sim_result = loaded_data["sim_result"]
    config = loaded_data["config"]
    pca_params = loaded_data["pca_params"]
    ground_truth_states = list(sim_result.ground_truth_ts.values)
    tracker_results = list(sim_result.tracker_results_ts.values)

    # Safety check for frame index
    if frame_idx >= len(ground_truth_states):
        return

    gt_state = ground_truth_states[frame_idx]
    tracker_result = tracker_results[frame_idx]
    est_state = tracker_result.state_posterior.mean

    if frame_idx == 0:
        fig = generate_initial_plotly_fig(gt_state, est_state, config, pca_params)
    else:
        fig = generate_plotly_fig_for_frame(
            frame_idx=frame_idx,
            gt_state=gt_state,
            est_state=tracker_result.state_posterior.mean,
            z_lidar_cart=tracker_result.measurements.reshape((-1, 2)),
            ground_truth_history=ground_truth_states[:frame_idx],
            config=config,
            pca_params=pca_params
        )

    fig.update_layout(
        autosize=True,
        margin=dict(l=40, r=40, b=40, t=40, pad=4),
        title=f"Frame: {frame_idx}", plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(range=[-60, 60], constrain='domain',
                   gridcolor='rgb(200, 200, 200)', zerolinecolor='rgb(200, 200, 200)',
                   title='East [m]'),
        yaxis=dict(range=[-15, 35], scaleanchor="x", scaleratio=1,
                   gridcolor='rgb(200, 200, 200)', zerolinecolor='rgb(200, 200, 200)',
                   title='North [m]'),
        legend=dict(x=1.05, y=1)
    )
    
    # Update the object in-place
    persistent_plotly_pane.object = fig

# 3. Bind the update function to the widgets using watch=True
pn.bind(update_plotly_view, frame_player, file_selector, watch=True)

@pn.depends(nees_group_selector.param.value, custom_nees_selector.param.value, file_selector.param.value)
def get_nees_view(selected_groups, custom_states, filename):
    loaded_data = load_data(filename)
    if not loaded_data or (not selected_groups and not custom_states):
        return pn.pane.Markdown("### Select pre-defined groups or custom states to show NEES plot.")
    
    # selected_groups already contains the values (e.g. ['x', 'y'] or 'all') from the dict
    fields_to_plot = []
    fields_to_plot.extend(selected_groups)
    
    if custom_states:
        fields_to_plot.append(list(custom_states))
        
    consistency_analyzer = loaded_data["consistency_analyzer"]
    bokeh_plot = interactive_show_consistency(
        analysis=consistency_analyzer, 
        fields_nees=fields_to_plot
    )

    if bokeh_plot is None:
      return pn.pane.Markdown("### No data to plot for the selected fields.")
      
    return pn.pane.Bokeh(bokeh_plot, sizing_mode='stretch_both')


# src/visualization/holoviz_dashboard.py

# ... existing code ...

@pn.depends(cov_matrix_selector.param.value, frame_player.param.value, file_selector.param.value)
def get_covariance_view(matrix_name, frame_idx, filename):
    loaded_data = load_data(filename)
    if not loaded_data:
        return pn.pane.Markdown("### Select a file to begin.")

    # Get the specific result for this frame
    tracker_result = loaded_data["sim_result"].tracker_results_ts.values[frame_idx]
    
    attr_name = COV_MATRIX_MAPPING[matrix_name]
    gauss_obj = getattr(tracker_result, attr_name)
    
    # Handle optional None values (e.g. initial frame prior)
    if gauss_obj is None:
        return pn.pane.Markdown(f"### {matrix_name} not available for Frame {frame_idx}")

    cov_matrix = gauss_obj.cov

    if 'state' in attr_name:
        # Check the type of the state estimate to determine labels
        # We look at the posterior mean to decide the structure
        state_mean = tracker_result.state_posterior.mean
        
        if isinstance(state_mean, State_GP):
            # --- GP Labels ---
            n_radii = len(state_mean.radii)
            labels = ['x', 'y', 'yaw', 'vel_x', 'vel_y', 'yaw_rate'] + [f'radius_{i}' for i in range(n_radii)]
            
        elif isinstance(state_mean, State_PCA):
            # --- PCA Labels ---
            n_pca = loaded_data["config"].tracker.N_pca
            labels = ['x', 'y', 'yaw', 'vel_x', 'vel_y', 'yaw_rate', 'length', 'width'] + [f'pca_{i}' for i in range(n_pca)]
            
        else:
            # Fallback based on size if type check fails or is generic
            N = cov_matrix.shape[0]
            labels = [f'state_{i}' for i in range(N)]
            
    else: 
        # Measurement Covariance (S_k)
        num_rays = cov_matrix.shape[0] // 2
        labels = [f'x{i}' for i in range(num_rays)] + [f'y{i}' for i in range(num_rays)]

    # Safety check to prevent crashing if shapes still don't match due to some other edge case
    if len(labels) != cov_matrix.shape[0]:
        return pn.pane.Markdown(f"### Dimension Mismatch: Covariance is {cov_matrix.shape}, but generated {len(labels)} labels.")

    df = pd.DataFrame(cov_matrix, index=labels, columns=labels)
    
    # Check condition number to warn about numerical instability
    try:
        cond_number = np.linalg.cond(cov_matrix)
        title_text = f"{matrix_name} at Frame {frame_idx} (cond={cond_number:.2e})"
    except np.linalg.LinAlgError:
        title_text = f"{matrix_name} at Frame {frame_idx} (Singular Matrix)"

    heatmap = df.hvplot.heatmap(
        cmap='viridis',
        rot=90,
        title=title_text
    ).opts(responsive=True, xrotation=90, invert_yaxis=True)
    
    return pn.pane.HoloViews(heatmap, sizing_mode="stretch_both")


@pn.depends(nis_field_selector.param.value, file_selector.param.value)
def get_nis_view(selected_field, filename):
    loaded_data = load_data(filename)
    if not loaded_data:
        return pn.pane.Markdown("### Select a file to begin.")
    
    consistency_analyzer = loaded_data["consistency_analyzer"]
    
    fields_to_plot = ["all"]
    
    bokeh_plot = interactive_show_consistency(
        analysis=consistency_analyzer, 
        fields_nis=fields_to_plot,
        title="Measurement Consistency"
    )

    if bokeh_plot is None:
      return pn.pane.Markdown("### No NIS data available.")
      
    return pn.pane.Bokeh(bokeh_plot, sizing_mode='stretch_both')


# --- Build Panel objects ---
controls = pn.Column(
    pn.pane.Markdown("## Controls"),
    file_selector,
    frame_player,
    nees_group_selector,
    custom_nees_selector,
    cov_matrix_selector,
    nis_field_selector,
    sizing_mode="stretch_width",
)

plotly_view = pn.Column(persistent_plotly_pane, sizing_mode="stretch_both")
nees_view = pn.Column(get_nees_view, sizing_mode="stretch_both")
covariance_view = pn.Column(get_covariance_view, sizing_mode="stretch_both")
nis_view = pn.Column(get_nis_view, sizing_mode="stretch_both")


# --- Custom GoldenLayout Template ---
template_file = Path(__file__).parent / 'golden_template.html'

with open(template_file, 'r') as f:
    template_str = f.read()

tmpl = pn.Template(template_str)

tmpl.nb_template.globals['get_id'] = make_globally_unique_id

tmpl.add_panel('controls', controls)
tmpl.add_panel('plotly_view', plotly_view)
tmpl.add_panel('nees_view', nees_view)
tmpl.add_panel('covariance_view', covariance_view)
tmpl.add_panel('nis_view', nis_view)
tmpl.servable(title="GP-PCA-EOT Simulation Analysis Dashboard")

if __name__ == "__main__":
    pn.serve(tmpl, port=5006, show=True)