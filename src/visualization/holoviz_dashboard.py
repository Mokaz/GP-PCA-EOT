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

from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result
from src.global_project_paths import SIMDATA_PATH
from src.senfuslib.plotting import (
    interactive_show_consistency, 
    show_consistency, 
    interactive_show_error,
    matplotlib_show_consistency,
    matplotlib_show_error
)
from src.visualization.plotly_offline_generator import (
    generate_plotly_fig_for_frame,
    generate_initial_plotly_fig,
)
from src.states.states import State_PCA, State_GP 

ASSETS_DIR = Path(__file__).parent / 'assets'

js_files = {
    'jquery': 'assets/jquery-1.11.1.min.js',
    'goldenlayout': 'assets/goldenlayout.min.js',
}
css_files = [
    'assets/goldenlayout-base.css',
    'assets/goldenlayout-light-theme.css',
]
pn.extension('plotly', 'tabulator', js_files=js_files, css_files=css_files)

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

frame_input = pn.widgets.IntInput(
    name='Jump to Frame', value=0, start=0, step=1,
    visible=False, sizing_mode='stretch_width'
)

# Link player and input
frame_player.link(frame_input, value='value')
frame_input.link(frame_player, value='value')

# --- NEES Widgets ---
nees_group_selector = pn.widgets.MultiSelect(
    name='NEES Consistency Groups', 
    visible=False, 
    sizing_mode='stretch_width'
)

# --- Error Widgets ---
error_group_selector = pn.widgets.MultiSelect(
    name='Error Groups', 
    visible=False, 
    sizing_mode='stretch_width'
)

custom_states_selector = pn.widgets.MultiSelect(
    name='Custom States for NEES/Error',
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

# --- Plotting Control Widgets ---
plotting_divider = pn.layout.Divider(visible=False)
plotting_header = pn.pane.Markdown("### Plotting & Saving", visible=False)

plot_backend_selector = pn.widgets.Select(
    name='Plotting Backend', 
    options=['Bokeh', 'Matplotlib'], 
    value='Bokeh',
    sizing_mode='stretch_width',
    visible=False
)

save_filename_input = pn.widgets.TextInput(
    name='Save Filename (no ext)', 
    value='plot_output',
    placeholder='Enter filename...',
    sizing_mode='stretch_width',
    visible=False
)

save_button = pn.widgets.Button(
    name='Save Matplotlib Plots', 
    button_type='primary',
    sizing_mode='stretch_width',
    visible=False
)

save_status = pn.pane.Markdown("", sizing_mode='stretch_width', visible=False)

# --- Data Browser Widgets ---
data_browser_mode = pn.widgets.Select(
    name='Data Browser Mode',
    options=[
        'Current Frame Tracker Result', 
        'Current Frame GT', 
        'Current Frame State Error', 
        'Current Frame Measurement Error',
        'Consistency Analysis (Summary)',
        'Config', 
        'Full Simulation Result (Summary)'
    ],
    value='Current Frame Tracker Result',
    sizing_mode='stretch_width',
    visible=False
)

# --- Interactive functions ---
@pn.depends(file_selector.param.value, watch=True)
def update_widgets(filename):
    loaded_data = load_data(filename)
    if not loaded_data:
        frame_player.visible = False
        frame_input.visible = False
        nees_group_selector.visible = False
        error_group_selector.visible = False
        cov_matrix_selector.visible = False
        nis_field_selector.visible = False
        custom_states_selector.visible = False
        
        data_browser_mode.visible = False
        plotting_divider.visible = False
        plotting_header.visible = False
        plot_backend_selector.visible = False
        save_filename_input.visible = False
        save_button.visible = False
        save_status.visible = False
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

    # Create a copy for error mapping and remove 'Total NEES' if present
    error_mapping = dynamic_nees_mapping.copy()
    if 'Total NEES' in error_mapping:
        del error_mapping['Total NEES']

    # 3. Update Widgets
    frame_player.end = sim_result.config.sim.num_frames
    frame_player.value = 0
    frame_player.visible = True
    
    frame_input.end = sim_result.config.sim.num_frames
    frame_input.value = 0
    frame_input.visible = True
    
    # Update NEES Group Selector
    nees_group_selector.options = dynamic_nees_mapping
    nees_group_selector.value = []
    nees_group_selector.visible = True
    nees_group_selector.size = len(dynamic_nees_mapping)
    nees_group_selector.height = min(len(dynamic_nees_mapping) * 20, 400)

    # Update Error Group Selector
    error_group_selector.options = error_mapping
    error_group_selector.value = []
    error_group_selector.visible = True
    error_group_selector.size = len(error_mapping)
    error_group_selector.height = min(len(error_mapping) * 20, 400)

    # Update Custom NEES Selector
    custom_states_selector.options = custom_state_options
    custom_states_selector.value = []

    # Visbility
    cov_matrix_selector.visible = True
    nis_field_selector.visible = True

    data_browser_mode.visible = True
    plotting_divider.visible = True
    plotting_header.visible = True
    plot_backend_selector.visible = True
    save_filename_input.visible = True
    save_button.visible = True
    save_status.visible = True
    custom_states_selector.visible = True
    
    n_items = len(custom_state_options)
    custom_states_selector.size = min(n_items, 15)
    custom_states_selector.height = min(n_items * 18, 400) 

    cov_matrix_selector.visible = True
    nis_field_selector.visible = True

def serialize_state_object(state_obj):
    """Helper to convert State_PCA/State_GP objects into a labeled dictionary."""
    cls_name = state_obj.__class__.__name__
    
    # Prioritize attribute detection over class name to handle potential mismatches
    # Check for GP characteristics (radii)
    if hasattr(state_obj, 'radii'):
        return {
            "__type__": cls_name,
            "x": float(state_obj.x),
            "y": float(state_obj.y),
            "yaw": float(state_obj.yaw),
            "vel_x": float(state_obj.vel_x),
            "vel_y": float(state_obj.vel_y),
            "yaw_rate": float(state_obj.yaw_rate),
            "radii": state_obj.radii.tolist()
        }
        
    # Check for PCA characteristics (pca_coeffs)
    if hasattr(state_obj, 'pca_coeffs'):
        return {
            "__type__": cls_name,
            "x": float(state_obj.x),
            "y": float(state_obj.y),
            "yaw": float(state_obj.yaw),
            "vel_x": float(state_obj.vel_x),
            "vel_y": float(state_obj.vel_y),
            "yaw_rate": float(state_obj.yaw_rate),
            "length": float(state_obj.length),
            "width": float(state_obj.width),
            "pca_coeffs": state_obj.pca_coeffs.tolist()
        }
    
    return None

def safe_serialize(obj, max_depth=3, current_depth=0):
    """Recursively converts objects to JSON-serializable dicts/lists."""
    if current_depth > max_depth:
        return str(obj)
    
    if obj is None:
        return None
    
    # Check for State objects first
    # We try to serialize if it has state-like attributes or matching class name
    if hasattr(obj, 'radii') or hasattr(obj, 'pca_coeffs') or obj.__class__.__name__ in ['State_PCA', 'State_GP']:
        res = serialize_state_object(obj)
        if res is not None:
            return res
        
    if isinstance(obj, (str, int, float, bool)):
        return obj
        
    if isinstance(obj, (np.integer, int)):
        return int(obj)
        
    if isinstance(obj, (np.floating, float)):
        return float(obj)
        
    if isinstance(obj, np.ndarray):
        if obj.size <= 100:
            return obj.tolist()
        return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
    
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(x, max_depth, current_depth + 1) for x in obj]
    
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v, max_depth, current_depth + 1) for k, v in obj.items()}
        
    # Handle Dataclasses
    if hasattr(obj, '__dataclass_fields__'):
        return {k: safe_serialize(getattr(obj, k), max_depth, current_depth + 1) for k in obj.__dataclass_fields__}
        
    # Handle generic objects
    if hasattr(obj, '__dict__'):
        return {k: safe_serialize(v, max_depth, current_depth + 1) for k, v in obj.__dict__.items() if not k.startswith('_')}
        
    return str(obj)

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

persistent_plotly_pane = pn.pane.Plotly(
    object=create_empty_figure(),
    sizing_mode='stretch_both', 
    config={'responsive': True}
)

def update_plotly_view(frame_idx, filename):
    loaded_data = load_data(filename)

    if not loaded_data:
        persistent_plotly_pane.object = create_empty_figure()
        return
    
    sim_result = loaded_data["sim_result"]
    config = loaded_data["config"]
    pca_params = loaded_data["pca_params"]
    ground_truth_states = list(sim_result.ground_truth_ts.values)
    tracker_results = list(sim_result.tracker_results_ts.values)

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
    
    persistent_plotly_pane.object = fig

pn.bind(update_plotly_view, frame_player, file_selector, watch=True)

@pn.depends(nees_group_selector.param.value, custom_states_selector.param.value, file_selector.param.value, plot_backend_selector.param.value)
def get_nees_view(selected_groups, custom_states, filename, backend):
    loaded_data = load_data(filename)
    if not loaded_data or (not selected_groups and not custom_states):
        return pn.pane.Markdown("### Select pre-defined groups or custom states to show NEES plot.")
    
    # selected_groups already contains the values (e.g. ['x', 'y'] or 'all') from the dict
    fields_to_plot = []
    fields_to_plot.extend(selected_groups)
    
    if custom_states:
        fields_to_plot.append(list(custom_states))
        
    consistency_analyzer = loaded_data["consistency_analyzer"]
    
    if backend == 'Matplotlib':
        fig = matplotlib_show_consistency(
            analysis=consistency_analyzer, 
            fields_nees=fields_to_plot
        )
        if fig is None:
             return pn.pane.Markdown("### No data to plot for the selected fields.")
        return pn.pane.Matplotlib(fig, sizing_mode='stretch_both')
    else:
        bokeh_plot = interactive_show_consistency(
            analysis=consistency_analyzer, 
            fields_nees=fields_to_plot
        )

        if bokeh_plot is None:
            return pn.pane.Markdown("### No data to plot for the selected fields.")
        
        return pn.pane.Bokeh(bokeh_plot, sizing_mode='stretch_both')

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


@pn.depends(nis_field_selector.param.value, file_selector.param.value, plot_backend_selector.param.value)
def get_nis_view(selected_field, filename, backend):
    loaded_data = load_data(filename)
    if not loaded_data:
        return pn.pane.Markdown("### Select a file to begin.")
    
    consistency_analyzer = loaded_data["consistency_analyzer"]
    
    fields_to_plot = ["all"]
    
    if backend == 'Matplotlib':
        fig = matplotlib_show_consistency(
            analysis=consistency_analyzer, 
            fields_nis=fields_to_plot
        )
        if fig is None:
             return pn.pane.Markdown("### No NIS data available.")
        return pn.pane.Matplotlib(fig, sizing_mode='stretch_both')
    else:
        bokeh_plot = interactive_show_consistency(
            analysis=consistency_analyzer, 
            fields_nis=fields_to_plot,
        )

        if bokeh_plot is None:
            return pn.pane.Markdown("### No NIS data available.")
        
        return pn.pane.Bokeh(bokeh_plot, sizing_mode='stretch_both')


@pn.depends(error_group_selector.param.value, custom_states_selector.param.value, file_selector.param.value, plot_backend_selector.param.value)
def get_error_view(selected_groups, custom_states, filename, backend):
    loaded_data = load_data(filename)
    if not loaded_data or (not selected_groups and not custom_states):
        return pn.pane.Markdown("### Select pre-defined groups or custom states to show Error plot.")
    
    # Flatten the selected groups into a single list of fields
    fields_to_plot = []
    
    # Handle group selections (which might be lists of fields)
    for group in selected_groups:
        if isinstance(group, list):
            fields_to_plot.extend(group)
        else:
            fields_to_plot.append(group)
    
    # Handle custom state selections
    if custom_states:
        fields_to_plot.extend(custom_states)
        
    consistency_analyzer = loaded_data["consistency_analyzer"]
    
    if backend == 'Matplotlib':
        fig = matplotlib_show_error(
            analysis=consistency_analyzer, 
            fields_err=fields_to_plot
        )
        if fig is None:
             return pn.pane.Markdown("### No data to plot for the selected fields.")
        return pn.pane.Matplotlib(fig, sizing_mode='stretch_both')
    else:
        bokeh_plot = interactive_show_error(
            analysis=consistency_analyzer, 
            fields_err=fields_to_plot
        )

        if bokeh_plot is None:
            return pn.pane.Markdown("### No data to plot for the selected fields.")
        
        return pn.pane.Bokeh(bokeh_plot, sizing_mode='stretch_both')


@pn.depends(data_browser_mode.param.value, frame_player.param.value, file_selector.param.value)
def get_data_browser_view(mode, frame_idx, filename):
    loaded_data = load_data(filename)
    if not loaded_data:
        return pn.pane.Markdown("### Select a file to begin.")
    
    sim_result = loaded_data["sim_result"]
    consistency_analyzer = loaded_data["consistency_analyzer"]
    
    data_to_show = {}
    
    if mode == 'Current Frame Tracker Result':
        if frame_idx < len(sim_result.tracker_results_ts.values):
            res = sim_result.tracker_results_ts.values[frame_idx]
            data_to_show = safe_serialize(res, max_depth=4)
        else:
            data_to_show = {"info": "Frame index out of range"}
            
    elif mode == 'Current Frame GT':
        if frame_idx < len(sim_result.ground_truth_ts.values):
            res = sim_result.ground_truth_ts.values[frame_idx]
            data_to_show = safe_serialize(res, max_depth=4)
        else:
            data_to_show = {"info": "Frame index out of range"}

    elif mode == 'Current Frame State Error':
        if hasattr(consistency_analyzer, 'x_err_gauss') and frame_idx < len(consistency_analyzer.x_err_gauss.values):
            err_gauss = consistency_analyzer.x_err_gauss.values[frame_idx]
            
            # Try to recover labels for the error vector
            err_mean = err_gauss.mean
            # We assume the error has the same structure as the posterior mean of the first frame
            if len(sim_result.tracker_results_ts.values) > 0:
                ref_state = sim_result.tracker_results_ts.values[0].state_posterior.mean
                
                if not isinstance(err_mean, (State_PCA, State_GP)) and isinstance(err_mean, np.ndarray):
                    try:
                        # Attempt to view as the reference state class
                        err_mean = err_mean.view(ref_state.__class__)
                    except Exception:
                        pass # Fallback to array

            data_to_show = {
                "error_mean": safe_serialize(err_mean, max_depth=4),
                "covariance_diag": safe_serialize(np.diag(err_gauss.cov), max_depth=4)
            }
        else:
            data_to_show = {"info": "Frame index out of range or no state error data."}

    elif mode == 'Current Frame Measurement Error':
        if hasattr(consistency_analyzer, 'z_err_gauss') and frame_idx < len(consistency_analyzer.z_err_gauss.values):
            err_gauss = consistency_analyzer.z_err_gauss.values[frame_idx]
            data_to_show = {
                "error_mean": safe_serialize(err_gauss.mean, max_depth=4),
                "covariance_diag": safe_serialize(np.diag(err_gauss.cov), max_depth=4)
            }
        else:
            data_to_show = {"info": "Frame index out of range or no measurement error data."}

    elif mode == 'Consistency Analysis (Summary)':
        data_to_show = {
            "num_frames_analyzed": len(consistency_analyzer.x_err_gauss) if hasattr(consistency_analyzer, 'x_err_gauss') else 0,
            "has_ground_truth": consistency_analyzer.x_gts is not None,
            "average_nees": safe_serialize(consistency_analyzer.get_nees(indices='all').a) if consistency_analyzer.x_gts else "N/A",
            "average_nis": safe_serialize(consistency_analyzer.get_nis(indices='all').a),
        }
            
    elif mode == 'Config':
        data_to_show = safe_serialize(sim_result.config, max_depth=5)
        
    elif mode == 'Full Simulation Result (Summary)':
        # Custom summary for the huge object
        data_to_show = {
            "config": "See Config mode",
            "num_frames": len(sim_result.tracker_results_ts.values),
            "static_covariances": safe_serialize(sim_result.static_covariances),
            "tracker_results_ts": f"TimeSequence with {len(sim_result.tracker_results_ts.values)} items",
            "ground_truth_ts": f"TimeSequence with {len(sim_result.ground_truth_ts.values)} items",
        }

    return pn.pane.JSON(data_to_show, sizing_mode='stretch_both', depth=5, theme='light')


def save_plots(event):
    filename = save_filename_input.value
    if not filename:
        save_status.object = "Please enter a filename."
        return
        
    loaded_data = load_data(file_selector.value)
    if not loaded_data:
        save_status.object = "No data loaded."
        return
        
    consistency_analyzer = loaded_data["consistency_analyzer"]
    saved_files = []
    
    # Save NEES
    nees_fields = []
    nees_fields.extend(nees_group_selector.value)
    if custom_states_selector.value:
        nees_fields.append(list(custom_states_selector.value))
        
    if nees_fields:
        fig = matplotlib_show_consistency(consistency_analyzer, fields_nees=nees_fields)
        if fig:
            path = Path(PROJECT_ROOT) / 'figures' / f'{filename}_nees.pdf'
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            saved_files.append(path.name)
            plt.close(fig)

    # Save Error
    error_fields = []
    for group in error_group_selector.value:
        if isinstance(group, list):
            error_fields.extend(group)
        else:
            error_fields.append(group)
    if custom_states_selector.value:
        error_fields.extend(custom_states_selector.value)
        
    if error_fields:
        fig = matplotlib_show_error(consistency_analyzer, fields_err=error_fields)
        if fig:
            path = Path(PROJECT_ROOT) / 'figures' / f'{filename}_error.pdf'
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            saved_files.append(path.name)
            plt.close(fig)

    # Save NIS
    if nis_field_selector.value:
        fig = matplotlib_show_consistency(consistency_analyzer, fields_nis=['all'])
        if fig:
            path = Path(PROJECT_ROOT) / 'figures' / f'{filename}_nis.pdf'
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            saved_files.append(path.name)
            plt.close(fig)
            
    if saved_files:
        save_status.object = f"Saved: {', '.join(saved_files)} in figures/"
    cov_matrix_selector,
    nis_field_selector,
    plotting_divider,
    plotting_header,
    plot_backend_selector,

# --- Build Panel objects ---
controls = pn.Column(
    pn.pane.Markdown("## Controls"),
    file_selector,
    frame_player,
    frame_input,
    data_browser_mode,
    nees_group_selector,
    error_group_selector,
    custom_states_selector,
    cov_matrix_selector,
    nis_field_selector,
    pn.layout.Divider(),
    pn.pane.Markdown("### Plotting & Saving"),
    plot_backend_selector,
    save_filename_input,
    save_button,
    save_status,
    sizing_mode="stretch_width",
)

plotly_view = pn.Column(persistent_plotly_pane, sizing_mode="stretch_both")
nees_view = pn.Column(get_nees_view, sizing_mode="stretch_both")
error_view = pn.Column(get_error_view, sizing_mode="stretch_both")
covariance_view = pn.Column(get_covariance_view, sizing_mode="stretch_both")
nis_view = pn.Column(get_nis_view, sizing_mode="stretch_both")
data_browser_view = pn.Column(get_data_browser_view, sizing_mode="stretch_both")

# --- Custom GoldenLayout Template ---
template_file = Path(__file__).parent / 'golden_template.html'

with open(template_file, 'r') as f:
    template_str = f.read()

tmpl = pn.Template(template_str)

tmpl.nb_template.globals['get_id'] = make_globally_unique_id

tmpl.add_panel('controls', controls)
tmpl.add_panel('plotly_view', plotly_view)
tmpl.add_panel('nees_view', nees_view)
tmpl.add_panel('error_view', error_view)
tmpl.add_panel('covariance_view', covariance_view)
tmpl.add_panel('nis_view', nis_view)
tmpl.add_panel('data_browser_view', data_browser_view)
tmpl.servable(title="GP-PCA-EOT Simulation Analysis Dashboard")

if __name__ == "__main__":
    pn.serve(tmpl, port=5006, show=True, static_dirs={'assets': str(ASSETS_DIR)})