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
NEES_GROUP_MAPPING = {
    'Total NEES': 'all',
    'Position (x, y)': ['x', 'y'],
    'Heading (yaw)': ['yaw'],
    'Velocity (vx, vy)': ['vel_x', 'vel_y'],
    'Extent (l, w)': ['length', 'width'],
    'PCA Shape': ['pca_coeffs'],
}
nees_group_selector = pn.widgets.MultiSelect(
    name='NEES Consistency Groups', 
    options=list(NEES_GROUP_MAPPING.keys()), 
    visible=False, 
    size=len(NEES_GROUP_MAPPING),
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
    options=['Lidar'], # Assuming 'Lidar' is the only sensor for now
    value='Lidar',
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

    # Dynamically get the list of state labels
    # n_pca = loaded_data["config"].tracker.N_pca
    all_state_labels = ['x', 'y', 'yaw', 'vel_x', 'vel_y', 'yaw_rate', 'length', 'width'] + [f'pca_coeff_{i}' for i in range(sim_result.config.tracker.N_pca)]

    frame_player.end = sim_result.config.sim.num_frames
    frame_player.value = 0
    frame_player.visible = True
    
    nees_group_selector.value = []
    nees_group_selector.visible = True

    custom_nees_selector.options = all_state_labels
    custom_nees_selector.value = []
    custom_nees_selector.visible = True
    custom_nees_selector.size = len(all_state_labels)

    cov_matrix_selector.visible = True
    nis_field_selector.visible = True


@pn.depends(frame_player.param.value, file_selector.param.value)
def get_plotly_view(frame_idx, filename):
    loaded_data = load_data(filename)
    if not loaded_data:
        return pn.pane.Markdown("### Select a file to begin.")
    
    sim_result = loaded_data["sim_result"]
    config = loaded_data["config"]
    pca_params = loaded_data["pca_params"]
    ground_truth_states = list(sim_result.ground_truth_ts.values)
    tracker_results = list(sim_result.tracker_results_ts.values)

    gt_state = ground_truth_states[frame_idx]
    tracker_result = tracker_results[frame_idx]

    est_state = tracker_result.state_posterior.mean

    fig = go.Figure()

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
    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_both')


@pn.depends(nees_group_selector.param.value, custom_nees_selector.param.value, file_selector.param.value)
def get_nees_view(selected_groups, custom_states, filename):
    loaded_data = load_data(filename)
    if not loaded_data or (not selected_groups and not custom_states):
        return pn.pane.Markdown("### Select pre-defined groups or custom states to show NEES plot.")
    
    # Start with the fields from the pre-defined groups
    fields_to_plot = [NEES_GROUP_MAPPING[name] for name in selected_groups]
    
    # If the user selected any custom states, add them as a new, separate plot
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



@pn.depends(cov_matrix_selector.param.value, frame_player.param.value, file_selector.param.value)
def get_covariance_view(matrix_name, frame_idx, filename):
    loaded_data = load_data(filename)
    if not loaded_data:
        return pn.pane.Markdown("### Select a file to begin.")

    tracker_result = loaded_data["sim_result"].tracker_results_ts.values[frame_idx]
    
    attr_name = COV_MATRIX_MAPPING[matrix_name]
    gauss_obj = getattr(tracker_result, attr_name)
    cov_matrix = gauss_obj.cov

    if 'state' in attr_name:
        n_pca = loaded_data["config"].tracker.N_pca
        labels = ['x', 'y', 'yaw', 'vel_x', 'vel_y', 'yaw_rate', 'length', 'width'] + [f'pca_{i}' for i in range(n_pca)]
    else: 
        num_rays = cov_matrix.shape[0] // 2
        labels = [f'x{i}' for i in range(num_rays)] + [f'y{i}' for i in range(num_rays)]

    df = pd.DataFrame(cov_matrix, index=labels, columns=labels)
    

    cond_number = np.linalg.cond(cov_matrix)
    heatmap = df.hvplot.heatmap(
        cmap='viridis',
        rot=90,
        title=f"{matrix_name} at Frame {frame_idx} (cond={cond_number:.2e})"
    ).opts(responsive=True, xrotation=90, invert_yaxis=True)
    
    return pn.pane.HoloViews(heatmap, sizing_mode="stretch_both")


@pn.depends(nis_field_selector.param.value, file_selector.param.value)
def get_nis_view(selected_field, filename):
    return pn.pane.Markdown("### NIS View not yet implemented.")
    # loaded_data = load_data(filename)
    # if not loaded_data or not selected_field:
    #     return pn.pane.Markdown("### Select a file to begin.")

    # consistency_analyzer = loaded_data["consistency_analyzer"]
    
    # # For NIS, we plot one field at a time.
    # fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    
    # # Ensure 'ax' is treated as an iterable list for the plotting function
    # axs = np.atleast_1d(ax)
    
    # show_consistency(analysis=consistency_analyzer, fields_nis=[selected_field], axs_nis=axs)
    # plt.tight_layout()
    # return pn.pane.Matplotlib(fig, tight=True, sizing_mode='stretch_width')


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

plotly_view = pn.Column(get_plotly_view, sizing_mode="stretch_both")
nees_view = pn.Column(get_nees_view, sizing_mode="stretch_both")
covariance_view = pn.Column(get_covariance_view, sizing_mode="stretch_both")
nis_view = pn.Column(get_nis_view, sizing_mode="stretch_both")


# --- Custom GoldenLayout Template ---
template_file = Path(__file__).parent / 'golden_template.html'

with open(template_file, 'r') as f:
    template_str = f.read()

tmpl = pn.Template(template_str)

tmpl.nb_template.globals['get_id'] = make_globally_unique_id

# Add panels to the template with the names used in the 'embed' calls
tmpl.add_panel('controls', controls)
tmpl.add_panel('plotly_view', plotly_view)
tmpl.add_panel('nees_view', nees_view)
tmpl.add_panel('covariance_view', covariance_view)
tmpl.add_panel('nis_view', nis_view)
tmpl.servable(title="GP-PCA-EOT Simulation Analysis Dashboard")