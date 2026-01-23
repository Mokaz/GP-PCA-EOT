import sys
import numpy as np
import os
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

from pathlib import Path

# Go up TWO levels to the 'src' directory
SRC_ROOT = Path(__file__).resolve().parent.parent
# Add the 'src' directory to the path to match the pickling environment
sys.path.append(str(SRC_ROOT))

# Now that 'src' is on the path, these imports will fail.
# We must add the project root for the 'from src...' imports to work for type hinting.
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))


from src.global_project_paths import SIMDATA_PATH, FIGURES_PATH
from src.utils.geometry_utils import compute_estimated_shape_from_params, compute_estimated_shape_global, compute_exact_vessel_shape_global

# These imports are for type hinting; the actual classes are found by pickle
from src.utils.SimulationResult import SimulationResult
from src.senfuslib.timesequence import TimeSequence
from src.tracker.TrackerUpdateResult import TrackerUpdateResult

def generate_plotly_html_from_pickle(filename: str):
    with open(os.path.join(SIMDATA_PATH, filename), "rb") as f:
        sim_result: SimulationResult = pickle.load(f)

    config = sim_result.config
    name = config.sim.name
    num_frames = config.sim.num_frames
    
    # Load PCA params only if needed (TrackerConfig usually has the path)
    pca_params = None
    if hasattr(config.tracker, 'PCA_parameters_path'):
        pca_params = np.load(Path(config.tracker.PCA_parameters_path))

    tracker_results_ts = sim_result.tracker_results_ts
    ground_truth_ts = sim_result.ground_truth_ts

    state_posteriors = [r.state_posterior.mean for r in tracker_results_ts.values]
    z = [r.measurements for r in tracker_results_ts.values]
    ground_truth_states = list(ground_truth_ts.values)

    fig = go.Figure()
    # ... (Layout setup remains the same) ...
    fig.update_layout(
        sliders=[{
            # ... slider config ...
            "active": 0, "yanchor": "top", "xanchor": "left",
            "currentvalue": {"prefix": "Frame: ", "font": {"size": 20}},
            "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0,
            "steps": [{"args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": str(i)} for i in range(num_frames+1)]
        }],
        title='Vessel Movement',
        xaxis_title='West-East Position [m]', yaxis_title='North-South Position [m]',
        xaxis=dict(scaleanchor='y', title_font=dict(size=16)), yaxis=dict(title_font=dict(size=16)),
        showlegend=True,
    )

    plot_frames = []
    # Histories for plotting path
    gt_history_x, gt_history_y = [], []

    for frame_idx in range(1, num_frames+1):
        gt_state = ground_truth_states[frame_idx]
        est_state = state_posteriors[frame_idx]
        
        gt_history_x.append(gt_state.x)
        gt_history_y.append(gt_state.y)

        # Ground Truth Shape (Always Explicit Polygon)
        shape_x, shape_y = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)

        z_lidar_cart = z[frame_idx].reshape((-1, 2))

        plot_frame = create_frame_from_state(
            est_state, 
            z_lidar_cart, 
            (frame_idx+1), 
            gt_history_x, gt_history_y, 
            shape_x, shape_y, 
            config, 
            pca_params
        )
        plot_frames.append(plot_frame)

    fig = create_sim_figure(fig, plot_frames, len(plot_frames))
    # ... (Saving logic remains the same) ...
    plot_filename = os.path.join(FIGURES_PATH, f"{name}.html")
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plotly.offline.plot(fig, filename=plot_filename, auto_open=False)
    print(f"HTML plot simulation run saved to {plot_filename}")

def create_frame_from_state(est_state, z_lidar_cart, current_timestep, locationx, locationy, shape_x, shape_y, config, pca_params):
    # Compute Estimated Shape
    est_shape_x, est_shape_y = compute_estimated_shape_global(est_state, config, pca_params)

    # Process Lidar Rays
    lidar_pos = config.lidar.lidar_position
    max_dist = config.lidar.max_distance
    
    lidar_ray_x = []
    lidar_ray_y = []

    for z_pos in z_lidar_cart:
        dist = np.linalg.norm(z_pos - np.array(lidar_pos))
        if dist < max_dist:
            lidar_ray_x.extend([lidar_pos[0], z_pos[0], None])
            lidar_ray_y.extend([lidar_pos[1], z_pos[1], None])

    # Create Arrow for Heading
    est_pos = np.array([est_state.x, est_state.y])
    est_heading = est_state.yaw
    arrow_length = 5.0
    arrow_end = est_pos + arrow_length * np.array([np.cos(est_heading), np.sin(est_heading)])
    
    # Arrowhead logic
    arrowhead_length = arrow_length * 0.3
    arrowhead_angle = np.pi / 6
    ah_left = arrow_end - arrowhead_length * np.array([np.cos(est_heading - arrowhead_angle), np.sin(est_heading - arrowhead_angle)])
    ah_right = arrow_end - arrowhead_length * np.array([np.cos(est_heading + arrowhead_angle), np.sin(est_heading + arrowhead_angle)])

    frame = go.Frame(
        data=[
            go.Scatter(x=locationy, y=locationx, mode='lines', name='Vessel Path', line=dict(color='royalblue')),
            go.Scatter(x=shape_y, y=shape_x, mode='lines', name='Vessel Extent (GT)', line=dict(color='black')),
            go.Scatter(x=est_shape_y, y=est_shape_x, mode='lines', name='Estimated Extent', line=dict(color='green')),
            go.Scatter(x=lidar_ray_y, y=lidar_ray_x, mode='lines+markers', name='LiDAR Rays', line=dict(color='red', width=1)),
            
            # Heading Arrow
            go.Scatter(x=[est_pos[1], arrow_end[1]], y=[est_pos[0], arrow_end[0]], mode='lines', name='Estimated Heading', line=dict(color='purple', width=2)),
            go.Scatter(x=[arrow_end[1], ah_left[1]], y=[arrow_end[0], ah_left[0]], mode='lines', showlegend=False, line=dict(color='purple', width=2)),
            go.Scatter(x=[arrow_end[1], ah_right[1]], y=[arrow_end[0], ah_right[0]], mode='lines', showlegend=False, line=dict(color='purple', width=2))
        ],
        name=str(current_timestep)
    )
    return frame

def create_sim_figure(fig, frames, num_frames):
    fig.add_traces(frames[0].data)
    fig.update(frames=frames)
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            range=[-60, 60],
            constrain='domain',
            gridcolor='rgb(200, 200, 200)',
            zerolinecolor='rgb(200, 200, 200)',
            title=dict(text='East [m]'),
        ),
        yaxis=dict(
            range=[-15, 35],
            scaleanchor="x",
            scaleratio=1,
            gridcolor='rgb(200, 200, 200)',
            zerolinecolor='rgb(200, 200, 200)',
            title=dict(text='North [m]'),
        ),
        updatemenus=[{
            'buttons': [
                {
                    'args': [
                        [None],
                        {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }
                    ],
                    'label': 'Pause',
                    'method': 'animate'
                },
                {
                    'args': [
                        None,
                        {
                            'frame': {'duration': 0, 'redraw': True},
                            'fromcurrent': True
                        }
                    ],
                    'label': 'Play',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 10},
                'prefix': 'Time step: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 0},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [{
                'args': [
                    [str(k)],
                    {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate'
                    }
                ],
                'label': str(k),
                'method': 'animate',
            } for k in range(num_frames + 1)]
        }]
    )
    return fig


def generate_initial_plotly_fig(gt_state, est_state, config, pca_params):
    """Generates the Plotly figure for the initial state (Frame 0)."""
    fig = go.Figure()

    # Add GT shape
    shape_x, shape_y = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)
    fig.add_trace(go.Scatter(x=shape_y, y=shape_x, mode='lines', name='Vessel Extent (GT)', line=dict(color='black')))
    
    # Add Estimated shape
    est_shape_x, est_shape_y = compute_estimated_shape_global(est_state, config, pca_params)
    fig.add_trace(go.Scatter(x=est_shape_y, y=est_shape_x, mode='lines', name='Estimated Extent', line=dict(color='green')))
    
    return fig

def generate_plotly_fig_for_frame(frame_idx, gt_state, est_state, z_lidar_cart, ground_truth_history, config, pca_params):
    """Generates the Plotly figure for a given frame's data."""
    fig = go.Figure()

    # Vessel Path
    locationx = [s.x for s in ground_truth_history]
    locationy = [s.y for s in ground_truth_history]
    fig.add_trace(go.Scatter(x=locationy, y=locationx, mode='lines', name='Vessel Path', line=dict(color='royalblue')))

    # Ground Truth Shape
    shape_x, shape_y = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)
    fig.add_trace(go.Scatter(x=shape_y, y=shape_x, mode='lines', name='Vessel Extent (GT)', line=dict(color='black')))

    # Estimated Shape
    est_shape_x, est_shape_y = compute_estimated_shape_global(est_state, config, pca_params)
    fig.add_trace(go.Scatter(x=est_shape_y, y=est_shape_x, mode='lines', name='Estimated Extent', line=dict(color='green')))

    # LiDAR Rays
    lidar_pos = config.lidar.lidar_position
    lidar_ray_x, lidar_ray_y = [], []
    for z_pos in z_lidar_cart:
        dist = np.linalg.norm(z_pos - np.array(lidar_pos))
        if dist < config.lidar.max_distance:
            lidar_ray_x.extend([lidar_pos[0], z_pos[0], None])
            lidar_ray_y.extend([lidar_pos[1], z_pos[1], None])
    fig.add_trace(go.Scatter(x=lidar_ray_y, y=lidar_ray_x, mode='lines+markers', name='LiDAR Rays', line=dict(color='red', width=1)))

    # Estimated Heading
    est_pos = np.array([est_state.x, est_state.y])
    est_heading = est_state.yaw
    arrow_length = 5.0
    arrow_end = est_pos + arrow_length * np.array([np.cos(est_heading), np.sin(est_heading)])

    # Compute arrowhead points
    arrowhead_length = arrow_length * 0.3  # Shorter than main arrow
    arrowhead_angle = np.pi / 6  # 30 degrees for arrowhead
    arrowhead_left = arrow_end - arrowhead_length * np.array([np.cos(est_heading - arrowhead_angle), np.sin(est_heading - arrowhead_angle)])
    arrowhead_right = arrow_end - arrowhead_length * np.array([np.cos(est_heading + arrowhead_angle), np.sin(est_heading + arrowhead_angle)])
    fig.add_trace(go.Scatter(x=[est_pos[1], arrow_end[1]], y=[est_pos[0], arrow_end[0]], mode='lines', name='Estimated Heading', line=dict(color='purple', width=2)))
    fig.add_trace(go.Scatter(x=[arrow_end[1], arrowhead_left[1]], y=[arrow_end[0], arrowhead_left[0]], mode='lines', showlegend=False, line=dict(color='purple', width=2)))
    fig.add_trace(go.Scatter(x=[arrow_end[1], arrowhead_right[1]], y=[arrow_end[0], arrowhead_right[0]], mode='lines', showlegend=False, line=dict(color='purple', width=2)))

    return fig

if __name__ == "__main__":
    filename = "bfgs_42_ellipse_300frames_2760336220.pkl"
    generate_plotly_html_from_pickle(filename)