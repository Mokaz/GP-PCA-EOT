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
from src.utils.geometry_utils import compute_estimated_shape_from_params, compute_exact_vessel_shape_global

# These imports are for type hinting; the actual classes are found by pickle
from src.utils.SimulationResult import SimulationResult
from src.senfuslib.timesequence import TimeSequence
from src.tracker.tracker import TrackerUpdateResult


def generate_plotly_html_from_pickle(filename: str):
    with open(os.path.join(SIMDATA_PATH, filename), "rb") as f:
        sim_result: SimulationResult = pickle.load(f)

    # Extract config objects for convenience
    config = sim_result.config
    name = config.sim.name
    num_frames = config.sim.num_frames
    lidar_max_distance = config.lidar.max_distance
    lidar_position = config.lidar.lidar_position
    angles = config.extent.angles
    N_fourier = config.extent.N_fourier
    shape_coords_body = config.extent.shape_coords_body

    # Extract static PCA parameters from the config within the simulation result
    pca_params = np.load(Path(config.tracker.PCA_parameters_path))
    PCA_eigenvectors_M = pca_params['eigenvectors'][:, :config.tracker.N_pca].real
    fourier_coeff_mean = pca_params['mean']

    # Extract TimeSequences
    tracker_results_ts = sim_result.tracker_results_ts
    ground_truth_ts = sim_result.ground_truth_ts

    # Iterate through TimeSequences to create lists of data for each frame
    # Access .values as a property, not a method
    state_posteriors = [r.estimate_posterior.mean for r in tracker_results_ts.values]
    z = [r.measurements for r in tracker_results_ts.values]
    ground_truth_states = list(ground_truth_ts.values)

    # Initialize the figure
    fig = go.Figure()
    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {"prefix": "Frame: ", "font": {"size": 20}},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": str(i)
                    } for i in range(num_frames+1)
                ]
            }
        ],
        title='Vessel Movement',
        xaxis_title='West-East Position [m]',
        yaxis_title='North-South Position [m]',
        xaxis=dict(scaleanchor='y', title_font=dict(size=16)),
        yaxis=dict(title_font=dict(size=16)),
        showlegend=True,
    )
    plot_frames, locationx, locationy = [], [], []

    for frame_idx in range(num_frames):
        # Create plot frame
        # The first update is at t=dt, which corresponds to the second GT state (index 1).
        gt_state = ground_truth_states[frame_idx + 1]
        est_state = state_posteriors[frame_idx]
        
        locationx.append(gt_state.x)
        locationy.append(gt_state.y)

        # Use named attributes for clarity and correctness
        x_pos_est = est_state.x
        y_pos_est = est_state.y
        heading_est = est_state.yaw
        L = est_state.length
        W = est_state.width
        PCA_coeffs = est_state.pca_coeffs

        shape_x, shape_y = compute_exact_vessel_shape_global(
            gt_state, shape_coords_body
        )

        z_lidar_cart = z[frame_idx].reshape((-1, 2))

        plot_frame = create_frame_from_params(x_pos_est, y_pos_est, heading_est, L, W, PCA_coeffs, z_lidar_cart, (frame_idx+1), locationx, locationy, 
                                shape_x, shape_y, lidar_max_distance, lidar_position, angles, fourier_coeff_mean, PCA_eigenvectors_M, N_fourier)
        plot_frames.append(plot_frame)

    fig = create_sim_figure(fig, plot_frames, len(plot_frames))
    plot_filename = os.path.join(FIGURES_PATH, f"{name}.html")
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plotly.offline.plot(fig, filename=plot_filename, auto_open=False)
    print(f"Plot for first simulation run saved to {plot_filename}")

def create_frame_from_params(x_pos_est, y_pos_est, heading_est, L, W, PCA_coeffs, z_lidar_cart, current_timestep, locationx, locationy, shape_x, shape_y, lidar_max_distance, lidar_position, angles, fourier_coeff_mean, PCA_eigenvectors_M, N_fourier):
    # Compute estimated shape based on current state estimate
    est_shape_x, est_shape_y = compute_estimated_shape_from_params(x_pos_est, y_pos_est, heading_est, L, W, PCA_coeffs, fourier_coeff_mean, PCA_eigenvectors_M, angles, N_fourier)

    lidar_ray_x = []
    lidar_ray_y = []

    for z_pos in z_lidar_cart:
        dist = np.linalg.norm(z_pos - np.array(lidar_position))
        if dist < lidar_max_distance:
            lidar_ray_x.extend([lidar_position[0], z_pos[0], None])
            lidar_ray_y.extend([lidar_position[1], z_pos[1], None])

    # Extract estimated position and heading
    est_pos = np.array([x_pos_est, y_pos_est])  # [North, East]
    est_heading = heading_est  # Heading angle

    # Define arrow properties
    arrow_length = 5.0  # Adjust as needed
    arrowhead_length = arrow_length * 0.3  # Shorter than main arrow
    arrowhead_angle = np.pi / 6  # 30 degrees for arrowhead

    # Compute arrow endpoint
    arrow_end = est_pos + arrow_length * np.array([np.cos(est_heading), np.sin(est_heading)])

    # Compute arrowhead points
    arrowhead_left = arrow_end - arrowhead_length * np.array([np.cos(est_heading - arrowhead_angle), np.sin(est_heading - arrowhead_angle)])
    arrowhead_right = arrow_end - arrowhead_length * np.array([np.cos(est_heading + arrowhead_angle), np.sin(est_heading + arrowhead_angle)])

    frame = go.Frame(
        data=[
            go.Scatter(x=locationy, y=locationx, mode='lines', name='Vessel Path', line=dict(color='royalblue')),
            go.Scatter(x=shape_y, y=shape_x, mode='lines', name='Vessel Extent', line=dict(color='black')),
            go.Scatter(x=est_shape_y, y=est_shape_x, mode='lines', name='Estimated Extent', line=dict(color='green')),
            go.Scatter(x=lidar_ray_y, y=lidar_ray_x, mode='lines+markers', name='LiDAR Rays', line=dict(color='red', width=1)),

            # Heading arrow
            go.Scatter(x=[est_pos[1], arrow_end[1]], y=[est_pos[0], arrow_end[0]],
                    mode='lines', name='Estimated Heading', line=dict(color='purple', width=2)),

            # Arrowhead
            go.Scatter(x=[arrow_end[1], arrowhead_left[1]], y=[arrow_end[0], arrowhead_left[0]],
                    mode='lines', showlegend=False, line=dict(color='purple', width=2)),
            go.Scatter(x=[arrow_end[1], arrowhead_right[1]], y=[arrow_end[0], arrowhead_right[0]],
                    mode='lines', showlegend=False, line=dict(color='purple', width=2))
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

if __name__ == "__main__":
    filename = "bfgs_ellipse_100frames.pkl"
    generate_plotly_html_from_pickle(filename)