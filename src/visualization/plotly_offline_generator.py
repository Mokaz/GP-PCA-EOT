import sys
import numpy as np
import os
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

# Go up two levels to the project root (GP-PCA-EOT)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(PROJECT_ROOT)

from src.config_paths import SIMDATA_PATH, FIGURES_PATH
from src.extent_model.geometry_utils import compute_estimated_shape_from_params

# Import necessary classes from main.py to access configuration dataclasses
from src.config_classes import Config, SimulationParams
from src.sensors.lidar import LidarConfig
from src.utils.ekf_config import EKFConfig
from src.utils.SimulationResult import SimulationResult


def generate_plotly_html_from_pickle(filename: str, sim_selection: int = 0):

    with open(os.path.join(SIMDATA_PATH, filename), "rb") as f:
        sim_data = pickle.load(f)

    # Extract the data
    state_posteriors = sim_data.state_posteriors[sim_selection]
    ground_truth    = sim_data.ground_truth[sim_selection]
    static_covariances = sim_data.static_covariances[sim_selection]
    true_extent     = sim_data.true_extent[sim_selection]
    P_prior         = sim_data.P_prior[sim_selection]
    P_post          = sim_data.P_post[sim_selection]
    S               = sim_data.S[sim_selection]
    y               = sim_data.y[sim_selection]
    z_flattened     = sim_data.z[sim_selection]
    x_dim           = sim_data.x_dim[sim_selection]
    z_dim           = sim_data.z_dim[sim_selection]
    shape_x_list    = sim_data.shape_x[sim_selection]
    shape_y_list    = sim_data.shape_y[sim_selection]

    PCA_eigenvectors_M = sim_data.PCA_eigenvectors
    fourier_coeff_mean = sim_data.PCA_mean
    initial_condition  = sim_data.initial_condition
    num_frames         = sim_data.num_frames
    angles             = sim_data.angles
    lidar_max_distance = sim_data.lidar_max_distance
    lidar_position     = sim_data.lidar_position

    config = sim_data.config
    name = config.sim.name # TODO Martin: fix this hacky way to get the name

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
        locationx.append(ground_truth[frame_idx][0])
        locationy.append(ground_truth[frame_idx][1])

        x_pos_est = state_posteriors[frame_idx][0]
        y_pos_est = state_posteriors[frame_idx][1]
        heading_est = state_posteriors[frame_idx][2]
        L = state_posteriors[frame_idx][6]
        W = state_posteriors[frame_idx][7]
        PCA_coeffs = state_posteriors[frame_idx][8:]

        shape_x = shape_x_list[frame_idx]
        shape_y = shape_y_list[frame_idx]

        z_lidar_cart = z_flattened[frame_idx].reshape((-1, 2))

        plot_frame = create_frame_from_params(x_pos_est, y_pos_est, heading_est, L, W, PCA_coeffs, z_lidar_cart, (frame_idx+1), locationx, locationy, 
                                shape_x, shape_y, lidar_max_distance, lidar_position, angles, fourier_coeff_mean, PCA_eigenvectors_M)
        plot_frames.append(plot_frame)

    fig = create_sim_figure(fig, plot_frames, len(plot_frames))
    plot_filename = os.path.join(FIGURES_PATH, f"{name}.html")
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plotly.offline.plot(fig, filename=plot_filename, auto_open=False)
    print(f"Plot for first simulation run saved to {plot_filename}")

def create_frame_from_params(x_pos_est, y_pos_est, heading_est, L, W, PCA_coeffs, z_lidar_cart, current_timestep, locationx, locationy, shape_x, shape_y, lidar_max_distance, lidar_position, angles, fourier_coeff_mean, PCA_eigenvectors_M):
    # Compute estimated shape based on current state estimate
    est_shape_x, est_shape_y = compute_estimated_shape_from_params(x_pos_est, y_pos_est, heading_est, L, W, PCA_coeffs, fourier_coeff_mean, PCA_eigenvectors_M, angles)

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
    filename = "bfgs_ellipsis_100frames.pkl"
    generate_plotly_html_from_pickle(filename, sim_selection=0)