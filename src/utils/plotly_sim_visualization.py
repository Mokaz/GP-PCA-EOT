import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Plotly figure setup
def initialize_plotly_figure(num_frames):
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
    return fig

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

def create_frame(tracker, noisy_measurements, step, locationx, locationy, shape_x, shape_y, max_distance, lidar_position, angles, compute_estimated_shape):
    est_shape_x, est_shape_y = compute_estimated_shape(tracker=tracker, angles=angles)

    lidar_ray_x = []
    lidar_ray_y = []

    for angle, dist in noisy_measurements:
        if dist < max_distance:
            lidar_ray_x.extend([lidar_position[0], lidar_position[0] + dist * np.cos(angle), None])
            lidar_ray_y.extend([lidar_position[1], lidar_position[1] + dist * np.sin(angle), None])

    # Extract estimated position and heading
    est_pos = np.array([tracker.state[0], tracker.state[1]])  # [North, East]
    est_heading = tracker.state[2]  # Heading angle

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
        name=str(step)
    )
    return frame

def create_dashboard_frame(step, sim_traces_data, analysis_data):
    """
    Creates a single frame for the dashboard, containing both simulation and analysis traces.
    """
    # --- Simulation Traces (from existing create_frame logic) ---
    sim_traces = [
        go.Scatter(x=sim_traces_data['locationy'], y=sim_traces_data['locationx'], mode='lines', name='Vessel Path', line=dict(color='royalblue')),
        go.Scatter(x=sim_traces_data['shape_y'], y=sim_traces_data['shape_x'], mode='lines', name='Vessel Extent', line=dict(color='black')),
        go.Scatter(x=sim_traces_data['est_shape_y'], y=sim_traces_data['est_shape_x'], mode='lines', name='Estimated Extent', line=dict(color='green')),
        go.Scatter(x=sim_traces_data['lidar_ray_y'], y=sim_traces_data['lidar_ray_x'], mode='lines+markers', name='LiDAR Rays', line=dict(color='red', width=1)),
        go.Scatter(x=[sim_traces_data['est_pos'][1], sim_traces_data['arrow_end'][1]], y=[sim_traces_data['est_pos'][0], sim_traces_data['arrow_end'][0]], mode='lines', name='Estimated Heading', line=dict(color='purple', width=2)),
        go.Scatter(x=[sim_traces_data['arrow_end'][1], sim_traces_data['arrowhead_left'][1]], y=[sim_traces_data['arrow_end'][0], sim_traces_data['arrowhead_left'][0]], mode='lines', showlegend=False, line=dict(color='purple', width=2)),
        go.Scatter(x=[sim_traces_data['arrow_end'][1], sim_traces_data['arrowhead_right'][1]], y=[sim_traces_data['arrow_end'][0], sim_traces_data['arrowhead_right'][0]], mode='lines', showlegend=False, line=dict(color='purple', width=2))
    ]

    # --- Analysis Traces ---
    # 1. The moving NEES marker
    nees_marker_trace = go.Scatter(
        x=[step], y=[analysis_data['nees_value']],
        mode='markers', name='Current NEES',
        marker=dict(color='red', size=12, symbol='cross-thin', line=dict(width=2))
    )

    # 2. Heatmaps for the matrices
    heatmap_pprior = go.Heatmap(z=analysis_data['P_prior'], name='P_prior', coloraxis="coloraxis", visible=True)
    heatmap_ppost = go.Heatmap(z=analysis_data['P_post'], name='P_post', coloraxis="coloraxis", visible=False)
    heatmap_s = go.Heatmap(z=analysis_data['S'], name='S', coloraxis="coloraxis", visible=False)

    # Combine all traces for the frame
    frame = go.Frame(
        data=sim_traces + [nees_marker_trace, heatmap_pprior, heatmap_ppost, heatmap_s],
        name=str(step),
        # Assign traces to their respective subplots
        # The first 7 are for sim (1,1), the next 4 are for analysis (1,2)
        traces=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    )
    return frame


def create_dashboard_figure(num_frames, nees_values, all_frames_data):
    """
    Creates a full interactive dashboard with simulation animation and analysis plots.
    """
    # Create a figure with 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        specs=[[{"type": "xy"}, {"type": "heatmap"}]],
        subplot_titles=("Vessel Simulation", "Filter Analysis")
    )

    # --- Add Initial Traces (for frame 0) ---
    # 1. Simulation traces
    initial_sim_traces = all_frames_data[0]['sim_traces']
    fig.add_trace(go.Scatter(x=initial_sim_traces['locationy'], y=initial_sim_traces['locationx'], mode='lines', name='Vessel Path', line=dict(color='royalblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=initial_sim_traces['shape_y'], y=initial_sim_traces['shape_x'], mode='lines', name='Vessel Extent', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=initial_sim_traces['est_shape_y'], y=initial_sim_traces['est_shape_x'], mode='lines', name='Estimated Extent', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=initial_sim_traces['lidar_ray_y'], y=initial_sim_traces['lidar_ray_x'], mode='lines+markers', name='LiDAR Rays', line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[initial_sim_traces['est_pos'][1], initial_sim_traces['arrow_end'][1]], y=[initial_sim_traces['est_pos'][0], initial_sim_traces['arrow_end'][0]], mode='lines', name='Estimated Heading', line=dict(color='purple', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[initial_sim_traces['arrow_end'][1], initial_sim_traces['arrowhead_left'][1]], y=[initial_sim_traces['arrow_end'][0], initial_sim_traces['arrowhead_left'][0]], mode='lines', showlegend=False, line=dict(color='purple', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[initial_sim_traces['arrow_end'][1], initial_sim_traces['arrowhead_right'][1]], y=[initial_sim_traces['arrow_end'][0], initial_sim_traces['arrowhead_right'][0]], mode='lines', showlegend=False, line=dict(color='purple', width=2)), row=1, col=1)

    # 2. Analysis traces
    fig.add_trace(go.Scatter(x=np.arange(num_frames), y=nees_values, mode='lines', name='NEES', line=dict(color='darkgrey')), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0], y=[nees_values[0]], mode='markers', name='Current NEES', marker=dict(color='red', size=12, symbol='cross-thin', line=dict(width=2))), row=1, col=2)
    
    initial_analysis_data = all_frames_data[0]['analysis_data']
    fig.add_trace(go.Heatmap(z=initial_analysis_data['P_prior'], name='P_prior', coloraxis="coloraxis", visible=True), row=1, col=2)
    fig.add_trace(go.Heatmap(z=initial_analysis_data['P_post'], name='P_post', coloraxis="coloraxis", visible=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=initial_analysis_data['S'], name='S', coloraxis="coloraxis", visible=False), row=1, col=2)

    # --- Create all animation frames ---
    frames = [create_dashboard_frame(k, all_frames_data[k]['sim_traces'], all_frames_data[k]['analysis_data']) for k in range(num_frames)]
    fig.frames = frames

    # --- Layout Configuration ---
    fig.update_layout(
        title='Interactive Simulation Dashboard',
        xaxis=dict(range=[-60, 60], title='East [m]', constrain='domain'),
        yaxis=dict(range=[-15, 35], scaleanchor="x", scaleratio=1, title='North [m]'),
        xaxis2=dict(title='Timestep'),
        yaxis2=dict(title='NEES / Matrix Values'),
        coloraxis={'colorscale': 'viridis', 'colorbar': {'title': 'Value'}},
        plot_bgcolor='white',
        paper_bgcolor='white',
        updatemenus=[
            { # Play/Pause
                'buttons': [{'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}], 'label': 'Play', 'method': 'animate'},
                            {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': 'Pause', 'method': 'animate'}],
                'direction': 'left', 'pad': {'r': 10, 't': 70}, 'showactive': False, 'type': 'buttons', 'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'
            },
            { # Matrix Selector
                "buttons": [
                    {"label": "P_prior", "method": "restyle", "args": ["visible", [False]*9 + [True, False, False]]},
                    {"label": "P_post", "method": "restyle", "args": ["visible", [False]*9 + [False, True, False]]},
                    {"label": "S", "method": "restyle", "args": ["visible", [False]*9 + [False, False, True]]},
                ],
                'direction': 'down', 'pad': {'r': 10, 't': 10}, 'showactive': True, 'x': 0.7, 'xanchor': 'left', 'y': 1.05, 'yanchor': 'top'
            }
        ],
        sliders=[{
            'active': 0, 'yanchor': 'top', 'xanchor': 'left',
            'currentvalue': {'font': {'size': 14}, 'prefix': 'Time: ', 'visible': True, 'xanchor': 'right'},
            'transition': {'duration': 0}, 'pad': {'b': 10, 't': 30}, 'len': 0.9, 'x': 0.1, 'y': 0,
            'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}], 'label': f.name, 'method': 'animate'} for f in frames]
        }]
    )
    return fig

