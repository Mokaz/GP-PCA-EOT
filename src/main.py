import os
import sys
import numpy as np


# Correct the path to be relative to this file's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from src.config_paths import SIMDATA_PATH, FIGURES_PATH
from src.utils.ekf_config import EKFConfig
from src.sensors.lidar import LidarConfig
from src.visualization.plotly_offline_generator import generate_plotly_html_from_pickle

from src.simulation import run_single_simulation
from src.config_classes import SimulationParams, Config

if __name__ == "__main__":
    GENERATE_PLOTLY_HTML = True

    # Simulation Parameters
    sim_params = SimulationParams(
        name = "",
        num_simulations=1,
        num_frames=100,
        timestep=0.1,
        seed=42,
        d_angle=np.deg2rad(1.0),
        L_gt=20.0,
        W_gt=6.0,
    )

    # LiDAR Parameters
    lidar_config = LidarConfig(
        lidar_position=(30.0, 0.0),
        num_rays=360,
        max_distance=140.0,
        noise_mean=0.0,
        noise_std_dev=0.0,
    )

    # method_list = ["iekf", "ukf", "bfgs", "slsqp", "gauss_newton", "levenberg_marquardt", "smoothing_slsqp"]
    method_list = ["bfgs"]

    for method in method_list:
        print(f"Running method: {method}")
        
        # Tracker Parameters
        N_pca = 4
        initial_state = np.zeros(8 + N_pca)
        initial_state[0] = 0.0      # North position
        initial_state[1] = -40.0    # East position
        initial_state[2] = np.pi / 2# Heading angle
        initial_state[4] = 3.0      # East Velocity
        initial_state[6] = 20.0     # Length
        initial_state[7] = 6.0      # Width

        ekf_config = EKFConfig(
            N_pca=N_pca,
            pos_north_std_dev=0.3,
            pos_east_std_dev=0.3,
            heading_std_dev=0.1,
            lidar_std_dev=0.15,
            state=initial_state,
            lidar_pos=np.array(lidar_config.lidar_position)
        )

        # Combine into a single config object
        config = Config(sim=sim_params, lidar=lidar_config, tracker=ekf_config)
        config.sim.name = f"{method}_{sim_params.param_true.get('type')}_{sim_params.num_frames}frames"

        run_single_simulation(config=config, method=method)

        if GENERATE_PLOTLY_HTML:
            pickle_filename = os.path.join(SIMDATA_PATH, f"{config.sim.name}.pkl")
            generate_plotly_html_from_pickle(pickle_filename, sim_selection=0)
