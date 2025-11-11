import os
import sys
import pickle
import numpy as np
from pathlib import Path
from zlib import crc32

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

# Correct the path to be relative to this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from global_project_paths import SIMDATA_PATH
from utils.config_classes import TrackerConfig, SimulationConfig, Config, ExtentConfig, LidarConfig
# Import the State_PCA class
from src.states.states import State_PCA
from src.visualization.plotly_offline_generator import generate_plotly_html_from_pickle

from src.simulation import run_single_simulation
from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result
from src.analysis.consistency_analysis import PlotterTrackerPCA
from src.utils import SimulationResult

if __name__ == "__main__":
    GENERATE_PLOTLY_HTML = True
    CONSISTENCY_ANALYSIS = True
    LOAD_SIM_RESULT = False # TODO MARTIN: ONLY USES CONFIG FOR ID GENERATION FOR NOW

    # Simulation Parameters
    sim_config = SimulationConfig(
        name = "",
        num_simulations=1,
        num_frames=300,
        dt=0.1,
        seed=42,
    )

    # LiDAR Parameters
    lidar_config = LidarConfig(
        lidar_position=(30.0, 0.0),
        num_rays=360,
        max_distance=140.0,
        lidar_noise_mean=0.0,
        lidar_std_dev=0.15,
    )

    # Tracker Parameters
    N_pca = 4
    
    initial_state_obj = State_PCA(
        x=0.0,          # North position
        y=-40.0,        # East position
        yaw=np.pi / 2,  # Heading angle
        vel_x=0.0,      # North Velocity
        vel_y=3.0,      # East Velocity
        yaw_rate=0.0,   # Yaw Rate
        length=20.0,    # Length
        width=6.0,      # Width
        pca_coeffs=np.zeros(N_pca)
    )

    tracker_config = TrackerConfig(
        use_gt_state_for_bodyangles_calc = True, # NOTE using gt for bodyangles calc 
        N_pca=N_pca,
        pos_north_std_dev=0.3,
        pos_east_std_dev=0.3,
        heading_std_dev=0.1,
        lidar_std_dev=0.0,
        initial_state=initial_state_obj,
        lidar_position=np.array(lidar_config.lidar_position)
    )
    
    # Extent config
    L_gt = 20.0
    W_gt = 6.0
    extent_config = ExtentConfig(
        N_fourier=64,
        d_angle=np.deg2rad(1.0),
        shape_params_true = {
            "type": "ellipse", 
            "L": L_gt, 
            "W": W_gt, 
            "P": L_gt * 0.2, 
            "S": L_gt * 0.1
        }
    )

    # method_list = ["iekf", "ukf", "bfgs", "slsqp", "gauss_newton", "levenberg_marquardt", "smoothing_slsqp"]
    method_list = ["bfgs"]

    for method in method_list:
        print(f"Running method: {method}")

        # Combine into a single config object
        config = Config(sim=sim_config, lidar=lidar_config, tracker=tracker_config, extent=extent_config)

        id_number = crc32(repr(config).encode())

        config.sim.name = f"{method}_{config.sim.seed}_{extent_config.shape_params_true.get('type')}_{sim_config.num_frames}frames_{id_number:010d}"

        filename = f"{config.sim.name}.pkl"
        pickle_path = Path(SIMDATA_PATH) / filename
        if pickle_path.exists() and LOAD_SIM_RESULT:
            # Load existing simulation data
            with open(pickle_path, "rb") as f:
                sim_result: SimulationResult = pickle.load(f)
        else:
            sim_result = run_single_simulation(config=config, method=method)

        if GENERATE_PLOTLY_HTML:
            pickle_filename = os.path.join(SIMDATA_PATH, f"{config.sim.name}.pkl")
            generate_plotly_html_from_pickle(pickle_filename)

        if CONSISTENCY_ANALYSIS:
            analysis = create_consistency_analysis_from_sim_result(sim_result)
            plotter = PlotterTrackerPCA(sim_result, analysis)
            plotter.show()