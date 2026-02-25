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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from global_project_paths import SIMDATA_PATH
from src.utils.config_classes import TrackerConfig, SimulationConfig, Config, ExtentConfig, LidarConfig
from src.states.states import State_GP, State_PCA
from src.visualization.plotly_offline_generator import generate_plotly_html_from_pickle

from src.simulation import run_single_simulation
from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result
from src.analysis.consistency_analysis import PlotterTrackerPCA
from src.utils import SimulationResult

def get_common_configs(N_pca=4):
    """Returns configs shared by all methods (Sim, Lidar, Extent)."""

    initial_state_gt = State_PCA(
        x=0.0,          # North position
        y=-40.0,        # East position
        yaw=np.pi / 2,  # Heading angle
        vel_x=0.0,      # North Velocity
        vel_y=3.0,      # East Velocity
        yaw_rate=0.0,   # Yaw Rate
        length=20.0,    # Length
        width=6.0,      # Width
        pca_coeffs=np.array([-1.1716024210647493, 0.1598108552333908, 0.06624525016067562, 0.07254312701009336]) # GT for ellipse shape L=20m, W=6m
    )

    sim_config = SimulationConfig(
        name = "",
        num_simulations=1,
        num_frames=500,
        dt=0.1,
        seed=42,
        initial_state_gt=initial_state_gt,
    )

    # sim_config.trajectory.type = "linear" # "linear", "circle", "waypoints"
    # Configure orbit
    sim_config.trajectory.type = "circle"
    sim_config.trajectory.center = (30.0, 0.0) # LiDAR pos
    sim_config.trajectory.radius = 40.0        # Orbit at 40m distance
    sim_config.trajectory.speed = 5.0          # Go faster

    # Ensure initial state matches the start of the trajectory to avoid "snap"
    # Starting at (30+40, 0) -> (70, 0) facing North (pi/2) for CCW orbit
    initial_state_gt.x = 70.0
    initial_state_gt.y = 0.0
    initial_state_gt.yaw = np.pi / 2

    # LiDAR Parameters
    lidar_config = LidarConfig(
        lidar_position=(30.0, 0.0),
        num_rays=360,
        max_distance=140.0,
        lidar_gt_mean=0.0,# NOTE: Currently unused
        lidar_gt_std_dev=0.0,
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
    return sim_config, lidar_config, extent_config

def get_pca_tracker_config(lidar_pos, N_pca=4):
    """Returns TrackerConfig for PCA methods (EKF, IEKF, BFGS)."""
    # Tracker Config & Initial State
    initial_state_tracker = State_PCA(
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

    # Define initial std devs for PCA
    initial_std_devs_tracker = State_PCA(
        x=2.0, y=2.0, yaw=0.2, 
        vel_x=2.0, vel_y=2.0, yaw_rate=0.1,
        length=2.0, width=2.0,
        pca_coeffs=np.ones(N_pca) * 0.5 # NOTE Overwritten by eigenvalues in tracker initialization
    )

    tracker_config = TrackerConfig(
        use_gt_state_for_bodyangles_calc = False,
        use_initialize_centroid = False,
        N_pca=N_pca,
        # PCA_parameters_path="data/input_parameters/BoatPCAParameters.npz",
        PCA_parameters_path="data/input_parameters/FourierPCAParameters_scaled.npz",
        pos_north_std_dev=0.3,
        pos_east_std_dev=0.3,
        heading_std_dev=0.1,
        lidar_std_dev=0.05,
        initial_state=initial_state_tracker,
        initial_std_devs=initial_std_devs_tracker,
        lidar_position=np.array(lidar_pos)
    )
    return tracker_config

def get_gp_tracker_config(lidar_pos, N_gp=20):
    """Returns TrackerConfig for GP methods."""
    
    # Initialize radii (circle of radius 10 m)
    initial_radii = np.ones(N_gp) * 10.0
    
    initial_state_tracker = State_GP(
        x=0.0, y=-40.0, yaw=np.pi / 2, vel_x=0.0, vel_y=3.0, yaw_rate=0.0,
        radii=initial_radii
    )

    initial_std_devs_tracker = State_GP(
        x=2.0, y=2.0, yaw=0.2, 
        vel_x=2.0, vel_y=2.0, yaw_rate=0.1,
        radii=np.ones(N_gp) * 5.0 # (5m std dev)
    )

    return TrackerConfig(
        use_gt_state_for_bodyangles_calc=False,
        
        # GP Specific Params
        N_gp_points=N_gp,
        gp_length_scale=np.pi / 2,
        gp_signal_var=1.0,
        gp_forgetting_factor=0.001,
        gp_use_negative_info=True,

        initial_state=initial_state_tracker,
        initial_std_devs=initial_std_devs_tracker,
        lidar_position=np.array(lidar_pos),
        
        # Standard noise params
        pos_north_std_dev=0.3, pos_east_std_dev=0.3, heading_std_dev=0.1, lidar_std_dev=0.15
    )


if __name__ == "__main__":
    GENERATE_PLOTLY_HTML = False
    CONSISTENCY_ANALYSIS = False
    LOAD_SIM_RESULT = False # TODO: ONLY USES CONFIG FOR ID GENERATION FOR NOW

    N_pca = 4
    N_gp = 20

    # Load base configs
    sim_base, lidar_base, extent_base = get_common_configs(N_pca)

    # method_list = ["bfgs", "ekf", "iekf", "gp_iekf"]
    method_list = ["ekf", "iekf"]
    # method_list = ["implicit_iekf"]

    for method in method_list:
        print(f"--- Setting up for method: {method} ---")

        if "gp" in method:
            tracker_cfg = get_gp_tracker_config(lidar_base.lidar_position, N_gp)
        else:
            tracker_cfg = get_pca_tracker_config(lidar_base.lidar_position, N_pca)
            
            # --- Process Model Selection ---
            # Options: 'cv' (Constant Velocity), 'temporal' (OU Process), 'inflation' (Covariance Inflation)
            tracker_cfg.process_model = 'cv' 
            # tracker_cfg.inflation_lambda = 0.99
            # tracker_cfg.temporal_eta = 1.0

        config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_cfg, extent=extent_base)

        # Create a unique name for this simulation configuration
        id_number = crc32(repr(config).encode())
        config.sim.name = f"{config.sim.trajectory.type}_{method}_seed_{config.sim.seed}"

        filename = f"{config.sim.name}.pkl"
        pickle_path = Path(SIMDATA_PATH) / filename

        if pickle_path.exists() and LOAD_SIM_RESULT:
            print(f"Loading existing result: {filename}")
            with open(pickle_path, "rb") as f:
                sim_result: SimulationResult = pickle.load(f)
        else:
            sim_result = run_single_simulation(config=config, method=method)