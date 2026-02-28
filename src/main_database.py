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

# Imports 
from global_project_paths import SIMDATA_PATH
from src.utils.config_classes import TrackerConfig, SimulationConfig, Config, ExtentConfig, LidarConfig, TrajectoryConfig
from src.states.states import State_GP, State_PCA
from src.experiment_runner import run_single_simulation
from src.utils import SimulationResult
from src.extent_model.boat_pca_utils import get_gt_pca_coeffs_for_boat, get_boat_dimensions

PCA_parameters_path = "data/input_parameters/ShipDatasetPCAParameters.npz" # NOTE Global variable for PCA parameters path

def get_common_configs(traj_type="circle", N_pca=4):
    """Returns configs shared by all methods (Sim, Lidar, Extent)."""
    
    # --- BOAT SELECTION ---
    # Select a boat from processed_ships.json
    selected_boat_id = "1" # Example: "1" = Sailing Yacht, "112" = Multihull
    
    # Get GT Dimensions and PCA Coeffs
    try:
        # L_gt, W_gt = get_boat_dimensions(selected_boat_id) # NOTE We can get L and W from the database but in practice we set them manually
        L_gt = 20.0
        W_gt = 6.0
        gt_pca_coeffs = get_gt_pca_coeffs_for_boat(selected_boat_id, N_pca=N_pca, pca_path=PCA_parameters_path)
    except Exception as e:
        logging.error(f"Could not load boat {selected_boat_id}: {e}")
        # Fallback to simple ellipse
        L_gt, W_gt = 20.0, 6.0
        gt_pca_coeffs = np.zeros(N_pca)
        selected_boat_id = None

    # --- TRAJECTORY ---
    if traj_type == "circle":
        trajectory = TrajectoryConfig(
            type="circle",   
            center=(30.0, 0.0),
            radius=30.0,
            speed=5.0,
            clockwise=False
        )
        start_x, start_y, start_yaw = 0.0, 0.0, np.pi/2

    elif traj_type == "linear":
        trajectory = TrajectoryConfig(
            type="linear",
            speed=5.0
        )
        start_x, start_y, start_yaw = 0.0, -40.0, np.pi/2
    
    elif traj_type == "waypoints":
        trajectory = TrajectoryConfig(
            type="waypoints",
            speed=5.0,
            waypoints=[(0, -40), (0, 40), (60, 40), (60, -40)]
        )
        start_x, start_y, start_yaw = 0.0, -40.0, np.pi/2
        
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    # --- GT STATE ---
    initial_state_gt = State_PCA(
        x=start_x,      
        y=start_y,      
        yaw=start_yaw,  
        vel_x=0.0,     
        vel_y=3.0,      
        yaw_rate=0.0,   
        length=L_gt,   
        width=W_gt,      
        pca_coeffs=gt_pca_coeffs[:N_pca] # Truncate to state size
    )

    sim_config = SimulationConfig(
        name = "",
        num_simulations=1,
        num_frames=500,
        dt=0.1,
        seed=42,
        initial_state_gt=initial_state_gt,
        gt_yaw_rate_std_dev= 0.1 if traj_type == "linear" else 0.0, 
        trajectory=trajectory
    )

    # LiDAR Parameters
    lidar_config = LidarConfig(
        lidar_position=(30.0, 0.0),
        num_rays=360,
        max_distance=140.0,
        lidar_gt_std_dev=0.0, # Perfect measurements for testing
    )

    # Extent config
    # Use "database" type to load the real shape from JSON
    if selected_boat_id:
        shape_params = {
            "type": "database", 
            "id": selected_boat_id,
            "L": L_gt, 
            "W": W_gt 
        }
    else:
        shape_params = {
            "type": "ellipse", 
            "L": L_gt, 
            "W": W_gt 
        }
        
    extent_config = ExtentConfig(
        N_fourier=64,
        d_angle=np.deg2rad(1.0),
        shape_params_true=shape_params
    )
    
    return sim_config, lidar_config, extent_config


def get_pca_tracker_config(lidar_pos, initial_state_gt, N_pca=4):
    """Returns TrackerConfig for PCA methods."""
    
    # Initialize Tracker closer to GT for stability in this test
    # (Or add noise if testing robustness)
    initial_state_tracker = State_PCA(
        x=initial_state_gt.x,
        y=initial_state_gt.y,
        yaw=initial_state_gt.yaw,
        vel_x=initial_state_gt.vel_x, 
        vel_y=initial_state_gt.vel_y, 
        yaw_rate=0.0,
        length=initial_state_gt.length,    
        width=initial_state_gt.width,      
        # pca_coeffs=np.zeros(N_pca) # Start with mean shape of dataset
        pca_coeffs=initial_state_gt.pca_coeffs.copy() # Start with perfect shape
    )

    initial_std_devs_tracker = State_PCA(
        x=2.0, y=2.0, yaw=0.2, 
        vel_x=2.0, vel_y=2.0, yaw_rate=0.1,
        length=2.0, width=2.0,
        pca_coeffs=np.ones(N_pca) * 0.5 
    )

    tracker_config = TrackerConfig(
        use_gt_state_for_bodyangles_calc = False,
        use_initialize_centroid = False,
        N_pca=N_pca,
        PCA_parameters_path=PCA_parameters_path,
        pos_north_std_dev=0.3,
        pos_east_std_dev=0.3,
        heading_std_dev=0.1,
        lidar_std_dev=0.15,
        initial_state=initial_state_tracker,
        initial_std_devs=initial_std_devs_tracker,
        lidar_position=np.array(lidar_pos)
    )
    return tracker_config


if __name__ == "__main__":
    N_pca = 4
    
    # Switch Scenario Here
    # selected_trajectory = "circle" 
    # selected_trajectory = "linear"
    selected_trajectory = "waypoints"

    # Load Configs
    sim_base, lidar_base, extent_base = get_common_configs(traj_type=selected_trajectory, N_pca=N_pca)
    
    print(f"Simulating boat from database (ID: {extent_base.shape_params_true.get('id', 'Unknown')})")
    print(f"L={sim_base.initial_state_gt.length:.2f}, W={sim_base.initial_state_gt.width:.2f}")

    method = "ekf"
    # method = "bfgs"
    # method = "iekf"
    
    tracker_cfg = get_pca_tracker_config(lidar_base.lidar_position, sim_base.initial_state_gt, N_pca)
    tracker_cfg.process_model = 'cv' 

    config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_cfg, extent=extent_base)

    # Unique Name
    boat_id = extent_base.shape_params_true.get('id', 'custom')
    config.sim.name = f"ShipDataset_{boat_id}_{config.sim.trajectory.type}_{method}_GT_init"

    # Run
    sim_result = run_single_simulation(config=config, method=method)
