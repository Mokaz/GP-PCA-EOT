import os
import sys
import numpy as np
from dataclasses import dataclass, field

from utils.ekf_config import EKFConfig
from sensors.LidarConfig import LidarConfig
from simulation import run_simulation_with_plot

# Initialize project and import modules
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

@dataclass
class SimulationParams:
    name: str = "default_simulation"
    # method : str = "iekf"

    num_simulations: int = 1
    num_frames: int = 100
    timestep: float = 0.1
    seed: int = 42

    # Target Vessel Parameters
    d_angle: float = np.deg2rad(1.0)
    L_gt: float = 20.0
    W_gt: float = 6.0
    
    angles: np.ndarray = field(init=False)
    param_true: dict = field(init=False)

    def __post_init__(self):
        self.angles = np.arange(-np.pi, np.pi, self.d_angle)
        self.param_true = {
            "type": "ellipsis", 
            "L": self.L_gt, 
            "W": self.W_gt, 
            "P": self.L_gt * 0.2, 
            "S": self.L_gt * 0.1
        }

@dataclass
class Config:
    sim: SimulationParams
    lidar: LidarConfig
    tracker: EKFConfig

if __name__ == "__main__":
    # Simulation Parameters
    sim_params = SimulationParams(
        name = "",
        num_simulations=1,
        num_frames=90,
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
        config.sim.name = f"martin_sim_{method}_ellipse_{sim_params.num_frames}frames"

        # Pass the single config object
        # monte_carlo(config=config, method=method, create_plot=True)
        run_simulation_with_plot(config=config, method=method)