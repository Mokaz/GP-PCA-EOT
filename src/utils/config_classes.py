import numpy as np
from typing import Optional

from dataclasses import dataclass, field

from sensors.lidar import LidarConfig
from src.states.states import State_PCA

@dataclass
class SimulationConfig:
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
class TrackerConfig:
    N_pca: int = 4
    N_fourier: int = 64

    # Standard deviations for process noise
    pos_north_std_dev: float = 0.3
    pos_east_std_dev: float = 0.3
    heading_std_dev: float = 0.1

    # Standard deviations for measurement noise
    lidar_std_dev: float = 0.15

    # Define initial standard deviations for each state variable.
    initial_std_devs: State_PCA = field(default_factory=lambda: State_PCA(
        x=2.0,
        y=2.0,
        yaw=0.2,
        vel_x=2.0,
        vel_y=2.0,
        yaw_rate=0.1,
        length=2.0,
        width=2.0,
        # PCA coeffs will be handled separately
    ))

    # Initial state vector. If not provided, a default is created in __post_init__.
    initial_state: Optional[State_PCA] = None
    
    # Lidar position, to be set from LidarConfig
    lidar_pos: Optional[np.ndarray] = None

    PCA_parameters_path : str = 'data/input_parameters/FourierPCAParameters_scaled.npz'

    def __post_init__(self):
        if self.initial_state is None:
            self.initial_state = State_PCA(
                x=0.0,
                y=-40.0,
                yaw=np.pi / 2,
                vel_x=0.0,
                vel_y=0.0,
                yaw_rate=0.0,
                length=20.0,
                width=6.0,
                pca_coeffs=np.zeros(self.N_pca)
            )

@dataclass
class Config:
    sim: SimulationConfig
    lidar: LidarConfig
    tracker: TrackerConfig