import numpy as np

from dataclasses import dataclass, field

from src.sensors.lidar import LidarConfig
from src.utils.ekf_config import EKFConfig

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