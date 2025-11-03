import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EKFConfig:
    N_pca: int = 4

    # Standard deviations for process noise
    pos_north_std_dev: float = 0.3
    pos_east_std_dev: float = 0.3
    heading_std_dev: float = 0.1

    # Standard deviations for measurement noise
    lidar_std_dev: float = 0.15
    ais_pos_std_dev: float = 3.0
    ais_heading_std_dev: float = 0.1
    ais_length_std_dev: float = 0.2
    ais_width_std_dev: float = 0.2

    # Initial state vector. If not provided, a default is created in __post_init__.
    state: Optional[np.ndarray] = None
    
    # Lidar position, to be set from LidarConfig
    lidar_pos: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.state is None:
            # The state vector has 8 kinematic/extent parts + N_pca shape parts
            self.state = np.zeros(8 + self.N_pca)
            self.state[0] = 0.0       # North position
            self.state[1] = -40.0     # East position
            self.state[2] = np.pi / 2 # Heading angle
            # state[3:6] are velocities/rates, default to 0
            self.state[6] = 20.0      # Length
            self.state[7] = 6.0       # Width
            # state[8:] are PCA coefficients, default to 0

