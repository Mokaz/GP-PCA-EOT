from dataclasses import dataclass
from typing import Tuple

@dataclass
class LidarConfig:
    """
    Configuration parameters for the LiDAR sensor.
    """
    # LiDAR Parameters
    lidar_position: Tuple[float, float] = (30.0, 0.0)
    num_rays: int = 360
    max_distance: float = 140.0
    noise_mean: float = 0.0
    noise_std_dev: float = 0.0
