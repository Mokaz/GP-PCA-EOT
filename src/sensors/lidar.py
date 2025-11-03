import numpy as np

from dataclasses import dataclass
from typing import Tuple
from src.utils.tools import cast_rays, add_noise_to_distances

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

def simulate_lidar_measurements(shape_x, shape_y, lidar_config, rng: np.random.Generator):
    """
    Simulate LiDAR measurements by casting rays from the LiDAR position to the vessel shape.
    Returns noisy distance measurements in polar coordinates.
    """
    # Noise characteristics
    lidar_position = lidar_config.lidar_position
    num_rays = lidar_config.num_rays
    max_distance = lidar_config.max_distance
    lidar_noise_mean = lidar_config.noise_mean
    lidar_noise_std_dev = lidar_config.noise_std_dev

    angles, distances = cast_rays(lidar_position, num_rays, max_distance, shape_x, shape_y)
    noisy_measurements = add_noise_to_distances(rng, distances, angles, lidar_noise_mean, lidar_noise_std_dev)
    return noisy_measurements