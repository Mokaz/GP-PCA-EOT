from dataclasses import dataclass, field
from typing import List, Any, Dict
import numpy as np

@dataclass
class SimulationResult:
    """
    A data class to store the results of a simulation run.
    """
    # --- Simulation run data ---
    state_predictions: List[Any]
    state_posteriors: List[Any]
    ground_truth: List[Any]
    P_prior: List[Any]
    P_post: List[Any]
    S: List[Any]
    y: List[Any]
    z: List[Any]
    x_dim: List[int]
    z_dim: List[int]
    shape_x: List[Any]
    shape_y: List[Any]
    initial_condition: List[Any]

    # --- Config data ---
    config: Any
    lidar_position: np.ndarray
    lidar_max_distance: float
    true_extent: np.ndarray
    true_extent_radius: np.ndarray
    N_pca: int
    angles: np.ndarray
    num_simulations: int
    num_frames: int

    # --- Static data ---
    PCA_mean: np.ndarray
    PCA_eigenvectors: np.ndarray
    static_covariances: List[np.ndarray]