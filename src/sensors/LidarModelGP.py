import numpy as np
from dataclasses import dataclass, field
from typing import Sequence, Tuple, List

from src.senfuslib import SensorModel
from src.states.states import State_GP, LidarScan
from src.utils.geometry_utils import compute_exact_vessel_shape_global
from src.utils.tools import cast_rays, add_noise_to_distances, rot2D, drot2D, pol2cart
from src.utils.GaussianProcess import GaussianProcess

@dataclass
class LidarModelGP(SensorModel[Sequence[LidarScan]]):
    """
    A sensor model for a LiDAR that measures the shape of a vessel modeled by a GP.
    """
    # Standard Sensor parameters
    lidar_position: np.ndarray
    num_rays: int
    max_distance: float
    lidar_std_dev: float
    
    # GP Utilities for interpolation logic
    gp_utils: GaussianProcess
    
    # Simulation specific
    rng: np.random.Generator = field(repr=False)

    # Simulation helper to define 'True' shape for data generation
    shape_coords_body: np.ndarray = field(default=None, repr=False)

    def h_lidar(self, x: State_GP, body_angles: list[float]) -> np.ndarray:
        """
        Predicts LiDAR measurements given the state and incidence angles.
        Returns: np.ndarray of shape (N_meas, 2) in Global Coordinates.
        """
        # 1. GP Interpolation (Kriging)
        k_vector = self.gp_utils.compute_k_vector(body_angles)
        
        # Calculate interpolation weights: W = k * Ktt_inv
        interpolation_weights = k_vector @ self.gp_utils.Ktt_inv # (N_meas, N_gp)
        
        # Predicted radius at each body angle
        x_radii = x.radii.reshape(-1, 1) # (N_gp, 1)
        r_pred = (interpolation_weights @ x_radii).flatten() # (N_meas,)

        # 2. Convert to Global Cartesian Frame
        # Explicitly build (2, N) unit vectors to avoid 'ur' shape ambiguity
        u_vecs = np.vstack([np.cos(body_angles), np.sin(body_angles)]) # Shape (2, N)
        
        # r_pred is (N,). Broadcasting (2, N) * (N,) works as (2, N) * (1, N)
        body_points = u_vecs * r_pred 

        # Global transformation: pos + R(yaw) * body_points
        pos = x.pos.reshape(2, 1)
        R = rot2D(x.yaw)
        
        z_pred_global = pos + R @ body_points
        
        return z_pred_global.T # Return (N, 2)

    def lidar_jacobian(self, x: State_GP, body_angles: list[float]) -> np.ndarray:
        """
        Compute the Jacobian of the measurement function h with respect to the state x.
        H matrix shape: (2 * N_meas, N_state)
        """
        N_meas = len(body_angles)
        
        # 1. Precompute Interpolation Weights
        k_vector = self.gp_utils.compute_k_vector(body_angles)
        W_weights = k_vector @ self.gp_utils.Ktt_inv # (N_meas, N_gp)

        # 2. Precompute Predicted Radius
        r_pred = (W_weights @ x.radii).flatten()
        
        # 3. Precompute matrices
        R = rot2D(x.yaw)      # Rotation
        dR = drot2D(x.yaw)    # Derivative of Rotation w.r.t yaw
        
        # Explicitly build (2, N) unit vectors
        u_vecs = np.vstack([np.cos(body_angles), np.sin(body_angles)])

        jacobians = []

        # Loop per measurement point
        for i in range(N_meas):
            r = r_pred[i]
            # Take the i-th column (2,) and reshape to (2, 1)
            u = u_vecs[:, i].reshape(2, 1) 
            
            # --- Kinematics ---
            dp_dpos = np.eye(2)
            dp_dyaw = dR @ (r * u)
            dp_dvel = np.zeros((2, 2))
            dp_dyawrate = np.zeros((2, 1))

            # --- Extent (GP) ---
            # d(pos)/d(radii) = R * u * w_i
            # w_i is the row of interpolation weights for this angle (1, N_gp)
            w_i = W_weights[i, :].reshape(1, -1)
            dp_dradii = (R @ u) @ w_i
            
            # Stack row: [Eye, dYaw, 0, 0, dRadii]
            H_i = np.hstack([dp_dpos, dp_dyaw, dp_dvel, dp_dyawrate, dp_dradii])
            jacobians.append(H_i)

        return np.vstack(jacobians)

    def R(self, num_measurements: int) -> np.ndarray:
        """
        Measurement noise covariance (R).
        """
        R_single_point = self.R_single_point()
        return np.kron(np.eye(num_measurements), R_single_point)
    
    def R_single_point(self) -> np.ndarray:
        return np.eye(2) * self.lidar_std_dev**2

    def sample_from_state(self, x_gt: State_GP) -> LidarScan:
        """
        Simulate LiDAR measurements from a ground truth state.
        Uses explicit polygon shape defined in shape_coords_body.
        """
        if self.shape_coords_body is None:
            # Fallback: Sample directly from the GP shape if no explicit GT polygon is provided
            return self._sample_from_gp_shape(x_gt)

        shape_x_coords, shape_y_coords = compute_exact_vessel_shape_global(
            x_gt, self.shape_coords_body
        )

        polar_measurements = np.array(self.simulate_lidar_measurements(shape_x_coords, shape_y_coords))
        
        x_coords, y_coords = pol2cart(polar_measurements[:, 0], polar_measurements[:, 1])
        return LidarScan(x=x_coords, y=y_coords)

    def _sample_from_gp_shape(self, x_gt: State_GP) -> LidarScan:
        """
        Fallback simulation: Raycast against the GP radius function directly.
        (Simplified implementation: just samples the support points)
        """
        N_dense = 360
        dense_angles = np.linspace(0, 2*np.pi, N_dense, endpoint=False)
        z_global = self.h_lidar(x_gt, list(dense_angles)) # (N, 2)
        
        return LidarScan(x=z_global[:, 0], y=z_global[:, 1])

    def simulate_lidar_measurements(self, shape_x, shape_y) -> List[Tuple[float, float]]:
        lidar_noise_mean = 0.0
        angles, distances = cast_rays(
            self.lidar_position, self.num_rays, self.max_distance, shape_x, shape_y
        )
        noisy_measurements = add_noise_to_distances(
            self.rng, distances, angles, lidar_noise_mean, self.lidar_std_dev
        )
        return noisy_measurements