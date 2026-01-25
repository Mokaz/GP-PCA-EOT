import numpy as np

from dataclasses import dataclass, field
from typing import Tuple, Sequence, List

from src.senfuslib import SensorModel
from src.states.states import State_PCA, LidarScan
from src.utils.geometry_utils import compute_exact_vessel_shape_global
from src.utils.tools import cast_rays, add_noise_to_distances, ur, rot2D, drot2D, fourier_basis_matrix, pol2cart
from src.utils.config_classes import ExtentConfig

@dataclass
class LidarMeasurementModel(SensorModel[Sequence[LidarScan]]):
    """
    A measurement model for the tracker.
    Represents the mathematical model of the sensor (h(x), H(x), R).
    """
    lidar_position: np.ndarray
    lidar_std_dev: float
    pca_mean: np.ndarray
    pca_eigenvectors: np.ndarray
    extent_cfg: ExtentConfig
    
    def h_lidar(self, x, body_angles: list[float]):
        L = x[6]
        W = x[7]
        
        normalized_angles = np.arctan2(np.sin(body_angles) / W, np.cos(body_angles) / L)

        pos = x[:2].reshape(-1, 1)  # shape (2, 1)
        R_heading = rot2D(x[2])    # shape (2, 2)
        LW_scaling = np.diag([L, W])  # shape (2, 2)

        fourier_coeffs = self.pca_mean + self.pca_eigenvectors @ x[8:].reshape(-1, 1)  # shape (N_f, 1)

        # Vectorized direction vectors: shape (2, N) -- one column per angle
        ur_all = ur(normalized_angles)  # shape (2, N), ur returns [cos(θ); sin(θ)]

        # Vectorized g(theta): shape (N_f, N)
        g_all = fourier_basis_matrix(normalized_angles, N_fourier=self.extent_cfg.N_fourier)  # shape (N_f, N)

        # Compute predicted measurement points in global frame
        #   ur_all * (g_all.T @ fourier_coeffs): shape (2, N)
        #   (g_all.T @ fourier_coeffs): shape (N,)
        #   full expression: shape (2, N) → pos + ... → shape (2, N)
        z_pred = np.squeeze(pos + R_heading @ LW_scaling @ ur_all * (g_all.T @ fourier_coeffs).flatten())

        return z_pred.T  # shape (N, 2)

    def lidar_jacobian(self, x, body_angles: list[float]):
        
        """
        Compute the Jacobian of the measurement function h with respect to the state x.     
        
        Parameters:
        x (np.array): Current state (n-dimensional).
        - x[0:2]: Position (North, East)
        - x[2]: Heading
        - x[3:5]: Velocity (North, East)
        - x[5]: Rate of turn
        - x[6]: Length
        - x[7]: Width
        - x[8:]: Fourier coefficients

        body_angles (list[float]): Body angles for the measurements.

        Returns:
        np.array: Jacobian matrix (m x n).
        """

        L = x[6]
        W = x[7]

        # Normalize the angles
        normalized_angles = np.arctan2(np.sin(body_angles) / W, np.cos(body_angles) / L)

        fourier_coeffs = self.pca_mean + self.pca_eigenvectors @ x[8:].reshape(-1, 1)

        jacobians = []

        for angle in normalized_angles:
            fourier_approx = (fourier_basis_matrix(angle, self.extent_cfg.N_fourier).T @ fourier_coeffs).item()

            dp_dpc = np.eye(2)
            dp_dphi = drot2D(x[2]) @ np.diag([L, W]) @ (ur(angle) * fourier_approx).reshape(-1, 1)
            dp_dv = np.zeros([2, 2])
            dp_dr = np.zeros([2, 1])
            dp_dLW = rot2D(x[2]) @ np.diag(ur(angle).flatten()) * fourier_approx
            dp_de = rot2D(x[2]) @ np.diag([L, W]) @ ur(angle).T * (fourier_basis_matrix(angle, self.extent_cfg.N_fourier).T @ self.pca_eigenvectors).flatten()

            H_i = np.hstack([dp_dpc, dp_dphi, dp_dv, dp_dr, dp_dLW, dp_de])
            jacobians.append(H_i)

        return np.vstack(jacobians)

    def R(self, num_measurements: int) -> np.ndarray:
        """
        Measurement noise covariance (R).
        Assumes independent noise for each point's x and y coordinate.
        """

        R_single_point = self.R_single_point()
        return np.kron(np.eye(num_measurements), R_single_point)
    
    def R_single_point(self) -> np.ndarray:
        """
        Measurement noise covariance for a single point.
        May be used externally
        """
        return np.eye(2) * self.lidar_std_dev**2


@dataclass
class LidarSimulator(SensorModel[Sequence[LidarScan]]):
    """
    A simulation model for a LiDAR.
    Handles ray-casting and noise generation.
    """
    lidar_position: np.ndarray
    num_rays: int
    max_distance: float
    lidar_gt_std_dev: float
    rng: np.random.Generator = field(repr=False)
    extent_cfg: ExtentConfig

    def sample_from_state(self, x_gt: State_PCA) -> LidarScan:
        """
        Overrides the base method to simulate LiDAR measurements from the ground truth state.
        """
        shape_x_coords, shape_y_coords = compute_exact_vessel_shape_global(
            x_gt, self.extent_cfg.shape_coords_body
        )

        polar_measurements = np.array(self.simulate_lidar_measurements(shape_x_coords, shape_y_coords))
        assert polar_measurements.shape[1] == 2, "Expected polar_measurements to have shape (N, 2)"

        x_coords, y_coords = pol2cart(polar_measurements[:, 0], polar_measurements[:, 1])
        
        return LidarScan(x=x_coords, y=y_coords)

    def simulate_lidar_measurements(self, shape_x, shape_y) -> List[Tuple[float, float]]:
        """
        Simulate LiDAR measurements by casting rays from the LiDAR position to the vessel shape.
        Returns noisy distance measurements in polar coordinates.
        """
        # Noise characteristics
        lidar_position = self.lidar_position
        num_rays = self.num_rays
        max_distance = self.max_distance

        angles, distances = cast_rays(lidar_position, num_rays, max_distance, shape_x, shape_y)
        noisy_measurements = add_noise_to_distances(self.rng, distances, angles, 0.0, self.lidar_gt_std_dev)
        return noisy_measurements