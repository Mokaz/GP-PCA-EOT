import numpy as np
from utils.tools import ssa, initialize_centroid
from src.tracker.tracker import Tracker

class EKF(Tracker):
    def __init__(self, process_model, timestep, config):
        """
        Initializes the Extended Kalman Filter (EKF) tracker.
        Parameters:
        - process_model: The process model to be used.
        - timestep: The time step for the tracker.
        - config: Configuration object containing parameters for the tracker.
        """
        print(f"Position North Std Dev inside EKF: {config.pos_north_std_dev}")
        rng = None
        super().__init__(process_model, timestep, rng, config)

    def predict(self, F_jacobian):
        """
        Prediction step.
        
        Parameters:
        - F_jacobian: Function to calculate the Jacobian of the state transition model w.r.t. state (F)
        
        Updates:
        - self.state: Predicted state estimate
        - self.P: Predicted covariance estimate
        """
        # Compute the predicted state using the process model
        self.state = self.dynamic_model(self.state, self.T)

        # Compute the Jacobian of the transition model
        F = F_jacobian(self.state, self.T, self.N_pca)
        
        # Compute the predicted covariance
        self.P = F @ self.P @ F.T + self.Q 

    def update(self, lidar_measurements_polar, lidar_pos, ais_measurements=None, ground_truth=None):
        """
        Combined update function for LiDAR and AIS measurements, with ground truth corrections.
        
        Parameters:
        - lidar_measurements: LiDAR measurements (available always)
        - ais_measurements: AIS measurements (optional, may be None)
        - vessel_gt: Ground truth of the vessel
        - L_gt: Ground truth length of the vessel
        - W_gt: Ground truth width of the vessel
        """

        # State vector before and after
        state_iterates = [self.state.copy()]
        state_vector = self.state.copy()
        
        # LiDAR Update
        lidar_measurements_polar = np.array(lidar_measurements_polar)
        num_meas = len(lidar_measurements_polar)

        # Find correct initialization point for vessel position
        state_vector[:2] = initialize_centroid(
            state_vector[:2], lidar_pos, lidar_measurements_polar, 
            L_est=state_vector[6], W_est=state_vector[7]
        )

        # LiDAR measurement points in cartesian coordinates and global frame
        lidar_measurements = lidar_pos + lidar_measurements_polar[:, 1].reshape(-1, 1) * np.array([np.cos(lidar_measurements_polar[:, 0]), np.sin(lidar_measurements_polar[:, 0])]).T
        
        ssa_vec = np.vectorize(ssa)
        body_angles = ssa_vec(np.arctan2(
            lidar_measurements[:, 1] - self.state[1], 
            lidar_measurements[:, 0] - self.state[0]
        ) - self.state[2])

        # Get measurement vector
        if ais_measurements is not None:
            z_combined = np.array([*lidar_measurements.flatten(), *ais_measurements])
        else:
            z_combined = lidar_measurements.flatten()

        # Predict LiDAR measurement
        z_pred_lidar = self.h_lidar(self.state, body_angles)
        y_lidar = (lidar_measurements - z_pred_lidar).reshape(-1, 1)
        
        # Compute Jacobian and Kalman gain for LiDAR
        H_lidar = self.lidar_jacobian(self.state, body_angles)
        S_lidar = H_lidar @ self.P @ H_lidar.T + np.kron(np.eye(num_meas), self.R_lidar)
        K_lidar = self.P @ H_lidar.T @ np.linalg.inv(S_lidar)
        
        if ais_measurements is not None:
            # AIS Update if AIS data is available
            pos = self.state[:2]
            heading = self.state[2]
            L = self.state[6]
            W = self.state[7]

            # Predict AIS measurement
            z_pred_ais = np.array([*pos, heading, L, W])
            y_ais = ais_measurements - z_pred_ais
            y_ais[0] = ssa(y_ais[0])
            y_ais = y_ais.reshape(-1, 1)  # Reshape to (5, 1) for compatibility

            # Construct the Jacobian for AIS
            H_ais = self.ais_jacobian()

            # Combine LiDAR and AIS updates
            y_combined = np.vstack([y_lidar, y_ais])
            combined_measurements = y_combined
            H_combined = np.vstack([H_lidar, H_ais])
            S_combined = H_combined @ self.P @ H_combined.T + np.block([
                [np.kron(np.eye(num_meas), self.R_lidar), np.zeros((2*num_meas, 5))],
                [np.zeros((5, 2*num_meas)), self.R_ais]
            ])
            K_combined = self.P @ H_combined.T @ np.linalg.inv(S_combined)

            # Apply update
            update_vector = K_combined @ combined_measurements
        else:
            # Only LiDAR is available
            update_vector = K_lidar @ y_lidar
            y_combined = y_lidar
            K_combined = K_lidar  # Fallback to LiDAR-only Kalman gain
            H_combined = H_lidar  # Fallback to LiDAR-only measurement Jacobian
            S_combined = S_lidar  # Fallback to LiDAR-only innovation covariance

        state_update =  update_vector.flatten()

        # Update state estimates
        self.state += state_update

        # Dimensions of the state and measurement vector
        x_dim = self.state_dim
        z_dim = 2 * num_meas

        # Ensure heading angle remains within -π to π range
        self.state[2] = ssa(self.state[2])

        # Append the updated state after ground truth corrections
        state_iterates.append(self.state.copy())

        P_pred = self.P.copy()

        # Update covariance matrix with the combined Kalman gain
        I = np.eye(self.state_dim)
        self.P = (I - K_combined @ H_combined) @ self.P @ (I - K_combined @ H_combined).T + K_combined @ S_combined @ K_combined.T

        return state_iterates, z_combined, y_combined, S_combined, P_pred, self.P, z_dim, x_dim

