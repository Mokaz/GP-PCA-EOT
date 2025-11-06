import numpy as np
from utils.tools import ssa, initialize_centroid
from src.tracker.tracker import Tracker

class IterativeEKF(Tracker):
    def __init__(self, process_model, timestep, rng, max_iterations=100, convergence_threshold=1e-6, config=None):
        """
        Initializes the Iterative Extended Kalman Filter (EKF) tracker.

        Parameters:
        - process_model: The process model to be used.
        - timestep: The time step for the tracker.
        - rng: Random number generator for noise.
        - max_iterations: Maximum number of iterations for the update step.
        - convergence_threshold: Threshold for convergence of the state estimate.
        - config: Configuration object containing parameters for the tracker.
        """
        super().__init__(process_model, timestep, rng, config)
        
        # Parameters
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def predict(self, F_jacobian):
        """
        Prediction step of the EKF.
        
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
        iteration = 0
        state_vector = self.state.copy()

        prev_state = state_vector.copy()

        ais_available = ais_measurements is not None

        # LiDAR update preparation
        lidar_pos = np.array(lidar_pos)  # Ensure lidar_pos is defined
        lidar_measurements_polar = np.array(lidar_measurements_polar)
        num_meas = len(lidar_measurements_polar)

        if ais_available:
            z_dim = 2 * num_meas + 5
        else:
            z_dim = 2 * num_meas

        ssa_vec = np.vectorize(ssa)

        # Consistency Analysis
        y_combined = None
        S_combined = None

        # Find correct initialization point for vessel position
        state_vector[:2] = initialize_centroid(state_vector[:2], lidar_pos, lidar_measurements_polar, L_est=state_vector[6], W_est=state_vector[7])

        # LiDAR measurement points in cartesian coordinates and global frame
        lidar_measurements = lidar_pos + lidar_measurements_polar[:, 1].reshape(-1, 1) * np.array([np.cos(lidar_measurements_polar[:, 0]), np.sin(lidar_measurements_polar[:, 0])]).T

        # Get measurement vector
        if ais_available:
            z_combined = np.array([*lidar_measurements.flatten(), *ais_measurements])
        else:
            z_combined = lidar_measurements.flatten()

        # State vector updates
        state_iterates = [state_vector.copy()]

        while iteration < self.max_iterations:
            # Compute body angles
            body_angles = ssa_vec(np.arctan2(
                lidar_measurements[:, 1] - state_vector[1], 
                lidar_measurements[:, 0] - state_vector[0]
            ) - state_vector[2])

            # Predict LiDAR measurement
            z_pred_lidar = self.h_lidar(state_vector, body_angles)
            y_lidar = (lidar_measurements - z_pred_lidar).reshape(-1, 1)
            
            # Compute Jacobian and Kalman gain for LiDAR
            H_lidar = self.lidar_jacobian(state_vector, body_angles)
            S_lidar = H_lidar @ self.P @ H_lidar.T + np.kron(np.eye(num_meas), self.R_lidar)
            K_lidar = self.P @ H_lidar.T @ np.linalg.inv(S_lidar)
            
            if ais_measurements is not None:
                # AIS Update if AIS data is available
                pos = state_vector[:2]
                heading = state_vector[2]
                L = state_vector[6]
                W = state_vector[7]

                # Predict AIS measurement
                z_pred_ais = np.array([*pos, heading, L, W])
                y_ais = ais_measurements - z_pred_ais
                y_ais[2] = ssa(y_ais[2])
                y_ais = y_ais.reshape(-1, 1)

                # Construct the Jacobian for AIS
                H_ais = self.ais_jacobian()

                # Combine LiDAR and AIS updates
                y_combined = np.vstack([y_lidar, y_ais])
                H_combined = np.vstack([H_lidar, H_ais])
                R_combined = np.block([
                    [np.kron(np.eye(num_meas), self.R_lidar), np.zeros((2*num_meas, 5))],
                    [np.zeros((5, 2*num_meas)), self.R_ais]
                ])
                S_combined = H_combined @ self.P @ H_combined.T + R_combined
                K_combined = self.P @ H_combined.T @ np.linalg.inv(S_combined)

                # Apply update
                update_vector = K_combined @ y_combined

                z_dim = 2 * num_meas + 5
            else:
                # Only LiDAR is available
                update_vector = K_lidar @ y_lidar
                y_combined = y_lidar
                K_combined = K_lidar  # Fallback to LiDAR-only Kalman gain
                H_combined = H_lidar  # Fallback to LiDAR-only measurement Jacobian
                R_combined = np.kron(np.eye(num_meas), self.R_lidar)
                S_combined = S_lidar  # Fallback to LiDAR-only innovation covariance

                z_dim = 2 * num_meas

            # Apply the update to the state
            state_vector += update_vector.flatten()

            # Ensure heading angle remains within -π to π range
            state_vector[2] = ssa(state_vector[2])

            #state_iterates.append(state_vector.copy())

            # Check for convergence
            diff = state_vector - prev_state
            diff[2] = ssa(diff[2])
            change = np.linalg.norm(diff)
            if np.max(change) < self.convergence_threshold:
                break

            # print("Eigenvalues of Hessian: ", np.all(np.linalg.eigvals(H_combined.T @ np.linalg.inv(R_combined) @ H_combined) > 0))

            prev_state = state_vector.copy()
            iteration += 1

        # Update final state
        self.state = state_vector

        state_iterates.append(self.state.copy())

        x_dim = self.state_dim

        P_pred = self.P.copy()

        # Update covariance matrix with the combined Kalman gain
        I = np.eye(self.state_dim)
        self.P = (I - K_combined @ H_combined) @ self.P @ (I - K_combined @ H_combined).T + K_combined @ S_combined @ K_combined.T

        return state_iterates, z_combined, y_combined, S_combined, P_pred, self.P, z_dim, x_dim
    
