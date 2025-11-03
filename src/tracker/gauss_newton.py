import numpy as np

from src.tracker.tracker import Tracker
from src.utils.tools import ssa, initialize_centroid
from src.utils.ekf_config import EKFConfig

class GaussNewton(Tracker):
    def __init__(self, process_model, timestep: float, rng, max_iterations=10, convergence_threshold=1e-3, config: EKFConfig=None):
        super().__init__(process_model, timestep, rng, config)

        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
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
        self.state = self.process_model(self.state, self.T)

        # Compute the Jacobian of the transition model
        F = F_jacobian(self.state, self.T, self.N_pca)
        
        # Compute the predicted covariance
        self.P = F @ self.P @ F.T + self.Q 

    def update(self, lidar_measurements_polar, lidar_pos, ais_measurements=None, ground_truth=None):
        iteration = 0
        state_vector = self.state.copy()

        ais_available = ais_measurements is not None

        # LiDAR update preparation
        lidar_pos = np.array(lidar_pos)
        lidar_measurements_polar = np.array(lidar_measurements_polar)
        num_meas = len(lidar_measurements_polar)

        if ais_available:
            z_dim = 2 * num_meas + 5
        else:
            z_dim = 2 * num_meas

        # Find correct initialization point for vessel position
        state_vector[:2] = initialize_centroid(
            state_vector[:2], lidar_pos, lidar_measurements_polar, 
            L_est=state_vector[6], W_est=state_vector[7]
        )

        # LiDAR measurement points in cartesian coordinates
        lidar_measurements = lidar_pos + lidar_measurements_polar[:, 1].reshape(-1, 1) * np.array([np.cos(lidar_measurements_polar[:, 0]), np.sin(lidar_measurements_polar[:, 0])]).T

        # Get measurement vector
        if ais_available:
            z = np.array([*lidar_measurements.flatten(), *ais_measurements])
        else:
            z = lidar_measurements.flatten()

        state_iterates = [state_vector.copy()]
        prev_state = state_vector.copy()

        # Gauss-Newton Iteration
        while iteration < self.max_iterations:
            # Compute body angles for LiDAR
            body_angles = np.arctan2(
                lidar_measurements[:, 1] - state_vector[1], 
                lidar_measurements[:, 0] - state_vector[0]
            ) - state_vector[2]

            # Predict measurement
            z_pred = self.h(state_vector, body_angles, ais_available)
            y= z - np.array(z_pred).flatten()

            if ais_available:
                y[-3] = ssa(y[-3])  # Ensure heading difference is within [-π, π]

            # Compute Jacobian
            H = self.jacobian(state_vector, body_angles, ais_available)

            # Compute Gauss-Newton step
            if ais_available:
                R = np.block([
                    [np.kron(np.eye(num_meas), self.R_lidar), np.zeros((2*num_meas, 5))],
                    [np.zeros((5, 2*num_meas)), self.R_ais]
                ])
            else:
                R = np.kron(np.eye(num_meas), self.R_lidar)

            # Gauss-Newton Hessian approximation (Marquardt damping)
            R_inv = np.linalg.inv(R)
            H_T_R_inv = H.T @ R_inv
            Hessian_approx = H_T_R_inv @ H
            grad = H_T_R_inv @ y

            # print eigenvalues of Hessian
            # if not np.all(np.linalg.eigvals(Hessian_approx)) > 0:
            #     print("Eigenvalues of Hessian: ", np.all(np.linalg.eigvals(Hessian_approx)) > 0)

            # Solve for update step (Gauss-Newton update)
            delta_state = np.linalg.solve(Hessian_approx, grad)

            # Apply updated
            state_vector += delta_state.flatten()

            # Ensure heading angle remains within -π to π range
            state_vector[2] = ssa(state_vector[2])

            state_iterates.append(state_vector.copy())

            # Convergence check
            diff = state_vector - prev_state
            diff[2] = ssa(diff[2])
            change = np.linalg.norm(diff)

            if change < self.convergence_threshold:
                break

            prev_state = state_vector.copy()
            iteration += 1

        # Update final state
        self.state = state_vector

        S = None

        x_dim = self.state_dim

        P_pred = self.P.copy()

        # Compute covariance update (Gauss-Newton approximation)
        self.P = np.linalg.inv(Hessian_approx)  # Inverse of approximate Hessian

        return state_iterates, z, y, S, P_pred, self.P, z_dim, x_dim
