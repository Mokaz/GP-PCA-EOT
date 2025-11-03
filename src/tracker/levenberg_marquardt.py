import numpy as np
from scipy.optimize import minimize

from src.tracker.tracker import Tracker
from src.utils.tools import ssa, initialize_centroid


class LevenbergMarquardt(Tracker):
    def __init__(self, process_model, timestep: float, rng, max_iterations=100, convergence_threshold=1e-6, config=None):
        """
        Initializes the Levenberg-Marquardt tracker.

        Parameters:
        - process_model: The process model to be used.
        - timestep: The time step for the tracker.
        - config: Configuration object containing parameters for the tracker.
        """
        super().__init__(process_model, timestep, rng, config)
        
        # Parameters
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.damping_factor = 1e-3

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

        z_dim = 2 * num_meas + 5 if ais_available else 2 * num_meas

        # Find correct initialization point for vessel position
        state_vector[:2] = initialize_centroid(
            state_vector[:2], lidar_pos, lidar_measurements_polar, 
            L_est=state_vector[6], W_est=state_vector[7]
        )

        # LiDAR measurement points in Cartesian coordinates
        angles, ranges = lidar_measurements_polar[:, 0], lidar_measurements_polar[:, 1]
        lidar_measurements = lidar_pos + ranges.reshape(-1, 1) * np.column_stack((np.cos(angles), np.sin(angles)))

        # Get measurement vector
        z = np.hstack((lidar_measurements.flatten(), ais_measurements)) if ais_available else lidar_measurements.flatten()

        state_iterates = [state_vector.copy()]
        prev_state = state_vector.copy()

        P_pred = self.P.copy()

        # Gauss-Newton Iteration
        while iteration < self.max_iterations:
            # Compute body angles for LiDAR
            body_angles = np.arctan2(
                lidar_measurements[:, 1] - state_vector[1], 
                lidar_measurements[:, 0] - state_vector[0]
            ) - state_vector[2]

            # Predict measurement
            z_pred = self.h(state_vector, body_angles, ais_available)
            y = z - np.array(z_pred).flatten()

            if ais_available:
                y[-3] = ssa(y[-3])  # Ensure heading difference is within [-π, π]

            # Compute Jacobian
            H = self.jacobian(state_vector, body_angles, ais_available)

            # Compute Gauss-Newton step
            R = np.block([
                [np.kron(np.eye(num_meas), self.R_lidar), np.zeros((2*num_meas, 5))],
                [np.zeros((5, 2*num_meas)), self.R_ais]
            ]) if ais_available else np.kron(np.eye(num_meas), self.R_lidar)

            # Gauss-Newton Hessian approximation (Marquardt damping)
            R_inv = np.linalg.inv(R)
            H_T_R_inv = H.T @ R_inv
            Hessian_approx = H_T_R_inv @ H + self.damping_factor * np.diag(np.diagonal(H_T_R_inv @ H))
            grad = H_T_R_inv @ y

            # Check eigenvalues of Hessian
            # if not np.all(np.linalg.eigvals(Hessian_approx) > 0):
            #     print("Eigenvalues of Hessian: ", np.linalg.eigvals(Hessian_approx))

            # Solve for update step (Gauss-Newton update)
            delta_state = np.linalg.solve(Hessian_approx, grad)

            # Trust-region constraint
            trust_radius = self.damping_factor
            delta_norm = np.linalg.norm(delta_state)
            if delta_norm > trust_radius:
                delta_state *= trust_radius / delta_norm  # Scale down to stay within trust region

            # Apply update
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

        x_dim = self.state_dim

        # Compute covariance update (Gauss-Newton approximation)
        self.P = np.linalg.inv(Hessian_approx)  # Inverse of approximate Hessian

        return state_iterates, z, y, None, P_pred, self.P, z_dim, x_dim
    
    # def update(self, lidar_measurements, lidar_pos, ais_measurements=None, ground_truth=None):
    #     ais_available = ais_measurements is not None

    #     # Convert inputs
    #     lidar_pos = np.array(lidar_pos)
    #     lidar_measurements = np.array(lidar_measurements)
    #     num_meas = len(lidar_measurements)

    #     z_dim = 2 * num_meas + 5 if ais_available else 2 * num_meas

    #     # Initial guess
    #     state_vector = self.state.copy()
    #     state_vector[:2] = initialize_centroid(
    #         state_vector[:2], lidar_pos, lidar_measurements, 
    #         L_est=state_vector[6], W_est=state_vector[7]
    #     )

    #     # Cartesian LiDAR points
    #     angles, ranges = lidar_measurements[:, 0], lidar_measurements[:, 1]
    #     lidar_measurements = lidar_pos + ranges.reshape(-1, 1) * np.column_stack((np.cos(angles), np.sin(angles)))

    #     # Measurement vector
    #     z = np.hstack((lidar_measurements.flatten(), ais_measurements)) if ais_available else lidar_measurements.flatten()

    #     ssa_vec = np.vectorize(ssa)

    #     # Covariance matrix
    #     R = np.block([
    #         [np.kron(np.eye(num_meas), self.R_lidar), np.zeros((2 * num_meas, 5))],
    #         [np.zeros((5, 2 * num_meas)), self.R_ais]
    #     ]) if ais_available else np.kron(np.eye(num_meas), self.R_lidar)

    #     R_inv = np.linalg.inv(R)

    #     P_pred = self.P.copy()

    #     result = minimize(
    #         fun=self.object_function,
    #         args=(z, self.h, R, state_vector, P_pred, ssa_vec, ais_available, ground_truth),
    #         x0=state_vector,
    #         jac=self.compute_jacobian_hessian_numerical,
    #         method='trust-constr',
    #         options={'maxiter': self.max_iterations, 'gtol': self.convergence_threshold}
    #     )

    #     self.state = result.x
    #     self.state[2] = ssa(self.state[2])  # Normalize heading

    #     # Approximate final Jacobian for covariance update
    #     body_angles = np.arctan2(
    #         lidar_measurements[:, 1] - self.state[1],
    #         lidar_measurements[:, 0] - self.state[0]
    #     ) - self.state[2]
    #     H_final = self.jacobian(self.state, body_angles, ais_available)

    #     P_pred = self.P.copy()
    #     Hessian_approx = H_final.T @ R_inv @ H_final
    #     self.P = np.linalg.inv(Hessian_approx  + self.damping_factor * np.eye(Hessian_approx.shape[0]))

    #     x_dim = self.state_dim
    #     z_pred = self.h(self.state, body_angles, ais_available)
    #     y = z - np.array(z_pred).flatten()
    #     if ais_available:
    #         y[-3] = ssa(y[-3])

    #     return [self.state.copy()], z, y, None, P_pred, self.P, z_dim, x_dim
