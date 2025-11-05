import numpy as np
from scipy.optimize import minimize

from src.tracker.tracker import Tracker
from src.utils.tools import ssa, initialize_centroid, compute_angle_range
from src.utils.ekf_config import EKFConfig

class BFGS(Tracker):
    def __init__(self, process_model, timestep: float, rng, max_iterations=10, convergence_threshold=1e-9, config=None):
        """
        Initializes the BFGS tracker.

        Parameters:
        - process_model: The process model to be used.
        - timestep: The time step for the tracker.
        - config: Configuration object containing parameters for the tracker.
        """
        super().__init__(process_model, timestep, rng, config)
        
        # Parameters
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        self.measurements_polar = None
        self.mean_lidar_measurement_angle = None
        self.upper_diff = None
        self.lower_diff = None
    
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
        state_vector = self.state.copy()

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
        v_combined = None
        S_combined = None

        # Find correct initialization point for vessel position
        state_vector[:2] = initialize_centroid(state_vector[:2], lidar_pos, lidar_measurements_polar, L_est=state_vector[6], W_est=state_vector[7]) # TODO Martin This is weird

        # Extract the angles from the measurements
        angles = np.array([ssa(measurement[0]) for measurement in lidar_measurements_polar])

        # Calculate the minimum and maximum angles from the measurements
        lower_diff, upper_diff, mean_lidar_angle = compute_angle_range(angles)

        self.mean_lidar_measurement_angle = mean_lidar_angle

        # LiDAR measurement points in cartesian coordinates and global frame
        lidar_measurements = lidar_pos + lidar_measurements_polar[:, 1].reshape(-1, 1) * np.array([np.cos(lidar_measurements_polar[:, 0]), np.sin(lidar_measurements_polar[:, 0])]).T

        # Get measurement vector
        if ais_available:
            z_combined = np.array([*lidar_measurements.flatten(), *ais_measurements])
        else:
            z_combined = lidar_measurements.flatten()

        # State vector updates
        state_iterates = [state_vector.copy()]

        if ais_available:
            R = np.block([[np.kron(np.eye(num_meas), self.R_lidar), np.zeros((2*num_meas, 5))],
                      [np.zeros((5, 2*num_meas)), self.R_ais]
                      ])
        else:
            R = np.kron(np.eye(num_meas), self.R_lidar)

        P_pred = self.P.copy()

        # Test the 'trust-exact' optimizer to potentially improve convergence and accuracy
        # of the state estimation by minimizing the negative log-posterior.
        res = minimize(self.prob_with_penalty, state_vector, 
                       args=(z_combined, self.h, R, state_vector, P_pred, ssa_vec, ais_available, ground_truth, mean_lidar_angle, lower_diff, upper_diff, lidar_pos), 
                       method='BFGS')#, options={'ftol': self.convergence_threshold})
        
        state_pred = self.state.copy()

        # Update final state
        self.state = res.x

        state_post = self.state.copy()

        x_dim = self.state_dim

        self.P = res.hess_inv

        return state_pred, state_post, z_combined, v_combined, S_combined, P_pred, self.P, z_dim, x_dim

    def penalty_function(self, x, z, h, mean_angle, lower_diff, upper_diff, lidar_pos):
        L = x[6]
        W = x[7]
        angles = np.array([0.0, np.arctan2(W, L), np.pi/2, np.arctan2(W, -L), np.pi, np.arctan2(-W, -L), -np.pi / 2, np.arctan2(-W, L)])

        pred_meas = h(x, angles, False)
        pred_meas = np.array(pred_meas).reshape(-1, 2)

        # Ensure the angles of prediced measurements are within the range
        pred_angles = np.arctan2(pred_meas[:, 1] - lidar_pos[1], pred_meas[:, 0] - lidar_pos[0])
        pred_angles = np.array([ssa(angle) for angle in pred_angles])

        alpha_min = mean_angle + lower_diff
        alpha_max = mean_angle + upper_diff

        # Calculate the penalty
        gain = 50.0
        penalty = 0.0
        for i in range(len(pred_angles)):
            penalty += gain * max(0.0, ssa(pred_angles[i] - alpha_max)) + gain * max(0.0, ssa(alpha_min - pred_angles[i]))

        return penalty

    def prob_with_penalty(self, x, z, h, R, x_pred, P_pred, ssa_func, ais_received, ground_truth, mean_angle, lower_diff, upper_diff, lidar_pos):
        return self.object_function(x, z, h, R, x_pred, P_pred, ssa_func, ais_received, ground_truth) + self.penalty_function(x, z, h, mean_angle, lower_diff, upper_diff, lidar_pos)
