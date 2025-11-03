import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, show_options

from src.tracker.tracker import Tracker
from src.utils.tools import ssa, ur, ut, initialize_centroid, compute_angle_range, cart2pol
from src.utils.ekf_config import EKFConfig

class SLSQP(Tracker):
    def __init__(self, process_model, timestep: float, rng, max_iterations=10, convergence_threshold=1e-9, config: EKFConfig=None):
        """
        Initializes the SLSQP tracker.

        Parameters:
        - process_model: The process model to be used.
        - timestep: The time step for the tracker.
        - config: Configuration object containing parameters for the tracker.
        """
        super().__init__(process_model, timestep, rng, config)
        
        # Parameters
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        #self.damping_factor = 1e-6

        self.measurements_polar = None
        self.mean_lidar_measurement_angle = None
        self.upper_diff = None
        self.lower_diff = None
        self.lidar_pos = None  # Will be set during the update step
    
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
        self.lidar_pos = np.array(lidar_pos)  # Ensure lidar_pos is defined
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

        # Extract the angles from the measurements
        angles = np.array([ssa(measurement[0]) for measurement in lidar_measurements_polar])

        # Calculate the minimum and maximum angles from the measurements
        lower_diff, upper_diff, mean_lidar_angle = compute_angle_range(angles)

        self.mean_lidar_measurement_angle = mean_lidar_angle
        self.upper_diff = upper_diff
        self.lower_diff = lower_diff

        self.measurements_polar = lidar_measurements_polar

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

        self.measurements = lidar_measurements

        # Create constraints
        constraints = []

        # Add buffer to lower and upper constraint limits
        buffer = np.deg2rad(0.5)
        lower_diff = lower_diff - buffer
        upper_diff = upper_diff + buffer
        constraints.append(NonlinearConstraint(self.create_neg_info_constraints, lower_diff, upper_diff, keep_feasible=False))

        # Add the extent normalization constraint
        constraints.append(NonlinearConstraint(self.extent_normalization_constraint, -0.5, 0.5, keep_feasible=False))

        # Ensure length and width are positive (index 6 and 7)
        constraints.append(LinearConstraint(np.eye(self.state_dim)[6:8], 0.1, np.inf))

        # Position constraint
        constraints.append(NonlinearConstraint(self.position_constraint, 0.0, np.inf, keep_feasible=False))

        # Test the 'trust-exact' optimizer to potentially improve convergence and accuracy
        # of the state estimation by minimizing the negative log-posterior.
        res = minimize(self.object_function, state_vector, 
                       args=(z_combined, self.h, R, state_vector, P_pred, ssa_vec, ais_available, ground_truth), #, alpha_min, alpha_max, lidar_pos), 
                       method='SLSQP', constraints=constraints) #, options={'ftol': self.convergence_threshold})

        # Update final state
        self.state = res.x

        state_iterates.append(self.state.copy())

        _, hessian = self.compute_jacobian_hessian_numerical(res.x, z_combined, self.h, R, state_vector, self.P, ssa_vec, ais_available, ground_truth=ground_truth)

        x_dim = self.state_dim

        # cond_number = np.linalg.cond(hessian)
        # print("Condition number of Hessian: ", cond_number, ",  Determinant: ", np.linalg.det(hessian))

        self.P = np.linalg.inv(hessian) # + self.damping_factor * np.eye(hessian.shape[0]))  

        # print("Covariance condition number: ", np.linalg.cond(self.P), ", Determinant: ", np.linalg.det(self.P), "\n")

        return state_iterates, z_combined, y_combined, S_combined, P_pred, self.P, z_dim, x_dim
    
    def create_neg_info_constraints(self, x):
        L = x[6]
        W = x[7]

        epsilon = 1e-6  # Small value to avoid division by zero
        num_angles = 16  # Number of angles to sample
        # Generate 16 angles evenly spaced around the ellipse, avoiding division by zero
        angles = []
        for i in range(num_angles):
            theta = i * (2 * np.pi / num_angles)
            # Avoid exact multiples of pi/2 to prevent division by zero
            L_safe = L + epsilon if np.isclose(np.cos(theta), 0) else L
            W_safe = W + epsilon if np.isclose(np.sin(theta), 0) else W

            angles.append(np.arctan2(W_safe * np.sin(theta), L_safe * np.cos(theta)))
        angles = np.array(angles)

        normalized_angles = np.arctan2(np.sin(angles) / W, np.cos(angles) / L)

        lidar_pos = self.lidar_pos

        pred_meas = self.h(x, normalized_angles, False)
        pred_meas = np.array(pred_meas).reshape(-1, 2)

        # Ensure the angles of prediced measurements are within the range
        pred_angles = np.arctan2(pred_meas[:, 1] - lidar_pos[1], pred_meas[:, 0] - lidar_pos[0])
        pred_angles = np.array([ssa(angle) for angle in pred_angles])

        pred_angles -= self.mean_lidar_measurement_angle 

        return ssa(pred_angles)
    
    def extent_normalization_constraint(self, x):
        """
        Constraint function to ensure the extent of the vessel is normalized.
        """
        L = x[6]
        W = x[7]
        
        epsilon = 1e-6  # Small value to avoid division by zero
        num_angles = 16  # Number of angles to sample
        # Generate 16 angles evenly spaced around the ellipse, avoiding division by zero
        angles = []
        for i in range(num_angles):
            theta = i * (2 * np.pi / num_angles)
            # Avoid exact multiples of pi/2 to prevent division by zero
            L_safe = L + epsilon if np.isclose(np.cos(theta), 0) else L
            W_safe = W + epsilon if np.isclose(np.sin(theta), 0) else W

            angles.append(np.arctan2(W_safe * np.sin(theta), L_safe * np.cos(theta)))
        angles = np.array(angles)
        
        # Normalize the angles
        normalized_angles = np.arctan2(np.sin(angles) / W, np.cos(angles) / L)

        pred_meas = []
        for i in range(len(angles)):
            # Compute the predicted measurement z_pred for each normalized angle
            norm_pred = ur(normalized_angles[i]) * (self.g(normalized_angles[i]).T @ (self.fourier_coeff_mean + self.M @ x[8:].reshape(-1, 1))).item()
            pred_meas.append(norm_pred.flatten())

        pred_meas = np.array(pred_meas)

        return pred_meas.flatten()
    
        # TODO: fix naming

    def position_constraint(self, x):
        """
        Constraint function to ensure the position of the vessel is behind the wall of measurements.
        """
        vessel_pos = x[:2]
        L = x[6]
        W = x[7]

        lidar_pos = np.array(self.lidar_pos)  # Ensure lidar_pos is defined

        relative = vessel_pos - lidar_pos
        angle_est, distance_est = cart2pol(*relative)

        distances = np.array([m[1] for m in self.measurements_polar])
        min_rz = np.min(distances)

        min_xf = max(1.0, min(L / 3, W / 3))
        min_distance = min_rz + min_xf

        return np.array([
            ssa(angle_est - (self.mean_lidar_measurement_angle + self.lower_diff)),
            ssa((self.mean_lidar_measurement_angle + self.upper_diff) - angle_est),
            distance_est - min_distance
        ])