import numpy as np
from pathlib import Path
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from scipy.linalg import block_diag
from queue import Queue

from src.tracker.tracker import Tracker
from src.utils.tools import ssa, cart2pol, ur, initialize_centroid, compute_angle_range, generate_fourier_function
from utils.config_classes import TrackerConfig

class SmoothingSLSQP(Tracker):
    def __init__(self, process_model, timestep: float, rng, max_iterations=10, convergence_threshold=1e-3, config=None):
        """
        Initializes the Smoothing SLSQP tracker.

        Parameters:
        - process_model: The process model to be used.
        - timestep: The time step for the tracker.
        - config: Configuration object containing parameters for the tracker.
        """
        super().__init__(process_model, timestep, rng, config)

        # State initialization
        #self.state = config.state.copy()
        #self.state[:8] = rng.multivariate_normal(mean=self.state[:8], cov=self.P[:8, :8])
        
        # self.damping_factor = 1e-6

        self.window_size = 5
        self.state_window = Queue(maxsize=self.window_size+1)
        self.measurement_window = Queue(maxsize=self.window_size)
        
        self.mean_lidar_measurement_angle = Queue(maxsize=self.window_size)
        self.lower_diff = Queue(maxsize=self.window_size)
        self.upper_diff = Queue(maxsize=self.window_size)

        self.num_angles = 16  # Number of angles to sample for constraints

        self.state_window.put(self.state[:6].copy())

        self.state = np.array([*self.state[6:], *self.state[:6]])
        self.P_prior = self.P[:6,:6]
        self.P = block_diag(self.P[6:,6:], self.P[:6,:6]) 
        # TODO: does not work with initial covariance matrix containing cross-covariance between extent and kinematic state

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
        prev_state = np.array([*self.state[-6:], *self.state[:self.N_extent]])

        new_state = self.dynamic_model(prev_state, self.T)

        if self.state_window.full():
            self.state_window.get()
        self.state_window.put(new_state[:6].copy())

        self.state = np.array([*new_state[6:], *np.concatenate(list(self.state_window.queue)).flatten()])
        
    def update(self, lidar_measurements_polar, lidar_pos, ais_measurements=None, ground_truth=None):
        
        ais_available = ais_measurements is not None

        self.state_dim = self.state.shape[0]

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

        # # Find correct initialization point for vessel position
        self.state[-6:-4] = initialize_centroid(self.state[-6:-4], lidar_pos, lidar_measurements_polar, L_est=self.state[0], W_est=self.state[1])

        # Extract the angles from the measurements
        angles = np.array([ssa(measurement[0]) for measurement in lidar_measurements_polar])

        # Calculate the minimum and maximum angles from the measurements
        lower_diff, upper_diff, mean_lidar_angle = compute_angle_range(angles)

        if self.lower_diff.full():
            self.lower_diff.get()
        self.lower_diff.put(lower_diff)

        if self.upper_diff.full():
            self.upper_diff.get()
        self.upper_diff.put(upper_diff)

        if self.mean_lidar_measurement_angle.full():
            self.mean_lidar_measurement_angle.get()
        self.mean_lidar_measurement_angle.put(mean_lidar_angle)
     
        # LiDAR measurement points in cartesian coordinates and global frame
        lidar_measurements = lidar_pos + lidar_measurements_polar[:, 1].reshape(-1, 1) * np.array([np.cos(lidar_measurements_polar[:, 0]), np.sin(lidar_measurements_polar[:, 0])]).T

        # Get measurement vector
        if ais_available:
            z_combined = np.array([*lidar_measurements.flatten(), *ais_measurements])
        else:
            z_combined = lidar_measurements.flatten()

        # State vector updates
        state_iterates = [self.state.copy()]

        # Add new measurements to the measurement window
        if self.measurement_window.full():
            self.measurement_window.get()
        self.measurement_window.put(z_combined)

        x_window = np.array(list(self.state_window.queue)).flatten()
        z_window = self.measurement_window.queue

        extent_covariance = self.P[:self.N_extent, :self.N_extent]

        unknowns = np.concatenate((self.state[:self.N_extent], x_window))

        state_length = len(unknowns)

        # Calculate weight matrix
        num_total_meas = 0
        num_meas_window = []
        for i in range(len(z_window)):
            lidar_measurements = z_window[i].reshape(-1, 2)
            num_total_meas += len(lidar_measurements)
            num_meas_window.append(len(lidar_measurements))

        I_z = np.eye(num_total_meas)
        I_x = np.eye(len(x_window) // 6 - 1) # cleanup

        if ais_available:
            R = np.block([[np.kron(I_z, self.R_lidar), np.zeros((2*num_meas, 5))],
                          [np.zeros((5, 2*num_meas)), self.R_ais]])
        else:
            R = np.kron(I_z, self.R_lidar)
        
        P_pred = self.P.copy()

        weight_matrix = np.linalg.inv(block_diag(extent_covariance, self.P_prior, np.kron(I_x, self.Q[:6, :6]), R))

        x_prior = unknowns.copy()

        # Create constraints
        constraints = []

        # Add buffer to lower and upper constraint limits
        buffer = np.deg2rad(0.5)
        lower_diff = np.array(list(self.lower_diff.queue)).flatten()
        upper_diff = np.array(list(self.upper_diff.queue)).flatten()
        lower_diff = lower_diff - buffer
        upper_diff = upper_diff + buffer

        lower_diff_array = np.array([val for val in lower_diff for _ in range(self.num_angles)])
        upper_diff_array = np.array([val for val in upper_diff for _ in range(self.num_angles)])
        constraints.append(NonlinearConstraint(self.create_neg_info_constraints, lower_diff_array, upper_diff_array))

        # Add the extent normalization constraint
        constraints.append(NonlinearConstraint(self.extent_normalization_constraint, -0.5, 0.5))

        # Ensure length and width are positive (index 0 and 1)
        constraints.append(LinearConstraint(np.eye(self.state_dim)[:2], 0.1, np.inf))

        # Position constraint
        constraints.append(NonlinearConstraint(self.position_constraint, 0.0, np.inf, keep_feasible=False))

        # Test the 'trust-exact' optimizer to potentially improve convergence and accuracy
        # of the state estimation by minimizing the negative log-posterior.

        # Optimization
        res = minimize(self.object_function, unknowns, 
                       args=(self.h, self.state[:self.N_extent], x_prior, z_window, extent_covariance, weight_matrix, ssa_vec, ais_available), #, alpha_min, alpha_max, lidar_pos), 
                       method='SLSQP', constraints=constraints) #, options={'disp': True})

        # Update final state
        self.state = np.array([*res.x[:self.N_extent], *res.x[self.N_extent:state_length]])

        state_iterates.append(self.state)

        _, hessian = self.compute_jacobian_hessian_numerical(res.x, self.h, self.state[:self.N_extent], x_prior, z_window, extent_covariance, weight_matrix, ssa_vec, ais_available)
       
        x_dim = self.state_dim

        self.P = np.linalg.inv(hessian)

        # Move to the new prior covariance if the window is going to shift
        if self.state_window.full():
            self.P_prior = self.P[(self.N_extent + 6):(self.N_extent + 12), (self.N_extent + 6):(self.N_extent + 12)]
        else:
            self.P_prior = self.P[self.N_extent:(self.N_extent + 6), self.N_extent:(self.N_extent + 6)]
        # TODO: check that this is correct
        
        # cond_number = np.linalg.cond(hessian)
        # print("Condition number of Hessian: ", cond_number)

        return state_iterates, z_combined, y_combined, S_combined, P_pred, self.P, z_dim, x_dim
    
    def object_function(self, x, h, extent_pred, x_prior, z_window, extent_covariance, weight_matrix, ssa_func, ais_received):
        """
        Compute the negative log-posterior for the given state and measurements.
        
        Parameters:
        x (np.array): Current state (n-dimensional).
        z (np.array): Measurement vector (m-dimensional).
        h (function): Measurement model function, h(x).
        R (np.array): Measurement noise covariance (m x m).
        x_pred (np.array): Predicted state (n-dimensional).
        P_pred (np.array): Predicted covariance (n x n).
        
        Returns:
        float: Negative log-posterior value.
        """

        extent_state = x[:self.N_extent]
        kinematic_state = x[self.N_extent:]

        # extent_state_pred = x_prior[:self.N_extent]
        # kinematic_state_pred = x_prior[self.N_extent:]

        kinematic_prior = x_prior[self.N_extent:(self.N_extent + 6)]

        kinematic_pred = x_prior[self.N_extent:-6]

        t = np.array([[1, self.T],
                  [0, 1]])
        F = np.kron(t, np.eye(3))

        x_pred = np.array([*extent_pred, *kinematic_prior])

        state = np.concatenate((extent_state, kinematic_state))

        # Multiply each 6-section of x with F except the first and last
        num_sections = len(kinematic_pred) // 6
        for i in range(num_sections):
            start_idx = i * 6
            end_idx = start_idx + 6
            x_pred = np.concatenate((x_pred, F @ kinematic_pred[start_idx:end_idx]))

        for k in range(len(z_window)):
            lidar_measurements = z_window[k].reshape(-1, 2)

            pose = kinematic_state[(k * 6):((k + 1) * 6)]
            # pose = kinematic_state_pred[(k * 6):((k + 1) * 6)]

            # Calculate body angles
            body_angles = ssa_func(np.arctan2(
                lidar_measurements[:, 1] - pose[1], 
                lidar_measurements[:, 0] - pose[0]
            ) - pose[2])

            state_k = np.array([*pose, *extent_state])

            x_pred = np.concatenate((x_pred, np.array(h(state_k, body_angles, ais_received)).flatten()))
            state = np.concatenate((state, z_window[k]))

        x_residual = state - x_pred

        return x_residual.T @ weight_matrix @ x_residual
    
    def compute_jacobian_hessian_numerical(self, x, h, extent_pred, x_window, z_window, extent_covariance, weight_matrix, ssa_func, ais_received, epsilon=1e-3):
        """
        Numerically compute Jacobian and Hessian of the negative log-posterior.
        """

        n = len(x)
        J = np.zeros(n)
        H = np.zeros((n, n))

        # Compute gradient (Jacobian)
        for i in range(n):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += epsilon
            x2[i] -= epsilon
            J[i] = (self.object_function(x1, h, extent_pred, x_window, z_window, extent_covariance, weight_matrix, ssa_func, ais_received) - self.object_function(x2, h, extent_pred, x_window, z_window, extent_covariance, weight_matrix, ssa_func, ais_received)) / (2 * epsilon)

        # Compute Hessian
        for i in range(n):
            for j in range(n):
                x_ijp = x.copy()
                x_ijp[i] += epsilon
                x_ijp[j] += epsilon
                
                x_ijm = x.copy()
                x_ijm[i] -= epsilon
                x_ijm[j] -= epsilon

                x_ipjm = x.copy()
                x_ipjm[i] += epsilon
                x_ipjm[j] -= epsilon

                x_imjp = x.copy()
                x_imjp[i] -= epsilon
                x_imjp[j] += epsilon

                H[i, j] = (self.object_function(x_ijp, h, extent_pred, x_window, z_window, extent_covariance, weight_matrix, ssa_func, ais_received) - self.object_function(x_ipjm, h, extent_pred, x_window, z_window, extent_covariance, weight_matrix, ssa_func, ais_received)
                        - self.object_function(x_imjp, h, extent_pred, x_window, z_window, extent_covariance, weight_matrix, ssa_func, ais_received) + self.object_function(x_ijm, h, extent_pred, x_window, z_window, extent_covariance, weight_matrix, ssa_func, ais_received)) / (4 * epsilon ** 2)

        return J, H
    
    def create_neg_info_constraints(self, x):
        L = x[0]
        W = x[1]

        epsilon = 1e-6  # Small value to avoid division by zero
        # Generate 16 angles evenly spaced around the ellipse, avoiding division by zero
        angles = []
        for i in range(self.num_angles):
            theta = i * (2 * np.pi / self.num_angles)
            # Avoid exact multiples of pi/2 to prevent division by zero
            L_safe = L + epsilon if np.isclose(np.cos(theta), 0) else L
            W_safe = W + epsilon if np.isclose(np.sin(theta), 0) else W

            angles.append(np.arctan2(W_safe * np.sin(theta), L_safe * np.cos(theta)))
        angles = np.array(angles)

        normalized_angles = np.arctan2(np.sin(angles) / W, np.cos(angles) / L)

        total_pred_angles = np.array([])
        for i in range(len(self.measurement_window.queue)):
            lidar_pos = self.lidar_pos
            kinematic_state = self.state_window.queue[i+1]

            state_k = np.array([*kinematic_state, *x[:self.N_extent]])

            pred_meas = self.h(state_k, normalized_angles, False)
            pred_meas = np.array(pred_meas).reshape(-1, 2)

            # Ensure the angles of prediced measurements are within the range
            pred_angles = np.arctan2(pred_meas[:, 1] - lidar_pos[1], pred_meas[:, 0] - lidar_pos[0])
            pred_angles = np.array([ssa(angle) for angle in pred_angles])

            total_pred_angles = np.append(total_pred_angles, pred_angles - self.mean_lidar_measurement_angle.queue[i]) 

        return ssa(total_pred_angles)
    
    def extent_normalization_constraint(self, x):
        """
        Constraint function to ensure the extent of the vessel is normalized.
        """
        L = x[0]
        W = x[1]
        
        epsilon = 1e-6  # Small value to avoid division by zero
        # Generate 16 angles evenly spaced around the ellipse, avoiding division by zero
        angles = []
        for i in range(self.num_angles):
            theta = i * (2 * np.pi / self.num_angles)
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
            norm_pred = ur(normalized_angles[i]) * (self.g(normalized_angles[i]).T @ (self.fourier_coeff_mean + self.M @ x[2:self.N_extent].reshape(-1, 1))).item()
            pred_meas.append(norm_pred.flatten())

        pred_meas = np.array(pred_meas)

        return pred_meas.flatten()
    
        # TODO: fix naming

    def position_constraint(self, x):
        """
        Constraint function to ensure the position of the vessel is behind the wall of measurements.
        """
        constraints = np.array([])

        for i in range(len(self.state_window.queue)-1):
            lidar_pos = np.array(self.lidar_pos)
            kinematic_state = self.state_window.queue[i+1]

            vessel_pos = kinematic_state[:2]
            L = x[0]
            W = x[1]

            relative = vessel_pos - lidar_pos
            angle_est, distance_est = cart2pol(*relative)

            measurements = np.array(self.measurement_window.queue[i]).reshape(-1, 2)
            distances = np.linalg.norm(measurements - lidar_pos, axis=1, ord=2)
            min_rz = np.min(distances)

            min_xf = max(1.0, min(L / 3, W / 3))
            min_distance = min_rz + min_xf

            constraints = np.append(constraints, 
                                          (ssa(angle_est - (self.mean_lidar_measurement_angle.queue[i] + self.lower_diff.queue[i])),
                                          ssa((self.mean_lidar_measurement_angle.queue[i] + self.upper_diff.queue[i]) - angle_est),
                                          distance_est - min_distance))

        return constraints

