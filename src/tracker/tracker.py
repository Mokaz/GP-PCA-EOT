import numpy as np
from pathlib import Path
from utils.tools import ssa, generate_fourier_function, ur, ut, rot2D, drot2D, initialize_centroid, compute_angle_range
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

class Tracker:
    def __init__(self, process_model, timestep, rng, config=None):
        
        self.N_pca = config.N_pca
        self.N_extent = 2 + self.N_pca

        self.state_dim = 8 + self.N_pca
        self.T = timestep

        # Extent and Fourier parameters
        PCA_parameters = np.load(Path('data/input_parameters/FourierPCAParameters_scaled.npz'))
        self.g = generate_fourier_function(N_f=64)
        self.fourier_coeff_mean = PCA_parameters['mean']
        self.M = PCA_parameters['eigenvectors'][:, :self.N_pca].real
        PCA_eigenvalues = PCA_parameters['eigenvalues'][:self.N_pca].real

        # Standard deviations
        pos_north_std_dev = config.pos_north_std_dev
        pos_east_std_dev = config.pos_east_std_dev
        heading_std_dev = config.heading_std_dev

        lidar_std_dev = config.lidar_std_dev

        ais_pos_std_dev = config.ais_pos_std_dev
        ais_heading_std_dev = config.ais_heading_std_dev
        ais_length_std_dev = config.ais_length_std_dev
        ais_width_std_dev = config.ais_width_std_dev

        # Process and measurement covariances
        t = np.array([[(self.T**3)/3, self.T**2/2],
                     [self.T**2/2, self.T]])
        Qk = np.kron(t, np.diag([pos_north_std_dev**2, pos_east_std_dev**2, heading_std_dev**2]))
        Qe = np.zeros((2+self.N_pca, 2+self.N_pca))

        self.Q = np.block([[Qk, np.zeros((6, 2+self.N_pca))], 
                           [np.zeros((2+self.N_pca, 6)), Qe]])
        
        self.P = np.diag([2.0**2, 2.0**2, 0.2**2, 2.0**2, 2.0**2, 0.1**2, 2.0**2, 2.0**2, *PCA_eigenvalues])

        # LiDAR
        self.R_lidar = (lidar_std_dev**2) * np.eye(2)

        # AIS
        self.R_ais = np.diag([ais_pos_std_dev**2, ais_pos_std_dev**2, ais_heading_std_dev**2, ais_length_std_dev**2, ais_width_std_dev**2])

        # State initialization
        self.state = config.state.copy()
        #self.state[:8] = rng.multivariate_normal(mean=self.state[:8], cov=self.P[:8, :8])

        self.process_model = process_model  # State transition matrix

        # Sensor position
        self.lidar_pos = config.lidar_pos

    def predict(self):
        raise NotImplementedError("Predict method not implemented for the Tracker class.")

    def update(self):
        raise NotImplementedError("Update method not implemented for the Tracker class.")

    def h_lidar(self, x, body_angles: list[float]):
        L = x[6]
        W = x[7]
        
        # Normalize the angles
        normalized_angles = np.arctan2(np.sin(body_angles) / W, np.cos(body_angles) / L)

        # Precompute values that don’t change inside the loop
        pos = x[:2].reshape(-1, 1)  # shape (2, 1)
        R_heading = rot2D(x[2])    # shape (2, 2)
        LW_scaling = np.diag([L, W])  # shape (2, 2)

        fourier_coeffs = self.fourier_coeff_mean + self.M @ x[8:].reshape(-1, 1)  # shape (N_f, 1)

        # Vectorized direction vectors: shape (2, N) -- one column per angle
        ur_all = ur(normalized_angles)  # shape (2, N) if ur returns [cos(θ); sin(θ)]

        # Vectorized g(theta): shape (N_f, N)
        g_all = self.g(normalized_angles)  # shape (N_f, N)  

        # Compute predicted measurement points in global frame
        #   ur_all * (g_all.T @ fourier_coeffs): shape (2, N)
        #   (g_all.T @ fourier_coeffs): shape (N,)
        #   full expression: shape (2, N) → pos + ... → shape (2, N)
        z_pred = np.squeeze(pos + R_heading @ LW_scaling @ ur_all * (g_all.T @ fourier_coeffs).flatten())

        return z_pred.T  # shape (N, 2)
    
    def h_ais(self, x):
        return np.array([*x[:3], *x[6:8]])
    
    def h(self, x, body_angles: list[float], is_ais_available):
        if is_ais_available:
            z_pred_lidar = self.h_lidar(x, body_angles)
            z_pred_ais = self.h_ais(x)
            return np.vstack([z_pred_lidar, z_pred_ais])
        else:
            return self.h_lidar(x, body_angles)

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

        fourier_coeffs = self.fourier_coeff_mean + self.M @ x[8:].reshape(-1, 1)

        jacobians = []

        for angle in normalized_angles:
            fourier_approx = (self.g(angle).T @ fourier_coeffs).item()

            dp_dpc = np.eye(2)
            dp_dphi = drot2D(x[2]) @ np.diag([L, W]) @ (ur(angle) * fourier_approx).reshape(-1, 1)
            dp_dv = np.zeros([2, 2])
            dp_dr = np.zeros([2, 1])
            dp_dLW = rot2D(x[2]) @ np.diag(ur(angle).flatten()) * fourier_approx
            dp_de = rot2D(x[2]) @ np.diag([L, W]) @ ur(angle).T * (self.g(angle).T @ self.M).flatten()

            H_i = np.hstack([dp_dpc, dp_dphi, dp_dv, dp_dr, dp_dLW, dp_de])
            jacobians.append(H_i)

        return np.vstack(jacobians)

    def ais_jacobian(self):
        H_ais = np.zeros((5, 8 + self.N_pca))
        H_ais[0, 0] = 1  # North position
        H_ais[1, 1] = 1  # East position
        H_ais[2, 2] = 1  # Heading
        H_ais[3, 6] = 1  # Length
        H_ais[4, 7] = 1  # Width

        return H_ais
    
    def jacobian(self, x, body_angles: list[float], is_ais_available=False):
        if is_ais_available:
            H_lidar = self.lidar_jacobian(x, body_angles)
            H_ais = self.ais_jacobian()
            return np.vstack([H_lidar, H_ais])
        else:
            return self.lidar_jacobian(x, body_angles)
        
    def object_function(self, x, z, h, R, x_pred, P_pred, ssa_func, ais_received=False, ground_truth=None):
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
        # Get lidar
        if ais_received:
            lidar_measurements = z[:-5].reshape(-1, 2)
        else:
            lidar_measurements = z.reshape(-1, 2)

        # Calculate body angles
        body_angles = ssa_func(np.arctan2(
            lidar_measurements[:, 1] - ground_truth[1], 
            lidar_measurements[:, 0] - ground_truth[0]
        ) - ground_truth[2])

        # Residuals
        z_residual = z - h(x, body_angles, ais_received).flatten()
        x_residual = x - x_pred
        
        # Negative log of each term
        term1 = 0.5 * z_residual.T @ np.linalg.inv(R) @ z_residual
        term2 = 0.5 * x_residual.T @ np.linalg.inv(P_pred) @ x_residual
        
        return term1 + term2
    
    def compute_jacobian_hessian(self, x, z, h, R, x_pred, P_pred, ssa_func, ais_received, ground_truth=None):
        """
        Compute the Jacobian and Hessian of the negative log-posterior function.
        
        Parameters:
        x (np.array): Current state (n-dimensional).
        z (np.array): Measurement vector (m-dimensional).
        h (function): Measurement model function, h(x).
        R (np.array): Measurement noise covariance (m x m).
        x_pred (np.array): Predicted state (n-dimensional).
        P_pred (np.array): Predicted covariance (n x n).
        ssa_func (function): Function to normalize angles.
        ais_received (bool): Flag indicating if AIS data is received.
        
        Returns:
        J (np.array): Jacobian vector (n-dimensional).
        H (np.array): Hessian matrix (n x n).
        """
        # Get lidar measurements
        if ais_received:
            lidar_measurements = z[:-5].reshape(-1, 2)
        else:
            lidar_measurements = z.reshape(-1, 2)

        # Compute body angles
        body_angles = ssa_func(np.arctan2(
            lidar_measurements[:, 1] - x[1], 
            lidar_measurements[:, 0] - x[0]
        ) - x[2])
        
        # Compute measurement residuals
        z_expected = np.array(h(x, body_angles, ais_received)).flatten()
        z_residual = z - z_expected
        x_residual = x - x_pred

        P_inv = np.linalg.inv(P_pred)
        R_inv = np.linalg.inv(R)
        
        # Compute Jacobian of measurement function
        H_x = self.lidar_jacobian(x, body_angles)

        # Compute Jacobian
        J = 0.5 * P_inv @ x_residual - 0.5 * H_x.T @ R_inv @ z_residual
        
        # Compute Hessian
        H = 0.5 * P_inv + 0.5 * H_x.T @ R_inv @ H_x

        return J, H
    
    def compute_jacobian_hessian_numerical(self, x, z, h, R, x_pred, P_pred, ssa_func, ais_received, ground_truth=None, epsilon=1e-3):
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
            J[i] = (self.object_function(x1, z, h, R, x_pred, P_pred, ssa_func, ais_received, ground_truth) 
                    - self.object_function(x2, z, h, R, x_pred, P_pred, ssa_func, ais_received, ground_truth)) / (2 * epsilon)

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

                H[i, j] = (self.object_function(x_ijp, z, h, R, x_pred, P_pred, ssa_func, ais_received, ground_truth) 
                           - self.object_function(x_ipjm, z, h, R, x_pred, P_pred, ssa_func, ais_received, ground_truth)
                           - self.object_function(x_imjp, z, h, R, x_pred, P_pred, ssa_func, ais_received, ground_truth) 
                           + self.object_function(x_ijm, z, h, R, x_pred, P_pred, ssa_func, ais_received, ground_truth)) / (4 * epsilon ** 2)

        return J, H

    def get_state(self):
        """
        Returns the current state estimate.
        """
        return self.state

    def get_state_covariance(self):
        """
        Returns the current state covariance estimate.
        """
        return self.P

