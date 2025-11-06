import numpy as np
from utils.tools import ssa, ur, rot2D, initialize_centroid
from src.tracker.tracker import Tracker

class UKF(Tracker):
    def __init__(self, process_model, timestep, rng, config):
        super().__init__(process_model, timestep, rng, config)
        
        # UKF parameters
        self.alpha = 1.0
        self.beta = 2
        self.kappa = 0.0
        self.lambda_ = self.alpha ** 2 * (self.state_dim + self.kappa) - self.state_dim

        # Weights
        self.Wm = np.full(2 * self.state_dim + 1, 1 / (2 * (self.state_dim + self.lambda_)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lambda_ / (self.state_dim + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha ** 2 + self.beta)

    def predict(self):
        x = self.state
        sigma_pts = self._generate_sigma_points(x, self.P)
        sigma_pts_pred = np.array([self.dynamic_model(pt.copy(), self.T) for pt in sigma_pts])
        x_pred, self.P = self._unscented_transform_state(sigma_pts_pred, self.Q)

        self.state = x_pred

    def update(self, lidar_measurements_polar, lidar_pos=None, ais_measurements=None, ground_truth=None):
        P_pred = self.P.copy()
        state_iterates = [self.state.copy()]
        z_dim = 2 * len(lidar_measurements_polar)
        x_dim = self.state_dim

        lidar_measurements_polar = np.array(lidar_measurements_polar)

        x = self.state.copy()

        x[:2] = initialize_centroid(x[:2], lidar_pos, lidar_measurements_polar, L_est=x[6], W_est=x[7])

        lidar_measurements = lidar_pos + lidar_measurements_polar[:, 1].reshape(-1, 1) * np.array([np.cos(lidar_measurements_polar[:, 0]), np.sin(lidar_measurements_polar[:, 0])]).T

        sigma_pts = self._generate_sigma_points(x, self.P)

        body_angles = np.arctan2(lidar_measurements[:, 1] - x[1],
                                 lidar_measurements[:, 0] - x[0]) - x[2]
        body_angles = np.vectorize(ssa)(body_angles)

        z_pred_pts = np.array([self.h(pt.copy(), body_angles, False) for pt in sigma_pts]).reshape(-1, z_dim)
        R = np.kron(np.eye(len(body_angles)), self.R_lidar)
        z_pred, S = self._unscented_transform_measurement(z_pred_pts, R)

        P_xz = self._cross_covariance(sigma_pts, x, z_pred_pts, z_pred)

        z = lidar_measurements.flatten()
        y = z - z_pred
        K = P_xz @ np.linalg.inv(S)
        x_update = x + K @ y
        self.P -= K @ S @ K.T
        self.state = x_update
        self.state[2] = ssa(self.state[2])

        state_iterates.append(self.state.copy())

        return state_iterates, z, y, S, P_pred, self.P, z_dim, x_dim

    def _generate_sigma_points(self, x, P):
        sqrt_P = np.linalg.cholesky((self.state_dim + self.lambda_) * P)
        sigma_pts = np.hstack([x[:, None], x[:, None] + sqrt_P, x[:, None] - sqrt_P])
        return sigma_pts.T

    def _unscented_transform_state(self, sigma_pts, noise_cov):
        mean = np.sum(self.Wm[:, None] * sigma_pts, axis=0)
        cov = noise_cov.copy()
        for i in range(sigma_pts.shape[0]):
            diff = sigma_pts[i] - mean
            diff[2] = ssa(diff[2])
            cov += self.Wc[i] * np.outer(diff, diff)
        return mean, cov

    def _unscented_transform_measurement(self, sigma_pts, noise_cov):
        mean = np.sum(self.Wm[:, None] * sigma_pts, axis=0)
        cov = noise_cov.copy()
        for i in range(sigma_pts.shape[0]):
            diff = sigma_pts[i] - mean
            cov += self.Wc[i] * np.outer(diff, diff)
        return mean, cov

    def _cross_covariance(self, X, x_mean, Z, z_mean):
        P_xz = np.zeros((X.shape[1], Z.shape[1]))
        for i in range(X.shape[0]):
            dx = X[i] - x_mean
            dz = Z[i] - z_mean
            dx[2] = ssa(dx[2])  # Normalize heading
            P_xz += self.Wc[i] * np.outer(dx, dz)
        return P_xz

    def get_state(self):
        return self.state

    def get_covariance(self):
        return self.P
