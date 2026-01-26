import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

from src.utils.config_classes import Config
from src.states.states import State_PCA, State_GP 
from src.senfuslib.gaussian import MultiVarGauss

from src.tracker.TrackerUpdateResult import TrackerUpdateResult

class Tracker:
    def __init__(self, dynamic_model, sensor_model, config: Config):
        self.dynamic_model = dynamic_model
        self.sensor_model = sensor_model
        self.config = config
        self.T: float = config.sim.dt
        
        self.use_gt_state_for_bodyangles_calc = config.tracker.use_gt_state_for_bodyangles_calc

        initial_mean = config.tracker.initial_state
        initial_std_devs = config.tracker.initial_std_devs

        if initial_mean is None or initial_std_devs is None:
            raise ValueError("Tracker configuration missing initial_state or initial_std_devs.")

        # Gaussian Process Initialization
        if isinstance(initial_mean, State_GP):
            std_devs_arr = np.array(initial_std_devs).flatten()
            
            # Safety check
            if len(std_devs_arr) != len(initial_mean):
                raise ValueError(f"State_GP size ({len(initial_mean)}) does not match initial_std_devs size ({len(std_devs_arr)})")
            
            initial_cov = np.diag(std_devs_arr**2)

        # PCA Initialization
        else:
            self.N_pca = config.tracker.N_pca 
            PCA_parameters = np.load(Path(config.tracker.PCA_parameters_path))
            PCA_eigenvalues = PCA_parameters['eigenvalues'][:self.N_pca].real
            
            kinematic_extent_std_devs = np.array(initial_std_devs).flatten()
            kinematic_extent_variances = kinematic_extent_std_devs[:8]**2
            
            initial_cov_diag = np.concatenate([
                kinematic_extent_variances,
                PCA_eigenvalues
            ])
            initial_cov = np.diag(initial_cov_diag)

        self.state_estimate = MultiVarGauss(mean=initial_mean, cov=initial_cov)
        self.body_angles: np.ndarray = None

    def get_initial_update_result(self) -> "TrackerUpdateResult":
        """
        Creates a TrackerUpdateResult for the initial state (t=0).
        """
        return TrackerUpdateResult(
            state_prior=self.state_estimate, # Best guess at t=0
            state_posterior=self.state_estimate, # This is the initial state x_0|0
            measurements=None,
            predicted_measurement=None,
            innovation_gauss=None
        )

    def predict(self):
        raise NotImplementedError("Predict method not implemented for the Tracker class.")

    def update(self):
        raise NotImplementedError("Update method not implemented for the Tracker class.")

    def jacobian(self, x, body_angles: list[float]): # Used in GN and LM only
        return self.sensor_model.lidar_jacobian(x, body_angles)

    def objective_function(self, x, x_pred, P_pred, z, ground_truth=None):
        """
        Compute the negative log-posterior for the given state and measurements.
        """
        assert self.body_angles is not None, "Body angles must be set before computing the object function."

        num_measurements = len(self.body_angles)
        R = self.sensor_model.R(num_measurements)

        # Residuals
        z_residual = z - self.sensor_model.h_lidar(x, self.body_angles).flatten()
        x_residual = x - x_pred
        
        # Negative log of each term
        # Note: Using solve or inv? Inv is expensive but explicit here.
        term1 = 0.5 * z_residual.T @ np.linalg.inv(R) @ z_residual
        term2 = 0.5 * x_residual.T @ np.linalg.inv(P_pred) @ x_residual
        
        return term1 + term2
    
    # NOTE Martin: Used by SLSQP and smoothing_SLSQP
    def compute_jacobian_hessian_numerical(self, x, z, h, R, x_pred, P_pred, ground_truth=None, epsilon=1e-3):
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
            J[i] = (self.objective_function(x1, z, h, R, x_pred, P_pred, ground_truth) 
                    - self.objective_function(x2, z, h, R, x_pred, P_pred, ground_truth)) / (2 * epsilon)

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

                H[i, j] = (self.objective_function(x_ijp, z, h, R, x_pred, P_pred, ground_truth) 
                           - self.objective_function(x_ipjm, z, h, R, x_pred, P_pred, ground_truth)
                           - self.objective_function(x_imjp, z, h, R, x_pred, P_pred, ground_truth) 
                           + self.objective_function(x_ijm, z, h, R, x_pred, P_pred, ground_truth)) / (4 * epsilon ** 2)
        return J, H