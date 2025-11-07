import numpy as np
from pathlib import Path
from utils.tools import generate_fourier_function, ur, ut, rot2D, drot2D, initialize_centroid, compute_angle_range
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from src.utils.tools import ssa 
from src.utils.config_classes import Config

from src.states.states import State_PCA
from sensors.LidarModel import LidarModel
from src.dynamics.process_models import Model_PCA_CV

from src.senfuslib.gaussian import MultiVarGauss

from dataclasses import dataclass
from typing import Any

class Tracker:
    def __init__(self, dynamic_model: Model_PCA_CV, sensor_model: LidarModel, config: Config):

        self.dynamic_model: Model_PCA_CV = dynamic_model
        self.sensor_model: LidarModel = sensor_model
        self.T: float = config.sim.dt
        self.N_pca = config.tracker.N_pca
        self.N_extent = 2 + self.N_pca # NOTE Only smoothing_SLSQP uses this

        # Extent and Fourier parameters
        PCA_parameters = np.load(Path(config.tracker.PCA_parameters_path))
        PCA_eigenvalues = PCA_parameters['eigenvalues'][:self.N_pca].real

        # Initialize state estimate and covariance
        initial_mean: State_PCA = config.tracker.initial_state
        
        kinematic_extent_std_devs = config.tracker.initial_std_devs.copy()
        kinematic_extent_variances = kinematic_extent_std_devs[:8]**2
        initial_cov_diag = np.concatenate([
            kinematic_extent_variances,
            PCA_eigenvalues
        ])
        initial_cov = np.diag(initial_cov_diag)

        self.state_estimate = MultiVarGauss(mean=initial_mean, cov=initial_cov)
        self.body_angles: np.ndarray = None

    def predict(self):
        raise NotImplementedError("Predict method not implemented for the Tracker class.")

    def update(self):
        raise NotImplementedError("Update method not implemented for the Tracker class.")

    def jacobian(self, x, body_angles: list[float]): # Used in GN and LM only
        return self.sensor_model.lidar_jacobian(x, body_angles)

    def object_function(self, x, x_pred, P_pred, z, ground_truth=None):
        """
        Compute the negative log-posterior for the given state and measurements.
        
        Parameters:
        x (np.array): Current state (n-dimensional).
        z (np.array): Measurement vector (m-dimensional).
        
        Returns:
        float: Negative log-posterior value.
        """
        num_measurements = len(self.body_angles)
        R = self.sensor_model.R(num_measurements)

        # Residuals
        z_residual = z - self.sensor_model.h_lidar(x, self.body_angles).flatten()
        x_residual = x - x_pred
        
        # Negative log of each term
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
            J[i] = (self.object_function(x1, z, h, R, x_pred, P_pred, ground_truth) 
                    - self.object_function(x2, z, h, R, x_pred, P_pred, ground_truth)) / (2 * epsilon)

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

                H[i, j] = (self.object_function(x_ijp, z, h, R, x_pred, P_pred, ground_truth) 
                           - self.object_function(x_ipjm, z, h, R, x_pred, P_pred, ground_truth)
                           - self.object_function(x_imjp, z, h, R, x_pred, P_pred, ground_truth) 
                           + self.object_function(x_ijm, z, h, R, x_pred, P_pred, ground_truth)) / (4 * epsilon ** 2)
        return J, H

@dataclass
class TrackerUpdateResult:
    """
    Holds all relevant data from a single tracker update step.
    """
    # Core filter states
    estimate_prior: MultiVarGauss       # State estimate before the update (x_k|k-1)
    estimate_posterior: MultiVarGauss   # State estimate after the update (x_k|k)

    # Measurement and Innovation
    measurements: np.ndarray             # The raw measurement vector used (z_k)
    innovation_gauss: MultiVarGauss     # The innovation (y_k) and its covariance (S_k)

    # Optional Debugging / Analysis Info
    iterations: int = None              # For iterative optimizers
    cost: float = None                  # Final value of the objective function
    raw_optimizer_result: Any = None    # The full result object from scipy.minimize