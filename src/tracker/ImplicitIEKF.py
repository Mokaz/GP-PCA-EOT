import numpy as np

from src.senfuslib import MultiVarGauss
from src.tracker.tracker import Tracker
from src.tracker.TrackerUpdateResult import TrackerUpdateResult
from src.utils.tools import rot2D, ssa, initialize_centroid
from src.states.states import State_PCA, LidarScan
from src.dynamics.process_models import Model_PCA_CV
from src.sensors.LidarModel import LidarMeasurementModel
from src.utils.config_classes import Config

class ImplicitIEKF(Tracker):
    """
    Implicit Iterated Extended Kalman Filter (I-IEKF).
    Uses the Implicit Measurement Model constraint g(x, z) = 0.
    
    This implementation projects the constraint onto the surface normals 
    to avoid rank deficiency caused by the lack of tangential information.
    """
    def __init__(self, 
                 dynamic_model: Model_PCA_CV, 
                 lidar_model: LidarMeasurementModel,
                 config: Config,
                 max_iterations=10, 
                 convergence_threshold=1e-6):
        super().__init__(dynamic_model=dynamic_model, sensor_model=lidar_model, config=config)
        
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_initialize_centroid = config.tracker.use_initialize_centroid

    def predict(self):
        self.state_estimate = self.dynamic_model.pred_from_est(self.state_estimate, self.T)

    def update(self, measurements_local: LidarScan, ground_truth: State_PCA = None) -> TrackerUpdateResult:
        polar_measurements = list(zip(measurements_local.angle, measurements_local.range))

        # 1. Prior Setup
        state_prior_mean = self.state_estimate.mean.copy()        
        P_pred = self.state_estimate.cov.copy()
        
        # --- OPTIMIZATION: Precompute Prior Information ---
        # Instead of inverting S (360x360), we will invert the Information Matrix (12x12)
        # This is much more stable.
        P_prior_inv = np.linalg.inv(P_pred)
        
        state_pred = MultiVarGauss(mean=state_prior_mean, cov=P_pred)

        # 2. Measurement Setup
        lidar_pos = self.sensor_model.lidar_position.reshape(2, 1)
        measurements_global_coords = measurements_local + lidar_pos
        z_flat = measurements_global_coords.flatten('F') 
        
        # 3. Initialization
        state_iter_mean = state_prior_mean.copy()
        
        if self.use_initialize_centroid:
            state_iter_mean.pos = initialize_centroid(
                position=state_prior_mean.pos,
                lidar_pos=self.sensor_model.lidar_position,
                measurements=polar_measurements,
                L_est=state_prior_mean.length,
                W_est=state_prior_mean.width
            )

        iterates = [state_iter_mean.copy()]

        # Variables for logging
        H_scal = None
        innovation_scalar = None
        z_pred_2d = None
        Info_Matrix = None # The (J) matrix
        
        # Noise parameters (Scalar R is just sigma^2 * I)
        sigma2 = self.sensor_model.lidar_std_dev**2
        inv_sigma2 = 1.0 / sigma2

        # 4. Iteration Loop
        for i in range(self.max_iterations):
            
            # A. Get Scalar Jacobian
            H_scal, normals, theta_imp = self.sensor_model.get_implicit_matrices(state_iter_mean, measurements_global_coords)
            
            # B. Predict and Calculate Scalar Residual
            z_pred_2d = self.sensor_model.h_from_theta(state_iter_mean, theta_imp).reshape(-1, 2)
            z_meas_2d = measurements_global_coords.T 
            
            resid_2d = z_meas_2d - z_pred_2d
            innovation_scalar = np.sum(resid_2d * normals, axis=1) # Shape (N,)

            # C. Update Step (Using Information Form)
            # We solve: (P^-1 + H^T R^-1 H) * correction = P^-1(x_prior - x_curr) + H^T R^-1 y
            
            # 1. Compute Fisher Information (H^T R^-1 H)
            # Since R is scalar diagonal, this is just (H^T H) / sigma^2
            HtH = H_scal.T @ H_scal
            Fisher_Information = HtH * inv_sigma2
            
            # 2. Compute Total Information Matrix (J)
            # J = P_prior^-1 + Fisher_Info
            # This is 12x12 (for PCA=4). Very easy to invert.
            Info_Matrix = P_prior_inv + Fisher_Information
            
            # 3. Compute the "Right Hand Side" vector for the linear system
            
            # Term from measurement innovation: H^T * R^-1 * y
            term_meas = (H_scal.T @ innovation_scalar) * inv_sigma2
            
            # Term from prior constraint: P^-1 * (x_prior - x_current)
            diff_state = state_prior_mean - state_iter_mean
            diff_state[2] = ssa(diff_state[2])
            term_prior = P_prior_inv @ diff_state
            
            rhs = term_prior + term_meas
            
            # 4. Solve for correction (12x12 solve)
            correction = np.linalg.solve(Info_Matrix, rhs)
            
            # Update state
            state_next = state_iter_mean + correction
            state_next[2] = ssa(state_next[2])
            
            state_iter_mean = state_next
            iterates.append(state_iter_mean.copy())

            # Check convergence
            if np.linalg.norm(correction) < self.convergence_threshold:
                break

        # 5. Final Covariance Update
        # P_post = J^-1 (Inverse of Information Matrix)
        state_post_cov = np.linalg.inv(Info_Matrix)
        
        self.state_estimate = MultiVarGauss(mean=state_iter_mean, cov=state_post_cov)

        # 6. Logging
        z_pred_flat = z_pred_2d.flatten() 
        z_pred_gauss = MultiVarGauss(mean=z_pred_flat, cov=np.eye(len(z_pred_flat))) 
        
        # Approximate R for logging purposes (we didn't use S in the update)
        R_diag = np.eye(len(innovation_scalar)) * sigma2
        innovation_gauss = MultiVarGauss(mean=innovation_scalar, cov=R_diag) 

        return TrackerUpdateResult(
            state_prior=state_pred,
            state_posterior=self.state_estimate,
            measurements=z_flat,
            predicted_measurement=z_pred_gauss,
            iterates=iterates,
            innovation_gauss=innovation_gauss,
            iterations=i + 1,
            H_jacobian=H_scal,
            R_covariance=R_diag,
        )