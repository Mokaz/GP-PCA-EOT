import numpy as np

from src.senfuslib import MultiVarGauss
from src.tracker.tracker import Tracker
from src.tracker.TrackerUpdateResult import TrackerUpdateResult
from src.utils.tools import ssa, calculate_body_angles
from src.states.states import State_PCA, LidarScan
from src.dynamics.process_models import Model_PCA_CV
from src.sensors.LidarModel import LidarModel
from src.utils.config_classes import Config

class EKF(Tracker):
    def __init__(self, 
                 dynamic_model: Model_PCA_CV, 
                 lidar_model: LidarModel,
                 config: Config):
        super().__init__(dynamic_model, sensor_model=lidar_model, config=config)

    def predict(self):
        """
        Prediction step.
        Updates the internal state estimate.
        """
        self.state_estimate = self.dynamic_model.pred_from_est(self.state_estimate, self.T)

    def update(self, measurements_local: LidarScan, ground_truth: State_PCA = None) -> TrackerUpdateResult:
        """
        Update step of the Extended Kalman Filter
        """
        state_prior_mean = self.state_estimate.mean.copy()
        P_pred = self.state_estimate.cov.copy()
        state_pred = MultiVarGauss(mean=state_prior_mean, cov=P_pred) 

        measurements_global_coords = measurements_local + self.sensor_model.lidar_position.reshape(2, 1)
        
        z = measurements_global_coords.flatten('F')

        self.body_angles = calculate_body_angles(measurements_global_coords, ground_truth if self.use_gt_state_for_bodyangles_calc else state_prior_mean)
        
        # Predict LiDAR measurement
        z_pred = self.sensor_model.h_lidar(state_prior_mean, self.body_angles).flatten()
        innovation = z - z_pred
        
        # Compute Jacobian and Kalman gain for LiDAR
        num_meas = len(self.body_angles)
        H_lidar = self.sensor_model.lidar_jacobian(state_prior_mean, self.body_angles)
        R_lidar = self.sensor_model.R(num_meas)
        S_lidar = H_lidar @ P_pred @ H_lidar.T + R_lidar
        K_lidar = np.linalg.solve(S_lidar.T, (H_lidar @ P_pred.T)).T
        
        # Update internal state estimate
        state_post_mean = state_prior_mean + K_lidar @ innovation
        state_post_mean.yaw = ssa(state_post_mean.yaw)  # Normalize heading angle
        I = np.eye(len(state_prior_mean))
        state_post_cov = (I - K_lidar @ H_lidar) @ P_pred @ (I - K_lidar @ H_lidar).T + K_lidar @ R_lidar @ K_lidar.T
        self.state_estimate = MultiVarGauss(mean=state_post_mean, cov=state_post_cov)

        z_pred_gauss = MultiVarGauss(mean=z_pred, cov=S_lidar) # Inserting S_k as covariance as expected by ConsistencyAnalysis
        innovation_gauss = MultiVarGauss(mean=innovation, cov=S_lidar)

        return TrackerUpdateResult(
            state_prior=state_pred,
            state_posterior=self.state_estimate,
            measurements=z,
            predicted_measurement=z_pred_gauss,
            innovation_gauss=innovation_gauss,
        )