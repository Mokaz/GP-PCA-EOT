import numpy as np

from src.senfuslib import MultiVarGauss
from src.tracker.tracker import Tracker
from src.tracker.TrackerUpdateResult import TrackerUpdateResult
from src.utils.tools import ssa, initialize_centroid, calculate_body_angles
from src.states.states import State_PCA, LidarScan
from src.dynamics.process_models import Model_PCA_CV
from src.sensors.LidarModel import LidarMeasurementModel
from src.utils.config_classes import Config

class IterativeEKF(Tracker):
    def __init__(self, 
                 dynamic_model: Model_PCA_CV, 
                 lidar_model: LidarMeasurementModel,
                 config: Config,
                 max_iterations=10, 
                 convergence_threshold=1e-6):
        super().__init__(dynamic_model=dynamic_model, sensor_model=lidar_model, config=config)
        
        # Parameters
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def predict(self):
        """
        Prediction step.
        Updates the internal state estimate.
        """
        self.state_estimate = self.dynamic_model.pred_from_est(self.state_estimate, self.T)

    def update(self, measurements_local: LidarScan, ground_truth: State_PCA = None) -> TrackerUpdateResult:
        polar_measurements = list(zip(measurements_local.angle, measurements_local.range))

        # Initialize the centroid for the optimization
        state_prior_mean = self.state_estimate.mean.copy()        
        P_pred = self.state_estimate.cov.copy()
        state_pred = MultiVarGauss(mean=state_prior_mean, cov=P_pred)

        measurements_global_coords = measurements_local + self.sensor_model.lidar_position.reshape(2, 1)
        
        z = measurements_global_coords.flatten('F')

        body_angles_prior = calculate_body_angles(
            measurements_global_coords, 
            ground_truth if self.use_gt_state_for_bodyangles_calc else state_prior_mean
        )
        z_pred = self.sensor_model.h_lidar(state_prior_mean, body_angles_prior).flatten()
        innovation = z - z_pred

        state_iter_mean = state_prior_mean.copy()
        state_iter_mean.pos = initialize_centroid(
            position=state_prior_mean.pos,
            lidar_pos=self.sensor_model.lidar_position,
            measurements=polar_measurements,
            L_est=state_prior_mean.length,
            W_est=state_prior_mean.width
        )

        prev_state_iter_mean = state_prior_mean.copy()

        for i in range(self.max_iterations):
            self.body_angles = calculate_body_angles(measurements_global_coords, ground_truth if self.use_gt_state_for_bodyangles_calc else state_iter_mean)

            # Predict LiDAR measurement
            z_pred_iter = self.sensor_model.h_lidar(state_iter_mean, self.body_angles).flatten()
            innovation_iter = z - z_pred_iter

            num_meas = len(self.body_angles) 
            
            # Compute Jacobian and Kalman gain for LiDAR
            H_lidar = self.sensor_model.lidar_jacobian(state_iter_mean, self.body_angles)
            R_lidar = self.sensor_model.R(num_meas)
            S_lidar = H_lidar @ P_pred @ H_lidar.T + R_lidar
            K_lidar = np.linalg.solve(S_lidar.T, (H_lidar @ P_pred.T)).T

            # Update state mean
            state_iter_mean = state_prior_mean + K_lidar @ (innovation_iter + H_lidar @ (state_iter_mean - state_prior_mean))
            state_iter_mean.yaw = ssa(state_iter_mean.yaw)

            # Check for convergence
            diff = state_iter_mean - prev_state_iter_mean
            diff.yaw = ssa(diff.yaw)
            if np.linalg.norm(diff) < self.convergence_threshold:
                break
            
            prev_state_iter_mean = state_iter_mean.copy()

        state_post_mean = state_iter_mean

        # Recalculate for final state
        self.body_angles = calculate_body_angles(measurements_global_coords, ground_truth if self.use_gt_state_for_bodyangles_calc else state_post_mean)
        
        num_meas = len(self.body_angles) 
        H_lidar = self.sensor_model.lidar_jacobian(state_post_mean, self.body_angles)
        R_lidar = self.sensor_model.R(num_meas)
        S_lidar = H_lidar @ P_pred @ H_lidar.T + R_lidar
        K_lidar = np.linalg.solve(S_lidar.T, (H_lidar @ P_pred.T)).T

        # Update covariance (Joseph form for stability)
        I = np.eye(len(state_post_mean))
        state_post_cov = (I - K_lidar @ H_lidar) @ P_pred @ (I - K_lidar @ H_lidar).T + K_lidar @ R_lidar @ K_lidar.T
        
        self.state_estimate = MultiVarGauss(mean=state_post_mean, cov=state_post_cov)

        z_pred_gauss = MultiVarGauss(mean=z_pred, cov=S_lidar)
        innovation_gauss = MultiVarGauss(mean=innovation, cov=S_lidar)

        return TrackerUpdateResult(
            state_prior=state_pred,
            state_posterior=self.state_estimate,
            measurements=z,
            predicted_measurement=z_pred_gauss,
            innovation_gauss=innovation_gauss,
            iterations=i + 1,
            H_jacobian=H_lidar,
            R_covariance=R_lidar,
        )

