import numpy as np

from src.senfuslib import MultiVarGauss
from src.tracker.tracker import Tracker, TrackerUpdateResult
from src.utils.tools import ssa, initialize_centroid, calculate_body_angles, cart2pol, pol2cart
from src.states.states import State_GP, LidarScan
from src.dynamics.process_models import Model_GP_CV
from src.sensors.LidarModelGP import LidarModelGP
from src.utils.config_classes import Config

class GP_IEKF(Tracker):
    """
    Iterative EKF specialized for Gaussian Process Extended Object Tracking.
    Standalone implementation to handle State_GP specific attributes.
    """
    def __init__(self, 
                 dynamic_model: Model_GP_CV, 
                 lidar_model: LidarModelGP, 
                 config: Config,
                 max_iterations=10, 
                 convergence_threshold=1e-6,
                 use_negative_info: bool = True):
        
        super().__init__(dynamic_model=dynamic_model, sensor_model=lidar_model, config=config)
        
        # Parameters
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_negative_info = use_negative_info

    def predict(self):
        """
        Prediction step.
        """
        self.state_estimate = self.dynamic_model.pred_from_est(self.state_estimate, self.T)

    def update(self, measurements_local: LidarScan, ground_truth: State_GP = None) -> TrackerUpdateResult:
        """
        Update step for GP State.
        """
        # 1. Prediction State
        state_prior_mean = self.state_estimate.mean.copy()        
        P_pred = self.state_estimate.cov.copy()
        state_pred = MultiVarGauss(mean=state_prior_mean, cov=P_pred)

        # 2. Negative Information (Virtual Measurements)
        if self.use_negative_info and len(measurements_local.x) > 2:
            measurements_augmented = self._augment_with_negative_info(
                measurements_local, state_prior_mean
            )
        else:
            measurements_augmented = measurements_local

        # 3. Prepare for Iteration
        measurements_global_coords = measurements_augmented + self.sensor_model.lidar_position.reshape(2, 1)
        z = measurements_global_coords.flatten('F')
        
        polar_measurements = list(zip(measurements_augmented.angle, measurements_augmented.range))

        # 4. Initialize Centroid
        # For State_GP, we don't have explicit Length/Width. 
        # We approximate L/W from the max radius of the current estimate for the centroid logic.
        max_radius = np.max(state_prior_mean.radii)
        L_est_approx = 2.0 * max_radius
        W_est_approx = 2.0 * max_radius

        state_iter_mean = state_prior_mean.copy()
        state_iter_mean.pos = initialize_centroid(
            position=state_prior_mean.pos,
            lidar_pos=self.sensor_model.lidar_position,
            measurements=polar_measurements,
            L_est=L_est_approx, 
            W_est=W_est_approx
        )

        prev_state_iter_mean = state_prior_mean.copy()
        i = 0

        # 5. Iterative Update Loop
        for i in range(self.max_iterations):
            # Calculate body angles based on current iteration's position/heading
            self.body_angles = calculate_body_angles(
                measurements_global_coords, 
                ground_truth if self.use_gt_state_for_bodyangles_calc else state_iter_mean
            )

            # Predict Measurement h(x)
            z_pred_iter = self.sensor_model.h_lidar(state_iter_mean, self.body_angles).flatten()
            innovation_iter = z - z_pred_iter

            num_meas = len(self.body_angles) 
            
            # Jacobian H
            H_lidar = self.sensor_model.lidar_jacobian(state_iter_mean, self.body_angles)
            
            # Measurement Noise R
            R_lidar = self.sensor_model.R(num_meas)
            
            # Kalman Gain K
            S_lidar = H_lidar @ P_pred @ H_lidar.T + R_lidar
            K_lidar = np.linalg.solve(S_lidar.T, (H_lidar @ P_pred.T)).T

            # Gauss-Newton Update Step
            # x_{i+1} = x_pred + K * ( z - h(x_i) - H * (x_pred - x_i) )
            # Note: innovation_iter is (z - h(x_i))
            correction = H_lidar @ (state_iter_mean - state_prior_mean)
            state_iter_mean = state_prior_mean + K_lidar @ (innovation_iter + correction)
            
            # Normalize Heading
            state_iter_mean.yaw = ssa(state_iter_mean.yaw)

            # Check Convergence
            diff = state_iter_mean - prev_state_iter_mean
            diff.yaw = ssa(diff.yaw)
            if np.linalg.norm(diff) < self.convergence_threshold:
                break
            
            prev_state_iter_mean = state_iter_mean.copy()

        # 6. Final Posterior Construction
        state_post_mean = state_iter_mean

        # Re-calculate matrices at the optimal point for covariance update
        self.body_angles = calculate_body_angles(measurements_global_coords, ground_truth if self.use_gt_state_for_bodyangles_calc else state_post_mean)
        num_meas = len(self.body_angles) 
        H_lidar = self.sensor_model.lidar_jacobian(state_post_mean, self.body_angles)
        R_lidar = self.sensor_model.R(num_meas)
        S_lidar = H_lidar @ P_pred @ H_lidar.T + R_lidar
        K_lidar = np.linalg.solve(S_lidar.T, (H_lidar @ P_pred.T)).T

        # Update Covariance (Joseph Form)
        I = np.eye(len(state_post_mean))
        state_post_cov = (I - K_lidar @ H_lidar) @ P_pred @ (I - K_lidar @ H_lidar).T + K_lidar @ R_lidar @ K_lidar.T
        
        self.state_estimate = MultiVarGauss(mean=state_post_mean, cov=state_post_cov)

        # For plotting/analysis
        # Note: z_pred and S are approximations at the solution point
        z_pred_final = self.sensor_model.h_lidar(state_post_mean, self.body_angles).flatten()
        z_pred_gauss = MultiVarGauss(mean=z_pred_final, cov=S_lidar)
        innovation_gauss = MultiVarGauss(mean=z - z_pred_final, cov=S_lidar)

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

    def _augment_with_negative_info(self, measurements: LidarScan, state_pred: State_GP) -> LidarScan:
        """
        Generates virtual measurements if the predicted extent exceeds the measured extent.
        """
        # 1. Analyze Measurements
        meas_angles, meas_ranges = cart2pol(measurements.x, measurements.y)
        
        if len(meas_angles) == 0:
            return measurements

        # Mean-centered unwrapping
        mean_angle = np.arctan2(np.mean(np.sin(meas_angles)), np.mean(np.cos(meas_angles)))
        diff_angles = ssa(meas_angles - mean_angle)
        
        min_idx = np.argmin(diff_angles)
        max_idx = np.argmax(diff_angles)
        
        min_meas_angle = meas_angles[min_idx] # Global frame
        max_meas_angle = meas_angles[max_idx] # Global frame

        # 2. Analyze Predicted Shape Extent
        gp_utils = self.sensor_model.gp_utils
        
        # Get support points in body frame
        bx, by = pol2cart(gp_utils.theta_test, state_pred.radii)
        
        # Transform to Global Frame
        rot = np.array([[np.cos(state_pred.yaw), -np.sin(state_pred.yaw)],
                        [np.sin(state_pred.yaw),  np.cos(state_pred.yaw)]])
        
        # Be careful with shapes here: pos is (2,) or (2,1)
        pos = np.array([state_pred.x, state_pred.y]).reshape(2, 1)
        global_points = pos + rot @ np.vstack([bx, by])
        
        # Transform to Sensor Frame (relative to lidar)
        sensor_points = global_points - self.sensor_model.lidar_position.reshape(2,1)
        pred_angles, _ = cart2pol(sensor_points[0,:], sensor_points[1,:])
        
        pred_diff_angles = ssa(pred_angles - mean_angle)
        min_pred_angle = pred_angles[np.argmin(pred_diff_angles)]
        max_pred_angle = pred_angles[np.argmax(pred_diff_angles)]

        # 3. Logic: Check for "Overhang"
        virtual_x = []
        virtual_y = []
        
        threshold = np.deg2rad(5.0) 
        
        # Left Side
        if ssa(min_meas_angle - min_pred_angle) > threshold:
            virtual_x.append(measurements.x[min_idx])
            virtual_y.append(measurements.y[min_idx])
            
        # Right Side
        if ssa(max_pred_angle - max_meas_angle) > threshold:
            virtual_x.append(measurements.x[max_idx])
            virtual_y.append(measurements.y[max_idx])

        # 4. Concatenate
        if len(virtual_x) > 0:
            new_x = np.concatenate([measurements.x, np.array(virtual_x)])
            new_y = np.concatenate([measurements.y, np.array(virtual_y)])
            return LidarScan(x=new_x, y=new_y)
            
        return measurements