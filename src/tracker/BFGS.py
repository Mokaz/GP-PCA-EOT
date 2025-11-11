import numpy as np
from scipy.optimize import minimize

from typing import Optional

from src.senfuslib import MultiVarGauss
from src.tracker.tracker import Tracker, TrackerUpdateResult
from src.utils.tools import ssa, initialize_centroid, compute_angle_range, calculate_body_angles
from src.states.states import State_PCA, LidarScan
from src.dynamics.process_models import Model_PCA_CV
from src.sensors.LidarModel import LidarModel
from src.utils.config_classes import Config

class BFGS(Tracker):
    def __init__(self, 
                 dynamic_model: Model_PCA_CV, 
                 lidar_model: LidarModel,
                 config: Config,
                 max_iterations=10, 
                 convergence_threshold=1e-9):
        super().__init__(dynamic_model=dynamic_model, sensor_model=lidar_model, config=config)
        
        # Parameters
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def predict(self):
        """
        Prediction step.
        Updates the internal state estimate.
        """
        # Compute the predicted state using the process model
        self.state_estimate = self.dynamic_model.pred_from_est(self.state_estimate, self.T)

    def update(self, measurements_local: LidarScan, ground_truth: State_PCA = None) -> TrackerUpdateResult:
        """
        Update step using BFGS optimization.
        
        Args:
            measurements: A LidarScan of measurements (2, N)
        
        Returns:
            A TrackerUpdateResult dataclass containing the results of the update step.
        """

        angles_local = measurements_local.angle
        lower_diff, upper_diff, mean_lidar_angle = compute_angle_range(angles_local)

        polar_measurements = list(zip(measurements_local.angle, measurements_local.range))

        # Initialize the centroid for the optimization
        state_prior = self.state_estimate.mean.copy()
        state_prior.pos = initialize_centroid(
            position=state_prior.pos,
            lidar_pos=self.sensor_model.lidar_position,
            measurements=polar_measurements,
            L_est=state_prior.length,
            W_est=state_prior.width
        ) # TODO Martin: Investigate if this should be used in penalty too

        measurements_global_coords = measurements_local + self.sensor_model.lidar_position.reshape(2, 1)

        self.body_angles = calculate_body_angles(measurements_global_coords, ground_truth if self.use_gt_state_for_bodyangles_calc else state_prior)

        # Use 'F' (column-major) order to get the interleaved [x1, y1, x2, y2, ...] format.
        z = measurements_global_coords.flatten('F')
        
        x_pred = state_prior # NOTE Using initial_state_guess (centroid-corrected) as state_pred
        P_pred = self.state_estimate.cov.copy()
        state_pred = MultiVarGauss(mean=x_pred, cov=P_pred) # NOTE Martin: Has adjusted initial centroid for optimization!!

        res = minimize(
            fun=self.prob_with_penalty, 
            x0=state_prior,
            args=(z, x_pred, P_pred, ground_truth, mean_lidar_angle, lower_diff, upper_diff),
            method='BFGS',
            # jac='3-point', # Use numerical differentiation for the gradient
            # options={'maxiter': self.max_iterations, 'gtol': self.convergence_threshold}
        )

        # --- NEW DEBUGGING CALCULATIONS ---
        state_post_mean = State_PCA.from_array(res.x)
        
        # Recalculate components at the final solution (res.x)
        num_meas = len(self.body_angles)
        R = self.sensor_model.R(num_meas)
        P_pred_inv = np.linalg.inv(P_pred)
        R_inv = np.linalg.inv(R)

        z_residual = z - self.sensor_model.h_lidar(state_post_mean, self.body_angles).flatten()
        x_residual = state_post_mean - x_pred

        cost_likelihood = 0.5 * z_residual.T @ R_inv @ z_residual
        cost_prior = 0.5 * x_residual.T @ P_pred_inv @ x_residual
        cost_penalty = self.penalty_function(state_post_mean, mean_lidar_angle, lower_diff, upper_diff)

        # Calculate Jacobian at the solution
        H = self.sensor_model.lidar_jacobian(state_post_mean, self.body_angles)
        
        # Calculate innovation and innovation covariance for analysis
        z_pred = self.sensor_model.h_lidar(state_post_mean, self.body_angles).flatten()
        innovation = z - z_pred
        S = H @ P_pred @ H.T + R
        
        z_pred_gauss = MultiVarGauss(mean=z_pred, cov=S)
        innovation_gauss = MultiVarGauss(mean=innovation, cov=S)
        # --- END NEW DEBUGGING CALCULATIONS ---
        
        # Update internal state estimate
        state_post_cov = res.hess_inv # The inverse Hessian is an approximation of the posterior covariance
        self.state_estimate = MultiVarGauss(mean=state_post_mean, cov=state_post_cov)

        return TrackerUpdateResult(
            state_prior=state_pred,
            state_posterior=self.state_estimate,
            measurements=z,
            predicted_measurement=z_pred_gauss,
            innovation_gauss=innovation_gauss,
            # Populate new fields
            cost_prior=cost_prior,
            cost_likelihood=cost_likelihood,
            cost_penalty=cost_penalty,
            H_jacobian=H,
            R_covariance=R,
            # Populate existing optional fields
            iterations=res.nit,
            cost=res.fun,
            raw_optimizer_result=res,
        )

    def penalty_function(self, x, mean_angle, lower_diff, upper_diff):
        lidar_pos = self.sensor_model.lidar_position

        L = x[6]
        W = x[7]
        angles = np.array([0.0, np.arctan2(W, L), np.pi/2, np.arctan2(W, -L), np.pi, np.arctan2(-W, -L), -np.pi / 2, np.arctan2(-W, L)])

        pred_meas = self.sensor_model.h_lidar(x, angles)
        pred_meas = np.array(pred_meas).reshape(-1, 2)

        # Ensure the angles of prediced measurements are within the range
        pred_angles = np.arctan2(pred_meas[:, 1] - lidar_pos[1], pred_meas[:, 0] - lidar_pos[0])

        alpha_min = mean_angle + lower_diff
        alpha_max = mean_angle + upper_diff

        # Calculate the penalty
        gain = 50.0
        penalty = 0.0
        for i in range(len(pred_angles)):
            penalty += gain * max(0.0, ssa(pred_angles[i] - alpha_max)) + gain * max(0.0, ssa(alpha_min - pred_angles[i]))

        return penalty

    def prob_with_penalty(self, x, z, x_pred, P_pred, ground_truth, mean_lidar_angle, lower_diff, upper_diff):
        penalty = self.penalty_function(x, mean_lidar_angle, lower_diff, upper_diff)
        objective = self.object_function(x, x_pred, P_pred, z, ground_truth)
        return objective + penalty