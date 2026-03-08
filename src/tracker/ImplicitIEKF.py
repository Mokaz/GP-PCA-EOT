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
        self.time_counter = 0

    def predict(self):
        self.state_estimate = self.dynamic_model.pred_from_est(self.state_estimate, self.T)

    def update(self, measurements_local: LidarScan, ground_truth: State_PCA = None) -> TrackerUpdateResult:
        polar_measurements = list(zip(measurements_local.angle, measurements_local.range))

        state_prior_mean = self.state_estimate.mean.copy()        
        P_pred = self.state_estimate.cov.copy()
        state_pred = MultiVarGauss(mean=state_prior_mean, cov=P_pred)

        lidar_pos = self.sensor_model.lidar_position.reshape(2, 1)
        measurements_global_coords = measurements_local + lidar_pos
        z_flat = measurements_global_coords.flatten('F') # Stacked [x1, y1, x2, y2...]
        
        state_iter_mean = state_prior_mean.copy()
        
        if self.use_initialize_centroid:
            state_iter_mean.pos = initialize_centroid(
                position=state_prior_mean.pos,
                lidar_pos=self.sensor_model.lidar_position,
                measurements=polar_measurements,
                L_est=state_prior_mean.length,
                W_est=state_prior_mean.width
            )

        prev_state_iter_mean = state_prior_mean.copy()
        iterates = [state_iter_mean.copy()]
        predicted_measurements_iterates = []

        for i in range(self.max_iterations):
            H_imp, D_imp, theta_implicit = self.sensor_model.get_implicit_matrices(state_iter_mean, measurements_global_coords)
            # print("H_implicit shape:", H_imp.shape)
            # print("rank(H_implicit):", np.linalg.matrix_rank(H_imp))

            if i == 0 and self.time_counter < 5:  # Only analyze the Jacobian in the first few iterations to avoid clutter
                print(f"\nIteration {i+1} - Analyzing Implicit Jacobian at time step {self.time_counter}:")
                # 1. Check Condition Number (High = Ill-conditioned/Explosive)
                cond_num = np.linalg.cond(H_imp)
                print(f"\nCondition Number of H_imp: {cond_num:.2e}")
                if cond_num > 1e4:
                    print("WARNING: Jacobian is ill-conditioned. Optimizer will likely explode.")

                # 2. Singular Value Decomposition
                U, S_vals, V_t = np.linalg.svd(H_imp, full_matrices=False)
                
                print("\nSingular Values (Small values indicate unobservable directions):")
                print(np.round(S_vals, 4))

                # 3. Analyze the Null-Space Vector (The smallest singular value's direction)
                # Find the smallest NON-ZERO singular value (Assuming 3 velocity states are always 0)
                # The observable states are the first 9 singular values.
                null_vector = V_t[-4, :]  # -1, -2, -3 are velocities. -4 is the weakest shape parameter
                
                print("\nNull-Space Direction (How the filter can move state without changing measurements):")
                print(f"Δ Pos_x:      {null_vector[0]:.4f}")
                print(f"Δ Pos_y:      {null_vector[1]:.4f}")
                print(f"Δ Heading:    {null_vector[2]:.4f}")
                print(f"Δ Length (L): {null_vector[6]:.4f}")
                print(f"Δ Width (W):  {null_vector[7]:.4f}")
                print(f"Δ PCA_0:      {null_vector[8]:.4f}")
                print(f"Δ PCA_1:      {null_vector[9]:.4f}")
                print(f"Δ PCA_2:      {null_vector[10]:.4f}")
                print(f"Δ PCA_3:      {null_vector[11]:.4f}")

                self.time_counter += 1
            
            z_pred_iter = self.sensor_model.h_from_theta(state_iter_mean, theta_implicit)
            predicted_measurements_iterates.append(z_pred_iter.copy())
            
            # The residual y = z - h(x, theta(x,z))
            innovation_iter = z_flat - z_pred_iter

            # Effective Measurement Noise
            # R_eff = D * R * D.T
            num_meas = measurements_global_coords.shape[1]
            R_std = self.sensor_model.R(num_meas)
            R_eff = D_imp @ R_std @ D_imp.T
            
            S = H_imp @ P_pred @ H_imp.T + R_eff
            
            # Numerical stability: use solve instead of inv
            # K = P H^T S^-1
            # S K^T = H P
            K_transpose = np.linalg.solve(S, H_imp @ P_pred.T)
            K = K_transpose.T
            
            # IEKF State Update Equation:
            # x_{i+1} = x_prior + K * ( (z - h(x_i)) - H * (x_prior - x_i) )
            diff_state = state_prior_mean - state_iter_mean
            diff_state[2] = ssa(diff_state[2])
            
            state_next = state_prior_mean + K @ (innovation_iter + H_imp @ diff_state)
            state_next[2] = ssa(state_next[2])

            # ==========================================
            # SAFETY CLAMPS: Prevent Mathematical Collapse
            # ==========================================
            # 1. Do not allow Length or Width to become negative or zero
            state_next.length = max(2.0, state_next.length)
            state_next.width = max(1.0, state_next.width)
            
            # 2. Prevent PCA coefficients from warping the shape to infinity
            # (Assuming PCA coeffs should mostly stay within +/- 2.0 based on your dataset)
            state_next.pca_coeffs = np.clip(state_next.pca_coeffs, -3.0, 3.0)
            # ==========================================
            
            state_iter_mean = state_next
            iterates.append(state_iter_mean.copy())

            # Check convergence
            step_diff = state_iter_mean - prev_state_iter_mean
            step_diff[2] = ssa(step_diff[2])
            if np.linalg.norm(step_diff) < self.convergence_threshold:
                break
            
            prev_state_iter_mean = state_iter_mean.copy()

        I = np.eye(len(state_iter_mean))
        state_post_cov = (I - K @ H_imp) @ P_pred @ (I - K @ H_imp).T + K @ R_eff @ K.T
        
        self.state_estimate = MultiVarGauss(mean=state_iter_mean, cov=state_post_cov)

        z_pred_gauss = MultiVarGauss(mean=z_pred_iter, cov=S)
        innovation_gauss = MultiVarGauss(mean=innovation_iter, cov=S)

        return TrackerUpdateResult(
            state_prior=state_pred,
            state_posterior=self.state_estimate,
            measurements=z_flat,
            predicted_measurement=z_pred_gauss,
            iterates=iterates,
            predicted_measurements_iterates=predicted_measurements_iterates,
            innovation_gauss=innovation_gauss,
            iterations=i + 1,
            H_jacobian=H_imp,
            R_covariance=R_eff,
        )