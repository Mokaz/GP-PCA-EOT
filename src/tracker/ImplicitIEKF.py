import numpy as np
from scipy.stats import chi2

from src.senfuslib import MultiVarGauss
from src.tracker.tracker import Tracker
from src.tracker.TrackerUpdateResult import TrackerUpdateResult
from src.utils.tools import rot2D, ssa, initialize_centroid, cart2pol
from src.states.states import State_PCA, LidarScan
from src.dynamics.process_models import Model_PCA_CV
from src.sensors.LidarModel import LidarMeasurementModel
from src.utils.config_classes import Config

class ImplicitIEKF(Tracker):
    """
    Implicit Iterated Extended Kalman Filter (I-IEKF).
    Uses the Implicit Measurement Model constraint g(x, z) = 0.
    """
    sensor_model: LidarMeasurementModel

    def __init__(self, 
                 dynamic_model: Model_PCA_CV, 
                 lidar_model: LidarMeasurementModel,
                 config: Config,
                 max_iterations: int | None = None, 
                 convergence_threshold : float | None = None):
        super().__init__(dynamic_model=dynamic_model, sensor_model=lidar_model, config=config)
        
        self.max_iterations = max_iterations if max_iterations is not None else config.tracker.max_iterations
        self.convergence_threshold = convergence_threshold if convergence_threshold is not None else config.tracker.convergence_threshold
        self.use_initialize_centroid = config.tracker.use_initialize_centroid

        self.use_state_clamping = config.tracker.use_state_clamping
        self.use_mahalanobis_projection = config.tracker.use_mahalanobis_projection
        if self.use_mahalanobis_projection:
            prob = getattr(config.tracker, 'mahalanobis_projection_prob', 0.99)
            self.chi2_thresh = chi2.ppf(prob, df=config.tracker.N_pca)
        self.use_negative_info = getattr(config.tracker, 'use_negative_info', False)
        self.use_D_imp_for_R = getattr(config.tracker, 'use_D_imp_for_R', True)
        # TODO: DEBUG
        self.debug_time_counter = 0

    def predict(self):
        self.state_estimate = self.dynamic_model.pred_from_est(self.state_estimate, self.T)

    def update(self, measurements_local: LidarScan, ground_truth: State_PCA = None) -> TrackerUpdateResult:
        state_prior_mean = self.state_estimate.mean.copy()        
        P_pred = self.state_estimate.cov.copy()
        state_pred = MultiVarGauss(mean=state_prior_mean, cov=P_pred)

        if self.use_negative_info and len(measurements_local.x) > 2:
            measurements_augmented = measurements_local
            # measurements_augmented = self._augment_with_negative_info(
            #     measurements_local, state_prior_mean
            # )
        else:
            measurements_augmented = measurements_local

        polar_measurements = list(zip(measurements_augmented.angle, measurements_augmented.range))

        lidar_pos = self.sensor_model.lidar_position.reshape(2, 1)
        measurements_global_coords = measurements_augmented + lidar_pos
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

        # Tracking variables for applied constraints across iterations
        final_clamped_length = None
        final_clamped_width = None
        final_mahalanobis_projection = None

        for i in range(self.max_iterations):
            H_imp, D_imp, theta_implicit = self.sensor_model.get_implicit_matrices(state_iter_mean, measurements_global_coords)
            # print("H_implicit shape:", H_imp.shape)
            # print("rank(H_implicit):", np.linalg.matrix_rank(H_imp))

            # --- DEBUGGING: Analyze the Jacobian to understand observability and convergence behavior ---
            # if i == 0 and self.time_counter < 5:  # Only analyze the Jacobian in the first few iterations to avoid clutter
            #     print(f"\nIteration {i+1} - Analyzing Implicit Jacobian at time step {self.time_counter}:")
            #     # 1. Check Condition Number (High = Ill-conditioned/Explosive)
            #     cond_num = np.linalg.cond(H_imp)
            #     print(f"\nCondition Number of H_imp: {cond_num:.2e}")
            #     if cond_num > 1e4:
            #         print("WARNING: Jacobian is ill-conditioned. Optimizer will likely explode.")

            #     # 2. Singular Value Decomposition
            #     U, S_vals, V_t = np.linalg.svd(H_imp, full_matrices=False)
                
            #     print("\nSingular Values (Small values indicate unobservable directions):")
            #     print(np.round(S_vals, 4))

            #     # 3. Analyze the Null-Space Vector (The smallest singular value's direction)
            #     # Find the smallest NON-ZERO singular value (Assuming 3 velocity states are always 0)
            #     # The observable states are the first 9 singular values.
            #     null_vector = V_t[-4, :]  # -1, -2, -3 are velocities. -4 is the weakest shape parameter
                
            #     print("\nNull-Space Direction (How the filter can move state without changing measurements):")
            #     print(f"Δ Pos_x:      {null_vector[0]:.4f}")
            #     print(f"Δ Pos_y:      {null_vector[1]:.4f}")
            #     print(f"Δ Heading:    {null_vector[2]:.4f}")
            #     print(f"Δ Length (L): {null_vector[6]:.4f}")
            #     print(f"Δ Width (W):  {null_vector[7]:.4f}")
            #     print(f"Δ PCA_0:      {null_vector[8]:.4f}")
            #     print(f"Δ PCA_1:      {null_vector[9]:.4f}")
            #     print(f"Δ PCA_2:      {null_vector[10]:.4f}")
            #     print(f"Δ PCA_3:      {null_vector[11]:.4f}")

            #     self.time_counter += 1

            # if i == 0:
            #     # Save the angles from the first iteration
            #     first_iter_thetas = theta_implicit.copy()
                
            #     # Calculate rho (assuming you can access zloc_x, zloc_y, L, W here)
            #     # rho = sqrt( (zloc_x / L)**2 + (zloc_y / W)**2 )
            #     # Alternatively, just look at the raw H_imp matrix:
            #     max_heading_derivative = np.max(np.abs(H_imp[:, 2]))  # Column 2 is Heading
                
            #     if max_heading_derivative > 10.0:
            #         print(f"\n[PROOF] EXPLOSION DETECTED! Max Heading Jacobian: {max_heading_derivative:.2f}")
            #         # Find the index of the measurement causing the explosion
            #         bad_idx = np.argmax(np.abs(H_imp[:, 2])) // 2  # Integer divide by 2 because of x/y rows
            #         print(f"-> This is caused by Measurement Index {bad_idx}.")
                    
            # elif i == 1:
            #     # See how much the angle SLID between iter 0 and iter 1
            #     bad_idx = np.argmax(np.abs(H_imp[:, 2])) // 2
            #     angle_shift_rads = theta_implicit[bad_idx] - first_iter_thetas[bad_idx]
            #     angle_shift_deg = np.degrees(angle_shift_rads)
            #     print(f"-> Between iteration 0 and 1, the point on the boundary SLID by {angle_shift_deg:.2f} degrees!")
                        
            z_pred_iter = self.sensor_model.h_from_theta(state_iter_mean, theta_implicit)
            predicted_measurements_iterates.append(z_pred_iter.copy())
            
            # The residual y = z - h(x, theta(x,z))
            innovation_iter = z_flat - z_pred_iter

            # Effective Measurement Noise
            # R_eff = D * R * D.T
            num_meas = measurements_global_coords.shape[1]
            R_std = self.sensor_model.R(num_meas)

            if self.use_D_imp_for_R:
                # R_eff = D_imp @ R_std @ D_imp.T
                R_eff = D_imp @ R_std @ D_imp.T
            else:
                R_eff = R_std

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
            
            state_next = state_prior_mean + K @ (innovation_iter - H_imp @ diff_state)
            state_next[2] = ssa(state_next[2])

            # =================================================================
            # ENFORCE OPTIONAL STABILITY CONSTRAINTS
            if self.use_state_clamping:
                # Prevent completely impossible sizes (divergence safety net)
                orig_length = state_next.length
                orig_width = state_next.width
                if orig_length < 1.0:
                    state_next.length = 1.0
                    print(f"Clamped length: {orig_length:.3f} -> {state_next.length:.3f}")
                    final_clamped_length = (float(orig_length), 1.0)
                if orig_width < 0.5:
                    state_next.width = 0.5
                    print(f"Clamped width:  {orig_width:.3f} -> {state_next.width:.3f}")
                    final_clamped_width = (float(orig_width), 0.5)

            if self.use_mahalanobis_projection:
                # Constrain PCA coefficients to the requested probability ellipsoid
                eigenvalues = self.config.tracker.pca_eigenvalues
                e = state_next.pca_coeffs

                # Calculate squared Mahalanobis distance D_M^2
                mahalanobis_sq = np.sum((e ** 2) / eigenvalues)

                if mahalanobis_sq > self.chi2_thresh:
                    orig_coeffs = e.copy()
                    # Project the vector back onto the surface of the ellipsoid
                    scale_factor = np.sqrt(self.chi2_thresh / mahalanobis_sq)
                    state_next.pca_coeffs = e * scale_factor
                    final_mahalanobis_projection = (orig_coeffs, state_next.pca_coeffs.copy(), mahalanobis_sq)
            # =================================================================
            
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
            K_gain=K,
            clamped_length=final_clamped_length,
            clamped_width=final_clamped_width,
            mahalanobis_projection=final_mahalanobis_projection
        )

    # def _augment_with_negative_info(self, measurements: LidarScan, state_pred: np.ndarray) -> LidarScan:
    #     """
    #     Generates virtual measurements in the free space if the predicted extent exceeds the measured extent.
    #     """
    #     # 1. Analyze Measurements
    #     meas_angles, meas_ranges = cart2pol(measurements.x, measurements.y)
        
    #     if len(meas_angles) == 0:
    #         return measurements

    #     # Mean-centered unwrapping
    #     mean_angle = np.arctan2(np.mean(np.sin(meas_angles)), np.mean(np.cos(meas_angles)))
    #     diff_angles = ssa(meas_angles - mean_angle)
        
    #     min_idx = np.argmin(diff_angles)
    #     max_idx = np.argmax(diff_angles)
        
    #     min_meas_angle = meas_angles[min_idx] # Local sensor frame
    #     max_meas_angle = meas_angles[max_idx] # Local sensor frame

    #     # 2. Analyze Predicted Shape Extent
    #     body_angles = np.linspace(-np.pi, np.pi, 360).tolist()
    #     global_points = self.sensor_model.h_lidar(state_pred, body_angles).T 
        
    #     # Transform to Sensor Frame (relative to lidar)
    #     sensor_points = global_points - self.sensor_model.lidar_position.reshape(2,1)
    #     pred_angles, _ = cart2pol(sensor_points[0,:], sensor_points[1,:])
        
    #     pred_diff_angles = ssa(pred_angles - mean_angle)
    #     min_pred_angle = pred_angles[np.argmin(pred_diff_angles)]
    #     max_pred_angle = pred_angles[np.argmax(pred_diff_angles)]

    #     # 3. Logic: Check for "Overhang" into empty space
    #     virtual_x =[]
    #     virtual_y =[]
        
    #     margin = np.deg2rad(5.0) 
        
    #     # Left Side (Minimum Angle Overhang)
    #     if ssa(min_meas_angle - min_pred_angle) > margin:
    #         # Place virtual point in the free space just outside the measured object
    #         virt_angle = ssa(min_meas_angle - margin)
    #         # Cap the range using the adjacent measurement
    #         virt_range = meas_ranges[min_idx]
            
    #         virtual_x.append(virt_range * np.cos(virt_angle))
    #         virtual_y.append(virt_range * np.sin(virt_angle))
            
    #     # Right Side (Maximum Angle Overhang)
    #     if ssa(max_pred_angle - max_meas_angle) > margin:
    #         # Place virtual point in the free space just outside the measured object
    #         virt_angle = ssa(max_meas_angle + margin)
    #         virt_range = meas_ranges[max_idx]
            
    #         virtual_x.append(virt_range * np.cos(virt_angle))
    #         virtual_y.append(virt_range * np.sin(virt_angle))

    #     # 4. Concatenate
    #     if len(virtual_x) > 0:
    #         new_x = np.concatenate([measurements.x, np.array(virtual_x)])
    #         new_y = np.concatenate([measurements.y, np.array(virtual_y)])
    #         return LidarScan(x=new_x, y=new_y)
            
    #     return measurements
