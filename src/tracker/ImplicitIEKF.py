import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize_scalar
from scipy.linalg import block_diag

from src.senfuslib import MultiVarGauss
from src.tracker.tracker import Tracker
from src.tracker.TrackerUpdateResult import TrackerUpdateResult
from src.utils.tools import rot2D, ssa, initialize_centroid, cart2pol
from src.states.states import State_PCA, LidarScan
from src.dynamics.process_models import Model_PCA_CV, DynamicModel
from src.sensors.LidarModel import LidarMeasurementModel
from src.utils.config_classes import Config

class ImplicitIEKF(Tracker):
    """
    Implicit Iterated Extended Kalman Filter (I-IEKF).
    Uses the Implicit Measurement Model constraint g(x, z) = 0.
    Fuses explicit Negative Information angular bounds if enabled.
    """
    sensor_model: LidarMeasurementModel
    dynamic_model: DynamicModel

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
        self.use_exact_extreme_angle = getattr(config.tracker, 'use_exact_extreme_angle', False)
        self.use_D_imp_for_R = getattr(config.tracker, 'use_D_imp_for_R', True)
        self.use_scaled_R = getattr(config.tracker, 'use_scaled_R', False)

    def predict(self):
        self.state_estimate = self.dynamic_model.pred_from_est(self.state_estimate, self.T)

    def update(self, measurements_local: LidarScan, ground_truth: State_PCA = None) -> TrackerUpdateResult:
        state_prior_mean = self.state_estimate.mean.copy()        
        P_pred = self.state_estimate.cov.copy()
        state_pred = MultiVarGauss(mean=state_prior_mean, cov=P_pred)

        polar_measurements = list(zip(measurements_local.angle, measurements_local.range))

        lidar_pos = self.sensor_model.lidar_position.reshape(2, 1)
        measurements_global_coords = measurements_local + lidar_pos
        z_flat = measurements_global_coords.flatten('F') # Stacked[x1, y1, x2, y2...]
        
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

        # Tracking variables
        final_clamped_length = None
        final_clamped_width = None
        final_mahalanobis_projection = None
        final_negative_info_count = 0
        virtual_constraints_iterates = []

        # Determine angular margin based on sensor resolution
        num_rays = getattr(self.config.lidar, 'num_rays', 360)
        self.angular_margin = (2 * np.pi) / num_rays

        # Latch for Active Set Method to prevent optimization jitter
        active_min_angle = False
        active_max_angle = False

        for i in range(self.max_iterations):
            # 1. Implicit Measurement Calculations (Positive Info)
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
            
            innovation_iter = z_flat - z_pred_iter

            num_meas = measurements_global_coords.shape[1]
            if self.use_scaled_R:
                R_std = self.sensor_model.R_scaled(num_meas)
            else:
                R_std = self.sensor_model.R(num_meas) 

            if self.use_D_imp_for_R:
                R_eff = D_imp @ R_std @ D_imp.T
            else:
                R_eff = R_std

            # 2. Explicit Angular Constraints Fusion (Negative Info)
            H_fused = H_imp
            innovation_fused = innovation_iter
            R_fused = R_eff
            current_virtual_constraints = []

            if self.use_negative_info and num_meas > 2:
                # Get virtual constraints, passing the active state to lock them once triggered
                current_virtual_constraints, active_min_angle, active_max_angle = self._get_virtual_angular_constraints(
                    measurements_local, state_iter_mean, active_min_angle, active_max_angle
                )
                
                final_negative_info_count = len(current_virtual_constraints)

                for vc in current_virtual_constraints:
                    # Analytical Jacobian (rad/state)
                    H_virt, gamma_pred = self.sensor_model.get_virtual_measurement_jacobian(
                        state_iter_mean, vc['body_angle']
                    )
                    ang_residual = ssa(vc['measured_angle'] - gamma_pred)
                    
                    # Extract distance to the edge point
                    pt_global = self.sensor_model.h_lidar(state_iter_mean, [vc['body_angle']]).flatten()
                    u_x = pt_global[0] - self.sensor_model.lidar_position[0]
                    u_y = pt_global[1] - self.sensor_model.lidar_position[1]
                    rho = np.maximum(np.sqrt(u_x**2 + u_y**2), 1.0)
                    
                    # Convert to arc length
                    arc_residual = rho * ang_residual
                    H_arc = rho * H_virt
                    
                    vc['ang_residual'] = float(ang_residual)
                    vc['arc_residual'] = float(arc_residual)
                    vc['rho'] = float(rho)
                    
                    # Stack fused matrices
                    H_fused = np.vstack((H_fused, H_arc))
                    innovation_fused = np.append(innovation_fused, arc_residual)
                    
                    # Set Strict Cartesian Variance (e.g. 1cm virtual wall vs 15cm lidar hits)
                    R_arc = np.array([[self.config.tracker.R_arc_std ** 2]])
                    R_fused = block_diag(R_fused, R_arc)
                    
            virtual_constraints_iterates.append(current_virtual_constraints)

            # 3. IEKF State Update Equation
            S = H_fused @ P_pred @ H_fused.T + R_fused
            
            K_transpose = np.linalg.solve(S, H_fused @ P_pred.T)
            K = K_transpose.T
            
            diff_state = state_prior_mean - state_iter_mean
            diff_state[2] = ssa(diff_state[2])
            
            state_next = state_prior_mean + K @ (innovation_fused - H_fused @ diff_state)
            state_next[2] = ssa(state_next[2])

            # =================================================================
            # ENFORCE OPTIONAL STABILITY CONSTRAINTS
            if self.use_state_clamping:
                orig_length = state_next.length
                orig_width = state_next.width
                if orig_length < 1.0:
                    state_next.length = 1.0
                    final_clamped_length = (float(orig_length), 1.0)
                if orig_width < 0.5:
                    state_next.width = 0.5
                    final_clamped_width = (float(orig_width), 0.5)

            if self.use_mahalanobis_projection:
                eigenvalues = self.config.tracker.pca_eigenvalues
                e = state_next.pca_coeffs
                mahalanobis_sq = np.sum((e ** 2) / eigenvalues)

                if mahalanobis_sq > self.chi2_thresh:
                    orig_coeffs = e.copy()
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

        # Final Update
        I = np.eye(len(state_iter_mean))
        state_post_cov = (I - K @ H_fused) @ P_pred @ (I - K @ H_fused).T + K @ R_fused @ K.T
        
        self.state_estimate = MultiVarGauss(mean=state_iter_mean, cov=state_post_cov)
        
        # NOTE: z_pred_gauss will just represent the positive measurements for visualization/metrics
        z_pred_gauss = MultiVarGauss(mean=z_pred_iter, cov=(H_imp @ P_pred @ H_imp.T + R_eff))
        innovation_gauss = MultiVarGauss(mean=innovation_iter, cov=(H_imp @ P_pred @ H_imp.T + R_eff))

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
            mahalanobis_projection=final_mahalanobis_projection,
            negative_info_used=final_negative_info_count,
            virtual_constraints_info=virtual_constraints_iterates
        )

    def _get_virtual_angular_constraints(self, measurements: LidarScan, state_pred: np.ndarray, 
                                         active_min: bool, active_max: bool) -> tuple[list[dict], bool, bool]:
        """
        Determines the maximum and minimum angle constraints (Negative Information).
        Treats triggered edges as "Active Set" equality constraints to assure IEKF convergence.
        """
        meas_angles, _ = cart2pol(measurements.x, measurements.y)
        if len(meas_angles) < 2:
            return [], active_min, active_max

        # Find the measured bounds
        mean_angle = np.arctan2(np.mean(np.sin(meas_angles)), np.mean(np.cos(meas_angles)))
        diff_angles = ssa(meas_angles - mean_angle)
        
        min_meas_angle = meas_angles[np.argmin(diff_angles)]
        max_meas_angle = meas_angles[np.argmax(diff_angles)]

        # Sample the predicted shape discretely
        body_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        global_points = self.sensor_model.h_lidar(state_pred, body_angles).T 
        
        sensor_points = global_points - self.sensor_model.lidar_position.reshape(2, 1)
        pred_angles, _ = cart2pol(sensor_points[0, :], sensor_points[1, :])
        
        pred_diff_angles = ssa(pred_angles - mean_angle)
        
        min_idx = np.argmin(pred_diff_angles)
        max_idx = np.argmax(pred_diff_angles)
        
        min_pred_angle = pred_angles[min_idx]
        max_pred_angle = pred_angles[max_idx]

        virtual_constraints =[]
        
        # Minimum Angle (Left) - Trigger once, latch for all iterations
        if active_min or ssa(min_meas_angle - min_pred_angle) > self.angular_margin:
            active_min = True
            theta_min = body_angles[min_idx]
            
            if self.use_exact_extreme_angle:
                theta_min = self._get_exact_extreme_angle(state_pred, theta_min, mean_angle, is_max=False)
            
            pt_global_min = self.sensor_model.h_lidar(state_pred,[theta_min]).flatten()

            virtual_constraints.append({
                'measured_angle': ssa(min_meas_angle - self.angular_margin),
                'body_angle': theta_min,
                'predicted_point': pt_global_min,
                'type': 'min_angle'
            })

        # Maximum Angle (Right) - Trigger once, latch for all iterations
        if active_max or ssa(max_pred_angle - max_meas_angle) > self.angular_margin:
            active_max = True
            theta_max = body_angles[max_idx]
            
            if self.use_exact_extreme_angle:
                theta_max = self._get_exact_extreme_angle(state_pred, theta_max, mean_angle, is_max=True)
            
            pt_global_max = self.sensor_model.h_lidar(state_pred, [theta_max]).flatten()

            virtual_constraints.append({
                'measured_angle': ssa(max_meas_angle + self.angular_margin),
                'body_angle': theta_max,
                'predicted_point': pt_global_max,
                'type': 'max_angle'
            })

        return virtual_constraints, active_min, active_max

    def _get_exact_extreme_angle(self, state_pred: np.ndarray, guess_theta: float, mean_angle: float, is_max: bool) -> float:
        """
        Uses a continuous 1D optimizer to find the exact tangent angle theta_body*,
        preventing discretization error and satisfying Danskin's theorem.
        """
        def objective(theta):
            pt_global = self.sensor_model.h_lidar(state_pred,[theta]).flatten()
            u_x = pt_global[0] - self.sensor_model.lidar_position[0]
            u_y = pt_global[1] - self.sensor_model.lidar_position[1]
            
            gamma = np.arctan2(u_y, u_x)
            
            # Unwrap relative to the mean measured angle to prevent optimizer Pi wraparound
            unwrapped_gamma = ssa(gamma - mean_angle)
            
            return -unwrapped_gamma if is_max else unwrapped_gamma

        delta = np.deg2rad(2.0)
        bounds = (guess_theta - delta, guess_theta + delta)
        
        res = minimize_scalar(objective, bounds=bounds, method='bounded')
        return res.x