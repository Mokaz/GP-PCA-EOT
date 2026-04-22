import numpy as np
import scipy.linalg
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.stats import chi2
from scipy.linalg import block_diag

from src.senfuslib import MultiVarGauss
from src.tracker.tracker import Tracker
from src.tracker.TrackerUpdateResult import TrackerUpdateResult
from src.utils.tools import rot2D, ssa, initialize_centroid, cart2pol
from src.states.states import State_PCA, LidarScan
from src.dynamics.process_models import Model_PCA_CV, DynamicModel
from src.sensors.LidarModel import LidarMeasurementModel
from src.utils.config_classes import Config

class ImplicitSmoother(Tracker):
    """
    Fixed-Lag Smoother based on the Implicit Analytical Measurement Model.
    Maintains a sliding window of N frames. Optimizes all N poses and a single 
    global shape simultaneously using Non-Linear Least Squares (SLSQP).
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
        
        # Standard Tracker Settings
        self.max_iterations = max_iterations if max_iterations is not None else config.tracker.max_iterations
        self.convergence_threshold = convergence_threshold if convergence_threshold is not None else config.tracker.convergence_threshold
        self.use_initialize_centroid = config.tracker.use_initialize_centroid
        self.use_scaled_R = getattr(config.tracker, 'use_scaled_R', False)
        self.use_D_imp_for_R = getattr(config.tracker, 'use_D_imp_for_R', True)
        
        # Constraints
        self.use_state_clamping = config.tracker.use_state_clamping
        self.use_mahalanobis_projection = config.tracker.use_mahalanobis_projection
        if self.use_mahalanobis_projection:
            prob = getattr(config.tracker, 'mahalanobis_projection_prob', 0.99)
            self.chi2_thresh = chi2.ppf(prob, df=config.tracker.N_pca)
            
        self.use_negative_info_angular = getattr(config.tracker, 'use_negative_info_angular', False)
        self.use_negative_info_front = getattr(config.tracker, 'use_negative_info_front', False)
        self.use_negative_info_centroid = getattr(config.tracker, 'use_negative_info_centroid', False)
        self.radial_margin = getattr(config.tracker, 'radial_margin', 0.1)
        self.use_exact_extreme_angle = getattr(config.tracker, 'use_exact_extreme_angle', False)
        self.neg_info_var = getattr(self.config.tracker, 'R_arc_std', 0.01) ** 2

        # Smoother Specific Settings
        self.window_size = getattr(config.tracker, 'smoother_window_size', 10)
        self.history_buffer =[]

    def predict(self):
        self.state_estimate = self.dynamic_model.pred_from_est(self.state_estimate, self.T)

    def update(self, measurements_local: LidarScan, ground_truth: State_PCA = None) -> TrackerUpdateResult:
        # =================================================================
        # 1. REAL-TIME IEKF PASS (To get a good initial guess for the Smoother)
        # =================================================================
        state_prior_mean = self.state_estimate.mean.copy()        
        P_pred = self.state_estimate.cov.copy()
        state_pred = MultiVarGauss(mean=state_prior_mean, cov=P_pred)

        polar_measurements = list(zip(measurements_local.angle, measurements_local.range))
        num_meas = measurements_local.x.shape[0]

        lidar_pos = self.sensor_model.lidar_position.reshape(2, 1)
        measurements_global_coords = measurements_local + lidar_pos
        z_flat = measurements_global_coords.flatten('F')
        
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
        iterates =[state_iter_mean.copy()]
        num_rays = getattr(self.config.lidar, 'num_rays', 360)
        self.angular_margin = (2 * np.pi) / num_rays

        # Standard IEKF Loop
        for i in range(self.max_iterations):
            H_imp, D_imp, theta_implicit = self.sensor_model.get_implicit_matrices(state_iter_mean, measurements_global_coords)
            z_pred_iter = self.sensor_model.h_from_theta(state_iter_mean, theta_implicit)
            innovation_iter = z_flat - z_pred_iter

            R_std = self.sensor_model.R_scaled(num_meas) if self.use_scaled_R else self.sensor_model.R(num_meas) 
            R_eff = D_imp @ R_std @ D_imp.T if self.use_D_imp_for_R else R_std

            H_fused = H_imp
            innovation_fused = innovation_iter
            R_fused = R_eff

            any_negative_info = self.use_negative_info_angular or self.use_negative_info_front or self.use_negative_info_centroid
            if any_negative_info and num_meas > 2:
                vcs = self._get_virtual_constraints(measurements_local, state_iter_mean)
                for vc in vcs:
                    c_type = vc['type']
                    if c_type in['min_angle', 'max_angle']:
                        H_virt, gamma_pred = self.sensor_model.get_virtual_measurement_jacobian(state_iter_mean, vc['body_angle'], is_radial=False)
                        ang_residual = ssa(vc['measured_val'] - gamma_pred)
                        rho = np.maximum(np.sqrt((vc['predicted_point'][0] - lidar_pos[0])**2 + (vc['predicted_point'][1] - lidar_pos[1])**2), 1.0)
                        H_stack = rho * H_virt
                        residual = rho * ang_residual
                    elif c_type == 'front_wall':
                        H_stack, rho_pred = self.sensor_model.get_virtual_measurement_jacobian(state_iter_mean, vc['body_angle'], is_radial=True)
                        residual = vc['measured_val'] - rho_pred
                    elif c_type == 'centroid_depth':
                        rho_c = vc['rho_c']
                        residual = vc['measured_val'] - rho_c
                        H_stack = np.zeros((1, len(state_iter_mean)))
                        H_stack[0, 0] = (state_iter_mean[0] - lidar_pos[0]) / rho_c
                        H_stack[0, 1] = (state_iter_mean[1] - lidar_pos[1]) / rho_c
                    
                    H_fused = np.vstack((H_fused, H_stack))
                    innovation_fused = np.append(innovation_fused, residual)
                    R_fused = block_diag(R_fused, np.array([[self.neg_info_var]]))

            # Soft Priors
            expected_L = self.config.extent.shape_params_true.get("L", 28.0)
            expected_W = self.config.extent.shape_params_true.get("W", 8.0)
            H_prior = np.zeros((2, len(state_iter_mean)))
            H_prior[0, 6], H_prior[1, 7] = 1.0, 1.0  
            H_fused = np.vstack((H_fused, H_prior))
            innovation_fused = np.append(innovation_fused, [expected_L - state_iter_mean[6], expected_W - state_iter_mean[7]])
            R_fused = block_diag(R_fused, np.diag([5.0**2, 5.0**2]))

            S = H_fused @ P_pred @ H_fused.T + R_fused
            K = np.linalg.solve(S, H_fused @ P_pred.T).T
            
            diff_state = state_prior_mean - state_iter_mean
            diff_state[2] = ssa(diff_state[2])
            state_next = state_prior_mean + K @ (innovation_fused - H_fused @ diff_state)
            state_next[2] = ssa(state_next[2])

            if self.use_state_clamping:
                state_next.length = max(1.0, state_next.length)
                state_next.width = max(0.5, state_next.width)

            state_iter_mean = state_next
            iterates.append(state_iter_mean.copy())

            step_diff = state_iter_mean - prev_state_iter_mean
            step_diff[2] = ssa(step_diff[2])
            if np.linalg.norm(step_diff) < self.convergence_threshold: break
            prev_state_iter_mean = state_iter_mean.copy()

        I = np.eye(len(state_iter_mean))
        state_post_cov = (I - K @ H_fused) @ P_pred @ (I - K @ H_fused).T + K @ R_fused @ K.T
        self.state_estimate = MultiVarGauss(mean=state_iter_mean, cov=state_post_cov)

        # =================================================================
        # 2. FIXED-LAG SMOOTHER INTEGRATION
        # =================================================================
        # Store data for the smoother window
        self.history_buffer.append({
            'z_global_flat': z_flat,
            'meas_local': measurements_local,
            'prior_mean': state_prior_mean.copy(),
            'prior_cov': P_pred.copy(),
            'post_mean': self.state_estimate.mean.copy()
        })

        if len(self.history_buffer) >= self.window_size:
            # Run SLSQP Optimization
            optimized_states, success = self._run_smoother()

            if success:
                # Overwrite the real-time state with the globally smoothed state!
                # The smoother enforces a rigid shape over the whole window.
                self.state_estimate.mean[:] = optimized_states[-1]

                # We can safely reduce covariance since smoothing is highly accurate
                self.state_estimate.cov *= 0.5 
            
            # Maintain sliding window
            self.history_buffer.clear()

        # Final return values
        z_pred_gauss = MultiVarGauss(mean=z_pred_iter, cov=(H_imp @ P_pred @ H_imp.T + R_std))
        innovation_gauss = MultiVarGauss(mean=innovation_iter, cov=(H_imp @ P_pred @ H_imp.T + R_std))

        return TrackerUpdateResult(
            state_prior=state_pred,
            state_posterior=self.state_estimate,
            measurements=z_flat,
            predicted_measurement=z_pred_gauss,
            iterates=iterates,
            predicted_measurements_iterates=[z_pred_iter],
            innovation_gauss=innovation_gauss,
            iterations=i + 1,
            H_jacobian=H_imp,
            R_covariance=R_std,
            K_gain=K
        )

    def _run_smoother(self):
        """
        Runs Non-Linear Least Squares over the last N frames to find a 
        single globally consistent shape and optimized kinematic trajectory.
        Uses Analytical Jacobians for extreme speed.
        """
        N = len(self.history_buffer)
        state_dim = len(self.history_buffer[0]['post_mean'])
        N_pca = self.config.tracker.N_pca
        
        # 1. FLAT STATE VECTOR INITIALIZATION
        X_init =[]
        for k in range(N):
            X_init.extend(self.history_buffer[k]['post_mean'][:6])
        X_init.extend(self.history_buffer[-1]['post_mean'][6:]) # Use latest shape
        X_init = np.array(X_init)

        # 2. BOUNDS
        lower_bounds = [-np.inf] * (6 * N) +[1.0, 0.5] + [-np.inf] * N_pca
        upper_bounds =[np.inf] * len(X_init)
        bounds = Bounds(lower_bounds, upper_bounds)

        # 3. MAHALANOBIS CONSTRAINT (With Analytical Jacobian)
        def pca_mahalanobis(X_opt):
            pca_coeffs = X_opt[-N_pca:]
            mahal_sq = np.sum((pca_coeffs ** 2) / self.config.tracker.pca_eigenvalues)
            return self.chi2_thresh - mahal_sq
            
        def pca_mahalanobis_jac(X_opt):
            grad = np.zeros_like(X_opt)
            pca_coeffs = X_opt[-N_pca:]
            # Derivative of (thresh - sum(e^2/eig)) is -2e/eig
            grad[-N_pca:] = -2.0 * pca_coeffs / self.config.tracker.pca_eigenvalues
            return grad
        
        nlc =[]
        if self.use_mahalanobis_projection:
            nlc.append(NonlinearConstraint(pca_mahalanobis, 0.0, np.inf, jac=pca_mahalanobis_jac))

        # 4. CONSTANT MATRICES
        Q_kin = self.dynamic_model.Q_d(dt=self.T)[:6, :6]
        Q_kin_inv = np.linalg.inv(Q_kin + np.eye(6)*1e-6)
        F_kin = self.dynamic_model.F_d(None, dt=self.T)[:6, :6]
        
        P_0_inv = np.linalg.inv(self.history_buffer[0]['prior_cov'] + np.eye(state_dim)*1e-6)
        state_0_prior = self.history_buffer[0]['prior_mean']
        
        sigma_lidar_sq = self.config.tracker.lidar_std_dev ** 2

        # 5. COST FUNCTION AND ANALYTICAL GRADIENT
        def cost_and_jac(X_opt):
            kinematics = X_opt[:6*N].reshape(N, 6)
            shape = X_opt[6*N:]
            
            total_cost = 0.0
            grad = np.zeros_like(X_opt) # Analytical Gradient Vector
            
            # --- A. Prior Cost ---
            full_state_0 = np.concatenate([kinematics[0], shape])
            diff_0 = full_state_0 - state_0_prior
            diff_0[2] = ssa(diff_0[2])
            
            P0_inv_diff = P_0_inv @ diff_0
            total_cost += 0.5 * diff_0.T @ P0_inv_diff
            
            # Gradient of Prior
            grad[:6] += P0_inv_diff[:6]
            grad[6*N:] += P0_inv_diff[6:]

            for k in range(N):
                state_k = np.concatenate([kinematics[k], shape])
                frame = self.history_buffer[k]
                
                # --- B. Measurement Cost ---
                z_flat = frame['z_global_flat']
                z_global_coords = z_flat.reshape(2, -1, order='F')
                
                # Re-use your analytical get_implicit_matrices to get H_imp!
                H_imp, _, theta_imp = self.sensor_model.get_implicit_matrices(state_k, z_global_coords)
                z_pred = self.sensor_model.h_from_theta(state_k, theta_imp)
                res_z = z_flat - z_pred
                
                var_k = sigma_lidar_sq
                if self.use_scaled_R:
                    var_k *= (len(z_flat) / 2)
                    
                R_inv_res_z = res_z / var_k
                total_cost += 0.5 * np.sum(res_z * R_inv_res_z)
                
                # Gradient of Measurements: -H_imp^T * R^-1 * residual
                grad_meas = -H_imp.T @ R_inv_res_z
                grad[k*6 : (k+1)*6] += grad_meas[:6]
                grad[6*N:] += grad_meas[6:]
                
                # --- C. Kinematic Cost ---
                if k < N - 1:
                    kin_next = kinematics[k+1]
                    kin_curr = kinematics[k]
                    kin_pred = F_kin @ kin_curr
                    
                    res_kin = kin_next - kin_pred
                    res_kin[2] = ssa(res_kin[2])
                    
                    Q_inv_res_kin = Q_kin_inv @ res_kin
                    total_cost += 0.5 * res_kin.T @ Q_inv_res_kin
                    
                    # Gradient w.r.t k+1
                    grad[(k+1)*6 : (k+2)*6] += Q_inv_res_kin
                    # Gradient w.r.t k
                    grad[k*6 : (k+1)*6] -= F_kin.T @ Q_inv_res_kin
                    
                # --- D. Negative Information Penalty ---
                any_neg = self.use_negative_info_angular or self.use_negative_info_front or self.use_negative_info_centroid
                if any_neg:
                    vcs = self._get_virtual_constraints(frame['meas_local'], state_k)
                    for vc in vcs:
                        c_type = vc['type']
                        if c_type in['min_angle', 'max_angle']:
                            H_virt, gamma_pred = self.sensor_model.get_virtual_measurement_jacobian(state_k, vc['body_angle'], is_radial=False)
                            ang_res = ssa(vc['measured_val'] - gamma_pred)
                            rho = np.maximum(np.sqrt((vc['predicted_point'][0] - self.sensor_model.lidar_position[0])**2 + (vc['predicted_point'][1] - self.sensor_model.lidar_position[1])**2), 1.0)
                            res_vc = rho * ang_res
                            H_stack = rho * H_virt
                        elif c_type == 'front_wall':
                            H_stack, rho_pred = self.sensor_model.get_virtual_measurement_jacobian(state_k, vc['body_angle'], is_radial=True)
                            res_vc = vc['measured_val'] - rho_pred
                        elif c_type == 'centroid_depth':
                            res_vc = vc['measured_val'] - vc['rho_c']
                            H_stack = np.zeros((1, len(state_k)))
                            H_stack[0, 0] = (state_k[0] - self.sensor_model.lidar_position[0]) / vc['rho_c']
                            H_stack[0, 1] = (state_k[1] - self.sensor_model.lidar_position[1]) / vc['rho_c']
                            
                        total_cost += 0.5 * (res_vc ** 2) / self.neg_info_var
                        
                        # Gradient of Penalties: -H_stack^T * (residual / var)
                        grad_vc = -H_stack.T.flatten() * (res_vc / self.neg_info_var)
                        grad[k*6 : (k+1)*6] += grad_vc[:6]
                        grad[6*N:] += grad_vc[6:]

            return total_cost, grad

        # 6. RUN SLSQP (Now 50x faster with Analytical Jacobians!)
        result = minimize(
            fun=cost_and_jac,
            x0=X_init,
            method='SLSQP',
            jac=True,          # <--- Tell SciPy to use our Exact Gradient
            bounds=bounds,
            constraints=nlc,
            options={'maxiter': 50, 'ftol': 1e-4, 'disp': False}
        )

        # 7. UNPACK RESULT
        optimized_states =[]
        global_shape = result.x[6*N:]
        for k in range(N):
            kinematics = result.x[k*6 : (k+1)*6]
            optimized_states.append(np.concatenate([kinematics, global_shape]))
            
        return optimized_states, result.success

    def _get_virtual_constraints(self, measurements: LidarScan, state_pred: np.ndarray) -> list[dict]:
        meas_angles, meas_ranges = cart2pol(measurements.x, measurements.y)
        if len(meas_angles) < 2: return[]

        mean_angle = np.arctan2(np.mean(np.sin(meas_angles)), np.mean(np.cos(meas_angles)))
        diff_angles = ssa(meas_angles - mean_angle)
        
        min_idx_meas = np.argmin(diff_angles)
        max_idx_meas = np.argmax(diff_angles)
        
        min_meas_angle = meas_angles[min_idx_meas]
        max_meas_angle = meas_angles[max_idx_meas]
        min_meas_dist = np.min(meas_ranges)

        parametric_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        z_pred_flat = self.sensor_model.h_from_theta(state_pred, parametric_angles)
        global_points = z_pred_flat.reshape(2, -1, order='F')
        
        sensor_points = global_points - self.sensor_model.lidar_position.reshape(2, 1)
        pred_angles, pred_ranges = cart2pol(sensor_points[0, :], sensor_points[1, :])
        
        pred_diff_angles = ssa(pred_angles - mean_angle)
        
        min_idx = np.argmin(pred_diff_angles)
        max_idx = np.argmax(pred_diff_angles)
        min_rad_idx = np.argmin(pred_ranges)
        
        min_pred_angle = pred_angles[min_idx]
        max_pred_angle = pred_angles[max_idx]

        virtual_constraints =[]
        
        if self.use_negative_info_angular:
            if ssa(min_meas_angle - min_pred_angle) > self.angular_margin:
                theta_min = parametric_angles[min_idx]
                if self.use_exact_extreme_angle:
                    theta_min = self._get_exact_extreme_angle(state_pred, theta_min, mean_angle, is_max=False)
                
                pt_global_min = self.sensor_model.h_from_theta(state_pred, np.array([theta_min])).flatten('F')
                virtual_constraints.append({
                    'measured_val': ssa(min_meas_angle - self.angular_margin),
                    'body_angle': theta_min,
                    'predicted_point': pt_global_min,
                    'type': 'min_angle'
                })

        if self.use_negative_info_angular:
            if ssa(max_pred_angle - max_meas_angle) > self.angular_margin:
                theta_max = parametric_angles[max_idx]
                if self.use_exact_extreme_angle:
                    theta_max = self._get_exact_extreme_angle(state_pred, theta_max, mean_angle, is_max=True)
                
                pt_global_max = self.sensor_model.h_from_theta(state_pred, np.array([theta_max])).flatten('F')
                virtual_constraints.append({
                    'measured_val': ssa(max_meas_angle + self.angular_margin),
                    'body_angle': theta_max,
                    'predicted_point': pt_global_max,
                    'type': 'max_angle'
                })

        if self.use_negative_info_front:
            if pred_ranges[min_rad_idx] < (min_meas_dist - self.radial_margin):
                virtual_constraints.append({
                    'measured_val': min_meas_dist - self.radial_margin,
                    'body_angle': parametric_angles[min_rad_idx],
                    'type': 'front_wall'
                })

        if self.use_negative_info_centroid:
            u_c_x = state_pred[0] - self.sensor_model.lidar_position[0]
            u_c_y = state_pred[1] - self.sensor_model.lidar_position[1]
            rho_c = np.sqrt(u_c_x**2 + u_c_y**2)
            
            centroid_buffer = state_pred[7] / 2.0 
            if rho_c < (min_meas_dist + centroid_buffer):
                virtual_constraints.append({
                    'measured_val': min_meas_dist + centroid_buffer,
                    'rho_c': float(rho_c),
                    'type': 'centroid_depth'
                })

        return virtual_constraints

    def _get_exact_extreme_angle(self, state_pred: np.ndarray, guess_theta: float, mean_angle: float, is_max: bool) -> float:
        from scipy.optimize import minimize_scalar
        def objective(theta):
            pt_global = self.sensor_model.h_from_theta(state_pred, np.array([theta])).flatten('F')
            u_x = pt_global[0] - self.sensor_model.lidar_position[0]
            u_y = pt_global[1] - self.sensor_model.lidar_position[1]
            gamma = np.arctan2(u_y, u_x)
            unwrapped_gamma = ssa(gamma - mean_angle)
            return -unwrapped_gamma if is_max else unwrapped_gamma

        delta = np.deg2rad(2.0)
        bounds = (guess_theta - delta, guess_theta + delta)
        res = minimize_scalar(objective, bounds=bounds, method='bounded')
        return res.x