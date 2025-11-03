import numpy as np

from src.dynamics.vessel import Vessel

from src.utils.tools import rot2D, generate_fourier_function

def get_vessel_shape(vessel: Vessel):
    shape_coords = vessel.extent.cartesian
    shape_coords = np.matmul(rot2D(vessel.kinematic_state.yaw), shape_coords)
    shape_x = shape_coords[0] + vessel.kinematic_state.pos[0]
    shape_y = shape_coords[1] + vessel.kinematic_state.pos[1]
    return shape_x, shape_y

def compute_estimated_shape(tracker, angles):
    L = tracker.state[6]
    W = tracker.state[7]

    est_shape_coords = np.array([
        (tracker.g(angle).T @ (tracker.fourier_coeff_mean + tracker.M @ tracker.state[8:].reshape(-1, 1))).item()
        for angle in angles
    ])
    est_shape_coords_x = est_shape_coords * L * np.cos(angles)
    est_shape_coords_y = est_shape_coords * W * np.sin(angles)
    est_shape_coords = np.stack([est_shape_coords_x, est_shape_coords_y], axis=0)
    est_shape_coords = np.matmul(rot2D(tracker.state[2]), est_shape_coords)
    return est_shape_coords[0] + tracker.state[0], est_shape_coords[1] + tracker.state[1]

def compute_estimated_shape_from_params(x_pos_est, y_pos_est, heading_est, L, W, PCA_coeffs, fourier_coeff_mean, PCA_eigenvectors_M, angles):
    g_func = generate_fourier_function(N_f=64) # TODO Martin Hardcoded value
    est_shape_coords = np.array([
        (g_func(angle).T @ (fourier_coeff_mean + PCA_eigenvectors_M @ PCA_coeffs.reshape(-1, 1))).item()
        for angle in angles
    ])
    est_shape_coords_x = est_shape_coords * L * np.cos(angles)
    est_shape_coords_y = est_shape_coords * W * np.sin(angles)
    est_shape_coords = np.stack([est_shape_coords_x, est_shape_coords_y], axis=0)
    est_shape_coords = np.matmul(rot2D(heading_est), est_shape_coords)
    return est_shape_coords[0] + x_pos_est, est_shape_coords[1] + y_pos_est