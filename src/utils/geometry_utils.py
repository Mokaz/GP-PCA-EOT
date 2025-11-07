import numpy as np

from src.utils.tools import rot2D, generate_fourier_function
from src.states.states import State_PCA

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

def compute_estimated_shape_from_params(x_pos_est, y_pos_est, heading_est, L, W, PCA_coeffs, fourier_coeff_mean, PCA_eigenvectors_M, angles, N_fourier):
    g_func = generate_fourier_function(N_f=N_fourier)
    est_shape_coords = np.array([
        (g_func(angle).T @ (fourier_coeff_mean + PCA_eigenvectors_M @ PCA_coeffs.reshape(-1, 1))).item()
        for angle in angles
    ])
    est_shape_coords_x = est_shape_coords * L * np.cos(angles)
    est_shape_coords_y = est_shape_coords * W * np.sin(angles)
    est_shape_coords = np.stack([est_shape_coords_x, est_shape_coords_y], axis=0)
    est_shape_coords = np.matmul(rot2D(heading_est), est_shape_coords)
    return est_shape_coords[0] + x_pos_est, est_shape_coords[1] + y_pos_est

def compute_exact_vessel_shape_global(state: State_PCA, shape_coords_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape_coords_oriented = rot2D(state.yaw) @ shape_coords_body

    shape_x_world = shape_coords_oriented[0] + state.x
    shape_y_world = shape_coords_oriented[1] + state.y

    return shape_x_world, shape_y_world