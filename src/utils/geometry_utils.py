import numpy as np

from src.utils.tools import rot2D, generate_fourier_function, pol2cart
from src.states.states import State_PCA, State_GP

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

def compute_estimated_shape_from_params(x_pos_est, y_pos_est, heading_est, L, W, PCA_coeffs, fourier_coeff_mean, PCA_eigenvectors_M, angles, N_fourier): # TODO OLD
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

def compute_estimated_shape_global(state, config, pca_params=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the global (North, East) coordinates of the estimated shape boundary
    for either State_PCA or State_GP.
    """
    
    if isinstance(state, State_GP):
        # 1. Get Radii
        radii = state.radii
        N_points = len(radii)
        
        # 2. Reconstruct Angles (Assumes uniform distribution 0 to 2pi)
        angles = np.linspace(0, 2 * np.pi, N_points, endpoint=False)
        
        # 3. Convert to Body Cartesian
        bx, by = pol2cart(angles, radii)

        # NOTE Hacky FIX: Close the loop
        bx = np.append(bx, bx[0])
        by = np.append(by, by[0])

        body_points = np.vstack([bx, by])
        
        # 4. Transform to Global
        pos = state.pos.reshape(2, 1)
        R = rot2D(state.yaw)
        
        global_points = pos + R @ body_points
        return global_points[0, :], global_points[1, :]

    elif isinstance(state, State_PCA):
        # Extract params
        L = state.length
        W = state.width
        pca_coeffs = state.pca_coeffs
        
        # Extract config params
        angles = config.extent.angles
        N_fourier = config.extent.N_fourier
        
        # PCA projection vectors
        fourier_coeff_mean = pca_params['mean']
        PCA_eigenvectors_M = pca_params['eigenvectors'][:, :len(pca_coeffs)].real

        # Reconstruct shape function (Fourier)
        g_func = generate_fourier_function(N_f=N_fourier)
        
        # Vectorized calculation of shape radius factor
        # fourier_coeffs = mean + EigenVectors * StateCoeffs
        f_coeffs = fourier_coeff_mean + PCA_eigenvectors_M @ pca_coeffs.reshape(-1, 1)
        
        # Compute radius factor for all angles: g(theta)^T * f_coeffs
        # g_all shape: (N_f, N_angles)
        g_all = g_func(angles) 
        radius_factors = (g_all.T @ f_coeffs).flatten()

        # Scale by L/W to get body coordinates
        bx = radius_factors * L * np.cos(angles)
        by = radius_factors * W * np.sin(angles)
        body_points = np.vstack([bx, by])

        # Transform to Global
        pos = state.pos.reshape(2, 1)
        R = rot2D(state.yaw)
        
        global_points = pos + R @ body_points
        return global_points[0, :], global_points[1, :]

    else:
        raise TypeError(f"Unknown state type for shape reconstruction: {type(state)}")
    
def compute_exact_vessel_shape_global(state: State_PCA, shape_coords_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape_coords_oriented = rot2D(state.yaw) @ shape_coords_body

    shape_x_world = shape_coords_oriented[0] + state.x
    shape_y_world = shape_coords_oriented[1] + state.y

    return shape_x_world, shape_y_world