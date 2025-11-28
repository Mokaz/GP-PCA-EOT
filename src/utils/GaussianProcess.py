import numpy as np
from numba import njit
from dataclasses import dataclass, field

@njit(cache=True)
def ssa_scalar_jit(theta):
    """
    Self-contained SSA for Numba functions to avoid dependency issues.
    """
    return ((theta + np.pi) % (2 * np.pi)) - np.pi

@njit(cache=True)
def covariance_function_numba(theta1, theta2, signal_var, bias_var, noise_var, lengthscale, symmetric: bool):
    """
    Calculates the Periodic Kernel value between two angles.
    """
    if symmetric:
        # For symmetric objects (ships/cars), 180 deg rotation should yield same radius
        s1 = abs(ssa_scalar_jit(theta1))
        s2 = abs(ssa_scalar_jit(theta2))
        # Squared Exponential Kernel on the mapped angles
        k = signal_var * np.exp(-(1 / (2 * lengthscale ** 2)) * (s1 - s2) ** 2)
    else:
        # Standard Periodic Kernel
        k = signal_var * np.exp(-(2 / lengthscale ** 2) * np.sin((theta1 - theta2) / 2) ** 2)
    
    k += bias_var
    
    # Add noise variance (nugget) only on the diagonal (when angles are identical)
    if np.abs(theta1 - theta2) < 1e-16:
        k += noise_var
    return k

@njit(cache=True)
def d_covariance_function_dtheta1_numba(theta1, theta2, signal_var, lengthscale, symmetric: bool):
    """
    Calculates the derivative of the kernel w.r.t theta1. 
    Needed for the Jacobian matrix (H).
    """
    if symmetric:
        s1 = abs(ssa_scalar_jit(theta1))
        s2 = abs(ssa_scalar_jit(theta2))
        k = signal_var * np.exp(-(1 / (2 * lengthscale ** 2)) * (s1 - s2) ** 2)
        # Chain rule derivative
        dkdtheta1 = k * (-1 / lengthscale ** 2) * (s1 - s2) * np.sign(ssa_scalar_jit(theta1))
    else:
        k = signal_var * np.exp(-(2 / lengthscale ** 2) * np.sin((theta1 - theta2) / 2) ** 2)
        dkdtheta1 = k * (-1 / lengthscale ** 2) * np.sin(theta1 - theta2)
    return dkdtheta1

@njit(cache=True)
def build_covariance_matrix(thetas1, thetas2, signal_var, bias_var, noise_var, lengthscale, symmetric: bool):
    """
    Builds the full covariance matrix K between two sets of angles.
    """
    N1 = len(thetas1)
    N2 = len(thetas2)
    K = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            K[i, j] = covariance_function_numba(thetas1[i], thetas2[j],
                                                signal_var, bias_var, noise_var,
                                                lengthscale, symmetric)
    return K

@njit(cache=True)
def build_d_covariance_matrix(thetas1, thetas2, signal_var, lengthscale, symmetric: bool):
    """
    Builds the derivative matrix dK/dtheta1.
    """
    N1 = len(thetas1)
    N2 = len(thetas2)
    dK = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            dK[i, j] = d_covariance_function_dtheta1_numba(thetas1[i], thetas2[j],
                                                           signal_var, lengthscale, symmetric)
    return dK

class GaussianProcess:
    """
    Utility class for handling Gaussian Process Radial Function math.
    Stores the configuration and the inverse covariance of the state angles (Ktt_inv).
    """
    def __init__(self, 
                 n_test_points: int,
                 length_scale: float,
                 signal_var: float, 
                 bias_var: float = 0.0, 
                 noise_var: float = 1e-4, 
                 symmetric: bool = True):
        
        self.n_test_points = n_test_points
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.bias_var = bias_var
        self.noise_var = noise_var
        self.symmetric = symmetric

        # Define the fixed angles for the state vector (e.g., 0, 18, 36... degrees)
        self.theta_test = np.linspace(0, 2 * np.pi, n_test_points, endpoint=False)
        
        # Precompute Ktt (Covariance of test points with themselves)
        self.Ktt = build_covariance_matrix(
            self.theta_test, self.theta_test,
            self.signal_var, self.bias_var, self.noise_var, 
            self.length_scale, self.symmetric
        )

        # Precompute Inverse Ktt (Used heavily in interpolation/Kriging)
        # Using pinv for stability, though inv is usually fine with the noise_var nugget
        self.Ktt_inv = np.linalg.inv(self.Ktt)

    def F(self, dt: float, forgetting_factor: float) -> np.ndarray:
        """
        Transition Matrix for the extent states (Ornstein-Uhlenbeck process).
        """
        return np.exp(-dt * forgetting_factor) * np.eye(self.n_test_points)

    def Q(self, dt: float, forgetting_factor: float) -> np.ndarray:
        """
        Process Noise Matrix for the extent states.
        """
        return (1 - np.exp(-2 * dt * forgetting_factor)) * self.Ktt

    def compute_k_vector(self, query_angles: np.ndarray) -> np.ndarray: # Likely wrong
        """
        Computes the covariance vector 'k' between arbitrary query angles and 
        the fixed state angles (theta_test).
        
        Args:
            query_angles: array of shape (N,)
        Returns:
            matrix of shape (N, n_test_points)
        """
        query_angles = np.atleast_1d(query_angles)
        return build_covariance_matrix(
            query_angles, self.theta_test,
            self.signal_var, self.bias_var, self.noise_var,
            self.length_scale, self.symmetric
        )

    def compute_dk_vector(self, query_angles: np.ndarray) -> np.ndarray:
        """
        Computes the derivative covariance vector 'dk/dtheta' between 
        query angles and state angles.
        
        Args:
            query_angles: array of shape (N,)
        Returns:
            matrix of shape (N, n_test_points)
        """
        query_angles = np.atleast_1d(query_angles)
        return build_d_covariance_matrix(
            query_angles, self.theta_test,
            self.signal_var, self.length_scale, self.symmetric
        )