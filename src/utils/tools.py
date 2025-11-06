import numpy as np
from numpy.linalg import norm
import random
from typing import Tuple, List, Optional, Callable
from enum import IntEnum

def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    return angle, r

def pol2cart(angle, r):
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return x, y

def ssa(angle: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the Smallest Signed Angle, wrapping the input to the range [-pi, pi].
    This function is vectorized and works on both scalars and NumPy arrays.
    """
    return np.pi - np.mod(np.pi - angle, 2 * np.pi)

def rot2D(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])

def drot2D(angle):
    return np.array([[-np.sin(angle), -np.cos(angle)],
                     [np.cos(angle),  -np.sin(angle)]])

def ur(angles):
    return np.stack([[np.cos(angles)], [np.sin(angles)]], axis=1) 

def ut(angles):
    return np.stack([[-np.sin(angles)], [np.cos(angles)]], axis=1)

def isPositiveDefinite(A):
    return np.all(np.linalg.eigvals(A) >= -1e-6)

def generate_fourier_function(N_f: int) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a vectorized g(theta) function for multiple angles.

    Parameters:
    N_f (int): Number of Fourier coefficients.

    Returns:
    Callable[[np.ndarray], np.ndarray]: A function g(angles) that returns an (N_f, N) array,
        where N is the number of angles.
    """
    def g_vectorized(angles: np.ndarray) -> np.ndarray:
        angles = np.atleast_1d(angles)  # ensure angles is an array
        n = np.arange(1, N_f).reshape(-1, 1)  # shape (N_f - 1, 1)
        cos_terms = np.cos(n * angles.reshape(1, -1))  # shape (N_f - 1, N)
        constant = np.full((1, angles.shape[0]), 0.5)  # shape (1, N)
        return np.vstack((constant, cos_terms))  # shape (N_f, N)

    return g_vectorized

def fourier_basis_matrix(angles: np.ndarray, N_fourier: int) -> np.ndarray:
    """
    Computes the Fourier basis matrix g(theta) for a given set of angles. Assumes symmetry in vessel shape and thus only cosine terms are used.

    Parameters:
    angles (np.ndarray): A 1D array of angles.
    N_fourier (int): The number of Fourier coefficients.

    Returns:
    np.ndarray: An (N_fourier, N) array, where N is the number of angles.
    """
    angles = np.atleast_1d(angles)  # ensure angles is an array
    
    # The first coefficient is a constant
    constant_term = np.full((1, angles.shape[0]), 0.5)
    if N_fourier == 1:
        return constant_term

    # The rest are cosine terms
    n = np.arange(1, N_fourier).reshape(-1, 1)
    cos_terms = np.cos(n * angles.reshape(1, -1))

    return np.vstack((constant_term, cos_terms))

def fourier_transform(angles: np.ndarray, func: np.ndarray, num_coeff: int = 64, symmetry: bool = True) -> np.ndarray:
    """
    Perform Fourier transform on a given function sampled at specific angles.

    Args:
        angles (np.ndarray): Array of angles at which the function is sampled.
        func (np.ndarray): Array of function values corresponding to the angles.
        num_coeff (int, optional): Number of Fourier coefficients to compute. Defaults to 64.
        symmetry (bool, optional): Whether to return only the symmetric part of the coefficients. Defaults to True.

    Returns:
        np.ndarray: Array of Fourier coefficients.
    """
    f_sample = 2 * num_coeff
    angles_fft = np.linspace(-np.pi, np.pi, f_sample, endpoint=False)
    r_fft = np.interp(angles_fft, angles, func, period=2 * np.pi)

    y = np.fft.rfft(r_fft) / angles_fft.size
    y *= 2
    a0 = y[0].real
    a = y[1:-1].real
    b = -y[1:-1].imag

    if symmetry:
        return np.concatenate((a0, a), axis=None).reshape((-1, 1))
    else:
        return np.concatenate((a0, a, b), axis=None).reshape((-1, 1))
    
def get_intersection(ray_origin: Tuple[float, float], 
                     ray_direction: Tuple[float, float], 
                     segment_start: Tuple[float, float], 
                     segment_end: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """
    Calculate the intersection point between a ray and a line segment.

    Args:
        ray_origin (Tuple[float, float]): The origin of the ray.
        ray_direction (Tuple[float, float]): The direction of the ray (normalized).
        segment_start (Tuple[float, float]): Start point of the line segment.
        segment_end (Tuple[float, float]): End point of the line segment.

    Returns:
        Optional[Tuple[float, float]]: The intersection point (x, y) if it exists; otherwise, None.
    """
    r_px, r_py = ray_origin
    r_dx, r_dy = ray_direction
    s_px, s_py = segment_start
    s_dx, s_dy = segment_end[0] - s_px, segment_end[1] - s_py

    # Solve for intersection
    denominator = r_dx * s_dy - r_dy * s_dx

    # Handle very small values of denominator
    if abs(denominator) < 1e-10:  # Use a small tolerance to avoid ambiguity
        return None  # Parallel lines

    t = ((s_px - r_px) * s_dy - (s_py - r_py) * s_dx) / denominator
    u = ((r_px - s_px) * r_dy - (r_py - s_py) * r_dx) / (-denominator)

    if t >= 0 and u >= 0 and u <= 1:
        # Calculate intersection point
        intersect_x = r_px + t * r_dx
        intersect_y = r_py + t * r_dy
        return (intersect_x, intersect_y)
    return None

def cast_rays(lidar_pos: Tuple[float, float], 
              num_rays: int, 
              max_dist: float, 
              obstacle_x: List[float], 
              obstacle_y: List[float]
             ):
    """
    Cast rays from a LiDAR position and find intersections with obstacles.

    Args:
        lidar_pos (Tuple[float, float]): The position of the LiDAR sensor.
        num_rays (int): Number of rays to cast.
        max_dist (float): Maximum distance a ray can travel.
        obstacle_x (List[float]): List of x coordinates of the obstacle vertices.
        obstacle_y (List[float]): List of y coordinates of the obstacle vertices.

    Returns:
        Tuple[List[List[Optional[Tuple[float, float]], float]], np.ndarray]: 
            - A list of lists. Each inner list contains:
                - An optional tuple with the coordinates of the intersection point (or None if no intersection).
                - A float representing the distance to the intersection or max_dist if no intersection occurred.
            - A numpy array of angles corresponding to each ray.
    """
    angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
    dist_meas = []
    angle_meas = []

    # Combine obstacle_x and obstacle_y into a list of vertices
    vertices = list(zip(obstacle_x, obstacle_y))

    # Create obstacle line segments from vertices
    segments = [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]

    for angle in angles:
        ray_dir = (np.cos(angle), np.sin(angle))
        closest_dist = max_dist
        # closest_point = None
        is_intersect = False

        # Cast ray against each obstacle segment
        for segment_start, segment_end in segments:
            intersect = get_intersection(lidar_pos, ray_dir, segment_start, segment_end)
            if intersect:
                dist = np.sqrt((intersect[0] - lidar_pos[0]) ** 2 + (intersect[1] - lidar_pos[1]) ** 2)
                if dist < closest_dist:
                    closest_dist = dist
                    # closest_point = intersect
                    is_intersect = True

        # Append as a list instead of a tuple
        if is_intersect:
            angle_meas.append(angle)
            dist_meas.append(closest_dist)

    return angle_meas, dist_meas

def add_noise_to_distances(rng: np.random.Generator,
                           distances: List[float],
                           angles: List[float],
                           noise_mean: float = 0.0,
                           noise_std_dev: float = 0.5) -> List:
    """
    Add Gaussian noise to the distances measured by LiDAR and shift the points accordingly.

    Args:
        distances (List[float]): Original distances from LiDAR.
        angles (List[float]): Angles of the LiDAR rays.
        noise_mean (float): Mean of the Gaussian noise.
        noise_std_dev (float): Standard deviation of the Gaussian noise.

    Returns:
        List[Tuple[Optional[Tuple[float, float]], float]]: Distances and points with added noise.
    """
    noisy_measurements = []
    
    for angle, dist in zip(angles, distances):
        noise = rng.normal(noise_mean, noise_std_dev)
        noisy_dist = dist + noise
        noisy_measurements.append([angle, noisy_dist])
    
    return noisy_measurements
    
def compute_angle_range(angles):
    """
    Finds the smallest enclosing angular interval that contains all angles.

    Parameters:
    - angles: Array of angles in radians.

    Returns:
    - lower_diff: difference between the minimum angle and the mean angle.
    - upper_diff: difference between the maximum angle and the mean angle.
    - alpha_mean: Mean angle in the wrapped sense.
    """

    # Unwrap angles to avoid 2pi jumps
    angles_unwrapped = np.unwrap(angles)

    # Compute mean of unwrapped angles
    mean_angle_unwrapped = np.mean(angles_unwrapped)

    # Calculate differences relative to mean (already unwrapped)
    angle_diffs = angles_unwrapped - mean_angle_unwrapped

    # Find min and max deviation
    lower_diff = np.min(angle_diffs)
    upper_diff = np.max(angle_diffs)

    # Wrap mean angle back to [-pi, pi]
    alpha_mean = ssa(mean_angle_unwrapped)

    return lower_diff, upper_diff, alpha_mean

def initialize_centroid(position, lidar_pos, measurements, L_est, W_est, min_xf=1.0):
    """
    Initialize the centroid (xc) for optimization with constraints:
    - The centroid's angle should be within the min/max angular bounds.
    - The centroid's radial distance should be greater than the closest measurement.

    Parameters:
    - measurements: list of tuples, where each tuple is (angle, distance) from the sensor
    - min_rz: minimum radial distance from the sensor (closest measurement)
    - min_xf: minimum offset added to the radial distance (additional safety margin)

    Returns:
    - xc: the initialized centroid (xp, yp) in Cartesian coordinates
    """

    relative_position = np.array(position) - np.array(lidar_pos)
    angle_est, distance_est = cart2pol(*relative_position)

    # Extract the angles and distances from the measurements
    angles = np.array([ssa(measurement[0]) for measurement in measurements])
    distances = np.array([measurement[1] for measurement in measurements])

    # Calculate the minimum radial distance from the sensor
    min_rz = np.min(distances)

    # Update the minimum offset if the current minimum is greater
    min_xf = max(min_xf, min(L_est / 2, W_est / 2))

    # Calculate the minimum and maximum angles from the measurements
    lower_diff, upper_diff, alpha_mean = compute_angle_range(angles)
    
    # Calculate the radial distance for the centroid
    min_distance = min_rz + min_xf
    
    if ssa((alpha_mean + lower_diff) - angle_est) * ssa((alpha_mean + upper_diff) - angle_est) < 0:
        angle_est = angle_est
    elif ssa((alpha_mean + lower_diff) - angle_est) > 0:
        angle_est = alpha_mean + lower_diff
    elif ssa((alpha_mean + upper_diff) - angle_est) < 0:
        angle_est = alpha_mean + upper_diff
    else:
        angle_est = alpha_mean
    
    distance_est = max(distance_est, min_distance)

    # Convert the centroid to global cartesian coordinates
    xc_x, xc_y = pol2cart(angle_est, distance_est)
    xc_x += lidar_pos[0]
    xc_y += lidar_pos[1]

    return xc_x, xc_y

    # TODO: make sure its correct

def compute_iou_radial(r1, r2, theta=None):
    """
    Compute the IoU (Intersection over Union) between two radial extent functions in polar coordinates.

    Parameters:
    - r1, r2: numpy arrays of radial distances (same length)
    - theta: array of angles corresponding to the radial values (optional, defaults to 0 to 2π)

    Returns:
    - iou: scalar value of the intersection over union
    """
    if theta is None:
        theta = np.linspace(0, 2 * np.pi, len(r1), endpoint=False)

    # Compute differential angle element
    dtheta = theta[1] - theta[0]

    # Intersection and Union areas (using polar area formula)
    intersection_area = 0.5 * np.sum(np.minimum(r1, r2)**2) * dtheta
    union_area = 0.5 * np.sum(np.maximum(r1, r2)**2) * dtheta

    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def calculate_body_angles(lidar_measurements: np.ndarray, state: np.ndarray) -> np.ndarray:
    """
    Calculate the angles of LiDAR measurements in the body frame of an object.

    Args:
        lidar_measurements (np.ndarray): A Nx2 array of LiDAR measurement points (x, y).
        state (np.ndarray): A 1D array representing the object's state
                                    [x_pos, y_pos, heading].

    Returns:
        np.ndarray: A 1D array of angles in the body frame.
    """
    # Calculate angles of measurements relative to the object's position in the world frame
    world_angles = np.arctan2(
        lidar_measurements[:, 1] - state[1],
        lidar_measurements[:, 0] - state[0]
    )
    
    # Transform world frame angles to body frame angles by subtracting the object's heading
    body_angles = ssa(world_angles - state[2])
    
    return body_angles

class StateIdxToName(IntEnum):
    """
    Enum mapping state vector indices to human-readable names.
    """
    X_POS = 0
    Y_POS = 1
    HEADING = 2
    X_VEL = 3
    Y_VEL = 4
    YAW_RATE = 5
    LENGTH = 6
    WIDTH = 7
    PCA_COMPONENTS = 8  # Starting index for PCA components

    def __new__(cls, value):
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj

    @property
    def name(self):
        if self.value >= self.PCA_COMPONENTS:
            return f"PCA_{self.value - self.PCA_COMPONENTS + 1}"
        return super().name

    def __str__(self):
        return self.name
