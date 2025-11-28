from dataclasses import dataclass, field
import numpy as np
from src.senfuslib.named_array import NamedArray, AtIndex

@dataclass
class State_PCA(NamedArray):
    """
    Defines the full state vector for the GP-PCA EOT tracker.

    x -- North position
    y -- East position
    yaw -- Heading angle
    vel_x -- Velocity in North direction
    vel_y -- Velocity in East direction
    yaw_rate -- Yaw rate
    length -- Vessel length
    width -- Vessel width
    pca_coeffs -- PCA coefficients for shape representation

    """
    # --- Field definitions are unchanged ---
    x: AtIndex[0]
    y: AtIndex[1]
    yaw: AtIndex[2]
    vel_x: AtIndex[3]
    vel_y: AtIndex[4]
    yaw_rate: AtIndex[5]
    length: AtIndex[6]
    width: AtIndex[7]
    pca_coeff_0: AtIndex[8] = field(init=False)
    pca_coeff_1: AtIndex[9] = field(init=False)
    pca_coeff_2: AtIndex[10] = field(init=False)
    pca_coeff_3: AtIndex[11] = field(init=False) # TODO Martin Adjust number of PCA coeffs as needed

    # --- Replace the custom __init__ with a custom __new__ method ---
    def __new__(cls, x, y, yaw, vel_x, vel_y, yaw_rate, length, width, pca_coeffs):
        # Assemble the full state array from the input arguments
        kinematics = np.array([x, y, yaw, vel_x, vel_y, yaw_rate])
        extent = np.array([length, width])
        pca_coeffs = np.atleast_1d(pca_coeffs)
        full_state = np.concatenate([kinematics, extent, pca_coeffs])

        # Create the final object by viewing the data as an instance of this class
        obj = np.asarray(full_state).view(cls)
        return obj


    # --- Convenient group accessors ---
    pos: AtIndex[slice(0, 2)] = field(init=False)
    kinematics: AtIndex[slice(0, 6)] = field(init=False)
    extent: AtIndex[slice(6, 8)] = field(init=False)
    pca_coeffs: AtIndex[slice(8, None)]

@dataclass
class LidarScan(NamedArray):
    """
    Represents a LiDAR scan in the sensor's local Cartesian coordinate frame.
    The origin (0,0) is the position of the LiDAR sensor.

    x -- np.array of x-coordinates of LiDAR points (in meters)
    y -- np.array of y-coordinates of LiDAR points (in meters)
    """
    x: AtIndex[0]
    y: AtIndex[1]

    # --- Add a custom __new__ method to correctly construct the array ---
    def __new__(cls, x: np.ndarray, y: np.ndarray):
        # Ensure inputs are arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        # Stack the coordinates into a single (2, N) array
        full_scan_data = np.stack([x, y], axis=0)

        # Create the final object by viewing the data as an instance of this class
        obj = np.asarray(full_scan_data).view(cls)
        return obj

    @property
    def range(self) -> np.ndarray:
        """Computes the range (distance from the origin) for all points."""
        return np.sqrt(self.x**2 + self.y**2)

    @property
    def angle(self) -> np.ndarray:
        """Computes the angle in radians for all points."""
        return np.arctan2(self.y, self.x)

@dataclass
class State_GP(NamedArray):
    """
    Defines the full state vector for the GP-EOT tracker (Radial Basis).

    x -- North position
    y -- East position
    yaw -- Heading angle
    vel_x -- Velocity in North direction
    vel_y -- Velocity in East direction
    yaw_rate -- Yaw rate
    radii -- The radial extent states (radius at each test angle)
    """
    x: AtIndex[0]
    y: AtIndex[1]
    yaw: AtIndex[2]
    vel_x: AtIndex[3]
    vel_y: AtIndex[4]
    yaw_rate: AtIndex[5]
    
    # The radii are dynamic in length (N_gp), starting at index 6
    radii: AtIndex[slice(6, None)]

    # --- Convenient group accessors ---
    pos: AtIndex[slice(0, 2)] = field(init=False)
    kinematics: AtIndex[slice(0, 6)] = field(init=False)

    def __new__(cls, x, y, yaw, vel_x, vel_y, yaw_rate, radii):
        kinematics = np.array([x, y, yaw, vel_x, vel_y, yaw_rate], dtype=float)
        radii = np.atleast_1d(radii).astype(float)
        full_state = np.concatenate([kinematics, radii])
        return np.asarray(full_state).view(cls)