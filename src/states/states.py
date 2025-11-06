from dataclasses import dataclass, field
from senfuslib import AtIndex, NamedArray, MetaData
import numpy as np

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
    # Kinematic part (6 dimensions)
    x: AtIndex[0]
    y: AtIndex[1]
    yaw: AtIndex[2]
    vel_x: AtIndex[3]
    vel_y: AtIndex[4]
    yaw_rate: AtIndex[5]

    # Extent part (2 dimensions)
    length: AtIndex[6]
    width: AtIndex[7]

    # PCA coefficients part (dynamic size)
    pca_coeffs: AtIndex[slice(8, None)]

    # --- Convenient group accessors ---
    pos: AtIndex[slice(0, 2)] = field(init=False)
    kinematics: AtIndex[slice(0, 6)] = field(init=False)
    extent: AtIndex[slice(6, 8)] = field(init=False)

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

    @property
    def range(self) -> float:
        """Computes the range (distance from the origin)."""
        return np.sqrt(self.x**2 + self.y**2)

    @property
    def angle(self) -> float:
        """Computes the angle in radians."""
        return np.arctan2(self.y, self.x)
