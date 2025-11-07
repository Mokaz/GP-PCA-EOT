import numpy as np
from typing import Any, Dict, Optional, Tuple

from dataclasses import dataclass, field

from src.utils.tools import cart2pol, pol2cart, ssa
from src.states.states import State_PCA

@dataclass
class SimulationConfig:
    name: str = "default_simulation"
    # method : str = "iekf"

    num_simulations: int = 1
    num_frames: int = 100
    dt: float = 0.1
    seed: int = 42

@dataclass
class ExtentConfig:
    N_fourier: int = 64
    d_angle: float = np.deg2rad(1.0)
    shape_params_true: Dict[str, Any] = field(default_factory=lambda: {"type": "ellipse", "L": 20.0, "W": 6.0})

    angles: np.ndarray = field(init=False)
    shape_coords_body: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.angles = np.arange(-np.pi, np.pi, self.d_angle)

        self.shape_coords_body = self.compute_exact_vessel_shape_local_from_params(
            self.angles, self.shape_params_true
        )

    def compute_exact_vessel_shape_local_from_params(self, angles: np.ndarray, shape_params_true: Dict[str, Any]) -> np.ndarray:
        """
        Generates the ground truth Cartesian coordinates of the vessel's shape in local body frame.

        Args:
            angles: Angles at which to compute the vessel shape.
            shape_params_true: A dictionary containing the true shape parameters of the vessel.
        Returns:
            A tuple (shape_x, shape_y) of the vessel's global coordinates.
        """
        radii = np.zeros_like(angles)
        nAngles = angles.size

        # Extract parameters from state and config
        L = shape_params_true.get("L")
        W = shape_params_true.get("W")
        S = shape_params_true.get("S")
        P = shape_params_true.get("P")
        shape_type = shape_params_true.get("type")

        if shape_type == "box":
            theta_0 = np.arctan2(W, L)
            radii = np.zeros(angles.shape)

            for idx, angle in enumerate(angles):
                if angle <= theta_0:
                    radii[idx] = L / 2 / np.cos(angle)
                elif angle <= np.pi - theta_0:
                    radii[idx] = W / 2 / np.cos(np.pi/2 - angle)
                else:
                    radii[idx] = L / 2 / np.cos(np.pi - angle)
        
        elif shape_type == "ellipse":
            x_interpol = np.linspace(-L / 2, L / 2, num=nAngles, endpoint=True)
            y_interpol = (W / 2) * np.sqrt(1 - ((2 / L) * x_interpol)**2)

            angles_interpol, r_interpol = cart2pol(x_interpol, y_interpol)

            radii = np.interp(np.abs(ssa(angles)), angles_interpol, r_interpol, period=2*np.pi)

        elif shape_type == "box_elliptic_sides":
            x_ellipse = np.linspace(-L / 2, L / 2, num=nAngles, endpoint=True)
            y_ellipse = S/2 + ((W - S) / 2) * np.sqrt(1 - ((2 / L) * x_ellipse)**2)

            angles_interpol, r_interpol = cart2pol(x_ellipse, y_ellipse)

            theta_corner = np.arctan2(S, L)
            radii = np.zeros(angles.shape)

            for idx, angle in enumerate(angles):
                if angle <= theta_corner:
                    radii[idx] = (L / 2) / np.cos(angle)
                elif angle <= np.pi - theta_corner:
                    radii[idx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                else:
                    radii[idx] = (L / 2) / np.cos(np.pi - angle)

        elif shape_type == "box_parabolic_bow_and_stern":
            x_interpol = np.linspace((L / 2) - P, L / 2, num=nAngles, endpoint=True)
            y_interpol = (-L**2 * W + 4*L*P*W) / (8 * P**2) + ((L*W - 2*P*W) * x_interpol) / (2*P**2) - (W * x_interpol**2) / (2 * P**2)  
            angles_interpol, r_interpol = cart2pol(x_interpol, y_interpol)
            theta_corner = np.arctan2(W, L - 2*P)
            for idx, angle in enumerate(angles):
                if angle <= theta_corner:
                    radii[idx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                elif angle <= np.pi - theta_corner:
                    radii[idx] = (W / 2) / np.sin(angle)
                else:
                    radii[idx] = np.interp(np.pi - angle, angles_interpol, r_interpol, period=2*np.pi)

        elif shape_type == "elliptic_bow_and_stern":
            y_ellipse = np.linspace(-W / 2, W / 2, num=nAngles, endpoint=True)
            x_ellipse = (L / 2) - P + P * np.sqrt(1 - ((2 / W) * y_ellipse)**2)
            angles_interpol, r_interpol = cart2pol(x_ellipse, y_ellipse)
            theta_corner = np.arctan2(W, L - 2*P)
            for idx, angle in enumerate(angles):
                if angle <= theta_corner:
                    radii[idx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                elif angle <= np.pi - theta_corner:
                    radii[idx] = (W / 2) / np.sin(angle)
                else:
                    radii[idx] = np.interp(np.pi - angle, angles_interpol, r_interpol, period=2*np.pi)

        else:
            raise ValueError(f"Unknown ground truth shape type: {shape_type}")

        # Convert the perfect polar shape to Cartesian coordinates in the body frame
        shape_coords_body = np.stack(pol2cart(angles, radii), axis=0)

        return shape_coords_body


@dataclass
class TrackerConfig:
    N_pca: int = 4

    # Standard deviations for process noise
    pos_north_std_dev: float = 0.3
    pos_east_std_dev: float = 0.3
    heading_std_dev: float = 0.1

    # Standard deviations for measurement noise
    lidar_std_dev: float = 0.15

    # Initial state vector. If not provided, a default is created in __post_init__.
    initial_state: Optional[State_PCA] = None # TODO Martin Consider separating intial state and tracker init state
    initial_std_devs: Optional[State_PCA] = None
    
    # Lidar position, to be set from LidarConfig
    lidar_position: Optional[np.ndarray] = None

    PCA_parameters_path : str = 'data/input_parameters/FourierPCAParameters_scaled.npz'

    def __post_init__(self):
        # Initialize initial_state if it wasn't provided
        if self.initial_state is None:
            self.initial_state = State_PCA(
                x=0.0,
                y=-40.0,
                yaw=np.pi / 2,
                vel_x=0.0,
                vel_y=0.0,
                yaw_rate=0.0,
                length=20.0,
                width=6.0,
                pca_coeffs=np.zeros(self.N_pca)
            )
        
        if self.initial_std_devs is None:
            self.initial_std_devs = State_PCA(
                x=2.0,
                y=2.0,
                yaw=0.2,
                vel_x=2.0,
                vel_y=2.0,
                yaw_rate=0.1,
                length=2.0,
                width=2.0,
                pca_coeffs=np.ones(self.N_pca) * 0.5 
            )

@dataclass
class LidarConfig:
    """
    Configuration parameters for the LiDAR sensor.
    """
    # LiDAR Parameters
    lidar_position: Tuple[float, float] = (30.0, 0.0)
    num_rays: int = 360
    max_distance: float = 140.0
    lidar_noise_mean: float = 0.0
    lidar_std_dev: float = 0.15

@dataclass
class Config:
    sim: SimulationConfig
    lidar: LidarConfig
    tracker: TrackerConfig
    extent: ExtentConfig
