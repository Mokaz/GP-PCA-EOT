from dataclasses import dataclass, field
from typing import Generic, Optional, Sequence, TypeVar
import numpy as np
from senfuslib import DynamicModel
from scipy.linalg import expm

from states.states import State_PCA
from utils.tools import rot2D, ssa

S = TypeVar('S', bound=np.ndarray)  # State type

@dataclass
class Model_PCA_CV(DynamicModel):
    """
    Implements the decoupled Constant Velocity process model for the full GP-PCA state.
    """
    # Noise parameters
    x_pos_std_dev: float
    y_pos_std_dev: float
    yaw_std_dev: float

    # Number of PCA components
    N_pca: int

    def F_d(self, x: State_PCA, dt: float) -> np.ndarray:
        """
        Discrete-time state transition jacobian (F).
        """
        F = np.eye(8 + self.N_pca)
        t = np.array([[1, dt], [0, 1]])
        # Create the 6x6 kinematic block
        F_kin = np.kron(t, np.eye(3))
        F[:6, :6] = F_kin
        return F

    def Q_d(self, x: State_PCA, dt: float) -> np.ndarray:
        """
        Discrete-time process noise covariance (Q).
        """
        Q_c_matrix = np.diag([
            self.x_pos_std_dev**2, 
            self.y_pos_std_dev**2, 
            self.yaw_std_dev**2
        ])
        
        t_int = np.array([[(dt**3)/3, (dt**2)/2], [(dt**2)/2, dt]])
        Qk = np.kron(t_int, Q_c_matrix)
        
        # Assume zero process noise on extent and PCA coefficients TODO Martin: investigate this
        Q_extent_pca = np.zeros((2 + self.N_pca, 2 + self.N_pca))
        
        # Combine into the full Q matrix
        Q = np.block([
            [Qk,                  np.zeros((6, 2 + self.N_pca))],
            [np.zeros((2 + self.N_pca, 6)), Q_extent_pca]
        ])
        return Q

@dataclass
class GroundTruthModel(DynamicModel):
    rng: np.random.Generator

    # Noise parameters
    yaw_rate_std_dev: float = 0.01

    _initial_speed: Optional[float] = field(default=None, init=False, repr=False)

    def step_simulation(self, x: State_PCA, dt: float) -> State_PCA:
        """
        Propagates the ground truth state forward one time step.
        This model now derives its speed from the initial state's velocity.
        """
        new_state = x.copy()

        # On the very first step, calculate and store the initial speed.
        if self._initial_speed is None:
            self._initial_speed = np.hypot(x.vel_x, x.vel_y)

        # 1. Add true process noise (random fluctuation in yaw rate)
        true_yaw_rate = self.rng.normal(0.0, self.yaw_rate_std_dev)
        new_state.yaw_rate = true_yaw_rate

        # 2. Propagate the state forward using the non-linear motion equations
        new_state.yaw = ssa(x.yaw + new_state.yaw_rate * dt)
        
        # 3. Velocity is now determined by the stored initial speed and the new heading
        velocity_in_body_frame = np.array([self._initial_speed, 0.0])
        new_state.vel_x, new_state.vel_y = rot2D(new_state.yaw) @ velocity_in_body_frame

        # 4. Update position using an exact discrete-time step
        new_state.x += new_state.vel_x * dt
        new_state.y += new_state.vel_y * dt
        
        return new_state