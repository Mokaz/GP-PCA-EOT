from dataclasses import dataclass, field
from typing import Optional, TypeVar
import numpy as np
from scipy.linalg import block_diag

from src.senfuslib import DynamicModel
from src.states.states import State_PCA, State_GP
from src.utils.tools import rot2D, ssa

from src.utils.GaussianProcess import GaussianProcess

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

    def Q_d(self, x: State_PCA = None, dt: float = 0.1) -> np.ndarray:
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
    rng: np.random.Generator = field(repr=False)

    # Noise parameters
    yaw_rate_std_dev: float = 0.01 # TODO Make configurable

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

@dataclass
class Model_GP_CV(DynamicModel):
    """
    Implements the decoupled Constant Velocity process model for the full GP-EOT state.
    Kinematics evolve via CV, Extent evolves via an Ornstein-Uhlenbeck process.
    """
    # GP Utility class (holds the math for F and Q of the extent)
    gp_utils: GaussianProcess
    
    # Noise parameters
    x_pos_std_dev: float
    y_pos_std_dev: float
    yaw_std_dev: float
    
    # Extent parameters
    forgetting_factor: float

    def F_d(self, x: State_GP, dt: float) -> np.ndarray:
        """
        Discrete-time state transition jacobian (F).
        """
        # 1. Kinematic Transition (Constant Velocity)
        # Order: [x, y, yaw, vx, vy, yaw_rate]
        # Maps to: pos += vel*dt, vel += 0
        t = np.array([[1, dt], [0, 1]])
        F_kin = np.kron(t, np.eye(3)) # 6x6 Matrix
        
        # 2. Extent Transition (GP Forgetting Factor)
        F_gp = self.gp_utils.F(dt, self.forgetting_factor)
        
        # 3. Combine into full block diagonal matrix
        F = block_diag(F_kin, F_gp)
        
        return F

    def Q_d(self, x: State_GP = None, dt: float = 0.1) -> np.ndarray:
        """
        Discrete-time process noise covariance (Q).
        """
        # 1. Kinematic Noise
        # Integrated White Noise assumption
        Q_c_matrix = np.diag([
            self.x_pos_std_dev**2, 
            self.y_pos_std_dev**2, 
            self.yaw_std_dev**2
        ])
        
        t_int = np.array([[(dt**3)/3, (dt**2)/2], [(dt**2)/2, dt]])
        Q_kin = np.kron(t_int, Q_c_matrix) # 6x6 Matrix
        
        # 2. Extent Noise
        # Noise injected to maintain the stationary variance of the GP
        Q_gp = self.gp_utils.Q(dt, self.forgetting_factor)
        
        # 3. Combine
        Q = block_diag(Q_kin, Q_gp)
        
        return Q