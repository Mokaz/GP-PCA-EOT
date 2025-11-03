import numpy as np

from utils.tools import ssa, rot2D

class KinematicState:
    def __init__(self) -> None:
        self.pos = np.array([0.0, -40.0])
        self.yaw = ssa(np.pi / 2)

        self.vel = np.matmul(rot2D(self.yaw), np.array([3.0, 0.0]))
        self.yawrate = 0.0

        self.accel = np.array([0.0, 0.0])
    
    def step(self, T: float, rng: np.random.Generator) -> None:
        self.yawrate = rng.normal(0.0, 0.1)

        self.yaw = ssa(self.yaw + self.yawrate * T) 

        self.vel = np.matmul(rot2D(self.yaw), np.array([3.0, 0.0])) + self.accel * T
        
        self.pos += self.vel * T
        pass