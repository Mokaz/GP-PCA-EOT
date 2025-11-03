import sys
import os
import numpy as np 
sys.path.append(os.path.abspath('../'))

from src.extent_model.extent import PCAExtentModel, Extent
from src.dynamics.kinematic_state import KinematicState

class Vessel:
    def __init__(self, extent: Extent, extent_model: PCAExtentModel, kinematic_state: KinematicState):
        self.extent = extent
        self.kinematic_state = kinematic_state
        self.extent_model = extent_model

    def step(self, T: float, rng: np.random.Generator):
        self.kinematic_state.step(T, rng)

    def get_state(self):
        return np.array([*self.kinematic_state.pos, self.kinematic_state.yaw,
                         *self.kinematic_state.vel, self.kinematic_state.yawrate,
                         self.extent_model.L, self.extent_model.W, *self.extent_model.pca_params])

    

