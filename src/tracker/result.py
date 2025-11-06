from dataclasses import dataclass
from typing import Any
import numpy as np
from senfuslib import MultiVarGauss

@dataclass
class TrackerUpdateResult:
    """
    Holds all relevant data from a single tracker update step.
    """
    # Core filter states
    estimate_prior: MultiVarGauss       # State estimate before the update (x_k|k-1)
    estimate_posterior: MultiVarGauss   # State estimate after the update (x_k|k)

    # Measurement and Innovation
    measurement: np.ndarray             # The raw measurement vector used (z_k)
    innovation_gauss: MultiVarGauss     # The innovation (y_k) and its covariance (S_k)

    # Optional Debugging / Analysis Info
    iterations: int = None              # For iterative optimizers
    cost: float = None                  # Final value of the objective function
    raw_optimizer_result: Any = None    # The full result object from scipy.minimize