import numpy as np
from dataclasses import dataclass
from typing import Any, Optional

from src.senfuslib.gaussian import MultiVarGauss

@dataclass
class TrackerUpdateResult:
    """
    Holds all relevant data from a single tracker update step.
    Moved here to avoid circular imports between Tracker logic and SimulationResult.
    """
    # Core filter states
    state_prior: MultiVarGauss           # State estimate before the update (x_k|k-1)
    state_posterior: MultiVarGauss       # State estimate after the update (x_k|k)

    # Measurement and Innovation
    measurements: Optional[np.ndarray]      # The flattened measurement vector used (z_k)
    predicted_measurement: Optional[MultiVarGauss]    # The predicted measurement
    innovation_gauss: Optional[MultiVarGauss]         # The innovation

    # --- DEBUGGING VALUES ---
    cost_prior: Optional[float] = None
    cost_likelihood: Optional[float] = None
    cost_penalty: Optional[float] = None
    H_jacobian: Optional[np.ndarray] = None
    R_covariance: Optional[np.ndarray] = None
    
    # Optional Debugging / Analysis Info
    iterations: Optional[int] = None
    iterates: Optional[list] = None
    cost: Optional[float] = None
    raw_optimizer_result: Any = None