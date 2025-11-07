from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

from senfuslib.timesequence import TimeSequence
from src.states.states import State_PCA, LidarScan
from src.tracker.tracker import TrackerUpdateResult
from src.utils.config_classes import Config


@dataclass
class SimulationResult:
    """
    A data class to store the results of a complete simulation run.
    It holds the configuration and the time-indexed sequences of data.
    """
    config: Config
    ground_truth_ts: TimeSequence[State_PCA]
    measurements_ts: TimeSequence[LidarScan]
    results_ts: TimeSequence[TrackerUpdateResult]
    static_covariances: Dict[str, np.ndarray]