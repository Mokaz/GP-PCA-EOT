import numpy as np

from src.utils.SimulationResult import SimulationResult
from src.senfuslib.analysis import ConsistencyAnalysis
from src.senfuslib.timesequence import TimeSequence
from src.senfuslib.gaussian import MultiVarGauss

from src.states.states import LidarScan, State_PCA

def create_consistency_analysis_from_sim_result(sim_result: SimulationResult) -> ConsistencyAnalysis:
    """
    Unpacks a SimulationResult object and prepares the data for ConsistencyAnalysis.
    """
    
    tracker_results_ts = sim_result.tracker_results_ts

    x_gts: TimeSequence[State_PCA] = sim_result.ground_truth_ts

    zs_flattened_ts: TimeSequence[np.ndarray] = sim_result.measurements_global_ts.map(
        lambda scan: scan.flatten('F')
    )

    x_ests_posterior: TimeSequence[MultiVarGauss] = tracker_results_ts.map(
        lambda r: r.state_posterior
    )

    z_preds_ts: TimeSequence[MultiVarGauss] = tracker_results_ts.map(
        lambda r: r.predicted_measurement
    )

    return ConsistencyAnalysis(
        x_gts=x_gts,
        zs=zs_flattened_ts,
        x_ests=x_ests_posterior,
        z_preds=z_preds_ts,
    )