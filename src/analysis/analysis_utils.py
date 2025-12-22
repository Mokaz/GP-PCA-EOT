import numpy as np

from src.utils.SimulationResult import SimulationResult
from src.senfuslib.analysis import ConsistencyAnalysis
from src.senfuslib.timesequence import TimeSequence
from src.senfuslib.gaussian import MultiVarGauss

from src.states.states import LidarScan, State_PCA, State_GP
from src.utils.tools import cart2pol

def create_consistency_analysis_from_sim_result(sim_result: SimulationResult) -> ConsistencyAnalysis:
    """
    Unpacks a SimulationResult object and prepares the data for ConsistencyAnalysis.
    Handles conversion between State_PCA (GT) and State_GP (Est) if necessary.
    Handles dimension mismatch due to virtual measurements in GP tracker.
    """
    
    tracker_results_ts = sim_result.tracker_results_ts
    x_gts: TimeSequence = sim_result.ground_truth_ts

    x_ests_posterior: TimeSequence[MultiVarGauss] = tracker_results_ts.map(
        lambda r: r.state_posterior
    )

    # --- 1. State Conversion (GT PCA -> GP) ---
    if len(x_ests_posterior) > 0 and len(x_gts) > 0:
        first_est = x_ests_posterior.values[0].mean
        first_gt = x_gts.values[0]

        if isinstance(first_est, State_GP) and isinstance(first_gt, State_PCA):
            print("[INFO] Converting Ground Truth from PCA to GP representation for consistency analysis...")
            x_gts = _convert_gt_pca_to_gp(x_gts, sim_result.config)

    # -----------------------------------------------------------------

    # Real Measurements (Ground Truth Z)
    zs_flattened_ts: TimeSequence[np.ndarray] = sim_result.measurements_global_ts.map(
        lambda scan: scan.flatten('F')
    )

    # Predicted Measurements (Z_hat)
    # We need a custom mapper to handle the dimension mismatch caused by virtual measurements
    def process_predicted_measurement(res, real_meas):
        pred_gauss = res.predicted_measurement
        
        # Check dimensions
        dim_pred = pred_gauss.mean.shape[0]
        dim_real = real_meas.shape[0]
        
        if dim_pred == dim_real:
            return pred_gauss
        
        elif dim_pred > dim_real:
            # Assume virtual measurements were appended to the END
            # Slice the mean and covariance to keep only the real part
            valid_idx = slice(0, dim_real)
            
            new_mean = pred_gauss.mean[valid_idx]
            new_cov = pred_gauss.cov[valid_idx, valid_idx]
            
            return MultiVarGauss(mean=new_mean, cov=new_cov)
        else:
            # Should not happen (prediction smaller than measurement?)
            return pred_gauss

    # We iterate manually to zip real measurements with predictions
    z_preds_ts = TimeSequence()
    for t, res in tracker_results_ts.items():
        if t in zs_flattened_ts:
            real_meas = zs_flattened_ts.get_t(t)
            processed_pred = process_predicted_measurement(res, real_meas)
            z_preds_ts.insert(t, processed_pred)

    return ConsistencyAnalysis(
        x_gts=x_gts,
        zs=zs_flattened_ts,
        x_ests=x_ests_posterior,
        z_preds=z_preds_ts,
    )

def _convert_gt_pca_to_gp(gt_ts: TimeSequence[State_PCA], config) -> TimeSequence[State_GP]:
    new_ts = TimeSequence()
    N_gp = config.tracker.N_gp_points
    gp_angles = np.linspace(0, 2 * np.pi, N_gp, endpoint=False)
    true_shape_cart = config.extent.shape_coords_body
    true_angles, true_radii = cart2pol(true_shape_cart[0], true_shape_cart[1])

    for t, gt in gt_ts.items():
        sampled_radii = np.interp(gp_angles, true_angles, true_radii, period=2*np.pi)
        new_state = State_GP(
            x=gt.x, y=gt.y, yaw=gt.yaw, vel_x=gt.vel_x, vel_y=gt.vel_y, yaw_rate=gt.yaw_rate,
            radii=sampled_radii
        )
        new_ts.insert(t, new_state)
    return new_ts