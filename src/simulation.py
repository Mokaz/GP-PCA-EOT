import os
import sys
import numpy as np
import pickle
from pathlib import Path

from global_project_paths import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)

from global_project_paths import SIMDATA_PATH

from src.dynamics.process_models import GroundTruthModel, Model_PCA_CV
from sensors.LidarModel import LidarModel

from src.tracker.tracker import TrackerUpdateResult
from src.tracker.ExtendedKalmanFilter import EKF
from src.tracker.IterativeEKF import IterativeEKF
from src.tracker.gauss_newton import GaussNewton
from src.tracker.levenberg_marquardt import LevenbergMarquardt
from src.tracker.BFGS import BFGS
from src.tracker.SLSQP import SLSQP
from src.tracker.smoothing_SLSQP import SmoothingSLSQP
from src.tracker.UnscentedKalmanFilter import UKF

from src.senfuslib.simulator import Simulator
from src.senfuslib.timesequence import TimeSequence

from src.utils.SimulationResult import SimulationResult

from src.utils.config_classes import Config
from tqdm import tqdm


def run_single_simulation(config: Config, method: str) -> SimulationResult:
    """
    Runs a single simulation using the senfuslib.Simulator and new architecture.
    """
    sim_cfg = config.sim
    tracker_cfg = config.tracker
    lidar_cfg = config.lidar
    extent_cfg = config.extent

    rng = np.random.default_rng(seed=sim_cfg.seed)

    # --- Initialize Models ---
    filter_dyn_model = Model_PCA_CV(
        x_pos_std_dev=tracker_cfg.pos_north_std_dev,
        y_pos_std_dev=tracker_cfg.pos_east_std_dev,
        yaw_std_dev=tracker_cfg.heading_std_dev,
        N_pca=tracker_cfg.N_pca
    )

    gt_dynamic_model = GroundTruthModel(rng=rng, yaw_rate_std_dev=sim_cfg.gt_yaw_rate_std_dev)

    pca_params = np.load(Path(tracker_cfg.PCA_parameters_path))
    sensor_model = LidarModel(
        lidar_position=np.array(lidar_cfg.lidar_position),
        num_rays=lidar_cfg.num_rays,
        max_distance=lidar_cfg.max_distance,
        lidar_std_dev=lidar_cfg.lidar_std_dev,
        extent_cfg=extent_cfg,
        pca_mean=pca_params['mean'],
        pca_eigenvectors=pca_params['eigenvectors'][:, :tracker_cfg.N_pca].real,
        rng=rng
    )

    # --- Initialize the Tracker ---
    if method == "bfgs":
        tracker = BFGS(dynamic_model=filter_dyn_model, lidar_model=sensor_model, config=config)
    else:
        raise NotImplementedError(f"Tracker method '{method}' is not yet refactored.")
    
    simulator = Simulator(
        dynamic_model=gt_dynamic_model,
        sensor_model=sensor_model,
        sensor_setter=None,
        init_state=tracker_cfg.initial_state,  # TODO Martin the GT initial tracker state should be separate
        dt=sim_cfg.dt,
        end_time=sim_cfg.num_frames * sim_cfg.dt,
        seed=str(sim_cfg.seed)  # Ensure seed is a string for crc32
    )

    # --- Generate Simulation Data ---
    print(f"Generating simulation data for {sim_cfg.num_frames} frames...")
    ground_truth_ts, measurements_lidar_frame_ts = simulator.get_gt_and_meas()

    lidar_pos_global = np.array(lidar_cfg.lidar_position).reshape(2, 1)
    measurements_global_ts = measurements_lidar_frame_ts.map(lambda scan: scan + lidar_pos_global)

    # --- Run Filtering Loop ---
    results_ts: TimeSequence[TrackerUpdateResult] = TimeSequence() 
    
    for ts, measurement in tqdm(measurements_lidar_frame_ts.items(), desc="Filtering measurements"):
        tracker.predict()
        update_result = tracker.update(measurement, ground_truth=ground_truth_ts.get_t(ts))
        results_ts.insert(ts, update_result)

    filename = os.path.join(SIMDATA_PATH, f"{sim_cfg.name}.pkl")

    static_covariances = {
        "Q": filter_dyn_model.Q_d(dt=sim_cfg.dt),
        "R_point": sensor_model.R_single_point()
    }

    data_to_save = SimulationResult(
        config=config,
        ground_truth_ts=ground_truth_ts,
        measurements_global_ts=measurements_global_ts,
        tracker_results_ts=results_ts,
        static_covariances=static_covariances
    )
    with open(filename, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Simulation run data saved to {filename}")

    return data_to_save