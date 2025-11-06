import os
import sys
import numpy as np
import pickle
from copy import deepcopy
from tqdm import trange
from pathlib import Path

from global_project_paths import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)

from global_project_paths import SIMDATA_PATH

from src.dynamics.vessel import Vessel
from src.dynamics.kinematic_state import KinematicState
from src.dynamics.process_models import GroundTruthModel, Model_PCA_CV, decoupled_CV_model, decoupled_CV_model_jacobian
from sensors.lidar import LidarModel

from src.extent_model.extent import Extent, PCAExtentModel
from src.extent_model.geometry_utils import get_vessel_shape, compute_estimated_shape

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
from src.states.states import State_PCA

from src.utils.config_classes import Config

def run_single_simulation(config: Config, method: str):
    """
    Runs a single simulation using the senfuslib.Simulator and new architecture.
    """
    sim_cfg = config.sim
    tracker_cfg = config.tracker
    lidar_cfg = config.lidar

    # --- 1. Initialize Models ---
    # The FILTER's belief model
    filter_dyn_model = Model_PCA_CV(
        x_pos_std_dev=tracker_cfg.pos_north_std_dev,
        y_pos_std_dev=tracker_cfg.pos_east_std_dev,
        yaw_std_dev=tracker_cfg.heading_std_dev,
        N_pca=tracker_cfg.N_pca
    )
    
    # The "REAL WORLD" model for the simulator
    rng = np.random.default_rng(seed=sim_cfg.seed)
    gt_dynamic_model = GroundTruthModel(rng=rng, yaw_rate_std_dev=0.01)

    pca_params = np.load(Path(tracker_cfg.PCA_parameters_path))
    sensor_model = LidarModel(
        lidar_position=np.array(lidar_cfg.lidar_position),
        num_rays=lidar_cfg.num_rays,
        max_distance=lidar_cfg.max_distance,
        lidar_std_dev=lidar_cfg.lidar_std_dev,
        pca_mean=pca_params['mean'],
        pca_eigenvectors=pca_params['eigenvectors'],
        rng=rng
    )

    # Initialize the Tracker
    if method == "bfgs":
        tracker = BFGS(dynamic_model=filter_dyn_model, lidar_model=sensor_model, config=config)
    else:
        raise NotImplementedError(f"Tracker method '{method}' is not yet refactored.")
    
    # The simulator orchestrates the data generation process
    simulator = Simulator(
        dynamic_model=gt_dynamic_model,
        sensor_model=sensor_model,
        sensor_setter=None,  # No special sensor setter needed
        init_state=tracker_cfg.initial_state,  # TODO Martin the GT initial state should be separate
        dt=sim_cfg.timestep,
        end_time=sim_cfg.num_frames * sim_cfg.timestep,
        seed=sim_cfg.seed
    )

    # Generate a TimeSequence of ground truth states
    ground_truth_ts, measurements_ts = simulator.get_gt_and_meas()

    # --- 4. Run Filtering Loop ---
    results_ts = TimeSequence() # Store results in a TimeSequence as well
    
    print(f"Filtering {len(measurements_ts)} measurements...")
    for ts, measurement in measurements_ts.items():
        tracker.predict()
        update_result = tracker.update(measurement, ground_truth=ground_truth_ts.get_t(ts))
        results_ts.insert(ts, update_result)

    # --- 5. Save the results ---
    filename = os.path.join(SIMDATA_PATH, f"{sim_cfg.name}.pkl")
    data_to_save = {
        "config": config,
        "ground_truth_ts": ground_truth_ts,
        "measurements_ts": measurements_ts,
        "results_ts": results_ts,
    }
    with open(filename, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Simulation run data saved to {filename}")

# def run_single_simulation_old(config, method):
#     sim_params = config.sim
#     ekfconfig = config.tracker
#     lidar_config = config.lidar

#     num_frames = sim_params.num_frames
#     num_simulations = sim_params.num_simulations
#     seed = sim_params.seed
#     name = sim_params.name
#     timestep = sim_params.timestep
#     param_true = sim_params.param_true
#     d_angle = sim_params.d_angle
#     angles = sim_params.angles

#     lidar_position = lidar_config.lidar_position
#     lidar_max_distance = lidar_config.max_distance

#     all_state_posteriors = []
#     all_state_predictions = []
#     all_gt = []
#     all_P_prior = []
#     all_P_post = []
#     all_S = []
#     all_v = []
#     all_z = []
#     all_x_dim = []
#     all_z_dim = []
#     all_shape_x = []
#     all_shape_y = []
#     all_init_conditions = []
#     static_covariances = []


#     rng = np.random.default_rng(seed=seed)

#     my_config = deepcopy(ekfconfig)
#     extent_true = Extent(param_true, d_angle)
#     pca_extent_true = PCAExtentModel(extent_true, my_config.N_pca)
#     kinematics_true = KinematicState()
#     target_vessel = Vessel(extent_true, pca_extent_true, kinematics_true)

#     tracker = _initialize_tracker(timestep, method, rng, my_config)
    
#     # --- Data and Plotting Initialization ---
#     state_predictions, state_posteriors, gt, P_prior, P_post, S, v, z, x_dim, z_dim = [], [], [], [], [], [], [], [], [], []
#     shape_x_list, shape_y_list = [], []
#     init_condition = [tracker.state.copy(), tracker.P.copy(), target_vessel.get_state()]
    
#     for i in trange(num_frames):
#         try:
#             tracker.predict(decoupled_CV_model_jacobian)
            
#             P_prior_i = tracker.P.copy()

#             # Step and measure ground truth
#             target_vessel.step(timestep, rng)
#             shape_x, shape_y = get_vessel_shape(target_vessel)
#             measurements_polar = simulate_lidar_measurements(shape_x, shape_y, lidar_config, rng)

#             # Run update
#             state_pred_i, state_post_i, z_i, v_i, S_i, _, P_post_i, z_dim_i, x_dim_i = tracker.update(measurements_polar, lidar_pos=lidar_position, ais_measurements=None, ground_truth=target_vessel.get_state())
            
#             # Store data
#             state_predictions.append(state_pred_i)
#             state_posteriors.append(state_post_i)
#             gt.append(target_vessel.get_state())
#             P_prior.append(P_prior_i)
#             P_post.append(P_post_i)
#             S.append(S_i)
#             v.append(v_i)
#             z.append(z_i)
#             x_dim.append(x_dim_i)
#             z_dim.append(z_dim_i)
#             shape_x_list.append(shape_x)
#             shape_y_list.append(shape_y)

#         except Exception as e:
#             print(f"Error in plotting simulation step {i}: {e}")
#             break

#     all_state_posteriors.append(state_posteriors)
#     all_state_predictions.append(state_predictions)
#     all_gt.append(gt)
#     all_P_prior.append(P_prior)
#     all_P_post.append(P_post)
#     all_S.append(S)
#     all_v.append(v)
#     all_z.append(z)
#     all_x_dim.append(x_dim)
#     all_z_dim.append(z_dim)
#     all_init_conditions.append(init_condition)
#     all_shape_x.append(shape_x_list)
#     all_shape_y.append(shape_y_list)

#     true_extent = Extent(param_true, d_angle)
#     true_extent_pca = PCAExtentModel(true_extent, ekfconfig.N_pca)
#     static_covariances = [tracker.Q, tracker.R_ais, tracker.R_lidar]

#     # TODO Martin
#     # Consistency analysis
#     # NEES(all_state_posteriors, all_gt, ekfconfig, name)
#     # NIS(all_y, all_S, name)

#     data_to_save = SimulationResult(
#         # Lists of simulation run data
#         state_predictions=all_state_predictions,
#         state_posteriors=all_state_posteriors,
#         ground_truth=all_gt,
#         P_prior=all_P_prior,
#         P_post=all_P_post,
#         S=all_S,
#         y=all_v,
#         z=all_z,
#         x_dim=all_x_dim,
#         z_dim=all_z_dim,
#         shape_x=all_shape_x,
#         shape_y=all_shape_y,
#         initial_condition=all_init_conditions,

#         # Config data
#         config=config,
#         lidar_position=lidar_position,
#         lidar_max_distance=lidar_max_distance,
#         true_extent=true_extent.cartesian,
#         true_extent_radius=true_extent.radii,
#         N_pca=ekfconfig.N_pca,
#         angles=angles,
#         num_simulations=num_simulations,
#         num_frames=num_frames,

#         # Static data
#         PCA_mean=true_extent_pca.fourier_coeff_mean,
#         PCA_eigenvectors=true_extent_pca.M,
#         static_covariances=static_covariances,
#     )

#     filename = os.path.join(SIMDATA_PATH, f"{name}.pkl")
#     with open(filename, "wb") as f:
#         pickle.dump(data_to_save, f)
#     print(f"Simulation run data saved to {filename}")

# def _initialize_tracker(timestep, method, rng, config):
#     """Initializes and returns a tracker instance based on the method name."""
#     match method:
#         case "ekf":
#             return EKF(process_model=decoupled_CV_model, timestep=timestep, config=config)
#         case "iekf":
#             return IterativeEKF(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
#         case "ukf":
#             return UKF(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
#         case "bfgs":
#             return BFGS(dynamic_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
#         case "slsqp":
#             return SLSQP(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
#         case "gauss_newton":
#             return GaussNewton(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
#         case "levenberg_marquardt":
#             return LevenbergMarquardt(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
#         case "smoothing_slsqp":
#             return SmoothingSLSQP(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
#         case _:
#             raise ValueError(f"Unknown tracking method: {method}")