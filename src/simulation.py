import os
import sys
import numpy as np
import pickle
from copy import deepcopy
from tqdm import trange

from src.config_paths import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)

from src.config_paths import SIMDATA_PATH

from src.dynamics.vessel import Vessel
from src.dynamics.kinematic_state import KinematicState
from src.dynamics.process_models import decoupled_CV_model, decoupled_CV_model_jacobian
from src.sensors.lidar import simulate_lidar_measurements

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

from src.utils.SimulationResult import SimulationResult

def run_single_simulation(config, method):
    sim_params = config.sim
    ekfconfig = config.tracker
    lidar_config = config.lidar

    num_frames = sim_params.num_frames
    num_simulations = sim_params.num_simulations
    seed = sim_params.seed
    name = sim_params.name
    timestep = sim_params.timestep
    param_true = sim_params.param_true
    d_angle = sim_params.d_angle
    angles = sim_params.angles

    lidar_position = lidar_config.lidar_position
    lidar_max_distance = lidar_config.max_distance

    all_state_posteriors = []
    all_state_predictions = []
    all_gt = []
    all_P_prior = []
    all_P_post = []
    all_S = []
    all_y = []
    all_z = []
    all_x_dim = []
    all_z_dim = []
    all_shape_x = []
    all_shape_y = []
    all_init_conditions = []
    static_covariances = []


    rng = np.random.default_rng(seed=seed)

    my_config = deepcopy(ekfconfig)
    extent_true = Extent(param_true, d_angle)
    pca_extent_true = PCAExtentModel(extent_true, my_config.N_pca)
    kinematics_true = KinematicState()
    target_vessel = Vessel(extent_true, pca_extent_true, kinematics_true)

    tracker = _initialize_tracker(timestep, method, rng, my_config)
    
    # --- Data and Plotting Initialization ---
    state_predictions, state_posteriors, gt, P_prior, P_post, S, y, z, x_dim, z_dim = [], [], [], [], [], [], [], [], [], []
    shape_x_list, shape_y_list = [], []
    init_condition = [tracker.state.copy(), tracker.P.copy(), target_vessel.get_state()]
    
    for i in trange(num_frames):
        try:
            tracker.predict(decoupled_CV_model_jacobian)
            
            P_prior_i = tracker.P.copy()

            # Step and measure ground truth
            target_vessel.step(timestep, rng)
            shape_x, shape_y = get_vessel_shape(target_vessel)
            measurements_polar = simulate_lidar_measurements(shape_x, shape_y, lidar_config, rng)

            # Run update
            state_pred_i, state_post_i, z_i, y_i, S_i, _, P_post_i, z_dim_i, x_dim_i = tracker.update(measurements_polar, lidar_pos=lidar_position, ais_measurements=None, ground_truth=target_vessel.get_state())
            
            # Store data
            state_predictions.append(state_pred_i)
            state_posteriors.append(state_post_i)
            gt.append(target_vessel.get_state())
            P_prior.append(P_prior_i)
            P_post.append(P_post_i)
            S.append(S_i)
            y.append(y_i)
            z.append(z_i)
            x_dim.append(x_dim_i)
            z_dim.append(z_dim_i)
            shape_x_list.append(shape_x)
            shape_y_list.append(shape_y)

        except Exception as e:
            print(f"Error in plotting simulation step {i}: {e}")
            break

    all_state_posteriors.append(state_posteriors)
    all_state_predictions.append(state_predictions)
    all_gt.append(gt)
    all_P_prior.append(P_prior)
    all_P_post.append(P_post)
    all_S.append(S)
    all_y.append(y)
    all_z.append(z)
    all_x_dim.append(x_dim)
    all_z_dim.append(z_dim)
    all_init_conditions.append(init_condition)
    all_shape_x.append(shape_x_list)
    all_shape_y.append(shape_y_list)

    true_extent = Extent(param_true, d_angle)
    true_extent_pca = PCAExtentModel(true_extent, ekfconfig.N_pca)
    static_covariances = [tracker.Q, tracker.R_ais, tracker.R_lidar]

    # TODO Martin
    # Consistency analysis
    # NEES(all_state_posteriors, all_gt, ekfconfig, name)
    # NIS(all_y, all_S, name)

    data_to_save = SimulationResult(
        # Lists of simulation run data
        state_predictions=all_state_predictions,
        state_posteriors=all_state_posteriors,
        ground_truth=all_gt,
        P_prior=all_P_prior,
        P_post=all_P_post,
        S=all_S,
        y=all_y,
        z=all_z,
        x_dim=all_x_dim,
        z_dim=all_z_dim,
        shape_x=all_shape_x,
        shape_y=all_shape_y,
        initial_condition=all_init_conditions,

        # Config data
        config=config,
        lidar_position=lidar_position,
        lidar_max_distance=lidar_max_distance,
        true_extent=true_extent.cartesian,
        true_extent_radius=true_extent.radii,
        N_pca=ekfconfig.N_pca,
        angles=angles,
        num_simulations=num_simulations,
        num_frames=num_frames,

        # Static data
        PCA_mean=true_extent_pca.fourier_coeff_mean,
        PCA_eigenvectors=true_extent_pca.M,
        static_covariances=static_covariances,
    )

    filename = os.path.join(SIMDATA_PATH, f"{name}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Simulation run data saved to {filename}")

def _initialize_tracker(timestep, method, rng, config):
    """Initializes and returns a tracker instance based on the method name."""
    match method:
        case "ekf":
            return EKF(process_model=decoupled_CV_model, timestep=timestep, config=config)
        case "iekf":
            return IterativeEKF(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
        case "ukf":
            return UKF(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
        case "bfgs":
            return BFGS(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
        case "slsqp":
            return SLSQP(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
        case "gauss_newton":
            return GaussNewton(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
        case "levenberg_marquardt":
            return LevenbergMarquardt(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
        case "smoothing_slsqp":
            return SmoothingSLSQP(process_model=decoupled_CV_model, timestep=timestep, rng=rng, config=config)
        case _:
            raise ValueError(f"Unknown tracking method: {method}")

# def _run_single_simulation(sim_params, lidar_config, ekf_config, method, rng, create_plot=False):
#     """
#     Runs a single simulation for a given configuration and returns all time-series data.
#     Optionally creates a plot for this run.
#     """
#     # --- Object Initialization ---
#     my_config = deepcopy(ekf_config)
#     extent_true = Extent(sim_params.param_true, sim_params.d_angle)
#     pca_extent_true = PCAExtentModel(extent_true, my_config.N_pca)
#     kinematics_true = KinematicState()
#     target_vessel = Vessel(extent_true, pca_extent_true, kinematics_true)
#     tracker = _initialize_tracker(sim_params.timestep, method, rng, my_config)

#     # --- Data and Plotting Initialization ---
#     state_dim = tracker.state.shape[0]
#     num_frames = sim_params.num_frames
    
#     # Pre-allocate NumPy arrays for efficiency
#     state_estimates = np.zeros((num_frames, state_dim))
#     ground_truth = np.zeros((num_frames, state_dim))
#     P_prior = np.zeros((num_frames, state_dim, state_dim))
#     P_post = np.zeros((num_frames, state_dim, state_dim))
    
#     initial_condition = {
#         "state": tracker.state.copy(), 
#         "P": tracker.P.copy(), 
#         "ground_truth": target_vessel.get_state()
#     }

#     plot_frames, locationx, locationy = ([], [], []) if create_plot else (None, None, None)
#     if create_plot:
#         fig = initialize_plotly_figure(num_frames)

#     # --- Simulation Loop ---
#     for i in trange(num_frames, desc="Sim Frame", leave=False):
#         try:
#             tracker.predict(decoupled_CV_model_jacobian)
#             P_prior[i] = tracker.P.copy()

#             target_vessel.step(sim_params.timestep, rng)
#             shape_x, shape_y = get_vessel_shape(target_vessel)
#             measurements_polar = simulate_lidar_measurements(shape_x, shape_y, lidar_config, rng)

#             update_results = tracker.update(measurements_polar, lidar_pos=lidar_config.lidar_position, ground_truth=target_vessel.get_state()) # NB new return signature
            
#             state_estimates[i] = update_results[0]
#             P_post[i] = update_results[5]
#             ground_truth[i] = target_vessel.get_state()

#             if create_plot:
#                 locationx.append(target_vessel.kinematic_state.pos[0])
#                 locationy.append(target_vessel.kinematic_state.pos[1])
#                 plot_frame = create_frame(tracker, measurements_polar, (i+1), locationx, locationy, 
#                                         shape_x, shape_y, lidar_config.max_distance, lidar_config.lidar_position, sim_params.angles, compute_estimated_shape)
#                 plot_frames.append(plot_frame)

#         except Exception as e:
#             print(f"Error in simulation step {i}: {e}")
#             traceback.print_exc()
#             break
            
#     if create_plot:
#         fig = create_sim_figure(fig, plot_frames, len(plot_frames))
#         plot_filename = f"figures/{sim_params.name}.html"
#         os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
#         plotly.offline.plot(fig, filename=plot_filename, auto_open=False)
#         print(f"Plot saved to {plot_filename}")

#     static_covariances = {"Q": tracker.Q, "R_ais": tracker.R_ais, "R_lidar": tracker.R_lidar}
    
#     return {
#         "state_estimates": state_estimates,
#         "ground_truth": ground_truth,
#         "P_prior": P_prior,
#         "P_post": P_post,
#         "initial_condition": initial_condition,
#         "static_covariances": static_covariances
#     }

# def monte_carlo(config, method, create_plot=False):
#     sim_params = config.sim
#     ekfconfig = config.tracker
#     lidar_config = config.lidar

#     seed_sequence = np.random.SeedSequence(sim_params.seed)
#     sim_seeds = seed_sequence.generate_state(sim_params.num_simulations)

#     state_dim = ekfconfig.state.shape[0]
#     num_sim = sim_params.num_simulations
#     num_frames = sim_params.num_frames

#     all_state_estimates = np.zeros((num_sim, num_frames, state_dim))
#     all_ground_truth = np.zeros((num_sim, num_frames, state_dim))
#     all_P_prior = np.zeros((num_sim, num_frames, state_dim, state_dim))
#     all_P_post = np.zeros((num_sim, num_frames, state_dim, state_dim))
#     all_initial_conditions = [] # This can remain a list of dicts

#     # --- Monte Carlo Loop ---
#     for i in trange(num_sim, desc="Monte Carlo Runs"):
#         print(f"Simulation {i+1}/{num_sim} started.")
#         rng = np.random.default_rng(sim_seeds[i])
        
#         # For the first run, we can optionally create a plot
#         should_plot_this_run = create_plot and i == 0
        
#         results = _run_single_simulation(sim_params, lidar_config, ekfconfig, method, rng, create_plot=should_plot_this_run)

#         # Store results in the master NumPy arrays
#         all_state_estimates[i, :, :] = results["state_estimates"]
#         all_ground_truth[i, :, :] = results["ground_truth"]
#         all_P_prior[i, :, :, :] = results["P_prior"]
#         all_P_post[i, :, :, :] = results["P_post"]
#         all_initial_conditions.append(results["initial_condition"])
        
#         print(f"Simulation {i+1}/{num_sim} completed.")

#     # --- Save Data to a Single Pickle File ---
#     true_extent = Extent(sim_params.param_true, sim_params.d_angle)
#     true_extent_pca = PCAExtentModel(true_extent, ekfconfig.N_pca)

#     data_to_save = {
#         "config": config,
#         "state_estimates": all_state_estimates,
#         "ground_truth": all_ground_truth,
#         "P_prior": all_P_prior,
#         "P_post": all_P_post,
#         "initial_conditions": all_initial_conditions,
#         "static_covariances": results["static_covariances"], # From the last run, should be the same
#         "true_extent_cartesian": true_extent.cartesian,
#         "PCA_mean": true_extent_pca.fourier_coeff_mean,
#         "PCA_eigenvectors": true_extent_pca.M,
#     }
    
#     output_dir = "data/results"
#     os.makedirs(output_dir, exist_ok=True)
#     filename = f"{output_dir}/mc_data_{sim_params.name}.pkl"
#     with open(filename, "wb") as f:
#         pickle.dump(data_to_save, f)
#     print(f"Monte Carlo simulation data saved to {filename}")