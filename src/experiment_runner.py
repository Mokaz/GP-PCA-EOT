import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from pathlib import Path
from tqdm import tqdm

from global_project_paths import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)

from global_project_paths import SIMDATA_PATH

from src.dynamics.process_models import GroundTruthModel, Model_GP_CV, Model_PCA_CV, Model_PCA_Temporal, Model_PCA_Inflation
from src.dynamics.trajectories import CircleTrajectory, WaypointTrajectory, ConstantVelocityTrajectory
from src.sensors.LidarModel import LidarSimulator

from tracker.EKF import EKF
from src.tracker.IterativeEKF import IterativeEKF
# from src.tracker.gauss_newton import GaussNewton
# from src.tracker.levenberg_marquardt import LevenbergMarquardt
from src.tracker.BFGS import BFGS
# from src.tracker.SLSQP import SLSQP
# from src.tracker.smoothing_SLSQP import SmoothingSLSQP
# from src.tracker.UnscentedKalmanFilter import UKF
from src.tracker.ImplicitIEKF import ImplicitIEKF

from src.tracker.TrackerUpdateResult import TrackerUpdateResult
from src.sensors.LidarModel import LidarMeasurementModel

from src.senfuslib.simulator import Simulator
from src.senfuslib.timesequence import TimeSequence

from src.utils.SimulationResult import SimulationResult
from src.utils.geometry_utils import compute_exact_vessel_shape_global

from src.utils.config_classes import Config

from src.utils.GaussianProcess import GaussianProcess
from src.sensors.LidarModelGP import LidarModelGP 
from src.tracker.GP_IEKF import GP_IEKF

def run_single_simulation(config: Config, method: str) -> SimulationResult:
    """
    Runs a single simulation using the senfuslib.Simulator and new architecture.
    """
    sim_cfg = config.sim
    tracker_cfg = config.tracker
    lidar_cfg = config.lidar
    extent_cfg = config.extent

    rng = np.random.default_rng(seed=sim_cfg.seed)

    traj_cfg = sim_cfg.trajectory
    if traj_cfg.type == "linear":
        trajectory_strategy = ConstantVelocityTrajectory()
    elif traj_cfg.type == "circle":
        trajectory_strategy = CircleTrajectory(
            center=traj_cfg.center,
            target_speed=traj_cfg.speed,
            radius=traj_cfg.radius,
            clockwise=traj_cfg.clockwise
        )
    elif traj_cfg.type == "waypoints":
        trajectory_strategy = WaypointTrajectory(
            waypoints=traj_cfg.waypoints,
            target_speed=traj_cfg.speed
        )
    else:
        raise ValueError(f"Unknown trajectory type: {traj_cfg.type}")

    gt_dynamic_model = GroundTruthModel(
        rng=rng, 
        yaw_rate_std_dev=sim_cfg.gt_yaw_rate_std_dev,
        trajectory_strategy=trajectory_strategy
    )

    if method == "gp_iekf":
        exit("GP-IEKF currently disabled")
        # NOTE GP disabled for now

        # # 1. Initialize GP Math Utils
        # gp_utils = GaussianProcess(
        #     n_test_points=tracker_cfg.N_gp_points, 
        #     length_scale=tracker_cfg.gp_length_scale,
        #     signal_var=tracker_cfg.gp_signal_var,
        #     symmetric=True
        # )

        # # 2. Initialize GP Process Model
        # filter_dyn_model = Model_GP_CV(
        #     gp_utils=gp_utils,
        #     x_pos_std_dev=tracker_cfg.pos_north_std_dev,
        #     y_pos_std_dev=tracker_cfg.pos_east_std_dev,
        #     yaw_std_dev=tracker_cfg.heading_std_dev,
        #     forgetting_factor=tracker_cfg.gp_forgetting_factor
        # )

        # # 3. Initialize GP Sensor Model
        # lidar_model = LidarModelGP(
        #     lidar_position=np.array(lidar_cfg.lidar_position),
        #     num_rays=lidar_cfg.num_rays,
        #     max_distance=lidar_cfg.max_distance,
        #     lidar_gt_std_dev=lidar_cfg.lidar_gt_std_dev,
        #     lidar_std_dev=tracker_cfg.lidar_std_dev, # Use tracker config for noise
        #     gp_utils=gp_utils,
        #     rng=rng,
        #     shape_coords_body=extent_cfg.shape_coords_body # Needed for GT generation
        # )

        # # 4. Initialize GP Tracker
        # tracker = GP_IEKF(
        #     dynamic_model=filter_dyn_model, 
        #     lidar_model=lidar_model, 
        #     config=config,
        #     use_negative_info=tracker_cfg.gp_use_negative_info
        # )

        # simulator = Simulator(
        #     dynamic_model=gt_dynamic_model,
        #     sensor_model=lidar_model, # TODO: Separate GT and filter sensor models for GP too
        #     sensor_setter=None,
        #     init_state=sim_cfg.initial_state_gt,
        #     dt=sim_cfg.dt,
        #     end_time=sim_cfg.num_frames * sim_cfg.dt,
        #     seed=str(sim_cfg.seed),
        #     use_cache=False
        # )

    else:
        common_kwargs = dict(
            x_pos_std_dev=tracker_cfg.pos_north_std_dev,
            y_pos_std_dev=tracker_cfg.pos_east_std_dev,
            yaw_std_dev=tracker_cfg.heading_std_dev,
            N_pca=tracker_cfg.N_pca
        )

        if tracker_cfg.process_model == "cv":
            filter_dyn_model = Model_PCA_CV(**common_kwargs)
        
        elif tracker_cfg.process_model == "temporal":
            filter_dyn_model = Model_PCA_Temporal(
                **common_kwargs,
                eta_f=tracker_cfg.temporal_eta,
                pca_process_var=tracker_cfg.temporal_pca_var
            )

        elif tracker_cfg.process_model == "inflation":
            filter_dyn_model = Model_PCA_Inflation(
                **common_kwargs,
                lambda_f=tracker_cfg.inflation_lambda
            )
        else:
            raise ValueError(f"Unknown process model: {tracker_cfg.process_model}")


        pca_params = np.load(Path(tracker_cfg.PCA_parameters_path))
        lidar_model = LidarMeasurementModel(
            lidar_position=np.array(lidar_cfg.lidar_position),
            lidar_std_dev=tracker_cfg.lidar_std_dev,
            pca_mean=pca_params['mean'],
            pca_eigenvectors=pca_params['eigenvectors'][:, :tracker_cfg.N_pca].real,
            extent_cfg=extent_cfg
        )

        if method == "bfgs":
            tracker = BFGS(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
        elif method == "ekf":
            tracker = EKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
        elif method == "iekf":
            tracker = IterativeEKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
        elif method == "implicit_iekf":
            tracker = ImplicitIEKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
        else:
            raise ValueError(f"Unknown method {method}")

        lidar_simulator = LidarSimulator(
            lidar_position=np.array(lidar_cfg.lidar_position),
            num_rays=lidar_cfg.num_rays,
            max_distance=lidar_cfg.max_distance,
            lidar_gt_std_dev=lidar_cfg.lidar_gt_std_dev,
            rng=rng,
            extent_cfg=extent_cfg
        )

        simulator = Simulator(
            dynamic_model=gt_dynamic_model,
            sensor_model=lidar_simulator,
            sensor_setter=None,
            init_state=sim_cfg.initial_state_gt,
            dt=sim_cfg.dt,
            end_time=sim_cfg.num_frames * sim_cfg.dt,
            seed=str(sim_cfg.seed),
            use_cache=False
        )

    # --- Generate Simulation Data ---
    print(f"Generating simulation data for {sim_cfg.num_frames} frames...")
    ground_truth_ts = simulator.get_gt()
    
    DEBUG = True
    # DEBUG: Plot GT Trajectory
    if DEBUG:
        ts_values = list(ground_truth_ts.values)
        # NED Frame: x is North, y is East.
        # We want North on Y-axis (vertical) and East on X-axis (horizontal).
        # So Plot X = state.y (East), Plot Y = state.x (North)
        
        east_vals = [state.y for state in ts_values]
        north_vals = [state.x for state in ts_values]
        
        p = figure(title="Debug: Ground Truth Trajectory with Vessel Extent", 
                   x_axis_label='East (y)', y_axis_label='North (x)',
                   match_aspect=True, width=800, height=800)

        # Plot Trajectory
        p.line(east_vals, north_vals, legend_label="GT Path", line_width=2, color="blue")
        p.circle(east_vals, north_vals, size=4, color="blue", alpha=0.5)

        # Start/End markers
        p.scatter([east_vals[0]], [north_vals[0]], size=10, color="green", legend_label="Start")
        p.scatter([east_vals[-1]], [north_vals[-1]], size=10, color="red", legend_label="End")

        # Plot Center of Trajectory (if circle)
        if traj_cfg.type == "circle":
            center_north, center_east = traj_cfg.center
            p.scatter([center_east], [center_north], size=10, color="orange", marker="cross", legend_label="Traj Center")

        # Plot Waypoints (if waypoints)
        if traj_cfg.type == "waypoints" and hasattr(traj_cfg, 'waypoints'):
            wp_north = [wp[0] for wp in traj_cfg.waypoints]
            wp_east = [wp[1] for wp in traj_cfg.waypoints]
            p.scatter(wp_east, wp_north, size=10, color="purple", marker="triangle", legend_label="Waypoints")
            # Connect waypoints with a dashed line to show intended path
            p.line(wp_east, wp_north, line_color="purple", line_dash="dashed", line_width=1, alpha=0.5)

        # Plot Lidar Position
        lidar_north, lidar_east = lidar_cfg.lidar_position
        p.scatter([lidar_east], [lidar_north], size=15, color="orange", marker="triangle", legend_label="Lidar")

        # Plot vessel extent every 20 frames
        for i in range(0, len(ts_values), 20):
            state = ts_values[i]
            shape_x, shape_y = compute_exact_vessel_shape_global(state, extent_cfg.shape_coords_body)
            # shape_x is North, shape_y is East. Plot (East, North) -> (shape_y, shape_x)
            p.line(shape_y, shape_x, line_color="black", line_width=1, line_alpha=0.5)

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        
        # Open in browser
        import webbrowser
        from bokeh.io import show
        # If running in a notebook, output_notebook() is needed, but assuming script execution here:
        try:
            show(p) 
        except Exception as e:
            print(f"Could not show Bokeh plot: {e}")

    measurements_lidar_frame_ts = simulator.get_meas()

    lidar_pos_global = np.array(lidar_cfg.lidar_position).reshape(2, 1)
    measurements_global_ts = measurements_lidar_frame_ts.map(lambda scan: scan + lidar_pos_global)

    # --- Run Filtering Loop ---
    results_ts: TimeSequence[TrackerUpdateResult] = TimeSequence() 
    
    # --- Insert the initial state at t=0 ---
    initial_result = tracker.get_initial_update_result()
    results_ts.insert(0.0, initial_result)

    for ts, measurement in tqdm(measurements_lidar_frame_ts.items(), desc="Filtering measurements"):
        tracker.predict()
        update_result = tracker.update(measurement, ground_truth=ground_truth_ts.get_t(ts))
        results_ts.insert(ts, update_result)

    filename = os.path.join(SIMDATA_PATH, f"{sim_cfg.name}.pkl")

    # NOTE Martin: static_covariances might need adjustment for GP, but keeping generic for now
    static_covariances = {
        "Q": filter_dyn_model.Q_d(dt=sim_cfg.dt),
        "R_point": lidar_model.R_single_point()
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