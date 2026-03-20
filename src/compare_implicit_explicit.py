import os
import sys
import numpy as np
from pathlib import Path
import pickle
from zlib import crc32
import logging
import panel as pn

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.tracker.EKF import EKF

import plotly.graph_objects as go
from plotly.subplots import make_subplots
pn.extension('plotly')

from src.utils.config_classes import Config
from src.dynamics.process_models import Model_PCA_CV
from src.sensors.LidarModel import LidarMeasurementModel, LidarSimulator
from src.tracker.IterativeEKF import IterativeEKF
from src.tracker.ImplicitIEKF import ImplicitIEKF
from src.utils.geometry_utils import compute_estimated_shape_global, compute_exact_vessel_shape_global
from src.utils.tools import calculate_body_angles
from src.senfuslib.gaussian import MultiVarGauss
from src.states.states import LidarScan, State_PCA

from global_project_paths import SIMDATA_PATH
from src.utils.config_classes import TrackerConfig, SimulationConfig, ExtentConfig, LidarConfig, TrajectoryConfig
from src.states.states import State_GP, State_PCA
from src.experiment_runner import run_single_simulation
from src.utils import SimulationResult
from src.extent_model.boat_pca_utils import get_gt_pca_coeffs_for_boat

PCA_parameters_path = "data/input_parameters/ShipDatasetPCAParameters.npz"


def get_common_configs(traj_type="linear", N_pca=4):
    """Returns configs shared by all methods (Sim, Lidar, Extent)."""
    selected_boat_id = "1"
    try:
        L_gt = 20.0
        W_gt = 6.0
        gt_pca_coeffs = get_gt_pca_coeffs_for_boat(selected_boat_id, N_pca=N_pca, pca_path=PCA_parameters_path)
    except Exception as e:
        logging.error(f"Could not load boat {selected_boat_id}: {e}")
        L_gt, W_gt = 20.0, 6.0
        gt_pca_coeffs = np.zeros(N_pca)
        selected_boat_id = None

    if traj_type == "circle":
        trajectory = TrajectoryConfig(type="circle", center=(30.0, 0.0), radius=30.0, speed=5.0, clockwise=False)
        start_x, start_y, start_yaw = 0.0, 0.0, np.pi/2
    elif traj_type == "linear":
        trajectory = TrajectoryConfig(type="linear", speed=5.0)
        start_x, start_y, start_yaw = 0.0, -40.0, np.pi/2
    elif traj_type == "waypoints":
        trajectory = TrajectoryConfig(type="waypoints", speed=5.0, waypoints=[(0, -40), (0, 40), (60, 40), (60, -40)])
        start_x, start_y, start_yaw = 0.0, -40.0, np.pi/2
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    initial_state_gt = State_PCA(
        x=start_x, y=start_y, yaw=start_yaw, vel_x=0.0, vel_y=3.0, yaw_rate=0.0,
        length=L_gt, width=W_gt, pca_coeffs=gt_pca_coeffs[:N_pca]
    )

    sim_config = SimulationConfig(
        name="", num_simulations=1, num_frames=500, dt=0.1, seed=42,
        initial_state_gt=initial_state_gt, gt_yaw_rate_std_dev=0.1 if traj_type == "linear" else 0.0,
        trajectory=trajectory, use_cache=True
    )

    lidar_config = LidarConfig(lidar_position=(30.0, 0.0), num_rays=360, max_distance=140.0, lidar_gt_std_dev=0.0)

    if selected_boat_id:
        shape_params = {"type": "database", "id": selected_boat_id, "L": L_gt, "W": W_gt}
    else:
        shape_params = {"type": "ellipse", "L": L_gt, "W": W_gt}
        
    extent_config = ExtentConfig(N_fourier=64, d_angle=np.deg2rad(1.0), shape_params_true=shape_params)
    return sim_config, lidar_config, extent_config

def get_pca_tracker_config(lidar_pos, initial_state, initial_std_devs, pos_n_std, pos_e_std, head_std, lidar_std, N_pca=4):
    pca_data = np.load(PCA_parameters_path)
    eigenvalues = pca_data['eigenvalues'][:N_pca].real

    tracker_config = TrackerConfig(
        use_gt_state_for_bodyangles_calc=False,
        use_initialize_centroid=False,
        N_pca=N_pca,
        PCA_parameters_path=PCA_parameters_path,
        pos_north_std_dev=pos_n_std,
        pos_east_std_dev=pos_e_std,
        heading_std_dev=head_std,
        lidar_std_dev=lidar_std,
        initial_state=initial_state,
        initial_std_devs=initial_std_devs,
        lidar_position=np.array(lidar_pos),
        pca_eigenvalues=eigenvalues
    )
    return tracker_config

class CompareDashboard(pn.viewable.Viewer):
    def __init__(self):
        self.N_pca = 4
        self.pca_params = np.load(PCA_parameters_path)
        self.cached_sim_data = {}

        # ---------------- MODE ----------------
        self.mode_selector = pn.widgets.Select(name="Mode", options=["Manual Setup", "Load from Simulation"], sizing_mode='stretch_width')

        # ---------------- WIDGETS (MANUAL) ----------------
        self.widgets = []
        self.manual_widgets = []
        def create_widget_pair(name, start, end, step, value):
            slider = pn.widgets.FloatSlider(name=name, start=start, end=end, step=step, value=value, sizing_mode='stretch_width')
            inp = pn.widgets.FloatInput(name='', start=start, end=end, step=step, value=value, width=70)
            slider.link(inp, value='value', bidirectional=True)
            self.widgets.append(slider)
            self.manual_widgets.append(slider)
            return slider, inp

        # 1. GT State
        self.w_gt_x, self.i_gt_x = create_widget_pair("GT X", -10.0, 10.0, 0.1, 0.0)
        self.w_gt_y, self.i_gt_y = create_widget_pair("GT Y", -10.0, 10.0, 0.1, 0.0)
        self.w_gt_yaw, self.i_gt_yaw = create_widget_pair("GT Yaw (deg)", 0.0, 360.0, 1.0, 90.0)
        
        # 2. Prior Offsets
        self.w_pr_offset_x, self.i_pr_offset_x = create_widget_pair("Prior Offset X", -10.0, 10.0, 0.1, -2.0)
        self.w_pr_offset_y, self.i_pr_offset_y = create_widget_pair("Prior Offset Y", -10.0, 10.0, 0.1, 1.0)
        self.w_pr_offset_yaw, self.i_pr_offset_yaw = create_widget_pair("Prior Offset Yaw (deg)", -45.0, 45.0, 1.0, 10.0)
        
        # 3. Initial Covariance std_devs
        self.w_init_std_x, self.i_init_std_x = create_widget_pair("Init Std X", 0.1, 10.0, 0.1, 2.0)
        self.w_init_std_y, self.i_init_std_y = create_widget_pair("Init Std Y", 0.1, 10.0, 0.1, 2.0)
        self.w_init_std_yaw, self.i_init_std_yaw = create_widget_pair("Init Std Yaw (rad)", 0.01, 3.14, 0.01, 0.2)
        
        self.w_init_std_v, self.i_init_std_v = create_widget_pair("Init Std Vel", 0.1, 5.0, 0.1, 2.0)
        self.w_init_std_rate, self.i_init_std_rate = create_widget_pair("Init Std Yaw Rate", 0.01, 1.0, 0.01, 0.1)
        self.w_init_std_dim, self.i_init_std_dim = create_widget_pair("Init Std L/W", 0.1, 5.0, 0.1, 2.0)
        self.w_init_std_pca, self.i_init_std_pca = create_widget_pair("Init Std PCA", 0.1, 5.0, 0.1, 1.0)

        # 4. Tracker Std Devs
        self.w_trk_pos_n, self.i_trk_pos_n = create_widget_pair("Tracker Pos N Std", 0.01, 2.0, 0.01, 0.3)
        self.w_trk_pos_e, self.i_trk_pos_e = create_widget_pair("Tracker Pos E Std", 0.01, 2.0, 0.01, 0.3)
        self.w_trk_head, self.i_trk_head = create_widget_pair("Tracker Head Std", 0.01, 1.0, 0.01, 0.1)
        self.w_trk_lidar, self.i_trk_lidar = create_widget_pair("Tracker LiDAR Std", 0.01, 1.0, 0.01, 0.15)
        
        # ---------------- WIDGETS (SIMULATION) ----------------
        pickle_files = sorted([f.name for f in Path(SIMDATA_PATH).glob("*.pkl")], reverse=True)
        self.file_selector = pn.widgets.Select(name="Select Simulation File", options=pickle_files, sizing_mode="stretch_width")
        self.frame_slider = pn.widgets.IntSlider(name="Frame Index", start=0, end=100, step=1, value=0, sizing_mode="stretch_width")
        self.frame_input = pn.widgets.IntInput(name="", start=0, step=1, value=0, width=70)
        self.frame_slider.link(self.frame_input, value='value', bidirectional=True)
        self.sim_widgets = [self.file_selector, self.frame_input, self.frame_slider]
        
        self.plotly_pane = pn.pane.Plotly(sizing_mode='stretch_both', min_height=700)
        
        def row(*args):
            return pn.Row(*args, sizing_mode='stretch_width')

        self.manual_sidebar = pn.Column(
            pn.pane.Markdown("### 1. Ground Truth"), row(self.w_gt_x, self.i_gt_x), row(self.w_gt_y, self.i_gt_y), row(self.w_gt_yaw, self.i_gt_yaw),
            pn.pane.Markdown("### 2. Prior Offset (from GT)"), row(self.w_pr_offset_x, self.i_pr_offset_x), row(self.w_pr_offset_y, self.i_pr_offset_y), row(self.w_pr_offset_yaw, self.i_pr_offset_yaw),
            pn.pane.Markdown("### 3. Initial Std Devs"), row(self.w_init_std_x, self.i_init_std_x), row(self.w_init_std_y, self.i_init_std_y), row(self.w_init_std_yaw, self.i_init_std_yaw), row(self.w_init_std_v, self.i_init_std_v), row(self.w_init_std_rate, self.i_init_std_rate), row(self.w_init_std_dim, self.i_init_std_dim), row(self.w_init_std_pca, self.i_init_std_pca),
            pn.pane.Markdown("### 4. Tracker Config Std Devs"), row(self.w_trk_pos_n, self.i_trk_pos_n), row(self.w_trk_pos_e, self.i_trk_pos_e), row(self.w_trk_head, self.i_trk_head), row(self.w_trk_lidar, self.i_trk_lidar),
            sizing_mode='stretch_width'
        )

        self.sim_sidebar = pn.Column(
            pn.pane.Markdown("### Load from Simulation"),
            self.file_selector, row(self.frame_slider, self.frame_input),
            sizing_mode='stretch_width',
            visible=False
        )

        # Watchers for auto-update
        self.mode_selector.param.watch(self.on_mode_change, 'value')
        for w in self.manual_widgets + self.sim_widgets:
            w.param.watch(self.update_plot, 'value')
        
        # Initial run
        self.update_plot(None)

    def on_mode_change(self, event):
        if event.new == "Manual Setup":
            self.manual_sidebar.visible = True
            self.sim_sidebar.visible = False
        else:
            self.manual_sidebar.visible = False
            self.sim_sidebar.visible = True
        self.update_plot(None)

    def load_sim_data(self, filename):
        if filename in self.cached_sim_data:
            return self.cached_sim_data[filename]
        
        filepath = Path(SIMDATA_PATH) / filename
        if not filepath.exists():
            return None
            
        with open(filepath, "rb") as f:
            sim_result = pickle.load(f)
            
        config = sim_result.config
        pca_params = self.pca_params
        if hasattr(config.tracker, 'PCA_parameters_path'):
            pca_params = np.load(PROJECT_ROOT / config.tracker.PCA_parameters_path)
            
        data = {
            "sim_result": sim_result,
            "config": config,
            "pca_params": pca_params
        }
        self.cached_sim_data[filename] = data
        return data

    def get_manual_setup(self):
        sim_base, lidar_base, extent_base = get_common_configs(traj_type="linear", N_pca=self.N_pca)
        
        gt_state = sim_base.initial_state_gt.copy()
        gt_state.x = self.w_gt_x.value
        gt_state.y = self.w_gt_y.value
        gt_state.yaw = np.deg2rad(self.w_gt_yaw.value)
        
        prior_state = gt_state.copy()
        prior_state.x += self.w_pr_offset_x.value
        prior_state.y += self.w_pr_offset_y.value
        prior_state.yaw += np.deg2rad(self.w_pr_offset_yaw.value)
        
        initial_std_devs_tracker = State_PCA(
            x=self.w_init_std_x.value, y=self.w_init_std_y.value, yaw=self.w_init_std_yaw.value, 
            vel_x=self.w_init_std_v.value, vel_y=self.w_init_std_v.value, yaw_rate=self.w_init_std_rate.value,
            length=self.w_init_std_dim.value, width=self.w_init_std_dim.value,
            pca_coeffs=np.ones(self.N_pca) * self.w_init_std_pca.value
        )
        
        tracker_cfg = get_pca_tracker_config(
            lidar_base.lidar_position, initial_state=prior_state, initial_std_devs=initial_std_devs_tracker,
            pos_n_std=self.w_trk_pos_n.value, pos_e_std=self.w_trk_pos_e.value, 
            head_std=self.w_trk_head.value, lidar_std=self.w_trk_lidar.value, N_pca=self.N_pca
        )
        tracker_cfg.use_initialize_centroid = False 
        config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_cfg, extent=extent_base)

        rng = np.random.default_rng(42)
        lidar_simulator = LidarSimulator(
            lidar_position=np.array(lidar_base.lidar_position), num_rays=lidar_base.num_rays,
            max_distance=lidar_base.max_distance, lidar_gt_std_dev=0.0, rng=rng, extent_cfg=extent_base
        )
        meas_local = lidar_simulator.sample_from_state(gt_state)
        
        std_dev_list = [
            initial_std_devs_tracker.x, initial_std_devs_tracker.y, initial_std_devs_tracker.yaw,
            initial_std_devs_tracker.vel_x, initial_std_devs_tracker.vel_y, initial_std_devs_tracker.yaw_rate,
            initial_std_devs_tracker.length, initial_std_devs_tracker.width
        ]
        if hasattr(initial_std_devs_tracker, 'pca_coeffs'):
            std_dev_list.extend(initial_std_devs_tracker.pca_coeffs)
        prior_cov = np.diag(np.array(std_dev_list) ** 2)
        prior_gauss = MultiVarGauss(mean=prior_state, cov=prior_cov)
        
        return config, gt_state, meas_local, prior_gauss, self.pca_params

    def get_sim_setup(self):
        filename = self.file_selector.value
        if not filename:
            return None
        data = self.load_sim_data(filename)
        if not data:
            return None
            
        sim_result = data["sim_result"]
        config = data["config"]
        pca_params = data["pca_params"]
        
        frame_idx = self.frame_input.value
        if frame_idx < 0:
            frame_idx = 0
            self.frame_input.value = 0
            
        gt_states = list(sim_result.ground_truth_ts.values)
        tracker_results = list(sim_result.tracker_results_ts.values)
        
        self.frame_slider.end = max(0, len(gt_states) - 1)
        
        if frame_idx >= len(gt_states):
            frame_idx = len(gt_states) - 1
            self.frame_input.value = frame_idx
            
        gt_state = gt_states[frame_idx]
        tracker_result = tracker_results[frame_idx]
        
        meas_raw = tracker_result.measurements
        if meas_raw is not None and len(meas_raw) > 0:
            meas_array = np.asarray(meas_raw)
            if meas_array.ndim == 1:
                # The .pkl saves measurements as a flattened array of GLOBAL coordinates.
                # Expected: [x1, y1, x2, y2 ...] => shape (-1, 2)
                meas_global_cart = meas_array.reshape((-1, 2)).T # shape (2, N)
                lidar_pos = np.array(config.lidar.lidar_position).reshape(2, 1)
                meas_local_cart = meas_global_cart - lidar_pos
                meas_local = LidarScan(x=meas_local_cart[0], y=meas_local_cart[1])
            else:
                meas_local = LidarScan(x=meas_array[0], y=meas_array[1])
        else:
            meas_local = None
            
        prior_gauss = tracker_result.state_prior
        
        return config, gt_state, meas_local, prior_gauss, pca_params

    def update_plot(self, event):
        if self.mode_selector.value == "Manual Setup":
            setup = self.get_manual_setup()
        else:
            setup = self.get_sim_setup()
            
        if not setup:
            self.plotly_pane.object = go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return
            
        config, gt_state, meas_local, prior_gauss, pca_params = setup
        prior_state = prior_gauss.mean
        
        if meas_local is None or len(meas_local) == 0:
            self.plotly_pane.object = go.Figure().add_annotation(text="No measurements in this frame", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return

        lidar_pos_global = np.array(config.lidar.lidar_position).reshape(2, 1)
        meas_global_raw = meas_local + lidar_pos_global
        meas_global = meas_local.__class__(x=meas_global_raw[0], y=meas_global_raw[1])

        N_pca = config.tracker.N_pca
        pca_mean = pca_params['mean']
        pca_eigenvectors = pca_params['eigenvectors'][:, :N_pca].real

        sensor_model = LidarMeasurementModel(
            lidar_position=np.array(config.lidar.lidar_position),
            lidar_std_dev=config.tracker.lidar_std_dev,
            pca_mean=pca_mean,
            pca_eigenvectors=pca_eigenvectors,
            extent_cfg=config.extent
        )

        dyn_model = Model_PCA_CV(
            x_pos_std_dev=config.tracker.pos_north_std_dev, y_pos_std_dev=config.tracker.pos_east_std_dev,
            yaw_std_dev=config.tracker.heading_std_dev, N_pca=N_pca
        )

        # Run trackers
        tracker_exp = IterativeEKF(dynamic_model=dyn_model, lidar_model=sensor_model, config=config, max_iterations=5)
        tracker_exp.state_estimate = prior_gauss
        res_exp = tracker_exp.update(meas_local)

        tracker_imp = ImplicitIEKF(dynamic_model=dyn_model, lidar_model=sensor_model, config=config, max_iterations=5)
        tracker_imp.state_estimate = prior_gauss
        res_imp = tracker_imp.update(meas_local)

        tracker_imp_ekf = ImplicitIEKF(dynamic_model=dyn_model, lidar_model=sensor_model, config=config, max_iterations=1)
        tracker_imp_ekf.state_estimate = prior_gauss
        res_imp_ekf = tracker_imp_ekf.update(meas_local)

        # Re-compute z_pred using prior
        if hasattr(sensor_model, 'h_lidar') and getattr(config.tracker, 'use_gt_state_for_bodyangles_calc', False) == False:
            body_angles = calculate_body_angles(meas_global, prior_state)
            z_pred_exp = sensor_model.h_lidar(prior_state, body_angles).flatten().reshape(-1, 2).T
        else:
            # Fallback if h_lidar fails
            try:
                body_angles = calculate_body_angles(meas_global, prior_state)
                z_pred_exp = sensor_model.h_lidar(prior_state, body_angles).flatten().reshape(-1, 2).T
            except:
                z_pred_exp = np.zeros_like(meas_global)

        # Plotting
        gt_x, gt_y = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)
        pr_x, pr_y = compute_estimated_shape_global(prior_state, config, pca_params)
        exp_x, exp_y = compute_estimated_shape_global(res_exp.state_posterior.mean, config, pca_params)
        imp_x, imp_y = compute_estimated_shape_global(res_imp.state_posterior.mean, config, pca_params)
        imp_ekf_x, imp_ekf_y = compute_estimated_shape_global(res_imp_ekf.state_posterior.mean, config, pca_params)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("1. Prior State & Association Rays", "2. Iteration Path & Final Updates"), 
                            horizontal_spacing=0.1)

        fig.add_trace(go.Scatter(x=gt_y, y=gt_x, mode='lines', line=dict(color='black', dash='dash'), name='Ground Truth Shape'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[gt_state.y], y=[gt_state.x], mode='markers', marker=dict(color='black', symbol='diamond', size=8), name='GT Centroid'), row=1, col=1)
        fig.add_trace(go.Scatter(x=meas_global[1, :], y=meas_global[0, :], mode='markers', marker=dict(color='red', size=5), name='Measurements(z)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=pr_y, y=pr_x, mode='lines', line=dict(color='blue'), name='Prior Shape'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[prior_state.y], y=[prior_state.x], mode='markers', marker=dict(color='blue', symbol='diamond', size=8), name='Prior Centroid'), row=1, col=1)
        if z_pred_exp is not None and z_pred_exp.shape[1] > 0:
            fig.add_trace(go.Scatter(x=z_pred_exp[1, :], y=z_pred_exp[0, :], mode='markers', marker=dict(color='blue', symbol='x', size=6), name='z_pred (Prior)'), row=1, col=1)

            ray_x, ray_y = [],[]
            for i in range(meas_global.shape[1]):
                ray_x.extend([meas_global[1, i], z_pred_exp[1, i], None])
                ray_y.extend([meas_global[0, i], z_pred_exp[0, i], None])
            fig.add_trace(go.Scatter(x=ray_x, y=ray_y, mode='lines', line=dict(color='gray', width=1, dash='dot'), opacity=0.5, name='Assoc Rays'), row=1, col=1)

        fig.add_trace(go.Scatter(x=gt_y, y=gt_x, mode='lines', line=dict(color='black', dash='dash'), opacity=0.4, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[gt_state.y], y=[gt_state.x], mode='markers', marker=dict(color='black', symbol='diamond', size=6), opacity=0.4, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=meas_global[1, :], y=meas_global[0, :], mode='markers', marker=dict(color='red', size=4), opacity=0.4, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=pr_y, y=pr_x, mode='lines', line=dict(color='blue'), opacity=0.3, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[prior_state.y], y=[prior_state.x], mode='markers', marker=dict(color='blue', symbol='diamond', size=6), opacity=0.3, showlegend=False), row=1, col=2)

        fig.add_trace(go.Scatter(x=exp_y, y=exp_x, mode='lines', line=dict(color='orange', width=2), name='Explicit IEKF'), row=1, col=2)
        if hasattr(res_exp, 'iterates') and res_exp.iterates:
            fig.add_trace(go.Scatter(x=[s.y for s in res_exp.iterates], y=[s.x for s in res_exp.iterates], 
                mode='lines+markers', marker=dict(color='orange', size=8), name='Exp Path'), row=1, col=2)

        fig.add_trace(go.Scatter(x=imp_y, y=imp_x, mode='lines', line=dict(color='green', width=3), name='Implicit IEKF'), row=1, col=2)
        if hasattr(res_imp, 'iterates') and res_imp.iterates:
            fig.add_trace(go.Scatter(x=[s.y for s in res_imp.iterates], y=[s.x for s in res_imp.iterates], 
                mode='lines+markers', marker=dict(color='green', size=8, symbol='x'), name='Imp Path'), row=1, col=2)

        fig.add_trace(go.Scatter(x=imp_ekf_y, y=imp_ekf_x, mode='lines', line=dict(color='purple', width=2), name='Implicit EKF'), row=1, col=2)
        if hasattr(res_imp_ekf, 'iterates') and res_imp_ekf.iterates:
            fig.add_trace(go.Scatter(x=[s.y for s in res_imp_ekf.iterates], y=[s.x for s in res_imp_ekf.iterates], 
                mode='lines+markers', marker=dict(color='purple', size=8, symbol='triangle-up'), name='Imp EKF Path'), row=1, col=2)

        lidar_x = float(lidar_pos_global[0, 0])
        lidar_y = float(lidar_pos_global[1, 0])
        fig.add_trace(go.Scatter(x=[lidar_y], y=[lidar_x], mode='markers', marker=dict(color='darkorange', symbol='cross', size=10), name='LiDAR Origin'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[lidar_y], y=[lidar_x], mode='markers', marker=dict(color='darkorange', symbol='cross', size=10), showlegend=False), row=1, col=2)

        fig.update_layout(title="Explicit vs Implicit IEKF", plot_bgcolor='white', margin=dict(l=20, r=20, t=60, b=20))
        # Ensure correct range centered on GT
        x_min, x_max = gt_state.x - 30, gt_state.x + 30
        y_min, y_max = gt_state.y - 30, gt_state.y + 30
        
        fig.update_xaxes(title_text="East (y) [m]", range=[y_min, y_max], showgrid=True, gridcolor='lightgrey', row=1, col=1)
        fig.update_yaxes(title_text="North (x) [m]", range=[x_min, x_max], scaleanchor="x", scaleratio=1, showgrid=True, gridcolor='lightgrey', row=1, col=1)
        fig.update_xaxes(title_text="East (y) [m]", range=[y_min, y_max], showgrid=True, gridcolor='lightgrey', row=1, col=2)
        fig.update_yaxes(title_text="North (x) [m]", range=[x_min, x_max], scaleanchor="x2", scaleratio=1, showgrid=True, gridcolor='lightgrey', row=1, col=2)
        
        self.plotly_pane.object = fig

    def view(self):
        sidebar = pn.Column(
            self.mode_selector,
            pn.layout.Divider(),
            self.manual_sidebar,
            self.sim_sidebar,
            sizing_mode='stretch_width'
        )

        main_area = pn.Column(self.plotly_pane, sizing_mode='stretch_both')
        
        return pn.template.FastListTemplate(
            title="Comparison of Explicit vs Implicit Dashboard",
            sidebar=[sidebar],
            main=[main_area],
            sidebar_width=450
        )

if __name__ == "__main__":
    app = CompareDashboard()
    app.view().show()
