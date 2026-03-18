import os
import sys
import numpy as np
from pathlib import Path
import sys
import pickle
import numpy as np
from pathlib import Path
from zlib import crc32
import logging

# Plotly imports
from plotly.subplots import make_subplots

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config_classes import Config
from src.dynamics.process_models import Model_PCA_CV
from src.sensors.LidarModel import LidarMeasurementModel, LidarSimulator
from src.tracker.IterativeEKF import IterativeEKF
from src.tracker.ImplicitIEKF import ImplicitIEKF
from src.utils.geometry_utils import compute_estimated_shape_global, compute_exact_vessel_shape_global
from src.senfuslib.gaussian import MultiVarGauss
from src.utils.tools import calculate_body_angles

# Imports 
from global_project_paths import SIMDATA_PATH
from src.utils.config_classes import TrackerConfig, SimulationConfig, Config, ExtentConfig, LidarConfig, TrajectoryConfig
from src.states.states import State_GP, State_PCA
from src.experiment_runner import run_single_simulation
from src.utils import SimulationResult
from src.extent_model.boat_pca_utils import get_gt_pca_coeffs_for_boat

PCA_parameters_path = "data/input_parameters/ShipDatasetPCAParameters.npz" # NOTE Global variable for PCA parameters path

def get_common_configs(traj_type="circle", N_pca=4):
    """Returns configs shared by all methods (Sim, Lidar, Extent)."""
    
    # --- BOAT SELECTION ---
    # Select a boat from processed_ships.json
    selected_boat_id = "1" # Example: "1" = Sailing Yacht, "112" = Multihull
    
    # Get GT Dimensions and PCA Coeffs
    try:
        # L_gt, W_gt = get_boat_dimensions(selected_boat_id) # NOTE We can get L and W from the database but in practice we set them manually
        L_gt = 20.0
        W_gt = 6.0
        gt_pca_coeffs = get_gt_pca_coeffs_for_boat(selected_boat_id, N_pca=N_pca, pca_path=PCA_parameters_path)
    except Exception as e:
        logging.error(f"Could not load boat {selected_boat_id}: {e}")
        # Fallback to simple ellipse
        L_gt, W_gt = 20.0, 6.0
        gt_pca_coeffs = np.zeros(N_pca)
        selected_boat_id = None

    # --- TRAJECTORY ---
    if traj_type == "circle":
        trajectory = TrajectoryConfig(
            type="circle",   
            center=(30.0, 0.0),
            radius=30.0,
            speed=5.0,
            clockwise=False
        )
        start_x, start_y, start_yaw = 0.0, 0.0, np.pi/2

    elif traj_type == "linear":
        trajectory = TrajectoryConfig(
            type="linear",
            speed=5.0
        )
        start_x, start_y, start_yaw = 0.0, -40.0, np.pi/2
    
    elif traj_type == "waypoints":
        trajectory = TrajectoryConfig(
            type="waypoints",
            speed=5.0,
            waypoints=[(0, -40), (0, 40), (60, 40), (60, -40)]
        )
        start_x, start_y, start_yaw = 0.0, -40.0, np.pi/2
        
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    # --- GT STATE ---
    initial_state_gt = State_PCA(
        x=start_x,      
        y=start_y,      
        yaw=start_yaw,  
        vel_x=0.0,     
        vel_y=3.0,      
        yaw_rate=0.0,   
        length=L_gt,   
        width=W_gt,      
        pca_coeffs=gt_pca_coeffs[:N_pca] # Truncate to state size
    )

    sim_config = SimulationConfig(
        name = "",
        num_simulations=1,
        num_frames=500,
        dt=0.1,
        seed=42,
        initial_state_gt=initial_state_gt,
        gt_yaw_rate_std_dev= 0.1 if traj_type == "linear" else 0.0, 
        trajectory=trajectory,
        use_cache=True
    )

    # LiDAR Parameters
    lidar_config = LidarConfig(
        lidar_position=(30.0, 0.0),
        num_rays=360,
        max_distance=140.0,
        lidar_gt_std_dev=0.0, # Perfect measurements for testing
    )

    # Extent config
    # Use "database" type to load the real shape from JSON
    if selected_boat_id:
        shape_params = {
            "type": "database", 
            "id": selected_boat_id,
            "L": L_gt, 
            "W": W_gt 
        }
    else:
        shape_params = {
            "type": "ellipse", 
            "L": L_gt, 
            "W": W_gt 
        }
        
    extent_config = ExtentConfig(
        N_fourier=64,
        d_angle=np.deg2rad(1.0),
        shape_params_true=shape_params
    )
    
    return sim_config, lidar_config, extent_config

def get_pca_tracker_config(lidar_pos, initial_state_gt, N_pca=4):
    """Returns TrackerConfig for PCA methods."""
    
    # Initialize Tracker closer to GT for stability in this test
    # (Or add noise if testing robustness)
    initial_state_tracker = State_PCA(
        x=initial_state_gt.x,
        y=initial_state_gt.y,
        yaw=initial_state_gt.yaw,
        vel_x=initial_state_gt.vel_x, 
        vel_y=initial_state_gt.vel_y, 
        yaw_rate=0.0,
        length=initial_state_gt.length,    
        width=initial_state_gt.width,      
        # pca_coeffs=np.zeros(N_pca) # Start with mean shape of dataset
        pca_coeffs=initial_state_gt.pca_coeffs.copy() # Start with perfect shape
    )

    initial_std_devs_tracker = State_PCA(
        x=2.0, y=2.0, yaw=0.2, 
        vel_x=2.0, vel_y=2.0, yaw_rate=0.1,
        length=2.0, width=2.0,
        pca_coeffs=np.ones(N_pca) * 1
    )

    pca_data = np.load(PCA_parameters_path)
    eigenvalues = pca_data['eigenvalues'][:N_pca].real

    tracker_config = TrackerConfig(
        use_gt_state_for_bodyangles_calc = False,
        use_initialize_centroid = False,
        N_pca=N_pca,
        PCA_parameters_path=PCA_parameters_path,
        pos_north_std_dev=0.3,
        pos_east_std_dev=0.3,
        heading_std_dev=0.1,
        lidar_std_dev=0.15,
        initial_state=initial_state_tracker,
        initial_std_devs=initial_std_devs_tracker,
        lidar_position=np.array(lidar_pos),
        pca_eigenvalues=eigenvalues
    )
    return tracker_config

def run_comparison():
    N_pca = 4
    
    # 1. Load Configurations
    sim_base, lidar_base, extent_base = get_common_configs(traj_type="linear", N_pca=N_pca)
    tracker_cfg = get_pca_tracker_config(lidar_base.lidar_position, sim_base.initial_state_gt, N_pca)
    
    # Disable centroid initialization to test pure Jacobian optimization from our bad prior
    tracker_cfg.use_initialize_centroid = False 
    config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_cfg, extent=extent_base)
    pca_params = np.load(PCA_parameters_path)

    # 2. Setup Ground Truth State
    gt_state = sim_base.initial_state_gt.copy()
    gt_state.x = 0.0
    gt_state.y = 0.0
    gt_state.yaw = np.pi / 2 

    # 3. Simulate Actual LiDAR Measurements
    rng = np.random.default_rng(42)
    lidar_simulator = LidarSimulator(
        lidar_position=np.array(lidar_base.lidar_position),
        num_rays=lidar_base.num_rays,
        max_distance=lidar_base.max_distance,
        lidar_gt_std_dev=0.0,  # Minor measurement noise
        rng=rng,
        extent_cfg=extent_base
    )
    meas_local = lidar_simulator.sample_from_state(gt_state)
    lidar_pos_global = np.array(lidar_base.lidar_position).reshape(2, 1)
    meas_global = meas_local + lidar_pos_global

    # 4. Create an intentionally BAD Prior State
    prior_state = gt_state.copy()
    prior_state.x += -2.0  
    prior_state.y += 1.0
    prior_state.yaw += np.deg2rad(10)
    
    # Create a realistic initial covariance matrix
    initial_std_devs = tracker_cfg.initial_std_devs
    
    std_dev_list = [
        initial_std_devs.x,
        initial_std_devs.y,
        initial_std_devs.yaw,
        initial_std_devs.vel_x,
        initial_std_devs.vel_y,
        initial_std_devs.yaw_rate,
        initial_std_devs.length,
        initial_std_devs.width
    ]
    if hasattr(initial_std_devs, 'pca_coeffs'):
        std_dev_list.extend(initial_std_devs.pca_coeffs)

    prior_cov = np.diag(np.array(std_dev_list) ** 2)
    prior_gauss = MultiVarGauss(mean=prior_state, cov=prior_cov)

    # 5. Extract Models to manually test z_pred
    sensor_model = LidarMeasurementModel(
        lidar_position=np.array(lidar_base.lidar_position),
        lidar_std_dev=tracker_cfg.lidar_std_dev,
        pca_mean=pca_params['mean'],
        pca_eigenvectors=pca_params['eigenvectors'][:, :N_pca].real,
        extent_cfg=extent_base
    )

    # --- EXPLICIT MODEL MATH ---
    body_angles = calculate_body_angles(meas_global, prior_state)
    z_pred_exp_flat = sensor_model.h_lidar(prior_state, body_angles).flatten()
    z_pred_exp = z_pred_exp_flat.reshape(-1, 2).T
    H_explicit = sensor_model.lidar_jacobian(prior_state, body_angles)

    # --- IMPLICIT MODEL MATH ---
    H_implicit, D_imp, theta_imp = sensor_model.get_implicit_matrices(prior_state, meas_global)
    z_pred_imp_flat = sensor_model.h_from_theta(prior_state, theta_imp)
    z_pred_imp = z_pred_imp_flat.reshape(-1, 2).T

    print("--- Math Verification ---")
    print(f"Are the z_pred associations identical? {np.allclose(z_pred_exp, z_pred_imp)}")
    print(f"Are the Jacobians (H) identical?      {np.allclose(H_explicit, H_implicit)}")

    print("H_explicit shape:", H_explicit.shape)
    print("rank(H_explicit):", np.linalg.matrix_rank(H_explicit))
    print("H_implicit shape:", H_implicit.shape)
    print("rank(H_implicit):", np.linalg.matrix_rank(H_implicit))
    print("Ranks equal?", np.linalg.matrix_rank(H_explicit) == np.linalg.matrix_rank(H_implicit))

    # 6. Run Trackers (5 Iterations)
    dyn_model = Model_PCA_CV(
        x_pos_std_dev=tracker_cfg.pos_north_std_dev, y_pos_std_dev=tracker_cfg.pos_east_std_dev,
        yaw_std_dev=tracker_cfg.heading_std_dev, N_pca=N_pca
    )

    tracker_exp = IterativeEKF(dynamic_model=dyn_model, lidar_model=sensor_model, config=config, max_iterations=5)
    tracker_exp.state_estimate = prior_gauss
    res_exp = tracker_exp.update(meas_local)

    tracker_imp = ImplicitIEKF(dynamic_model=dyn_model, lidar_model=sensor_model, config=config, max_iterations=5)
    tracker_imp.state_estimate = prior_gauss
    res_imp = tracker_imp.update(meas_local)

    tracker_imp_ekf = ImplicitIEKF(dynamic_model=dyn_model, lidar_model=sensor_model, config=config, max_iterations=1)
    tracker_imp_ekf.state_estimate = prior_gauss
    res_imp_ekf = tracker_imp_ekf.update(meas_local)

    # 7. INTERACTIVE VISUALIZATION (Plotly)
    # Shape coordinations
    gt_x, gt_y = compute_exact_vessel_shape_global(gt_state, extent_base.shape_coords_body)
    pr_x, pr_y = compute_estimated_shape_global(prior_state, config, pca_params)
    exp_x, exp_y = compute_estimated_shape_global(res_exp.state_posterior.mean, config, pca_params)
    imp_x, imp_y = compute_estimated_shape_global(res_imp.state_posterior.mean, config, pca_params)
    imp_ekf_x, imp_ekf_y = compute_estimated_shape_global(res_imp_ekf.state_posterior.mean, config, pca_params)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # ================= PLOT: Identical Associations & Iteration Paths =================
    fig = make_subplots(rows=1, cols=2, subplot_titles=("1. Prior State & Association Rays", "2. Iteration Path & Final Updates"), 
                        horizontal_spacing=0.1)
    
    # Subplot 1
    fig.add_trace(go.Scatter(x=gt_y, y=gt_x, mode='lines', line=dict(color='black', dash='dash'), name='Ground Truth Shape'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[gt_state.y], y=[gt_state.x], mode='markers', marker=dict(color='black', symbol='diamond', size=8), name='GT Centroid'), row=1, col=1)
    fig.add_trace(go.Scatter(x=meas_global.y, y=meas_global.x, mode='markers', marker=dict(color='red', size=5), name='Actual Measurements (z)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pr_y, y=pr_x, mode='lines', line=dict(color='blue'), name='Offset Prior Shape'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[prior_state.y], y=[prior_state.x], mode='markers', marker=dict(color='blue', symbol='diamond', size=8), name='Prior Centroid'), row=1, col=1)
    fig.add_trace(go.Scatter(x=z_pred_exp[1, :], y=z_pred_exp[0, :], mode='markers', marker=dict(color='blue', symbol='x', size=6), name='z_pred (Both Models)'), row=1, col=1)

    # Interleave Ray Data so it plots efficiently as one trace
    ray_x, ray_y = [],[]
    for i in range(meas_global.shape[1]):
        ray_x.extend([meas_global.y[i], z_pred_exp[1, i], None])
        ray_y.extend([meas_global.x[i], z_pred_exp[0, i], None])
    
    fig.add_trace(go.Scatter(x=ray_x, y=ray_y, mode='lines', line=dict(color='gray', width=1, dash='dot'), opacity=0.5, name='Association Rays'), row=1, col=1)

    # Subplot 2
    # Faded background elements
    fig.add_trace(go.Scatter(x=gt_y, y=gt_x, mode='lines', line=dict(color='black', dash='dash'), opacity=0.4, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[gt_state.y], y=[gt_state.x], mode='markers', marker=dict(color='black', symbol='diamond', size=6), opacity=0.4, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=meas_global.y, y=meas_global.x, mode='markers', marker=dict(color='red', size=4), opacity=0.4, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=pr_y, y=pr_x, mode='lines', line=dict(color='blue'), opacity=0.3, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[prior_state.y], y=[prior_state.x], mode='markers', marker=dict(color='blue', symbol='diamond', size=6), opacity=0.3, showlegend=False), row=1, col=2)

    # Explicit Trace
    fig.add_trace(go.Scatter(x=exp_y, y=exp_x, mode='lines', line=dict(color='orange', width=2), name='Explicit IEKF (Updated)'), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[s.y for s in res_exp.iterates], y=[s.x for s in res_exp.iterates], 
        mode='lines+markers', marker=dict(color='orange', size=8), name='Exp Centroid Path'
    ), row=1, col=2)

    # Implicit Trace
    fig.add_trace(go.Scatter(x=imp_y, y=imp_x, mode='lines', line=dict(color='green', width=3), name='Implicit IEKF (Updated)'), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[s.y for s in res_imp.iterates], y=[s.x for s in res_imp.iterates], 
        mode='lines+markers', marker=dict(color='green', size=8, symbol='x'), name='Imp Centroid Path'
    ), row=1, col=2)

    # Implicit EKF Trace
    fig.add_trace(go.Scatter(x=imp_ekf_y, y=imp_ekf_x, mode='lines', line=dict(color='purple', width=2), name='Implicit EKF (Updated)'), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[s.y for s in res_imp_ekf.iterates], y=[s.x for s in res_imp_ekf.iterates], 
        mode='lines+markers', marker=dict(color='purple', size=8, symbol='triangle-up'), name='Imp EKF Centroid Path'
    ), row=1, col=2)

    fig.update_layout(
        title="Comparison of Explicit vs Implicit IEKF",
        plot_bgcolor='white',
        height=800,
        width=1600
    )
    
    fig.update_xaxes(title_text="East (y) [m]", row=1, col=1)
    fig.update_yaxes(title_text="North (x) [m]", scaleanchor="x", scaleratio=1, row=1, col=1)
    
    fig.update_xaxes(title_text="East (y) [m]", row=1, col=2)
    fig.update_yaxes(title_text="North (x) [m]", scaleanchor="x2", scaleratio=1, row=1, col=2)

    # Show the interactive plot (opens in one single tab/window)
    fig.show()

if __name__ == "__main__":
    run_comparison()