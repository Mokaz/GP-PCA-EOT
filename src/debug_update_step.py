import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.tracker.EKF import EKF
from src.tracker.IterativeEKF import IterativeEKF
from src.utils.tools import calculate_body_angles
from global_project_paths import SIMDATA_PATH
from src.utils.SimulationResult import SimulationResult
from copy import deepcopy

# 1. Load the IEKF simulation result (contains the problematic priors/measurements)
SIM_FILE = "iekf_initialize_centroid_True_seed_42.pkl"
FRAME_IDX = 90

# Check if file exists to avoid crashes if user doesn't have it
sim_file_path = Path(SIMDATA_PATH) / SIM_FILE
if not sim_file_path.exists():
    print(f"Error: {sim_file_path} does not exist. Please run the simulation first.")
    sys.exit(1)

with open(sim_file_path, "rb") as f:
    sim_result = pickle.load(f)

# 2. Extract Data for that Frame
res = sim_result.tracker_results_ts.values[FRAME_IDX]
measurements_global = sim_result.measurements_global_ts.values[FRAME_IDX]
ground_truth = sim_result.ground_truth_ts.values[FRAME_IDX]

# Reconstruct LidarScan in LOCAL frame (expected by update())
lidar_pos = np.array(sim_result.config.lidar.lidar_position)
measurements_local = measurements_global.copy()
measurements_local.x -= lidar_pos[0]
measurements_local.y -= lidar_pos[1]

# 3. Setup Trackers with IDENTICAL Config
# (Reuse the setup logic from main.py or CostLandscapeComponent)
from src.dynamics.process_models import Model_PCA_CV
from src.sensors.LidarModel import LidarMeasurementModel

config = sim_result.config
# Ensure PCA params are loaded
pca_params = np.load(config.tracker.PCA_parameters_path)

dyn_model = Model_PCA_CV(
    x_pos_std_dev=config.tracker.pos_north_std_dev,
    y_pos_std_dev=config.tracker.pos_east_std_dev,
    yaw_std_dev=config.tracker.heading_std_dev,
    N_pca=config.tracker.N_pca
)

sensor_model = LidarMeasurementModel(
    lidar_position=lidar_pos,
    lidar_std_dev=config.tracker.lidar_std_dev,
    pca_mean=pca_params['mean'],
    pca_eigenvectors=pca_params['eigenvectors'][:, :config.tracker.N_pca].real,
    extent_cfg=config.extent
)

ekf = EKF(dyn_model, sensor_model, config)
iekf = IterativeEKF(dyn_model, sensor_model, config)

# 4. FORCE both trackers to have the EXACT SAME PRIOR
prior = res.state_prior 
ekf.state_estimate = prior
iekf.state_estimate = prior

# 5. Run Updates
print(f"--- Frame {FRAME_IDX} Analysis ---")
print(f"Prior Yaw: {prior.mean.yaw:.4f}")

ekf_res = ekf.update(measurements_local)
iekf_res = iekf.update(measurements_local)

# 6. Compare
print(f"\nEKF Posterior Yaw: {ekf_res.state_posterior.mean.yaw:.4f}")
print(f"IEKF Posterior Yaw: {iekf_res.state_posterior.mean.yaw:.4f}")
print(f"Ground Truth Yaw:   {ground_truth.yaw:.4f}")

# Check Cost Function
# Calculate cost for EKF result
ekf_cost = iekf.objective_function(
    ekf_res.state_posterior.mean, 
    prior.mean, prior.cov, 
    measurements_global.flatten('F')
)
# Calculate cost for IEKF result
iekf_cost = iekf.objective_function(
    iekf_res.state_posterior.mean, 
    prior.mean, prior.cov, 
    measurements_global.flatten('F')
)

print(f"\nEKF Final Cost:  {ekf_cost:.2f}")
print(f"IEKF Final Cost: {iekf_cost:.2f}")

# 7. Export to PKL for Dashboard
from src.utils.SimulationResult import SimulationResult
from copy import deepcopy

def save_debug_result(update_result, filename_part, original_sim_result):
    # Shallow copy is fine for most, but we need to modify tracker_results_ts
    new_sim = deepcopy(original_sim_result) 
    
    # We only update the specific frame in the timesequence
    # Use direct access to .times because TimeSequence.__iter__ returns a list, conflicting with list() constructor
    timestamps = getattr(new_sim.tracker_results_ts, 'times', [])
    if not timestamps and hasattr(new_sim.tracker_results_ts, 'keys'):
         # Fallback if times attribute isn't there (though it should be)
         timestamps = list(new_sim.tracker_results_ts.keys())
         
    if FRAME_IDX < len(timestamps):
        ts = timestamps[FRAME_IDX]
        print(f"Update frame {FRAME_IDX} at timestamp {ts}")
        
        # Truncate before appending (keeps frames 0 to FRAME_IDX-1)
        new_sim.tracker_results_ts = new_sim.tracker_results_ts.slice_idx(stop=FRAME_IDX)
        new_sim.tracker_results_ts.insert(ts, update_result)
    else:
        print("Frame index out of bounds for existing timestamps?")

    # Update config name so we know
    new_sim.config.sim.name = f"{filename_part}"
    
    out_path = Path(SIMDATA_PATH) / f"{filename_part}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(new_sim, f)
    print(f"Saved {out_path}")

save_debug_result(ekf_res, "ekf_debug_frame_90", sim_result)
save_debug_result(iekf_res, "iekf_debug_frame_90", sim_result)
