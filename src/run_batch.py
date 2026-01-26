import itertools
import pickle
from pathlib import Path
import sys
import logging
from zlib import crc32

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

from src.global_project_paths import SIMDATA_PATH
from src.utils.config_classes import Config
from src.simulation import run_single_simulation
from src.main import get_common_configs, get_pca_tracker_config, get_gp_tracker_config


def set_nested_value(obj, path, value):
    """Sets a value deep in a nested object using a dot-notation string."""
    parts = path.split('.')
    last = parts.pop()
    for part in parts:
        obj = getattr(obj, part)
    setattr(obj, last, value)

if __name__ == "__main__":
    
    # --- 1. Define the Parameter Grid ---
    # Keys: dot-notation path to attribute (or "method" for the tracker type)
    # Values: List of options to sweep over
    param_grid = {
        "method": ["ekf", "iekf"],
        "tracker.use_initialize_centroid": [True, False],
        # "sim.seed": [42, 100],  # Uncomment to test robustness
        # "tracker.lidar_std_dev": [0.05, 0.1, 0.2] # Uncomment to sweep noise
    }

    # --- 2. Generate Combinations ---
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"--- Queued {len(combinations)} simulations ---")
    
    # Constants
    N_pca = 4
    N_gp = 20

    # --- 3. Run Loop ---
    for i, combination in enumerate(combinations):
        current_params = dict(zip(keys, combination))
        method_name = current_params.pop("method")
        
        # --- A. Reset Base Config ---
        sim_base, lidar_base, extent_base = get_common_configs(N_pca)

        if "gp" in method_name:
            tracker_cfg = get_gp_tracker_config(lidar_base.lidar_position, N_gp)
        else:
            tracker_cfg = get_pca_tracker_config(lidar_base.lidar_position, N_pca)

        config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_cfg, extent=extent_base)

        # --- B. Apply Parameters ---
        param_desc_parts = []
        for path, val in current_params.items():
            set_nested_value(config, path, val)
            
            short_key = path.split('.')[-1].replace("use_", "")
            param_desc_parts.append(f"{short_key}_{val}")

        # --- C. Naming & Execution ---
        param_suffix = ("_" + "_".join(param_desc_parts)) if param_desc_parts else ""
        
        # Standardize naming: method + params + seed
        config.sim.name = f"{method_name}{param_suffix}_seed_{config.sim.seed}"
        
        print(f"\n[{i+1}/{len(combinations)}] Running: {config.sim.name}")
        print(f"   > Params: {current_params}")
        
        filename = f"{config.sim.name}.pkl"
        pickle_path = Path(SIMDATA_PATH) / filename

        # Check if already exists? (Optional)
        # if pickle_path.exists():
        #     print(f"   > Skipping, file exists: {filename}")
        #     continue

        try:
            run_single_simulation(config=config, method=method_name)
            print(f"   > Success. Saved to {filename}")
        except Exception as e:
            logging.error(f"Simulation {config.sim.name} failed!", exc_info=True)
