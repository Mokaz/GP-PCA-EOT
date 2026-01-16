import pickle
from pathlib import Path
import sys
import numpy as np

# Add project root to path
SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.global_project_paths import SIMDATA_PATH

def load_sim_result(filename):
    path = Path(SIMDATA_PATH) / filename
    if not path.exists():
        print(f"Error: File {path} not found.", file=sys.stderr)
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def analyze_optimizer_success(filename, label):
    sim_result = load_sim_result(filename)
    if sim_result is None:
        return None

    total_updates = 0
    success_count = 0
    
    for result in sim_result.tracker_results_ts.values:
        if result.raw_optimizer_result is not None:
            total_updates += 1
            
            is_success = False
            if hasattr(result.raw_optimizer_result, 'success'):
                is_success = result.raw_optimizer_result.success
            elif isinstance(result.raw_optimizer_result, dict):
                is_success = result.raw_optimizer_result.get('success', False)
            
            if is_success:
                success_count += 1
    
    success_rate = (success_count / total_updates * 100) if total_updates > 0 else 0.0
    return {
        'label': label,
        'total': total_updates,
        'success': success_count,
        'rate': success_rate
    }

def generate_latex_table(stats_list):
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"Scenario & Total Updates & Successful Updates & Success Rate (\%) \\")
    print(r"\hline")
    
    for stats in stats_list:
        if stats:
            print(f"{stats['label']} & {stats['total']} & {stats['success']} & {stats['rate']:.2f} \\\\")
            
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Optimizer Success Rates for BFGS Tracker with varying Lidar Noise}")
    print(r"\label{tab:optimizer_success}")
    print(r"\end{table}")

def main():
    files = [
        ("casestudy_newest_bfgs_43_tracker_lidarstd_0.05.pkl", "Lidar Std: 0.05 (Original)"),
        ("casestudy_highlidarstd_bfgs_43_tracker_lidarstd_0.25.pkl", "Lidar Std: 0.25"),
        ("casestudy_highlidarstd_bfgs_43_tracker_lidarstd_0.5.pkl", "Lidar Std: 0.5"),
        ("casestudy_highlidarstd_bfgs_43_tracker_lidarstd_1.0.pkl", "Lidar Std: 1.0")
    ]
    
    stats_list = []
    for filename, label in files:
        stats = analyze_optimizer_success(filename, label)
        if stats:
            stats_list.append(stats)
            
    generate_latex_table(stats_list)

if __name__ == "__main__":
    main()
