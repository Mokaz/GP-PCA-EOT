import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.global_project_paths import SIMDATA_PATH
from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result

def load_analysis(filename):
    path = Path(SIMDATA_PATH) / filename
    if not path.exists():
        print(f"Error: File {path} not found.")
        return None
    
    with open(path, "rb") as f:
        sim_result = pickle.load(f)
    
    return create_consistency_analysis_from_sim_result(sim_result)

def plot_nees_on_ax(ax, analysis, title):
    data = analysis.get_nees('all')

    times = data.mahal_dist_tseq.times
    values = data.mahal_dist_tseq.values
    timesteps = np.arange(len(values))

    aconf = data.aconf
    insym = '$\\in$' if aconf[0] < data.a < aconf[1] else '$\\notin$'
    lab = f"Avg={data.a:.3f} {insym} CI({aconf[0]:.3f}, {aconf[1]:.3f})"
    
    ax.plot(timesteps, values, label=lab, color="royalblue", linewidth=1)
    
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    dofs_sub = str(data.dofs[0]).translate(subscript_map)
    sym = rf"$\chi^2_{{{data.dofs[0]}}}$" 
    
    ci_label = (f"{sym}, {data.in_interval:.0%} $\\in$ CI")
    median_label = f"{sym}, {data.above_median:.0%} > median"
    
    lmu = data.low_med_upp_tseq.values_as_array()
    ax.plot(timesteps, lmu[:, 0], label=ci_label, color="darkorange", linestyle='--')
    ax.plot(timesteps, lmu[:, 2], color="darkorange", linestyle='--')
    ax.plot(timesteps, lmu[:, 1], label=median_label, color="green", linestyle='--')
    
    ax.set_ylabel("NEES")
    ax.set_xlabel("Timestep")
    ax.set_yscale('log')
    ax.legend(loc="lower right", fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(title)

def main():
    # Files in order of increasing noise for better comparison
    files = [
        ("casestudy_newest_bfgs_43_tracker_lidarstd_0.05.pkl", "BFGS Lidar Std: 0.05 (Original)"),
        ("casestudy_highlidarstd_bfgs_43_tracker_lidarstd_0.25.pkl", "BFGS Lidar Std: 0.25"),
        ("casestudy_highlidarstd_bfgs_43_tracker_lidarstd_0.5.pkl", "BFGS Lidar Std: 0.5"),
        ("casestudy_highlidarstd_bfgs_43_tracker_lidarstd_1.0.pkl", "BFGS Lidar Std: 1.0")
    ]
    
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
    
    for i, (filename, title) in enumerate(files):
        print(f"Loading {filename}...")
        analysis = load_analysis(filename)
        if analysis:
            plot_nees_on_ax(axs[i], analysis, title)
        else:
            axs[i].text(0.5, 0.5, f"File not found:\n{filename}", ha='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
