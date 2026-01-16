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

def plot_consistency_on_ax(ax, analysis, metric_name, field='all', title_prefix=''):
    if metric_name == 'NIS':
        data = analysis.get_nis(field)
    else:
        data = analysis.get_nees(field)

    times = data.mahal_dist_tseq.times
    values = data.mahal_dist_tseq.values
    timesteps = np.arange(len(values))

    aconf = data.aconf
    insym = '$\\in$' if aconf[0] < data.a < aconf[1] else '$\\notin$'
    lab = f"Avg={data.a:.3f} {insym} CI({aconf[0]:.3f}, {aconf[1]:.3f})"
    
    ax.plot(timesteps, values, label=lab, color="royalblue", linewidth=2)
    
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    dofs_sub = str(data.dofs[0]).translate(subscript_map)
    sym = rf"$\chi^2_{{{data.dofs[0]}}}$" 
    
    ci_label = (f"{sym}, {data.in_interval:.0%} $\\in$ CI")
    median_label = f"{sym}, {data.above_median:.0%} > median"
    
    lmu = data.low_med_upp_tseq.values_as_array()
    ax.plot(timesteps, lmu[:, 0], label=ci_label, color="darkorange", linestyle='--')
    ax.plot(timesteps, lmu[:, 2], color="darkorange", linestyle='--')
    ax.plot(timesteps, lmu[:, 1], label=median_label, color="green", linestyle='--')
    
    ax.set_ylabel(f"{metric_name} ({field})")
    ax.set_yscale('log')
    ax.legend(loc="lower right", fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"{title_prefix} {metric_name}")

def main():
    file_iekf = "casestudy_newest_iekf_43_tracker_lidarstd_0.05.pkl"
    file_bfgs = "casestudy_newest_bfgs_43_tracker_lidarstd_0.05.pkl"
    
    print(f"Loading {file_iekf}...")
    analysis_iekf = load_analysis(file_iekf)
    
    print(f"Loading {file_bfgs}...")
    analysis_bfgs = load_analysis(file_bfgs)
    
    if not analysis_iekf or not analysis_bfgs:
        return

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    
    # Row 1: NEES
    plot_consistency_on_ax(axs[0, 0], analysis_iekf, 'NEES', title_prefix='IEKF')
    plot_consistency_on_ax(axs[0, 1], analysis_bfgs, 'NEES', title_prefix='BFGS')
    
    # Row 2: NIS
    plot_consistency_on_ax(axs[1, 0], analysis_iekf, 'NIS', title_prefix='IEKF')
    plot_consistency_on_ax(axs[1, 1], analysis_bfgs, 'NIS', title_prefix='BFGS')
    
    axs[1, 0].set_xlabel('Timestep')
    axs[1, 1].set_xlabel('Timestep')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
