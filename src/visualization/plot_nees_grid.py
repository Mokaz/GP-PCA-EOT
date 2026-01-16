import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add project root to path to import modules
SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.global_project_paths import SIMDATA_PATH
from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result
from src.states.states import State_PCA
        
def plot_nees_grid():
    filename = "casestudy_newest_bfgs_43_tracker_lidarstd_0.05.pkl"
    filepath = Path(SIMDATA_PATH) / filename
    
    if not filepath.exists():
        print(f"Error: File {filepath} not found.")
        return

    print(f"Loading {filename}...")
    with open(filepath, "rb") as f:
        sim_result = pickle.load(f)

    # Create consistency analysis
    print("Analyzing consistency...")
    consistency_analyzer = create_consistency_analysis_from_sim_result(sim_result)
    
    first_state = sim_result.tracker_results_ts.values[0].state_posterior.mean
    
    labels = []
    if isinstance(first_state, State_PCA):
        # Kinematic states + Extent
        labels = ['x', 'y', 'yaw', 'vel_x', 'vel_y', 'yaw_rate', 'length', 'width']
        # PCA coefficients
        n_pca = len(first_state.pca_coeffs)
        labels.extend([f'pca_{i}' for i in range(n_pca)])
    else:
        # Fallback
        labels = [f'State {i}' for i in range(12)]

    if len(labels) != 12:
        print(f"Warning: Expected 12 states, but found {len(labels)} labels: {labels}")
        # Adjust if necessary, but proceed with what we have or truncate/extend
    
    # Setup plot
    fig, axs = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    axs = axs.flatten()
    
    print("Plotting...")
    for i in range(12):
        if i >= len(labels):
            break
            
        ax = axs[i]
        label = labels[i]
        
        query = None
        if label in ['x', 'y', 'yaw', 'vel_x', 'vel_y', 'yaw_rate', 'length', 'width']:
            query = label
        elif label.startswith('pca_'):
            query = i 
        else:
            query = i

        try:
            data = consistency_analyzer.get_nees(query)
            
            # Plot NEES value
            times = data.mahal_dist_tseq.times
            values = data.mahal_dist_tseq.values
            timesteps = np.arange(len(values))
            ax.plot(timesteps, values, label='NEES', color='royalblue', linewidth=1)
            
            # Plot CI bounds and Median
            lmu = data.low_med_upp_tseq.values_as_array()
            # lmu columns: 0=lower CI, 1=median, 2=upper CI
            
            ax.plot(timesteps, lmu[:, 2], color='darkorange', linestyle='--', label='95% CI')
            ax.plot(timesteps, lmu[:, 1], color='green', linestyle='--', label='Median')
            ax.plot(timesteps, lmu[:, 0], color='darkorange', linestyle='--')
            
            ax.set_title(f'BFGS NEES: {label}')
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="-", alpha=0.2)
            
            # Add stats to legend or title
            avg_nees = data.a
            in_ci = data.in_interval
            stats_text = f"Avg: {avg_nees:.2f}, In CI: {in_ci:.1%}"
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        except Exception as e:
            print(f"Could not plot state {i} ({label}): {e}")
            ax.text(0.5, 0.5, "Error", ha='center')

    # Layout adjustments
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    
    # Add global labels
    fig.text(0.5, 0.01, 'Timestep', ha='center', fontsize=12)
    fig.text(0.01, 0.5, 'NEES (Log Scale)', va='center', rotation='vertical', fontsize=12)
    
    plt.show()

if __name__ == "__main__":
    plot_nees_grid()
