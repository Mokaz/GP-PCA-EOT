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

def plot_error_grid():
    filename = "casestudy_bfgs.pkl"
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
            # For PCA components, we need to access them by index relative to the whole state vector
            # Or if get_x_err handles 'pca_coeffs' and we can slice it?
            # Actually, get_x_err handles integer indices into the full state vector.
            # We need to know the index of this pca component.
            # In State_PCA, pca_coeffs start after the 8 named fields.
            query = 8 + int(label.split('_')[1])
        else:
            query = i

        try:
            err_gauss_tseq = consistency_analyzer.get_x_err(query)
            
            # Extract mean
            raw_values = np.array([e.mean for e in err_gauss_tseq.values])
            
            # Determine if we are plotting a vector norm or scalar absolute error
            if raw_values.ndim > 1 and raw_values.shape[1] > 1:
                # Vector quantity: Plot Euclidean Norm
                values = np.linalg.norm(raw_values, axis=1)
                label_prefix = "||err||"
            else:
                # Scalar quantity: Plot Absolute Error
                values = np.abs(raw_values).flatten()
                label_prefix = "|err|"

            timesteps = np.arange(len(values))
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean(values**2))
            
            # Plot Error
            ax.plot(timesteps, values, label=f'{label_prefix} (RMSE={rmse:.2e})', color='royalblue', linewidth=1)
            
            # Zero line
            ax.axhline(0, color='black', linestyle=':', alpha=0.5)
            
            ax.set_title(f'BFGS Error: {label}')
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend(loc='upper right', fontsize=8)

        except Exception as e:
            print(f"Could not plot state {i} ({label}): {e}")
            ax.text(0.5, 0.5, "Error", ha='center')

    # Layout adjustments
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    # Add global labels
    fig.text(0.5, 0.01, 'Timestep', ha='center', fontsize=12)
    
    plt.show()

if __name__ == "__main__":
    plot_error_grid()
