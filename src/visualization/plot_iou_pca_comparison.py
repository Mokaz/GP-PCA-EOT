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
from src.states.states import State_PCA
from src.utils.tools import fourier_basis_matrix

# Define input parameters path manually if not in global_paths
INPUT_PARAMETERS_PATH = PROJECT_ROOT / "data" / "input_parameters"

def load_pca_parameters():
    try:
        data = np.load(INPUT_PARAMETERS_PATH / "FourierPCAParameters_scaled.npz")
        return data['mean'].flatten(), data['eigenvectors']
    except FileNotFoundError:
        print(f"Warning: PCA parameters not found at {INPUT_PARAMETERS_PATH}")
        return None, None

PCA_MEAN, PCA_EIGENVECTORS = load_pca_parameters()

def get_radius_at_angle_pca(theta, L, W, pca_coeffs, pca_mean, pca_eigenvectors, N_fourier=32):
    """
    Calculates the radius of the vessel at a given body angle theta for PCA model.
    """
    # Ensure inputs are flattened/correct shape
    pca_coeffs = np.asarray(pca_coeffs).flatten()
    
    # Normalize the angle
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    normalized_angle = np.arctan2(sin_t / W, cos_t / L)
    
    # Slice eigenvectors to match pca_coeffs dimension if needed
    # pca_coeffs might be smaller than total components available in eigenvectors
    if pca_eigenvectors.shape[1] > len(pca_coeffs):
        current_eigenvectors = pca_eigenvectors[:, :len(pca_coeffs)]
    else:
        current_eigenvectors = pca_eigenvectors

    fourier_coeffs_full = pca_mean + current_eigenvectors @ pca_coeffs
    
    # Use the dimension of the reconstructed coefficients for the basis matrix
    # This overrides the passed N_fourier if they differ, ensuring valid matrix multiplication
    N_reconstruct = len(fourier_coeffs_full)
    
    # Calculate radius in normalized space
    # fourier_basis_matrix returns (N_fourier, N_angles)
    g = fourier_basis_matrix(normalized_angle, N_reconstruct) 
    
    if np.isscalar(theta):
        radius_norm = np.dot(g.flatten(), fourier_coeffs_full)
        v_x = L * np.cos(normalized_angle) * radius_norm
        v_y = W * np.sin(normalized_angle) * radius_norm
        return np.sqrt(v_x**2 + v_y**2)
    else:
        # g.T is (N_angles, N_fourier), fourier_coeffs_full is (N_fourier,)
        radius_norm = g.T @ fourier_coeffs_full
        v_x = L * np.cos(normalized_angle) * radius_norm
        v_y = W * np.sin(normalized_angle) * radius_norm
        return np.sqrt(v_x**2 + v_y**2)

def load_sim_result(filename):
    path = Path(SIMDATA_PATH) / filename
    if not path.exists():
        print(f"Error: File {path} not found.")
        return None
    
    with open(path, "rb") as f:
        sim_result = pickle.load(f)
    return sim_result

def calculate_iou_radial(r_true_func, r_est_func, num_samples=360):
    """
    Calculates IoU using numerical integration over theta.
    """
    thetas = np.linspace(-np.pi, np.pi, num_samples, endpoint=False)
    
    r_true = r_true_func(thetas)
    r_est = r_est_func(thetas)
    
    # Ensure 1D arrays
    r_true = np.asarray(r_true).flatten()
    r_est = np.asarray(r_est).flatten()
        
    min_r_sq = np.minimum(r_true, r_est)**2
    max_r_sq = np.maximum(r_true, r_est)**2
    
    intersection_area = 0.5 * np.trapz(min_r_sq, thetas)
    union_area = 0.5 * np.trapz(max_r_sq, thetas)
    
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area

def get_radius_function(state, config=None):
    """Returns a function r(theta) for the given state."""
    if isinstance(state, State_PCA):
        if PCA_MEAN is None:
            raise ValueError("PCA parameters not loaded.")
        
        N_fourier = 32
        if config and hasattr(config, 'extent') and hasattr(config.extent, 'N_fourier'):
            N_fourier = config.extent.N_fourier
            
        return lambda theta: get_radius_at_angle_pca(theta, state.length, state.width, state.pca_coeffs, PCA_MEAN, PCA_EIGENVECTORS, N_fourier)
    else:
        raise ValueError(f"Expected State_PCA, got {type(state)}")

def calculate_iou_sequence(sim_result):
    ious = []
    times = []
    
    gt_ts = sim_result.ground_truth_ts
    est_ts = sim_result.tracker_results_ts
    
    for t, est_result in est_ts.items():
        if t not in gt_ts:
            continue
            
        gt_state = gt_ts.get_t(t)
        est_state = est_result.state_posterior.mean
        
        r_func_true = get_radius_function(gt_state, sim_result.config)
        r_func_est = get_radius_function(est_state, sim_result.config)
        
        iou = calculate_iou_radial(r_func_true, r_func_est)
        ious.append(iou)
        times.append(t)
        
    return times, ious

def main():
    files = [
        ("casestudy_newest_bfgs_43_tracker_lidarstd_0.05.pkl", "BFGS"),
        ("casestudy_newest_ekf_43_tracker_lidarstd_0.05.pkl", "EKF"),
        ("casestudy_newest_iekf_43_tracker_lidarstd_0.05.pkl", "IEKF")
    ]
    
    plt.figure(figsize=(10, 6))
    
    for filename, label in files:
        print(f"Processing {label} ({filename})...")
        sim_result = load_sim_result(filename)
        if sim_result:
            times, ious = calculate_iou_sequence(sim_result)
            avg_iou = np.mean(ious) if len(ious) > 0 else 0
            plt.plot(times, ious, label=f"{label} (Avg: {avg_iou:.3f})", linewidth=1.5)
            
    plt.xlabel("Timestep")
    plt.ylabel("Intersection over Union (IoU)")
    plt.title("Extent Estimation Accuracy (IoU) over Time")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
