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
from src.states.states import State_PCA, State_GP
from src.utils.tools import fourier_basis_matrix, ur
from src.utils.GaussianProcess import GaussianProcess

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
    # Normalize the angle
    # theta can be scalar or array
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    normalized_angle = np.arctan2(sin_t / W, cos_t / L)
    
    # Ensure dimensions match N_fourier
    # Slice mean and eigenvectors to N_fourier rows
    if pca_mean.shape[0] > N_fourier:
        pca_mean_sliced = pca_mean[:N_fourier]
        pca_eigenvectors_sliced = pca_eigenvectors[:N_fourier, :]
    else:
        pca_mean_sliced = pca_mean
        pca_eigenvectors_sliced = pca_eigenvectors

    # Slice eigenvectors columns to match pca_coeffs dimension
    if pca_eigenvectors_sliced.shape[1] > len(pca_coeffs):
        pca_eigenvectors_sliced = pca_eigenvectors_sliced[:, :len(pca_coeffs)]

    # Reconstruct Fourier coefficients
    # pca_coeffs shape: (N_pca,)
    # pca_mean_sliced shape: (N_fourier,)
    # pca_eigenvectors_sliced shape: (N_fourier, N_pca)
    
    fourier_coeffs_full = pca_mean_sliced + pca_eigenvectors_sliced @ pca_coeffs
    
    # Calculate radius in normalized space
    # fourier_basis_matrix returns (N_fourier, N_angles)
    g = fourier_basis_matrix(normalized_angle, N_fourier) 
    
    # If theta is scalar, g is (N_fourier, 1). If array, (N_fourier, N)
    # fourier_coeffs_full is (N_fourier,)
    
    if np.isscalar(theta):
        radius_norm = np.dot(g.flatten(), fourier_coeffs_full)
        # Body frame vector
        v_x = L * np.cos(normalized_angle) * radius_norm
        v_y = W * np.sin(normalized_angle) * radius_norm
        return np.sqrt(v_x**2 + v_y**2)
    else:
        radius_norm = np.dot(g.T, fourier_coeffs_full) # (N,)
        v_x = L * np.cos(normalized_angle) * radius_norm
        v_y = W * np.sin(normalized_angle) * radius_norm
        return np.sqrt(v_x**2 + v_y**2)

def get_radius_at_angle_gp(theta, radii, gp_utils):
    """
    Calculates the radius of the vessel at a given body angle theta for GP model.
    """
    # theta can be scalar or array
    # compute_k_vector handles array input
    k_vector = gp_utils.compute_k_vector(theta) # (N_theta, N_gp)
    
    # Interpolation weights
    weights = k_vector @ gp_utils.Ktt_inv # (N_theta, N_gp)
    
    # Predicted radius
    r_pred = weights @ radii
    
    return r_pred

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
    IoU = Integral(min(r_t, r_e)^2) / Integral(max(r_t, r_e)^2)
    """
    thetas = np.linspace(-np.pi, np.pi, num_samples, endpoint=False)
    
    # Vectorized evaluation if functions support it, otherwise loop
    try:
        r_true = r_true_func(thetas)
        r_est = r_est_func(thetas)
    except Exception as e:
        # print(f"Vectorized eval failed: {e}")
        r_true = np.array([r_true_func(t) for t in thetas])
        r_est = np.array([r_est_func(t) for t in thetas])
        
    min_r_sq = np.minimum(r_true, r_est)**2
    max_r_sq = np.maximum(r_true, r_est)**2
    
    intersection_area = 0.5 * np.trapz(min_r_sq, thetas) # 0.5 * r^2 dtheta
    union_area = 0.5 * np.trapz(max_r_sq, thetas)
    
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area

def get_radius_function(state, config=None, gp_utils=None):
    """Returns a function r(theta) for the given state."""
    if isinstance(state, State_PCA):
        if PCA_MEAN is None:
            raise ValueError("PCA parameters not loaded.")
        
        # N_fourier should be in config, but default to 32 if not found or if config is None
        N_fourier = 32
        if config and hasattr(config, 'extent') and hasattr(config.extent, 'N_fourier'):
            N_fourier = config.extent.N_fourier
            
        return lambda theta: get_radius_at_angle_pca(theta, state.length, state.width, state.pca_coeffs, PCA_MEAN, PCA_EIGENVECTORS, N_fourier)
        
    elif isinstance(state, State_GP):
        if gp_utils is None:
            # Try to reconstruct gp_utils from config if provided
            if config and hasattr(config, 'tracker'):
                # We need N_gp, length_scale, etc.
                # Assuming config.tracker has these
                # But State_GP has radii size which is N_gp
                N_gp = len(state.radii)
                
                # Default values if not in config (fallback)
                ls = getattr(config.tracker, 'gp_length_scale', np.pi/6)
                ff = getattr(config.tracker, 'gp_forgetting_factor', 0.01)
                sv = getattr(config.tracker, 'gp_signal_var', 1.0)
                
                gp_utils = GaussianProcess(N_gp, ls, ff, sv)
            else:
                raise ValueError("GP Utils or Config required for GP state radius calculation.")
                
        return lambda theta: get_radius_at_angle_gp(theta, state.radii, gp_utils)
    
    else:
        raise ValueError(f"Unknown state type: {type(state)}")

def calculate_iou_sequence(sim_result):
    ious = []
    times = []
    
    gt_ts = sim_result.ground_truth_ts
    est_ts = sim_result.tracker_results_ts
    
    # Prepare GP utils if needed
    gp_utils = None
    if hasattr(sim_result.config.tracker, 'gp_length_scale'):
         # It might be a GP tracker
         # But we don't know N_gp easily without checking the state or config
         # Let's do it lazily in the loop or check the first state
         pass

    # Iterate through frames where we have both GT and Estimate
    for t, est_result in est_ts.items():
        if t not in gt_ts:
            continue
            
        gt_state = gt_ts.get_t(t)
        est_state = est_result.state_posterior.mean
        
        # Define radial functions in BODY FRAME (Shape accuracy)
        r_func_true = get_radius_function(gt_state, sim_result.config)
        r_func_est = get_radius_function(est_state, sim_result.config)
        
        iou = calculate_iou_radial(r_func_true, r_func_est)
        ious.append(iou)
        times.append(t)
        
    return times, ious

def main():
    # Find all simulation files matching the pattern
    import glob
    import os
    
    pattern = "study_gp_iekf_*.pkl"
    files = sorted(glob.glob(str(SIMDATA_PATH / pattern)))
    
    if not files:
        print(f"No files found matching {pattern} in {SIMDATA_PATH}")
        return

    plt.figure(figsize=(12, 8))
    
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")
        sim_result = load_sim_result(filename)
        if sim_result:
            times, ious = calculate_iou_sequence(sim_result)
            
            # Calculate average IoU
            avg_iou = np.mean(ious) if len(ious) > 0 else 0
            
            # Parse parameters from filename for label
            label = filename.replace("study_gp_iekf_43_", "").replace(".pkl", "")
            
            plt.plot(times, ious, label=f"{label} (Avg: {avg_iou:.3f})", linewidth=1.5)
            
    plt.xlabel("Timestep")
    plt.ylabel("Intersection over Union (IoU)")
    plt.title("Extent Estimation Accuracy (IoU) over Time - GP Parameter Sweep")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right", fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

