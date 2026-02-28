import numpy as np
import sys
from pathlib import Path

# Adjust path to find sibling modules if running as script
current_dir = Path(__file__).resolve().parent
src_root = current_dir.parent
if str(src_root) not in sys.path:
    sys.path.append(str(src_root))

from src.utils.tools import fourier_transform     # <--- Changed this line
from src.utils.ship_database import ShipDatabase  # <--- Changed this line

def get_gt_pca_coeffs_for_boat(boat_id: str, N_pca: int, pca_path: str = None) -> np.ndarray:
    """
    Calculates the Ground Truth PCA coefficients for a specific boat from the database.
    
    Args:
        boat_id: ID of the boat in processed_ships.json
        N_pca: Number of PCA components to return
        pca_path: Path to the NPZ file containing PCA parameters (mean, eigenvectors)
    
    Returns:
        coeffs: Numpy array of shape (N_pca,)
    """
    
    if pca_path is None:
        # Default path
        pca_path = src_root.parent / 'data' / 'input_parameters' / 'FourierPCAParameters_scaled.npz'
    
    pca_path = Path(pca_path)
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA parameters not found at {pca_path}")

    # Load PCA Model
    pca_data = np.load(pca_path)
    mean_fourier = pca_data['mean'].flatten()
    eigenvectors = pca_data['eigenvectors'] 

    # Load Boat Data
    db = ShipDatabase()
    boat = db.get_boat_by_id(str(boat_id))
    if not boat:
        raise ValueError(f"Boat ID {boat_id} not found")
    
    radii = np.array(boat['radii'])
    # The PCA model expects normalized radii (normalized by length usually, or whatever the training data was)
    # The JSON radii are already normalized by Original Length (max radius in JSON is ~0.5)
    # The PCA training (generate_pca_from_boats.py) typically uses these normalized radii directly.
    
    # Compute Fourier Transform
    # Standard angle grid for the radii is linearly spaced -pi to pi
    angles = np.linspace(-np.pi, np.pi, len(radii), endpoint=False)
    
    # Num coeffs must match the PCA model
    # Usually 64 coeffs (Check if encoded in NPZ or infer from mean vector size)
    # mean_fourier size = 64 (real) for 64 coefficients? 
    # If using rfft, size is N/2 + 1 complex, but usually we flatten real/imag parts.
    # From calculate_gt_coeffs_basicshapes.py: "num_coeff must match... typically 64"
    # And we assume symmetry=True for boats generally unless specified otherwise.
    
    # Infer num_coeff from mean vector size?
    # mean_fourier size = 64 if simple real coeffs?
    # Let's use 64 as default constant based on other files.
    
    vec = fourier_transform(angles, radii, num_coeff=64, symmetry=True)
    vec = vec.flatten()
    
    if vec.shape != mean_fourier.shape:
        # Try symmetry=False if shapes mismatch
        vec_asym = fourier_transform(angles, radii, num_coeff=64, symmetry=False).flatten()
        if vec_asym.shape == mean_fourier.shape:
            vec = vec_asym
        else:
            raise ValueError(f"Shape mismatch: PCA Mean {mean_fourier.shape}, Calculated Vec {vec.shape}")

    # Project onto Basis
    # c = M.T @ (x - mu) / s^2  (if M is scaled orthogonal)
    # Assuming standard eigenvectors from sklearn PCA (orthonormal), then c = M.T @ (x - mu)
    
    # CHECK SCALING:
    # In calculate_gt_coeffs_basicshapes.py, it does:
    # s = np.linalg.norm(eigenvectors[:, 0])
    # scaling_factor_sq = s**2
    # all_coeffs = (eigenvectors.T @ centered_vec) / scaling_factor_sq
    
    # If the eigenvectors are scaled by singular values (S), then M^T M = S^2.
    # The projection to get normalized coefficients (z ~ N(0,1)) is distinct from coefficients used for reconstruction (c = M z).
    # Wait, usually PCA coeffs are 'scores'. x = mu + scores @ components.
    # If 'eigenvectors' are the principal components (axes), then scores = (x - mu) @ components.T
    
    # Let's follow the logic in `calculate_gt_coeffs_basicshapes.py` exactly to be consistent.
    s = np.linalg.norm(eigenvectors[:, 0])
    scaling_factor_sq = s**2
    centered_vec = vec - mean_fourier
    all_coeffs = (eigenvectors.T @ centered_vec) / scaling_factor_sq
    
    return all_coeffs[:N_pca]

def get_boat_dimensions(boat_id: str):
    db = ShipDatabase()
    boat = db.get_boat_by_id(str(boat_id))
    if not boat:
        raise ValueError(f"Boat ID {boat_id} not found")
    
    L = boat['original_length_m']
    # Estimate Width from max radius in Y direction roughly
    # Or just use L/3 as placeholder if not in DB.
    # JSON has "radii". Max width is roughly 2 * max(y_projection(radii)).
    
    radii = np.array(boat['radii'])
    angles = np.linspace(-np.pi, np.pi, len(radii), endpoint=False)
    
    # y = r * sin(theta)
    ys = radii * np.sin(angles)
    W_norm = (np.max(ys) - np.min(ys)) # Normalized width
    W = W_norm * L
    
    return L, W
