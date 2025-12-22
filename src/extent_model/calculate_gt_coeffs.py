import numpy as np
from pathlib import Path
import sys
import os

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_ROOT))
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.extent_model.extent import Extent
from src.utils.tools import fourier_transform

def calculate_gt_pca_coeffs():
    # 1. Load PCA Parameters
    pca_path = Path('data/input_parameters/FourierPCAParameters_scaled.npz')
    if not pca_path.exists():
        print(f"Error: Could not find {pca_path}")
        return

    pca_data = np.load(pca_path)
    mean_fourier = pca_data['mean'].flatten()
    eigenvectors = pca_data['eigenvectors'] # These are scaled eigenvectors (M)

    # 2. Define Ground Truth Shape (Same as in main_casestudy.py)
    L_gt = 20.0
    W_gt = 6.0
    d_angle = np.deg2rad(1.0)
    
    shape_params = {
        "type": "ellipse", 
        "L": L_gt, 
        "W": W_gt, 
        # P and S are not used for ellipse but kept for consistency
        "P": L_gt * 0.2, 
        "S": L_gt * 0.1  
    }

    print(f"Generating GT Shape: {shape_params}")

    # 3. Generate Extent and Fourier Coefficients
    # The PCA model is trained on 'norm_radii' (normalized by L and W)
    extent = Extent(shape_params, d_angle)
    
    # Compute Fourier coefficients of the normalized shape
    # num_coeff must match what was used during PCA generation (usually 64)
    gt_fourier_vec = fourier_transform(extent.angles, extent.norm_radii, num_coeff=64, symmetry=True)
    gt_fourier_vec = gt_fourier_vec.flatten()

    # 4. Project onto PCA Basis
    # The reconstruction formula used in geometry_utils is: x = mu + M @ c
    # Since M is orthogonal but scaled (M^T @ M = s^2 * I), the projection is:
    # c = (1/s^2) * M^T @ (x - mu)
    
    # Calculate scaling factor s from the first eigenvector
    s = np.linalg.norm(eigenvectors[:, 0])
    scaling_factor_sq = s**2

    # Center the data
    centered_vec = gt_fourier_vec - mean_fourier

    # Project
    # We calculate all coefficients, then you can pick the top N_pca
    all_coeffs = (eigenvectors.T @ centered_vec) / scaling_factor_sq

    print("\n--- Calculated PCA Coefficients ---")
    print(f"Scaling Factor (s): {s:.4f}")
    print(f"First 4 Coeffs: {all_coeffs[:4]}")
    print(f"First 8 Coeffs: {all_coeffs[:8]}")
    
    print(f"pca_coeffs=np.array({list(all_coeffs[:4])})")

if __name__ == "__main__":
    calculate_gt_pca_coeffs()