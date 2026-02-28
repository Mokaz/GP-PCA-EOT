import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- SETUP PROJECT PATHS ---
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_ROOT))
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.tools import fourier_transform

# --- CONFIGURATION ---
INPUT_JSON = "./data/processed_ships.json"
OUTPUT_FILENAME = "ShipDatasetPCAParameters.npz"
OUTPUT_DIR = "data/input_parameters"

# Filtering Toggles
INCLUDE_BOATS = True
INCLUDE_KAYAKS = False

# Fourier Parameters
NUM_COEFF = 64
SYMMETRY = True

def process_boats_to_pca():
    # 1. Load Data
    if not os.path.exists(INPUT_JSON):
        print(f"Error: Input file {INPUT_JSON} not found.")
        return

    with open(INPUT_JSON, 'r') as f:
        boats_data = json.load(f)

    print(f"Loaded {len(boats_data)} entries from {INPUT_JSON}")

    # 2. Extract and Transform Vectors
    extent_vector_list = []
    skipped_count = 0
    used_count = 0

    desc = "Processing boats"
    for entry in tqdm(boats_data, desc=desc):
        # -- Filtering Logic --
        is_boat = entry.get('is_boat', 0) == 1
        is_kayak = entry.get('is_kayak', 0) == 1
        
        # Determine if we keep this entry based on toggles
        keep = False
        if is_boat and INCLUDE_BOATS:
            keep = True
        if is_kayak and INCLUDE_KAYAKS:
            keep = True
        
        # If both flags are effectively mutually exclusive in your data, the above works.
        # If an entry is both boat and kayak, it stays. 
        # If filtering out specifically, we verify here:
        if is_kayak and not INCLUDE_KAYAKS:
            keep = False
        if is_boat and not INCLUDE_BOATS:
            keep = False
            
        if not keep:
            skipped_count += 1
            continue

        # -- Data Extraction --
        radii = np.array(entry['radii'])
        
        if len(radii) == 0:
            print(f"Warning: Boat ID {entry.get('id')} has empty radii.")
            skipped_count += 1
            continue

        # -- Angle Generation --
        # Assuming radii are sampled evenly from -pi to pi (standard for the STL processing)
        # matching the endpoint=False convention of the processing script
        angles = np.linspace(-np.pi, np.pi, len(radii), endpoint=False)

        # -- Fourier Transform --
        # This converts the spatial radii into the frequency domain (shape coefficients)
        try:
            extent_fourier_vec = fourier_transform(
                angles, 
                radii, 
                num_coeff=NUM_COEFF, 
                symmetry=SYMMETRY
            )
            extent_vector_list.append(extent_fourier_vec)
            used_count += 1
        except Exception as e:
            print(f"Error processing boat ID {entry.get('id')}: {e}")
            skipped_count += 1

    print(f"\nProcessing complete.")
    print(f"Used: {used_count} | Skipped: {skipped_count}")

    if used_count < 2:
        print("Error: Not enough data points to compute PCA.")
        return

    # 3. Prepare Matrices
    # Stack vectors into matrix of shape (num_features, num_samples)
    # Each column is one boat
    extent_vectors = np.hstack(extent_vector_list) 
    
    # 4. Compute PCA (Logic from generate_extent_2D.py)
    print("Computing PCA...")
    
    # Calculate Mean
    extent_vector_mean = np.mean(extent_vectors, axis=1).reshape((-1, 1))
    
    # Calculate Covariance
    extent_vector_cov = np.cov(extent_vectors)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(extent_vector_cov)

    # Ensure real numbers
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Sort eigenvalues and eigenvectors (descending)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Scale eigenvalues for better conditioned covariance matrix (legacy logic)
    # This logic matches generate_extent_2D.py exactly
    scale_factor = np.sqrt(np.max(np.abs(eigenvalues)))
    
    # Avoid division by zero if all eigenvalues are 0 (unlikely)
    if scale_factor > 0:
        eigenvalues_scaled = eigenvalues / scale_factor ** 2
        eigenvectors_scaled = eigenvectors * scale_factor
    else:
        eigenvalues_scaled = eigenvalues
        eigenvectors_scaled = eigenvectors

    # Verify Variance Preservation
    scaled_variance = eigenvectors_scaled @ np.diag(eigenvalues_scaled) @ eigenvectors_scaled.T
    variance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Check consistency
    is_close = np.allclose(variance, scaled_variance, atol=1e-9, rtol=1e-12)
    print(f"Variance preserved after scaling: {is_close}")

    # 5. Save Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    np.savez(
        output_path, 
        mean=extent_vector_mean, 
        eigenvectors=eigenvectors_scaled, 
        eigenvalues=eigenvalues_scaled,
        # Optional: Save metadata about the source
        # meta_num_samples=used_count,
        meta_scale_factor=scale_factor
    )

    print(f"\nSuccess! PCA model saved to:")
    print(f"-> {output_path}")
    
    # Print Explained Variance Ratio for the first few components
    total_var = np.sum(eigenvalues)
    print("\nExplained Variance by Top 5 Components:")
    for i in range(min(5, len(eigenvalues))):
        print(f"PC{i+1}: {eigenvalues[i]/total_var*100:.2f}%")

if __name__ == "__main__":
    process_boats_to_pca()