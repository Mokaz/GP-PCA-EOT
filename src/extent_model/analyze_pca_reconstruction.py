import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_ROOT))
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "input_parameters"
INPUT_JSON = "./data/processed_ships.json"
PCA_FILE = DATA_DIR / "ShipDatasetPCAParameters.npz"
PDF_FILENAME = "all_boats_reconstruction.pdf"

# Analysis Settings
NUM_COEFF_FFT = 64
SYMMETRY = True
MAX_COMPONENTS_PLOT = 50 

# Filtering (Must match training logic)
INCLUDE_BOATS = True
INCLUDE_KAYAKS = False 

# PDF Settings
BOATS_PER_PAGE = 6  # 2 rows x 3 columns
COMPONENTS_TO_VISUALIZE = [3, 10, 40]
COLORS_VISUALIZE = ['#e74c3c', '#3498db', '#2ecc71'] # Red, Blue, Green

# --- UTILITIES ---

def get_basis_orthonormal(eigenvectors):
    # Normalize column-wise to ensure unit length basis for projection
    norms = np.linalg.norm(eigenvectors, axis=0)
    norms[norms == 0] = 1.0
    return eigenvectors / norms

def fourier_transform(angles, func, num_coeff=64, symmetry=True):
    f_sample = 2 * num_coeff
    angles_fft = np.linspace(-np.pi, np.pi, f_sample, endpoint=False)
    r_fft = np.interp(angles_fft, angles, func, period=2 * np.pi)
    y = np.fft.rfft(r_fft) / angles_fft.size
    y *= 2
    a0 = y[0].real
    a = y[1:-1].real
    b = -y[1:-1].imag
    if symmetry:
        return np.concatenate((a0, a), axis=None).reshape((-1, 1))
    else:
        return np.concatenate((a0, a, b), axis=None).reshape((-1, 1))

def inverse_fourier_transform(coeffs, num_points=360, symmetry=True):
    coeffs = coeffs.flatten()
    a0 = coeffs[0]
    
    if symmetry:
        a = coeffs[1:]
        b = np.zeros_like(a)
    else:
        n_coeffs = (len(coeffs) - 1) // 2
        a = coeffs[1:n_coeffs+1]
        b = coeffs[n_coeffs+1:]

    complex_coeffs = [a0 / 2.0] # DC term scaled for IRFFT
    for ac, bc in zip(a, b):
        z = (ac - 1j * bc) / 2.0
        complex_coeffs.append(z)
        
    reconstructed_sig = np.fft.irfft(complex_coeffs)
    reconstructed_sig *= len(reconstructed_sig) # Scale back up
    
    current_angles = np.linspace(-np.pi, np.pi, len(reconstructed_sig), endpoint=False)
    target_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
    
    return np.interp(target_angles, current_angles, reconstructed_sig, period=2*np.pi)

# --- PDF GENERATION ---

def generate_pdf_report(valid_boats, mean_vec, basis, W_full):
    print(f"\nGenerating PDF report: {PDF_FILENAME}...")
    
    output_path = PDF_FILENAME
    num_boats = len(valid_boats)
    
    # Pre-compute reconstruction vectors for all boats at visualization levels
    # to avoid re-calculating inside the plot loop (though fast enough either way)
    
    with PdfPages(output_path) as pdf:
        # Create batches
        for i in tqdm(range(0, num_boats, BOATS_PER_PAGE), desc="Pages"):
            batch_indices = range(i, min(i + BOATS_PER_PAGE, num_boats))
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': 'polar'})
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            fig.suptitle(f"Boats {i+1} to {min(i+BOATS_PER_PAGE, num_boats)} of {num_boats}", fontsize=14)
            
            axes_flat = axes.flatten()
            
            for ax_idx, boat_idx in enumerate(batch_indices):
                ax = axes_flat[ax_idx]
                boat = valid_boats[boat_idx]
                
                # 1. Plot Truth
                angles_plot = np.linspace(-np.pi, np.pi, 360, endpoint=False)
                radii_true_raw = boat['radii_true']
                angles_raw = np.linspace(-np.pi, np.pi, len(radii_true_raw), endpoint=False)
                radii_true_interp = np.interp(angles_plot, angles_raw, radii_true_raw, period=2*np.pi)
                
                ax.plot(angles_plot, radii_true_interp, 'k-', linewidth=2, label='Truth', alpha=0.4)
                
                # 2. Plot Reconstructions
                for n_comp, color in zip(COMPONENTS_TO_VISUALIZE, COLORS_VISUALIZE):
                    # Reconstruct
                    weights = W_full[:n_comp, boat_idx].reshape(-1, 1)
                    vec_rec = mean_vec + basis[:, :n_comp] @ weights
                    radii_rec = inverse_fourier_transform(vec_rec, num_points=360, symmetry=SYMMETRY)
                    
                    ax.plot(angles_plot, radii_rec, color=color, linewidth=1.5, linestyle='--', label=f'N={n_comp}')
                
                # Styling
                boat_name = boat.get('name', 'Unknown')
                # calculate simple MSE for title
                vec_diff = boat['vector'] - (mean_vec + basis[:, :10] @ W_full[:10, boat_idx].reshape(-1, 1))
                mse_10 = np.mean(vec_diff**2)
                
                ax.set_title(f"ID: {boat['id']} | {boat_name[:15]}\nMSE(10): {mse_10:.1e}", fontsize=9)
                ax.set_yticklabels([])
                ax.grid(True, alpha=0.3)
                
                # Add legend only to the first subplot of the page
                if ax_idx == 0:
                    ax.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.15), fontsize='x-small', ncol=4)

            # Turn off unused axes
            for j in range(len(batch_indices), len(axes_flat)):
                axes_flat[j].axis('off')

            pdf.savefig(fig)
            plt.close(fig)
            
    print(f"PDF saved to {output_path}")

# --- MAIN ANALYSIS ---

def analyze_boats():
    # 1. Load PCA Model
    if not PCA_FILE.exists():
        print(f"Error: PCA file not found at {PCA_FILE}")
        return

    print(f"Loading PCA model from {PCA_FILE}...")
    model_data = np.load(PCA_FILE)
    mean_vec = model_data['mean']
    eigenvectors_scaled = model_data['eigenvectors']
    
    basis = get_basis_orthonormal(eigenvectors_scaled)
    
    # 2. Load and Filter Boat Data
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found")
        return

    with open(INPUT_JSON, 'r') as f:
        boats_data = json.load(f)

    valid_boats = []
    print("Filtering and processing boat data...")
    
    for entry in tqdm(boats_data, desc="Loading JSON"):
        is_boat = entry.get('is_boat', 0) == 1
        is_kayak = entry.get('is_kayak', 0) == 1
        
        if is_kayak and not INCLUDE_KAYAKS: continue
        if is_boat and not INCLUDE_BOATS: continue
        if len(entry['radii']) == 0: continue

        radii = np.array(entry['radii'])
        angles = np.linspace(-np.pi, np.pi, len(radii), endpoint=False)
        
        try:
            vec = fourier_transform(angles, radii, num_coeff=NUM_COEFF_FFT, symmetry=SYMMETRY)
            valid_boats.append({
                'id': entry['id'],
                'name': entry.get('name', 'Unknown'),
                'vector': vec,
                'radii_true': radii
            })
        except Exception:
            continue

    if not valid_boats:
        print("No valid boats found.")
        return

    print(f"Analyzing {len(valid_boats)} boats.")

    # 3. Calculate Global Weights
    X = np.hstack([b['vector'] for b in valid_boats])
    X_centered = X - mean_vec
    # W_full: Weights for every component for every boat
    W_full = basis.T @ X_centered

    # 4. Generate PDF Report (All Boats)
    generate_pdf_report(valid_boats, mean_vec, basis, W_full)

    # 5. Generate Aggregate Summary Plot (Single Image)
    print("\nGenerating Aggregate Analysis Image...")
    mse_history = []
    components_range = range(1, min(MAX_COMPONENTS_PLOT, basis.shape[1]) + 1)
    
    for k in components_range:
        X_rec = mean_vec + basis[:, :k] @ W_full[:k, :]
        mse = np.mean((X - X_rec)**2)
        mse_history.append(mse)

    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(wspace=0.3)
    
    # Cumulative Variance
    ax1 = fig.add_subplot(1, 2, 1)
    comp_vars = np.var(W_full, axis=1)
    total_var = np.sum(comp_vars)
    cumulative_var = np.cumsum(comp_vars / total_var)
    ax1.plot(cumulative_var[:MAX_COMPONENTS_PLOT], 'o-', markersize=4, color='#2c3e50')
    ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95%')
    ax1.set_title("Cumulative Variance")
    ax1.set_xlabel("Components")
    ax1.grid(True)
    ax1.legend()

    # Reconstruction Error
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(components_range, mse_history, 'o-', markersize=4, color='#e74c3c')
    ax2.set_title("Mean Squared Error")
    ax2.set_xlabel("Components")
    ax2.set_yscale('log')
    ax2.grid(True, which='both')

    summary_img = "pca_summary.png"
    plt.savefig(summary_img)
    print(f"Summary image saved to {summary_img}")
    # plt.show() # Uncomment if running interactively

if __name__ == "__main__":
    analyze_boats()