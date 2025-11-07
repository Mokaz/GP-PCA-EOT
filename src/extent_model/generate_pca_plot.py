import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Initialize project and import modules
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from utils.tools import fourier_transform
from extent_model.extent import Extent

# === Config ===
prior_filename = Path("data/input_parameters/extent_prior_fourier.npz")
prior_filename_pca = Path("data/input_parameters/FourierPCAParameters_scaled.npz")

# === Load PCA Data ===
prior_data = np.load(prior_filename)
extent_vector_mean = prior_data["mean"]
extent_vectors_covm = prior_data["covm"]

prior_data_pca = np.load(prior_filename_pca)
eigenvectors = prior_data_pca["eigenvectors"]
eigenvalues = prior_data_pca["eigenvalues"]

# === Shape generation params ===
l_types = ["box", "ellipse", "box_elliptic_sides", "box_parabolic_bow_and_stern", "elliptic_bow_and_stern"]
L = 1.0
W = 1.0
P = 0.25
S = 0.5
d_angle = np.deg2rad(1.0)
num_fourier_coeff = 64

# === Prepare plot ===
plt.figure(figsize=(10, 6))

for type_ in l_types:
    # Generate shape parameters
    params = {"type": type_, "L": L, "W": W, "P": P, "S": S}
    extent = Extent(params, d_angle)

    # Transform shape to Fourier vector
    extent_fourier = fourier_transform(extent.angles, extent.norm_radii, num_coeff=num_fourier_coeff, symmetry=True)
    x = extent_fourier.reshape(-1, 1)  # shape (D, 1)
    x_zero_mean = x - extent_vector_mean

    # Compute reconstruction error vs number of components
    reconstruction_errors = []
    for k in range(1, 15):
        V_k = eigenvectors[:, :k]
        x_proj = V_k.T @ x_zero_mean
        x_recon = V_k @ x_proj + extent_vector_mean
        mse = np.mean((x - x_recon) ** 2)
        reconstruction_errors.append(mse)

    plt.plot(range(1, len(reconstruction_errors)+1), reconstruction_errors, marker='o', label=type_)

# === Configure plot ===
title = "PCA Reconstruction Error by Shape Type"
subtitle = f"W={W}, L={L}, P={P}, S={S}"
save_figures = True
show_figures = True

# === Finalize plot ===
plt.suptitle(title, fontsize=16)
plt.title(subtitle)
plt.xlabel('Number of Principal Components')
plt.ylabel('Mean Squared Reconstruction Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
if save_figures:
    plt.savefig("figures/pca_reconstruction_error.svg", format="svg", dpi=1200)
if show_figures:
    plt.show()

