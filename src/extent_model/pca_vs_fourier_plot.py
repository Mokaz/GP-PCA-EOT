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

from utils.tools import fourier_transform, generate_fourier_function, pol2cart
from extent_model.extent import Extent, PCAExtentModel

def plot_pca_vs_fourier(extent_true, N_pca, add_title, save_figures):
    plt.figure(figsize=(8, 6))

    angles = extent_true.angles
    
    body_coordinates = extent_true.cartesian
    plt.plot(body_coordinates[0, :], body_coordinates[1, :], linewidth=1.5, label="True")

    extent_pca_model = PCAExtentModel(extent_true, N_pca=N_pca)

    pca_extent_estimate_x, pca_extent_estimate_y = pol2cart(angles, g(angles).T @ (extent_pca_model.fourier_coeff_mean + extent_pca_model.M[:, :N_pca] @ extent_pca_model.pca_params.reshape(-1, 1)).flatten())

    plt.plot(pca_extent_estimate_x, pca_extent_estimate_y, linewidth=1.5, label="PCA Estimate")

    fourier_extent_estimate_x, fourier_extent_estimate_y = pol2cart(angles, g(angles).T[:, :N_pca] @ extent_pca_model.extent_fourier.flatten()[:N_pca])

    plt.plot(fourier_extent_estimate_x, fourier_extent_estimate_y, linewidth=1.5, label="Fourier Estimate")

    #plt.xlabel("x [m]")
    #plt.ylabel("y [m]")
    plt.axis("equal")
    plt.legend()
    if add_title:
        plt.title(f"Truncated Fourier vs PCA Fourier with {N_pca} parameters")
    if save_figures:
        plt.savefig(f"figures/truncated_vs_PCA_fourier_{type_}_Nparam_{N_pca}.svg")

    # plt.show()
    # plt.waitforbuttonpress()

g = generate_fourier_function(N_f=64)

# === Config ===
prior_filename = Path("data/input_parameters/extent_prior_fourier.npz")
prior_filename_pca = Path("data/input_parameters/FourierPCAParameters.npz")

# === Load PCA Data ===
prior_data = np.load(prior_filename)
extent_vector_mean = prior_data["mean"]
extent_vectors_covm = prior_data["cov"]

prior_data_pca = np.load(prior_filename_pca)
eigenvectors = prior_data_pca["eigenvectors"]
eigenvalues = prior_data_pca["eigenvalues"]

# === Shape generation params ===
l_types = ["box", "ellipsis", "box_elliptic_sides", "box_parabolic_bow_and_stern", "elliptic_bow_and_stern"]
L = 1.0
W = 1.0
P = 0.25
S = 0.5
d_angle = np.deg2rad(1.0)
num_fourier_coeff = 64
N_pca = 4

# === Prepare plot ===
#plt.figure(figsize=(10, 6))

for type_ in l_types:
    # Generate shape parameters
    params = {"type": type_, "L": L, "W": W, "P": P, "S": S}
    extent = Extent(params, d_angle)

    plot_pca_vs_fourier(extent, N_pca, add_title=True, save_figures=True)
