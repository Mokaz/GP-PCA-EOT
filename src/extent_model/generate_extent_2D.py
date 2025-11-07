import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Initialize project and import modules
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from extent_model.extent import Extent
from utils.tools import fourier_transform
#from plotting import plot_correlation_matrix, plot_eigenvalues, plot_mean_and_principal_components, plot_pca_vs_fourier, plot_cosine_fourier_coefficients

# Configurations
normalize_width = True
add_title = True
save_figures = True
show_figures = True
generate_model = True

# Extent prior parameters
prior_filename = Path("data/input_parameters/extent_prior_fourier.npz")
prior_filename_pca = Path("data/input_parameters/FourierPCAParameters_scaled.npz")
d_angle = np.deg2rad(1.0)

Nparam = 6  # Number of PCA components to use

if generate_model:
    # Types and ranges for sampling
    l_types = ["box", "ellipse", "box_elliptic_sides", "box_parabolic_bow_and_stern", "elliptic_bow_and_stern"]
    num_samples = 20
    L_array = np.linspace(0.7, 1.3, num=num_samples)
    W_per_L_array = np.linspace(0.7, 1.3, num=num_samples)
    P_per_L_array = np.linspace(0.1, 0.4, num=num_samples)
    S_per_W_array = np.linspace(0.2, 0.8, num=num_samples)
    num_fourier_coeff = 64

    # Generate prior for Fourier coefficients
    extent_vector_list = []
    total = len(l_types) * len(L_array) * len(W_per_L_array) * len(P_per_L_array) * len(S_per_W_array)
    with tqdm(total=total, desc="Generating extent vectors") as pbar:
        for type_ in l_types:
            for L_prior in L_array:
                for W_per_L in W_per_L_array:
                    W_prior = W_per_L * L_prior
                    for P_per_L in P_per_L_array:
                        P_prior = P_per_L * L_prior
                        for S_per_W in S_per_W_array:
                            S_prior = S_per_W * W_prior
                            params = {"type": type_, "L": L_prior, "W": W_prior, "P": P_prior, "S": S_prior}
                            extent = Extent(params, d_angle)
                            extent_fourier_vec = fourier_transform(extent.angles, extent.norm_radii, num_coeff=64, symmetry=True)
                            extent_vector_list.append([extent_fourier_vec])
                            pbar.update(1)

    extent_vectors = np.squeeze(np.array(extent_vector_list)).transpose()
    extent_vector_mean = np.mean(extent_vectors, axis=1).reshape((-1, 1))
    extent_vector_cov = np.cov(extent_vectors)

    # Save mean and covariance
    np.savez(prior_filename, mean=extent_vector_mean, cov=extent_vector_cov)

    eigenvalues, eigenvectors = np.linalg.eig(extent_vector_cov)

    # Convert eigenvalues to real numbers and ensure eigenvectors are real
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Sort eigenvalues and eigenvectors
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Scale eigenvalues for better conditioned covariance matrix
    scale_factor = np.sqrt(np.max(np.abs(eigenvalues)))
    eigenvalues_scaled = eigenvalues / scale_factor ** 2
    eigenvectors_scaled = eigenvectors * scale_factor

    # Check that total variance is preserved
    scaled_variance = eigenvectors_scaled @ np.diag(eigenvalues_scaled) @ eigenvectors_scaled.T
    variance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    assert np.allclose(variance, scaled_variance, atol=1e-9, rtol=1e-12), "Total variance not preserved after scaling."

    np.savez(prior_filename_pca, mean=extent_vector_mean, eigenvectors=eigenvectors_scaled, eigenvalues=eigenvalues_scaled)
else:
    # Load prior
    prior_data = np.load(prior_filename)
    extent_vector_mean = prior_data["mean"]
    extent_vector_covm = prior_data["covm"]

    prior_data_pca = np.load(prior_filename_pca)
    eigenvectors = prior_data_pca["eigenvectors"]
    eigenvalues = prior_data_pca["eigenvalues"]

    num_fourier_coeff = extent_vector_mean.size

# Plotting
# if generate_model:
#     plot_correlation_matrix(extent_vectors_zero_mean, num_fourier_coeff, add_title, save_figures)
# plot_eigenvalues(eigenvalues, num_fourier_coeff, add_title, save_figures)
# plot_cosine_fourier_coefficients(eigenvectors, num_fourier_coeff, add_title, save_figures)
# plot_mean_and_principal_components(extent_vector_mean, eigenvectors, d_angle, add_title, save_figures, num_fourier_coeff)

# # PCA and Fourier comparison
# shape_params_true = {"type": "box_parabolic_bow_and_stern", "L": 1.0, "W": 0.8, "P": 0.15, "S": 0.6}
# extent_true = Extent(shape_params_true, d_angle)
# extent_true_vec = fourier_transform(extent_true.angles, extent_true.radii, num_coeff=num_fourier_coeff, symmetry=True)

# Sigma = eigenvectors[:, :Nparam]
# PCA_parameters = Sigma.T @ (extent_true_vec - extent_vector_mean)
# extent_est_pca_vec = extent_vector_mean + Sigma @ PCA_parameters

# print("PCAparameters: ", PCA_parameters)

# param_est_pca = {"type": "Fourier", "symmetry": True, "vector": extent_est_pca_vec}
# extent_est_pca = Extent(param_est_pca, d_angle)

# extent_est_fourier_vec = extent_true_vec[:Nparam]
# param_est_fourier = {"type": "Fourier", "symmetry": True, "vector": extent_est_fourier_vec}
# extent_est_fourier = Extent(param_est_fourier, d_angle)

# # plot_pca_vs_fourier(extent_true, extent_est_pca, extent_est_fourier, Nparam, add_title, save_figures)

# # Show figures if enabled
# if show_figures:
#     plt.show()
