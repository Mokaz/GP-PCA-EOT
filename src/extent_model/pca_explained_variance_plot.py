import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Load PCA eigenvalues ===
prior_filename_pca = Path("data/input_parameters/FourierPCAParameters_scaled.npz")
prior_data_pca = np.load(prior_filename_pca)
eigenvalues = prior_data_pca["eigenvalues"]

# === Compute explained variance percentages ===
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# === Plot individual and cumulative explained variance ===
plt.figure(figsize=(10, 6))
#plt.plot(range(1, 20), explained_variance_ratio * 100, marker='o', label='Individual')
plt.plot(range(1, 21), cumulative_variance[:20] * 100, marker='s', label='Cumulative')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.suptitle('Explained Variance by Principal Components', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/explained_variance_plot.svg", format="svg", dpi=1200)
plt.show()
