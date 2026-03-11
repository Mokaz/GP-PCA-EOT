import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import json
import itertools
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Setup paths
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_ROOT))
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.tools import generate_fourier_function
from src.extent_model.boat_pca_utils import get_gt_pca_coeffs_for_boat

PCA_FILE = PROJECT_ROOT / "data" / "input_parameters" / "ShipDatasetPCAParameters.npz"
JSON_FILE = PROJECT_ROOT / "data" / "processed_ships.json"

def interactive_4d_pca_space(N_pca=4):
    print("Loading PCA Model and Dataset...")
    pca_data = np.load(PCA_FILE)
    pca_mean = pca_data['mean'].flatten()
    pca_eigenvectors = pca_data['eigenvectors'][:, :N_pca].real

    angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
    G = generate_fourier_function(N_f=64)(angles).T  # Shape: (180, 64)

    # Pre-compute transformation matrices for extreme speed
    M = G @ pca_eigenvectors  # Shape: (180, 4) - How each PC affects radius
    R_mean = G @ pca_mean     # Shape: (180,)   - Mean radius
    
    cos_ang = np.cos(angles)[:, None, None] # For fast 2D grid broadcasting
    sin_ang = np.sin(angles)[:, None, None]

    # --- 1. Evaluate True 4D Feasibility of Dataset ---
    with open(JSON_FILE, 'r') as f:
        boats_data = json.load(f)
        
    actual_coeffs = []
    actual_status =[] # 1: Safe, -1: Spill, -2: Neg R

    for boat in boats_data:
        if not boat.get('is_boat') or boat.get('is_kayak'): continue
        try:
            coeffs = get_gt_pca_coeffs_for_boat(boat['id'], N_pca=N_pca, pca_path=PCA_FILE)
        except Exception: continue
        
        # Calculate full 4D radius
        r_4d = R_mean + M @ coeffs
        x_4d = r_4d * np.cos(angles)
        y_4d = r_4d * np.sin(angles)
        
        if np.any(r_4d < 0):
            status = -2
        elif np.max(np.abs(x_4d)) > 0.505 or np.max(np.abs(y_4d)) > 0.505:
            status = -1
        else:
            status = 1
            
        actual_coeffs.append(coeffs)
        actual_status.append(status)

    actual_coeffs = np.array(actual_coeffs)
    actual_status = np.array(actual_status)

    safe_count = np.sum(actual_status == 1)
    spill_count = np.sum(actual_status == -1)
    neg_count = np.sum(actual_status == -2)
    
    print("\n=== TRUE 4D FEASIBILITY OF ACTUAL BOATS ===")
    print(f"Perfectly Safe (Inside 1x1): {safe_count}")
    print(f"Overshoots bounding box:     {spill_count}")
    print(f"Self-Intersecting (Neg R):   {neg_count}")
    print("===========================================\n")

    # --- 2. Setup Interactive Plot ---
    pairs = list(itertools.combinations(range(N_pca), 2))
    cols = 3
    rows = int(np.ceil(len(pairs) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 8))
    plt.subplots_adjust(bottom=0.25, hspace=0.3, wspace=0.3)
    axes = axes.flatten()
    
    cmap = ListedColormap(['#e74c3c', '#f39c12', '#2ecc71']) # Red, Orange, Green
    
    res = 100
    limit = 15
    grid_vals = np.linspace(-limit, limit, res)
    X_grid, Y_grid = np.meshgrid(grid_vals, grid_vals)
    
    images =[]
    
    # Initialize plots
    for idx, (d1, d2) in enumerate(pairs):
        ax = axes[idx]
        # We use imshow for blazing fast real-time updates
        img = ax.imshow(np.zeros((res, res)), origin='lower', extent=[-limit, limit, -limit, limit], 
                        cmap=cmap, vmin=-2, vmax=1, alpha=0.6)
        images.append(img)
        
        # Plot the actual dataset boats OVER the grid. 
        # We color them by their TRUE 4D status!
        ax.scatter(actual_coeffs[actual_status==1, d1], actual_coeffs[actual_status==1, d2], 
                   c='green', edgecolors='k', s=20, label='4D Safe')
        ax.scatter(actual_coeffs[actual_status==-1, d1], actual_coeffs[actual_status==-1, d2], 
                   c='orange', edgecolors='k', s=20, label='4D Spill')
        ax.scatter(actual_coeffs[actual_status==-2, d1], actual_coeffs[actual_status==-2, d2], 
                   c='red', edgecolors='k', s=20, label='4D Neg R')

        ax.set_title(f"PC{d1} vs PC{d2}")
        ax.set_xlabel(f"PC{d1}")
        ax.set_ylabel(f"PC{d2}")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

    for idx in range(len(pairs), len(axes)):
        axes[idx].axis('off')

    # Legend
    green_patch = mpatches.Patch(color='#2ecc71', alpha=0.6, label='Cross-Section: Safe')
    orange_patch = mpatches.Patch(color='#f39c12', alpha=0.6, label='Cross-Section: Spill')
    red_patch = mpatches.Patch(color='#e74c3c', alpha=0.6, label='Cross-Section: Neg R')
    fig.legend(handles=[green_patch, orange_patch, red_patch], loc='upper right', title="Grid Background")

    # --- 3. Create Sliders ---
    slider_axes = []
    sliders =[]
    slider_height = 0.03
    for i in range(N_pca):
        ax_sl = plt.axes([0.15, 0.05 + i*0.04, 0.65, slider_height])
        sl = Slider(ax_sl, f'Set PC{i}', -limit, limit, valinit=0.0)
        slider_axes.append(ax_sl)
        sliders.append(sl)

    def update(val):
        """Called every time a slider moves."""
        e_vals = [s.val for s in sliders]
        
        for idx, (d1, d2) in enumerate(pairs):
            # Calculate the fixed contribution from the dimensions NOT on the X/Y axes of this specific plot
            R_fixed = R_mean.copy()
            for d in range(N_pca):
                if d != d1 and d != d2:
                    R_fixed += M[:, d] * e_vals[d]
            
            # Broadcast the grid! Super fast array math.
            # R shape: (180, 100, 100)
            R = R_fixed[:, None, None] + M[:, d1, None, None] * X_grid + M[:, d2, None, None] * Y_grid
            
            X = R * cos_ang
            Y = R * sin_ang
            
            # Evaluate constraints
            feasibility = np.ones((res, res))
            feasibility[np.any(np.abs(X) > 0.505, axis=0) | np.any(np.abs(Y) > 0.505, axis=0)] = -1
            feasibility[np.any(R < 0, axis=0)] = -2
            
            # Update the image data instantly
            images[idx].set_data(feasibility)
            
        fig.canvas.draw_idle()

    # Link sliders to update function
    for sl in sliders:
        sl.on_changed(update)

    # Initial draw
    update(0)
    plt.suptitle("Interactive 4D PCA Convex Space Slicer", fontsize=16)
    plt.show()

if __name__ == "__main__":
    interactive_4d_pca_space(N_pca=4)