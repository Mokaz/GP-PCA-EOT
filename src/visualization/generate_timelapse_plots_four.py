import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import matplotlib.patches as patches

# Add project root to path
SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.global_project_paths import SIMDATA_PATH, FIGURES_PATH
from src.utils.geometry_utils import compute_estimated_shape_global, compute_exact_vessel_shape_global
from src.states.states import State_PCA, State_GP

def generate_timelapse_for_file(filename: Path):
    print(f"Processing {filename.name}...")
    
    try:
        with open(filename, "rb") as f:
            sim_result = pickle.load(f)
    except Exception as e:
        print(f"Failed to load {filename.name}: {e}")
        return

    config = sim_result.config
    
    # Load PCA params if needed
    pca_params = None
    if hasattr(config.tracker, 'PCA_parameters_path'):
        pca_path = Path(config.tracker.PCA_parameters_path)
        if pca_path.exists():
            pca_params = np.load(pca_path)
        # If path is relative, try resolving from project root
        elif (PROJECT_ROOT / config.tracker.PCA_parameters_path).exists():
             pca_params = np.load(PROJECT_ROOT / config.tracker.PCA_parameters_path)

    tracker_results = list(sim_result.tracker_results_ts.values)
    ground_truth_states = list(sim_result.ground_truth_ts.values)
    
    num_frames = len(tracker_results)
    if num_frames < 4:
        print(f"Skipping {filename.name}: Not enough frames ({num_frames})")
        return

    # Select 4 evenly spaced frames (always include first and last)
    # e.g., 0, 33, 66, 99 for 100 frames
    indices = np.linspace(0, num_frames - 1, 4, dtype=int)
    
    # Calculate global bounds for the selected frames to fix the camera
    all_x = []
    all_y = []
    
    # Include LiDAR position in bounds so it's always visible
    lidar_pos = config.lidar.lidar_position
    all_x.append(lidar_pos[0])
    all_y.append(lidar_pos[1])

    for idx in indices:
        s = ground_truth_states[idx]
        all_x.append(s.x)
        all_y.append(s.y)
    
    # Add some padding
    padding = 20
    min_x, max_x = min(all_x) - padding, max(all_x) + padding
    min_y, max_y = min(all_y) - padding, max(all_y) + padding
    
    # Calculate data aspect ratio to size the figure correctly
    # x is North (plotted on Y), y is East (plotted on X)
    data_height = max_x - min_x
    data_width = max_y - min_y
    
    # We have 4 subplots in a row
    # Aspect ratio of one subplot = data_height / data_width
    # Total figure aspect ratio (H/W) approx = data_height / (4 * data_width)
    
    fig_width = 20
    # Calculate height, adding some extra for titles/labels/legend (e.g. +2 inches)
    fig_height = (fig_width / 4) * (data_height / data_width) + 2
    
    # Setup Figure - switch back to tight_layout for manual control
    fig, axs = plt.subplots(1, 4, figsize=(fig_width, fig_height), sharey=True, sharex=True)
    
    # Global limits for consistent view (optional, but good for comparison)
    # Or we can let each frame auto-scale to focus on the ship
    # Let's auto-scale but keep aspect ratio equal
    
    for i, idx in enumerate(indices):
        ax = axs[i]
        
        # Data for this frame
        gt_state = ground_truth_states[idx]
        tracker_res = tracker_results[idx]
        est_state = tracker_res.state_posterior.mean
        
        # Handle potential None measurements
        if tracker_res.measurements is not None:
            z_lidar = tracker_res.measurements.reshape((-1, 2))
        else:
            z_lidar = np.empty((0, 2))
        
        # 1. Plot Path (History up to this frame)
        history_x = [s.x for s in ground_truth_states[:idx+1]]
        history_y = [s.y for s in ground_truth_states[:idx+1]]
        ax.plot(history_y, history_x, color='royalblue', linewidth=1, label='Path' if i == 0 else "", zorder=1)
        
        # 3. Plot GT Shape (Moved up to ensure correct layering logic, though zorder handles it)
        gt_shape_x, gt_shape_y = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)
        ax.plot(gt_shape_y, gt_shape_x, color='black', linewidth=1.5, label='GT Shape' if i == 0 else "", zorder=2)
        
        # 4. Plot Estimated Shape
        est_shape_x, est_shape_y = compute_estimated_shape_global(est_state, config, pca_params)
        ax.plot(est_shape_y, est_shape_x, color='green', linewidth=1.5, linestyle='-', label='Est Shape' if i == 0 else "", zorder=3)

        # 2. Plot LiDAR Rays (High zorder to be on top)
        # lidar_pos = config.lidar.lidar_position # Already defined above
        for z in z_lidar:
            dist = np.linalg.norm(z - np.array(lidar_pos))
            if dist < config.lidar.max_distance:
                ax.plot([lidar_pos[1], z[1]], [lidar_pos[0], z[0]], color='red', alpha=0.3, linewidth=0.5, zorder=4)
                ax.scatter(z[1], z[0], s=5, color='red', marker='.', zorder=5) # Draw measurement point
        # Add a dummy point for legend
        ax.plot([], [], color='red', label='LiDAR' if i == 0 else "", zorder=4)
        
        # 5. Plot Heading Arrow
        arrow_len = 5.0
        ax.arrow(est_state.y, est_state.x, 
                 arrow_len * np.sin(est_state.yaw), arrow_len * np.cos(est_state.yaw),
                 head_width=1.5, head_length=1.5, fc='purple', ec='purple', label='Heading' if i == 0 else "", zorder=6)

        # Formatting
        ax.set_title(f"Frame {idx}")
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Zoom in on the ship for detail? Or show full path?
        # Usually for "timelapse" showing the context (path) is good, but if the path is huge, the ship is tiny.
        # Let's center on the ship with a fixed window size
        # window_size = 40 # meters
        # ax.set_xlim(est_state.y - window_size, est_state.y + window_size)
        # ax.set_ylim(est_state.x - window_size, est_state.x + window_size)
        
        # Set fixed global limits
        ax.set_xlim(min_y, max_y)
        ax.set_ylim(min_x, max_x)
        
        if i == 0:
            ax.set_ylabel("North [m]")
        ax.set_xlabel("East [m]")

    # Add single legend
    handles, labels = axs[0].get_legend_handles_labels()
    # Filter duplicates just in case
    by_label = dict(zip(labels, handles))
    
    # Place legend at the very top, centered
    # Using bbox_to_anchor with loc='lower center' to place the bottom of the legend 
    # just above the plot area.
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.92), frameon=False)

    # Adjust layout to leave room for the legend at the top
    # rect=[left, bottom, right, top]
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_dir = FIGURES_PATH / "timelapses"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{filename.stem}_timelapse.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved to {output_path}")

def main():
    files = sorted(list(SIMDATA_PATH.glob("*.pkl")))
    if not files:
        print("No .pkl files found in results folder.")
        return

    for f in files:
        generate_timelapse_for_file(f)

if __name__ == "__main__":
    main()