import os
import json
import math
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "./data/stl_ships"
CSV_FILE = os.path.join(DATA_DIR, "source.csv")
OUTPUT_JSON = "./data/processed_ships.json"
OUTPUT_IMAGE = "./data/ship_inspection.pdf"
OUTPUT_IMAGE_POLAR = "./data/ship_inspection_polar.pdf"
OUTPUT_IMAGE_POLAR_PROJ = "./data/ship_inspection_polar_proj.pdf"
DEBUG_DIR = "./data/debug_ships"
NUM_RAYS = 360
DOWNSAMPLE = 1
ROTATED_SHIP_IDS = ["103", "143", "144", "145", "149", "82"]

if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

def safe_int(value):
    """Safely converts string to int, returning 0 if empty or invalid."""
    try:
        if pd.isna(value) or str(value).strip() == '':
            return 0
        return int(float(value))
    except:
        return 0

def load_metadata(filepath):
    """
    Robust metadata loader.
    """
    if not os.path.exists(filepath):
        print(f"ERROR: CSV file not found at {filepath}")
        return {}

    try:
        df = pd.read_csv(filepath, sep=';', dtype=str, encoding='utf-8-sig', on_bad_lines='skip')
        
        df.columns = df.columns.str.strip()
        
        if 'BoatID' not in df.columns:
            print("CRITICAL ERROR: 'BoatID' column not found in CSV.")
            return {}

        df['BoatID'] = df['BoatID'].str.strip()
        df = df[df['BoatID'].notna() & (df['BoatID'] != '')]
        df = df.drop_duplicates(subset=['BoatID'], keep='first')
        
        meta_dict = df.set_index('BoatID').to_dict('index')
        print(f"Loaded {len(meta_dict)} metadata entries.")
        return meta_dict
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}

def get_ray_intersection(poly_points, angle):
    """
    Finds the intersection of a ray from (0,0) at 'angle' 
    with the polygon defined by 'poly_points'.
    """
    # Ray direction
    rd = np.array([np.cos(angle), np.sin(angle)])
    
    # Check intersection with every segment of the polygon
    # Segment is P1 to P2
    # Ray is P = t * rd (where t > 0)
    # Intersection: t * rd = P1 + u * (P2 - P1)
    # Solve for t and u. We want smallest positive t.
    
    best_t = None
    
    for i in range(len(poly_points) - 1):
        p1 = poly_points[i]
        p2 = poly_points[i+1]
        
        v1 = p1 # Origin to P1
        v2 = p2 - p1 # Segment vector
        
        # Cross product in 2D is determinant
        # t * rd = p1 + u * v2
        # t * rd - u * v2 = p1
        # Matrix form: [[rd.x, -v2.x], [rd.y, -v2.y]] * [[t], [u]] = [[p1.x], [p1.y]]
        
        det = rd[0] * (-v2[1]) - rd[1] * (-v2[0])
        
        if abs(det) < 1e-8: continue # Parallel
        
        # Cramer's rule or manual inverse
        t = (p1[0] * (-v2[1]) - p1[1] * (-v2[0])) / det
        u = (rd[0] * p1[1] - rd[1] * p1[0]) / det
        
        if t > 0 and 0 <= u <= 1:
            if best_t is None or t < best_t:
                best_t = t
                
    return best_t if best_t is not None else 0.0

def generate_debug_plot(boat_id, mesh_vertices, points_2d, hull_points, aligned_pts, radii, output_dir):
    """
    Generates a 6-panel debug plot for a single ship, similar to debug_stl_processing.py
    """
    fig = plt.figure(figsize=(18, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle(f"Step-by-Step Processing of ID: {boat_id}", fontsize=16)

    # --- STEP 1: RAW 3D MESH ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    step = max(1, len(mesh_vertices) // 1000)
    v = mesh_vertices[::step]
    
    ax1.scatter(v[:,0], v[:,1], v[:,2], c=v[:,2], cmap='viridis', s=1, alpha=0.5)
    ax1.set_title("1. Raw 3D Mesh (Downsampled)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    
    if len(v) > 0:
        max_range = np.array([v[:,0].max()-v[:,0].min(), v[:,1].max()-v[:,1].min(), v[:,2].max()-v[:,2].min()]).max() / 2.0
        mid_x = (v[:,0].max()+v[:,0].min()) * 0.5
        mid_y = (v[:,1].max()+v[:,1].min()) * 0.5
        mid_z = (v[:,2].max()+v[:,2].min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- STEP 2: PROJECTION (The Shadow) ---
    ax2 = fig.add_subplot(2, 3, 2)
    step_2d = max(1, len(points_2d) // 1000)
    ax2.scatter(points_2d[::step_2d, 0], points_2d[::step_2d, 1], s=1, c='gray', alpha=0.5)
    ax2.set_title("2. Top-Down Projection (Z dropped)")
    ax2.set_aspect('equal')
    ax2.grid(True)

    # --- STEP 3: CONVEX HULL (The Rubber Band) ---
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Close loop
    hull_viz = np.vstack((hull_points, hull_points[0]))
    
    ax3.scatter(points_2d[::step_2d, 0], points_2d[::step_2d, 1], s=1, c='gray', alpha=0.2)
    ax3.plot(hull_viz[:,0], hull_viz[:,1], 'r-', linewidth=2, label='Convex Hull')
    ax3.scatter(hull_viz[:-1,0], hull_viz[:-1,1], c="#33db43", s=20, marker='o', label='Hull Vertices')
    ax3.set_title("3. Convex Hull ('Rubber Band')")
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.grid(True)

    # --- STEP 4: ALIGNMENT & NORMALIZATION ---
    ax4 = fig.add_subplot(2, 3, 4)
    # Re-close loop for plotting
    aligned_viz = np.vstack((aligned_pts, aligned_pts[0]))
    
    ax4.plot(aligned_viz[:,0], aligned_viz[:,1], 'b-', linewidth=2, label='Aligned Hull')
    ax4.fill(aligned_viz[:,0], aligned_viz[:,1], 'b', alpha=0.1)
    ax4.set_title("4. Centered & Normalized (Original Orientation)")
    ax4.set_aspect('equal')
    ax4.grid(True)
    ax4.set_xlim(-0.6, 0.6)
    ax4.set_ylim(-0.6, 0.6)

    # --- STEP 5: RAY CASTING (Proper Intersection) ---
    ax5 = fig.add_subplot(2, 3, 5, projection='polar')
    
    target_angles = np.linspace(-np.pi, np.pi, len(radii), endpoint=False)
    
    ax5.plot(target_angles, radii, 'g-', label='Exact Radii')
    ax5.set_title("5. Polar Profile (Exact Intersection)")
    ax5.legend()

    # --- STEP 6: RECONSTRUCTION VS TRUTH ---
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Reconstruct from radii
    rec_x = np.array(radii) * np.cos(target_angles)
    rec_y = np.array(radii) * np.sin(target_angles)
    # Close loop
    rec_x = np.append(rec_x, rec_x[0])
    rec_y = np.append(rec_y, rec_y[0])
    
    aligned_viz = np.vstack((aligned_pts, aligned_pts[0]))

    ax6.plot(aligned_viz[:,0], aligned_viz[:,1], 'b-', linewidth=1, label='True Hull', alpha=0.5)
    ax6.plot(rec_x, rec_y, 'r--', linewidth=2, label='Reconstructed')
    ax6.set_title("6. Final Output vs True Hull")
    ax6.legend()
    ax6.set_aspect('equal')
    ax6.grid(True)
    ax6.set_xlim(-0.6, 0.6)
    ax6.set_ylim(-0.6, 0.6)

    output_file = os.path.join(output_dir, f"debug_{boat_id}.png")
    plt.savefig(output_file)
    plt.close(fig)

metadata = load_metadata(CSV_FILE)
processed_list = []
failed_log = []

stl_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.stl')]
stl_files.sort()

print(f"Found {len(stl_files)} STL files. Processing...")

for fname in tqdm(stl_files):
    try:
        boat_id = os.path.splitext(fname)[0]
        meta = metadata.get(boat_id, {})
        
        full_path = os.path.join(DATA_DIR, fname)
        mesh = trimesh.load(full_path)
        
        if mesh.is_empty:
            raise ValueError("Mesh is empty")

        # --- CORRESPONDS TO STEP 2 (Projection) ---
        points_2d = mesh.vertices[:, :2]

        # --- CORRESPONDS TO STEP 3 (Convex Hull) ---
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
        
        # --- CORRESPONDS TO STEP 4 (Alignment) ---
        # 1. Centering (Essential for Polar conversion)
        # Uses Bounding Box center instead of vertex mean
        min_xy = np.min(hull_points, axis=0) 
        max_xy = np.max(hull_points, axis=0)
        centroid = (min_xy + max_xy) / 2.0
        aligned_pts = hull_points - centroid

        # 1.5. Manual Rotation if needed (180 degrees)
        if boat_id in ROTATED_SHIP_IDS:
            print(f"Applying 180-degree rotation to Boat ID: {boat_id}")
            aligned_pts = -aligned_pts
        
        # 2. Scaling (Normalize length to 1.0)
        min_x = np.min(aligned_pts[:, 0])
        max_x = np.max(aligned_pts[:, 0])
        length = max_x - min_x
        
        if length > 0:
            aligned_pts = aligned_pts / length

        # Close the loop for intersection check
        aligned_pts_closed = np.vstack((aligned_pts, aligned_pts[0]))

        # --- CORRESPONDS TO STEP 5 (Ray Casting) ---   
        target_angles = np.linspace(-np.pi, np.pi, NUM_RAYS, endpoint=False)
        radii = []

        for angle in target_angles:
            r = get_ray_intersection(aligned_pts_closed, angle)
            radii.append(r)
        
        radii = np.array(radii)

        # Generate Debug Plot
        # Pass hull_points and aligned_pts (not closed) to plotting function
        generate_debug_plot(boat_id, mesh.vertices, points_2d, hull_points, aligned_pts, radii, DEBUG_DIR)
        
        entry = {
            "id": boat_id,
            "filename": fname,
            "original_length_m": float(length), 
            "radii": radii.tolist(),
            "name": meta.get('ProjectName', 'Unknown'),
            "type": meta.get('Designer', 'Unknown'),
            "is_boat": safe_int(meta.get('Is Boat', 0)),
            "is_kayak": safe_int(meta.get('Is Kayak', 0)),
            "is_solid": safe_int(meta.get('Is Solid', 0))
        }
        
        processed_list.append(entry)
        
    except Exception as e:
        print(f"FAILED {fname}: {e}")
        failed_log.append(f"{fname}: {str(e)}")

# Save JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(processed_list, f, indent=2)

print(f"\nSaved {len(processed_list)} processed ships to {OUTPUT_JSON}")

# --- VISUALIZATION ---
print("Generating verification images...")

N = len(processed_list)
if N > 0:
    cols = 6
    rows = math.ceil(N / cols)
    
    # --- 1. Cartesian Plot (X vs Y) ---
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows))
    fig.suptitle("Normalized 2D Shapes (Cartesian)", fontsize=16)
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    axes_flat = axes.flatten() if N > 1 else [axes]
    
    for i, ax in enumerate(axes_flat):
        if i < N:
            data = processed_list[i]
            r = np.array(data['radii'])
            
            # Reconstruct (X, Y) from (r, theta)
            theta = np.linspace(-np.pi, np.pi, len(r), endpoint=False)
            
            # Close loop for plotting
            r_plot = np.append(r, r[0])
            theta_plot = np.append(theta, theta[0])
            
            x = r_plot * np.cos(theta_plot)
            y = r_plot * np.sin(theta_plot)
            
            color = '#d62728'
            if data['is_boat'] == 1: color = '#1f77b4'
            if data['is_kayak'] == 1: color = '#2ca02c'
            
            ax.fill(x, y, alpha=0.3, fc=color)
            ax.plot(x, y, color=color, lw=1.5)
            
            # Plot Centroid
            ax.plot(0, 0, 'go', markersize=2)

            name_clean = data['name'][:20] if data['name'] else "Unknown"
            title_text = f"ID: {data['id']}\n{name_clean}"
            ax.set_title(title_text, fontsize=7)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle=':', alpha=0.6)
            # Expanded limits to avoid clipping
            ax.set_xlim(-0.7, 0.7)
            ax.set_ylim(-0.7, 0.7)
            
        else:
            ax.axis('off')
    
    # plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"Inspection image saved to {OUTPUT_IMAGE}")

    # --- 2. Polar Plot (Radius vs Angle) ---
    fig_polar, axes_polar = plt.subplots(rows, cols, figsize=(16, 3.2 * rows))
    fig_polar.suptitle("Polar Profiles (Radius vs Angle)\n0 = Stern, +/- Pi = Bow", fontsize=16)
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    axes_polar_flat = axes_polar.flatten() if N > 1 else [axes_polar]
    
    for i, ax in enumerate(axes_polar_flat):
        if i < N:
            data = processed_list[i]
            r = np.array(data['radii'])
            theta = np.linspace(-np.pi, np.pi, len(r), endpoint=False)
            
            color = '#d62728' # Red
            if data['is_boat'] == 1: color = '#1f77b4'
            if data['is_kayak'] == 1: color = '#2ca02c'

            ax.plot(theta, r, color=color, lw=1.5)
            
            name_clean = data['name'][:20] if data['name'] else "Unknown"
            title_text = f"ID: {data['id']}\n{name_clean}"
            ax.set_title(title_text, fontsize=7)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(0, max(r) * 1.1)
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.grid(True, linestyle=':', alpha=0.6)
            
        else:
            ax.axis('off')

    # plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_POLAR, dpi=150)
    print(f"Polar inspection image saved to {OUTPUT_IMAGE_POLAR}")

    # --- 3. True Polar Projection Plot ---
    fig_polar_proj, axes_polar_proj = plt.subplots(rows, cols, figsize=(16, 3.2 * rows), subplot_kw={'projection': 'polar'})
    fig_polar_proj.suptitle("Polar Projection\nRadius vs Angle", fontsize=16)
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    axes_polar_proj_flat = axes_polar_proj.flatten() if N > 1 else [axes_polar_proj]
    
    for i, ax in enumerate(axes_polar_proj_flat):
        if i < N:
            data = processed_list[i]
            r = np.array(data['radii'])
            theta = np.linspace(-np.pi, np.pi, len(r), endpoint=False)
            
            # Close loop
            r = np.append(r, r[0])
            theta = np.append(theta, theta[0])

            color = '#d62728'
            if data['is_boat'] == 1: color = '#1f77b4'
            if data['is_kayak'] == 1: color = '#2ca02c'

            ax.plot(theta, r, color=color, lw=1.5)
            ax.fill(theta, r, alpha=0.3, fc=color)
            
            name_clean = data['name'][:20] if data['name'] else "Unknown"
            title_text = f"ID: {data['id']}\n{name_clean}"
            ax.set_title(title_text, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle=':', alpha=0.6)
            
        else:
            ax.axis('off')

    # plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_POLAR_PROJ, dpi=150)
    print(f"Polar projection image saved to {OUTPUT_IMAGE_POLAR_PROJ}")