import os
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
TARGET_FILE = "./data/stl_ships/1.stl"
NUM_RAYS = 360
DOWNSAMPLE = 1  # For 3D plotting speed (plot every Nth point)

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

def debug_process(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"Debugging: {filepath}")
    mesh = trimesh.load(filepath)
    
    fig = plt.figure(figsize=(18, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle(f"Step-by-Step Processing of {os.path.basename(filepath)}", fontsize=16)

    # --- STEP 1: RAW 3D MESH ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    v = mesh.vertices[::DOWNSAMPLE] # Downsample for performance
    
    # Plot vertices
    ax1.scatter(v[:,0], v[:,1], v[:,2], c=v[:,2], cmap='viridis', s=1, alpha=0.5)
    ax1.set_title("1. Raw 3D Mesh (Downsampled)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    
    # Force equal aspect ratio for 3D
    max_range = np.array([v[:,0].max()-v[:,0].min(), v[:,1].max()-v[:,1].min(), v[:,2].max()-v[:,2].min()]).max() / 2.0
    mid_x = (v[:,0].max()+v[:,0].min()) * 0.5
    mid_y = (v[:,1].max()+v[:,1].min()) * 0.5
    mid_z = (v[:,2].max()+v[:,2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- STEP 2: PROJECTION (The Shadow) ---
    ax2 = fig.add_subplot(2, 3, 2)
    points_2d = mesh.vertices[:, :2]
    
    ax2.scatter(points_2d[::DOWNSAMPLE, 0], points_2d[::DOWNSAMPLE, 1], s=1, c='gray', alpha=0.5)
    ax2.set_title("2. Top-Down Projection (Z dropped)")
    ax2.set_aspect('equal')
    ax2.grid(True)

    # --- STEP 3: CONVEX HULL (The Rubber Band) ---
    ax3 = fig.add_subplot(2, 3, 3)
    
    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]
    # Close loop
    hull_points = np.vstack((hull_points, hull_points[0]))
    
    ax3.scatter(points_2d[::DOWNSAMPLE, 0], points_2d[::DOWNSAMPLE, 1], s=1, c='gray', alpha=0.2)
    ax3.plot(hull_points[:,0], hull_points[:,1], 'r-', linewidth=2, label='Convex Hull')
    ax3.scatter(hull_points[:-1,0], hull_points[:-1,1], c="#33db43", s=20, marker='o', label='Hull Vertices')
    ax3.set_title("3. Convex Hull ('Rubber Band')")
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.grid(True)

    # --- STEP 4: ALIGNMENT & NORMALIZATION ---
    ax4 = fig.add_subplot(2, 3, 4)
    
    # 1. Centering (Essential for Polar conversion)
    # Uses Bounding Box center instead of vertex mean
    min_xy = np.min(hull_points[:-1], axis=0)
    max_xy = np.max(hull_points[:-1], axis=0)
    centroid = (min_xy + max_xy) / 2.0
    aligned_pts = hull_points - centroid
    
    # 2. Scaling (Normalize length to 1.0)
    min_x = np.min(aligned_pts[:, 0])
    max_x = np.max(aligned_pts[:, 0])
    length = max_x - min_x
    
    if length > 0:
        aligned_pts = aligned_pts / length
    
    ax4.plot(aligned_pts[:,0], aligned_pts[:,1], 'b-', linewidth=2, label='Aligned Hull')
    ax4.fill(aligned_pts[:,0], aligned_pts[:,1], 'b', alpha=0.1)
    ax4.set_title("4. Centered & Normalized (Original Orientation)")
    ax4.set_aspect('equal')
    ax4.grid(True)
    ax4.set_xlim(-0.6, 0.6)
    ax4.set_ylim(-0.6, 0.6)

    # --- STEP 5: RAY CASTING (Proper Intersection) ---
    print(f"Calculating {NUM_RAYS} ray intersections...")
    ax5 = fig.add_subplot(2, 3, 5, projection='polar')
    
    target_angles = np.linspace(-np.pi, np.pi, NUM_RAYS, endpoint=False)
    radii = []
    
    # Intersect every ray with the hull polygon
    for angle in target_angles:
        r = get_ray_intersection(aligned_pts, angle)
        radii.append(r)
    
    radii = np.array(radii)
    
    ax5.plot(target_angles, radii, 'g-', label='Exact Radii')
    ax5.set_title("5. Polar Profile (Exact Intersection)")
    ax5.legend()

    # --- STEP 6: RECONSTRUCTION VS TRUTH ---
    # Goal: Overlay the final result on the hull to see the distortion
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Reconstruct from radii
    rec_x = radii * np.cos(target_angles)
    rec_y = radii * np.sin(target_angles)
    # Close loop
    rec_x = np.append(rec_x, rec_x[0])
    rec_y = np.append(rec_y, rec_y[0])
    
    ax6.plot(aligned_pts[:,0], aligned_pts[:,1], 'b-', linewidth=1, label='True Hull', alpha=0.5)
    ax6.plot(rec_x, rec_y, 'r--', linewidth=2, label='Reconstructed')
    ax6.set_title("6. Final Output vs True Hull")
    ax6.legend()
    ax6.set_aspect('equal')
    ax6.grid(True)
    ax6.set_xlim(-0.6, 0.6)
    ax6.set_ylim(-0.6, 0.6)

    output_file = "debug_step_by_step.png"
    plt.savefig(output_file)
    print(f"Saved debug image to {output_file}")
    plt.show()

if __name__ == "__main__":
    debug_process(TARGET_FILE)