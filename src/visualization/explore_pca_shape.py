import sys
import os
import json
import numpy as np
import panel as pn
import holoviews as hv
import plotly.graph_objects as go
from pathlib import Path
from matplotlib.colors import ListedColormap

# Setup paths
FILE_PATH = Path(__file__).resolve()
SRC_ROOT = FILE_PATH.parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.geometry_utils import compute_estimated_shape_from_params
from src.utils.tools import generate_fourier_function, fourier_transform

pn.extension('plotly', 'mathjax')
hv.extension('bokeh')

class ShapeExplorer(pn.viewable.Viewer):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
             raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        # Paths
        self.npz_files = list(self.data_dir.glob("*.npz"))
        if not self.npz_files:
             raise FileNotFoundError(f"No .npz files found in {data_dir}")
        self.npz_options = {f.name: str(f) for f in self.npz_files}
        self.json_path = PROJECT_ROOT / "data" / "processed_ships.json"

        # --- Internal State ---
        self.N_pca = 4
        self.current_coeffs = np.zeros(self.N_pca)
        self.boat_db = {}
        self.all_boat_coeffs =[]
        self.all_boat_status =[]
        self.all_boat_names =[]
        self._ignore_callbacks = False  # Prevents recursive jumping
        self.last_selected_gt_radii = None
        
        # --- Pre-compute Math Matrices ---
        self.angles_180 = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        self.G_mat = generate_fourier_function(N_f=64)(self.angles_180).T # Shape: (180, 64)
        self.cos_ang = np.cos(self.angles_180)[:, None, None]
        self.sin_ang = np.sin(self.angles_180)[:, None, None]

        # --- GUI Widgets ---
        default_key = next((k for k in self.npz_options.keys() if 'ShipDatasetPCAParameters' in k), list(self.npz_options.keys())[0] if self.npz_options else None)
        self.file_selector = pn.widgets.Select(name='Select PCA Model', options=self.npz_options, value=self.npz_options[default_key], sizing_mode='stretch_width')
        self.n_pca_input = pn.widgets.IntInput(name='Number of PCA Coeffs', value=self.N_pca, start=2, end=20, sizing_mode='stretch_width')
        
        self.boat_selector = pn.widgets.Select(name='Jump to Dataset Boat', options={"Custom": "Custom"}, sizing_mode='stretch_width')
        
        self.slider_L = pn.widgets.FloatSlider(name='Length (L)', start=1.0, end=150.0, step=0.5, value=20.0, sizing_mode='stretch_width')
        self.input_L = pn.widgets.FloatInput(name='Length Input', value=20.0, start=1.0, end=150.0, step=0.5, sizing_mode='stretch_width')
        self.slider_L.link(self.input_L, value='value', bidirectional=True)
        
        self.slider_W = pn.widgets.FloatSlider(name='Width (W)', start=0.5, end=50.0, step=0.1, value=6.0, sizing_mode='stretch_width')
        self.input_W = pn.widgets.FloatInput(name='Width Input', value=6.0, start=0.5, end=50.0, step=0.1, sizing_mode='stretch_width')
        self.slider_W.link(self.input_W, value='value', bidirectional=True)
        
        self.reset_lw_btn = pn.widgets.Button(name='Reset L & W', button_type='warning', sizing_mode='stretch_width')
        self.rotate_toggle = pn.widgets.Checkbox(name='Rotate 90°', value=False, sizing_mode='stretch_width')
        
        self.reset_btn = pn.widgets.Button(name='Reset Coefficients', button_type='warning', sizing_mode='stretch_width')
        
        # Feasibility controls
        self.x_axis_select = pn.widgets.Select(name='X Axis (Feasibility)', options=[], sizing_mode='stretch_width')
        self.y_axis_select = pn.widgets.Select(name='Y Axis (Feasibility)', options=[], sizing_mode='stretch_width')
        self.z_axis_select = pn.widgets.Select(name='Z Axis (3D Feasibility)', options=[], sizing_mode='stretch_width')
        self.show_3d_volume_toggle = pn.widgets.Checkbox(name='Show 3D Volume (Slower)', value=False, sizing_mode='stretch_width')
        self.color_by_feasibility_toggle = pn.widgets.Checkbox(name='Color boats by True Feasibility', value=True, sizing_mode='stretch_width')
        self.include_kayaks_toggle = pn.widgets.Checkbox(name='Include Kayaks', value=False, sizing_mode='stretch_width')
        self.heatmap_res_slider = pn.widgets.IntSlider(name='Heatmap Resolution', start=100, end=1000, step=50, value=300, sizing_mode='stretch_width')

        # Mahalanobis Controls
        self.show_mahalanobis_toggle = pn.widgets.Checkbox(name='Show Mahalanobis Bound', value=True, sizing_mode='stretch_width')
        self.chi2_input = pn.widgets.FloatInput(name='Mahalanobis Chi² Threshold (99% = 13.28)', value=13.28, start=0.1, end=100.0, step=0.1, sizing_mode='stretch_width')

        # Dynamic PCA Sliders Column
        self.pca_sliders_column = pn.Column(sizing_mode='stretch_width')
        self.pca_sliders =[]
        
        # Panes
        self.plotly_pane = pn.pane.Plotly(sizing_mode='stretch_both', config={'responsive': True})
        self.plotly_feasibility_3d = pn.pane.Plotly(sizing_mode='stretch_both', config={'responsive': True})
        self.update_trigger = pn.widgets.Button(visible=False) # Hidden trigger for Bokeh updates

        # Bokeh Tap Stream
        self.tap_stream = hv.streams.Tap(x=None, y=None)

        # --- Initialization ---
        self.load_model(self.file_selector.value)
        self.load_boat_database()
        
        # --- Event Watchers ---
        self.file_selector.param.watch(self.on_file_change, 'value')
        self.n_pca_input.param.watch(self.on_npca_change, 'value')
        self.boat_selector.param.watch(self.on_boat_select, 'value')
        self.reset_btn.on_click(self.reset_coefficients)
        self.reset_lw_btn.on_click(self.reset_lw)
        
        self.slider_L.param.watch(self._on_manual_change, 'value')
        self.slider_W.param.watch(self._on_manual_change, 'value')
        self.rotate_toggle.param.watch(lambda e: self.trigger_update(), 'value')
        
        # Axis Fallback Watchers
        self.x_axis_select.param.watch(self._on_axis_change, 'value')
        self.y_axis_select.param.watch(self._on_axis_change, 'value')
        self.z_axis_select.param.watch(self._on_axis_change, 'value')
        self.color_by_feasibility_toggle.param.watch(lambda e: self.trigger_update(), 'value')
        self.include_kayaks_toggle.param.watch(self.on_kayak_toggle, 'value')
        self.heatmap_res_slider.param.watch(lambda e: self.trigger_update(), 'value')
        self.show_3d_volume_toggle.param.watch(lambda e: self.trigger_update(), 'value')
        
        # Mahalanobis Watchers
        self.show_mahalanobis_toggle.param.watch(lambda e: self.trigger_update(), 'value')
        self.chi2_input.param.watch(lambda e: self.trigger_update(), 'value')

        self.tap_stream.param.watch(self._on_tap,['x', 'y'])

        self.trigger_update()

    # --- Loading & Data Processing ---
    
    def load_model(self, pca_path):
        data = np.load(pca_path)
        self.full_eigenvectors = data['eigenvectors'].real 
        self.mean_coeffs = data['mean'].flatten()
        
        if 'eigenvalues' in data:
            self.full_eigenvalues = data['eigenvalues'].real
        else:
            print("Warning: 'eigenvalues' not found in PCA .npz file. Defaulting to 1.0.")
            self.full_eigenvalues = np.ones(self.full_eigenvectors.shape[1])
            
        self.n_fourier_dim = self.mean_coeffs.shape[0]
        self.angles = np.linspace(0, 2*np.pi, 200)
        self._build_pca_dependencies()

    def _build_pca_dependencies(self):
        # Truncate to N_pca
        self.eigenvectors = self.full_eigenvectors[:, :self.N_pca]
        self.eigenvalues = self.full_eigenvalues[:self.N_pca]
        
        # Precompute projection matrices for fast heatmap
        self.M_mat = self.G_mat @ self.eigenvectors # Shape: (180, N_pca)
        self.R_mean = self.G_mat @ self.mean_coeffs # Shape: (180,)

        # Re-create internal coeffs
        old_coeffs = self.current_coeffs.copy()
        self.current_coeffs = np.zeros(self.N_pca)
        for i in range(min(len(old_coeffs), self.N_pca)):
            self.current_coeffs[i] = old_coeffs[i]

        # Re-create sliders
        self.pca_sliders =[]
        self.pca_sliders_column.clear()
        
        for i in range(self.N_pca):
            s = pn.widgets.FloatSlider(name=f'PC_{i}', start=-15.0, end=15.0, step=0.05, value=self.current_coeffs[i], sizing_mode='stretch_width')
            s.param.watch(self._on_slider_change, 'value')
            self.pca_sliders.append(s)
            self.pca_sliders_column.append(s)
            
        # Re-create Axis selectors
        opts =[f"PC_{i}" for i in range(self.N_pca)]
        self.x_axis_select.options = opts
        self.y_axis_select.options = opts
        self.z_axis_select.options = opts
        self.x_axis_select.value = opts[0]
        self.y_axis_select.value = opts[1] if self.N_pca > 1 else opts[0]
        self.z_axis_select.value = opts[2] if self.N_pca > 2 else opts[0]
        
        self.load_boat_database() # Re-project boats to new N_pca

    def load_boat_database(self):
        """Batch loads and projects all boats into PCA space, and calculates their 4D status."""
        if not self.json_path.exists(): return
        
        with open(self.json_path, 'r') as f:
            boats_data = json.load(f)
            
        self.boat_db = {}
        self.all_boat_coeffs = []
        self.all_boat_status =[]
        self.all_boat_names =[]
        
        for boat in boats_data:
            is_boat = boat.get('is_boat', False)
            is_kayak = boat.get('is_kayak', False) or 'kayak' in str(boat.get('name', '')).lower()
            
            if not is_boat and not is_kayak: continue
            if is_kayak and not self.include_kayaks_toggle.value: continue
            if len(boat.get('radii',[])) == 0: continue
            
            radii = np.array(boat['radii'])
            angs = np.linspace(-np.pi, np.pi, len(radii), endpoint=False)
            
            vec = fourier_transform(angs, radii, num_coeff=64, symmetry=True).flatten()
            if vec.shape != self.mean_coeffs.shape:
                vec = fourier_transform(angs, radii, num_coeff=64, symmetry=False).flatten()
                
            centered_vec = vec - self.mean_coeffs
            s = np.linalg.norm(self.full_eigenvectors[:, 0])
            scaling_factor_sq = s**2
            
            # Project to get coefficients up to N_pca
            coeffs = (self.full_eigenvectors.T @ centered_vec) / scaling_factor_sq
            coeffs = coeffs[:self.N_pca]

            # Determine true N-dimensional feasibility for these specific coeffs
            r_4d = self.R_mean + self.M_mat @ coeffs
            x_4d = r_4d * np.cos(self.angles_180)
            y_4d = r_4d * np.sin(self.angles_180)

            if np.any(r_4d < 0):
                status = -2  # Negative Radius
            elif np.max(np.abs(x_4d)) > 0.505 or np.max(np.abs(y_4d)) > 0.505:
                status = -1  # Spills outside 1x1
            else:
                status = 1   # Fully Feasible
            
            boat_id = str(boat['id'])
            self.boat_db[boat_id] = {
                'id': boat_id,
                'name': boat.get('name', 'Unknown'),
                'L': boat.get('original_length_m', 20.0),
                'W': boat.get('original_width_m', 6.0),
                'coeffs': coeffs,
                'status': status,
                'radii': radii
            }
            self.all_boat_coeffs.append(coeffs)
            self.all_boat_status.append(status)
            self.all_boat_names.append(f"ID: {boat_id} - {boat.get('name', 'Unknown')} | Boat: {is_boat} | Kayak: {is_kayak}")
            
        self.all_boat_coeffs = np.array(self.all_boat_coeffs)
        self.all_boat_status = np.array(self.all_boat_status)
        self.all_boat_names = np.array(self.all_boat_names)
        
        # Update dropdown
        opts = {"Custom": "Custom"}
        for k, v in self.boat_db.items():
            opts[f"{k} - {v['name'][:15]}"] = k
        self.boat_selector.options = opts

    # --- Events ---
    
    def on_file_change(self, event):
        self._ignore_callbacks = True
        self.load_model(event.new)
        self.boat_selector.value = "Custom"
        self._ignore_callbacks = False
        self.trigger_update()
        
    def on_npca_change(self, event):
        if event.new < 2:
            self.n_pca_input.value = 2
            return
        self.N_pca = event.new
        
        self._ignore_callbacks = True
        self._build_pca_dependencies()
        self.boat_selector.value = "Custom"
        self._ignore_callbacks = False
        
        self.trigger_update()

    def on_kayak_toggle(self, event):
        self._ignore_callbacks = True
        self.load_boat_database()
        self.boat_selector.value = "Custom"
        self._ignore_callbacks = False
        self.trigger_update()

    def on_boat_select(self, event):
        boat_id = event.new
        if boat_id == "Custom" or boat_id not in self.boat_db: return
        
        boat = self.boat_db[boat_id]
        self.last_selected_gt_radii = boat.get('radii', None)
        
        self._ignore_callbacks = True
        self.slider_L.value = boat['L']
        self.slider_W.value = boat['W']
        
        for i, val in enumerate(boat['coeffs']):
            self.pca_sliders[i].value = val
            self.current_coeffs[i] = val
        self._ignore_callbacks = False
            
        self.trigger_update()

    def _on_slider_change(self, event):
        if getattr(self, '_ignore_callbacks', False):
            return
        
        for i, s in enumerate(self.pca_sliders):
            self.current_coeffs[i] = s.value
            
        if self.boat_selector.value != "Custom":
            self._ignore_callbacks = True
            self.boat_selector.value = "Custom"
            self._ignore_callbacks = False
            
        self.trigger_update()

    def _on_manual_change(self, event):
        if getattr(self, '_ignore_callbacks', False):
            return
            
        if self.boat_selector.value != "Custom":
            self._ignore_callbacks = True
            self.boat_selector.value = "Custom"
            self._ignore_callbacks = False
            
        self.trigger_update()

    def _on_axis_change(self, event):
        """Prevents selecting the same PCA component for X, Y, and Z axes."""
        x_val = self.x_axis_select.value
        y_val = self.y_axis_select.value
        z_val = self.z_axis_select.value

        # Enforce uniqueness, prioritizing X and Y over Z. 
        if event.obj is self.x_axis_select:
            if x_val == y_val:
                fallback = next(p for p in self.y_axis_select.options if p not in [x_val, z_val])
                self.y_axis_select.value = fallback
            elif x_val == z_val:
                fallback = next(p for p in self.z_axis_select.options if p not in [x_val, y_val])
                self.z_axis_select.value = fallback
        elif event.obj is self.y_axis_select:
            if y_val == x_val:
                fallback = next(p for p in self.x_axis_select.options if p not in[y_val, z_val])
                self.x_axis_select.value = fallback
            elif y_val == z_val:
                fallback = next(p for p in self.z_axis_select.options if p not in [x_val, y_val])
                self.z_axis_select.value = fallback
        elif event.obj is self.z_axis_select:
            if z_val in [x_val, y_val]:
                fallback = next(p for p in self.z_axis_select.options if p not in [x_val, y_val])
                self.z_axis_select.value = fallback

        self.trigger_update()

    def _on_tap(self, *events):
        x, y = self.tap_stream.x, self.tap_stream.y
        if x is None or y is None: return
        
        d1 = int(self.x_axis_select.value.split('_')[1])
        d2 = int(self.y_axis_select.value.split('_')[1])
        
        self._ignore_callbacks = True
        self.pca_sliders[d1].value = x
        self.pca_sliders[d2].value = y
        self.current_coeffs[d1] = x
        self.current_coeffs[d2] = y
        self.boat_selector.value = "Custom"
        self._ignore_callbacks = False
        
        self.trigger_update()

    def reset_coefficients(self, event=None):
        self._ignore_callbacks = True
        for s in self.pca_sliders: 
            s.value = 0.0
        self.current_coeffs.fill(0.0)
        self.boat_selector.value = "Custom"
        self._ignore_callbacks = False
        self.trigger_update()

    def reset_lw(self, event=None):
        self._ignore_callbacks = True
        self.slider_L.value = 20.0
        self.slider_W.value = 6.0
        self._ignore_callbacks = False
        self.trigger_update()

    def trigger_update(self):
        # Update Plotly
        self.update_plotly()
        self.update_plotly_feasibility_3d()
        # Trigger Bokeh dynamic map update
        self.update_trigger.clicks += 1

    # --- Plotting ---

    def update_plotly(self):
        L = self.slider_L.value
        W = self.slider_W.value
        yaw = np.pi / 2 if self.rotate_toggle.value else 0.0

        est_shape_x, est_shape_y = compute_estimated_shape_from_params(
            0.0, 0.0, yaw, L, W, 
            self.current_coeffs, self.mean_coeffs.reshape(-1, 1), self.eigenvectors, 
            self.angles, self.n_fourier_dim
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=est_shape_y, y=est_shape_x, mode='lines', name='PCA Shape', line=dict(color='royalblue', width=3), fill='toself', fillcolor='rgba(65, 105, 225, 0.2)'))

        # L x W Reference Box
        box_local_x = np.array([L/2, L/2, -L/2, -L/2, L/2])
        box_local_y = np.array([-W/2, W/2, W/2, -W/2, -W/2])
        c, s = np.cos(yaw), np.sin(yaw)
        box_rot_x = c * box_local_x - s * box_local_y
        box_rot_y = s * box_local_x + c * box_local_y

        fig.add_trace(go.Scatter(x=box_rot_y, y=box_rot_x, mode='lines', name='L x W Ref Box', line=dict(color='gray', dash='dash')))

        fig.update_layout(
            title="Scaled Geometry (Real World Coordinates)",
            autosize=True,
            xaxis=dict(range=[-40, 40], constrain='domain', title="East / Y [m]"), 
            yaxis=dict(range=[-50, 50], scaleanchor="x", scaleratio=1, title="North / X[m]"), 
            template="plotly_white", margin=dict(l=20, r=20, t=40, b=20)
        )
        self.plotly_pane.object = fig 

    def update_plotly_feasibility_3d(self):
        d1 = int(self.x_axis_select.value.split('_')[1])
        d2 = int(self.y_axis_select.value.split('_')[1])
        d3 = int(self.z_axis_select.value.split('_')[1])
        
        fig = go.Figure()
        
        res_3d = 30
        limit = 15.0
        grid_vals = np.linspace(-limit, limit, res_3d)
        
        if self.show_3d_volume_toggle.value or self.show_mahalanobis_toggle.value:
            X_grid, Y_grid, Z_grid = np.meshgrid(grid_vals, grid_vals, grid_vals, indexing='ij')

        if self.show_3d_volume_toggle.value:
            fixed_coeffs = self.current_coeffs.copy()
            fixed_coeffs[d1] = 0.0
            fixed_coeffs[d2] = 0.0
            fixed_coeffs[d3] = 0.0

            R_fixed = self.R_mean + self.M_mat @ fixed_coeffs
            R_3d = (R_fixed[:, None, None, None] + 
                    self.M_mat[:, d1, None, None, None] * X_grid + 
                    self.M_mat[:, d2, None, None, None] * Y_grid + 
                    self.M_mat[:, d3, None, None, None] * Z_grid)
            
            cos_ang_3d = self.cos_ang[:, :, None]
            sin_ang_3d = self.sin_ang[:, :, None]

            X_3d_geom = R_3d * cos_ang_3d
            Y_3d_geom = R_3d * sin_ang_3d

            # Continuous metrics mapping distance to invalidity (Feasible where > 0)
            margin_radius = np.min(R_3d, axis=0)
            margin_spill_x = 0.505 - np.max(np.abs(X_3d_geom), axis=0)
            margin_spill_y = 0.505 - np.max(np.abs(Y_3d_geom), axis=0)
            feasibility_metric = np.minimum.reduce([margin_radius, margin_spill_x, margin_spill_y])

            fig.add_trace(go.Isosurface(
                x=X_grid.flatten(), y=Y_grid.flatten(), z=Z_grid.flatten(),
                value=feasibility_metric.flatten(),
                isomin=-0.05, isomax=0.05,
                surface_count=2,
                caps=dict(x_show=False, y_show=False, z_show=False),
                colorscale=[[0, 'rgba(46, 204, 113, 0.4)'],[1, 'rgba(46, 204, 113, 0.4)']],
                showscale=False, opacity=0.6,
                name='Feasible Boundary'
            ))

        if self.show_mahalanobis_toggle.value:
            # Accumulate Mahalanobis penalty from dimensions NOT in this 3D plot
            fixed_mahal_sq = sum(
                (self.current_coeffs[i]**2) / self.eigenvalues[i] 
                for i in range(self.N_pca) if i not in (d1, d2, d3)
            )
            
            # The remaining budget available for d1, d2, and d3
            budget = self.chi2_input.value - fixed_mahal_sq
            
            if budget > 0:
                # --- Analytical Parametric Ellipsoid ---
                # This guarantees it renders perfectly even if the ellipsoid is tiny!
                u = np.linspace(0, 2 * np.pi, 40)
                v = np.linspace(0, np.pi, 40)
                U, V = np.meshgrid(u, v)
                
                # Parametric equation of an ellipsoid
                x_ell = np.sqrt(budget * self.eigenvalues[d1]) * np.cos(U) * np.sin(V)
                y_ell = np.sqrt(budget * self.eigenvalues[d2]) * np.sin(U) * np.sin(V)
                z_ell = np.sqrt(budget * self.eigenvalues[d3]) * np.cos(V)
                
                fig.add_trace(go.Surface(
                    x=x_ell, y=y_ell, z=z_ell,
                    surfacecolor=np.zeros_like(z_ell),  # Forces solid color mapping
                    colorscale=[[0, 'mediumorchid'],[1, 'mediumorchid']],
                    showscale=False,
                    opacity=0.3,
                    name='Mahalanobis Bound',
                    showlegend=True
                ))

        if len(self.all_boat_coeffs) > 0:
            if self.color_by_feasibility_toggle.value:
                safe_mask = self.all_boat_status == 1
                spill_mask = self.all_boat_status == -1
                neg_mask = self.all_boat_status == -2
                
                if np.any(safe_mask):
                    fig.add_trace(go.Scatter3d(
                        x=self.all_boat_coeffs[safe_mask, d1], y=self.all_boat_coeffs[safe_mask, d2], z=self.all_boat_coeffs[safe_mask, d3],
                        text=self.all_boat_names[safe_mask], hoverinfo='text', mode='markers',
                        marker=dict(size=4, color='#2ecc71', line=dict(width=1, color='black')), name='Safe'
                    ))
                if np.any(spill_mask):
                    fig.add_trace(go.Scatter3d(
                        x=self.all_boat_coeffs[spill_mask, d1], y=self.all_boat_coeffs[spill_mask, d2], z=self.all_boat_coeffs[spill_mask, d3],
                        text=self.all_boat_names[spill_mask], hoverinfo='text', mode='markers',
                        marker=dict(size=4, color='#f39c12', line=dict(width=1, color='black')), name='Spill'
                    ))
                if np.any(neg_mask):
                    fig.add_trace(go.Scatter3d(
                        x=self.all_boat_coeffs[neg_mask, d1], y=self.all_boat_coeffs[neg_mask, d2], z=self.all_boat_coeffs[neg_mask, d3],
                        text=self.all_boat_names[neg_mask], hoverinfo='text', mode='markers',
                        marker=dict(size=4, color='#e74c3c', line=dict(width=1, color='black')), name='Neg Radius'
                    ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=self.all_boat_coeffs[:, d1], y=self.all_boat_coeffs[:, d2], z=self.all_boat_coeffs[:, d3],
                    text=self.all_boat_names, hoverinfo='text', mode='markers',
                    marker=dict(size=3, color='black', opacity=0.3), name='Boats'
                ))
                
        fig.add_trace(go.Scatter3d(
            x=[self.current_coeffs[d1]], y=[self.current_coeffs[d2]], z=[self.current_coeffs[d3]],
            mode='markers', marker=dict(size=8, color='blue', symbol='x', line=dict(width=2, color='white')), name='Current'
        ))
        
        fig.update_layout(
            title=f"3D Feasibility Space",
            scene=dict(xaxis_title=f"PC_{d1}", yaxis_title=f"PC_{d2}", zaxis_title=f"PC_{d3}"),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        self.plotly_feasibility_3d.object = fig

    def get_feasibility_heatmap(self, _trigger, x=None, y=None):
        if self.x_axis_select.value == self.y_axis_select.value:
            return hv.Curve([]).opts(title="Select different X and Y axes")

        d1 = int(self.x_axis_select.value.split('_')[1])
        d2 = int(self.y_axis_select.value.split('_')[1])

        # 1. Evaluate Heatmap Grid
        res = self.heatmap_res_slider.value 
        limit = 15.0
        grid_vals = np.linspace(-limit, limit, res)
        X_grid, Y_grid = np.meshgrid(grid_vals, grid_vals)

        fixed_coeffs = self.current_coeffs.copy()
        fixed_coeffs[d1] = 0.0
        fixed_coeffs[d2] = 0.0

        # R_fixed: (180,)
        R_fixed = self.R_mean + self.M_mat @ fixed_coeffs
        
        # Broadcast into (180, res, res)
        R_3d = R_fixed[:, None, None] + self.M_mat[:, d1, None, None] * X_grid + self.M_mat[:, d2, None, None] * Y_grid
        
        X_3d = R_3d * self.cos_ang
        Y_3d = R_3d * self.sin_ang

        feasibility = np.ones((res, res))
        feasibility[np.any(np.abs(X_3d) > 0.505, axis=0) | np.any(np.abs(Y_3d) > 0.505, axis=0)] = -1
        feasibility[np.any(R_3d < 0, axis=0)] = -2

        # 2. Build Heatmap Plot
        cmap = ListedColormap(['#e74c3c', '#f39c12', '#2ecc71']) # Red (Neg), Orange (Spill), Green (Safe)
        
        heatmap = hv.Image((grid_vals, grid_vals, feasibility), kdims=[f'PC_{d1}', f'PC_{d2}']).opts(
            cmap=cmap, clim=(-2, 1), width=500, height=500, 
            tools=['hover', 'tap'], active_tools=['tap'],
            title=f"Feasibility Space (PC_{d1} vs PC_{d2})",
            xlabel=f"PC_{d1}", ylabel=f"PC_{d2}"
        )

        final_plot = heatmap

        # 3. Add Mahalanobis Bound
        if self.show_mahalanobis_toggle.value:
            # Calculate distance consumed by the dimensions NOT on the X/Y axes
            fixed_mahal_sq = sum(
                (self.current_coeffs[i]**2) / self.eigenvalues[i] 
                for i in range(self.N_pca) if i not in [d1, d2]
            )
            
            # The remaining budget available for d1 and d2
            budget = self.chi2_input.value - fixed_mahal_sq
            
            if budget > 0:
                # Calculate ellipse boundary
                alpha = np.linspace(0, 2*np.pi, 100)
                el_x = np.sqrt(budget * self.eigenvalues[d1]) * np.cos(alpha)
                el_y = np.sqrt(budget * self.eigenvalues[d2]) * np.sin(alpha)
                
                # Close the curve
                el_x = np.append(el_x, el_x[0])
                el_y = np.append(el_y, el_y[0])
                
                ellipse = hv.Curve((el_x, el_y), label='Mahalanobis Bound').opts(
                    color='mediumorchid', line_width=3, line_dash='dashed'
                )
                final_plot = final_plot * ellipse

        # 4. Plot actual boats
        if len(self.all_boat_coeffs) > 0:
            scatter_kdims = [f'PC_{d1}']
            scatter_vdims =[f'PC_{d2}', 'Name']
            if self.color_by_feasibility_toggle.value:
                # Color by true dimensionality status
                safe_mask = self.all_boat_status == 1
                spill_mask = self.all_boat_status == -1
                neg_mask = self.all_boat_status == -2
                
                if np.any(safe_mask):
                    final_plot *= hv.Scatter((self.all_boat_coeffs[safe_mask, d1], self.all_boat_coeffs[safe_mask, d2], self.all_boat_names[safe_mask]), kdims=scatter_kdims, vdims=scatter_vdims).opts(
                        color='#2ecc71', size=6, line_color='black', alpha=0.9, tools=['hover'])
                if np.any(spill_mask):
                    final_plot *= hv.Scatter((self.all_boat_coeffs[spill_mask, d1], self.all_boat_coeffs[spill_mask, d2], self.all_boat_names[spill_mask]), kdims=scatter_kdims, vdims=scatter_vdims).opts(
                        color='#f39c12', size=6, line_color='black', alpha=0.9, tools=['hover'])
                if np.any(neg_mask):
                    final_plot *= hv.Scatter((self.all_boat_coeffs[neg_mask, d1], self.all_boat_coeffs[neg_mask, d2], self.all_boat_names[neg_mask]), kdims=scatter_kdims, vdims=scatter_vdims).opts(
                        color='#e74c3c', size=6, line_color='black', alpha=0.9, tools=['hover'])
            else:
                # Flat transparent color
                final_plot *= hv.Scatter((self.all_boat_coeffs[:, d1], self.all_boat_coeffs[:, d2], self.all_boat_names), kdims=scatter_kdims, vdims=scatter_vdims).opts(
                    color='black', size=3, alpha=0.3, tools=['hover']
                )

        # Plot Current Cursor ON TOP
        cursor = hv.Points([(self.current_coeffs[d1], self.current_coeffs[d2])]).opts(
            color='blue', marker='star', size=15, line_color='white'
        )
        
        return final_plot * cursor

    def get_normalized_geometry(self, _trigger):
        # 3. Build Normalized Geometry Plot
        r_curr = self.R_mean + self.M_mat @ self.current_coeffs
        x_curr = r_curr * np.cos(self.angles_180)
        y_curr = r_curr * np.sin(self.angles_180)
        
        # Close loop
        x_curr = np.append(x_curr, x_curr[0])
        y_curr = np.append(y_curr, y_curr[0])
        
        # Apply rotation if toggled
        if self.rotate_toggle.value:
            x_curr_rot = -y_curr
            y_curr_rot = x_curr
            x_curr, y_curr = x_curr_rot, y_curr_rot

        geom = hv.Curve((y_curr, x_curr), label="PCA Shape").opts(
            color='royalblue', line_width=2, line_dash='solid', width=400, height=500,
            title="Normalized Geometry (r(θ))",
            xlabel="Normalized Width (Y)", ylabel="Normalized Length (X)",
            xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), data_aspect=1, show_legend=True
        )
        
        final_plot = geom

        # Overlay GT shape faintly
        if self.last_selected_gt_radii is not None:
            r_gt = self.last_selected_gt_radii
            ang_gt = np.linspace(-np.pi, np.pi, len(r_gt), endpoint=False)
            x_gt = r_gt * np.cos(ang_gt)
            y_gt = r_gt * np.sin(ang_gt)
            
            x_gt = np.append(x_gt, x_gt[0])
            y_gt = np.append(y_gt, y_gt[0])
            
            if self.rotate_toggle.value:
                x_gt_rot = -y_gt
                y_gt_rot = x_gt
                x_gt, y_gt = x_gt_rot, y_gt
            
            gt_geom = hv.Curve((y_gt, x_gt), label="GT Shape").opts(color='black', alpha=0.3, line_width=2, line_dash='dotted')
            final_plot = gt_geom * final_plot
        
        # 1x1 Box
        box_x = np.array([-0.5, 0.5, 0.5, -0.5, -0.5])
        box_y = np.array([-0.5, -0.5, 0.5, 0.5, -0.5])
        if self.rotate_toggle.value:
            box_x_rot = -box_y
            box_y_rot = box_x
            box_x, box_y = box_x_rot, box_y_rot
            
        box = hv.Curve((box_y, box_x), label="Bounds").opts(color='gray', line_dash='dashed')

        return final_plot * box

    # --- View Layout ---

    def view(self):
        sidebar = pn.Column(
            pn.pane.Markdown("### Data & Extent Settings"),
            self.file_selector,
            self.n_pca_input,
            self.boat_selector,
            pn.layout.Divider(),
            pn.pane.Markdown("### Size & Rotation"),
            pn.Row(self.slider_L, self.input_L, sizing_mode='stretch_width'),
            pn.Row(self.slider_W, self.input_W, sizing_mode='stretch_width'),
            self.reset_lw_btn,
            self.rotate_toggle,
            pn.layout.Divider(),
            pn.pane.Markdown("### Feasibility Controls"),
            self.x_axis_select,
            self.y_axis_select,
            self.z_axis_select,
            self.show_3d_volume_toggle,
            self.color_by_feasibility_toggle,
            self.include_kayaks_toggle,
            self.heatmap_res_slider,
            pn.layout.Divider(),
            pn.pane.Markdown("### Mahalanobis Projection"),
            self.show_mahalanobis_toggle,
            self.chi2_input,
            pn.layout.Divider(),
            pn.pane.Markdown("### PCA Coefficients"),
            self.reset_btn,
            pn.Spacer(height=10),
            self.pca_sliders_column,
            sizing_mode='stretch_width' 
        )
        
        # Split into two DynamicMaps to resolve the Layout/Tap conflict
        dmap_heatmap = hv.DynamicMap(
            pn.bind(self.get_feasibility_heatmap, _trigger=self.update_trigger.param.clicks), 
            streams=[self.tap_stream]
        )
        dmap_geom = hv.DynamicMap(
            pn.bind(self.get_normalized_geometry, _trigger=self.update_trigger.param.clicks)
        )
        
        # Combine using Panel Row instead of HoloViews Layout (+)
        feasibility_layout = pn.Row(
            pn.pane.HoloViews(dmap_heatmap, backend='bokeh'),
            pn.pane.HoloViews(dmap_geom, backend='bokeh')
        )
        
        tabs = pn.Tabs(
            ("Interactive Feasibility Landscape", feasibility_layout),
            ("3D Feasibility Space", self.plotly_feasibility_3d),
            ("2D Shape Viewer", self.plotly_pane)
        )
        
        main_area = pn.Column(tabs, sizing_mode='stretch_both')
        
        return pn.template.FastListTemplate(
            title="PCA Shape Explorer",
            sidebar=[sidebar],
            main=[main_area],
            sidebar_width=450
        )

if __name__ == "__main__":
    DATA_DIR = PROJECT_ROOT / "data" / "input_parameters"
    
    if not DATA_DIR.exists():
        print(f"WARNING: Data directory not found at {DATA_DIR}")
    else:
        try:
            explorer = ShapeExplorer(str(DATA_DIR))
            explorer.view().show()
        except FileNotFoundError as e:
            print(f"Error initializing explorer: {e}")