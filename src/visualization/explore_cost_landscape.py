import sys
import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
import pickle
from pathlib import Path

# Setup Paths
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent.parent
sys.path.append(str(project_root))

from src.global_project_paths import SIMDATA_PATH
from src.tracker.BFGS import BFGS
from src.utils.tools import calculate_body_angles, ssa
from src.states.states import State_PCA, State_GP, LidarScan
from src.utils.geometry_utils import compute_estimated_shape_global, compute_exact_vessel_shape_global

pn.extension('katex', 'plotly', 'tabulator')
hv.extension('bokeh', 'plotly')

class CostLandscapeExplorer(pn.viewable.Viewer):
    def __init__(self, filename):
        self.filename = filename
        print(f"Loading {filename}...")
        
        with open(SIMDATA_PATH / filename, "rb") as f:
            self.sim_result = pickle.load(f)
            
        self.config = self.sim_result.config
        
        # Load PCA params
        self.pca_params = None
        if hasattr(self.config.tracker, 'PCA_parameters_path'):
            self.pca_params = np.load(project_root / self.config.tracker.PCA_parameters_path)

        # Re-instantiate Tracker Logic
        from src.dynamics.process_models import Model_PCA_CV
        from src.sensors.LidarModel import LidarMeasurementModel
        
        filter_dyn_model = Model_PCA_CV(
            x_pos_std_dev=self.config.tracker.pos_north_std_dev,
            y_pos_std_dev=self.config.tracker.pos_east_std_dev,
            yaw_std_dev=self.config.tracker.heading_std_dev,
            N_pca=self.config.tracker.N_pca
        )
        
        sensor_model = LidarMeasurementModel(
            lidar_position=np.array(self.config.lidar.lidar_position),
            lidar_std_dev=self.config.tracker.lidar_std_dev,
            extent_cfg=self.config.extent,
            pca_mean=self.pca_params['mean'],
            pca_eigenvectors=self.pca_params['eigenvectors'][:, :self.config.tracker.N_pca].real,
        )
        
        self.tracker = BFGS(dynamic_model=filter_dyn_model, lidar_model=sensor_model, config=self.config)

        # --- Internal State ---
        self._current_frame_idx = -1
        self._cache_grid_data = {} 
        self._cursor_state = None 
        
        # --- Widgets ---
        self.frame_slider = pn.widgets.IntSlider(name='Frame', start=1, end=self.config.sim.num_frames, value=200)
        
        self.param_options = ['x', 'y', 'yaw', 'length', 'width'] + [f'pca_{i}' for i in range(self.config.tracker.N_pca)]
        self.x_axis_select = pn.widgets.Select(name='X Axis', options=self.param_options, value='yaw')
        self.y_axis_select = pn.widgets.Select(name='Y Axis', options=self.param_options, value='length')
        
        self.center_select = pn.widgets.RadioButtonGroup(
            name='Center View On', options=['Estimate', 'Ground Truth'], value='Estimate', button_type='primary'
        )
        
        self.behavior_toggle = pn.widgets.RadioButtonGroup(
            name='Selection Mode', options=['Reset on Change', 'Keep Modifications'], value='Reset on Change', button_type='success'
        )
        
        self.cost_scale_toggle = pn.widgets.RadioButtonGroup(
            name='Cost Scale', options=['Raw', 'Logarithmic'], value='Raw', button_type='primary'
        )
        
        self.penalty_toggle = pn.widgets.Toggle(name='Include Penalty', value=True, button_type='success')
        
        self.range_slider = pn.widgets.FloatSlider(name='Grid Range (+/-)', start=0.1, end=10.0, value=4.0)
        self.resolution_slider = pn.widgets.IntSlider(name='Grid Resolution', start=10, end=100, value=50)
        
        self.reset_btn = pn.widgets.Button(name='Reset Cursor to Anchor', button_type='warning')
        self.reset_btn.on_click(self.reset_cursor)

        # --- Tap Stream & Trigger ---
        self.tap_stream = hv.streams.Tap(x=None, y=None)
        self.update_trigger = pn.widgets.Button(name='Trigger', visible=False)

        # Setup Watchers
        self.x_axis_select.param.watch(self._on_axis_change, 'value')
        self.y_axis_select.param.watch(self._on_axis_change, 'value')
        self.center_select.param.watch(self.reset_cursor, 'value')
        # Trigger update when scale changes
        self.cost_scale_toggle.param.watch(lambda e: setattr(self.update_trigger, 'clicks', self.update_trigger.clicks + 1), 'value')
        self.penalty_toggle.param.watch(lambda e: setattr(self.update_trigger, 'clicks', self.update_trigger.clicks + 1), 'value')

        super().__init__()

    # --- Event Handling ---

    @pn.depends('tap_stream.x', 'tap_stream.y', watch=True)
    def _on_tap(self, *events):
        x, y = self.tap_stream.x, self.tap_stream.y
        if x is None or y is None: return
        
        x_param = self.x_axis_select.value
        y_param = self.y_axis_select.value
        
        self._cursor_state = self._modify_state(self._cursor_state, x_param, x)
        self._cursor_state = self._modify_state(self._cursor_state, y_param, y)
        
        print(f"Tap Registered: {x_param}={x:.4f}, {y_param}={y:.4f}")
        self.update_trigger.clicks += 1

    def _on_axis_change(self, event):
        # Prevent selecting same axis
        if self.x_axis_select.value == self.y_axis_select.value:
            fallback = next(p for p in self.param_options if p != event.new)
            if event.obj is self.x_axis_select:
                self.y_axis_select.value = fallback
            else:
                self.x_axis_select.value = fallback
            return

        if self.behavior_toggle.value == 'Reset on Change':
            self.reset_cursor()
        else:
            self.update_trigger.clicks += 1

    def reset_cursor(self, event=None):
        if self._current_frame_idx == -1: return
        self._cursor_state = self._get_anchor_state().copy()
        print("Cursor reset to anchor.")
        self.update_trigger.clicks += 1

    # --- Data & State Logic ---

    def _get_frame_data(self, frame_idx):
        if frame_idx == self._current_frame_idx:
            return

        res = self.sim_result.tracker_results_ts.values[frame_idx]
        self._state_est = res.state_posterior.mean
        self._state_pred = res.state_prior.mean
        self._P_pred = res.state_prior.cov
        self._state_gt = self.sim_result.ground_truth_ts.values[frame_idx]
        self._meas_global = self.sim_result.measurements_global_ts.values[frame_idx]
        self._z_flat = self._meas_global.flatten('F')
        
        lidar_pos = self.config.lidar.lidar_position
        dx = self._meas_global.x - lidar_pos[0]
        dy = self._meas_global.y - lidar_pos[1]
        angles = np.arctan2(dy, dx)
        from src.utils.tools import compute_angle_range
        self._lower_diff, self._upper_diff, self._mean_lidar_angle = compute_angle_range(angles)
        
        self._current_frame_idx = frame_idx
        self._cache_grid_data = {} 
        self._cursor_state = self._get_anchor_state().copy()

    def _get_anchor_state(self):
        return self._state_est if self.center_select.value == 'Estimate' else self._state_gt

    def _modify_state(self, base_state, param_name, value):
        if base_state is None: return None
        new_state = base_state.copy()
        if param_name.startswith('pca_'):
            idx = int(param_name.split('_')[1])
            new_state.pca_coeffs[idx] = value
        else:
            setattr(new_state, param_name, value)
        return new_state

    def _get_param_value(self, state, param_name):
        if state is None: return 0.0
        if param_name.startswith('pca_'):
            idx = int(param_name.split('_')[1])
            return state.pca_coeffs[idx]
        return getattr(state, param_name)

    def calculate_single_cost(self, base_state, x_val, y_val, x_param, y_param):
        test_state = base_state.copy()
        test_state = self._modify_state(test_state, x_param, x_val)
        test_state = self._modify_state(test_state, y_param, y_val)
        
        body_angles = calculate_body_angles(self._meas_global, test_state)
        self.tracker.body_angles = body_angles 
        
        try:
            obj_cost = self.tracker.object_function(
                test_state, self._state_pred, self._P_pred, self._z_flat
            )
            
            total_cost = obj_cost
            if self.penalty_toggle.value:
                penalty = self.tracker.penalty_function(
                    test_state, self._mean_lidar_angle, self._lower_diff, self._upper_diff
                )
                # print(f"Cost: {obj_cost:.4f}, Penalty: {penalty:.4f}")
                total_cost += penalty

            return total_cost
        except Exception:
            return np.nan

    def calculate_single_penalty(self, base_state, x_val, y_val, x_param, y_param):
        test_state = base_state.copy()
        test_state = self._modify_state(test_state, x_param, x_val)
        test_state = self._modify_state(test_state, y_param, y_val)
        
        try:
            penalty = self.tracker.penalty_function(
                test_state, self._mean_lidar_angle, self._lower_diff, self._upper_diff
            )
            return penalty
        except Exception:
            return np.nan

    def _compute_cost_grid(self, x_param, y_param, rng, res, anchor_state):
        cache_key = (x_param, y_param, rng, res, id(anchor_state))
        if cache_key in self._cache_grid_data:
            return self._cache_grid_data[cache_key]

        x_center = self._get_param_value(anchor_state, x_param)
        y_center = self._get_param_value(anchor_state, y_param)
        
        est_x = self._get_param_value(self._state_est, x_param)
        est_y = self._get_param_value(self._state_est, y_param)
        gt_x = self._get_param_value(self._state_gt, x_param)
        gt_y = self._get_param_value(self._state_gt, y_param)

        x_scale = 1.0 if 'yaw' not in x_param else 0.5
        y_scale = 1.0 if 'yaw' not in y_param else 0.5
        
        xs = np.linspace(x_center - rng*x_scale, x_center + rng*x_scale, res)
        ys = np.linspace(y_center - rng*y_scale, y_center + rng*y_scale, res)
        
        grid_z = np.zeros((res, res))
        grid_penalty = np.zeros((res, res))
        base_frozen_state = self._cursor_state
        
        for i, xv in enumerate(xs):
            for j, yv in enumerate(ys):
                grid_z[j, i] = self.calculate_single_cost(base_frozen_state, xv, yv, x_param, y_param)
                grid_penalty[j, i] = self.calculate_single_penalty(base_frozen_state, xv, yv, x_param, y_param)
        
        min_cost = np.nanmin(grid_z)
        min_penalty = np.nanmin(grid_penalty)
        
        # Return RAW grid_z now
        result = (xs, ys, grid_z, min_cost, grid_penalty, min_penalty, est_x, est_y, gt_x, gt_y)
        self._cache_grid_data[cache_key] = result
        return result

    # --- Plot Generators ---

    def _get_scaled_grid(self, grid_z, min_cost):
        if self.cost_scale_toggle.value == 'Logarithmic':
            return np.log1p(np.maximum(0, grid_z - min_cost))
        return grid_z

    def _plot_2d_landscape(self, frame_idx, x_param, y_param, rng, res, center_mode, scale_mode, _trigger, x=None, y=None):
        self._get_frame_data(frame_idx)
        anchor_state = self._get_anchor_state()
        data = self._compute_cost_grid(x_param, y_param, rng, res, anchor_state)
        xs, ys, grid_z, min_cost, _, _, est_x, est_y, gt_x, gt_y = data
        
        # Apply scaling based on toggle
        z_vals = self._get_scaled_grid(grid_z, min_cost)
        
        heatmap = hv.Image((xs, ys, z_vals), kdims=[x_param, y_param]).opts(
            cmap='Viridis_r', title=f'2D Cost ({scale_mode})',
            tools=['hover', 'tap'], active_tools=['tap'], 
            width=600, height=500, colorbar=True, backend='bokeh',
            framewise=True
        )
        
        # Overlays
        est_pt = hv.Points([(est_x, est_y)], label='Estimate').opts(color='cyan', marker='+', size=15, line_width=3, backend='bokeh')
        gt_pt = hv.Points([(gt_x, gt_y)], label='GT').opts(color='black', marker='x', size=15, line_width=3, backend='bokeh')
        
        cur_x = self._get_param_value(self._cursor_state, x_param)
        cur_y = self._get_param_value(self._cursor_state, y_param)
        cur_pt = hv.Points([(cur_x, cur_y)], label='Cursor').opts(color='blue', marker='o', size=10, backend='bokeh')

        return (heatmap * est_pt * gt_pt * cur_pt)

    def _plot_3d_landscape(self, frame_idx, x_param, y_param, rng, res, center_mode, scale_mode, _trigger):
        self._get_frame_data(frame_idx)
        anchor_state = self._get_anchor_state()
        data = self._compute_cost_grid(x_param, y_param, rng, res, anchor_state)
        xs, ys, grid_z, min_cost, _, _, est_x, est_y, gt_x, gt_y = data

        # Apply scaling
        z_vals = self._get_scaled_grid(grid_z, min_cost)

        surface = hv.Surface((xs, ys, z_vals), kdims=[x_param, y_param], vdims=['Cost']).opts(
            cmap='Viridis_r', colorbar=True, width=600, height=500, backend='plotly',
            title=f'3D Surface ({scale_mode})'
        )
        
        # Helper to get scaled Z for points
        def get_z(px, py):
            c = self.calculate_single_cost(self._cursor_state, px, py, x_param, y_param)
            if scale_mode == 'Logarithmic':
                return np.log1p(np.maximum(0, c - min_cost))
            return c

        ez = get_z(est_x, est_y)
        est_pt = hv.Scatter3D([(est_x, est_y, ez)]).opts(color='cyan', size=4, backend='plotly')

        gz = get_z(gt_x, gt_y)
        gt_pt = hv.Scatter3D([(gt_x, gt_y, gz)]).opts(color='black', size=4, backend='plotly')
        
        cur_x = self._get_param_value(self._cursor_state, x_param)
        cur_y = self._get_param_value(self._cursor_state, y_param)
        cz = get_z(cur_x, cur_y)
        cur_pt = hv.Scatter3D([(cur_x, cur_y, cz)]).opts(color='blue', size=5, backend='plotly')

        return (surface * est_pt * gt_pt * cur_pt)

    def _plot_penalty_landscape(self, frame_idx, x_param, y_param, rng, res, center_mode, scale_mode, _trigger, mode='2D'):
        self._get_frame_data(frame_idx)
        anchor_state = self._get_anchor_state()
        data = self._compute_cost_grid(x_param, y_param, rng, res, anchor_state)
        xs, ys, _, _, grid_penalty, min_penalty, est_x, est_y, gt_x, gt_y = data
        
        z_vals = self._get_scaled_grid(grid_penalty, min_penalty)
        
        if mode == '2D':
            heatmap = hv.Image((xs, ys, z_vals), kdims=[x_param, y_param]).opts(
                cmap='Reds', title=f'Penalty ({scale_mode})',
                width=600, height=500, colorbar=True, backend='bokeh',
                framewise=True
            )
            est_pt = hv.Points([(est_x, est_y)], label='Estimate').opts(color='cyan', marker='+', size=15, line_width=3, backend='bokeh')
            return heatmap * est_pt
        else:
            surface = hv.Surface((xs, ys, z_vals), kdims=[x_param, y_param], vdims=['Penalty']).opts(
                cmap='Reds', colorbar=True, width=600, height=500, backend='plotly',
                title=f'3D Penalty ({scale_mode})'
            )
            return surface

    def _plot_geometry(self, frame_idx, _trigger):
        self._get_frame_data(frame_idx)
        
        sel_shape_x, sel_shape_y = compute_estimated_shape_global(self._cursor_state, self.config, self.pca_params)
        g_sel = hv.Curve((sel_shape_y, sel_shape_x), 'East', 'North', label='Cursor (Selected)').opts(color='blue', line_width=3, backend='bokeh')

        est_shape_x, est_shape_y = compute_estimated_shape_global(self._state_est, self.config, self.pca_params)
        g_est = hv.Curve((est_shape_y, est_shape_x), label='Estimate').opts(color='cyan', line_width=2, line_dash='dotted', backend='bokeh')
        
        gt_shape_x, gt_shape_y = compute_exact_vessel_shape_global(self._state_gt, self.config.extent.shape_coords_body)
        g_gt = hv.Curve((gt_shape_y, gt_shape_x), label='GT').opts(color='black', line_dash='dashed', backend='bokeh')
        
        meas = hv.Scatter((self._meas_global.y, self._meas_global.x), label='Lidar').opts(color='red', size=5, backend='bokeh')
        lidar = hv.Scatter([(self.config.lidar.lidar_position[1], self.config.lidar.lidar_position[0])], label='Sensor').opts(color='orange', marker='^', size=10, backend='bokeh')

        return (g_gt * g_est * g_sel * meas * lidar).opts(
            title='Geometry Comparison', 
            data_aspect=1, width=600, height=500, padding=0.2,
            legend_position='top_right', backend='bokeh'
        )

    def view_stats_table(self, frame_idx, x_param, y_param, _trigger):
        self._get_frame_data(frame_idx)
        anchor_state = self._get_anchor_state()
        data = []
        params_to_show = ['x', 'y', 'yaw', 'length', 'width'] + [f'pca_{i}' for i in range(self.config.tracker.N_pca)]
        
        for p in params_to_show:
            val_gt = self._get_param_value(self._state_gt, p)
            val_est = self._get_param_value(self._state_est, p)
            val_cur = self._get_param_value(self._cursor_state, p)
            
            data.append({
                'Parameter': p, 
                'Ground Truth': val_gt, 
                'Estimate': val_est, 
                'Cursor': val_cur, 
                'Error (GT)': val_cur - val_gt
            })
            
        df = pd.DataFrame(data)
        
        def style_rows(row):
            s = pd.Series('', index=row.index)
            p = row['Parameter']
            
            # Active Parameter Highlighting
            if p == x_param or p == y_param:
                s[:] = 'background-color: #fff9c4; font-weight: bold; color: black'
            
            # Modified Cursor Highlighting
            val_anchor = self._get_param_value(anchor_state, p)
            if abs(row['Cursor'] - val_anchor) > 1e-6:
                # Append red color to override any existing color
                s['Cursor'] = (s['Cursor'] or '') + '; color: #d32f2f; font-weight: bold;'
                
            return s

        tabulator = pn.widgets.Tabulator(
            df, disabled=True, width=580, height=400, show_index=False,
            configuration={'columnDefaults': {'headerSort': False}}
        )
        fmt = {'type': 'number', 'func': '0.0000'}
        tabulator.formatters = {'Ground Truth': fmt, 'Estimate': fmt, 'Cursor': fmt, 'Error (GT)': fmt}
        return tabulator.style.apply(style_rows, axis=1)

    def __panel__(self):
        # 2D Map (Bind scale mode)
        dmap_2d = hv.DynamicMap(
            pn.bind(self._plot_2d_landscape, 
                frame_idx=self.frame_slider, x_param=self.x_axis_select, y_param=self.y_axis_select, 
                rng=self.range_slider, res=self.resolution_slider, 
                center_mode=self.center_select, scale_mode=self.cost_scale_toggle, # <--- Bind toggle
                _trigger=self.update_trigger.param.clicks
            ),
            streams=[self.tap_stream]
        )

        dmap_3d = hv.DynamicMap(
            pn.bind(self._plot_3d_landscape, 
                frame_idx=self.frame_slider, x_param=self.x_axis_select, y_param=self.y_axis_select, 
                rng=self.range_slider, res=self.resolution_slider, 
                center_mode=self.center_select, scale_mode=self.cost_scale_toggle, # <--- Bind toggle
                _trigger=self.update_trigger.param.clicks
            )
        )

        dmap_penalty_2d = hv.DynamicMap(
            pn.bind(self._plot_penalty_landscape, 
                frame_idx=self.frame_slider, x_param=self.x_axis_select, y_param=self.y_axis_select, 
                rng=self.range_slider, res=self.resolution_slider, 
                center_mode=self.center_select, scale_mode=self.cost_scale_toggle,
                _trigger=self.update_trigger.param.clicks,
                mode='2D'
            )
        )

        dmap_penalty_3d = hv.DynamicMap(
            pn.bind(self._plot_penalty_landscape, 
                frame_idx=self.frame_slider, x_param=self.x_axis_select, y_param=self.y_axis_select, 
                rng=self.range_slider, res=self.resolution_slider, 
                center_mode=self.center_select, scale_mode=self.cost_scale_toggle,
                _trigger=self.update_trigger.param.clicks,
                mode='3D'
            )
        )

        dmap_geometry = hv.DynamicMap(
            pn.bind(self._plot_geometry, 
                frame_idx=self.frame_slider, _trigger=self.update_trigger.param.clicks
            )
        )

        dmap_stats = pn.bind(self.view_stats_table, 
                             self.frame_slider, self.x_axis_select, self.y_axis_select, 
                             self.update_trigger.param.clicks)

        sidebar = pn.Column(
            pn.pane.Markdown("## Controls"),
            self.frame_slider,
            pn.layout.Divider(),
            "### Axes & Center",
            self.x_axis_select,
            self.y_axis_select,
            self.center_select,
            pn.layout.Divider(),
            "### Interaction Mode",
            self.behavior_toggle,
            self.reset_btn,
            pn.layout.Divider(),
            "### View Options",
            self.cost_scale_toggle,
            self.penalty_toggle,
            self.range_slider,
            self.resolution_slider,
            pn.layout.Divider(),
            "### State Table",
            dmap_stats
        )

        tabs = pn.Tabs(
            ("2D Heatmap", pn.pane.HoloViews(dmap_2d, backend='bokeh')),
            ("3D Surface", pn.pane.HoloViews(dmap_3d, backend='plotly')),
            ("2D Penalty", pn.pane.HoloViews(dmap_penalty_2d, backend='bokeh')),
            ("3D Penalty", pn.pane.HoloViews(dmap_penalty_3d, backend='plotly'))
        )

        main = pn.Row(tabs, pn.pane.HoloViews(dmap_geometry, backend='bokeh'))

        return pn.template.FastListTemplate(
            title="BFGS Cost Landscape Explorer",
            sidebar=[sidebar],
            main=[main],
            sidebar_width=500
        )

if __name__ == "__main__":
    filename = "casestudy_newest_bfgs_43_tracker_lidarstd_0.05.pkl" 
    if not (SIMDATA_PATH / filename).exists():
        files = list(SIMDATA_PATH.glob("*bfgs*.pkl"))
        if files:
            filename = files[0].name
            print(f"Default file not found, using {filename}")
        else:
            print("No BFGS .pkl files found in data/results/")
            sys.exit()

    app = CostLandscapeExplorer(filename)
    pn.serve(app, port=5007)