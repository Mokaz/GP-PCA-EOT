import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from pathlib import Path
from src.tracker.BFGS import BFGS
from src.utils.tools import calculate_body_angles
from src.utils.geometry_utils import compute_estimated_shape_global, compute_exact_vessel_shape_global
from src.dynamics.process_models import Model_PCA_CV
from src.sensors.LidarModel import LidarMeasurementModel


pn.extension('katex', 'plotly', 'tabulator')
hv.extension('bokeh', 'plotly')

class CostLandscapeComponent(pn.viewable.Viewer):
    def __init__(self, sim_result, tracker_config_path, project_root):
        """
        Component for exploring the BFGS cost landscape.
        
        Args:
            sim_result: The loaded SimulationResult object.
            tracker_config_path: Path to tracker config (for PCA params).
            project_root: Path object for resolving relative paths.
        """
        self.sim_result = sim_result
        self.config = self.sim_result.config
        
        # Load PCA params
        self.pca_params = None
        if hasattr(self.config.tracker, 'PCA_parameters_path'):
            self.pca_params = np.load(project_root / self.config.tracker.PCA_parameters_path)

        filter_dyn_model = Model_PCA_CV(
            x_pos_std_dev=self.config.tracker.pos_north_std_dev,
            y_pos_std_dev=self.config.tracker.pos_east_std_dev,
            yaw_std_dev=self.config.tracker.heading_std_dev,
            N_pca=self.config.tracker.N_pca
        )
        
        sensor_model = LidarMeasurementModel(
            lidar_position=np.array(self.config.lidar.lidar_position),
            lidar_std_dev=self.config.tracker.lidar_std_dev,
            pca_mean=self.pca_params['mean'],
            pca_eigenvectors=self.pca_params['eigenvectors'][:, :self.config.tracker.N_pca].real,
            extent_cfg=self.config.extent
        )
        
        self.tracker = BFGS(dynamic_model=filter_dyn_model, lidar_model=sensor_model, config=self.config)

        # --- Internal State ---
        self._current_frame_idx = -1
        self._cache_grid_data = {} 
        self._cursor_state = None 
        
        # --- Widgets ---
        self._internal_frame_val = 0 
        
        self.param_options = ['x', 'y', 'yaw', 'length', 'width'] + [f'pca_{i}' for i in range(self.config.tracker.N_pca)]
        self.x_axis_select = pn.widgets.Select(name='X Axis', options=self.param_options, value='yaw')
        self.y_axis_select = pn.widgets.Select(name='Y Axis', options=self.param_options, value='length')
        
        self.center_select = pn.widgets.RadioButtonGroup(
            name='Center View On', options=['Estimate', 'Ground Truth'], value='Estimate', button_type='primary'
        )
        
        self.cursor_reset_on_change_toggle = pn.widgets.RadioButtonGroup(
            name='Selection Mode', options=['Reset on Change', 'Keep Modifications'], value='Reset on Change', button_type='success'
        )
        
        self.cost_scale_toggle = pn.widgets.RadioButtonGroup(
            name='Cost Scale', options=['Raw', 'Logarithmic'], value='Raw', button_type='primary'
        )

        self.penalty_toggle = pn.widgets.Toggle(name='Include Penalty', value=True, button_type='success')
        
        self.range_slider = pn.widgets.FloatSlider(name='Grid Range (+/-)', start=0.1, end=10.0, value=4.0)
        self.resolution_slider = pn.widgets.IntSlider(name='Grid Resolution', start=10, end=100, value=30)
        
        self.reset_btn = pn.widgets.Button(name='Reset Cursor to Anchor', button_type='warning')
        self.reset_btn.on_click(self.reset_cursor)

        self.iterate_player = pn.widgets.Player(name='Iterate', start=0, end=0, value=0, loop_policy='once', visible=False, width=200, visible_loop_options=[], show_value=True)
        self.show_iterates_toggle = pn.widgets.Toggle(name='Show Iterates', value=False, button_type='default', width=100)
        self.toggle_cursor_follow_iterate = pn.widgets.Checkbox(name='Cursor Follows Iterates', value=True, align='center')

        self.show_penalty_plots_toggle = pn.widgets.Toggle(name='Show Penalty Plots', value=False, button_type='default', width=150)

        # zoom controls
        self.zoom_to_iterates_btn = pn.widgets.Button(name='Zoom to Iterates', button_type='light')
        self.zoom_to_iterates_btn.on_click(self._zoom_to_iterates)
        self.keep_zoom_toggle = pn.widgets.Checkbox(name='Keep Zoom Level', value=False, align='center')

        # --- Tap Stream & Trigger ---
        self.tap_stream = hv.streams.Tap(x=None, y=None)
        self.range_xy_stream = hv.streams.RangeXY()
        self.update_trigger = pn.widgets.Button(name='Refresh values', button_type='primary')
        self.auto_refresh_toggle = pn.widgets.Checkbox(name='Auto Refresh', value=True, align='center')

        # Setup Watchers
        self.x_axis_select.param.watch(self._on_axis_change, 'value')
        self.y_axis_select.param.watch(self._on_axis_change, 'value')
        self.center_select.param.watch(self.reset_cursor, 'value')
        self.cost_scale_toggle.param.watch(lambda e: setattr(self.update_trigger, 'clicks', self.update_trigger.clicks + 1), 'value')
        self.penalty_toggle.param.watch(lambda e: setattr(self.update_trigger, 'clicks', self.update_trigger.clicks + 1), 'value')
        self.show_iterates_toggle.param.watch(self._on_iterate_view_change, 'value')
        self.iterate_player.param.watch(self._on_iterate_change, 'value')
        self.toggle_cursor_follow_iterate.param.watch(self._on_iterate_change, 'value')
        self.show_penalty_plots_toggle.param.watch(lambda e: setattr(self.update_trigger, 'clicks', self.update_trigger.clicks + 1), 'value')

        super().__init__()
    
    def update_frame(self, frame_idx):
        """Updates the component to display data for the given frame index."""
        if frame_idx != self._current_frame_idx:
            # Check iterates
            try:
                res = self.sim_result.tracker_results_ts.values[frame_idx]
                has_iterates = getattr(res, 'iterates', None) is not None and len(res.iterates) > 0
                
                if has_iterates:
                    self.show_iterates_toggle.disabled = False
                    self.show_iterates_toggle.button_type = 'success'
                    self.show_iterates_toggle.name = 'Show Iterates'
                    self.iterate_player.end = len(res.iterates) - 1
                    self.iterate_player.value = 0
                else:
                    self.show_iterates_toggle.value = False
                    self.show_iterates_toggle.disabled = True
                    self.show_iterates_toggle.button_type = 'default'
                    self.show_iterates_toggle.name = 'No Iterates'
            except Exception:
                self.show_iterates_toggle.disabled = True
                self.show_iterates_toggle.value = False
                self.show_iterates_toggle.button_type = 'default'
                self.show_iterates_toggle.name = 'No Iterates'

            self._current_frame_idx = frame_idx
            self._cache_grid_data = {}

            if self.auto_refresh_toggle.value:
                self.update_trigger.clicks += 1
                if self.cursor_reset_on_change_toggle.value == 'Reset on Change':
                    self.reset_cursor()

    # --- Event Handling ---

    def _on_iterate_view_change(self, event):
        self.iterate_player.visible = event.new
        self.reset_cursor()

    def _on_iterate_change(self, event):
        if not self.show_iterates_toggle.value:
            return

        if self.toggle_cursor_follow_iterate.value:
            try:
                res = self.sim_result.tracker_results_ts.values[self._current_frame_idx]
                if getattr(res, 'iterates', None) is not None and len(res.iterates) > 0:
                    idx = min(max(0, self.iterate_player.value), len(res.iterates)-1)
                    self._cursor_state = res.iterates[idx]
            except Exception:
                pass
        
        self.update_trigger.clicks += 1

    @pn.depends('tap_stream.x', 'tap_stream.y', watch=True)
    def _on_tap(self, *events):
        x, y = self.tap_stream.x, self.tap_stream.y
        if x is None or y is None: return
        
        x_param = self.x_axis_select.value
        y_param = self.y_axis_select.value
        
        self._cursor_state = self._modify_state(self._cursor_state, x_param, x)
        self._cursor_state = self._modify_state(self._cursor_state, y_param, y)
        
        self.update_trigger.clicks += 1

    def _on_axis_change(self, event):
        if self.x_axis_select.value == self.y_axis_select.value:
            fallback = next(p for p in self.param_options if p != event.new)
            if event.obj is self.x_axis_select:
                self.y_axis_select.value = fallback
            else:
                self.x_axis_select.value = fallback
            return

        if self.cursor_reset_on_change_toggle.value == 'Reset on Change':
            self.reset_cursor()
        else:
            self.update_trigger.clicks += 1

    def reset_cursor(self, event=None):
        if self._current_frame_idx == -1: return
        self._cursor_state = self._get_anchor_state().copy()
        self.update_trigger.clicks += 1

    def _zoom_to_iterates(self, event=None):
        if self._current_frame_idx == -1: return
        
        try:
            res = self.sim_result.tracker_results_ts.values[self._current_frame_idx]
            if getattr(res, 'iterates', None) is None or len(res.iterates) == 0:
                return
            
            x_param = self.x_axis_select.value
            y_param = self.y_axis_select.value
            
            self._get_frame_data()
            anchor_state = self._get_anchor_state()
            c_x = self._get_param_value(anchor_state, x_param)
            c_y = self._get_param_value(anchor_state, y_param)
            
            max_d = 0.0
            for s in res.iterates:
                val_x = self._get_param_value(s, x_param)
                val_y = self._get_param_value(s, y_param)
                d = max(abs(val_x - c_x), abs(val_y - c_y))
                if d > max_d: max_d = d
            
            # Check Est as well to keep them in view
            est_x = self._get_param_value(self._state_est, x_param)
            est_y = self._get_param_value(self._state_est, y_param)
            max_d = max(max_d, abs(est_x - c_x), abs(est_y - c_y))
            
            if max_d > 0:
                # Add padding
                new_rng = max_d * 1.5
                new_rng = max(new_rng, 0.1) # Minimum range
                
                self.keep_zoom_toggle.value = False
                self.range_slider.value = float(new_rng)
        except Exception:
            pass

    # --- Data Logic ---

    def _get_frame_data(self):
        # Uses self._current_frame_idx set by update_frame
        idx = self._current_frame_idx
        if idx >= len(self.sim_result.tracker_results_ts.values): return

        res = self.sim_result.tracker_results_ts.values[idx]
        self._state_est = res.state_posterior.mean
        self._state_pred = res.state_prior.mean
        self._P_pred = res.state_prior.cov
        self._state_gt = self.sim_result.ground_truth_ts.values[idx]
        self._meas_global = self.sim_result.measurements_global_ts.values[idx]
        self._z_flat = self._meas_global.flatten('F')
        
        lidar_pos = self.config.lidar.lidar_position
        dx = self._meas_global.x - lidar_pos[0]
        dy = self._meas_global.y - lidar_pos[1]
        angles = np.arctan2(dy, dx)
        
        from src.utils.tools import compute_angle_range
        self._lower_diff, self._upper_diff, self._mean_lidar_angle = compute_angle_range(angles)
        
        if self._cursor_state is None:
            self._cursor_state = self._get_anchor_state().copy()

    def _get_anchor_state(self):
        if not hasattr(self, '_state_est'): self._get_frame_data()
        return self._state_est if self.center_select.value == 'Estimate' else self._state_gt

    def _modify_state(self, base_state, param_name, value):
        if base_state is None: 
            self._get_frame_data()
            base_state = self._get_anchor_state()
            
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
        
        total_cost = self._calculate_cost_for_state(test_state)
        return total_cost

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
    
    def _calculate_cost_for_state(self, state):
        """Calculate the total cost for a given complete state."""
        try:
            body_angles = calculate_body_angles(self._meas_global, state)
            self.tracker.body_angles = body_angles
            
            obj_cost = self.tracker.objective_function(
                state, self._state_pred, self._P_pred, self._z_flat
            )
            
            total_cost = obj_cost
            if self.penalty_toggle.value:
                penalty = self.tracker.penalty_function(
                    state, self._mean_lidar_angle, self._lower_diff, self._upper_diff
                )
                total_cost += penalty
            return total_cost
        except Exception:
            return np.nan

    def _compute_cost_grid(self, x_param, y_param, rng, res, anchor_state):
        cache_key = (x_param, y_param, rng, res, id(anchor_state), id(self._cursor_state), self._current_frame_idx, self.penalty_toggle.value, self.show_penalty_plots_toggle.value)
        if cache_key in self._cache_grid_data:
            return self._cache_grid_data[cache_key]

        x_center = self._get_param_value(anchor_state, x_param)
        y_center = self._get_param_value(anchor_state, y_param)
        
        est_x = self._get_param_value(self._state_est, x_param)
        est_y = self._get_param_value(self._state_est, y_param)
        gt_x = self._get_param_value(self._state_gt, x_param)
        gt_y = self._get_param_value(self._state_gt, y_param)
        
        # ... existing scaling code ...
        x_scale = 1.0 if 'yaw' not in x_param else 0.5
        y_scale = 1.0 if 'yaw' not in y_param else 0.5
        
        xs = np.linspace(x_center - rng*x_scale, x_center + rng*x_scale, res)
        ys = np.linspace(y_center - rng*y_scale, y_center + rng*y_scale, res)
        
        grid_z = np.zeros((res, res))
        grid_penalty = np.zeros((res, res))
        base_frozen_state = self._cursor_state
        
        calc_penalty = self.show_penalty_plots_toggle.value

        for i, xv in enumerate(xs):
            for j, yv in enumerate(ys):
                grid_z[j, i] = self.calculate_single_cost(base_frozen_state, xv, yv, x_param, y_param)
                if calc_penalty:
                    grid_penalty[j, i] = self.calculate_single_penalty(base_frozen_state, xv, yv, x_param, y_param)
        
        min_cost = np.nanmin(grid_z)
        min_penalty = np.nanmin(grid_penalty) if calc_penalty else 0.0

        result = (xs, ys, grid_z, min_cost, grid_penalty, min_penalty, est_x, est_y, gt_x, gt_y)
        self._cache_grid_data[cache_key] = result
        return result

    def _get_iterates_trajectory(self, x_param, y_param, val_func=None):
        if not self.show_iterates_toggle.value:
            return None
        
        try:
            res = self.sim_result.tracker_results_ts.values[self._current_frame_idx]
            if getattr(res, 'iterates', None) is None or len(res.iterates) == 0:
                return None
            
            xs = [self._get_param_value(s, x_param) for s in res.iterates]
            ys = [self._get_param_value(s, y_param) for s in res.iterates]
            
            if val_func:
                vs = [val_func(s) for s in res.iterates]
                return xs, ys, vs
            return xs, ys
        except Exception:
            return None

    # --- Plot Generators ---

    def _get_scaled_grid(self, grid_z, min_cost):
        if self.cost_scale_toggle.value == 'Logarithmic':
            return np.log1p(np.maximum(0, grid_z - min_cost))
        return grid_z

    def _plot_2d_landscape(self, x_param, y_param, rng, res, center_mode, scale_mode, _trigger, x=None, y=None, x_range=None, y_range=None):
        self._get_frame_data()
        anchor_state = self._get_anchor_state()
        data = self._compute_cost_grid(x_param, y_param, rng, res, anchor_state)
        xs, ys, grid_z, min_cost, _, _, est_x, est_y, gt_x, gt_y = data
        
        z_vals = self._get_scaled_grid(grid_z, min_cost)
        
        heatmap = hv.Image((xs, ys, z_vals), kdims=[x_param, y_param]).opts(
            cmap='Viridis_r', title=f'2D Cost ({scale_mode})',
            tools=['hover', 'tap'], active_tools=['tap'], 
            width=600, height=500, colorbar=True, backend='bokeh',
            framewise=True
        )
        
        est_pt = hv.Points([(est_x, est_y)], label='Estimate').opts(color='cyan', marker='+', size=15, line_width=3, backend='bokeh')
        gt_pt = hv.Points([(gt_x, gt_y)], label='GT').opts(color='black', marker='x', size=15, line_width=3, backend='bokeh')
        
        cur_x = self._get_param_value(self._cursor_state, x_param)
        cur_y = self._get_param_value(self._cursor_state, y_param)
        cur_pt = hv.Points([(cur_x, cur_y)], label='Cursor').opts(color='blue', marker='o', size=10, backend='bokeh')

        plot = heatmap * est_pt * gt_pt * cur_pt

        traj_data = self._get_iterates_trajectory(x_param, y_param)
        if traj_data:
            tx, ty = traj_data
            traj_curve = hv.Curve((tx, ty)).opts(color='white', line_width=1, alpha=0.7, backend='bokeh')
            traj_pts = hv.Scatter((tx, ty)).opts(color='white', size=4, alpha=0.9, backend='bokeh')
            plot = plot * traj_curve * traj_pts
        
        if self.keep_zoom_toggle.value and x_range and y_range:
            plot = plot.opts(xlim=x_range, ylim=y_range)

        return plot

    def _plot_3d_landscape(self, x_param, y_param, rng, res, center_mode, scale_mode, _trigger):
        self._get_frame_data()
        anchor_state = self._get_anchor_state()
        data = self._compute_cost_grid(x_param, y_param, rng, res, anchor_state)
        xs, ys, grid_z, min_cost, _, _, est_x, est_y, gt_x, gt_y = data

        z_vals = self._get_scaled_grid(grid_z, min_cost)

        surface = hv.Surface((xs, ys, z_vals), kdims=[x_param, y_param], vdims=['Cost']).opts(
            cmap='Viridis_r', colorbar=True, width=600, height=500, backend='plotly',
            title=f'3D Surface ({scale_mode})'
        )
        
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

        plot = surface * est_pt * gt_pt * cur_pt

        # Iterates
        def cost_func(s):
            c = self._calculate_cost_for_state(s)
            if scale_mode == 'Logarithmic':
                return np.log1p(np.maximum(0, c - min_cost))
            return c

        traj_data = self._get_iterates_trajectory(x_param, y_param, val_func=cost_func)
        if traj_data:
            tx, ty, tz = traj_data
            # hv.Path3D is not always available or behaves differently across backends. 
            # Using Scatter3D with mode='lines+markers' is a Plotly specific tweak.
            traj_pts = hv.Scatter3D((tx, ty, tz)).opts(
                color='white', size=3, backend='plotly'
            )
            # To get lines in Plotly via HoloViews, we often need to use a specific Option.
            # Unfortunately, pure HoloViews support for 3D lines is spotty.
            # We will try passing the mode trace object directly if possible, or just markers.
            # However, typically Scatter3D in HoloViews maps to Scatter3d in plotly.
            # Trying to force lines via opts:
            traj_lines = hv.Scatter3D((tx, ty, tz)).opts(
                color='white', size=1, backend='plotly', 
            )
            # We can't easily force 'mode'='lines' via simple opts in all HV versions without using hooks. 
            # For now, let's just plot dense points or try 'line' marker if available? No.
            # We will stick to points for 3D to be safe, or try one experimental hook.
            plot = plot * traj_pts

        return plot

    def _plot_penalty_landscape(self, x_param, y_param, rng, res, center_mode, scale_mode, _trigger, mode='2D'):
        self._get_frame_data()
        anchor_state = self._get_anchor_state()
        data = self._compute_cost_grid(x_param, y_param, rng, res, anchor_state)
        xs, ys, _, _, grid_penalty, min_penalty, est_x, est_y, gt_x, gt_y = data
        
        z_vals = self._get_scaled_grid(grid_penalty, min_penalty)

        # Helper for penalty scaling
        def pen_func(s):
            try:
                p = self.tracker.penalty_function(s, self._mean_lidar_angle, self._lower_diff, self._upper_diff)
                return self._get_scaled_grid(p, min_penalty)
            except:
                return 0.0

        if mode == '2D':
            heatmap = hv.Image((xs, ys, z_vals), kdims=[x_param, y_param]).opts(
                cmap='Reds', title=f'Penalty ({scale_mode})',
                width=600, height=500, colorbar=True, backend='bokeh',
                framewise=True
            )
            est_pt = hv.Points([(est_x, est_y)], label='Estimate').opts(color='cyan', marker='+', size=15, line_width=3, backend='bokeh')
            
            plot = heatmap * est_pt
            
            traj_data = self._get_iterates_trajectory(x_param, y_param)
            if traj_data:
                tx, ty = traj_data
                traj_curve = hv.Curve((tx, ty)).opts(color='black', line_width=1, alpha=0.5, backend='bokeh')
                traj_pts = hv.Scatter((tx, ty)).opts(color='black', size=4, alpha=0.7, backend='bokeh')
                plot = plot * traj_curve * traj_pts
            
            return plot
        else:
            surface = hv.Surface((xs, ys, z_vals), kdims=[x_param, y_param], vdims=['Penalty']).opts(
                cmap='Reds', colorbar=True, width=600, height=500, backend='plotly',
                title=f'3D Penalty ({scale_mode})'
            )
            
            # Add Estimate point to ensure we return an Overlay (fixing DynamicMap type consistency)
            ez = pen_func(self._state_est)
            est_pt = hv.Scatter3D([(est_x, est_y, ez)]).opts(color='cyan', size=4, backend='plotly')
             
            plot = surface * est_pt
            
            traj_data = self._get_iterates_trajectory(x_param, y_param, val_func=pen_func)
            if traj_data:
                tx, ty, tz = traj_data
                traj_pts = hv.Scatter3D((tx, ty, tz)).opts(color='black', size=3, backend='plotly')
                plot = plot * traj_pts
                
            return plot

    def _plot_geometry(self, _trigger):
        self._get_frame_data()
        
        sel_shape_x, sel_shape_y = compute_estimated_shape_global(self._cursor_state, self.config, self.pca_params)
        g_sel = hv.Curve((sel_shape_y, sel_shape_x), 'East', 'North', label='Cursor (Selected)').opts(color='blue', line_width=3, backend='bokeh')

        est_shape_x, est_shape_y = compute_estimated_shape_global(self._state_est, self.config, self.pca_params)
        g_est = hv.Curve((est_shape_y, est_shape_x), label='Estimate').opts(color='cyan', line_width=2, line_dash='dotted', backend='bokeh')
        
        gt_shape_x, gt_shape_y = compute_exact_vessel_shape_global(self._state_gt, self.config.extent.shape_coords_body)
        g_gt = hv.Curve((gt_shape_y, gt_shape_x), label='GT').opts(color='black', line_dash='dashed', backend='bokeh')
        
        meas = hv.Scatter((self._meas_global.y, self._meas_global.x), label='Lidar').opts(color='red', size=5, backend='bokeh')
        lidar = hv.Scatter([(self.config.lidar.lidar_position[1], self.config.lidar.lidar_position[0])], label='Sensor').opts(color='orange', marker='^', size=10, backend='bokeh')

        plot = g_gt * g_est * g_sel * meas * lidar
        
        # Plot iterate trajectory (East vs North -> y vs x in state parameters)
        traj_data = self._get_iterates_trajectory('y', 'x') # 'y' is state.pos[1] (East), 'x' is state.pos[0] (North)
        if traj_data:
            ty, tx = traj_data # returns x_param vals, y_param vals -> East, North
            traj_curve = hv.Curve((ty, tx)).opts(color='blue', line_width=1, alpha=0.5, backend='bokeh')
            traj_pts = hv.Scatter((ty, tx)).opts(color='blue', size=4, alpha=0.6, backend='bokeh')
            plot = plot * traj_curve * traj_pts

        return plot.opts(
            title='Geometry Comparison', 
            data_aspect=1, width=600, height=500, padding=0.2,
            legend_position='top_right', backend='bokeh'
        )

    def view_stats_table(self, x_param, y_param, _trigger, include_penalty=None):
        self._get_frame_data()
        anchor_state = self._get_anchor_state()
        
        # --- Cost Table ---
        cost_gt = self._calculate_cost_for_state(self._state_gt)
        cost_est = self._calculate_cost_for_state(self._state_est)
        cost_cur = self._calculate_cost_for_state(self._cursor_state)
        
        cost_data = [{
            'Metric': 'Total Cost',
            'Ground Truth': cost_gt,
            'Estimate': cost_est,
            'Cursor': cost_cur,
            'Error (GT)': cost_cur - cost_gt,
        }]
        
        df_cost = pd.DataFrame(cost_data)
        
        # --- Parameters Table ---
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
            
        df_params = pd.DataFrame(data)
        
        # --- Styling ---
        
        def style_params(row):
            s = pd.Series('', index=row.index)
            p = row['Parameter']
            
            if p == x_param or p == y_param:
                s[:] = 'background-color: #fff9c4; font-weight: bold; color: black'
            
            # Highlight if modified
            val_anchor = self._get_param_value(anchor_state, p)
            if abs(row['Cursor'] - val_anchor) > 1e-6:
                s['Cursor'] = (s['Cursor'] or '') + '; color: #d32f2f; font-weight: bold;'
            return s

        def style_cost(row):
            s = pd.Series('', index=row.index)
            # Default style for cost row
            s[:] = 'background-color: #e3f2fd; font-weight: bold; color: black'
            
            # Highlight if modified from anchor
            cost_anchor = self._calculate_cost_for_state(anchor_state)
            if abs(row['Cursor'] - cost_anchor) > 1e-6:
                s['Cursor'] = (s['Cursor'] or '') + '; color: #d32f2f; font-weight: bold;'
            return s

        # Create Tabulators
        fmt = {'type': 'number', 'func': '0.0000'}
        
        tab_cost = pn.widgets.Tabulator(
            df_cost, disabled=True, width=580, height=80, show_index=False,
            configuration={'columnDefaults': {'headerSort': False}},
            formatters={'Ground Truth': fmt, 'Estimate': fmt, 'Cursor': fmt, 'Error (GT)': fmt}
        )
        
        tab_params = pn.widgets.Tabulator(
            df_params, disabled=True, width=580, height=350, show_index=False,
            configuration={'columnDefaults': {'headerSort': False}},
            formatters={'Ground Truth': fmt, 'Estimate': fmt, 'Cursor': fmt, 'Error (GT)': fmt}
        )
        
        return pn.Column(
            tab_cost.style.apply(style_cost, axis=1),
            tab_params.style.apply(style_params, axis=1)
        )

    def _get_tabs(self, show_penalty, keep_zoom):
        streams_2d = [self.tap_stream]
        if keep_zoom:
            streams_2d.append(self.range_xy_stream)

        dmap_2d = hv.DynamicMap(
            pn.bind(self._plot_2d_landscape, 
                x_param=self.x_axis_select, y_param=self.y_axis_select, 
                rng=self.range_slider, res=self.resolution_slider, 
                center_mode=self.center_select, scale_mode=self.cost_scale_toggle,
                _trigger=self.update_trigger.param.clicks
            ),
            streams=streams_2d
        )

        dmap_3d = hv.DynamicMap(
            pn.bind(self._plot_3d_landscape, 
                x_param=self.x_axis_select, y_param=self.y_axis_select, 
                rng=self.range_slider, res=self.resolution_slider, 
                center_mode=self.center_select, scale_mode=self.cost_scale_toggle,
                _trigger=self.update_trigger.param.clicks
            )
        )

        tabs_list = [
            ("2D Heatmap", pn.pane.HoloViews(dmap_2d, backend='bokeh')),
            ("3D Surface", pn.pane.HoloViews(dmap_3d, backend='plotly'))
        ]

        if show_penalty:
            dmap_penalty_2d = hv.DynamicMap(
                pn.bind(self._plot_penalty_landscape, 
                    x_param=self.x_axis_select, y_param=self.y_axis_select, 
                    rng=self.range_slider, res=self.resolution_slider, 
                    center_mode=self.center_select, scale_mode=self.cost_scale_toggle,
                    _trigger=self.update_trigger.param.clicks,
                    mode='2D'
                )
            )

            dmap_penalty_3d = hv.DynamicMap(
                pn.bind(self._plot_penalty_landscape, 
                    x_param=self.x_axis_select, y_param=self.y_axis_select, 
                    rng=self.range_slider, res=self.resolution_slider, 
                    center_mode=self.center_select, scale_mode=self.cost_scale_toggle,
                    _trigger=self.update_trigger.param.clicks,
                    mode='3D'
                )
            )
            tabs_list.append(("2D Penalty", pn.pane.HoloViews(dmap_penalty_2d, backend='bokeh')))
            tabs_list.append(("3D Penalty", pn.pane.HoloViews(dmap_penalty_3d, backend='plotly')))

        return pn.Tabs(*tabs_list)

    def __panel__(self):
        dmap_geometry = hv.DynamicMap(
            pn.bind(self._plot_geometry, 
                _trigger=self.update_trigger.param.clicks
            )
        )

        dmap_stats = pn.bind(self.view_stats_table, 
                             self.x_axis_select, self.y_axis_select, 
                             self.update_trigger.param.clicks,
                             include_penalty=self.penalty_toggle)

        sidebar = pn.Column(
            pn.pane.Markdown("### Cost Landscape Config"),
            pn.Row(self.update_trigger, self.auto_refresh_toggle),
            pn.layout.Divider(),
            "### Cost and State Tables",
            dmap_stats,
            pn.layout.Divider(),
            "### Iterate Visualization",
            pn.Row(self.show_iterates_toggle, self.zoom_to_iterates_btn, self.toggle_cursor_follow_iterate),
            self.iterate_player,
            pn.layout.Divider(),
            "### View Options",
            self.cost_scale_toggle,
            self.penalty_toggle,
            self.show_penalty_plots_toggle,
            pn.Row(self.range_slider, self.keep_zoom_toggle, align='end'),
            self.resolution_slider,

        )
        view_controls = pn.Column(
            "### View Controls",
            self.x_axis_select,
            self.y_axis_select,
            self.center_select,
            pn.layout.Divider(),
            "### Cursor Interaction Mode",
            self.cursor_reset_on_change_toggle,
            self.reset_btn,
            pn.layout.Divider(),
        )

        tabs_dynamic = pn.bind(self._get_tabs, self.show_penalty_plots_toggle, self.keep_zoom_toggle)

        # Main Layout
        return pn.Row(
            sidebar,
            pn.Column(tabs_dynamic, pn.pane.HoloViews(dmap_geometry, backend='bokeh')),
            view_controls
        )