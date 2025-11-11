from typing import Any, Callable, Optional, Sequence, Union
from matplotlib import pyplot as plt
import numpy as np
from src.senfuslib import TimeSequence, MultiVarGauss, ConsistencyAnalysis, plot_field, scatter_field, ax_config, fig_config, show_consistency
import itertools

from dataclasses import dataclass, field

from src.utils.SimulationResult import SimulationResult
from src.states.states import State_PCA, LidarScan

@dataclass
class PlotterTrackerPCA:
    """
    A plotter class to visualize the results of a tracker simulation
    for the GP-PCA-EOT project.
    """
    sim_result: SimulationResult
    ca: ConsistencyAnalysis
    est_means: TimeSequence[State_PCA] = field(init=False)

    def __post_init__(self):
        # Extract a TimeSequence of the mean of the posterior estimates for easier plotting
        self.est_means = self.ca.x_ests.map(lambda s: s.mean)

    def show(self):
        """
        Generates and displays all plots.
        """
        # self._plot_2d_overview()
        # self._plot_state_errors()
        # self._plot_NEES_by_state_fields()
        self._plot_NEES_all()
        plt.show()

    def _plot_2d_overview(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot ground truth path
        plot_field(ax, self.sim_result.ground_truth_ts, x='y', y='x', label="Ground Truth Path", color='black', linestyle='--')
        
        # Plot estimated path
        plot_field(ax, self.est_means, x='y', y='x', label="Estimated Path", color='C0')

        # Plot measurements from the last frame for context
        last_meas_ts = self.sim_result.measurements_global_ts.get_t(self.sim_result.measurements_global_ts.times[-1])
        if last_meas_ts is not None:
            scatter_field(ax, TimeSequence([(0, last_meas_ts)]), x='y', y='x', s=5, c='red', marker='.', label='Measurements (last frame)')

        ax_config(ax, 'East [m]', 'North [m]', aspect='equal', title='2D Path Overview')
        fig_config(fig, '2D Path Overview')

    def _plot_state_errors(self):
        """
        Plots the estimation error for key state variables with 1-sigma bounds.
        """
        # Define which state components to plot
        fields_to_plot = ['x', 'y', 'yaw', 'length', 'width']
        
        fig, axs = plt.subplots(len(fields_to_plot), 1, figsize=(12, 10), sharex=True)
        
        # Use the senfuslib helper to plot errors with uncertainty bounds
        show_consistency(self.ca, axs_err=axs, fields_err=fields_to_plot, title="State Estimation Errors")
        
        fig_config(fig, 'State Estimation Errors')

    def _plot_NEES_by_state_fields(self):
        """
        Plots the NEES consistency metrics, with one subplot for each state group.
        """
        # Define which state components to group for NEES calculation
        nees_fields = [
            ['x', 'y'],              # Position
            ['yaw'],                 # Heading
            ['length', 'width'],     # Extent
            # ['pca_coeffs']         # Uncomment to see NEES for PCA coefficients
        ]
        
        # 1. Create a figure with one subplot for each entry in nees_fields
        num_plots = len(nees_fields)
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)

        # 2. Call show_consistency, passing the array of axes
        show_consistency(self.ca, 
                         axs_nees=axs, # Pass the entire array of axes
                         fields_nees=nees_fields,
                         title="Filter Consistency (NEES)")
                         
        fig_config(fig, 'Consistency Analysis (NEES)')

    def _plot_NEES_all(self):
        """
        Plots all NEES consistency metrics, including total NEES and subgroups.
        """
        # Define which state components to group for NEES calculation
        nees_fields = [
            None,                    # Total NEES for the entire state vector
            ['x', 'y'],              # Position
            ['yaw'],                 # Heading
            ['length', 'width'],     # Extent
            ['pca_coeffs']           # NEES for all PCA coefficients
        ]
        
        # 1. Create a figure with one subplot for each entry in nees_fields
        num_plots = len(nees_fields)
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)

        # 2. Call show_consistency, passing the array of axes
        show_consistency(self.ca, 
                         axs_nees=axs, # Pass the entire array of axes
                         fields_nees=nees_fields,
                         title="Filter Consistency (NEES)")
                         
        fig_config(fig, 'Consistency Analysis (NEES)')

    def test_export_NEES_all_to_csv(self, filepath: str):
        """
        Exports the NEES data for all state components to a CSV file.
        """
        nees_fields = [
            None                    # Total NEES for the entire state vector
        ]

        nees_data = {}
        for fields in nees_fields:
            data = self.ca.get_nees(fields)
            key = 'total' if fields is None else '_'.join(fields)
            nees_data[key] = data.mahal_dist_tseq.values

        # Combine into a single array for saving
        times = self.ca.x_ests.times
        combined_data = np.column_stack([nees_data[key] for key in nees_data])

        # Save to CSV
        header = ','.join(nees_data.keys())
        np.savetxt(filepath, np.column_stack((times, combined_data)), delimiter=',', header='time,' + header, comments='')

        print(f"NEES data exported to {filepath}")