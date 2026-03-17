import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from .config import VisConfig
from .spatial import SpatialPlotter
from .temporal import TemporalPlotter
from .statistics import StatsPlotter
from .animation import AnimationPlotter
from .interactive import InteractivePlotter

class DataVisualizer:
    """
    Main orchestration class for data analysis and visualization.
    Integrates spatial, temporal, statistical, and interactive plotting components.
    """
    def __init__(self, vis_config=None):
        # Component initialization and configuration assignment
        self.cfg = vis_config or VisConfig()
        self.spatial = SpatialPlotter(self.cfg)
        self.temporal = TemporalPlotter(self.cfg)
        self.stats = StatsPlotter(self.cfg)
        self.animator = AnimationPlotter(self.cfg)
        self.interactive = InteractivePlotter(self.cfg)

    def analyze(self, file_path, scenario_idx=0, layout_idx=0, 
                plots=['spatial', 'trends'], 
                time_range=None):
        """
        Loads data from HDF5 and dispatches plotting commands based on provided keys.
        """
        with h5py.File(file_path, 'r') as f:
            # Scenario selection and temporal slicing logic
            sc_key = list(f.keys())[scenario_idx]
            grp = f[sc_key]
            ts = slice(*time_range) if time_range else slice(None)

            # Data extraction for ground truth, sensor readings, and metadata
            gt = grp['ground_truth'][ts]
            readings = grp['sensor_readings'][layout_idx, ts]
            coords = grp['sensor_coords'][layout_idx]
            meta = json.loads(grp.attrs['config'])
            wind = meta.get('scenario_initial_wind', [0, 0])

            print(f"--- Analysis Dashboard: {sc_key} ---")

            # Spatial plotter dispatch logic
            if 'spatial' in plots: self.spatial.plot_snapshot(gt[-1], coords, wind)
            if 'voronoi' in plots: self.spatial.plot_coverage_voronoi(coords, readings[-1], gt.shape[1:])
            if 'interp' in plots: self.spatial.plot_interpolation(coords, readings[-1], gt.shape[1:])
            if 'dose' in plots: self.spatial.plot_cumulative_dose(gt)
            if 'residuals' in plots: self.spatial.plot_residual_map(gt[-1], coords, readings[-1])

            # Temporal plotter dispatch logic
            if 'trends' in plots: self.temporal.plot_trends(gt)
            if 'ridge' in plots: self.temporal.plot_ridge_joyplot(gt)
            if 'diurnal' in plots: self.temporal.plot_diurnal_analysis(gt)
            if 'lag' in plots: self.temporal.plot_lag_analysis(readings)

            # Statistical plotter dispatch logic
            if 'wind_rose' in plots: self.stats.plot_pollution_wind_rose(gt, np.array([wind]*len(gt)))
            if 'corr' in plots: self.stats.plot_sensor_correlation(readings)
            if 'errors' in plots: self.stats.plot_error_dist(gt, readings, coords)

            # 3D and Interactive plotter dispatch logic
            if '3d' in plots: self.spatial.plot_3d_surface(gt[-1])
            if 'interactive_3d' in plots: self.interactive.plot_3d_interactive(gt[-1])
            if 'st_cube' in plots: self.interactive.plot_space_time_cube(gt)

            plt.show()

    def animate(self, file_path, scenario_idx=0, output_path="pollution.gif", fps=10):
        """
        Extracts historical ground truth data and triggers GIF generation.
        """
        with h5py.File(file_path, 'r') as f:
            sc_key = list(f.keys())[scenario_idx]
            gt = f[sc_key]['ground_truth'][:]
            self.animator.create_gif(gt, output_path, fps)