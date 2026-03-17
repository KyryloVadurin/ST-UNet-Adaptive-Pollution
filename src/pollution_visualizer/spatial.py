import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.interpolate import Rbf
from .config import VisConfig

class SpatialPlotter:
    """
    Handles 2D and 3D spatial visualizations including interpolation and coverage analysis.
    """
    def __init__(self, config: VisConfig):
        self.cfg = config

    def plot_snapshot(self, gt_step, sensor_coords=None, wind_vec=None, title="Spatial State"):
        """
        Renders a static 2D heatmap of the pollution state at a specific time step.
        """
        fig, ax = plt.subplots(figsize=self.cfg.figsize_square, dpi=self.cfg.dpi)
        im = ax.imshow(gt_step, cmap=self.cfg.cmap_main, origin='lower')
        
        # Sensor location overlay logic
        if sensor_coords is not None:
            ax.scatter(sensor_coords[:, 1], sensor_coords[:, 0], c=self.cfg.sensor_color, s=10, alpha=0.5)
        
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        return fig

    def plot_interpolation(self, sensor_coords, sensor_values, grid_shape):
        """
        Reconstructs the full map from sparse sensor data using Radial Basis Function (Rbf) interpolation.
        """
        x, y = sensor_coords[:, 0], sensor_coords[:, 1]

        # Interpolation execution with smoothness parameter to prevent matrix ill-conditioning
        try:
            rbf = Rbf(x, y, sensor_values, function='multiquadric', smooth=0.1)
            gx, gy = np.mgrid[0:grid_shape[0], 0:grid_shape[1]]
            interp_data = rbf(gx, gy)
        except Exception:
            # Failure fallback logic
            interp_data = np.zeros(grid_shape) 

        fig, ax = plt.subplots(figsize=self.cfg.figsize_square, dpi=self.cfg.dpi)
        im = ax.imshow(interp_data, cmap=self.cfg.cmap_main, origin='lower')
        plt.colorbar(im, ax=ax)
        ax.set_title("Interpolated Map (Smooth Rbf)")
        return fig

    def plot_3d_surface(self, gt_step, title="3D Surface"):
        """
        Generates a static 3D surface plot using Matplotlib's 3D toolkit.
        """
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8), dpi=self.cfg.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Grid coordinate generation
        X, Y = np.meshgrid(np.arange(gt_step.shape[1]), np.arange(gt_step.shape[0]))
        
        # Surface rendering logic
        surf = ax.plot_surface(X, Y, gt_step, cmap=self.cfg.cmap_main)
        ax.set_title(title)
        return fig

    def plot_cumulative_dose(self, gt_history):
        """
        Calculates and visualizes the total pollution exposure (integral over time).
        """
        dose = np.sum(gt_history, axis=0)
        fig, ax = plt.subplots(figsize=self.cfg.figsize_square, dpi=self.cfg.dpi)
        im = ax.imshow(dose, cmap='YlOrRd', origin='lower')
        plt.colorbar(im, ax=ax, label="Dose")
        ax.set_title("Cumulative Exposure")
        return fig

    def plot_residual_map(self, gt_step, sensor_coords, sensor_values):
        """
        Visualizes the difference between ground truth and interpolation to identify blind spots.
        """
        x, y = sensor_coords[:, 0], sensor_coords[:, 1]
        
        # Baseline interpolation logic for comparison
        rbf = Rbf(x, y, sensor_values, function='linear', smooth=0.1)
        gx, gy = np.mgrid[0:gt_step.shape[0], 0:gt_step.shape[1]]
        interp = rbf(gx, gy)
        
        # Residual calculation logic
        res = gt_step - interp
        fig, ax = plt.subplots(figsize=self.cfg.figsize_square, dpi=self.cfg.dpi)
        
        # Divergent colormap centered at zero
        im = ax.imshow(res, cmap='RdBu_r', origin='lower', 
                       vmin=-np.max(np.abs(res)), vmax=np.max(np.abs(res)))
        plt.colorbar(im, ax=ax)
        ax.set_title("Residuals (Blind Spots)")
        return fig

    def plot_coverage_voronoi(self, sensor_coords, sensor_values, grid_size):
        """
        Uses Voronoi tessellation to represent the area of influence for each physical sensor.
        """
        from scipy.spatial import Voronoi, voronoi_plot_2d
        fig, ax = plt.subplots(figsize=self.cfg.figsize_square, dpi=self.cfg.dpi)
        
        # Voronoi calculation and plotting
        vor = Voronoi(sensor_coords)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_alpha=0.3)
        
        # Colored scatter based on sensor intensity
        ax.scatter(sensor_coords[:, 1], sensor_coords[:, 0], c=sensor_values, cmap=self.cfg.cmap_main)
        ax.set_xlim(0, grid_size[1])
        ax.set_ylim(0, grid_size[0])
        return fig