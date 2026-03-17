import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .config import VisConfig

class StatsPlotter:
    """
    Handles statistical visualization of pollution data, including wind relationships,
    sensor correlations, and measurement error distributions.
    """
    def __init__(self, config: VisConfig):
        self.cfg = config

    def plot_pollution_wind_rose(self, gt_history, wind_vectors):
        """
        Generates a Wind Rose plot correlating mean pollution concentration with wind direction.
        Includes handling for empty directional bins.
        """
        # Directional angle calculation from wind vectors
        angles = np.arctan2(wind_vectors[:, 1], wind_vectors[:, 0]) * 180 / np.pi
        mean_pollution = np.mean(gt_history, axis=(1, 2))

        # Sector binning logic (8 sectors of 45 degrees)
        bins = np.linspace(-180, 180, 9)
        bin_indices = np.digitize(angles, bins)

        sector_means = []
        for i in range(1, 9):
            subset = mean_pollution[bin_indices == i]
            # Protection against empty bins if wind never blew in a specific sector
            sector_means.append(np.mean(subset) if len(subset) > 0 else 0.0)

        # Polar projection figure initialization
        fig = plt.figure(figsize=(6, 6), dpi=self.cfg.dpi)
        ax = fig.add_subplot(111, polar=True)
        theta = np.linspace(0, 2*np.pi, 8, endpoint=False)

        # Configure theta zero location to North and set clockwise direction
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # Bar chart rendering on polar axes
        ax.bar(theta, sector_means, width=0.7, color='red', alpha=0.5, edgecolor='black')
        ax.set_title("Pollution Wind Rose\n(Mean Conc. vs Direction)")
        return fig

    def plot_sensor_correlation(self, sensor_readings, max_sensors=20):
        """
        Computes and visualizes the Pearson correlation matrix between physical sensors.
        """
        # Sensor subset selection
        subset = sensor_readings[:, :max_sensors]
        
        # Correlation matrix calculation
        corr = np.corrcoef(subset.T)
        
        # Heatmap rendering logic
        fig, ax = plt.subplots(figsize=(8, 7), dpi=self.cfg.dpi)
        sns.heatmap(corr, cmap=self.cfg.cmap_div, center=0, ax=ax)
        ax.set_title(f"Correlation Matrix (First {max_sensors} Sensors)")
        return fig

    def plot_error_dist(self, gt_history, sensor_readings, coords):
        """
        Calculates and plots the distribution of measurement errors (Sensor vs Ground Truth).
        """
        errors = []
        # Error extraction loop for a representative subset of sensors
        for i in range(min(50, sensor_readings.shape[1])):
            x, y = coords[i].astype(int)
            # Calculate difference between sensor reading and actual value at grid node
            errors.extend(sensor_readings[:, i] - gt_history[:, x, y])
            
        # Histogram and KDE rendering
        fig, ax = plt.subplots(figsize=self.cfg.figsize_std, dpi=self.cfg.dpi)
        sns.histplot(errors, bins=50, kde=True, ax=ax, color='crimson')
        ax.set_title("Sensor Error Distribution")
        return fig