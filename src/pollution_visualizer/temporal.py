import matplotlib.pyplot as plt
import numpy as np
from .config import VisConfig

class TemporalPlotter:
    """
    Handles temporal data visualization, including trends, distribution shifts, 
    diurnal patterns, and cross-sensor lag analysis.
    """
    def __init__(self, config: VisConfig):
        self.cfg = config

    def plot_trends(self, gt_history):
        """
        Renders basic temporal trends (Global Mean) with min-max spatial range shading.
        """
        # Global statistics calculation across spatial dimensions
        mean_v = np.mean(gt_history, axis=(1, 2))
        max_v = np.max(gt_history, axis=(1, 2))
        min_v = np.min(gt_history, axis=(1, 2))

        # Figure and axis setup
        fig, ax = plt.subplots(figsize=self.cfg.figsize_std, dpi=self.cfg.dpi)
        steps = range(len(mean_v))
        
        # Plotting logic for trend line and uncertainty area
        ax.plot(steps, mean_v, label="Global Mean", color="blue", lw=2.5)
        ax.fill_between(steps, min_v, max_v, color="blue", alpha=0.1, label="Spatial Range")
        
        # Labeling and styling
        ax.set_title("Pollution Temporal Trends", fontsize=self.cfg.title_fontsize)
        ax.set_xlabel("Time (Steps)")
        ax.set_ylabel("Concentration")
        ax.legend()
        ax.grid(True, alpha=0.2)
        return fig

    def plot_ridge_joyplot(self, gt_history, skip_steps=6):
        """
        Generates a Ridge plot (Joyplot) to visualize concentration distribution shifts over time.
        """
        # Temporal slicing
        subset = gt_history[::skip_steps]
        
        # Multi-axis subplot configuration
        fig, axes = plt.subplots(len(subset), 1, figsize=(10, 8), sharex=True, dpi=self.cfg.dpi)
        colors = plt.cm.magma(np.linspace(0, 0.8, len(subset)))

        # Iterative distribution plotting for each time step
        for i, (data, ax) in enumerate(zip(subset, axes)):
            ax.hist(data.flatten(), bins=80, density=True, alpha=0.7, color=colors[i])
            
            # Formatting for the "Joyplot" aesthetic (minimalist spines/ticks)
            ax.set_yticks([])
            ax.patch.set_alpha(0)
            for s in ["top", "right", "left"]: 
                ax.spines[s].set_visible(False)
            if i < len(subset)-1: 
                ax.spines["bottom"].set_visible(False)

        plt.suptitle("Pollution Density Ridge Plot", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_diurnal_analysis(self, gt_history):
        """
        Calculates and plots the average diurnal (24-hour cycle) pattern.
        """
        if gt_history.shape[0] < 24: 
            return None
            
        # 24-hour cycle aggregation logic
        diurnal = [np.mean(gt_history[i::24], axis=(0, 1, 2)) for i in range(24)]
        
        # Visualization setup
        fig, ax = plt.subplots(figsize=self.cfg.figsize_std, dpi=self.cfg.dpi)
        ax.plot(range(24), diurnal, marker='o', color='darkviolet', lw=2)
        ax.set_title("Average Diurnal Pattern")
        ax.set_xticks(range(24))
        ax.grid(True, alpha=0.3)
        return fig

    def plot_lag_analysis(self, sensor_readings, sensor_a=0, sensor_b=10):
        """
        Performs cross-correlation analysis between two sensor time-series to detect signal lag.
        """
        s1, s2 = sensor_readings[:, sensor_a], sensor_readings[:, sensor_b]
        
        # Cross-correlation calculation for zero-centered signals
        corr = np.correlate(s1 - s1.mean(), s2 - s2.mean(), mode='full')
        lags = np.arange(-len(s1) + 1, len(s1))
        
        # Visualization
        fig, ax = plt.subplots(figsize=self.cfg.figsize_std, dpi=self.cfg.dpi)
        ax.plot(lags, corr, color='teal')
        ax.set_title(f"Lag Analysis (Sensor {sensor_a} vs {sensor_b})")
        return fig