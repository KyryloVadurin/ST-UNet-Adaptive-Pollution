import numpy as np
import scipy.ndimage as ndimage
from typing import Tuple
from .config import SimConfig

class Environment:
    """
    Simulation engine for atmospheric pollution dispersion.
    Handles wind dynamics, advection, diffusion, and dynamic background noise.
    """
    def __init__(self, config: SimConfig, initial_wind: Tuple[float, float]):
        self.config = config
        # Initialize grid with baseline background pollution
        self.grid = np.full((config.grid_x, config.grid_y), config.bg_pollution_base)
        self.current_wind = list(initial_wind)
        self.current_bg = config.bg_pollution_base

    def step(self, emissions: np.ndarray, step_idx: int):
        """
        Calculates the next state of the environment for a single time step.
        """
        # 1. Update wind vector with stochastic variability
        self.current_wind[0] += np.random.normal(0, self.config.wind_variability)
        self.current_wind[1] += np.random.normal(0, self.config.wind_variability)

        # 2. Update dynamic background levels (Diurnal cycle + Random drift)
        # Simulate a 24-hour periodic oscillation
        cycle = self.config.bg_fluctuation_amp * np.sin(2 * np.pi * step_idx / 24)
        self.current_bg += np.random.normal(0, self.config.bg_drift_std)
        effective_bg = max(0.5, self.config.bg_pollution_base + cycle + self.current_bg - self.config.bg_pollution_base)

        # 3. Transport physics: Advection and Diffusion
        # Add new emissions to the current grid
        self.grid += emissions
        # Shift the pollution cloud based on wind speed and direction
        self.grid = ndimage.shift(self.grid, self.current_wind, mode='nearest')

        # Apply Gaussian diffusion
        if self.config.diffusion_sigma > 0:
            self.grid = ndimage.gaussian_filter(self.grid, sigma=self.config.diffusion_sigma)

        # 4. Apply atmospheric decay and convergence to dynamic background level
        self.grid *= (1.0 - self.config.decay_rate)
        # Reintroduce background pollution uniformly across the grid
        self.grid += (effective_bg * self.config.decay_rate) 

        # Apply final spatial white noise
        noise = np.random.normal(0, self.config.bg_noise_std, self.grid.shape)
        self.grid = np.clip(self.grid + noise, 0, None)

    def get_state(self):
        """Returns the current pollution distribution map."""
        return self.grid.copy()