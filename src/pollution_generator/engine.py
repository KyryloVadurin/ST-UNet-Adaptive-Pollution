import numpy as np
from typing import Tuple
from .config import SimConfig
from .physics import Environment
from .sources import SourceManager
from .sensors import SensorManager

class Simulator:
    """
    Orchestrator class that integrates physics, emission sources, and sensor networks 
    to execute a complete pollution dispersion scenario.
    """
    def __init__(self, config: SimConfig):
        self.config = config
        
        # Stochastic initialization of the global wind vector
        speed = np.random.uniform(*config.wind_speed_range)
        angle = np.random.uniform(*config.wind_angle_range)
        self.start_wind = (speed * np.cos(angle), speed * np.sin(angle))

        # Component initialization
        self.env = Environment(config, self.start_wind)
        self.sources = SourceManager(config)
        self.sensors = SensorManager(config)

    def run_scenario(self):
        """
        Executes the simulation loop including burn-in and data collection phases.
        """
        total_steps = self.config.burn_in_steps + self.config.sampling_steps
        gt_history = []
        sensor_history = []

        for t in range(total_steps):
            # Calculate emissions for the current timestamp
            emissions = self.sources.get_emissions_grid()
            
            # Update environmental state (passing t for cyclical background logic)
            self.env.step(emissions, t)

            # Data collection logic after the burn-in period
            if t >= self.config.burn_in_steps:
                current_state = self.env.get_state()
                gt_history.append(current_state)
                # Sample the current state across all sensor layouts
                sensor_history.append(self.sensors.sample(current_state))

        # Result formatting:
        # Ground Truth: (Time, X, Y)
        # Sensors: Transposed to (Layout, Time, Sensor_Index)
        return (
            np.array(gt_history), 
            np.array(sensor_history).transpose(1, 0, 2), 
            self.sensors.layouts, 
            self.start_wind
        )