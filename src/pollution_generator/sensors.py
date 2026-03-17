import numpy as np
from .config import SimConfig

class SensorManager:
    """
    Logic for managing sensor topologies and simulating the data acquisition process.
    """
    def __init__(self, config: SimConfig):
        self.config = config
        # Initialization of multiple sensor topologies and persistent measurement offsets (drifts)
        self.layouts = [self._create_layout() for _ in range(config.num_layouts)]
        self.drifts = [np.random.normal(0, config.sensor_drift_std, config.num_sensors) 
                       for _ in range(config.num_layouts)]

    def _create_layout(self):
        # Stochastic generation of sensor coordinates within the grid boundaries
        x = np.random.randint(0, self.config.grid_x, self.config.num_sensors)
        y = np.random.randint(0, self.config.grid_y, self.config.num_sensors)
        return np.column_stack((x, y))

    def sample(self, ground_truth: np.ndarray):
        """
        Simulates the sampling process, incorporating measurement errors and network instability.
        """
        all_readings = []
        for i, layout in enumerate(self.layouts):
            # Extraction of ground truth values at sensor locations
            vals = ground_truth[layout[:, 0], layout[:, 1]]
            
            # Simulation of measurement noise and systematic sensor drift
            noise = np.random.normal(0, self.config.sensor_noise_std, len(vals))
            vals_noisy = vals + self.drifts[i] + noise
            
            # Simulation of packet loss (network reliability modeling)
            mask = np.random.rand(len(vals)) > self.config.packet_loss_prob
            vals_noisy = np.where(mask, vals_noisy, 0.0)
            
            # Result aggregation and non-negative value enforcement
            all_readings.append(np.clip(vals_noisy, 0, None))
            
        return np.array(all_readings)