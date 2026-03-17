from dataclasses import dataclass, field, asdict
from typing import Tuple, List

# Configuration data structure for simulation parameters
@dataclass
class SimConfig:
    # Grid dimensions and temporal sampling parameters
    grid_x: int = 64
    grid_y: int = 64
    cell_size_m: float = 100.0
    sampling_steps: int = 48
    burn_in_steps: int = 24

    # Physical properties and stochastic wind dynamics
    wind_speed_range: Tuple[float, float] = (0.5, 3.0) 
    wind_angle_range: Tuple[float, float] = (0.0, 6.283) 
    wind_variability: float = 0.1
    diffusion_sigma: float = 1.0
    decay_rate: float = 0.04

    # Dynamic background pollution level modeling
    bg_pollution_base: float = 2.0         
    bg_fluctuation_amp: float = 1.5       
    bg_drift_std: float = 0.1             
    bg_noise_std: float = 0.5             

    # Stationary and mobile emission source parameters
    num_static_sources: int = 5
    static_intensity_range: Tuple[float, float] = (70.0, 180.0)
    num_mobile_sources: int = 50
    mobile_intensity_range: Tuple[float, float] = (5.0, 15.0)
    num_main_routes: int = 4
    num_minor_routes: int = 6

    # Sensor network deployment and data quality metrics
    num_sensors: int = 150
    num_layouts: int = 10
    packet_loss_prob: float = 0.1
    sensor_drift_std: float = 2.0
    sensor_noise_std: float = 1.0

    # Serialization method for dictionary conversion
    def to_dict(self):
        return asdict(self)