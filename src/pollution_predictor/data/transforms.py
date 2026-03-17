import numpy as np
import torch
from typing import Dict

class PollutionTransforms:
    """
    Data transformation class for pollution sensor readings and target maps.
    """
    def __init__(self, noise_floor=0.02):
        # Initialization of normalization parameters and state
        self.noise_floor = noise_floor
        self.max_wind_speed = 8.0 
        self.is_fitted = True 
        self.r_mean, self.r_std = 0.0, 1.0

    def fit_from_h5(self, h5_path): 
        # Method for dataset-wide statistics calculation (currently bypassed)
        pass 

    def transform(self, readings, coords, wind, target=None, use_wind=True) -> Dict[str, torch.Tensor]:
        # Input sensor readings normalization logic (Instance-based min-max)
        r_min, r_max = readings.min(), readings.max()
        diff_r = r_max - r_min
        r_norm = np.zeros_like(readings) if diff_r < 1e-4 else (readings - r_min) / diff_r
        
        # Noise floor suppression
        r_norm[r_norm < self.noise_floor] = 0.0

        # Tensor conversion and axis alignment for model input
        t_readings = torch.tensor(r_norm, dtype=torch.float32)
        if t_readings.ndim == 2 and t_readings.shape[0] != coords.shape[0]: 
            t_readings = t_readings.transpose(0, 1)

        # Result dictionary initialization with sensor features
        out = {"readings": t_readings, "coords": torch.tensor(coords, dtype=torch.long)}

        # Wind vector normalization and inclusion
        if use_wind and wind is not None:
            out["wind"] = torch.tensor(wind, dtype=torch.float32) / self.max_wind_speed

        # Ground truth target processing
        if target is not None:
            # Temporal aggregation
            if target.ndim == 3: target = np.mean(target, axis=0)
            
            # Normalization of target map intensity
            t_min, t_max = target.min(), target.max()
            diff_t = t_max - t_min

            if diff_t < 1e-4:
                t_norm = np.zeros_like(target)
            else:
                t_norm = (target - t_min) / diff_t

            # Noise floor cutoff and contrast stretching logic
            cutoff = 0.10 
            t_norm = np.where(t_norm < cutoff, 0.0, t_norm)
            t_norm = np.where(t_norm > 0, (t_norm - cutoff) / (1.0 - cutoff + 1e-8), 0.0)

            # Final tensor formatting for target
            out["target"] = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(0)

        return out

    # Methods for serialization of the transformer configuration
    def get_state(self): return {"noise_floor": self.noise_floor}
    def load_state(self, s): self.noise_floor = s.get("noise_floor", 0.02)