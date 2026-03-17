import numpy as np
from typing import Tuple, Optional

class ClassicalBaseline:
    """
    Classical baseline model for pollution estimation.
    Uses weighted spatial averaging with heuristic wind compensation.
    """
    @staticmethod
    def predict(readings: np.ndarray, coords: np.ndarray, wind: Optional[np.ndarray], grid_shape: Tuple[int, int]):
        # Grid dimensions extraction
        grid_x, grid_y = grid_shape
        
        # Sensor data preprocessing and temporal aggregation
        r_proc = np.log1p(readings)
        mean_r = np.mean(r_proc, axis=0) 

        # Spatial weighted center of mass calculation
        total_p = np.sum(mean_r) + 1e-8
        cx = np.sum(coords[:, 0] * mean_r) / total_p
        cy = np.sum(coords[:, 1] * mean_r) / total_p

        # Wind-based advection compensation logic
        if wind is not None and np.linalg.norm(wind) > 1e-5:
            # Source location estimation offset
            est_x = cx - wind[0] * 4.0 
            est_y = cy - wind[1] * 4.0
        else:
            est_x, est_y = cx, cy

        # Boundary clipping logic
        est_x = np.clip(est_x, 0, grid_x - 1)
        est_y = np.clip(est_y, 0, grid_y - 1)

        # Gaussian dispersion synthesis
        yy, xx = np.mgrid[0:grid_x, 0:grid_y]
        dist_sq = (yy - est_x)**2 + (xx - est_y)**2
        sigma = grid_x / 8.0 
        heatmap = np.exp(-dist_sq / (2 * sigma**2))

        # Result intensity normalization
        return np.clip(heatmap, 0, 1)