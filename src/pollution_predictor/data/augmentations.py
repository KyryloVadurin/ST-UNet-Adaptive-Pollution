import numpy as np
import random

class SpatialAugmentations:
    """Physically consistent spatial augmentations for inverse problems."""
    def __init__(self, grid_size, prob=0.5):
        # Initialization of augmentation parameters
        self.prob = prob
        self.grid_x, self.grid_y = grid_size

    def __call__(self, readings, coords, wind, target):
        # Probabilistic bypass logic
        if random.random() > self.prob:
            return readings, coords, wind, target

        # 1. Sensor Dropout logic (randomly zeroing 10-20% of inputs)
        if random.random() < 0.5:
            # Handle different input dimensionalities
            num_sensors = readings.shape[0] if readings.ndim == 2 else readings.shape[1]
            drop_count = int(num_sensors * random.uniform(0.1, 0.2))
            drop_indices = random.sample(range(num_sensors), drop_count)

            # Apply dropout mask to readings
            if readings.ndim == 2: 
                readings[:, drop_indices] = 0.0
            else: 
                readings[drop_indices] = 0.0

        # 2. Discrete spatial rotations (90, 180, 270 degrees)
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            # Rotate the ground truth target map
            if target.ndim == 2:
                target = np.rot90(target, k=k)
            elif target.ndim == 3:
                target = np.rot90(target, k=k, axes=(1, 2))

            # Transform physical sensor coordinates and wind vectors
            for _ in range(k):
                # Apply 90-degree coordinate transformation
                coords = np.column_stack((coords[:, 1], self.grid_x - 1 - coords[:, 0]))
                # Apply 90-degree vector rotation
                wind = np.array([wind[1], -wind[0]])

        return readings, coords, wind, target