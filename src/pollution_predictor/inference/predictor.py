import torch
import numpy as np
from typing import Optional
from ..models import model_registry
from ..data.transforms import PollutionTransforms
from ..tracking.checkpointing import CheckpointManager

class InferencePredictor:
    """
    High-level wrapper for performing inference using trained pollution prediction models.
    Synchronizes model architecture and transformation states from saved artifacts.
    """
    def __init__(self, artifact_path: str, device: str = 'cpu'):
        # Compute device configuration
        self.device = torch.device(device)
        
        # Load unified checkpoint containing weights, config, and transforms state
        checkpoint = CheckpointManager.load(artifact_path, self.device)
        cfg = checkpoint['config']

        # Extract model-specific hyperparameters
        model_params = cfg['model'].get('params', {})

        # Dynamic model instantiation via the architectural registry
        self.model = model_registry.create(
            name=cfg['model']['architecture'],
            time_steps=cfg['model']['time_steps'],
            grid_x=cfg['data']['grid_size'][0],
            grid_y=cfg['data']['grid_size'][1],
            use_wind=cfg['model'].get('use_wind', True),
            **model_params
        )
        
        # Restoration of trained weights and state synchronization
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(self.device)
        self.model.eval()

        # Transformation logic initialization and state restoration
        self.transforms = PollutionTransforms()
        self.transforms.load_state(checkpoint['transforms_state'])

    def predict_deterministic(self, readings, coords, wind=None):
        """
        Executes a deterministic forward pass on raw sensor inputs.
        
        Args:
            readings: (B, Sensors, Time) or (Sensors, Time) raw signal data.
            coords: (B, Sensors, 2) spatial coordinates of sensors.
            wind: (B, 2) optional global wind vectors.
        """
        self.model.eval()

        # Batch size detection and dimension normalization
        batch_size = readings.shape[0] if readings.ndim == 3 else 1

        processed_readings = []
        processed_winds = []

        # Instance-based preprocessing loop to apply local normalization
        for i in range(batch_size):
            r = readings[i] if readings.ndim == 3 else readings
            c = coords[i] if coords.ndim == 3 else coords
            w = wind[i] if wind is not None else None

            # Apply data transformations (normalization, noise floor clipping)
            data = self.transforms.transform(r, c, w, use_wind=(wind is not None))
            processed_readings.append(data['readings'])
            if 'wind' in data:
                processed_winds.append(data['wind'])

        # Feature batch aggregation for GPU processing
        batch = {
            "readings": torch.stack(processed_readings).to(self.device),
            "coords": torch.tensor(coords, dtype=torch.long).to(self.device)
        }
        
        if processed_winds:
            batch["wind"] = torch.stack(processed_winds).to(self.device)

        # Gradient-free forward pass
        with torch.no_grad():
            output = self.model(batch)
            # Removal of channel dimension and conversion to NumPy
            return output.squeeze(1).cpu().numpy()