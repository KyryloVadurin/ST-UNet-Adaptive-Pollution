import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict

class BasePredictor(nn.Module, ABC):
    """
    Abstract base class for spatio-temporal architectures.
    Handles the transformation of sparse sensor readings into dense grid features.
    """
    def __init__(self, time_steps: int, grid_x: int, grid_y: int, use_wind: bool = True):
        # Dimension and feature flags initialization
        super().__init__()
        self.time_steps = time_steps
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.use_wind = use_wind

        # Pre-computed spatial coordinate grids setup
        cx = torch.linspace(0, 1, grid_x).view(1, 1, grid_x, 1).expand(1, 1, grid_x, grid_y).clone()
        cy = torch.linspace(0, 1, grid_y).view(1, 1, 1, grid_y).expand(1, 1, grid_x, grid_y).clone()
        
        # Buffer registration for persistent coordinate features
        self.register_buffer('coord_x', cx)
        self.register_buffer('coord_y', cy)

    def prepare_spatial_grid(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Projects sparse readings onto a dense grid and appends auxiliary features."""
        readings, coords = batch['readings'], batch['coords']
        B, N, T = readings.shape

        # 1. Vectorized sparse-to-dense grid projection
        grid = torch.zeros((B, T, self.grid_x, self.grid_y), device=readings.device)
        cx_idx = torch.clamp(coords[:, :, 0], 0, self.grid_x - 1)
        cy_idx = torch.clamp(coords[:, :, 1], 0, self.grid_y - 1)
        flat_indices = cy_idx + cx_idx * self.grid_y
        
        src = readings.transpose(1, 2)
        time_expanded_indices = flat_indices.unsqueeze(1).expand(-1, T, -1)
        grid.view(B, T, -1).scatter_add_(2, time_expanded_indices, src)

        # 2. Base input assembly (Grid + Spatial Coordinates)
        inputs =[grid, self.coord_x.expand(B, -1, -1, -1), self.coord_y.expand(B, -1, -1, -1)]

        # 3. Dynamic wind feature integration
        if self.use_wind:
            if 'wind' not in batch:
                raise RuntimeError("Model initialized with use_wind=True, but 'wind' missing in batch.")
            wind = batch['wind']
            wx = wind[:, 0].view(B, 1, 1, 1).expand(B, 1, self.grid_x, self.grid_y)
            wy = wind[:, 1].view(B, 1, 1, 1).expand(B, 1, self.grid_x, self.grid_y)
            inputs.extend([wx, wy])

        # Feature concatenation along the channel dimension
        return torch.cat(inputs, dim=1)

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Abstract method to be implemented by specific architectures."""
        pass

    def freeze_encoder(self, freeze: bool = True):
        """Standard interface for parameter freezing logic."""
        print(f"Warning: freeze_encoder is not implemented for {self.__class__.__name__}")