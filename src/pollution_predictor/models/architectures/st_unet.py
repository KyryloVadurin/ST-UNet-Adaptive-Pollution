import torch
import torch.nn as nn
from ..base import BasePredictor
from ..registry import model_registry

class ResBlock(nn.Module):
    """
    Core residual building block with spatial dropout for regularization.
    Utilizes skip connections to mitigate gradient vanishing in deep architectures.
    """
    def __init__(self, in_c, out_c, dropout_rate=0.2):
        super().__init__()
        # Convolutional pipeline: Conv -> BN -> ReLU -> Dropout -> Conv -> BN
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        # Identity shortcut logic
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False), 
                nn.BatchNorm2d(out_c)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): 
        # Additive residual connection followed by activation
        return self.relu(self.conv(x) + self.shortcut(x))

class UpBlock(nn.Module):
    """
    Upsampling block using bilinear interpolation to prevent checkerboard artifacts.
    Combines feature maps from the decoder and skip connections from the encoder.
    """
    def __init__(self, in_c, out_c, dropout_rate=0.2):
        super().__init__()
        # Smooth scaling using bilinear mode
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ResBlock(in_c, out_c, dropout_rate)

    def forward(self, x1, x2):
        # Feature concatenation followed by residual refinement
        x1 = self.up(x1)
        return self.conv(torch.cat([x1, x2], dim=1))

@model_registry.register("st_unet")
class STUNet(BasePredictor):
    """
    Spatio-Temporal UNet architecture tailored for pollution dispersion modeling.
    Optimized for numerical stability and spatial plume reconstruction.
    """
    def __init__(self, time_steps: int, grid_x: int, grid_y: int, use_wind: bool = True, 
                 hidden_dim: int = 64, dropout_rate: float = 0.2):
        # Parent class initialization (handles grid projection buffers)
        super().__init__(time_steps, grid_x, grid_y, use_wind)
        
        # Input channel composition: T steps + 2 Coords + 2 Wind
        in_channels = time_steps + 4 

        # Symmetric Encoder hierarchy
        self.inc = ResBlock(in_channels, hidden_dim, dropout_rate)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(hidden_dim, hidden_dim*2, dropout_rate))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(hidden_dim*2, hidden_dim*4, dropout_rate))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(hidden_dim*4, hidden_dim*8, dropout_rate))

        # Smooth Decoder hierarchy with skip-connection integration
        self.up1 = UpBlock(hidden_dim*8 + hidden_dim*4, hidden_dim*4, dropout_rate)
        self.up2 = UpBlock(hidden_dim*4 + hidden_dim*2, hidden_dim*2, dropout_rate)
        self.up3 = UpBlock(hidden_dim*2 + hidden_dim, hidden_dim, dropout_rate)

        # Final reconstruction head for intensity estimation
        self.outc = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, batch: dict) -> torch.Tensor:
        # 1. Project sparse sensor data to dense spatial features
        x = self.prepare_spatial_grid(batch)
        
        # 2. Downsampling path (Encoder)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 3. Upsampling path with skip connections (Decoder)
        u1 = self.up1(x4, x3)
        u2 = self.up2(u1, x2)
        u3 = self.up3(u2, x1)

        # 4. Result generation
        return self.outc(u3)