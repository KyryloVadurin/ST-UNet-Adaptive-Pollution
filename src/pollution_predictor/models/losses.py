import torch
import torch.nn as nn

class OmniStructuralLoss(nn.Module):
    """
    Spatio-temporal loss function designed for high-contrast pollution maps.
    Uses binary-mask based weighting to prioritize background cleaning and peak reconstruction.
    """
    def __init__(self, w_bg=10.0, w_peak=10.0):
        super().__init__()
        # Penalty scaling factors
        self.w_bg = w_bg     
        self.w_peak = w_peak 

    def forward(self, pred, target, struct_factor=1.0):
        # 1. Initialize base weight map (unit importance for all pixels)
        weight = torch.ones_like(target)

        # 2. Assign high weight to background pixels to suppress "overexposure" noise
        weight[target < 0.05] = self.w_bg

        # 3. Assign high weight to epicenter pixels to drive maximum intensity accuracy
        weight[target > 0.5] = self.w_peak

        # 4. Computation of numerically stable weighted Mean Squared Error (MSE)
        loss = (weight * (pred - target) ** 2).mean()

        return loss