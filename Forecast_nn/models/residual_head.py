"""Lead-specific residual projection head for multi-step forecasts."""


import torch
import torch.nn as nn

class HorizonResidualHead(nn.Module):
    """PyTorch module implementing the Horizon Residual Head architecture or layer."""
    def __init__(self, channels, out_channels, horizons):
        """Initialize Horizon Residual Head with ``channels``, ``out_channels``, ``horizons``."""
        super().__init__()
        self.emb = nn.Parameter(torch.randn(horizons, channels, 1, 1))
        self.head = nn.Conv2d(channels, out_channels, 1)

    def forward(self, h, last):
        """Project encoded features to the requested forecast horizon."""
        result = []
        for i in range(self.emb.shape[0]):
            result.append(last + self.head(h + self.emb[i]))
        return torch.stack(result, 1)
