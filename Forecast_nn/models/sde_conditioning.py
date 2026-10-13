"""Small conditioning layers shared by the SDE-informed map forecasters."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from Forecast_nn.models.unet import _groups, _resize_mask


class SDEFiLM2d(nn.Module):
    """Feature-wise affine modulation generated from local a/b coefficient maps.

    The final convolution is zero-initialized, so an informed model starts with
    exactly the same hidden representation as its non-informed counterpart.
    """
    def __init__(self, sde_channels, feature_channels, scale=0.1):
        super().__init__()
        hidden = max(int(feature_channels) // 2, 16)
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.Conv2d(int(sde_channels), hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * int(feature_channels), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature, sde_map, mask=None):
        sde_map = F.interpolate(sde_map.float(), size=feature.shape[-2:], mode='bilinear', align_corners=False)
        sde_map = sde_map.to(dtype=feature.dtype)
        if mask is not None:
            sde_map = sde_map * _resize_mask(mask, feature.shape[-2:], feature.dtype)
        gamma, beta = self.net(sde_map).chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        return feature * (1.0 + self.scale * gamma) + self.scale * beta
