"""Radial-basis-function KAN layers for neural forecasting."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelLayerNorm(nn.Module):
    """LayerNorm over the channel vector at every spatial point."""
    def __init__(self, channels):
        """Initialize Pixel Layer Norm with ``channels``."""
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """Apply the Pixel Layer Norm transformation to the input tensors."""
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class RBFKANHead(nn.Module):
    """Compact residual RBF-KAN head from the SDE-Energy paper."""
    def __init__(self, in_channels, out_channels, kan_dim=64, num_centers=32, gamma=1.0, dropout=0.05):
        """Initialize RBFKANHead with ``in_channels``, ``out_channels``, ``kan_dim``, ``num_centers``, ``gamma``, ``dropout``."""
        super().__init__()
        self.gamma = gamma
        self.project = nn.Conv2d(in_channels, kan_dim, 1)
        self.norm = PixelLayerNorm(kan_dim)
        self.register_buffer('centers', torch.linspace(-1.0, 1.0, num_centers))
        self.rbf_weights = nn.Parameter(torch.empty(kan_dim, num_centers))
        self.mix1 = nn.Conv2d(kan_dim, kan_dim, 1)
        self.mix2 = nn.Conv2d(kan_dim, out_channels, 1)
        self.dropout = nn.Dropout2d(dropout)
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.rbf_weights, mean=0.0, std=0.02)

    def forward(self, h):
        # u = tanh(Norm(P h + p)), so every KAN coordinate lies in [-1, 1].
        """Apply the RBFKANHead transformation to the input tensors."""
        u = torch.tanh(self.norm(self.project(h)))

        # Compute sum_m a_{r,m} exp(-gamma (u_r-c_m)^2) without creating a huge BxDxMxHxW tensor.
        g = torch.zeros_like(u)
        for m, center in enumerate(self.centers):
            coeff = self.rbf_weights[:, m].view(1, -1, 1, 1)
            g = g + coeff * torch.exp(-self.gamma * (u - center) ** 2)

        correction = self.mix2(self.dropout(F.relu(self.mix1(g), inplace=True)))
        return torch.sigmoid(self.gate_logit) * correction
