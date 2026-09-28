import torch
import torch.nn as nn
import torch.nn.functional as F

from Forecast_nn.models.rbf_kan import RBFKANHead


def _groups(channels):
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g


class ConvBlock(nn.Module):
    """Two convolutional layers with normalization and a pointwise nonlinearity."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, mask):
        return self.block(x) * mask


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, base_channels=16, latent_channels=64, dropout=0.05):
        super().__init__()
        # The binary ocean mask is appended as an extra channel so the encoder can distinguish land from a true zero.
        self.block1 = ConvBlock(in_channels + 1, base_channels, dropout)
        self.block2 = ConvBlock(base_channels, base_channels * 2, dropout)
        self.block3 = ConvBlock(base_channels * 2, latent_channels, dropout)

    @staticmethod
    def masked_pool(x, mask, pooled_mask):
        # Divide pooled features by the valid-ocean fraction so coastlines are not attenuated by land zeros.
        support = F.avg_pool2d(mask, 2)
        pooled = F.avg_pool2d(x * mask, 2) / support.clamp_min(1e-6)
        return pooled * pooled_mask

    def forward(self, x, mask0, mask1, mask2):
        x = torch.cat([x * mask0, mask0.expand(x.size(0), -1, -1, -1)], dim=1)
        x = self.block1(x, mask0)
        x = self.block2(self.masked_pool(x, mask0, mask1), mask1)
        return self.block3(self.masked_pool(x, mask1, mask2), mask2)


class SpatialDecoder(nn.Module):
    def __init__(self, base_channels=16, latent_channels=64, dropout=0.05):
        super().__init__()
        self.block1 = ConvBlock(latent_channels, base_channels * 2, dropout)
        self.block2 = ConvBlock(base_channels * 2, base_channels, dropout)

    def forward(self, z, mask0, mask1):
        z = F.interpolate(z, size=mask1.shape[-2:], mode='bilinear', align_corners=False)
        z = self.block1(z, mask1)
        z = F.interpolate(z, size=mask0.shape[-2:], mode='bilinear', align_corners=False)
        return self.block2(z, mask0)


class TemporalAttention(nn.Module):
    """Attention over encoded input frames using masked global spatial descriptors."""
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    def forward(self, z, mask):
        # z: B,T,C,H,W. Land features are already zero, but divide by the ocean area rather than H*W.
        denom = mask.sum(dim=(-2, -1)).clamp_min(1.0)
        q = (z * mask.unsqueeze(1)).sum(dim=(-2, -1)) / denom.unsqueeze(1)
        weights = torch.softmax(self.mlp(q).squeeze(-1), dim=1)
        return (z * weights[:, :, None, None, None]).sum(dim=1), weights


class SDEEnergyNet(nn.Module):
    """Encoder -> temporal attention -> decoder -> linear head + residual RBF-KAN correction."""
    def __init__(self, mask, output_length, in_channels=1, out_channels=1, base_channels=16, latent_channels=64,
                 kan_dim=64, rbf_centers=32, rbf_gamma=1.0, dropout=0.05):
        super().__init__()
        self.output_length = output_length
        self.out_channels = out_channels
        mask = torch.as_tensor(mask, dtype=torch.float32)
        self.register_buffer('ocean_mask', mask, persistent=False)
        self.encoder = SpatialEncoder(in_channels, base_channels, latent_channels, dropout)
        self.temporal = TemporalAttention(latent_channels)
        self.decoder = SpatialDecoder(base_channels, latent_channels, dropout)
        head_channels = output_length * out_channels
        self.linear_head = nn.Conv2d(base_channels, head_channels, 1)
        self.kan_head = RBFKANHead(base_channels, head_channels, kan_dim, rbf_centers, rbf_gamma, dropout)

    def _mask_pyramid(self, batch_size, dtype):
        mask0 = self.ocean_mask.to(dtype=dtype)[None, None]
        h, w = mask0.shape[-2:]
        mask1 = (F.avg_pool2d(mask0, 2) > 0).to(dtype)
        mask2 = (F.avg_pool2d(mask1, 2) > 0).to(dtype)
        return mask0.expand(batch_size, -1, -1, -1), mask1.expand(batch_size, -1, -1, -1), mask2.expand(batch_size, -1, -1, -1)

    def forward(self, x):
        # x: B,T,C,H,W. The model predicts all future frames in one pass, not autoregressively.
        batch, time, _, _, _ = x.shape
        mask0, mask1, mask2 = self._mask_pyramid(batch, x.dtype)
        encoded = [self.encoder(x[:, t], mask0, mask1, mask2) for t in range(time)]
        z = torch.stack(encoded, dim=1)
        z_star, attention = self.temporal(z, mask2)
        h = self.decoder(z_star, mask0, mask1)
        prediction = self.linear_head(h) + self.kan_head(h)
        prediction = prediction.view(batch, self.output_length, self.out_channels, mask0.shape[-2], mask0.shape[-1])
        prediction = prediction * mask0[:, None]
        return prediction, attention
