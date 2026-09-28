"""U-Net forecaster for sequences of masked 2-D geophysical fields.

The temporal axis is flattened into the channel dimension, so every input day has its
own channel position. The model therefore keeps the ordering of the input sequence while
using a standard 2-D U-Net to model the spatial structure of the maps.

Optional static channels:
- ocean/land mask;
- normalized y/x coordinates.

When residual=True the final convolution is initialized to zero and the network predicts
corrections to the last observed field. At initialization the forecast is exactly the
persistence baseline, which is useful when persistence is already strong at short horizons.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _resize_mask(mask, size, dtype):
    if mask is None:
        return None
    return F.interpolate(mask.to(dtype=dtype), size=size, mode='nearest')


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, dropout)

    def forward(self, x, skip):
        # Explicit interpolation makes odd grids such as 81x91 safe at every level.
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNetForecaster(nn.Module):
    """Direct multi-horizon U-Net forecast.

    Common Forecast_nn interface:
        input:  x    [B, T_in, C, H, W]
        mask:        [1, 1, H, W] or [B, 1, H, W]
        output:      [B, T_out, C, H, W]
    """

    def __init__(self, in_len, out_len, in_channels=1, base_channels=16, depth=3,
                 dropout=0.05, residual=True, use_mask_channel=True,
                 use_coord_channels=True, mask_features=True):
        super().__init__()
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.in_channels = int(in_channels)
        self.residual = bool(residual)
        self.use_mask_channel = bool(use_mask_channel)
        self.use_coord_channels = bool(use_coord_channels)
        self.mask_features = bool(mask_features)

        static_channels = int(self.use_mask_channel) + 2 * int(self.use_coord_channels)
        input_channels = self.in_len * self.in_channels + static_channels

        encoder_channels = [base_channels * (2 ** i) for i in range(depth)]
        self.stem = DoubleConv(input_channels, encoder_channels[0], dropout)
        self.down = nn.ModuleList([
            DownBlock(encoder_channels[i - 1], encoder_channels[i], dropout)
            for i in range(1, depth)
        ])

        bottleneck_channels = encoder_channels[-1] * 2
        self.bottleneck = DownBlock(encoder_channels[-1], bottleneck_channels, dropout)

        decoder = []
        current = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            decoder.append(UpBlock(current, skip_channels, skip_channels, dropout))
            current = skip_channels
        self.up = nn.ModuleList(decoder)

        self.output = nn.Conv2d(current, self.out_len * self.in_channels, 1)

        # For residual forecasting the initial network reproduces persistence exactly.
        if self.residual:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    @staticmethod
    def _coords(batch, height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([yy, xx], dim=0)[None]
        return coords.expand(batch, -1, -1, -1)

    def _apply_feature_mask(self, feature, mask):
        if not self.mask_features or mask is None:
            return feature
        resized = _resize_mask(mask, feature.shape[-2:], feature.dtype)
        return feature * resized

    def forward(self, x, mask=None):
        batch, time, channels, height, width = x.shape
        if time != self.in_len or channels != self.in_channels:
            raise ValueError(f'UNetForecaster expects T={self.in_len}, C={self.in_channels}; got {tuple(x.shape)}')

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            mask = mask.to(device=x.device, dtype=x.dtype)
            if mask.size(0) == 1:
                mask = mask.expand(batch, -1, -1, -1)

        parts = [x.reshape(batch, time * channels, height, width)]
        if self.use_mask_channel:
            if mask is None:
                raise ValueError('use_mask_channel=True requires mask in forward().')
            parts.append(mask)
        if self.use_coord_channels:
            parts.append(self._coords(batch, height, width, x.device, x.dtype))
        z = torch.cat(parts, dim=1)

        skips = []
        z = self._apply_feature_mask(self.stem(z), mask)
        skips.append(z)
        for block in self.down:
            z = self._apply_feature_mask(block(z), mask)
            skips.append(z)

        z = self._apply_feature_mask(self.bottleneck(z), mask)
        for block, skip in zip(self.up, reversed(skips)):
            z = self._apply_feature_mask(block(z, skip), mask)

        correction = self.output(z).view(batch, self.out_len, self.in_channels, height, width)
        prediction = x[:, -1:, :, :, :] + correction if self.residual else correction

        if mask is not None:
            prediction = prediction * mask[:, None]
        return prediction
