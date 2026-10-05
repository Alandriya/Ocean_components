"""SimVP-style forecaster with the common Forecast_nn interface.

Input / output are intentionally identical to UNetForecaster:
    x:          [B, T_in, C, H, W]
    mask:       [H, W], [B, H, W], [1, 1, H, W] or [B, 1, H, W]
    prediction: [B, T_out, C, H, W]

The model is adapted to masked geophysical maps rather than being a line-for-line
copy of the reference SimVP repository:
  1. every historical frame is encoded by one shared U-Net encoder;
  2. the encoded sequence is mixed by residual factorized 3-D CNN blocks;
  3. an ordered learned projection maps T_in latent states to T_out states;
  4. each horizon is decoded with the same U-Net decoder and skips from the last
     observed frame;
  5. residual=True predicts a correction to the last observed field.

This preserves the same data contract, mask handling, coordinate channels and
residual-persistence initialization used by the U-Net implementation.
"""

import torch
import torch.nn as nn

from Forecast_nn.models.unet import DoubleConv, DownBlock, UpBlock, _groups, _resize_mask


class FactorizedSTBlock(nn.Module):
    """Residual spatiotemporal block: temporal depthwise conv + spatial depthwise conv."""

    def __init__(self, channels, temporal_kernel=3, spatial_kernel=3, dropout=0.0):
        super().__init__()
        if temporal_kernel % 2 == 0 or spatial_kernel % 2 == 0:
            raise ValueError('temporal_kernel and spatial_kernel must be odd.')

        tp = temporal_kernel // 2
        sp = spatial_kernel // 2

        self.norm_temporal = nn.GroupNorm(_groups(channels), channels)
        self.temporal = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(temporal_kernel, 1, 1),
                padding=(tp, 0, 0),
                groups=channels,
                bias=False,
            ),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
        )

        self.norm_spatial = nn.GroupNorm(_groups(channels), channels)
        self.spatial = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, spatial_kernel, spatial_kernel),
                padding=(0, sp, sp),
                groups=channels,
                bias=False,
            ),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
        )

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.dropout(self.temporal(self.norm_temporal(x)))
        x = x + self.dropout(self.spatial(self.norm_spatial(x)))
        return x


class SimVPForecaster(nn.Module):
    """CNN-only direct multi-horizon forecaster for sequences of 2-D maps.

    Common Forecast_nn interface:
        input:  x    [B, T_in, C, H, W]
        mask:        [1, 1, H, W] or [B, 1, H, W]
        output:      [B, T_out, C, H, W]
    """

    def __init__(
        self,
        in_len,
        out_len,
        in_channels=1,
        base_channels=16,
        depth=3,
        dropout=0.05,
        residual=True,
        use_mask_channel=True,
        use_coord_channels=True,
        mask_features=True,
        temporal_blocks=4,
        temporal_kernel=3,
        spatial_kernel=3,
        horizon_embedding=True,
    ):
        super().__init__()
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.in_channels = int(in_channels)
        self.residual = bool(residual)
        self.use_mask_channel = bool(use_mask_channel)
        self.use_coord_channels = bool(use_coord_channels)
        self.mask_features = bool(mask_features)
        self.use_horizon_embedding = bool(horizon_embedding)

        if self.in_len < 1 or self.out_len < 1:
            raise ValueError('in_len and out_len must be positive.')
        if depth < 1:
            raise ValueError('depth must be >= 1.')

        static_channels = int(self.use_mask_channel) + 2 * int(self.use_coord_channels)
        frame_channels = self.in_channels + static_channels
        encoder_channels = [int(base_channels) * (2 ** i) for i in range(int(depth))]

        # Shared spatial encoder for every historical frame.
        self.stem = DoubleConv(frame_channels, encoder_channels[0], dropout)
        self.down = nn.ModuleList([
            DownBlock(encoder_channels[i - 1], encoder_channels[i], dropout)
            for i in range(1, len(encoder_channels))
        ])

        bottleneck_channels = encoder_channels[-1] * 2
        self.bottleneck = DownBlock(encoder_channels[-1], bottleneck_channels, dropout)

        # CNN-only spatiotemporal translator. Tensor layout here is B,C,T,h,w.
        self.temporal_mixer = nn.ModuleList([
            FactorizedSTBlock(
                bottleneck_channels,
                temporal_kernel=temporal_kernel,
                spatial_kernel=spatial_kernel,
                dropout=dropout,
            )
            for _ in range(int(temporal_blocks))
        ])

        # Ordered time mapping T_in -> T_out, shared across channels/spatial positions.
        self.time_projection = nn.Linear(self.in_len, self.out_len)

        if self.use_horizon_embedding:
            self.horizon_embedding = nn.Parameter(
                torch.empty(self.out_len, bottleneck_channels, 1, 1)
            )
            nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
        else:
            self.register_parameter('horizon_embedding', None)

        # Shared U-Net decoder. Skip features are taken from the last observed frame.
        decoder = []
        current = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            decoder.append(UpBlock(current, skip_channels, skip_channels, dropout))
            current = skip_channels
        self.up = nn.ModuleList(decoder)
        self.output = nn.Conv2d(current, self.in_channels, kernel_size=1)

        # Same initialization policy as UNetForecaster: start from persistence.
        if self.residual:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    @staticmethod
    def _coords(batch, height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([yy, xx], dim=0)[None].expand(batch, -1, -1, -1)

    @staticmethod
    def _prepare_mask(mask, batch, x):
        if mask is None:
            return None
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]
        elif mask.ndim != 4:
            raise ValueError(f'mask must have 2, 3 or 4 dimensions, got {mask.ndim}.')
        if mask.size(1) != 1:
            raise ValueError(f'mask must have one channel, got shape {tuple(mask.shape)}.')
        mask = mask.to(device=x.device, dtype=x.dtype)
        if mask.size(0) == 1:
            mask = mask.expand(batch, -1, -1, -1)
        elif mask.size(0) != batch:
            raise ValueError(f'mask batch must be 1 or {batch}, got {mask.size(0)}.')
        return mask

    def _feature_mask(self, feature, mask):
        if not self.mask_features or mask is None:
            return feature
        return feature * _resize_mask(mask, feature.shape[-2:], feature.dtype)

    def _sequence_mask(self, feature, mask):
        """Mask a B,C,T,h,w latent sequence without changing its time axis."""
        if not self.mask_features or mask is None:
            return feature
        m = _resize_mask(mask, feature.shape[-2:], feature.dtype)  # B,1,h,w
        return feature * m[:, :, None]

    def _frame_input(self, frame, mask, coords):
        parts = [frame]
        if self.use_mask_channel:
            if mask is None:
                raise ValueError('use_mask_channel=True requires mask in forward().')
            parts.append(mask)
        if self.use_coord_channels:
            parts.append(coords)
        return torch.cat(parts, dim=1)

    def _encode(self, frame, mask, coords):
        skips = []
        z = self._feature_mask(self.stem(self._frame_input(frame, mask, coords)), mask)
        skips.append(z)
        for block in self.down:
            z = self._feature_mask(block(z), mask)
            skips.append(z)
        z = self._feature_mask(self.bottleneck(z), mask)
        return z, skips

    def _decode(self, latent, last_skips, mask):
        z = self._feature_mask(latent, mask)
        for block, skip in zip(self.up, reversed(last_skips)):
            z = self._feature_mask(block(z, skip), mask)
        return self.output(z)

    def forward(self, x, mask=None):
        if x.ndim != 5:
            raise ValueError(f'x must be [B,T,C,H,W], got shape {tuple(x.shape)}.')

        batch, time, channels, height, width = x.shape
        if time != self.in_len or channels != self.in_channels:
            raise ValueError(
                f'SimVPForecaster expects T={self.in_len}, C={self.in_channels}; '
                f'got {tuple(x.shape)}.'
            )

        mask = self._prepare_mask(mask, batch, x)
        coords = (
            self._coords(batch, height, width, x.device, x.dtype)
            if self.use_coord_channels else None
        )

        encoded = []
        last_skips = None
        for t in range(time):
            z, skips = self._encode(x[:, t], mask, coords)
            encoded.append(z)
            if t == time - 1:
                last_skips = skips

        # B,T,C,h,w -> B,C,T,h,w.
        z = torch.stack(encoded, dim=1).permute(0, 2, 1, 3, 4).contiguous()
        z = self._sequence_mask(z, mask)
        for block in self.temporal_mixer:
            z = self._sequence_mask(block(z), mask)

        # Apply an ordered learned projection along the temporal axis.
        # B,C,T,h,w -> B,C,h,w,T -> B,C,h,w,T_out -> B,T_out,C,h,w.
        z = z.permute(0, 1, 3, 4, 2).contiguous()
        z = self.time_projection(z)
        z = z.permute(0, 4, 1, 2, 3).contiguous()

        corrections = []
        for horizon in range(self.out_len):
            zh = z[:, horizon]
            if self.horizon_embedding is not None:
                zh = zh + self.horizon_embedding[horizon]
            corrections.append(self._decode(zh, last_skips, mask))
        correction = torch.stack(corrections, dim=1)

        prediction = x[:, -1:] + correction if self.residual else correction
        if mask is not None:
            prediction = prediction * mask[:, None]
        return prediction
