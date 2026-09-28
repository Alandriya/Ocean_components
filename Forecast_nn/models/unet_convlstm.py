"""U-Net + ConvLSTM forecaster for masked 2-D geophysical fields.

Each historical map is encoded by the same U-Net encoder. The bottleneck feature maps
are passed through ConvLSTM in chronological order, so the temporal block preserves the
spatial grid instead of reducing every frame to a vector. The decoder uses skip features
from the last observed map, which is useful for short-range forecasting where persistence
is already a strong baseline.

The output is a direct multi-horizon residual forecast relative to the last input map.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from Forecast_nn.models.unet import DoubleConv, DownBlock, UpBlock, _resize_mask


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, state=None):
        if state is None:
            h = torch.zeros(x.size(0), self.hidden_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            c = torch.zeros_like(h)
        else:
            h, c = state
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class UNetConvLSTMForecaster(nn.Module):
    """Shared U-Net encoder -> ConvLSTM bottleneck -> U-Net decoder.

    Common interface:
        x:          [B,T_in,C,H,W]
        mask:       [1,1,H,W] or [B,1,H,W]
        prediction: [B,T_out,C,H,W]
    """

    def __init__(self, in_len, out_len, in_channels=1, base_channels=16, depth=3,
                 dropout=0.05, residual=True, use_mask_channel=True,
                 use_coord_channels=True, mask_features=True, lstm_kernel=3):
        super().__init__()
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.in_channels = int(in_channels)
        self.residual = bool(residual)
        self.use_mask_channel = bool(use_mask_channel)
        self.use_coord_channels = bool(use_coord_channels)
        self.mask_features = bool(mask_features)

        static_channels = int(self.use_mask_channel) + 2 * int(self.use_coord_channels)
        frame_channels = self.in_channels + static_channels
        encoder_channels = [base_channels * (2 ** i) for i in range(depth)]

        self.stem = DoubleConv(frame_channels, encoder_channels[0], dropout)
        self.down = nn.ModuleList([
            DownBlock(encoder_channels[i - 1], encoder_channels[i], dropout)
            for i in range(1, depth)
        ])
        bottleneck_channels = encoder_channels[-1] * 2
        self.bottleneck = DownBlock(encoder_channels[-1], bottleneck_channels, dropout)
        self.temporal = ConvLSTMCell(bottleneck_channels, bottleneck_channels, lstm_kernel)

        current = bottleneck_channels
        decoder = []
        for skip_channels in reversed(encoder_channels):
            decoder.append(UpBlock(current, skip_channels, skip_channels, dropout))
            current = skip_channels
        self.up = nn.ModuleList(decoder)
        self.output = nn.Conv2d(current, self.out_len * self.in_channels, 1)

        if self.residual:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    @staticmethod
    def _coords(batch, height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([yy, xx], dim=0)[None].expand(batch, -1, -1, -1)

    def _feature_mask(self, feature, mask):
        if not self.mask_features or mask is None:
            return feature
        return feature * _resize_mask(mask, feature.shape[-2:], feature.dtype)

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

    def forward(self, x, mask=None):
        batch, time, channels, height, width = x.shape
        if time != self.in_len or channels != self.in_channels:
            raise ValueError(f'UNetConvLSTMForecaster expects T={self.in_len}, C={self.in_channels}; got {tuple(x.shape)}')

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            mask = mask.to(device=x.device, dtype=x.dtype)
            if mask.size(0) == 1:
                mask = mask.expand(batch, -1, -1, -1)

        coords = self._coords(batch, height, width, x.device, x.dtype) if self.use_coord_channels else None
        state = None
        last_skips = None
        for t in range(time):
            bottleneck, skips = self._encode(x[:, t], mask, coords)
            state = self.temporal(bottleneck, state)
            if t == time - 1:
                last_skips = skips

        z = self._feature_mask(state[0], mask)
        for block, skip in zip(self.up, reversed(last_skips)):
            z = self._feature_mask(block(z, skip), mask)

        correction = self.output(z).view(batch, self.out_len, self.in_channels, height, width)
        prediction = x[:, -1:] + correction if self.residual else correction
        if mask is not None:
            prediction = prediction * mask[:, None]
        return prediction
