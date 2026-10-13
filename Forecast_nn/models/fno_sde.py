"""SDE-informed Fourier Neural Operator.

The a(x), b(x) fields are treated as physical coefficient fields and concatenated
with the ordered flux history before the 1x1 lifting layer. By default the full
SDE history is used; setting sde_history=False uses only a/b for the last input day.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from Forecast_nn.models.fno import FNOForecaster


class SDEFNOForecaster(FNOForecaster):
    def __init__(self, *args, sde_history=True, **kwargs):
        in_len = int(kwargs.get('in_len', args[0] if len(args) > 0 else 0))
        in_channels = int(kwargs.get('in_channels', 1))
        width = int(kwargs.get('width', 48))
        super().__init__(*args, **kwargs)
        self.sde_history = bool(sde_history)
        self.sde_channels = 2 * in_channels

        static_channels = int(self.use_mask_channel) + 2 * int(self.use_coord_channels)
        sde_input_channels = self.sde_channels * (in_len if self.sde_history else 1)
        total_input_channels = in_len * in_channels + static_channels + sde_input_channels
        self.lift = nn.Conv2d(total_input_channels, width, kernel_size=1)

    def forward(self, x, mask=None, sde_features=None):
        if x.ndim != 5:
            raise ValueError(f'x must be [B,T,C,H,W], got shape {tuple(x.shape)}.')
        batch, time, channels, height, width = x.shape
        if time != self.in_len or channels != self.in_channels:
            raise ValueError(
                f'SDEFNOForecaster expects T={self.in_len}, C={self.in_channels}; got {tuple(x.shape)}.'
            )
        if sde_features is None:
            raise ValueError('SDE-informed model requires sde_features=[B,T,2*C,H,W].')
        if sde_features.shape[:2] != (batch, time) or sde_features.size(2) != self.sde_channels:
            raise ValueError(f'Unexpected sde_features shape {tuple(sde_features.shape)}.')

        mask = self._prepare_mask(mask, batch, x)
        if mask is not None:
            sde_features = sde_features * mask[:, None]

        parts = [x.reshape(batch, time * channels, height, width)]
        if self.sde_history:
            parts.append(sde_features.reshape(batch, time * self.sde_channels, height, width))
        else:
            parts.append(sde_features[:, -1])
        if self.use_mask_channel:
            if mask is None:
                raise ValueError('use_mask_channel=True requires mask in forward().')
            parts.append(mask)
        if self.use_coord_channels:
            parts.append(self._coords(batch, height, width, x.device, x.dtype))

        z = self.lift(torch.cat(parts, dim=1))
        z = self._feature_mask(z, mask)

        if self.padding > 0:
            z = F.pad(z, (0, self.padding, 0, self.padding))
            block_mask = F.pad(mask, (0, self.padding, 0, self.padding)) if mask is not None else None
        else:
            block_mask = mask

        for block in self.blocks:
            z = block(z)
            z = self._feature_mask(z, block_mask)

        if self.padding > 0:
            z = z[..., :height, :width]

        correction = self.project(z).view(batch, self.out_len, self.in_channels, height, width)
        prediction = x[:, -1:] + correction if self.residual else correction
        if mask is not None:
            prediction = prediction * mask[:, None]
        return prediction
