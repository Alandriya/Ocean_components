"""SDE-informed U-Net + ConvLSTM.

The base architecture is unchanged, but every encoded historical state is
modulated at the U-Net bottleneck by a(x), b(x) fields before it enters ConvLSTM.
"""
import torch

from Forecast_nn.models.unet_convlstm import UNetConvLSTMForecaster
from Forecast_nn.models.sde_conditioning import SDEFiLM2d


class SDEUNetConvLSTMForecaster(UNetConvLSTMForecaster):
    def __init__(self, *args, sde_condition_scale=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        bottleneck_channels = self.temporal.hidden_channels
        self.sde_channels = 2 * self.in_channels
        self.sde_conditioner = SDEFiLM2d(self.sde_channels, bottleneck_channels, sde_condition_scale)

    def forward(self, x, mask=None, sde_features=None):
        batch, time, channels, height, width = x.shape
        if time != self.in_len or channels != self.in_channels:
            raise ValueError(f'SDEUNetConvLSTMForecaster expects T={self.in_len}, C={self.in_channels}; got {tuple(x.shape)}')
        if sde_features is None:
            raise ValueError('SDE-informed model requires sde_features=[B,T,2*C,H,W].')
        if sde_features.shape[:2] != (batch, time) or sde_features.size(2) != self.sde_channels:
            raise ValueError(f'Unexpected sde_features shape {tuple(sde_features.shape)}.')

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
            bottleneck = self.sde_conditioner(bottleneck, sde_features[:, t], mask)
            bottleneck = self._feature_mask(bottleneck, mask)
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
