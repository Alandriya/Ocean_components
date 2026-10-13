"""SDE-informed SimVP-style map forecaster.

For each historical frame, a/b maps modulate the spatial bottleneck through a
small FiLM conditioner. The conditioned sequence then goes through the original
SimVP temporal mixer and T_in -> T_out projection.
"""
import torch

from Forecast_nn.models.simvp import SimVPForecaster
from Forecast_nn.models.sde_conditioning import SDEFiLM2d


class SDESimVPForecaster(SimVPForecaster):
    def __init__(self, *args, sde_condition_scale=0.1, **kwargs):
        base_channels = int(kwargs.get('base_channels', 16))
        depth = int(kwargs.get('depth', 3))
        in_channels = int(kwargs.get('in_channels', 1))
        super().__init__(*args, **kwargs)
        bottleneck_channels = base_channels * (2 ** (depth - 1)) * 2
        self.sde_channels = 2 * in_channels
        self.sde_conditioner = SDEFiLM2d(self.sde_channels, bottleneck_channels, sde_condition_scale)

    def forward(self, x, mask=None, sde_features=None):
        if x.ndim != 5:
            raise ValueError(f'x must be [B,T,C,H,W], got shape {tuple(x.shape)}.')
        batch, time, channels, height, width = x.shape
        if time != self.in_len or channels != self.in_channels:
            raise ValueError(
                f'SDESimVPForecaster expects T={self.in_len}, C={self.in_channels}; got {tuple(x.shape)}.'
            )
        if sde_features is None:
            raise ValueError('SDE-informed model requires sde_features=[B,T,2*C,H,W].')
        if sde_features.shape[:2] != (batch, time) or sde_features.size(2) != self.sde_channels:
            raise ValueError(f'Unexpected sde_features shape {tuple(sde_features.shape)}.')

        mask = self._prepare_mask(mask, batch, x)
        coords = self._coords(batch, height, width, x.device, x.dtype) if self.use_coord_channels else None

        encoded = []
        last_skips = None
        for t in range(time):
            z, skips = self._encode(x[:, t], mask, coords)
            z = self.sde_conditioner(z, sde_features[:, t], mask)
            z = self._feature_mask(z, mask)
            encoded.append(z)
            if t == time - 1:
                last_skips = skips

        z = torch.stack(encoded, dim=1).permute(0, 2, 1, 3, 4).contiguous()
        z = self._sequence_mask(z, mask)
        for block in self.temporal_mixer:
            z = self._sequence_mask(block(z), mask)

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
