"""2-D Fourier Neural Operator forecaster with the common Forecast_nn interface.

Input / output are intentionally identical to UNetForecaster:
    x:          [B, T_in, C, H, W]
    mask:       [H, W], [B, H, W], [1, 1, H, W] or [B, 1, H, W]
    prediction: [B, T_out, C, H, W]

The ordered historical sequence is flattened into distinct channels before the
operator. Therefore day 1 and day 30 are not permutation-equivalent inputs. Optional
mask and normalized coordinate channels provide coastline and absolute-position
information. The output is a direct multi-horizon residual forecast.

FFT operations are explicitly evaluated in float32. This is important for AMP on
odd grids such as 81x91, where half-precision cuFFT support is restrictive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Forecast_nn.models.unet import _groups, _resize_mask


class SpectralConv2d(nn.Module):
    """Truncated 2-D Fourier convolution over low-frequency modes."""

    def __init__(self, in_channels, out_channels, modes_y=12, modes_x=12):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_y = int(modes_y)
        self.modes_x = int(modes_x)

        if self.modes_y < 1 or self.modes_x < 1:
            raise ValueError('modes_y and modes_x must be positive.')

        scale = 1.0 / max(self.in_channels * self.out_channels, 1) ** 0.5
        shape = (self.in_channels, self.out_channels, self.modes_y, self.modes_x)
        self.weight_pos = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _complex_mul(x, weight):
        # x:      B,Cin,my,mx
        # weight: Cin,Cout,my,mx
        # output: B,Cout,my,mx
        return torch.einsum('bixy,ioxy->boxy', x, weight)

    def forward(self, x):
        input_dtype = x.dtype

        # Keep Fourier transforms in float32/complex64 even under CUDA autocast.
        xf = x.float()
        batch, _, height, width = xf.shape
        x_ft = torch.fft.rfft2(xf, norm='ortho')

        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Keep positive and negative y bands disjoint for normal-sized grids.
        max_y = max(height // 2, 1)
        my = min(self.modes_y, max_y)
        mx = min(self.modes_x, width // 2 + 1)

        out_ft[:, :, :my, :mx] = self._complex_mul(
            x_ft[:, :, :my, :mx],
            self.weight_pos[:, :, :my, :mx],
        )
        out_ft[:, :, -my:, :mx] = self._complex_mul(
            x_ft[:, :, -my:, :mx],
            self.weight_neg[:, :, :my, :mx],
        )

        y = torch.fft.irfft2(out_ft, s=(height, width), norm='ortho')
        return y.to(dtype=input_dtype)


class FNOBlock(nn.Module):
    """Residual spectral block with global Fourier and local 1x1 branches."""

    def __init__(self, width, modes_y, modes_x, dropout=0.0):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(width), width)
        self.spectral = SpectralConv2d(width, width, modes_y, modes_x)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        z = self.norm(x)
        z = self.spectral(z) + self.local(z)
        z = self.dropout(self.activation(z))
        return x + z


class FNOForecaster(nn.Module):
    """Direct multi-horizon 2-D Fourier Neural Operator.

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
        width=48,
        modes_y=12,
        modes_x=12,
        layers=4,
        dropout=0.0,
        residual=True,
        use_mask_channel=True,
        use_coord_channels=True,
        mask_features=True,
        projection_width=96,
        padding=6,
    ):
        super().__init__()
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.in_channels = int(in_channels)
        self.residual = bool(residual)
        self.use_mask_channel = bool(use_mask_channel)
        self.use_coord_channels = bool(use_coord_channels)
        self.mask_features = bool(mask_features)
        self.padding = int(padding)

        if self.in_len < 1 or self.out_len < 1:
            raise ValueError('in_len and out_len must be positive.')
        if layers < 1:
            raise ValueError('layers must be >= 1.')
        if self.padding < 0:
            raise ValueError('padding must be >= 0.')

        static_channels = int(self.use_mask_channel) + 2 * int(self.use_coord_channels)
        input_channels = self.in_len * self.in_channels + static_channels

        self.lift = nn.Conv2d(input_channels, int(width), kernel_size=1)
        self.blocks = nn.ModuleList([
            FNOBlock(int(width), int(modes_y), int(modes_x), dropout)
            for _ in range(int(layers))
        ])
        self.project = nn.Sequential(
            nn.Conv2d(int(width), int(projection_width), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(projection_width), self.out_len * self.in_channels, kernel_size=1),
        )

        # Start from persistence, matching UNetForecaster residual initialization.
        if self.residual:
            nn.init.zeros_(self.project[-1].weight)
            nn.init.zeros_(self.project[-1].bias)

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

    def forward(self, x, mask=None):
        if x.ndim != 5:
            raise ValueError(f'x must be [B,T,C,H,W], got shape {tuple(x.shape)}.')

        batch, time, channels, height, width = x.shape
        if time != self.in_len or channels != self.in_channels:
            raise ValueError(
                f'FNOForecaster expects T={self.in_len}, C={self.in_channels}; '
                f'got {tuple(x.shape)}.'
            )

        mask = self._prepare_mask(mask, batch, x)

        # Same ordered-history convention as UNetForecaster: T and C become
        # distinct input channels, so chronological positions are preserved.
        parts = [x.reshape(batch, time * channels, height, width)]
        if self.use_mask_channel:
            if mask is None:
                raise ValueError('use_mask_channel=True requires mask in forward().')
            parts.append(mask)
        if self.use_coord_channels:
            parts.append(self._coords(batch, height, width, x.device, x.dtype))

        z = self.lift(torch.cat(parts, dim=1))
        z = self._feature_mask(z, mask)

        # Zero padding reduces the artificial periodic wrap-around of the Fourier
        # representation at the outer domain boundaries. Crop back afterwards.
        if self.padding > 0:
            z = F.pad(z, (0, self.padding, 0, self.padding))
            block_mask = None
            if mask is not None:
                block_mask = F.pad(mask, (0, self.padding, 0, self.padding))
        else:
            block_mask = mask

        for block in self.blocks:
            z = block(z)
            z = self._feature_mask(z, block_mask)

        if self.padding > 0:
            z = z[..., :height, :width]

        correction = self.project(z).view(
            batch, self.out_len, self.in_channels, height, width
        )
        prediction = x[:, -1:] + correction if self.residual else correction

        if mask is not None:
            prediction = prediction * mask[:, None]
        return prediction
