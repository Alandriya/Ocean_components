"""Ito-SDE utilities for time-aligned coefficient maps.

The current project stores precomputed coefficient maps in four NPY files:

    {files_path_prefix}/Components/sensible-latent/A1_1-17106.npy
    {files_path_prefix}/Components/sensible-latent/B1_1-17106.npy
    {files_path_prefix}/Components/sensible-latent/A2_1-17106.npy
    {files_path_prefix}/Components/sensible-latent/B2_1-17106.npy

where A1/B1 correspond to sensible flux and A2/B2 to latent flux.

Time alignment is deliberately causal. A_t and B_t were estimated from the
observed transition X_t - X_{t-1}; therefore coefficient file element 0 is
attached to state index t=1, not t=0.  During forecasting we only feed
coefficient maps aligned with historical input states. Future A/B maps are never
read for model input or for the SDE loss.
"""
from pathlib import Path
import os
import time
import numpy as np
import torch

from Forecast_nn.data import OceanFieldReader, OceanSequenceDataset


class SDEFieldReader:
    """Read time-aligned A/B coefficient maps without future leakage.

    Parameters
    ----------
    source : dict
        Must contain ``a_file`` and ``b_file``. ``time_offset`` defaults to 1,
        meaning coefficient position 0 is A_1/B_1 and corresponds to the
        already-observed transition X_1-X_0.
    feature_stats : dict
        Training-only statistics used to standardize A/B conditioning features.
    state_std : array-like
        Standard deviation used to normalize the forecasted flux. It is used to
        convert A/B into the normalized model state units for the SDE penalty.
    b_kind : {'diffusion', 'variance'}
        Interpretation of the saved B field.
    """

    def __init__(self, source, *, downsample=1, b_kind='diffusion',
                 feature_clip=5.0, feature_stats=None, state_std=None):
        source = dict(source or {})
        if 'a_file' not in source or 'b_file' not in source:
            raise ValueError('SDE field source must contain a_file and b_file.')

        self.a_file = os.fspath(source['a_file'])
        self.b_file = os.fspath(source['b_file'])
        self.time_offset = int(source.get('time_offset', 1))
        self.b_kind = str(b_kind).lower()
        if self.b_kind not in ('variance', 'diffusion'):
            raise ValueError("b_kind must be 'variance' or 'diffusion'.")
        self.feature_clip = float(feature_clip)

        if not Path(self.a_file).exists():
            raise FileNotFoundError(f'SDE A file not found: {self.a_file}')
        if not Path(self.b_file).exists():
            raise FileNotFoundError(f'SDE B file not found: {self.b_file}')

        self.a_reader = OceanFieldReader(self.a_file, downsample)
        self.b_reader = OceanFieldReader(self.b_file, downsample)
        if self.a_reader.length != self.b_reader.length:
            raise ValueError(
                f'A/B time lengths differ: {self.a_reader.length} vs {self.b_reader.length}.'
            )
        if self.a_reader.channels != self.b_reader.channels:
            raise ValueError(
                f'A/B channel counts differ: {self.a_reader.channels} vs {self.b_reader.channels}.'
            )
        if (self.a_reader.height, self.a_reader.width) != (self.b_reader.height, self.b_reader.width):
            raise ValueError('A/B spatial shapes differ.')

        self.length = self.a_reader.length
        self.channels = self.a_reader.channels
        self.height = self.a_reader.height
        self.width = self.a_reader.width

        self.feature_stats = None
        if feature_stats is not None:
            self.set_feature_stats(feature_stats)

        self.state_std = None
        if state_std is not None:
            self.set_state_std(state_std)

    def __getstate__(self):
        # OceanFieldReader already drops its mmap handles when pickled.
        return self.__dict__.copy()

    def set_feature_stats(self, feature_stats):
        stats = {k: np.asarray(v, dtype=np.float32) for k, v in dict(feature_stats).items()}
        required = ('a_mean', 'a_std', 'b_feature_mean', 'b_feature_std')
        missing = [k for k in required if k not in stats]
        if missing:
            raise KeyError(f'Missing SDE feature statistics: {missing}')
        self.feature_stats = stats

    def set_state_std(self, state_std):
        state_std = np.asarray(state_std, dtype=np.float32).reshape(-1)
        if state_std.size not in (1, self.channels):
            raise ValueError(
                f'state_std has {state_std.size} channels; SDE maps have {self.channels}.'
            )
        if state_std.size == 1 and self.channels > 1:
            state_std = np.repeat(state_std, self.channels)
        self.state_std = np.maximum(state_std, 1e-8).astype(np.float32)

    @property
    def first_flux_index(self):
        return self.time_offset

    @property
    def last_flux_index(self):
        # Inclusive flux-state index represented by the final coefficient map.
        return self.time_offset + self.length - 1

    def describe(self):
        return {
            'a_file': self.a_file,
            'b_file': self.b_file,
            'time_offset': self.time_offset,
            'coefficient_maps': self.length,
            'flux_index_coverage': (self.first_flux_index, self.last_flux_index),
            'channels': self.channels,
            'grid': (self.height, self.width),
            'b_kind': self.b_kind,
        }

    def _b_feature(self, b):
        if self.b_kind == 'variance':
            return np.sqrt(np.maximum(b, 0.0))
        return np.abs(b)

    def read_coefficient_indices(self, coefficient_indices):
        coefficient_indices = np.asarray(coefficient_indices, dtype=np.int64)
        if coefficient_indices.size == 0:
            shape = (0, self.channels, self.height, self.width)
            return np.empty(shape, np.float32), np.empty(shape, np.float32)
        if coefficient_indices.min() < 0 or coefficient_indices.max() >= self.length:
            raise IndexError(
                f'Coefficient indices [{coefficient_indices.min()}, {coefficient_indices.max()}] '
                f'outside [0, {self.length - 1}].'
            )
        return (
            self.a_reader.read_indices(coefficient_indices).astype(np.float32, copy=False),
            self.b_reader.read_indices(coefficient_indices).astype(np.float32, copy=False),
        )

    def read_for_flux_indices(self, flux_indices):
        """Return A/B maps aligned to observed X indices.

        With ``time_offset=1``:
            X_0 -> unavailable (zeros, valid=False)
            X_1 -> coefficient element 0 == A_1/B_1, estimated from X_1-X_0
            X_t -> coefficient element t-1

        Thus A_{t+1}/B_{t+1}, which use the future transition X_{t+1}-X_t,
        are never attached to X_t.
        """
        flux_indices = np.asarray(flux_indices, dtype=np.int64)
        out_shape = (len(flux_indices), self.channels, self.height, self.width)
        a = np.zeros(out_shape, dtype=np.float32)
        b = np.zeros(out_shape, dtype=np.float32)

        coefficient_indices = flux_indices - self.time_offset
        valid = (coefficient_indices >= 0) & (coefficient_indices < self.length)
        if np.any(valid):
            av, bv = self.read_coefficient_indices(coefficient_indices[valid])
            a[valid] = av
            b[valid] = bv
        return a, b, valid

    def conditioning_features_numpy(self, flux_indices, mask=None):
        if self.feature_stats is None:
            raise RuntimeError('SDE feature_stats are required for conditioning.')
        a, b, valid = self.read_for_flux_indices(flux_indices)
        s = self.feature_stats
        shape = (1, self.channels, 1, 1)
        a_z = (a - s['a_mean'].reshape(shape)) / np.maximum(s['a_std'].reshape(shape), 1e-8)
        bf = self._b_feature(b)
        b_z = (bf - s['b_feature_mean'].reshape(shape)) / np.maximum(
            s['b_feature_std'].reshape(shape), 1e-8
        )
        features = np.concatenate([a_z, b_z], axis=1).astype(np.float32, copy=False)
        if self.feature_clip > 0:
            np.clip(features, -self.feature_clip, self.feature_clip, out=features)
        # The only normally unavailable map is X_0. Zero means "neutral/average"
        # after standardization and, critically, does not use A_1/B_1 in advance.
        features[~valid] = 0.0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if mask is not None:
            features *= np.asarray(mask, dtype=np.float32)[None, None]
        return features

    def loss_coefficients_numpy(self, flux_index, mask=None):
        """Return lagged A_t/B_t in *normalized model-state units*.

        This is called for the last observed state X_t.  It never reads A_{t+1}
        or B_{t+1}. For normalized state Z=(X-mean)/std:

            a_Z = a_X / std(X)
            B_Z = B_X / std(X)              if B is diffusion amplitude
            var_Z = var_X / std(X)^2        if B stores variance
        """
        if self.state_std is None:
            raise RuntimeError('state_std is required for SDE loss coefficients.')
        a, b, valid = self.read_for_flux_indices([int(flux_index)])
        if not bool(valid[0]):
            raise IndexError(
                f'No causal A/B map is available for flux index {flux_index}. '
                f'Available flux indices are {self.first_flux_index}..{self.last_flux_index}.'
            )
        scale = self.state_std.reshape(1, self.channels, 1, 1)
        a_model = a / scale
        if self.b_kind == 'diffusion':
            b_model = b / scale
        else:
            b_model = b / np.square(scale)
        coeff = np.concatenate([a_model, b_model], axis=1)[0].astype(np.float32, copy=False)
        coeff = np.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)
        if mask is not None:
            coeff *= np.asarray(mask, dtype=np.float32)[None]
        return coeff


def _accumulate_stats(reader, mask, coefficient_end, chunk=32, verbose=True):
    channels = reader.channels
    sums_a = np.zeros(channels, np.float64)
    sums2_a = np.zeros(channels, np.float64)
    sums_b = np.zeros(channels, np.float64)
    sums2_b = np.zeros(channels, np.float64)
    counts_a = np.zeros(channels, np.int64)
    counts_b = np.zeros(channels, np.int64)

    starts = list(range(0, coefficient_end, chunk))
    started = time.time()
    for i, start in enumerate(starts, 1):
        end = min(start + chunk, coefficient_end)
        a, b = reader.read_coefficient_indices(np.arange(start, end, dtype=np.int64))
        bf = reader._b_feature(b)
        for c in range(channels):
            av = a[:, c][:, mask]
            bv = bf[:, c][:, mask]
            av = av[np.isfinite(av)]
            bv = bv[np.isfinite(bv)]
            sums_a[c] += av.sum(dtype=np.float64)
            sums2_a[c] += np.square(av, dtype=np.float64).sum(dtype=np.float64)
            counts_a[c] += av.size
            sums_b[c] += bv.sum(dtype=np.float64)
            sums2_b[c] += np.square(bv, dtype=np.float64).sum(dtype=np.float64)
            counts_b[c] += bv.size
        if verbose and (i == 1 or i % 50 == 0 or i == len(starts)):
            print(
                f'    SDE normalization {i}/{len(starts)} '
                f'({100*i/max(len(starts),1):.1f}%), elapsed={(time.time()-started)/60:.1f} min',
                flush=True,
            )

    a_mean = sums_a / np.maximum(counts_a, 1)
    a_var = np.maximum(sums2_a / np.maximum(counts_a, 1) - a_mean**2, 1e-12)
    b_mean = sums_b / np.maximum(counts_b, 1)
    b_var = np.maximum(sums2_b / np.maximum(counts_b, 1) - b_mean**2, 1e-12)
    return {
        'a_mean': a_mean.astype(np.float32),
        'a_std': np.sqrt(a_var).astype(np.float32),
        'b_feature_mean': b_mean.astype(np.float32),
        'b_feature_std': np.sqrt(b_var).astype(np.float32),
    }


def sde_normalization_file(stats_root, variable, downsample, train_end, b_kind, time_offset):
    return Path(stats_root) / (
        f'sde_field_norm_{variable}_ds{downsample}_train{train_end}_'
        f'{b_kind}_offset{time_offset}.npz'
    )


def load_or_create_sde_field_normalization(source, mask, variable, downsample,
                                           train_end, stats_root, b_kind='diffusion',
                                           feature_clip=5.0, rebuild=False):
    time_offset = int(dict(source).get('time_offset', 1))
    path = sde_normalization_file(stats_root, variable, downsample, train_end, b_kind, time_offset)
    if path.exists() and not rebuild:
        with np.load(path) as f:
            stats = {k: f[k].astype(np.float32) for k in f.files}
        return stats, path

    reader = SDEFieldReader(
        source,
        downsample=downsample,
        b_kind=b_kind,
        feature_clip=feature_clip,
    )
    # train_end is exclusive in the flux series.  A_1 corresponds to flux index 1,
    # so flux indices [time_offset, train_end-1] map to coefficient positions
    # [0, train_end-time_offset-1].
    coefficient_end = min(reader.length, max(0, int(train_end) - reader.time_offset))
    if coefficient_end <= 0:
        raise ValueError('No causal SDE coefficient maps overlap the training interval.')
    stats = _accumulate_stats(reader, mask, coefficient_end, verbose=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)
    return stats, path


class SDEOceanSequenceDataset(OceanSequenceDataset):
    """Ocean sequence dataset plus causally aligned historical A/B maps."""

    def __init__(self, *args, sde_reader, **kwargs):
        super().__init__(*args, **kwargs)
        self.sde_reader = sde_reader
        if self.sde_reader.channels != self.channels:
            raise ValueError(
                f'Flux data has C={self.channels}, SDE maps have C={self.sde_reader.channels}.'
            )

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        all_indices = self.start_idx + index + np.arange(self.in_len + self.out_len) * self.stride
        input_indices = all_indices[:self.in_len]
        sde_features = self.sde_reader.conditioning_features_numpy(input_indices, self.mask)
        # Only the coefficient map attached to the last OBSERVED input is returned
        # for the SDE loss. Target/future coefficient maps are never accessed.
        sde_last = self.sde_reader.loss_coefficients_numpy(input_indices[-1], self.mask)
        return (
            x,
            y,
            torch.from_numpy(sde_features.copy()),
            torch.from_numpy(sde_last.copy()),
        )


def source_to_serializable(source):
    out = {}
    for key, value in dict(source).items():
        out[key] = os.fspath(value) if isinstance(value, (str, os.PathLike, Path)) else value
    return out
