"""Utilities for stochastic (Ito-SDE) conditioning and regularization.

The preferred input is the statistics file produced by the earlier SDE-Energy
pipeline. Such files contain:
    bin_edges : [C, N+1]  (normalized state bins)
    a         : [C, N]    empirical drift E[dX | X]
    b         : [C, N]    empirical conditional variance Var[dX | X]

A more generic lookup table is also accepted when an NPZ contains a state grid
(`values`, `x`, `centers` or `bin_centers`) together with `a`/`drift` and
`b`/`variance`/`diffusion`.

All lookups are performed on tensors already in the model's state space. For the
legacy SDE-Energy statistics this is the normalized state space used by the
Forecast_nn datasets, so no denormalization is required.
"""
from pathlib import Path
import os
import numpy as np
import torch


_VALUE_KEYS = ('values', 'x', 'centers', 'bin_centers', 'flux_values', 'state_values')
_A_KEYS = ('a', 'drift', 'mu')
_B_KEYS = ('b', 'variance', 'diffusion', 'sigma2', 'sigma')
_EDGE_KEYS = ('bin_edges', 'edges')


def _first_key(mapping, candidates):
    for key in candidates:
        if key in mapping:
            return key
    return None


def _as_channel_table(arr, channels=None):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.ndim != 2:
        raise ValueError(f'SDE table arrays must be 1-D or 2-D, got shape {arr.shape}.')
    if channels is not None and arr.shape[0] not in (1, int(channels)):
        raise ValueError(f'SDE table has {arr.shape[0]} channels, expected 1 or {channels}.')
    if channels is not None and arr.shape[0] == 1 and int(channels) > 1:
        arr = np.repeat(arr, int(channels), axis=0)
    return arr


class SDECoefficientLookup:
    """Fixed a(x), b(x) lookup used both for conditioning and SDE loss."""

    def __init__(self, *, bin_edges=None, values=None, a=None, b=None,
                 b_kind='variance', feature_clip=5.0):
        if a is None or b is None:
            raise ValueError('Both SDE coefficients a and b are required.')
        self.a_np = _as_channel_table(a)
        self.b_np = _as_channel_table(b, self.a_np.shape[0])
        if self.a_np.shape != self.b_np.shape:
            raise ValueError(f'a and b shapes differ: {self.a_np.shape} vs {self.b_np.shape}.')
        self.channels = self.a_np.shape[0]
        self.b_kind = str(b_kind).lower()
        if self.b_kind not in ('variance', 'diffusion'):
            raise ValueError("b_kind must be 'variance' or 'diffusion'.")
        self.feature_clip = float(feature_clip)

        self.bin_edges_np = None
        self.values_np = None
        if bin_edges is not None:
            edges = _as_channel_table(bin_edges, self.channels)
            if edges.shape[1] != self.a_np.shape[1] + 1:
                raise ValueError(
                    f'bin_edges must have N+1 entries for N coefficients; '
                    f'got edges={edges.shape}, a={self.a_np.shape}.'
                )
            self.bin_edges_np = edges
            self.mode = 'bins'
        elif values is not None:
            values = _as_channel_table(values, self.channels)
            if values.shape != self.a_np.shape:
                raise ValueError(f'values and a shapes differ: {values.shape} vs {self.a_np.shape}.')
            order = np.argsort(values, axis=1)
            self.values_np = np.take_along_axis(values, order, axis=1)
            self.a_np = np.take_along_axis(self.a_np, order, axis=1)
            self.b_np = np.take_along_axis(self.b_np, order, axis=1)
            self.mode = 'values'
        else:
            raise ValueError('SDE source must contain either bin_edges or state values/centers.')

        # Conditioning uses order-one features. Drift is standardized directly;
        # diffusion information is represented as a standard-deviation-like term.
        b_feature = self._b_feature_numpy(self.b_np)
        self.a_mean = self.a_np.mean(axis=1).astype(np.float32)
        self.a_std = np.maximum(self.a_np.std(axis=1), 1e-6).astype(np.float32)
        self.bf_mean = b_feature.mean(axis=1).astype(np.float32)
        self.bf_std = np.maximum(b_feature.std(axis=1), 1e-6).astype(np.float32)
        self._cache = {}

    @classmethod
    def from_source(cls, source, *, b_kind='variance', feature_clip=5.0):
        """Load from a config dict or directly from an NPZ path.

        Supported config forms:
          {'file': 'stats.npz'}
        or
          {'value_file': 'x.npy', 'a_file': 'a.npy', 'b_file': 'b.npy'}
        """
        if isinstance(source, (str, os.PathLike, Path)):
            source = {'file': source}
        source = dict(source or {})

        file_path = source.get('file')
        if file_path:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(
                    f'SDE coefficient file not found: {path}\n'
                    'Set cfg.sde_sources[variable]["file"] or pass --sde-file.'
                )
            with np.load(path) as f:
                arrays = {k: f[k] for k in f.files}
            edge_key = source.get('edge_key') or _first_key(arrays, _EDGE_KEYS)
            value_key = source.get('value_key') or _first_key(arrays, _VALUE_KEYS)
            a_key = source.get('a_key') or _first_key(arrays, _A_KEYS)
            b_key = source.get('b_key') or _first_key(arrays, _B_KEYS)
            if a_key is None or b_key is None or (edge_key is None and value_key is None):
                raise KeyError(
                    f'Could not identify SDE arrays in {path}. Available keys: {sorted(arrays)}. '
                    'Expected bin_edges (or values/centers), a/drift, and b/variance/diffusion.'
                )
            return cls(
                bin_edges=arrays[edge_key] if edge_key else None,
                values=arrays[value_key] if value_key else None,
                a=arrays[a_key], b=arrays[b_key], b_kind=b_kind,
                feature_clip=feature_clip,
            )

        value_file, a_file, b_file = source.get('value_file'), source.get('a_file'), source.get('b_file')
        if value_file and a_file and b_file:
            return cls(
                values=np.load(value_file, mmap_mode='r'),
                a=np.load(a_file, mmap_mode='r'),
                b=np.load(b_file, mmap_mode='r'),
                b_kind=b_kind, feature_clip=feature_clip,
            )
        raise ValueError('Invalid SDE source. Provide an NPZ file or value_file/a_file/b_file.')

    def _b_feature_numpy(self, b):
        if self.b_kind == 'variance':
            return np.sqrt(np.maximum(b, 0.0))
        return np.abs(b)

    def describe(self):
        return {
            'mode': self.mode,
            'channels': self.channels,
            'bins': int(self.a_np.shape[1]),
            'b_kind': self.b_kind,
            'a_min': float(np.nanmin(self.a_np)),
            'a_max': float(np.nanmax(self.a_np)),
            'b_min': float(np.nanmin(self.b_np)),
            'b_max': float(np.nanmax(self.b_np)),
        }

    def _device_tensors(self, device):
        key = str(device)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = {
            'a': torch.as_tensor(self.a_np, device=device, dtype=torch.float32),
            'b': torch.as_tensor(self.b_np, device=device, dtype=torch.float32),
            'a_mean': torch.as_tensor(self.a_mean, device=device, dtype=torch.float32),
            'a_std': torch.as_tensor(self.a_std, device=device, dtype=torch.float32),
            'bf_mean': torch.as_tensor(self.bf_mean, device=device, dtype=torch.float32),
            'bf_std': torch.as_tensor(self.bf_std, device=device, dtype=torch.float32),
        }
        if self.bin_edges_np is not None:
            result['bin_edges'] = torch.as_tensor(self.bin_edges_np, device=device, dtype=torch.float32)
        if self.values_np is not None:
            result['values'] = torch.as_tensor(self.values_np, device=device, dtype=torch.float32)
        self._cache[key] = result
        return result

    def lookup(self, state):
        """Return a(state), b(state) with the same B,T,C,H,W shape as state."""
        if state.ndim != 5:
            raise ValueError(f'SDE lookup expects [B,T,C,H,W], got {tuple(state.shape)}.')
        if state.size(2) != self.channels:
            raise ValueError(f'SDE lookup has C={self.channels}, state has C={state.size(2)}.')
        x = state.float()
        tensors = self._device_tensors(x.device)
        a_out, b_out = [], []
        for c in range(self.channels):
            xc = x[:, :, c].contiguous()
            if self.mode == 'bins':
                edges = tensors['bin_edges'][c]
                idx = torch.bucketize(xc, edges[1:-1])
                ac = tensors['a'][c][idx]
                bc = tensors['b'][c][idx]
            else:
                values = tensors['values'][c]
                idx_hi = torch.bucketize(xc, values).clamp(1, values.numel() - 1)
                idx_lo = idx_hi - 1
                x0, x1 = values[idx_lo], values[idx_hi]
                w = ((xc - x0) / (x1 - x0).clamp_min(1e-8)).clamp(0.0, 1.0)
                ac = tensors['a'][c][idx_lo] * (1.0 - w) + tensors['a'][c][idx_hi] * w
                bc = tensors['b'][c][idx_lo] * (1.0 - w) + tensors['b'][c][idx_hi] * w
            a_out.append(ac[:, :, None])
            b_out.append(bc[:, :, None])
        return torch.cat(a_out, dim=2), torch.cat(b_out, dim=2)

    def conditioning_features(self, state, mask=None):
        """Return standardized [a, diffusion-scale] maps as [B,T,2*C,H,W]."""
        a, b = self.lookup(state)
        tensors = self._device_tensors(state.device)
        shape = (1, 1, self.channels, 1, 1)
        a_z = (a - tensors['a_mean'].view(shape)) / tensors['a_std'].view(shape)
        if self.b_kind == 'variance':
            bf = torch.sqrt(b.clamp_min(0.0))
        else:
            bf = b.abs()
        b_z = (bf - tensors['bf_mean'].view(shape)) / tensors['bf_std'].view(shape)
        features = torch.cat([a_z, b_z], dim=2)
        if self.feature_clip > 0:
            features = features.clamp(-self.feature_clip, self.feature_clip)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            if mask.ndim != 4:
                raise ValueError(f'mask must be 2-D, 3-D or 4-D, got {mask.ndim}.')
            mask = mask.to(device=features.device, dtype=features.dtype)
            if mask.size(0) == 1 and features.size(0) > 1:
                mask = mask.expand(features.size(0), -1, -1, -1)
            features = features * mask[:, None]
        return features
