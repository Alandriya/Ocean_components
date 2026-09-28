import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset


NATIVE_HEIGHT = 161
NATIVE_WIDTH = 181
NATIVE_SPACE_SIZE = NATIVE_HEIGHT * NATIVE_WIDTH


def downsampled_shape(factor=1):
    """Shape produced by the repository-style [::factor, ::factor] spatial slicing."""
    factor = max(1, int(factor))
    return len(range(0, NATIVE_HEIGHT, factor)), len(range(0, NATIVE_WIDTH, factor))


def load_full_mask(root_path=None, mask_file=None, downsample=1):
    """Read the native 161x181 ocean mask and optionally thin it as mask[::factor, ::factor]."""
    path = mask_file or os.path.join(root_path, 'DATA', 'mask')
    mask = np.fromfile(path, dtype=np.bool_, count=NATIVE_SPACE_SIZE).reshape(NATIVE_HEIGHT, NATIVE_WIDTH)
    factor = max(1, int(downsample))
    return mask[::factor, ::factor]


class OceanFieldReader:
    """Memory-mapped reader with optional repository-style spatial downsampling.

    Source files may be flattened native grids `(29141, T)` / `(T, 29141)` or time-first
    native grids. With `downsample=2`, returned frames are 81x91, exactly as in the older
    Forecast_nn preprocessing (`frame[::2, ::2]`). The source .npy file is never copied.

    The mmap object is deliberately NOT pickled. Windows DataLoader workers are created with
    ``spawn``, so each worker reopens the same .npy file read-only on first access.
    """
    def __init__(self, data_file, downsample=1):
        self.data_file = os.fspath(data_file)
        self.downsample = max(1, int(downsample))
        self.height, self.width = downsampled_shape(self.downsample)
        self._data = None
        data = self.data
        out_space = self.height * self.width

        if data.ndim == 2 and data.shape[0] == NATIVE_SPACE_SIZE:
            self.layout, self.length, self.channels, self.native_source = 'space_time', data.shape[1], 1, True
        elif data.ndim == 2 and data.shape[1] == NATIVE_SPACE_SIZE:
            self.layout, self.length, self.channels, self.native_source = 'time_space', data.shape[0], 1, True
        elif data.ndim == 3 and data.shape[-2:] == (NATIVE_HEIGHT, NATIVE_WIDTH):
            self.layout, self.length, self.channels, self.native_source = 'time_hw', data.shape[0], 1, True
        elif data.ndim == 4 and data.shape[-2:] == (NATIVE_HEIGHT, NATIVE_WIDTH):
            self.layout, self.length, self.channels, self.native_source = 'time_chw', data.shape[0], data.shape[1], True
        elif data.ndim == 2 and data.shape[0] == out_space:
            self.layout, self.length, self.channels, self.native_source = 'space_time_small', data.shape[1], 1, False
        elif data.ndim == 2 and data.shape[1] == out_space:
            self.layout, self.length, self.channels, self.native_source = 'time_space_small', data.shape[0], 1, False
        elif data.ndim == 3 and data.shape[-2:] == (self.height, self.width):
            self.layout, self.length, self.channels, self.native_source = 'time_hw_small', data.shape[0], 1, False
        elif data.ndim == 4 and data.shape[-2:] == (self.height, self.width):
            self.layout, self.length, self.channels, self.native_source = 'time_chw_small', data.shape[0], data.shape[1], False
        else:
            raise ValueError(f'Unsupported data shape {data.shape}; expected native 161x181 data or an already downsampled {self.height}x{self.width} grid.')

    @property
    def data(self):
        if self._data is None:
            self._data = np.load(self.data_file, mmap_mode='r')
        return self._data

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_data'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._data = None

    def read_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        data = self.data
        if self.layout == 'space_time':
            arr = np.asarray(data[:, indices], dtype=np.float32).T.reshape(len(indices), 1, NATIVE_HEIGHT, NATIVE_WIDTH)
        elif self.layout == 'time_space':
            arr = np.asarray(data[indices], dtype=np.float32).reshape(len(indices), 1, NATIVE_HEIGHT, NATIVE_WIDTH)
        elif self.layout == 'time_hw':
            arr = np.asarray(data[indices], dtype=np.float32)[:, None]
        elif self.layout == 'time_chw':
            arr = np.asarray(data[indices], dtype=np.float32)
        elif self.layout == 'space_time_small':
            arr = np.asarray(data[:, indices], dtype=np.float32).T.reshape(len(indices), 1, self.height, self.width)
        elif self.layout == 'time_space_small':
            arr = np.asarray(data[indices], dtype=np.float32).reshape(len(indices), 1, self.height, self.width)
        elif self.layout == 'time_hw_small':
            arr = np.asarray(data[indices], dtype=np.float32)[:, None]
        else:
            arr = np.asarray(data[indices], dtype=np.float32)

        if self.native_source and self.downsample > 1:
            arr = arr[..., ::self.downsample, ::self.downsample]
        return arr

    def read_range(self, start, end):
        return self.read_indices(np.arange(start, end, dtype=np.int64))


def _log_progress(label, done, total, started):
    elapsed = max(time.time() - started, 1e-9)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    print(f'    {label}: {done}/{total} ({100.0 * done / max(total, 1):5.1f}%), elapsed={elapsed / 60:.1f} min, ETA={eta / 60:.1f} min', flush=True)


def estimate_normalization(reader, mask, start_idx, end_idx, chunk=32, verbose=False, progress_every=25):
    """Estimate mean/std from training ocean points only."""
    sums = np.zeros(reader.channels, dtype=np.float64)
    sums2 = np.zeros(reader.channels, dtype=np.float64)
    counts = np.zeros(reader.channels, dtype=np.int64)
    stop = min(end_idx, reader.length)
    starts = list(range(start_idx, stop, chunk))
    started = time.time()
    for i, start in enumerate(starts, 1):
        part = reader.read_range(start, min(start + chunk, stop))
        for c in range(reader.channels):
            values = part[:, c][:, mask]
            values = values[np.isfinite(values)]
            sums[c] += values.sum(dtype=np.float64)
            sums2[c] += np.square(values, dtype=np.float64).sum(dtype=np.float64)
            counts[c] += values.size
        if verbose and (i == 1 or i % progress_every == 0 or i == len(starts)):
            _log_progress('normalization', i, len(starts), started)
    mean = sums / counts
    var = np.maximum(sums2 / counts - mean ** 2, 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _quantile_edges(reader, mask, start_idx, end_idx, mean, std, bins, stride, max_samples=1000000, seed=2025, verbose=False):
    rng = np.random.default_rng(seed)
    valid_y, valid_x = np.where(mask)
    n_frames = min(256, max(1, end_idx - start_idx - stride))
    frames = np.linspace(start_idx, end_idx - stride - 1, n_frames, dtype=int)
    per_frame = max(1, max_samples // n_frames)
    samples = [[] for _ in range(reader.channels)]
    started = time.time()
    for i, t in enumerate(frames, 1):
        frame = reader.read_indices([t])[0]
        choose = rng.integers(0, len(valid_y), size=min(per_frame, len(valid_y)))
        for c in range(reader.channels):
            values = frame[c, valid_y[choose], valid_x[choose]]
            values = values[np.isfinite(values)]
            samples[c].append((values - mean[c]) / std[c])
        if verbose and (i == 1 or i % 64 == 0 or i == len(frames)):
            _log_progress('quantile sampling', i, len(frames), started)
    edges = np.zeros((reader.channels, bins + 1), dtype=np.float32)
    for c in range(reader.channels):
        values = np.concatenate(samples[c])
        e = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)).astype(np.float32)
        for i in range(1, len(e)):
            if e[i] <= e[i - 1]:
                e[i] = np.nextafter(e[i - 1], np.float32(np.inf))
        edges[c] = e
    return edges


def estimate_sde_statistics(reader, mask, start_idx, end_idx, mean, std, bins=500, stride=1, chunk=16, verbose=False, progress_every=50):
    """Estimate a(x)=E[dX|X] and b(x)=Var[dX|X] on the training period."""
    if verbose:
        print(f'  Estimating {bins} SDE bins (stride={stride})...', flush=True)
    edges = _quantile_edges(reader, mask, start_idx, end_idx, mean, std, bins, stride, verbose=verbose)
    counts = np.zeros((reader.channels, bins), dtype=np.int64)
    sum_delta = np.zeros((reader.channels, bins), dtype=np.float64)
    sum_delta2 = np.zeros((reader.channels, bins), dtype=np.float64)
    stop = min(end_idx, reader.length) - stride
    starts = list(range(start_idx, stop, chunk))
    started = time.time()
    for i, start in enumerate(starts, 1):
        finish = min(start + chunk, stop)
        x = reader.read_range(start, finish)
        y = reader.read_range(start + stride, finish + stride)
        for c in range(reader.channels):
            x_c = x[:, c][:, mask].reshape(-1)
            y_c = y[:, c][:, mask].reshape(-1)
            valid = np.isfinite(x_c) & np.isfinite(y_c)
            x_n = (x_c[valid] - mean[c]) / std[c]
            delta = (y_c[valid] - x_c[valid]) / std[c]
            idx = np.searchsorted(edges[c, 1:-1], x_n, side='right')
            counts[c] += np.bincount(idx, minlength=bins)
            sum_delta[c] += np.bincount(idx, weights=delta, minlength=bins)
            sum_delta2[c] += np.bincount(idx, weights=delta * delta, minlength=bins)
        if verbose and (i == 1 or i % progress_every == 0 or i == len(starts)):
            _log_progress('SDE a/b accumulation', i, len(starts), started)
    a = np.zeros_like(sum_delta, dtype=np.float32)
    b = np.zeros_like(sum_delta, dtype=np.float32)
    for c in range(reader.channels):
        nonempty = counts[c] > 0
        a[c, nonempty] = (sum_delta[c, nonempty] / counts[c, nonempty]).astype(np.float32)
        b[c, nonempty] = (sum_delta2[c, nonempty] / counts[c, nonempty] - a[c, nonempty] ** 2).astype(np.float32)
        global_a = float(sum_delta[c].sum() / max(counts[c].sum(), 1))
        global_b = float(sum_delta2[c].sum() / max(counts[c].sum(), 1) - global_a ** 2)
        a[c, ~nonempty] = global_a
        b[c, ~nonempty] = global_b
    return edges, a, np.maximum(b, 1e-5)


def build_statistics(data_file, mask, start_idx, end_idx, bins=500, stride=1, downsample=1, verbose=False):
    reader = OceanFieldReader(data_file, downsample=downsample)
    if verbose:
        print('  Stage A: estimating training mean/std...', flush=True)
    mean, std = estimate_normalization(reader, mask, start_idx, end_idx, verbose=verbose)
    if verbose:
        print(f'  mean={mean.tolist()}, std={std.tolist()}', flush=True)
        print('  Stage B: estimating empirical SDE drift a(x) and variance b(x)...', flush=True)
    edges, a, b = estimate_sde_statistics(reader, mask, start_idx, end_idx, mean, std, bins, stride, verbose=verbose)
    if verbose:
        print('  SDE statistics are ready.', flush=True)
    return {'mean': mean, 'std': std, 'bin_edges': edges, 'a': a, 'b': b}


def save_statistics(path, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **stats)


def load_statistics(path):
    with np.load(path) as f:
        stats = {k: f[k] for k in f.files}
    if 'a' not in stats and 'drift' in stats:
        stats['a'] = stats['drift']
    if 'b' not in stats and 'variance' in stats:
        stats['b'] = stats['variance']
    return stats


class OceanSequenceDataset(Dataset):
    """Sequence loader returning time x channel x H x W tensors."""
    def __init__(self, data_file, mask, start_idx, end_idx, in_len, out_len, mean, std, stride=1, downsample=1):
        self.reader = OceanFieldReader(data_file, downsample=downsample)
        self.mask = mask.astype(bool)
        self.start_idx = start_idx
        self.end_idx = min(end_idx, self.reader.length)
        self.in_len = in_len
        self.out_len = out_len
        self.stride = stride
        self.downsample = max(1, int(downsample))
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.channels = self.reader.channels

    def __len__(self):
        return max(0, self.end_idx - self.start_idx - (self.in_len + self.out_len - 1) * self.stride)

    def __getitem__(self, index):
        indices = self.start_idx + index + np.arange(self.in_len + self.out_len) * self.stride
        seq = self.reader.read_indices(indices)
        seq = (seq - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        seq *= self.mask[None, None]
        return torch.from_numpy(seq[:self.in_len].copy()), torch.from_numpy(seq[self.in_len:].copy())
