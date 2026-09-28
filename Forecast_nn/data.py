"""Memory-efficient data access for 2-D ocean fields."""
from pathlib import Path
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset

NATIVE_HEIGHT = 161
NATIVE_WIDTH = 181
NATIVE_SPACE_SIZE = NATIVE_HEIGHT * NATIVE_WIDTH


def downsampled_shape(factor=1):
    factor = max(1, int(factor))
    return len(range(0, NATIVE_HEIGHT, factor)), len(range(0, NATIVE_WIDTH, factor))


def load_mask(mask_file, downsample=1):
    mask = np.fromfile(os.fspath(mask_file), dtype=np.bool_, count=NATIVE_SPACE_SIZE)
    if mask.size != NATIVE_SPACE_SIZE:
        raise ValueError(f'Mask {mask_file} contains {mask.size} values, expected {NATIVE_SPACE_SIZE}.')
    mask = mask.reshape(NATIVE_HEIGHT, NATIVE_WIDTH)
    factor = max(1, int(downsample))
    return mask[::factor, ::factor]


class OceanFieldReader:
    """Read native or already-downsampled arrays through numpy mmap.

    Supported layouts include the project format (29141, time), its transpose,
    and time-first HxW / CxHxW arrays. The mmap object is excluded from pickling,
    so Windows DataLoader workers reopen the file themselves instead of trying to
    send a multi-GB mmap through multiprocessing pipes.
    """
    def __init__(self, data_file, downsample=1):
        self.data_file = os.fspath(data_file)
        self.downsample = max(1, int(downsample))
        self.height, self.width = downsampled_shape(self.downsample)
        self._data = None
        data = self.data
        out_space = self.height * self.width

        if data.ndim == 2 and data.shape[0] == NATIVE_SPACE_SIZE:
            self.layout, self.length, self.channels, self.native = 'space_time', data.shape[1], 1, True
        elif data.ndim == 2 and data.shape[1] == NATIVE_SPACE_SIZE:
            self.layout, self.length, self.channels, self.native = 'time_space', data.shape[0], 1, True
        elif data.ndim == 3 and data.shape[-2:] == (NATIVE_HEIGHT, NATIVE_WIDTH):
            self.layout, self.length, self.channels, self.native = 'time_hw', data.shape[0], 1, True
        elif data.ndim == 4 and data.shape[-2:] == (NATIVE_HEIGHT, NATIVE_WIDTH):
            self.layout, self.length, self.channels, self.native = 'time_chw', data.shape[0], data.shape[1], True
        elif data.ndim == 2 and data.shape[0] == out_space:
            self.layout, self.length, self.channels, self.native = 'space_time_small', data.shape[1], 1, False
        elif data.ndim == 2 and data.shape[1] == out_space:
            self.layout, self.length, self.channels, self.native = 'time_space_small', data.shape[0], 1, False
        elif data.ndim == 3 and data.shape[-2:] == (self.height, self.width):
            self.layout, self.length, self.channels, self.native = 'time_hw_small', data.shape[0], 1, False
        elif data.ndim == 4 and data.shape[-2:] == (self.height, self.width):
            self.layout, self.length, self.channels, self.native = 'time_chw_small', data.shape[0], data.shape[1], False
        else:
            raise ValueError(f'Unsupported data shape {data.shape}.')

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
        d = self.data
        if self.layout == 'space_time':
            arr = np.asarray(d[:, indices], dtype=np.float32).T.reshape(len(indices), 1, NATIVE_HEIGHT, NATIVE_WIDTH)
        elif self.layout == 'time_space':
            arr = np.asarray(d[indices], dtype=np.float32).reshape(len(indices), 1, NATIVE_HEIGHT, NATIVE_WIDTH)
        elif self.layout == 'time_hw':
            arr = np.asarray(d[indices], dtype=np.float32)[:, None]
        elif self.layout == 'time_chw':
            arr = np.asarray(d[indices], dtype=np.float32)
        elif self.layout == 'space_time_small':
            arr = np.asarray(d[:, indices], dtype=np.float32).T.reshape(len(indices), 1, self.height, self.width)
        elif self.layout == 'time_space_small':
            arr = np.asarray(d[indices], dtype=np.float32).reshape(len(indices), 1, self.height, self.width)
        elif self.layout == 'time_hw_small':
            arr = np.asarray(d[indices], dtype=np.float32)[:, None]
        else:
            arr = np.asarray(d[indices], dtype=np.float32)
        if self.native and self.downsample > 1:
            arr = arr[..., ::self.downsample, ::self.downsample]
        return arr

    def read_range(self, start, end):
        return self.read_indices(np.arange(start, end, dtype=np.int64))


def estimate_normalization(reader, mask, start_idx, end_idx, chunk=32, verbose=True):
    sums = np.zeros(reader.channels, dtype=np.float64)
    sums2 = np.zeros(reader.channels, dtype=np.float64)
    counts = np.zeros(reader.channels, dtype=np.int64)
    starts = list(range(start_idx, min(end_idx, reader.length), chunk))
    started = time.time()
    for i, start in enumerate(starts, 1):
        part = reader.read_range(start, min(start + chunk, end_idx, reader.length))
        for c in range(reader.channels):
            values = part[:, c][:, mask]
            values = values[np.isfinite(values)]
            sums[c] += values.sum(dtype=np.float64)
            sums2[c] += np.square(values, dtype=np.float64).sum(dtype=np.float64)
            counts[c] += values.size
        if verbose and (i == 1 or i % 50 == 0 or i == len(starts)):
            elapsed = max(time.time() - started, 1e-9)
            print(f'    normalization {i}/{len(starts)} ({100*i/max(len(starts),1):.1f}%), elapsed={elapsed/60:.1f} min', flush=True)
    mean = sums / np.maximum(counts, 1)
    var = np.maximum(sums2 / np.maximum(counts, 1) - mean ** 2, 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def normalization_file(stats_root, variable, downsample, train_end):
    return Path(stats_root) / f'norm_{variable}_ds{downsample}_train{train_end}.npz'


def load_or_create_normalization(data_file, mask, variable, downsample, train_end, stats_root, rebuild=False):
    path = normalization_file(stats_root, variable, downsample, train_end)
    if path.exists() and not rebuild:
        with np.load(path) as f:
            return f['mean'].astype(np.float32), f['std'].astype(np.float32), path
    reader = OceanFieldReader(data_file, downsample)
    mean, std = estimate_normalization(reader, mask, 0, train_end, verbose=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, std=std)
    return mean, std, path


class OceanSequenceDataset(Dataset):
    """Return normalized input/target sequences as [T,C,H,W]."""
    def __init__(self, data_file, mask, start_idx, end_idx, in_len, out_len, mean, std, stride=1, downsample=1):
        self.reader = OceanFieldReader(data_file, downsample)
        self.mask = mask.astype(bool)
        self.start_idx = int(start_idx)
        self.end_idx = min(int(end_idx), self.reader.length)
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.stride = int(stride)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.channels = self.reader.channels

    def __len__(self):
        needed_span = (self.in_len + self.out_len - 1) * self.stride
        return max(0, self.end_idx - self.start_idx - needed_span)

    def __getitem__(self, index):
        indices = self.start_idx + index + np.arange(self.in_len + self.out_len) * self.stride
        seq = self.reader.read_indices(indices)
        seq = (seq - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        seq *= self.mask[None, None]
        x = torch.from_numpy(seq[:self.in_len].copy())
        y = torch.from_numpy(seq[self.in_len:].copy())
        return x, y
