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



class OceanMultiFieldReader:
    """Join flux histories along channel axis, without eagerly loading data."""
    def __init__(self, data_files, downsample=1):
        from types import SimpleNamespace
        if len(data_files) != 2:
            raise ValueError('Joint mode requires exactly two flux files')
        self.readers = [OceanFieldReader(f, downsample) for f in data_files]
        first = self.readers[0]
        if any((r.length, r.height, r.width) != (first.length, first.height, first.width)
               for r in self.readers[1:]):
            raise ValueError('Joint flux files have different grids or time lengths')
        self.height, self.width = first.height, first.width
        self.length = first.length
        self.channels = sum(r.channels for r in self.readers)
        if self.channels != 2:
            raise ValueError('Expected one sensible and one latent channel')
        self.data = SimpleNamespace(shape=(self.length, self.channels, self.height, self.width))

    def read_indices(self, indices):
        return np.concatenate([r.read_indices(indices) for r in self.readers], axis=1)

    def read_range(self, start, end):
        return self.read_indices(np.arange(start, end, dtype=np.int64))


def make_field_reader(data_file, downsample=1):
    if isinstance(data_file, (list, tuple)):
        return OceanMultiFieldReader(data_file, downsample)
    return OceanFieldReader(data_file, downsample)


def resolve_flux_source(variable, data_files):
    """The two-channel task uses the same files and dates as solo experiments."""
    if variable == 'joint':
        return [str(data_files[v]) for v in ('sensible', 'latent')]
    return str(data_files[variable])


def source_exists(source):
    if isinstance(source, (list, tuple)):
        return all(Path(item).is_file() for item in source)
    return Path(source).is_file()

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
    reader = make_field_reader(data_file, downsample)
    mean, std = estimate_normalization(reader, mask, 0, train_end, verbose=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, std=std)
    return mean, std, path


class OceanSequenceDataset(Dataset):
    """Return normalized input/target sequences as [T,C,H,W]."""
    def __init__(self, data_file, mask, start_idx, end_idx, in_len, out_len, mean, std, stride=1, downsample=1):
        self.reader = make_field_reader(data_file, downsample)
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


def psnr_file(stats_root, variable, downsample, train_end, percentiles, max_frames):
    low, high = (float(v) for v in percentiles)
    return Path(stats_root) / (
        f'psnr_{variable}_ds{downsample}_train{train_end}_'
        f'p{low:g}-{high:g}_f{int(max_frames)}.npz'
    )


def load_or_create_psnr_ranges(data_file, mask, variable, downsample, train_end,
                               stats_root, percentiles=(0.1, 99.9), max_frames=256,
                               rebuild=False):
    """Estimate channel-wise physical ranges from TRAIN samples only.

    Percentiles are computed from deterministic, evenly spaced training frames,
    rather than from held-out targets or each image (which would change the PSNR
    reference from sample to sample). Returns fixed ranges per channel.
    """
    low, high = map(float, percentiles)
    if not 0 <= low < high <= 100:
        raise ValueError(f'Invalid PSNR percentiles: {percentiles}.')
    if train_end <= 0 or int(max_frames) <= 0:
        raise ValueError('Training interval and PSNR sample count must be positive.')
    path = psnr_file(stats_root, variable, downsample, train_end,
                     percentiles, max_frames)
    if path.exists() and not rebuild:
        with np.load(path) as f:
            return f['data_range'].astype(np.float32), path

    reader = make_field_reader(data_file, downsample)
    if mask.shape != (reader.height, reader.width):
        raise ValueError('PSNR mask/grid mismatch.')
    indices = np.unique(np.linspace(0, train_end - 1,
                                    num=min(train_end, int(max_frames)), dtype=np.int64))
    samples = [[] for _ in range(reader.channels)]
    for start in range(0, len(indices), 8):
        values = reader.read_indices(indices[start:start + 8])
        for c in range(reader.channels):
            sea = values[:, c, mask].reshape(-1)
            sea = sea[np.isfinite(sea)]
            if sea.size:
                samples[c].append(sea)
    ranges = []
    for c in range(reader.channels):
        if not samples[c]:
            raise ValueError(f'No valid ocean pixels for PSNR channel {c}.')
        vals = np.concatenate(samples[c])
        q_low, q_high = np.percentile(vals, [low, high])
        width = float(q_high - q_low)
        if not np.isfinite(width) or width <= 0:
            raise ValueError(f'Non-positive PSNR data range for channel {c}: {width}.')
        ranges.append(width)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = np.asarray(ranges, dtype=np.float32)
    np.savez(path, data_range=result, percentiles=np.asarray([low, high]),
             sampled_frames=indices)
    return result, path
