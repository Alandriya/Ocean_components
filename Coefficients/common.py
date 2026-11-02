"""Shared input validation, water masks, and quantile grouping for NP and SP.

Conventions
-----------
- Mask value 1 means water (included); 0 means land (excluded).
- Scalar time series are stored as (T, H, W), with H=161 and W=181 by default.
- Vector transitions are represented internally as (H, W, 2).
- Bivariate diffusion channels are ordered [B00, B11, B01, B10].
- Quantile groups depend on the state at t-1, while increments are unquantized.
- The same GroupedTransition can be passed to both estimators.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

GRID_SHAPE = (161, 181)
MASK_WATER = 1
MASK_LAND = 0
B_CHANNELS = ("00", "11", "01", "10")


@dataclass(frozen=True)
class Group:
    bin_id: int
    flat_indices: np.ndarray
    increments: np.ndarray  # (n,) for 1D; (n, 2) for 2D.


@dataclass(frozen=True)
class GroupedTransition:
    shape: Tuple[int, int]
    groups: Tuple[Group, ...]
    n_valid_pairs: int
    n_total_positions: int
    n_requested_bins: int
    n_occupied_bins: int
    n_singleton_bins: int
    edges: object
    dimension: int = 1


def load_water_mask(mask: Optional[Union[str, Path, np.ndarray]],
                    shape: Tuple[int, int] = GRID_SHAPE) -> np.ndarray:
    """Load a binary water mask from .npy, raw byte flags, or an array.

    All mask entries must be 0/1 or bool. A missing mask means all water.
    A one-dimensional mask is reshaped in C order to ``shape``.
    Raw masks must contain exactly H*W bytes (one bool per position).
    """
    if mask is None:
        return np.ones(shape, dtype=bool)
    if isinstance(mask, (str, Path)):
        path = Path(mask)
        if path.suffix.lower() == '.npy':
            raw = np.load(path, allow_pickle=False)
        else:
            raw = np.fromfile(path, dtype=np.uint8)
    else:
        raw = np.asarray(mask)
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"Water mask needs {shape} = {np.prod(shape)} entries; received {raw.shape}")
    if raw.shape not in (shape, (int(np.prod(shape)),)):
        raise ValueError(f"Water mask shape must be {shape} or {(int(np.prod(shape)),)}")
    if raw.dtype.kind not in 'biuf' or not np.all(np.isin(raw, (0, 1))):
        raise ValueError('Water mask must contain only 1 (water) and 0 (land)')
    return raw.reshape(shape).astype(bool, copy=False)


def load_series(data: Union[str, Path, np.ndarray],
                grid_shape: Tuple[int, int] = GRID_SHAPE) -> np.ndarray:
    """Return a time-first view of a scalar time series without copying full data.

    Accepted shapes: (T,H,W), (H*W,T), and (T,H*W).
    The (H*W,T) convention is checked before (T,H*W). Ambiguous square
    flattened arrays are rejected. Files must be uncompressed .npy arrays.
    Land is not replaced globally; pass the mask to the estimators instead.
    """
    source = np.load(data, mmap_mode='r', allow_pickle=False) if isinstance(data, (str, Path)) else np.asarray(data)
    if source.dtype.kind not in 'biuf':
        raise TypeError(f"Expected a numeric array; received {source.dtype}")
    h, w = grid_shape
    cells = h * w
    if source.ndim == 3:
        if source.shape[1:] != grid_shape:
            raise ValueError(f"Expected (T,{h},{w}); received {source.shape}")
        result = source
    elif source.ndim == 2:
        if source.shape == (cells, cells):
            raise ValueError('Ambiguous flattened time orientation: both dimensions equal H*W')
        if source.shape[0] == cells:
            result = source.T.reshape((-1, h, w))
        elif source.shape[1] == cells:
            result = source.reshape((-1, h, w))
        else:
            raise ValueError(f"Expected ({cells},T), (T,{cells}), or (T,{h},{w}); received {source.shape}")
    else:
        raise ValueError(f"Expected a 2D flattened or 3D time-first array; received {source.shape}")
    if result.shape[0] < 2:
        raise ValueError('At least two time points are needed')
    return result


def coerce_series(source, grid_shape: Tuple[int, int] = GRID_SHAPE):
    """Normalize an array or filename; permit arbitrary H,W for 3D in-memory tests."""
    if isinstance(source, (str, Path)):
        return load_series(source, grid_shape)
    array = np.asarray(source)
    return load_series(array, array.shape[1:] if array.ndim == 3 else grid_shape)


def load_pair_series(data_0, data_1, grid_shape: Tuple[int, int] = GRID_SHAPE):
    """Load two aligned scalar series; return a pair of (T,H,W) arrays.

    Using a tuple instead of a full (T,H,W,2) stack avoids unnecessary
    memory allocation on long ERA5 time series.
    """
    first, second = coerce_series(data_0, grid_shape), coerce_series(data_1, grid_shape)
    if first.shape != second.shape:
        raise ValueError(f"Two series must have equal shapes: {first.shape} != {second.shape}")
    return first, second


def check_series(data_array, time_start: int, time_end: int) -> None:
    if data_array.ndim != 3:
        raise ValueError('Expected time series with shape (T,H,W)')
    if not 0 <= time_start < time_end <= data_array.shape[0] or time_end - time_start < 2:
        raise ValueError('Expected 0 <= time_start < time_end <= T and at least one transition')


def check_pair_series(first, second, time_start, time_end) -> None:
    check_series(first, time_start, time_end)
    check_series(second, time_start, time_end)
    if first.shape != second.shape:
        raise ValueError('Both variables must have the same (T,H,W) shape')


def _bin_ids(values: np.ndarray, n_bins: int):
    if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, (bool, np.bool_)) or n_bins < 1:
        raise ValueError('Number of quantile groups must be a positive integer')
    edges = np.quantile(values, np.linspace(0.0, 1.0, int(n_bins) + 1))
    ids = np.searchsorted(edges, values, side='right') - 1
    return np.clip(ids, 0, n_bins - 1), edges


def _build_groups(ids, positions, increments, shape, requested, edges, dimension):
    order = np.argsort(ids, kind='stable')
    unique, starts, counts = np.unique(ids[order], return_index=True, return_counts=True)
    groups = tuple(
        Group(int(group_id), positions[order[start:start + count]],
              increments[order[start:start + count]])
        for group_id, start, count in zip(unique, starts, counts)
    )
    return GroupedTransition(
        shape=shape, groups=groups, n_valid_pairs=len(positions),
        n_total_positions=int(np.prod(shape)), n_requested_bins=requested,
        n_occupied_bins=len(groups), n_singleton_bins=int(np.sum(counts == 1)),
        edges=edges, dimension=dimension,
    )


def prepare_transition(x_prev, x_curr, quantiles_amount: int,
                       mask: Optional[np.ndarray] = None) -> GroupedTransition:
    """Create 1D groups using X[t-1] and raw increments X[t]-X[t-1]."""
    prev, curr = np.asarray(x_prev, dtype=float), np.asarray(x_curr, dtype=float)
    if prev.ndim != 2 or prev.shape != curr.shape:
        raise ValueError('Previous and current 1D fields must share shape (H,W)')
    keep = load_water_mask(mask, prev.shape)
    valid = keep & np.isfinite(prev) & np.isfinite(curr)
    positions = np.flatnonzero(valid.ravel())
    if positions.size:
        increments = (curr.ravel()[positions] - prev.ravel()[positions])
        good = np.isfinite(increments)
        positions, increments = positions[good], increments[good]
    else:
        increments = np.empty(0)
    if not positions.size:
        return _build_groups(np.empty(0, dtype=np.int64), positions, increments,
                             prev.shape, int(quantiles_amount), np.empty(0), 1)
    ids, edges = _bin_ids(prev.ravel()[positions], quantiles_amount)
    return _build_groups(ids, positions, increments, prev.shape, quantiles_amount, edges, 1)


def prepare_transition_2d(x_prev, x_curr, quantiles_amount: int,
                          mask: Optional[np.ndarray] = None) -> GroupedTransition:
    """Jointly group two state coordinates by their previous-day quantiles.

    Inputs are (H,W,2). Each axis has ``quantiles_amount`` intervals, so
    there are at most N*N joint cells. Each group contains raw increments
    (delta X0, delta X1). This is shared by both NP and SP estimators.
    """
    prev, curr = np.asarray(x_prev, dtype=float), np.asarray(x_curr, dtype=float)
    if prev.ndim != 3 or prev.shape[-1] != 2 or prev.shape != curr.shape:
        raise ValueError('Both bivariate fields must share shape (H,W,2)')
    shape = prev.shape[:2]
    keep = load_water_mask(mask, shape)
    valid = keep & np.all(np.isfinite(prev), axis=-1) & np.all(np.isfinite(curr), axis=-1)
    positions = np.flatnonzero(valid.ravel())
    increments = curr.reshape((-1, 2))[positions] - prev.reshape((-1, 2))[positions]
    good = np.all(np.isfinite(increments), axis=1)
    positions, increments = positions[good], increments[good]
    if not positions.size:
        return _build_groups(np.empty(0, dtype=np.int64), positions, increments,
                             shape, quantiles_amount ** 2, (np.empty(0), np.empty(0)), 2)
    values = prev.reshape((-1, 2))[positions]
    first, edges_0 = _bin_ids(values[:, 0], quantiles_amount)
    second, edges_1 = _bin_ids(values[:, 1], quantiles_amount)
    ids = first.astype(np.int64) * int(quantiles_amount) + second.astype(np.int64)
    return _build_groups(ids, positions, increments, shape, quantiles_amount ** 2,
                         (edges_0, edges_1), 2)


def prepare_pair_transition(series0_prev, series0_curr, series1_prev, series1_curr,
                            quantiles_amount, mask=None):
    """Prepare a bivariate transition from four scalar (H,W) fields."""
    prev = np.stack((series0_prev, series1_prev), axis=-1)
    curr = np.stack((series0_curr, series1_curr), axis=-1)
    return prepare_transition_2d(prev, curr, quantiles_amount, mask)


def empty_maps(shape, channels=1):
    """Return empty A and B maps with NaN at all positions."""
    a_shape = shape if channels == 1 else (2, *shape)
    b_shape = shape if channels == 1 else (4, *shape)
    return np.full(a_shape, np.nan), np.full(b_shape, np.nan)


def covariance_to_diffusion(cov: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Return the symmetric PSD matrix B satisfying B @ B.T = Cov(delta X)/dt.

    Use one canonical square root because covariance alone does not identify
    the diffusion matrix uniquely (B and BQ share the same covariance for
    any orthogonal Q). Negative eigenvalues from roundoff are clipped.
    """
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError('dt must be finite and positive')
    matrix = np.asarray(cov, dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError('Expected finite 2x2 covariance matrix')
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    if min(values) < -1e-8 * max(1., float(np.max(np.abs(values)))):
        raise ValueError(f'Covariance matrix is not positive semidefinite: {values}')
    root = (vectors * np.sqrt(np.maximum(values, 0.) / dt)) @ vectors.T
    return (root + root.T) * 0.5


def pack_b_matrix(matrix: np.ndarray) -> np.ndarray:
    """Store a 2x2 diffusion matrix as [B00,B11,B01,B10]."""
    return np.array([matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0]])


def unpack_b_matrix(channels: np.ndarray) -> np.ndarray:
    """Restore a matrix from [B00,B11,B01,B10]."""
    b = np.asarray(channels)
    if b.shape[0] != 4:
        raise ValueError('Expected a leading dimension with four diffusion channels')
    result = np.empty((*b.shape[1:], 2, 2), dtype=b.dtype)
    result[..., 0, 0] = b[0]
    result[..., 1, 1] = b[1]
    result[..., 0, 1] = b[2]
    result[..., 1, 0] = b[3]
    return result
