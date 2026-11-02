"""Derived coefficients C and F for either two-dimensional estimator.

C maps (2,H,W):
    C[0] = temporal Pearson correlation between drift A0 and drift A1.
    C[1] = temporal Pearson correlation between diffusion B00 and B11.
F map (H,W):
    F = Euclidean norm of the mean drift / spectral norm of mean diffusion.
FS maps (2,H,W):
    FS[0] = abs(mean A0) / abs(mean B00)
    FS[1] = abs(mean A1) / abs(mean B11)

All statistics exclude land (mask=0). B uses channel order [00,11,01,10].
Correlations are undefined for a constant series, windows with missing values,
or fewer than three observations; the output is NaN at those positions.
"""
from pathlib import Path
import warnings
import numpy as np
from Coefficients.aligned_common import load_water_mask, unpack_b_matrix


def _stack_fields(a_timelist, b_timelist):
    """Accept sequences of A=(2,H,W) and B=(4,H,W) fields."""
    if len(a_timelist) != len(b_timelist) or len(a_timelist) == 0:
        raise ValueError('A and B time series must be nonempty and time-aligned')
    a = np.asarray(a_timelist, dtype=float)
    b = np.asarray(b_timelist, dtype=float)
    if a.ndim != 4 or b.ndim != 4 or a.shape[1] != 2 or b.shape[1] != 4 or a.shape[2:] != b.shape[2:]:
        raise ValueError(f'Expected A=(T,2,H,W) and B=(T,4,H,W), received {a.shape}, {b.shape}')
    return a, b


def _pearson_complete(x, y):
    """Vectorized Pearson correlation across the first (time) axis."""
    finite = np.all(np.isfinite(x) & np.isfinite(y), axis=0)
    out = np.full(x.shape[1:], np.nan)
    if x.shape[0] < 3 or not np.any(finite):
        return out
    xf = x[:, finite]
    yf = y[:, finite]
    xd = xf - np.mean(xf, axis=0)
    yd = yf - np.mean(yf, axis=0)
    denom = np.sqrt(np.sum(xd**2, axis=0) * np.sum(yd**2, axis=0))
    ok = denom > 0
    result = np.full(len(denom), np.nan)
    result[ok] = np.sum(xd[:, ok] * yd[:, ok], axis=0) / denom[ok]
    out[finite] = result
    return out


def compute_c(a_window, b_window, mask=None):
    """Compute temporal correlations C0=corr(A0,A1), C1=corr(B00,B11)."""
    a, b = _stack_fields(a_window, b_window)
    c = np.stack((
        _pearson_complete(a[:, 0], a[:, 1]),
        _pearson_complete(b[:, 0], b[:, 1]),
    ), axis=0)
    c[:, ~load_water_mask(mask, a.shape[2:])] = np.nan
    return c


def compute_f(a_map, b_map, mask=None):
    """Compute ||A||_2 / ||B||_2 using the spectral matrix norm of B."""
    a, b = np.asarray(a_map, dtype=float), np.asarray(b_map, dtype=float)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != 2 or b.shape[0] != 4 or a.shape[1:] != b.shape[1:]:
        raise ValueError('Expected A=(2,H,W), B=(4,H,W)')
    water = load_water_mask(mask, a.shape[1:])
    valid = water & np.all(np.isfinite(a), axis=0) & np.all(np.isfinite(b), axis=0)
    out = np.full(a.shape[1:], np.nan)
    if np.any(valid):
        matrices = unpack_b_matrix(b)  # (H,W,2,2)
        spectral = np.linalg.svd(matrices[valid], compute_uv=False)[:, 0]
        norm_a = np.linalg.norm(a[:, valid], axis=0)
        positive = spectral > 0
        values = np.full(spectral.shape, np.nan)
        values[positive] = norm_a[positive] / spectral[positive]
        out[valid] = values
    return out


def compute_f_legacy(a_map, b_map, mask=None):
    """Reproduce the alternative historical F formula from count_fraction.

    F_legacy = |A0+A1| / sqrt(B00^2+B11^2+B01+B10).
    This is not a matrix norm and may be undefined for some matrices;
    use compute_f (spectral norm) for the primary scientific analysis.
    """
    a, b = np.asarray(a_map, dtype=float), np.asarray(b_map, dtype=float)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != 2 or b.shape[0] != 4 or a.shape[1:] != b.shape[1:]:
        raise ValueError('Expected A=(2,H,W), B=(4,H,W)')
    denom_squared = b[0]**2 + b[1]**2 + b[2] + b[3]
    valid = load_water_mask(mask, a.shape[1:]) & (denom_squared > 0)
    valid &= np.all(np.isfinite(a), axis=0) & np.all(np.isfinite(b), axis=0)
    out = np.full(a.shape[1:], np.nan)
    out[valid] = np.abs(a[0, valid] + a[1, valid]) / np.sqrt(denom_squared[valid])
    return out


def compute_fs(a_map, b_map, mask=None):
    """Compute |A0|/|B00| and |A1|/|B11|; zero denominator gives NaN."""
    a, b = np.asarray(a_map, dtype=float), np.asarray(b_map, dtype=float)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != 2 or b.shape[0] != 4 or a.shape[1:] != b.shape[1:]:
        raise ValueError('Expected A=(2,H,W), B=(4,H,W)')
    out = np.full(a.shape, np.nan)
    water = load_water_mask(mask, a.shape[1:])
    for k in (0, 1):
        valid = water & np.isfinite(a[k]) & np.isfinite(b[k]) & (b[k] != 0)
        out[k, valid] = np.abs(a[k, valid]) / np.abs(b[k, valid])
    return out


def _target(files_path_prefix, pair_name):
    destination = Path(files_path_prefix) / 'Coeff_data_3d' / pair_name
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def count_c_coeff(files_path_prefix, a_timelist, b_timelist,
                 start_idx: int = 1, time_width: int = 14,
                 pair_name: str = '', mask=None):
    """Save C_<t>.npy for each complete rolling window, as in legacy code.

    The legacy implementation used an exclusive final window; this version
    includes the last valid window. C[1] now reads B[1] (B11), because the
    updated B channel convention is [00,11,01,10].
    """
    a, b = _stack_fields(a_timelist, b_timelist)
    if time_width < 3:
        raise ValueError('Correlation time_width must be at least 3')
    target = _target(files_path_prefix, pair_name)
    for start in range(len(a) - time_width + 1):
        c = compute_c(a[start:start + time_width], b[start:start + time_width], mask)
        np.save(target / f'{start_idx + start}_C.npy', c)


def count_fraction(files_path_prefix, a_timelist, b_timelist, mask=None,
                   mean_width: int = 7, start_idx: int = 1,
                   pair_name: str = '', formula: str = 'spectral'):
    """Save F_<t>.npy from mean fields; optional historical formula."""
    a, b = _stack_fields(a_timelist, b_timelist)
    if mean_width < 1:
        raise ValueError('mean_width must be positive')
    if formula not in ('spectral', 'legacy_scalar'):
        raise ValueError('formula must be spectral or legacy_scalar')
    target = _target(files_path_prefix, pair_name)
    for start in range(len(a) - mean_width + 1):
        aw, bw = a[start:start + mean_width], b[start:start + mean_width]
        valid = np.all(np.isfinite(aw), axis=(0, 1)) & np.all(np.isfinite(bw), axis=(0, 1))
        am = np.mean(aw, axis=0)
        bm = np.mean(bw, axis=0)
        am[:, ~valid], bm[:, ~valid] = np.nan, np.nan
        np.save(target / f'{start_idx + start}_F.npy',
                compute_f(am, bm, mask) if formula == 'spectral' else compute_f_legacy(am, bm, mask))


def count_f_separate_coeff(files_path_prefix, a_timelist, b_timelist,
                          start_idx: int = 1, mean_width: int = 7,
                          pair_name: str = '', mask=None):
    """Save FS_<t>.npy with componentwise drift/diffusion ratios."""
    a, b = _stack_fields(a_timelist, b_timelist)
    if mean_width < 1:
        raise ValueError('mean_width must be positive')
    target = _target(files_path_prefix, pair_name)
    for start in range(len(a) - mean_width + 1):
        aw, bw = a[start:start + mean_width], b[start:start + mean_width]
        valid = np.all(np.isfinite(aw), axis=(0, 1)) & np.all(np.isfinite(bw), axis=(0, 1))
        am = np.mean(aw, axis=0)
        bm = np.mean(bw, axis=0)
        am[:, ~valid], bm[:, ~valid] = np.nan, np.nan
        np.save(target / f'{start_idx + start}_FS.npy', compute_fs(am, bm, mask))


def count_cf_from_daily(files_path_prefix, method: str, time_start: int,
                        time_end: int, path: str = 'Coefficients/',
                        start_index: int = 0, pair_name: str = '',
                        correlation_window: int = 14, mean_width: int = 7,
                        mask=None):
    """Read previously saved daily 2D estimates and calculate C, F and FS.

    Method must be 'Nonparametric_2d' or 'Semiparametric_2d'.
    Only the requested interval is loaded. The output location follows the
    original 'Coeff_data_3d/<pair_name>/' convention.
    """
    if method not in ('Nonparametric_2d', 'Semiparametric_2d'):
        raise ValueError('method must be Nonparametric_2d or Semiparametric_2d')
    daily = Path(files_path_prefix) / path / method / 'daily'
    a_series, b_series = [], []
    for t in range(time_start + 1, time_end):
        a_series.append(np.load(daily / f'A_{t + start_index}.npy'))
        b_series.append(np.load(daily / f'B_{t + start_index}.npy'))
    first_stamp = time_start + 1 + start_index
    if len(a_series) < correlation_window:
        warnings.warn('Not enough saved days to compute C for the selected correlation_window', stacklevel=2)
    else:
        count_c_coeff(files_path_prefix, a_series, b_series,
                      start_idx=first_stamp, time_width=correlation_window,
                      pair_name=pair_name, mask=mask)
    if len(a_series) < mean_width:
        warnings.warn('Not enough saved days to compute F/FS for the selected mean_width', stacklevel=2)
    else:
        count_fraction(files_path_prefix, a_series, b_series, mask=mask,
                       mean_width=mean_width, start_idx=first_stamp,
                       pair_name=pair_name)
        count_f_separate_coeff(files_path_prefix, a_series, b_series,
                               mean_width=mean_width, start_idx=first_stamp,
                               pair_name=pair_name, mask=mask)
