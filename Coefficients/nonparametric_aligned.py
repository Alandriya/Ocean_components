"""Nonparametric estimators of 1D and 2D drift and diffusion.

All statistics use the original, unquantized increments inside groups formed
from the preceding state. The 2D diffusion is the symmetric PSD square root
of the empirical conditional covariance matrix, not the covariance itself.
"""
from pathlib import Path
from typing import Optional
import numpy as np
from Coefficients.aligned_common import (
    GroupedTransition, prepare_transition, prepare_pair_transition,
    check_series, check_pair_series, load_series, load_pair_series,
    load_water_mask, coerce_series, empty_maps, covariance_to_diffusion, pack_b_matrix,
)
from Coefficients.coefficients_cf import count_c_coeff, count_fraction, count_f_separate_coeff


def estimate_np(grouped: GroupedTransition, min_group_size: int = 2, dt: float = 1.0):
    """Estimate scalar drift and diffusion (a, b) from common raw increments."""
    if grouped.dimension != 1:
        raise ValueError('estimate_np expects a scalar grouped transition')
    if min_group_size < 2 or dt <= 0:
        raise ValueError('min_group_size must be >=2 and dt must be positive')
    a, b = empty_maps(grouped.shape)
    af, bf = a.ravel(), b.ravel()
    for group in grouped.groups:
        if group.increments.size < min_group_size:
            continue
        inc = group.increments
        af[group.flat_indices] = np.mean(inc) / dt
        bf[group.flat_indices] = np.std(inc, ddof=0) / np.sqrt(dt)
    return a, b


def estimate_np_2d(grouped: GroupedTransition, min_group_size: int = 2,
                   dt: float = 1.0):
    """Return A (2,H,W) and B (4,H,W), ordered [00,11,01,10].

    For each joint state group, estimate E[delta X]/dt and Cov(delta X)/dt.
    B is the symmetric positive-semidefinite square root of the covariance.
    """
    if grouped.dimension != 2:
        raise ValueError('estimate_np_2d expects bivariate grouped transitions')
    if min_group_size < 2 or dt <= 0:
        raise ValueError('min_group_size must be >=2 and dt must be positive')
    a, b = empty_maps(grouped.shape, channels=2)
    af, bf = a.reshape((2, -1)), b.reshape((4, -1))
    for group in grouped.groups:
        if len(group.increments) < min_group_size:
            continue
        y = group.increments
        mean = np.mean(y, axis=0)
        centered = y - mean
        cov = centered.T @ centered / len(y)
        root = covariance_to_diffusion(cov, dt)
        af[:, group.flat_indices] = (mean / dt)[:, None]
        bf[:, group.flat_indices] = pack_b_matrix(root)[:, None]
    return a, b


def count_1d_nonparam(files_path_prefix: str, flux, time_start: int,
                      time_end: int, path: str = 'Coefficients/',
                      start_index: int = 0, quantiles_amount: int = 100,
                      mask: Optional[np.ndarray] = None,
                      min_group_size: int = 2, overwrite: bool = False,
                      dt: float = 1.0):
    """Save scalar daily A_t and B_t under Nonparametric/daily."""
    series = coerce_series(flux)
    check_series(series, time_start, time_end)
    keep = load_water_mask(mask, series.shape[1:])
    target = Path(files_path_prefix) / path / 'Nonparametric' / 'daily'
    target.mkdir(parents=True, exist_ok=True)
    for t in range(time_start + 1, time_end):
        pa, pb = target / f'A_{t + start_index}.npy', target / f'B_{t + start_index}.npy'
        if not overwrite and pa.exists() and pb.exists():
            continue
        groups = prepare_transition(series[t - 1], series[t], quantiles_amount, keep)
        a, b = estimate_np(groups, min_group_size, dt)
        np.save(pa, a)
        np.save(pb, b)


def count_2d_nonparam(files_path_prefix: str, variable_0, variable_1,
                      time_start: int, time_end: int,
                      path: str = 'Coefficients/', start_index: int = 0,
                      quantiles_amount: int = 12, mask=None,
                      min_group_size: int = 2, overwrite: bool = False,
                      dt: float = 1.0):
    """Estimate a coupled 2D SDE from two time-aligned variables.

    Both inputs may be (T,H,W) arrays or paths to .npy files with shape
    (H*W,T), (T,H*W), or (T,H,W).

    Saved arrays:
        A_t.npy: (2,H,W) drift components [a0,a1].
        B_t.npy: (4,H,W) diffusion [B00,B11,B01,B10].
    """
    s0 = coerce_series(variable_0)
    s1 = coerce_series(variable_1)
    check_pair_series(s0, s1, time_start, time_end)
    keep = load_water_mask(mask, s0.shape[1:])
    target = Path(files_path_prefix) / path / 'Nonparametric_2d' / 'daily'
    target.mkdir(parents=True, exist_ok=True)
    for t in range(time_start + 1, time_end):
        pa, pb = target / f'A_{t + start_index}.npy', target / f'B_{t + start_index}.npy'
        if not overwrite and pa.exists() and pb.exists():
            continue
        grouped = prepare_pair_transition(s0[t - 1], s0[t], s1[t - 1],
                                          s1[t], quantiles_amount, keep)
        a, b = estimate_np_2d(grouped, min_group_size, dt)
        np.save(pa, a)
        np.save(pb, b)
