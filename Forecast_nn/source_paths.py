"""Canonical paths and calendar alignment for Forecast_nn, 1979--2026.

The final year may be incomplete. In particular the filename suffix 1979-2026
DOES NOT imply there are 17,532 daily observations or that 2026-12-31 exists.
Only memory-mapped .npy headers are inspected; the ERA5 dataset is never loaded.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import re

import numpy as np

PERIOD = '1979-2026'
BASE_DATE = '1979-01-01'
GRID_SHAPE = (161, 181)
GRID_SIZE = 161 * 181
DATA_LAYOUT = {
    'sensible': ('Fluxes', 'sensible'),
    'latent': ('Fluxes', 'latent'),
    'Flux': ('Fluxes', 'Flux'),
    'SST': ('SST', 'SST'),
    'Pressure': ('Pressure', 'Pressure'),
}


def data_paths(root, period=PERIOD):
    root = Path(root)
    return {
        key: root / 'DATA' / folder / f'{stem}_{period}_grouped.npy'
        for key, (folder, stem) in DATA_LAYOUT.items()
    }


def grouped_length(path):
    """Count daily observations for supported native field formats, via mmap."""
    data = np.load(path, mmap_mode='r', allow_pickle=False)
    shape = data.shape
    del data
    if len(shape) == 2 and shape[0] == GRID_SIZE:
        return int(shape[1])  # (space, time): original grouped layout
    if len(shape) == 2 and shape[1] == GRID_SIZE:
        return int(shape[0])
    if len(shape) in (3, 4) and shape[-2:] == GRID_SHAPE:
        return int(shape[0])
    raise ValueError(f'Unsupported grouped data shape {shape}: {path}')


def last_available_date(files, *, variables=('sensible', 'latent'),
                        base_date=BASE_DATE, fallback='2026-12-31'):
    """Infer last day from real files; validate equal lengths for joint mode.

    ``fallback`` is used only when *none* of the selected files are present,
    e.g. when the repository is imported before mounting the DATA volume.
    In that case training still fails fast if the field files remain missing.
    """
    available = {name: grouped_length(files[name])
                 for name in variables if Path(files[name]).is_file()}
    if not available:
        return fallback
    if len(set(available.values())) > 1:
        raise ValueError(f'Grouped variables have unequal time lengths: {available}')
    length = next(iter(available.values()))
    if length < 2:
        raise ValueError(f'Grouped series must have at least two days: {available}')
    final = date.fromisoformat(base_date) + timedelta(days=length - 1)
    if final.year > 2026:
        raise ValueError(f'{PERIOD} file extends beyond expected end year: {final}')
    return final.isoformat()


def _scalar_dir(root, name, bins=144, method='sp', period=PERIOD):
    if name not in DATA_LAYOUT:
        raise KeyError(f'Unsupported forecast variable: {name}')
    folder = {'sp': 'Semiparametric', 'np': 'Nonparametric'}[method]
    return Path(root) / 'Coefficients' / period / f'1d_bins_{int(bins)}' / name / folder


def scalar_sde_source(root, name, *, bins=144, method='sp', period=PERIOD,
                      field_length=None):
    """Resolve one *continuous*, one-dimensional coefficient A/B pair.

    Single-decade series must not be mistaken for full 1979--2026 histories.
    The first coefficient at endpoint label 2 is aligned with flux index 1.
    """
    folder = _scalar_dir(root, name, bins=bins, method=method, period=period)
    candidates = []
    for a in folder.glob('A_*-*.npy'):
        match = re.fullmatch(r'A_(\d+)-(\d+)\.npy', a.name)
        if match is None:
            continue
        first, last = (int(v) for v in match.groups())
        b = folder / f'B_{first}-{last}.npy'
        if b.is_file() and first == 2:
            candidates.append((last, a, b))
    if candidates:
        candidates.sort(key=lambda p: p[0], reverse=True)
        last, a, b = candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f'Multiple 1979--2026 scalar coefficient ranges found under {folder}: '
                f'{[p[0] for p in candidates]}; remove obsolete duplicates or select explicitly.')
        return {'a_file': a, 'b_file': b, 'time_offset': 1}
    last = int(field_length) if field_length is not None else 2
    return {
        'a_file': folder / f'A_2-{last}.npy',
        'b_file': folder / f'B_2-{last}.npy',
        'time_offset': 1,
    }


def joint_sde_source(root, *, names=('sensible', 'latent'), method='sp',
                     period=PERIOD):
    """Resolve the modern continuous *joint* 2D coefficients.

    The project's 2D estimator saves B in diagonal-first order
    [B00, B11, B01, B10], not row-major order.
    """
    if len(names) != 2 or names[0] == names[1]:
        raise ValueError('Joint SDE needs two distinct variables')
    if method not in ('sp', 'np'):
        raise ValueError('method must be sp or np')
    parent = Path(root) / 'Coefficients' / period / f'{names[0]}_{names[1]}'
    return {
        'a_file': parent / f'A_{period}_{method}.npy',
        'b_file': parent / f'B_{period}_{method}.npy',
        'time_offset': 1,
        'b_layout': 'diagonal_first',
    }
