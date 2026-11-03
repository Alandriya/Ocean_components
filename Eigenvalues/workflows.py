"""Period-wise spectral calculations and chronologically ordered collection."""

from __future__ import annotations

import json
from datetime import date, timedelta
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from Data_processing.io import load_spectral_field, start_label


def run_eigenvalues_for_period(files_path_prefix, start_year=2019, end_year=None,
                               n_bins=100, n_components=3, dt=1.0,
                               overwrite=True, save_counts=False,
                               spatial_shape=(161, 181),
                               variables=("Flux", "SST", "Pressure")):
    """Compute all self/cross spectral modes for one actual input period.

    Parameters
    ----------
    files_path_prefix : str or Path
        Input and output data root.
    start_year, end_year : int
        Half-open period labels.
    n_bins : int
        Quantile bins for each marginal.
    n_components : int
        Number of retained eigen/singular modes.
    dt : float
        Increment duration.
    overwrite : bool
        Replace existing per-day results.
    save_counts : bool
        Save quantile-bin observation counts.
    spatial_shape : tuple of int
        Spatial grid dimensions.
    variables : tuple of str
        Variables passed to original spectral estimator.

    Returns
    -------
    dict
        Period metadata and names of all unordered pairs.
    """
    from Eigenvalues.eigenvalues import count_eigenvalues_all

    if end_year is None:
        end_year = 2026 if start_year == 2019 else start_year + 10
    arrays = {name: load_spectral_field(files_path_prefix, name, start_year, end_year,
                                        spatial_shape) for name in variables}
    lengths = {name: data.shape[1] for name, data in arrays.items()}
    if len(set(lengths.values())) != 1 or min(lengths.values()) < 2:
        raise ValueError(f"Inconsistent spectral series lengths: {lengths}")
    days = next(iter(lengths.values()))
    label = start_label(start_year)
    count_eigenvalues_all(str(files_path_prefix), arrays, spatial_shape, label,
                          n_bins=n_bins, n_components=n_components, dt=dt,
                          overwrite=overwrite, save_counts=save_counts)
    return {"period": f"{start_year}-{end_year}", "days": days,
            "pairs": list(combinations_with_replacement(arrays, 2)),
            "first_index": label, "last_index": label + days - 2}


def collect_spectral_values(files_path_prefix, names, periods, *, overwrite=False):
    """Aggregate sorted eigenvalues or singular values without interchanging them.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory containing Eigenvalues/pair daily files.
    names : tuple of str
        Self or cross-variable pair.
    periods : sequence of tuple
        Chronological (start_year, end_year) periods.
    overwrite : bool
        Replace the aggregate when present.

    Returns
    -------
    pathlib.Path
        Aggregated numeric array with shape (T, n_bins).

    Notes
    -----
    Eigenvalue labels refer to STARTPOINT observations, unlike A/B labels.
    """
    root = Path(files_path_prefix) / "Eigenvalues" / f"{names[0]}-{names[1]}"
    kind = "eigenvalues" if names[0] == names[1] else "singular_values"
    labels = []
    for first_year, last_year in periods:
        fields = load_spectral_field(files_path_prefix, names[0], first_year, last_year)
        start = start_label(first_year)
        labels.extend(range(start, start + fields.shape[1] - 1))
    if not labels:
        raise ValueError("No spectral labels to aggregate")
    result = root / f"{kind}_collected_{labels[0]}-{labels[-1]}.npy"
    if result.exists() and not overwrite:
        return result
    source = [root / f"{kind}_{label}.npy" for label in labels]
    missing = next((p for p in source if not p.exists()), None)
    if missing is not None:
        raise FileNotFoundError(str(missing))
    first = np.load(source[0], mmap_mode="r", allow_pickle=False)
    dest = np.lib.format.open_memmap(result, mode="w+", dtype=first.dtype,
                                     shape=(len(source), *first.shape))
    for index, path in enumerate(source):
        field = np.load(path, mmap_mode="r", allow_pickle=False)
        if field.shape != first.shape or field.dtype != first.dtype:
            raise ValueError(f"Incompatible spectrum: {path}")
        dest[index] = field
    dest.flush()
    del dest
    (root / f"{kind}_collected_{labels[0]}-{labels[-1]}.json").write_text(
        json.dumps({"labels": labels, "convention": "startpoint", "kind": kind}),
        encoding="utf-8")
    return result
