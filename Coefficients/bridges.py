"""Estimate missing cross-period A/B transitions and assemble contiguous outputs.

The original mathematical estimators are imported unchanged from common,
nonparametric and semiparametric. This module adds only temporal orchestration.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from Data_processing.io import BASE_DATE, load_field, read_ocean_mask, start_label
from Coefficients.collection import (
    coefficient_source, load_coefficient_source, period_dir,
)

METHOD_FOLDERS = {"np": "Nonparametric_2d", "sp": "Semiparametric_2d"}


def period_daily_dir(root, period, names, n_bins_2d, method):
    """Locate daily maps produced by the existing decade estimator.

    Parameters
    ----------
    root : str or Path
        Data directory.
    period : tuple of int
        Start and end year.
    names : tuple of str
        Paired variables in estimator order.
    n_bins_2d : int
        Quantile bin count per coordinate.
    method : {'np', 'sp'}
        Estimation method identifier.

    Returns
    -------
    pathlib.Path
        Existing method-specific daily output directory.
    """
    return (period_dir(root, *period) /
            f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}" /
            METHOD_FOLDERS[method] / "daily")


def bridge_daily_dir(root, names, n_bins_2d, method):
    """Locate the dedicated daily output directory for bridge transitions.

    Parameters
    ----------
    root : str or Path
        Data directory.
    names : tuple of str
        Variable pair.
    n_bins_2d : int
        Quantile bin count.
    method : {'np', 'sp'}
        Estimation method.

    Returns
    -------
    pathlib.Path
        Directory for bridge A/B arrays.
    """
    return (Path(root) / "Coefficients" / "bridges" /
            f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}" /
            METHOD_FOLDERS[method] / "daily")


def calculate_boundary_transition(files_path_prefix, prior_period, next_period,
                                  names=("sensible", "latent"), *,
                                  mask=None, n_bins_2d=12, methods=("np", "sp"),
                                  n_components=3, n_init=10, dt=1.0,
                                  min_group_size=2, overwrite=False):
    """Compute the missing transition between two adjacent data periods.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory of existing data and coefficient arrays.
    prior_period, next_period : tuple of int
        Adjacent half-open calendar intervals (start_year, end_year).
    names : tuple of str
        Two different variable names, in unchanged channel order.
    mask : array_like, optional
        Boolean ocean mask. If omitted the stored ocean mask is loaded.
    n_bins_2d : int
        Quantile bin count per state coordinate.
    methods : tuple of {'np', 'sp'}
        Selected original estimators.
    n_components : int
        Gaussian mixture component count for SP.
    n_init : int
        Gaussian mixture initialization count.
    dt : float
        Time increment for drift and diffusion estimates.
    min_group_size : int
        Smallest group eligible for estimation.
    overwrite : bool
        Whether to replace existing complete bridge files.

    Returns
    -------
    dict
        Absolute coefficient label and saved output filenames.

    Notes
    -----
    The A/B transition is labeled by the ENDPOINT observation date.
    For the 1988-12-31 to 1989-01-01 transition, this is label 3654.
    """
    from Coefficients.common import prepare_pair_transition
    from Coefficients.nonparametric import estimate_np_2d
    from Coefficients.semiparametric import estimate_sp_2d

    if prior_period[1] != next_period[0]:
        raise ValueError("Periods are not adjacent")
    if len(names) != 2 or names[0] == names[1]:
        raise ValueError("Two different variable names are required")
    if any(method not in METHOD_FOLDERS for method in methods):
        raise ValueError("Only 'np' and 'sp' are supported")
    old_first = date(prior_period[0], 1, 1)
    boundary = date(next_period[0], 1, 1)
    left = [load_field(files_path_prefix, name, *prior_period) for name in names]
    right = [load_field(files_path_prefix, name, *next_period) for name in names]
    expected = (boundary - old_first).days
    if any(len(field) != expected for field in left):
        raise ValueError("Previous period must reach the day before the next period")
    if len({field.shape[1:] for field in (*left, *right)}) != 1:
        raise ValueError("Spatial dimensions differ across periods")
    if any(len(field) == 0 for field in right):
        raise ValueError("Next period has no observations")
    water = read_ocean_mask(files_path_prefix, left[0].shape[1:]) if mask is None else mask
    grouped = prepare_pair_transition(left[0][-1], right[0][0],
                                      left[1][-1], right[1][0], n_bins_2d, water)
    label = start_label(next_period[0])
    paths = {}
    for method in methods:
        directory = bridge_daily_dir(files_path_prefix, names, n_bins_2d, method)
        directory.mkdir(parents=True, exist_ok=True)
        expected_keys = ("A", "B", "B_with_reg") if method == "sp" else ("A", "B")
        outputs = {key: directory / f"{key}_{label}.npy" for key in expected_keys}
        if overwrite or not all(file.is_file() for file in outputs.values()):
            if method == "np":
                a, b = estimate_np_2d(grouped, min_group_size=min_group_size, dt=dt)
                arrays = {"A": a, "B": b}
            else:
                a, b, b_reg, _ = estimate_sp_2d(
                    grouped, n_components=n_components, min_group_size=min_group_size,
                    n_init=n_init, dt=dt)
                arrays = {"A": a, "B": b, "B_with_reg": b_reg}
            for key, array in arrays.items():
                np.save(outputs[key], array)
        paths[method] = {key: str(path) for key, path in outputs.items()}
    return {"label": label, "date": boundary.isoformat(), "paths": paths}


def _period_indices(files_path_prefix, period, names):
    """Return date-aligned coefficient labels for one input period."""
    first = start_label(period[0])
    sizes = [len(load_field(files_path_prefix, name, *period)) for name in names]
    if len(set(sizes)) != 1 or sizes[0] < 2:
        raise ValueError(f"Invalid or misaligned input period {period}: {sizes}")
    actual_last = date(period[0], 1, 1) + timedelta(days=sizes[0] - 1)
    if actual_last >= date(period[1], 1, 1):
        raise ValueError(f"Input for {period} exceeds its declared period")
    return range(first + 1, first + sizes[0])


def assemble_continuous_coefficients(files_path_prefix, periods, names=("sensible", "latent"),
                                     *, n_bins_2d=12, methods=("np", "sp"),
                                     overwrite=False):
    """Merge period A/B maps and boundary maps with no missing date labels.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory of the coefficient calculations.
    periods : sequence of (int, int)
        Chronologically ordered data periods.
    names : tuple of str
        Variable pair.
    n_bins_2d : int
        Quantile bins per coordinate.
    methods : sequence of str
        NP/SP method identifiers.
    overwrite : bool
        Rebuild complete arrays even when they already exist.

    Returns
    -------
    dict
        Metadata and output paths for each method.

    Raises
    ------
    FileNotFoundError
        If any required daily A/B field or bridge is missing.
    ValueError
        On noncontiguous time labels, invalid field shapes or dtypes.
    """
    periods = [tuple(period) for period in periods]
    if not periods:
        raise ValueError("At least one period is required")
    records = []
    for index, period in enumerate(periods):
        if index:
            if periods[index - 1][1] != period[0]:
                raise ValueError("Periods are not chronologically adjacent")
            prev_labels = _period_indices(files_path_prefix, periods[index - 1], names)
            boundary_label = start_label(period[0])
            if prev_labels.stop != boundary_label:
                raise ValueError("Previous data file ends before the requested bridge")
            records.append((boundary_label, None))
        records.extend((label, period) for label in _period_indices(files_path_prefix, period, names))
    labels = [label for label, _ in records]
    if any(second != first + 1 for first, second in zip(labels, labels[1:])):
        raise ValueError("Coefficient label sequence has gaps or duplicates")
    result = {"first_label": labels[0], "last_label": labels[-1], "count": len(labels), "methods": {}}
    for method in methods:
        if method not in METHOD_FOLDERS:
            raise ValueError(f"Invalid method: {method}")
        source_files = {}
        keys = ("A", "B", "B_with_reg") if method == "sp" else ("A", "B")
        for key in keys:
            filenames = [
                (coefficient_source(files_path_prefix, period, names,
                                    n_bins_2d, method, key, label)
                 if period is not None else
                 (bridge_daily_dir(files_path_prefix, names, n_bins_2d, method) /
                  f"{key}_{label}.npy", None))
                for label, period in records
            ]
            missing = next((path for path, _ in filenames if not path.is_file()), None)
            if missing is not None:
                raise FileNotFoundError(f"Missing coefficient for a continuous series: {missing}")
            source_files[key] = filenames
        parent = (Path(files_path_prefix) / "Coefficients" / "continuous" /
                  f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}" / METHOD_FOLDERS[method])
        parent.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for key, filenames in source_files.items():
            output = parent / f"{key}_{labels[0]}-{labels[-1]}.npy"
            if not output.exists() or overwrite:
                first = load_coefficient_source(filenames[0])
                shape, dtype = first.shape, first.dtype
                n_channels = 2 if key == "A" else 4
                if len(shape) != 3 or shape[0] != n_channels:
                    raise ValueError(f"Invalid first {key} shape: {shape}")
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(dir=parent, prefix=".stack_", suffix=".npy", delete=False) as handle:
                        temp_path = Path(handle.name)
                    merged = np.lib.format.open_memmap(temp_path, mode="w+", dtype=dtype,
                                                       shape=(len(filenames), *shape))
                    for step, item in enumerate(filenames):
                        array = load_coefficient_source(item)
                        if array.shape != shape or array.dtype != dtype:
                            raise ValueError(f"Incompatible coefficient array: {item[0]}")
                        merged[step] = array
                    merged.flush()
                    del merged
                    os.replace(temp_path, output)
                finally:
                    if temp_path is not None and temp_path.exists():
                        temp_path.unlink()
            outputs[key] = str(output)
        result["methods"][method] = outputs
    manifest = Path(files_path_prefix) / "Coefficients" / "continuous" / (
        f"manifest_{names[0]}_{names[1]}_{n_bins_2d}.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({**result, "periods": periods,
                                    "label_convention": "endpoint"}, indent=2), encoding="utf-8")
    result["manifest"] = str(manifest)
    return result


def run_boundary_workflow(files_path_prefix, periods, names, *, n_bins_2d=12,
                          n_components=3, n_init=10, dt=1.0,
                          methods=("np", "sp"), mask=None, overwrite=False):
    """Estimate every available boundary and assemble the continuous A/B arrays.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root directory.
    periods : sequence of (int, int)
        Consecutive calendar periods.
    names : tuple of str
        Variable pair.
    n_bins_2d : int
        Quantile groups per axis.
    n_components, n_init : int
        Original Gaussian-mixture settings.
    dt : float
        Observation increment.
    methods : sequence of str
        NP/SP method identifiers.
    mask : array_like, optional
        Ocean mask.
    overwrite : bool
        Replace saved bridge and concatenated arrays.

    Returns
    -------
    dict
        Assembled coefficient information including absolute labels.
    """
    for previous, following in zip(periods, periods[1:]):
        calculate_boundary_transition(files_path_prefix, previous, following, names,
                                      mask=mask, n_bins_2d=n_bins_2d, methods=methods,
                                      n_components=n_components, n_init=n_init,
                                      dt=dt, overwrite=overwrite)
    return assemble_continuous_coefficients(files_path_prefix, periods, names,
                                            n_bins_2d=n_bins_2d, methods=methods,
                                            overwrite=overwrite)
