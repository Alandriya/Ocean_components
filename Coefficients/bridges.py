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


def _period_indices(files_path_prefix, period, names, *, is_last=False):
    """Get within-period endpoint labels for an actual available data series.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory with input data.
    period : tuple[int, int]
        Nominal years. Older periods are half-open; the final year in the
        last period may be incomplete and is treated as inclusive.
    names : tuple[str, str]
        Variable names, in estimator order.
    is_last : bool
        Allow an incomplete final period and observations in its named end year.

    Returns
    -------
    range
        All within-period endpoint labels (excluding the first observation).
    """
    first = start_label(period[0])
    sizes = [len(load_field(files_path_prefix, name, *period)) for name in names]
    if len(set(sizes)) != 1 or sizes[0] < 2:
        raise ValueError(f"Invalid or misaligned input period {period}: {sizes}")
    last_date = date(period[0], 1, 1) + timedelta(days=sizes[0] - 1)
    upper_bound = (date(period[1] + 1, 1, 1) if is_last
                   else date(period[1], 1, 1))
    if last_date >= upper_bound:
        raise ValueError(
            f"Input period {period} contains data past its allowed end: {last_date}"
        )
    return range(first + 1, first + sizes[0])


def _collected_period_source(root, period, names, n_bins_2d, method,
                             field, label_range):
    """Prefer one memory-mapped period array over thousands of daily files.

    Parameters
    ----------
    root : str or Path
        Data root.
    period : tuple[int, int]
        Period names for the input coefficient folder.
    names : tuple[str, str]
        Variable pair.
    n_bins_2d : int
        Quantile bin count.
    method : str
        Estimator identifier.
    field : str
        A, B or B_with_reg.
    label_range : range
        Exact sequence of within-period endpoint labels.

    Returns
    -------
    list[tuple[Path, int | None, int]]
        Tuples of filename, optional offset and number of maps to copy.
    """
    from Coefficients.collection import coefficient_source

    root = Path(root)
    method_dir = (period_dir(root, *period) / f"2d_bins_{n_bins_2d}" /
                  f"{names[0]}_{names[1]}" / METHOD_FOLDERS[method])
    first_label, last_label = label_range.start, label_range.stop - 1
    exact = method_dir / f"{field}_{first_label}-{last_label}.npy"
    if exact.is_file():
        return [(exact, 0, len(label_range))]

    # A period may have multiple merged chunks. Resolve their first slices once
    # and group contiguous ranges to avoid rescanning the same folder per day.
    segments = []
    label = label_range.start
    while label < label_range.stop:
        path, index = coefficient_source(root, period, names, n_bins_2d,
                                         method, field, label)
        if not path.is_file():
            raise FileNotFoundError(f"Missing coefficient: {path}")
        if index is None:
            length = 1
        else:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            length = min(arr.shape[0] - index, label_range.stop - label)
            del arr
        if length < 1:
            raise ValueError(f"Invalid source length for {path}")
        segments.append((path, index, length))
        label += length
    return segments


def _input_segments(root, periods, names, n_bins_2d, method, field, ranges):
    """Return an ordered, gap-free list of input coefficient segments."""
    result = []
    for i, period in enumerate(periods):
        if i:
            label = start_label(period[0])
            path = (bridge_daily_dir(root, names, n_bins_2d, method) /
                    f"{field}_{label}.npy")
            if not path.is_file():
                raise FileNotFoundError(f"Missing cross-period bridge: {path}")
            result.append((path, None, 1))
        result.extend(_collected_period_source(root, period, names, n_bins_2d,
                                               method, field, ranges[i]))
    return result


def _describe_segments(segments, field):
    """Validate every segment's shape and dtype before writing output."""
    expected_shape = expected_dtype = None
    count = 0
    for filename, index, length in segments:
        arr = np.load(filename, mmap_mode="r", allow_pickle=False)
        shape = arr.shape if index is None else arr.shape[1:]
        if len(shape) != 3 or shape[0] != (2 if field == "A" else 4):
            raise ValueError(f"Bad {field} coefficient shape {shape}: {filename}")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"Non-numeric coefficient: {filename}")
        if index is not None and not (0 <= index < arr.shape[0] and
                                      index + length <= arr.shape[0]):
            raise ValueError(f"Invalid collected slice for {filename}: {index}, {length}")
        if expected_shape is None:
            expected_shape, expected_dtype = shape, arr.dtype
        elif shape != expected_shape or arr.dtype != expected_dtype:
            raise ValueError(f"Shape/dtype mismatch in {filename}")
        count += length
        del arr
    if count == 0:
        raise ValueError(f"No source coefficients for {field}")
    return expected_shape, expected_dtype, count


def _valid_merged(path, count, shape, dtype):
    """Validate a persisted merged NPY header without reading all data."""
    if not path.is_file():
        return False
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = arr.shape == (count, *shape) and arr.dtype == dtype
        del arr
        return bool(valid)
    except (OSError, ValueError, EOFError, TypeError):
        return False


def _publish_merged(destination, segments, count, shape, dtype):
    """Write a verified memory-mapped array and atomically publish it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.stem}_",
            suffix=".npy", delete=False,
        ) as f:
            temp = Path(f.name)
        dest = np.lib.format.open_memmap(temp, mode="w+", dtype=dtype,
                                         shape=(count, *shape))
        position = 0
        for filename, index, length in segments:
            arr = np.load(filename, mmap_mode="r", allow_pickle=False)
            if index is None:
                dest[position] = arr
            else:
                dest[position:position + length] = arr[index:index + length]
            position += length
            del arr
        if position != count:
            raise RuntimeError(f"Incomplete merge: {position} != {count}")
        dest.flush()
        del dest
        if not _valid_merged(temp, count, shape, dtype):
            raise RuntimeError(f"Temporary aggregate failed validation: {temp}")
        os.replace(temp, destination)
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _ensure_legacy_link(canonical, legacy, *, overwrite=False):
    """Keep older consumers working without a second multi-gigabyte array.

    Use a hard link on the same filesystem. If unavailable, do not duplicate
    a multi-year coefficient file: callers should consume the canonical path
    published in the manifest. Existing legacy files remain untouched.
    """
    if legacy.exists() and not overwrite:
        return
    legacy.parent.mkdir(parents=True, exist_ok=True)
    temporary = legacy.with_name(f".{legacy.name}.link_{os.getpid()}")
    try:
        os.link(canonical, temporary)
        os.replace(temporary, legacy)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        print(f"WARNING: cannot hard-link legacy output {legacy}: {exc}", flush=True)


def assemble_continuous_coefficients(files_path_prefix, periods,
                                     names=("sensible", "latent"), *,
                                     n_bins_2d=12, methods=("np", "sp"),
                                     overwrite=False):
    """Merge within-period arrays and cross-period bridges into one time axis.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root directory.
    periods : sequence[tuple[int, int]]
        Adjacent periods; the last may end with an incomplete final year.
    names : tuple[str, str]
        Variable pair.
    n_bins_2d : int
        Number of 2D quantile bins.
    methods : sequence[str]
        Requested NP/SP estimators.
    overwrite : bool
        Force rebuilding already valid continuous arrays.

    Returns
    -------
    dict
        Output paths and metadata, including actual last available date.

    Notes
    -----
    Outputs are ``Coefficients/1979-2026/<pair>/A_1979-2026_np.npy``, etc.
    The first array axis corresponds to transition endpoint dates.
    """
    from Coefficients.collection import normalize_methods

    root = Path(files_path_prefix)
    periods = [tuple(p) for p in periods]
    methods = normalize_methods(methods)
    if not periods:
        raise ValueError("At least one period must be provided")
    if len(set(periods)) != len(periods):
        raise ValueError("Duplicated period")
    if len(names) != 2 or names[0] == names[1]:
        raise ValueError("Two distinct variable names are required")

    ranges = []
    for index, period in enumerate(periods):
        if index and periods[index - 1][1] != period[0]:
            raise ValueError(f"Non-adjacent periods: {periods[index-1]} and {period}")
        current = _period_indices(root, period, names, is_last=index == len(periods)-1)
        if index and ranges[-1].stop != start_label(period[0]):
            raise ValueError(f"Gap before bridge to {period}: previous period incomplete")
        ranges.append(current)

    first_label = ranges[0].start
    last_label = ranges[-1].stop - 1
    count = last_label - first_label + 1
    actual_last_date = (BASE_DATE + timedelta(days=last_label - 1)).isoformat()
    period_name = f"{periods[0][0]}-{periods[-1][1]}"
    canonical_parent = root / "Coefficients" / period_name / f"{names[0]}_{names[1]}"
    canonical_parent.mkdir(parents=True, exist_ok=True)
    result = {
        "first_label": first_label,
        "last_label": last_label,
        "first_transition_date": (BASE_DATE + timedelta(days=first_label-1)).isoformat(),
        "last_transition_date": actual_last_date,
        "count": count,
        "period": period_name,
        "methods": {},
    }

    for method in methods:
        outputs = {}
        keys = ("A", "B", "B_with_reg") if method == "sp" else ("A", "B")
        for field in keys:
            segments = _input_segments(root, periods, names, n_bins_2d,
                                       method, field, ranges)
            shape, dtype, actual_count = _describe_segments(segments, field)
            if actual_count != count:
                raise ValueError(f"Missing/duplicated labels for {field} {method}: "
                                 f"{actual_count} != {count}")
            destination = canonical_parent / f"{field}_{period_name}_{method}.npy"
            if overwrite or not _valid_merged(destination, count, shape, dtype):
                print(f"Assembling {method}/{field}: {count} transitions -> {destination}", flush=True)
                _publish_merged(destination, segments, count, shape, dtype)
            else:
                print(f"Reusing {method}/{field}: {destination}", flush=True)
            if not _valid_merged(destination, count, shape, dtype):
                raise RuntimeError(f"Invalid merged array: {destination}")

            legacy = (root / "Coefficients" / "continuous" /
                      f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}" /
                      METHOD_FOLDERS[method] /
                      f"{field}_{first_label}-{last_label}.npy")
            _ensure_legacy_link(destination, legacy, overwrite=overwrite)
            outputs[field] = str(destination)
        result["methods"][method] = outputs

    manifest = (root / "Coefficients" / "continuous" /
                f"manifest_{names[0]}_{names[1]}_{n_bins_2d}.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {**result, "periods": periods, "label_convention": "endpoint"}
    temporary = manifest.with_name(f".{manifest.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, manifest)
    result["manifest"] = str(manifest)
    print(f"Continuous series: {count} transitions, last date {actual_last_date}", flush=True)
    return result


def run_boundary_workflow(files_path_prefix, periods, names, *, n_bins_2d=12,
                          n_components=3, n_init=10, dt=1.0,
                          methods=("np", "sp"), mask=None, overwrite=False):
    """Compute missing cross-period transitions and assemble full NP/SP arrays.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root.
    periods : sequence[tuple[int, int]]
        Chronologically adjacent nominal periods.
    names : tuple[str, str]
        Ordered variable pair.
    n_bins_2d : int
        Number of quantile groups per axis.
    n_components, n_init : int
        Gaussian mixture settings for SP.
    dt : float
        Temporal spacing.
    methods : sequence[str]
        Selected estimation methods.
    mask : array-like or None
        Ocean mask.
    overwrite : bool
        Rebuild bridge and collected outputs.

    Returns
    -------
    dict
        Final paths and transition metadata.
    """
    from Coefficients.collection import normalize_methods

    methods = normalize_methods(methods)
    for prior, following in zip(periods, periods[1:]):
        print(f"Checking bridge {prior[1]}-01-01: {prior} -> {following}", flush=True)
        calculate_boundary_transition(files_path_prefix, prior, following, names,
                                      mask=mask, n_bins_2d=n_bins_2d,
                                      methods=methods, n_components=n_components,
                                      n_init=n_init, dt=dt, overwrite=overwrite)
    return assemble_continuous_coefficients(files_path_prefix, periods, names,
                                            n_bins_2d=n_bins_2d,
                                            methods=methods, overwrite=overwrite)
