"""Estimate missing cross-period A/B transitions and assemble contiguous outputs.

The original mathematical estimators are imported unchanged from common,
nonparametric and semiparametric. This module adds only temporal orchestration.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from Data_processing.io import BASE_DATE, load_field, read_ocean_mask, start_label
from Coefficients.collection import (
    coefficient_source, load_coefficient_source, period_dir,
)

METHOD_FOLDERS = {"np": "Nonparametric_2d", "sp": "Semiparametric_2d"}

# A hard link avoids duplicated multi-year arrays. Windows volumes may reject it.
# In that case, try a symlink, then copy the file in bounded-memory chunks.
# Set to "skip" to keep only the canonical file and the JSON manifest when
# there is not enough free disk space for a duplicate.
LEGACY_FALLBACK = "copy"  # "copy" or "skip"


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
    """Publish a legacy NPY path, including on Windows volumes without links.

    Parameters
    ----------
    canonical : str or Path
        Verified assembled NPY in ``Coefficients/<period>/<pair>``.
    legacy : str or Path
        Backward-compatible NPY path in ``Coefficients/continuous``.
    overwrite : bool
        Replace a pre-existing legacy array. Without this flag, a valid
        existing legacy NPY is reused, but an invalid one is replaced.

    Returns
    -------
    str
        One of ``existing``, ``hardlink``, ``symlink``, ``copy``, ``skipped``.

    Notes
    -----
    Windows hard links can fail on network or non-NTFS volumes. Try a symbolic
    link next. If both are unavailable, copy the bytes with bounded memory into
    a temporary file and atomically replace the destination. A failed link
    never invalidates the already assembled canonical file.
    """
    canonical = Path(canonical).resolve(strict=True)
    legacy = Path(legacy)
    if LEGACY_FALLBACK not in ("copy", "skip"):
        raise ValueError('LEGACY_FALLBACK must be "copy" or "skip"')
    if canonical == legacy.resolve(strict=False):
        return "existing"

    source = np.load(canonical, mmap_mode="r", allow_pickle=False)
    source_shape, source_dtype = source.shape, source.dtype
    del source
    if legacy.exists() and not overwrite and _valid_merged(
            legacy, source_shape[0], source_shape[1:], source_dtype):
        print(f"Legacy output already exists: {legacy}", flush=True)
        return "existing"

    legacy.parent.mkdir(parents=True, exist_ok=True)
    temporary = legacy.with_name(f".{legacy.name}.tmp_{os.getpid()}_{uuid.uuid4().hex}")
    errors = []
    try:
        # A hard link is the most efficient solution: one file, two paths.
        try:
            os.link(canonical, temporary)
            os.replace(temporary, legacy)
            print(f"Legacy hard link: {legacy}", flush=True)
            return "hardlink"
        except OSError as exc:
            errors.append(f"hard link: {exc}")
            temporary.unlink(missing_ok=True)

        # On Windows, this may require Developer Mode or a special privilege.
        # A symlink points to the already verified canonical array.
        try:
            os.symlink(canonical, temporary, target_is_directory=False)
            os.replace(temporary, legacy)
            print(f"Legacy symbolic link: {legacy}", flush=True)
            return "symlink"
        except OSError as exc:
            errors.append(f"symbolic link: {exc}")
            temporary.unlink(missing_ok=True)

        if LEGACY_FALLBACK == "skip":
            print(
                f"WARNING: legacy link unavailable; canonical array remains at "
                f"{canonical}. Legacy output was not created at {legacy}. "
                f"Reasons: {'; '.join(errors)}",
                flush=True,
            )
            return "skipped"

        # Last resort: streaming byte copy, never materializing the complete
        # historical array in memory. The legacy path appears only after the
        # full copy has completed successfully.
        size = canonical.stat().st_size
        free = shutil.disk_usage(legacy.parent).free
        if free < size:
            raise OSError(
                f"Not enough free space for the backward-compatible copy "
                f"({size / 1024**3:.2f} GiB required, "
                f"{free / 1024**3:.2f} GiB free): {legacy}. "
                f"The main array is intact at {canonical}. "
                'Set LEGACY_FALLBACK = "skip" to avoid duplicating large arrays.'
            )
        print(
            f"Windows/file system rejected hard and symbolic links; "
            f"copying {size / 1024**3:.2f} GiB to legacy path: {legacy}",
            flush=True,
        )
        with canonical.open("rb") as source_file, temporary.open("xb") as target_file:
            copied = 0
            next_report = 1024 ** 3
            while True:
                block = source_file.read(16 * 1024 * 1024)
                if not block:
                    break
                target_file.write(block)
                copied += len(block)
                if copied >= next_report:
                    print(f"  Legacy copy: {copied / size:.1%} ({copied / 1024**3:.1f} GiB)",
                          flush=True)
                    next_report = copied + 1024 ** 3
            target_file.flush()
            try:
                os.fsync(target_file.fileno())
            except OSError as exc:
                # Some network and removable Windows drives do not support fsync.
                print(f"WARNING: explicit disk sync unavailable: {exc}", flush=True)
        if temporary.stat().st_size != size:
            raise IOError(f"Incomplete legacy copy: {temporary}")
        if not _valid_merged(temporary, source_shape[0], source_shape[1:], source_dtype):
            raise RuntimeError(f"Copied NPY failed validation: {temporary}")
        os.replace(temporary, legacy)
        print(f"Legacy output created: {legacy}", flush=True)
        return "copy"
    finally:
        temporary.unlink(missing_ok=True)


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
                          methods=("np", "sp"), mask=None, overwrite=False,
                          n_bins=None, dimension="auto"):
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

    Dimension routing
    -----------------
    The existing two-name call keeps the original 2D algorithm unchanged.
    A single variable (e.g. names=("Flux",)) automatically selects the new
    scalar 1D bridge workflow. For multiple independent scalar variables,
    call with dimension="1d" and optionally n_bins=144. The legacy main.py
    does not forward run_1d/run_2d flags to bridges; dimension is determined
    from names unless explicitly supplied to this function.
    """
    from Coefficients.collection import normalize_methods

    methods = normalize_methods(methods)
    names = tuple(names)
    if dimension not in ("auto", "1d", "2d"):
        raise ValueError("dimension must be 'auto', '1d' or '2d'")
    if dimension == "1d" or (dimension == "auto" and len(names) == 1):
        return run_scalar_boundary_workflow(
            files_path_prefix, periods, names, n_bins=n_bins,
            n_components=n_components, n_init=n_init, dt=dt,
            methods=methods, mask=mask, overwrite=overwrite,
        )
    if len(names) != 2 or names[0] == names[1]:
        raise ValueError(
            "2D bridges require two distinct names. For 1D bridges set "
            "names=('Flux',) or pass dimension='1d'.")
    for prior, following in zip(periods, periods[1:]):
        print(f"Checking bridge {prior[1]}-01-01: {prior} -> {following}", flush=True)
        calculate_boundary_transition(files_path_prefix, prior, following, names,
                                      mask=mask, n_bins_2d=n_bins_2d,
                                      methods=methods, n_components=n_components,
                                      n_init=n_init, dt=dt, overwrite=overwrite)
    return assemble_continuous_coefficients(files_path_prefix, periods, names,
                                            n_bins_2d=n_bins_2d,
                                            methods=methods, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Independent scalar (1D) bridge / continuous-collection workflow.
# This is deliberately separate from the established 2D path above.
# ---------------------------------------------------------------------------
SCALAR_METHOD_FOLDERS = {"np": "Nonparametric", "sp": "Semiparametric"}


def _scalar_bins(root, periods, names, n_bins):
    """Use explicit 1D quantile-bin count, or find one unambiguous value."""
    if n_bins is not None:
        if isinstance(n_bins, bool) or int(n_bins) < 1:
            raise ValueError("n_bins must be a positive integer")
        return int(n_bins)
    common = None
    for period in periods:
        folder = period_dir(root, *period)
        for name in names:
            valid = set()
            for candidate in folder.glob("1d_bins_*"):
                if candidate.is_dir():
                    suffix = candidate.name.removeprefix("1d_bins_")
                    if suffix.isdigit() and (candidate / name).is_dir():
                        valid.add(int(suffix))
            common = valid if common is None else common.intersection(valid)
    if not common:
        raise FileNotFoundError(
            "No common 1D coefficient bin count found in all periods for "
            f"{names}; expected Coefficients/<period>/1d_bins_<N>/<name>/. "
            "First run task='coefficients' with run_1d=True, run_2d=False."
        )
    if len(common) != 1:
        raise ValueError(
            f"Ambiguous 1D bin counts across periods: {sorted(common)}. "
            "Call run_boundary_workflow(..., dimension='1d', n_bins=144)."
        )
    return common.pop()


def _scalar_bridge_daily_dir(root, name, n_bins, method):
    return (Path(root) / "Coefficients" / "bridges" / f"1d_bins_{n_bins}" /
            name / SCALAR_METHOD_FOLDERS[method] / "daily")


def _scalar_period_folder(root, period, name, n_bins, method):
    return (period_dir(root, *period) / f"1d_bins_{n_bins}" / name /
            SCALAR_METHOD_FOLDERS[method])


def _scalar_boundary_transition(root, prior, next_period, name, n_bins, methods,
                                n_components, n_init, dt, mask, overwrite):
    """Estimate one endpoint-labeled scalar A/B transition between periods."""
    from Coefficients.common import prepare_transition
    from Coefficients.nonparametric import estimate_np
    from Coefficients.semiparametric import estimate_sp

    if prior[1] != next_period[0]:
        raise ValueError(f"Nonadjacent periods: {prior} -> {next_period}")
    left = load_field(root, name, *prior)
    right = load_field(root, name, *next_period)
    boundary = date(next_period[0], 1, 1)
    n_expected = (boundary - date(prior[0], 1, 1)).days
    if len(left) != n_expected:
        raise ValueError(
            f"{name}: prior period {prior} has {len(left)} days; "
            f"expected {n_expected} to bridge to {next_period}"
        )
    if len(right) < 1 or left.shape[1:] != right.shape[1:]:
        raise ValueError(f"{name}: invalid/misaligned boundary data arrays")
    field_shape = left.shape[1:]
    water = (read_ocean_mask(root, field_shape) if mask is None else
             np.asarray(mask).reshape(field_shape))
    label = start_label(next_period[0])
    grouped = None
    outputs = {}
    for method in methods:
        directory = _scalar_bridge_daily_dir(root, name, n_bins, method)
        keys = ("A", "B", "B_with_reg") if method == "sp" else ("A", "B")
        paths = {key: directory / f"{key}_{label}.npy" for key in keys}
        complete = not overwrite and all(
            path.is_file() and _scalar_valid_daily(path, field_shape)
            for path in paths.values()
        )
        if not complete:
            if grouped is None:
                grouped = prepare_transition(left[-1], right[0], n_bins, water)
            if method == "np":
                a, b = estimate_np(grouped, min_group_size=2, dt=dt)
                arrays = {"A": a, "B": b}
            else:
                a, b, b_reg, _ = estimate_sp(
                    grouped, n_components=n_components, min_group_size=2,
                    n_init=n_init, dt=dt,
                )
                arrays = {"A": a, "B": b, "B_with_reg": b_reg}
            directory.mkdir(parents=True, exist_ok=True)
            for key, value in arrays.items():
                np.save(paths[key], value)
            print(f"[1D bridge] {name} {method} label={label}: saved", flush=True)
        else:
            print(f"[1D bridge] {name} {method} label={label}: reused", flush=True)
        outputs[method] = {k: str(p) for k, p in paths.items()}
    return {"name": name, "label": label, "date": boundary.isoformat(),
            "paths": outputs}


def _scalar_valid_daily(path, spatial_shape):
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return (arr.shape == spatial_shape and
                np.issubdtype(arr.dtype, np.number))
    except (OSError, ValueError, EOFError, TypeError):
        return False


def _scalar_sources(root, period, name, n_bins, method, field, labels):
    """Discover memory-mapped merged pieces or individual daily scalar maps."""
    folder = _scalar_period_folder(root, period, name, n_bins, method)
    first, last = labels.start, labels.stop - 1
    exact = folder / f"{field}_{first}-{last}.npy"
    if exact.is_file():
        return [(exact, 0, len(labels))]
    # Detect chunk collections; segments must cover each label exactly once.
    pattern = re.compile(rf"^{re.escape(field)}_(\d+)-(\d+)\.npy$")
    archives = []
    for path in folder.glob(f"{field}_*-*.npy"):
        match = pattern.fullmatch(path.name)
        if match:
            lo, hi = map(int, match.groups())
            if lo <= hi:
                archives.append((lo, hi, path))
    segments = []
    label = first
    while label <= last:
        matching = [(lo, hi, path) for lo, hi, path in archives
                    if lo <= label <= hi]
        if len(matching) > 1:
            raise ValueError(f"Overlapping 1D collections for {field} {label}: {matching}")
        if matching:
            lo, hi, path = matching[0]
            length = min(last - label + 1, hi - label + 1)
            segments.append((path, label - lo, length))
            label += length
        else:
            path = folder / "daily" / f"{field}_{label}.npy"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing 1D {name} {method}/{field}, period={period}, "
                    f"label={label}: expected {path} or a collected {field}_start-end.npy"
                )
            segments.append((path, None, 1))
            label += 1
    return segments


def _scalar_input_segments(root, periods, name, n_bins, method, field, ranges):
    result = []
    for i, period in enumerate(periods):
        if i:
            label = start_label(period[0])
            bridge = (_scalar_bridge_daily_dir(root, name, n_bins, method) /
                      f"{field}_{label}.npy")
            if not bridge.is_file():
                raise FileNotFoundError(f"Missing 1D bridge {bridge}")
            result.append((bridge, None, 1))
        result.extend(_scalar_sources(root, period, name, n_bins, method,
                                      field, ranges[i]))
    return result


def _scalar_segment_info(segments, name, method, field):
    shape = dtype = None
    count = 0
    for filename, index, length in segments:
        a = np.load(filename, mmap_mode="r", allow_pickle=False)
        current = a.shape if index is None else a.shape[1:]
        if len(current) != 2 or not np.issubdtype(a.dtype, np.number):
            raise ValueError(f"Invalid 1D {name} {method}/{field} {current}: {filename}")
        if index is not None and (
            a.ndim != 3 or index < 0 or index + length > a.shape[0]
        ):
            raise ValueError(f"Collected 1D slice is out of range: {filename}")
        if shape is None:
            shape, dtype = current, a.dtype
        elif current != shape or a.dtype != dtype:
            raise ValueError(f"Mixed 1D shapes/dtypes for {name} {method}/{field}: {filename}")
        count += length
        del a
    if count <= 0:
        raise ValueError(f"No 1D {name} {method}/{field} coefficients found")
    return shape, dtype, count


def _publish_scalar_collection(path, segments, count, shape, dtype,
                               chunk_days=32):
    """Atomic bounded-memory concatenation; do not duplicate full-period arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}_", suffix=".npy", delete=False
        ) as f:
            temporary = Path(f.name)
        out = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype,
                                        shape=(count, *shape))
        cursor = 0
        for filename, offset, length in segments:
            a = np.load(filename, mmap_mode="r", allow_pickle=False)
            for k in range(0, length, chunk_days):
                size = min(chunk_days, length - k)
                if offset is None:
                    if k:
                        raise ValueError(f"Daily 1D segment has length >1: {filename}")
                    out[cursor] = a
                else:
                    out[cursor:cursor+size] = a[offset+k:offset+k+size]
                cursor += size
            del a
        out.flush()
        del out
        if cursor != count or not _valid_merged(temporary, count, shape, dtype):
            raise RuntimeError(f"Incomplete 1D collected output {temporary}: {cursor}/{count}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def assemble_scalar_continuous_coefficients(files_path_prefix, periods, names,
                                            *, n_bins=None, methods=("np", "sp"),
                                            overwrite=False):
    """Join scalar period collections and boundary transitions for each variable.

    Result is written to Coefficients/<first>-<last>/1d_bins_N/<variable>/
    <Nonparametric|Semiparametric>/<A|B|B_with_reg>_<firstlabel>-<lastlabel>.npy.
    Unlike 2D, scalar arrays do NOT create large copied legacy aliases.
    """
    from Coefficients.collection import normalize_methods
    root = Path(files_path_prefix)
    names = tuple(dict.fromkeys(names))
    periods = tuple(map(tuple, periods))
    methods = normalize_methods(methods)
    if not periods or not names:
        raise ValueError("At least one period and one 1D variable are required")
    for earlier, later in zip(periods, periods[1:]):
        if earlier[1] != later[0]:
            raise ValueError(f"Nonadjacent periods: {earlier}, {later}")
    bins = _scalar_bins(root, periods, names, n_bins)
    total_period = f"{periods[0][0]}-{periods[-1][1]}"
    payload = {"dimension": 1, "names": list(names), "period": total_period,
               "bins": bins, "methods": {}, "label_convention": "endpoint",
               "periods": [list(p) for p in periods]}
    for name in names:
        ranges = [_period_indices(root, period, (name,),
                                   is_last=i==len(periods)-1)
                  for i, period in enumerate(periods)]
        for i in range(1, len(ranges)):
            if ranges[i-1].stop != start_label(periods[i][0]):
                raise ValueError(f"{name}: gap before boundary {periods[i]}")
        first, last = ranges[0].start, ranges[-1].stop - 1
        expected_count = last - first + 1
        variable_results = {}
        for method in methods:
            files = {}
            for field in (("A", "B", "B_with_reg") if method == "sp" else ("A", "B")):
                segments = _scalar_input_segments(root, periods, name, bins,
                                                  method, field, ranges)
                shape, dtype, count = _scalar_segment_info(segments, name, method, field)
                if count != expected_count:
                    raise ValueError(f"{name}/{method}/{field}: {count} labels, expected {expected_count}")
                destination = (root / "Coefficients" / total_period /
                               f"1d_bins_{bins}" / name / SCALAR_METHOD_FOLDERS[method] /
                               f"{field}_{first}-{last}.npy")
                if overwrite or not _valid_merged(destination, count, shape, dtype):
                    print(f"[1D bridge] Assemble {name}/{method}/{field}: {count} "
                          f"transitions -> {destination}", flush=True)
                    _publish_scalar_collection(destination, segments, count, shape, dtype)
                else:
                    print(f"[1D bridge] Reuse {name}/{method}/{field}: {destination}", flush=True)
                files[field] = str(destination)
            variable_results[method] = files
        payload["methods"][name] = variable_results
        payload.setdefault("labels", {})[name] = {
            "first": first, "last": last,
            "last_date": (BASE_DATE + timedelta(days=last-1)).isoformat(),
            "transitions": expected_count,
        }
    manifest = root / "Coefficients" / "continuous" / f"manifest_1d_{'-'.join(names)}_{bins}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temp = manifest.with_name(f".{manifest.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, manifest)
    payload["manifest"] = str(manifest)
    print(f"[1D bridge] Complete: variables={names}, bin_count={bins}; "
          f"manifest={manifest}", flush=True)
    return payload


def run_scalar_boundary_workflow(files_path_prefix, periods, names, *,
                                 n_bins=None, n_components=3, n_init=10,
                                 dt=1.0, methods=("np", "sp"), mask=None,
                                 overwrite=False):
    """Compute scalar missing-boundary A/B and assemble historical trajectories."""
    from Coefficients.collection import normalize_methods
    names = tuple(dict.fromkeys(names))
    periods = tuple(map(tuple, periods))
    methods = normalize_methods(methods)
    if not names:
        raise ValueError("At least one 1D variable is required")
    if not periods:
        raise ValueError("At least one period is required")
    n_bins = _scalar_bins(files_path_prefix, periods, names, n_bins)
    print(f"[1D bridge] Variables={names}, bins={n_bins}, methods={methods}, "
          f"periods={periods}", flush=True)
    for prior, following in zip(periods, periods[1:]):
        for name in names:
            _scalar_boundary_transition(files_path_prefix, prior, following,
                                        name, n_bins, methods, n_components,
                                        n_init, dt, mask, overwrite)
    return assemble_scalar_continuous_coefficients(
        files_path_prefix, periods, names, n_bins=n_bins,
        methods=methods, overwrite=overwrite,
    )
