"""Reliable NP/SP coefficient aggregation and compact period folder layout.

Input maps are left intact until all requested methods have complete,
shape-validated aggregate outputs. Aggregates are written with a memory map
and atomically published. Deleting daily maps is strictly opt-in.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np

METHOD_FOLDERS = {
    "np": ("Nonparametric", "Nonparametric_2d"),
    "sp": ("Semiparametric", "Semiparametric_2d"),
}


def period_dir(root, start_year: int, end_year: int) -> Path:
    """Get a period folder without 'aligned_' and migrate a legacy folder.

    The old directory is renamed, never copied. If old and new both exist,
    refuse to merge them automatically, since they may contain different runs.
    """
    base = Path(root) / "Coefficients"
    modern = base / f"{start_year}-{end_year}"
    old = base / f"aligned_{start_year}-{end_year}"
    if old.exists():
        if modern.exists():
            raise FileExistsError(
                f"Both old and new period folders exist: {old} and {modern}. "
                "Inspect and consolidate them before continuing."
            )
        old.rename(modern)
        print(f"Migrated period directory: {old} -> {modern}", flush=True)
    return modern


def normalize_methods(methods) -> tuple[str, ...]:
    """Normalize a method name, 'both', or an iterable without duplicates."""
    if isinstance(methods, str):
        methods = ("np", "sp") if methods == "both" else (methods,)
    methods = tuple(dict.fromkeys(methods))
    if not methods or any(method not in METHOD_FOLDERS for method in methods):
        raise ValueError("methods must contain 'np', 'sp', or both")
    return methods


def _field_shape(method, field, spatial_shape):
    """Return the daily map shape for a method, field and spatial grid."""
    if method.endswith("_2d"):
        return (2 if field == "A" else 4, *spatial_shape)
    return tuple(spatial_shape)


def _aggregate_valid(path: Path, method: str, field: str, n_days: int,
                     daily_shape=None, daily_dtype=None) -> bool:
    """Check that a persisted aggregate is usable without reading its payload."""
    if not path.is_file():
        return False
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        minimum_ndim = 4 if method.endswith("_2d") else 3
        ready = (
            array.ndim == minimum_ndim
            and array.shape[0] == n_days
            and np.issubdtype(array.dtype, np.number)
            and (daily_shape is None or array.shape[1:] == tuple(daily_shape))
            and (daily_dtype is None or array.dtype == daily_dtype)
            and (
                not method.endswith("_2d")
                or array.shape[1] == (2 if field == "A" else 4)
            )
        )
        del array
        return bool(ready)
    except (OSError, ValueError, EOFError, TypeError):
        return False


def period_collections_complete(specs, first_label: int, last_label: int) -> bool:
    """Check that every enabled daily output has a complete collected counterpart.

    Parameters
    ----------
    specs : iterable
        Existing runner triples (daily_folder, field_names, expected_shapes).
    first_label, last_label : int
        Inclusive endpoint observation labels of this period.
    """
    n_days = last_label - first_label + 1
    if n_days <= 0 or not specs:
        return False
    for daily_dir, fields, shapes in specs:
        method = Path(daily_dir).parent.name
        for index, field in enumerate(fields):
            spatial = shapes[index] if isinstance(shapes[0], tuple) else shapes
            merged = Path(daily_dir).parent / f"{field}_{first_label}-{last_label}.npy"
            if not _aggregate_valid(merged, method, field, n_days, spatial):
                return False
    return True


def _validate_daily_files(daily: Path, method: str, coefficients: tuple[str, ...],
                          start_label: int, end_label: int):
    """Validate all daily inputs before producing any aggregate."""
    definitions = []
    for field in coefficients:
        files = [daily / f"{field}_{label}.npy"
                 for label in range(start_label, end_label + 1)]
        missing = next((p for p in files if not p.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(f"Missing daily file for {method}: {missing}")
        expected_shape = expected_dtype = None
        for file in files:
            try:
                arr = np.load(file, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError, TypeError, EOFError) as error:
                raise ValueError(f"Invalid daily file {file}: {error}") from error
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(f"Non-numeric coefficient map: {file}")
            expected_ndim = 3 if method.endswith("_2d") else 2
            if arr.ndim != expected_ndim:
                raise ValueError(f"Incorrect shape for {file}: {arr.shape}")
            if method.endswith("_2d") and arr.shape[0] != (2 if field == "A" else 4):
                raise ValueError(f"Incorrect coefficient channels for {file}: {arr.shape}")
            if expected_shape is None:
                expected_shape, expected_dtype = arr.shape, arr.dtype
            elif arr.shape != expected_shape or arr.dtype != expected_dtype:
                raise ValueError(f"Inconsistent daily shape/dtype: {file}")
            del arr
        definitions.append((field, files, expected_shape, expected_dtype))
    return definitions



def _validate_remaining_daily(daily: Path, destinations: dict[str, Path],
                              start_label: int, end_label: int) -> None:
    """Check surviving daily headers against preexisting collected arrays."""
    if not daily.is_dir():
        return
    for field, output in destinations.items():
        merged = np.load(output, mmap_mode="r", allow_pickle=False)
        shape, dtype = merged.shape[1:], merged.dtype
        del merged
        for label in range(start_label, end_label + 1):
            path = daily / f"{field}_{label}.npy"
            if path.is_file():
                source = np.load(path, mmap_mode="r", allow_pickle=False)
                if source.shape != shape or source.dtype != dtype:
                    raise ValueError(
                        f"Collected file differs from remaining daily input: {path}"
                    )
                del source


def collect_daily_coefficients(
    files_path_prefix: str,
    *,
    path: str,
    method: str,
    start_label: int,
    end_label: int,
    coefficients=None,
    overwrite: bool = False,
    show_progress: bool = True,
    delete_daily: bool = False,
) -> dict[str, Path]:
    """Collect one method's daily maps into validated, atomic NPY arrays.

    Parameters
    ----------
    files_path_prefix : str
        Data root directory.
    path : str
        Path to the pair/variable directory, relative to data root.
    method : str
        One of Nonparametric, Semiparametric, Nonparametric_2d,
        Semiparametric_2d.
    start_label, end_label : int
        Inclusive daily coefficient labels.
    coefficients : sequence of str, optional
        Field names to collect. Default includes B_with_reg for SP.
    overwrite : bool
        Replace collected arrays, requiring daily source files.
    show_progress : bool
        Log progress for lengthy collections.
    delete_daily : bool
        Remove only collected daily files after all outputs are verified.
        Use False when collecting multiple methods through the orchestration
        function, so cleanup can occur after every method succeeds.
    """
    allowed = {folder for pair in METHOD_FOLDERS.values() for folder in pair}
    if method not in allowed:
        raise ValueError(f"Invalid method: {method}")
    if not isinstance(start_label, int) or not isinstance(end_label, int) or end_label < start_label:
        raise ValueError("Expected integer labels, start_label <= end_label")
    if coefficients is None:
        coefficients = ("A", "B", "B_with_reg") if method.startswith("Semiparametric") else ("A", "B")
    coefficients = tuple(dict.fromkeys(coefficients))
    allowed_fields = ("A", "B", "B_with_reg") if method.startswith("Semiparametric") else ("A", "B")
    if not coefficients or any(name not in allowed_fields for name in coefficients):
        raise ValueError(f"Unexpected coefficients: {coefficients}")

    parent = Path(files_path_prefix) / path / method
    daily = parent / "daily"
    n_days = end_label - start_label + 1
    destinations = {field: parent / f"{field}_{start_label}-{end_label}.npy"
                    for field in coefficients}
    # Idempotent: a second invocation after cleanup returns existing aggregates.
    if not overwrite and all(_aggregate_valid(dest, method, field, n_days)
                             for field, dest in destinations.items()):
        _validate_remaining_daily(daily, destinations, start_label, end_label)
        if show_progress:
            print(f"[{method}] All collected arrays already exist and are valid.", flush=True)
        if delete_daily:
            _cleanup_daily(daily, coefficients, start_label, end_label)
        return destinations
    if not daily.is_dir():
        raise FileNotFoundError(
            f"Daily folder does not exist and complete merged arrays are missing: {daily}"
        )

    definitions = _validate_daily_files(daily, method, coefficients, start_label, end_label)
    for field, files, shape, dtype in definitions:
        destination = destinations[field]
        if not overwrite and _aggregate_valid(destination, method, field, n_days, shape, dtype):
            if show_progress:
                print(f"[{method}] Reusing validated {destination.name}", flush=True)
            continue
        parent.mkdir(parents=True, exist_ok=True)
        temp = None
        started = time.perf_counter()
        try:
            with tempfile.NamedTemporaryFile(
                dir=parent, prefix=f".{field}_{start_label}-{end_label}_",
                suffix=".npy", delete=False,
            ) as stream:
                temp = Path(stream.name)
            result = np.lib.format.open_memmap(temp, mode="w+", dtype=dtype,
                                               shape=(n_days, *shape))
            for index, source in enumerate(files):
                current = np.load(source, mmap_mode="r", allow_pickle=False)
                result[index] = current
                del current
                if show_progress and ((index + 1) % max(1, n_days // 10) == 0 or index + 1 == n_days):
                    print(f"[{method}] {field}: {index+1}/{n_days}", flush=True)
            result.flush()
            del result
            os.replace(temp, destination)
            temp = None
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        if show_progress:
            print(f"[{method}] Saved {destination} in {time.perf_counter()-started:.1f}s", flush=True)

    for field, destination in destinations.items():
        shape, dtype = next((shape, dtype) for name, _, shape, dtype in definitions if name == field)
        if not _aggregate_valid(destination, method, field, n_days, shape, dtype):
            raise RuntimeError(f"Collected output failed validation: {destination}")
    if delete_daily:
        _cleanup_daily(daily, coefficients, start_label, end_label)
    return destinations


def _cleanup_daily(daily: Path, coefficients, start_label: int, end_label: int) -> int:
    """Delete only successfully collected days and leave unrelated files intact."""
    removed = 0
    for field in coefficients:
        for label in range(start_label, end_label + 1):
            path = daily / f"{field}_{label}.npy"
            if path.exists() or path.is_symlink():
                path.unlink()
                removed += 1
    if daily.is_dir():
        try:
            daily.rmdir()
        except OSError:
            pass  # Preserve unrelated files, other dates and diagnostic artifacts.
    return removed


def collect_period_coefficients(
    files_path_prefix: str,
    *,
    names: Tuple[str, str],
    start_year: int,
    end_year: int,
    start_label: int,
    end_label: int,
    n_bins: int = 250,
    n_bins_2d: int = 15,
    run_1d: bool = False,
    run_2d: bool = True,
    methods: Tuple[str, ...] = ("np", "sp"),
    overwrite: bool = False,
    show_progress: bool = True,
    delete_daily: bool = False,
) -> dict[str, dict[str, Path]]:
    """Collect all requested NP/SP outputs, then optionally delete daily maps.

    All aggregate files are checked before the *first* deletion, even when
    some aggregates existed before this invocation. An interrupted collection
    therefore leaves all source daily files available for retry.
    """
    methods = normalize_methods(methods)
    if not run_1d and not run_2d:
        raise ValueError("At least one of run_1d and run_2d must be enabled")
    if len(names) != 2 or names[0] == names[1]:
        raise ValueError("Two distinct variable names are required")
    root = period_dir(files_path_prefix, start_year, end_year)
    targets = []
    if run_1d:
        for variable in names:
            branch = root / f"1d_bins_{n_bins}" / variable
            for method in methods:
                targets.append((f"1d/{variable}/{method}", branch,
                                METHOD_FOLDERS[method][0]))
    if run_2d:
        branch = root / f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}"
        for method in methods:
            targets.append((f"2d/{method}", branch, METHOD_FOLDERS[method][1]))
    results = {}
    for index, (key, branch, method) in enumerate(targets, 1):
        if show_progress:
            print(f"Collecting method {index}/{len(targets)}: {key}", flush=True)
        results[key] = collect_daily_coefficients(
            files_path_prefix, path=str(branch.relative_to(Path(files_path_prefix))),
            method=method, start_label=start_label, end_label=end_label,
            overwrite=overwrite, show_progress=show_progress, delete_daily=False,
        )

    # Do not remove any daily source until *every* selected method is valid.
    n_days = end_label - start_label + 1
    for key, branch, method in targets:
        expected_fields = ("A", "B", "B_with_reg") if method.startswith("Semiparametric") else ("A", "B")
        if set(results[key]) != set(expected_fields):
            raise RuntimeError(f"Incomplete collected method {key}: {results[key]}")
        for field, destination in results[key].items():
            if not _aggregate_valid(destination, method, field, n_days):
                raise RuntimeError(f"Invalid merged coefficient array {destination}")
    if delete_daily:
        for key, branch, method in targets:
            removed = _cleanup_daily(branch / method / "daily", results[key],
                                    start_label, end_label)
            if show_progress:
                print(f"[{key}] Deleted {removed} daily coefficient files after validation.", flush=True)
    return results


def coefficient_source(root, period, names, n_bins_2d, method, field, label):
    """Resolve either a daily file or a slice in a collected period array.

    Returns
    -------
    tuple[Path, int | None]
        Filename and optional time index. The latter is None for daily maps.
    """
    parent = (period_dir(root, *period) / f"2d_bins_{n_bins_2d}" /
              f"{names[0]}_{names[1]}" / METHOD_FOLDERS[method][1])
    daily = parent / "daily" / f"{field}_{label}.npy"
    if daily.is_file():
        return daily, None
    matches = []
    pattern = re.compile(rf"^{re.escape(field)}_(\d+)-(\d+)\.npy$")
    for candidate in parent.glob(f"{field}_*-*.npy"):
        match = pattern.fullmatch(candidate.name)
        if match and int(match.group(1)) <= label <= int(match.group(2)):
            matches.append((candidate, label - int(match.group(1))))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous collected coefficient for {field}_{label}: {matches}")
    return daily, None  # Let the calling workflow report a missing file.


@lru_cache(maxsize=48)
def _cached_merged(path: str):
    """Open each collected NPY at most once during a continuous assembly."""
    return np.load(path, mmap_mode="r", allow_pickle=False)


def load_coefficient_source(source):
    """Read a daily coefficient or one time slice of an existing aggregate."""
    path, day_index = source
    if day_index is None:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    return _cached_merged(str(path))[day_index]


def load_daily_or_collected(path: Path):
    """Load a daily coefficient, falling back to its aggregate time slice.

    Designed for consumers such as SDE condition checking that address files
    with an absolute daily label even after the daily NPYs were deleted.
    """
    path = Path(path)
    if path.is_file():
        return np.load(path, mmap_mode="r", allow_pickle=False)
    if path.parent.name != "daily":
        raise FileNotFoundError(str(path))
    match = re.fullmatch(r"(A|B|B_with_reg)_(\d+)\.npy", path.name)
    if match is None:
        raise FileNotFoundError(str(path))
    field, label = match.group(1), int(match.group(2))
    candidates = []
    pattern = re.compile(rf"{re.escape(field)}_(\d+)-(\d+)\.npy")
    for merged in path.parent.parent.glob(f"{field}_*-*.npy"):
        bounds = pattern.fullmatch(merged.name)
        if bounds is not None and int(bounds.group(1)) <= label <= int(bounds.group(2)):
            candidates.append((merged, label - int(bounds.group(1))))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one aggregate containing {path.name}; "
            f"found {len(candidates)} under {path.parent.parent}"
        )
    return load_coefficient_source(candidates[0])
