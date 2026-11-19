"""Calendar-aligned annual climatology for arbitrary time-first numerical arrays."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re

import numpy as np
from tqdm.auto import tqdm


def calculate_mean_year(values, dates, *, mask=None, leap_day="drop",
                        progress=True, description="Mean year"):
    """Average arbitrary numerical tensors by calendar month and day.

    Parameters
    ----------
    values : array_like
        Observations shaped (T, ...), including scalar or multi-channel fields.
    dates : sequence of date or datetime
        Exactly T calendar dates in strictly increasing order.
    mask : array_like of bool, optional
        Ocean mask matching the final two axes of spatial fields. Ignored
        for non-spatial arrays only when omitted.
    leap_day : {'drop'}
        Exclude February 29 to produce a 365-day climatology.
    progress : bool
        Show a tqdm day counter, processing rate and estimated time remaining.
    description : str
        Label for the progress bar.

    Notes
    -----
    Keep the source dtype and memory mapping intact. Convert only the current
    day to float64, and apply the ocean mask to that day only. Memory use is
    bounded by the 365-day sums and counts, not the length of the input record.

    Returns
    -------
    mean : numpy.ndarray
        Per-calendar-day averages of shape (365, ...).
    coverage : numpy.ndarray
        Finite-observation counts of shape (365, ...).
    """
    if leap_day != "drop":
        raise ValueError("Only leap_day='drop' is currently supported")
    values = np.asarray(values)  # a view for ndarray/memmap inputs
    if values.dtype.kind not in "biuf":
        raise TypeError(f"Expected a real numeric array, got {values.dtype}")
    if values.ndim < 1 or len(values) != len(dates):
        raise ValueError("The first array dimension must match dates")
    dates = list(dates)
    if not all(isinstance(d, (date, datetime)) for d in dates):
        raise TypeError("dates must contain datetime.date values")
    if any(current >= following for current, following in zip(dates, dates[1:])):
        raise ValueError("dates must be strictly increasing")
    ocean = None
    if mask is not None:
        ocean = np.asarray(mask, dtype=bool)
        if values.ndim < 3 or tuple(values.shape[-2:]) != ocean.shape:
            raise ValueError("mask must match the final two spatial dimensions")
    calendar = [(month, day) for month in range(1, 13) for day in range(1, 32)
                if _valid_calendar_day(month, day)]
    index = {key: i for i, key in enumerate(calendar)}
    sums = np.zeros((365, *values.shape[1:]), dtype=np.float64)
    counts = np.zeros((365, *values.shape[1:]), dtype=np.int32)
    with tqdm(total=len(dates), desc=description, unit="day", disable=not progress,
              mininterval=1.0, dynamic_ncols=True) as bar:
        for i, timestamp in enumerate(dates):
            if not (timestamp.month == 2 and timestamp.day == 29):
                index_day = index[(timestamp.month, timestamp.day)]
                field = np.asarray(values[i], dtype=np.float64)
                valid = np.isfinite(field)
                if ocean is not None:
                    valid &= ocean  # broadcasts across any coefficient channels
                sums[index_day] += np.where(valid, field, 0.0)
                counts[index_day] += valid
            bar.update(1)
    # Reuse the sums buffer as the final mean; no second 365-day float64 array.
    for i in range(365):
        dst = sums[i:i+1]
        count = counts[i:i+1]
        np.divide(dst, count, out=dst, where=count > 0)
        np.copyto(dst, np.nan, where=count == 0)
    return sums, counts


def _valid_calendar_day(month, day):
    """Return whether a month/day pair exists in a non-leap year."""
    try:
        date(2001, month, day)
    except ValueError:
        return False
    return True


def _safe_component(value):
    if (not isinstance(value, str) or not value or value in (".", "..")
            or any(c in value for c in "\\/:")):
        raise ValueError(f"Unsafe mean-year path component: {value!r}")
    return value


def mean_year_parts(category, *, pair=None, method=None, n_bins_2d=None,
                    variable=None, n_bins_1d=None):
    """Common relative directory for numerical results and plots.

    Pair order is significant: A/B channels depend on the variable order.
    Old unqualified coefficient files are deliberately not used implicitly.
    """
    parts = (_safe_component(category),)
    if category == "Coefficients_1d":
        variable = _safe_component(variable)
        if method not in ("np", "sp"):
            raise ValueError("mean_year method must be 'np' or 'sp'")
        if (not isinstance(n_bins_1d, (int, np.integer))
                or isinstance(n_bins_1d, (bool, np.bool_)) or n_bins_1d < 1):
            raise ValueError("mean_year n_bins_1d must be a positive integer")
        return (*parts, f"1d_bins_{n_bins_1d}", variable, method)
    if category in ("Coefficients", "Derived", "Eigenvalues"):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"mean_year {category} requires an ordered variable pair")
        left, right = map(_safe_component, pair)
        if category == "Eigenvalues":
            return (*parts, f"{left}-{right}")
        if method not in ("np", "sp"):
            raise ValueError("mean_year method must be 'np' or 'sp'")
        if (not isinstance(n_bins_2d, (int, np.integer))
                or isinstance(n_bins_2d, (bool, np.bool_)) or n_bins_2d < 1):
            raise ValueError("mean_year n_bins_2d must be a positive integer")
        return (*parts, f"2d_bins_{n_bins_2d}", f"{left}_{right}", method)
    return parts


def mean_year_paths(files_path_prefix, category, name, *, years=None,
                    pair=None, method=None, n_bins_2d=None,
                    variable=None, n_bins_1d=None):
    """Return the mean and coverage paths without accessing or creating files."""
    parts = mean_year_parts(category, pair=pair, method=method, n_bins_2d=n_bins_2d,
                            variable=variable, n_bins_1d=n_bins_1d)
    name = _safe_component(name)
    suffix = f"_{years[0]}-{years[1]}" if years else ""
    output = Path(files_path_prefix) / "Mean_year" / Path(*parts)
    return (output / f"{name}{suffix}.npy",
            output / f"{name}{suffix}_coverage.npy")


def save_mean_year(files_path_prefix, category, name, values, dates, *, mask=None, years=None,
                   pair=None, method=None, n_bins_2d=None,
                   variable=None, n_bins_1d=None, progress=True):
    """Calculate and save mean and coverage under the numeric Mean_year root.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root directory.
    category : str
        Numeric output category, for example Variables or Coefficients.
    name : str
        Filename stem identifying the field.
    values : array_like
        Time-first observations.
    dates : sequence of date
        Observation dates.
    mask : array_like, optional
        Optional spatial ocean mask.
    years : tuple of int, optional
        Two year labels for the filename.
    pair : tuple of str, optional
        Required for Coefficients, Derived and Eigenvalues; order is preserved.
    method : {"np", "sp"}, optional
        Required for Coefficients, Derived and Coefficients_1d.
    n_bins_2d : int, optional
        Required for Coefficients and Derived.
    variable : str, optional
        Required for Coefficients_1d.
    n_bins_1d : int, optional
        Required for Coefficients_1d.
    progress : bool
        Show processing progress and input/output information.

    Returns
    -------
    tuple of Path
        Paths of the mean and coverage NPY arrays.
    """
    result, count_path = mean_year_paths(
        files_path_prefix, category, name, years=years, pair=pair,
        method=method, n_bins_2d=n_bins_2d, variable=variable, n_bins_1d=n_bins_1d,
    )
    if progress:
        print(f"[mean_year] {category}/{name}: {len(values):,} days -> {result}", flush=True)
    mean, coverage = calculate_mean_year(values, dates, mask=mask, progress=progress,
                                        description=f"Mean year: {variable or name}{' / '+name if variable else ''}")
    result.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        print(f"[mean_year] Saving {mean.shape} mean and coverage...", flush=True)
    np.save(result, mean)
    np.save(count_path, coverage)
    if progress:
        print(f"[mean_year] Saved {result} and {count_path}", flush=True)
    return result, count_path


def load_scalar_mean_year_source(files_path_prefix, variable, name, start_year, end_year,
                                 *, method, n_bins_1d, spatial_shape=(161, 181)):
    """Read a collected scalar coefficient (T,H,W) via its current local path.

    The endpoint labels encoded in <A|B|B_with_reg>_<first>-<last>.npy
    define the dates, including the available part of the final year. Never
    follow absolute paths in a copied manifest or guess between multiple files.
    Supports both full bridged collections and individual collected periods.
    """
    from Data_processing.io import dates_from_labels

    mean_year_parts("Coefficients_1d", variable=variable, method=method,
                    n_bins_1d=n_bins_1d)
    allowed = ("A", "B", "B_with_reg") if method == "sp" else ("A", "B")
    if name not in allowed:
        raise ValueError(f"Invalid 1D coefficient {name!r} for {method}: choose {allowed}")
    folder = (Path(files_path_prefix) / "Coefficients" / f"{start_year}-{end_year}"
              / f"1d_bins_{n_bins_1d}" / variable
              / ("Nonparametric" if method == "np" else "Semiparametric"))
    pattern = re.compile(rf"{re.escape(name)}_(\d+)-(\d+)\.npy")
    matches = [(path, pattern.fullmatch(path.name))
               for path in sorted(folder.glob(f"{name}_*-*.npy"))]
    matches = [(path, match) for path, match in matches if match and path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one collected 1D {name} array under {folder}; "
                                f"found {[str(path) for path, _ in matches]}")
    path, match = matches[0]
    first, last = map(int, match.groups())
    if first < 1 or last < first:
        raise ValueError(f"Invalid endpoint labels in {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if (values.ndim != 3 or tuple(values.shape[1:]) != tuple(spatial_shape)
            or len(values) != last - first + 1 or values.dtype.kind not in "biuf"):
        raise ValueError(f"Unexpected scalar array {path}: shape={values.shape}, dtype={values.dtype}; "
                         f"expected ({last-first+1}, {spatial_shape[0]}, {spatial_shape[1]}) real numeric")
    dates = dates_from_labels(first, len(values), convention="end")
    print(f"[mean_year] Reading {path}; {dates[0]}..{dates[-1]}, {len(values):,} days", flush=True)
    return values, dates
