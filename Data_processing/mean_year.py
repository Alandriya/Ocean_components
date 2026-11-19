"""Calendar-aligned annual climatology for arbitrary time-first numerical arrays."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np


def calculate_mean_year(values, dates, *, mask=None, leap_day="drop"):
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

    Returns
    -------
    mean : numpy.ndarray
        Per-calendar-day averages of shape (365, ...).
    coverage : numpy.ndarray
        Finite-observation counts of shape (365, ...).
    """
    if leap_day != "drop":
        raise ValueError("Only leap_day='drop' is currently supported")
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or len(values) != len(dates):
        raise ValueError("The first array dimension must match dates")
    dates = list(dates)
    if not all(isinstance(d, (date, datetime)) for d in dates):
        raise TypeError("dates must contain datetime.date values")
    if any(current >= following for current, following in zip(dates, dates[1:])):
        raise ValueError("dates must be strictly increasing")
    if mask is not None:
        ocean = np.asarray(mask, dtype=bool)
        if values.ndim < 3 or tuple(values.shape[-2:]) != ocean.shape:
            raise ValueError("mask must match the final two spatial dimensions")
        values = np.where(ocean, values, np.nan)
    calendar = [(month, day) for month in range(1, 13) for day in range(1, 32)
                if _valid_calendar_day(month, day)]
    index = {key: i for i, key in enumerate(calendar)}
    sums = np.zeros((365, *values.shape[1:]), dtype=np.float64)
    counts = np.zeros((365, *values.shape[1:]), dtype=np.int32)
    for timestamp, field in zip(dates, values):
        if timestamp.month == 2 and timestamp.day == 29:
            continue
        index_day = index[(timestamp.month, timestamp.day)]
        valid = np.isfinite(field)
        sums[index_day] += np.where(valid, field, 0.0)
        counts[index_day] += valid
    means = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means, counts


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


def mean_year_parts(category, *, pair=None, method=None, n_bins_2d=None):
    """Common relative directory for numerical results and plots.

    Pair order is significant: A/B channels depend on the variable order.
    Old unqualified coefficient files are deliberately not used implicitly.
    """
    parts = (_safe_component(category),)
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
                    pair=None, method=None, n_bins_2d=None):
    """Return the mean and coverage paths without accessing or creating files."""
    parts = mean_year_parts(category, pair=pair, method=method, n_bins_2d=n_bins_2d)
    name = _safe_component(name)
    suffix = f"_{years[0]}-{years[1]}" if years else ""
    output = Path(files_path_prefix) / "Mean_year" / Path(*parts)
    return (output / f"{name}{suffix}.npy",
            output / f"{name}{suffix}_coverage.npy")


def save_mean_year(files_path_prefix, category, name, values, dates, *, mask=None, years=None,
                   pair=None, method=None, n_bins_2d=None):
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
        Required for Coefficients and Derived.
    n_bins_2d : int, optional
        Required for Coefficients and Derived.

    Returns
    -------
    tuple of Path
        Paths of the mean and coverage NPY arrays.
    """
    result, count_path = mean_year_paths(
        files_path_prefix, category, name, years=years, pair=pair,
        method=method, n_bins_2d=n_bins_2d,
    )
    mean, coverage = calculate_mean_year(values, dates, mask=mask)
    result.parent.mkdir(parents=True, exist_ok=True)
    np.save(result, mean)
    np.save(count_path, coverage)
    return result, count_path
