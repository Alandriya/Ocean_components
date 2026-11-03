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


def save_mean_year(files_path_prefix, category, name, values, dates, *, mask=None, years=None):
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

    Returns
    -------
    tuple of Path
        Paths of the mean and coverage NPY arrays.
    """
    if not category or not name or any(c in category + name for c in "\\/"):
        raise ValueError("category and name must be safe filename components")
    mean, coverage = calculate_mean_year(values, dates, mask=mask)
    output = Path(files_path_prefix) / "Mean_year" / category
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{years[0]}-{years[1]}" if years else ""
    result = output / f"{name}{suffix}.npy"
    count_path = output / f"{name}{suffix}_coverage.npy"
    np.save(result, mean)
    np.save(count_path, coverage)
    return result, count_path
