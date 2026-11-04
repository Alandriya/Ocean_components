"""Compare daily ERA5 arrays over actual overlapping calendar dates.

The comparison streams spatial blocks from memory-mapped NPY arrays and never
loads an entire 2019-2026 field collection into RAM. NaN disagreement counts
as a mismatch, while NaNs in the same positions are accepted.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np


def _as_space_time(array: np.ndarray, n_space: int) -> np.ndarray:
    """Expose historical (space, time), (time, space) and map layouts."""
    if array.ndim == 2:
        if array.shape[0] == n_space:
            return array
        if array.shape[1] == n_space:
            return array.T
    elif array.ndim == 3:
        if array.shape[1] * array.shape[2] == n_space:
            return array.reshape(array.shape[0], n_space).T
        if array.shape[0] * array.shape[1] == n_space:
            return array.reshape(n_space, array.shape[2])
    raise ValueError(f"Historical shape {array.shape} is incompatible with {n_space} cells")


def compare_overlap(old_npy, new_npy, new_dates_npy, *, old_start,
                    rtol=1e-5, atol=1e-4, block_pixels=512, max_days=None):
    """Calculate overlap equality, RMSE and the largest numerical difference.

    Parameters
    ----------
    old_npy, new_npy : str or Path
        Old and newly converted NPY files, typically (space, day).
    new_dates_npy : str or Path
        The calendar dates saved by the ERA5 GRIB parser.
    old_start : str
        ISO date of the first daily column of the historical array.
    rtol, atol : float
        Numerical comparison tolerances accepted by numpy.isclose.
    block_pixels : int
        Spatial block size for bounded memory usage.
    max_days : int, optional
        Optional number of overlapping dates to inspect first.

    Returns
    -------
    dict
        Equality verdict, dates, compared cells and error statistics.

    Notes
    -----
    This compares the same grid-cell indices; it cannot prove equal geographical
    orientation without the original latitude/longitude coordinates.
    """
    original = np.load(old_npy, mmap_mode="r", allow_pickle=False)
    converted = np.load(new_npy, mmap_mode="r", allow_pickle=False)
    dates = np.load(new_dates_npy, allow_pickle=False).astype("datetime64[D]")
    if converted.ndim != 2 or converted.shape[1] != len(dates):
        raise ValueError("New data and its dates file have inconsistent shapes")
    if len(dates) == 0 or np.any(np.diff(dates) != np.timedelta64(1, "D")):
        raise ValueError("New dates are empty, repeated or non-consecutive")
    historic = _as_space_time(original, converted.shape[0])
    old_first = np.datetime64(date.fromisoformat(old_start), "D")
    original_indices = (dates - old_first).astype("timedelta64[D]").astype(np.int64)
    selected = np.flatnonzero((original_indices >= 0) & (original_indices < historic.shape[1]))
    if max_days is not None:
        selected = selected[:int(max_days)]
    if not len(selected):
        raise ValueError("No overlapping dates: verify old_start and date coverage")
    previous_indices = original_indices[selected]
    total_compared = 0
    total_finite = 0
    total_different = 0
    missing_pattern_mismatch = 0
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    max_absolute_error = 0.0
    max_relative_error = 0.0
    sum_new = sum_old = 0.0
    finite_value_count = 0
    for start in range(0, converted.shape[0], block_pixels):
        end = min(start + block_pixels, converted.shape[0])
        old_values = np.asarray(historic[start:end, previous_indices], dtype=np.float64)
        new_values = np.asarray(converted[start:end, selected], dtype=np.float64)
        finite_old = np.isfinite(old_values)
        finite_new = np.isfinite(new_values)
        both_finite = finite_old & finite_new
        both_nan = np.isnan(old_values) & np.isnan(new_values)
        allowed = np.isclose(old_values, new_values, rtol=rtol, atol=atol,
                             equal_nan=True)
        count = old_values.size
        total_compared += count
        total_different += int(np.count_nonzero(~allowed))
        missing_pattern_mismatch += int(np.count_nonzero(finite_old != finite_new))
        total_finite += int(np.count_nonzero(both_finite))
        if np.any(both_finite):
            differences = old_values[both_finite] - new_values[both_finite]
            absolute = np.abs(differences)
            squared_error_sum += float(np.sum(differences ** 2))
            absolute_error_sum += float(np.sum(absolute))
            max_absolute_error = max(max_absolute_error, float(np.max(absolute)))
            relative = absolute / np.maximum(np.abs(old_values[both_finite]), atol)
            max_relative_error = max(max_relative_error, float(np.max(relative)))
            sum_old += float(np.sum(old_values[both_finite]))
            sum_new += float(np.sum(new_values[both_finite]))
            finite_value_count += int(np.count_nonzero(both_finite))
    report = {
        "match": total_different == 0,
        "overlap_first_date": str(dates[selected[0]]),
        "overlap_last_date": str(dates[selected[-1]]),
        "days": len(selected),
        "pixels": converted.shape[0],
        "compared_cells": total_compared,
        "both_finite_cells": total_finite,
        "different_cells": total_different,
        "different_percent": 100.0 * total_different / max(1, total_compared),
        "finite_mask_mismatches": missing_pattern_mismatch,
        "maximum_absolute_error": max_absolute_error,
        "mean_absolute_error": absolute_error_sum / total_finite if total_finite else None,
        "rmse": float(np.sqrt(squared_error_sum / total_finite)) if total_finite else None,
        "maximum_relative_error": max_relative_error,
        "old_mean": sum_old / finite_value_count if finite_value_count else None,
        "new_mean": sum_new / finite_value_count if finite_value_count else None,
        "rtol": rtol, "atol": atol,
    }
    if finite_value_count and abs(sum_new) > 1e-20:
        ratio = sum_old / sum_new
        report["old_to_new_mean_ratio"] = ratio
        if np.isclose(abs(ratio), 3600.0, rtol=0.1):
            report["hint"] = "A factor of about 3600 suggests J m-2 versus W m-2 units"
        elif np.isclose(abs(ratio), 1 / 3600.0, rtol=0.1):
            report["hint"] = "A factor of about 1/3600 suggests J m-2 versus W m-2 units"
        elif np.isclose(ratio, -1.0, rtol=0.1):
            report["hint"] = "Opposite sign conventions may explain disagreement"
    return report


def main():
    """Compare one pair of historical and new arrays from the command line."""
    parser = argparse.ArgumentParser(description="Compare ERA5 daily arrays by UTC date")
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--dates", type=Path, required=True)
    parser.add_argument("--old-start", required=True,
                        help="First daily observation in the historical file, YYYY-MM-DD")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()
    outcome = compare_overlap(args.old, args.new, args.dates, old_start=args.old_start,
                              rtol=args.rtol, atol=args.atol, max_days=args.max_days)
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    raise SystemExit(0 if outcome["match"] else 1)


if __name__ == "__main__":
    main()
