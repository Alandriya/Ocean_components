"""Daily scalar and bivariate quantile-cell diagnostics for Ocean_components.

Called from main.py via quantile_diagnostics or quantile_diagnostics_1d.
Uses the project's exact NP/SP grouping functions (scalar or joint).
Works period by period to avoid holding 1979--2026 fields in RAM together.
Does not estimate SP mixtures, recompute coefficient maps, or modify inputs.

Outputs for each period:
  daily_summary.csv: one row for each analysed transition and N;
  by_bin_summary.csv: per-N summary;
  config.json: reproducibility metadata.
For multiple periods, adds all_periods_summary.csv.

Spatial correlations use row/column pixel offsets. They are descriptive
and do not establish any asymptotic mixing assumptions.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np

from Coefficients.common import coerce_series, load_water_mask, prepare_pair_transition
from Data_processing.io import load_field


def _positive_unique(values: Iterable[int], label: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(v) for v in values))
    if not result or any(v < 1 for v in result):
        raise ValueError(f"{label} must contain positive integers")
    return result


def _mean_finite(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return round(float(arr.mean()), 6) if arr.size else None


def _weighted_p10_group_size(sizes: np.ndarray) -> int:
    """10th percentile of the size of the group containing a random valid pixel."""
    occupied = np.sort(sizes[sizes > 0])
    if not occupied.size:
        return 0
    counts = np.cumsum(occupied, dtype=np.int64)
    return int(occupied[np.searchsorted(counts, 0.1 * counts[-1], side="left")])


def _spatial_correlation(residual: np.ndarray, lag: int, axis: int, component: int) -> float:
    if lag >= residual.shape[axis]:
        return float("nan")
    if axis == 0:
        x, y = residual[:-lag, :, component], residual[lag:, :, component]
    else:
        x, y = residual[:, :-lag, component], residual[:, lag:, component]
    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 30:
        return float("nan")
    x, y = x[ok], y[ok]
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator else float("nan")


def diagnose_quantile_period(
    array1,
    array2,
    mask,
    out_dir,
    *,
    start_year: int,
    end_year: int,
    bins: Iterable[int] = (8, 12, 15, 20),
    thresholds: Iterable[int] = (2, 10, 20, 30, 50),
    first_transition: int = 1,
    stop_transition: int | None = None,
    every: int = 1,
    corr_every: int = 30,
    corr_bins: int = 15,
    corr_lags: Iterable[int] = (1, 4, 8),
    progress_every: int = 365,
) -> dict:
    """Analyze transitions for one period, writing CSV rows incrementally.

    ``array1`` and ``array2`` are time-first fields (T,H,W) or supported
    flattened formats. ``first_transition``/``stop_transition`` are endpoint
    indices in the given period: transitions X[t-1] -> X[t], with stop exclusive.
    The date is inferred from a contiguous daily record starting Jan 1
    ``start_year``; the function does not infer dates from ERA5 metadata.
    """
    bins = _positive_unique(bins, "bins")
    thresholds = _positive_unique(thresholds, "thresholds")
    lags = _positive_unique(corr_lags, "corr_lags")
    if first_transition < 1 or every < 1 or corr_every < 0 or progress_every < 1:
        raise ValueError("first_transition, every, progress_every >= 1 and corr_every >= 0 required")
    if corr_every and corr_bins not in bins:
        raise ValueError("corr_bins must be one of bins unless corr_every=0")
    a0, a1 = coerce_series(array1), coerce_series(array2)
    if a0.shape != a1.shape:
        raise ValueError(f"Field shapes do not match: {a0.shape} != {a1.shape}")
    water = load_water_mask(mask, a0.shape[1:])
    water_count = int(np.count_nonzero(water))
    if not water_count:
        raise ValueError("The water mask has no valid pixels")
    stop = len(a0) if stop_transition is None else min(len(a0), int(stop_transition))
    if first_transition >= stop:
        raise ValueError(f"No transitions: start={first_transition}, stop={stop}")

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    thresholds = tuple(sorted(set(thresholds)))
    lags = tuple(sorted(set(lags)))
    corr_fields = [f"resid_corr_lag{lag}_{direction}_component{component}"
                   for lag in lags for direction in ("row", "col") for component in (0, 1)]
    fields = [
        "period", "date", "transition_t", "N", "valid", "pct_valid_water",
        "occupied", "empty", "singletons", "min_occupied", "median_occupied",
        "p10_size_typical_observation", "max_occupied", "ties_axis0", "ties_axis1",
        "max_width_axis0", "max_width_axis1", "p95_width_axis0", "p95_width_axis1",
        "max_observed_cell_diameter",
    ]
    fields.extend(f"coverage_ge_{threshold}_pct" for threshold in thresholds)
    fields.extend(corr_fields)
    summary_stats = defaultdict(lambda: {
        "days": 0, "occupied": [], "empty": [], "ties": 0,
        "p10_typical": [], "diameters": [],
        "coverage": defaultdict(list), "corr": defaultdict(list),
    })
    total_days = len(range(first_transition, stop, every))
    first_date = date(int(start_year), 1, 1)
    label = f"{start_year}-{end_year}"
    with (directory / "daily_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, t in enumerate(range(first_transition, stop, every), start=1):
            day = (first_date + timedelta(days=t)).isoformat()
            for n in bins:
                grouped = prepare_pair_transition(a0[t-1], a0[t], a1[t-1], a1[t], n, water)
                sizes = np.zeros(n * n, dtype=np.int64)
                for group in grouped.groups:
                    sizes[group.bin_id] = len(group.increments)
                valid = int(grouped.n_valid_pairs)
                if int(sizes.sum()) != valid:
                    raise AssertionError("Joint group counts do not match valid transition pixels")
                positive = sizes[sizes > 0]
                edge0, edge1 = (np.asarray(edge, dtype=float) for edge in grouped.edges)
                width0, width1 = np.diff(edge0), np.diff(edge1)
                row = {
                    "period": label, "date": day, "transition_t": t, "N": n,
                    "valid": valid,
                    "pct_valid_water": round(100.0 * valid / water_count, 4),
                    "occupied": int(positive.size), "empty": int((sizes == 0).sum()),
                    "singletons": int((sizes == 1).sum()),
                    "min_occupied": int(positive.min()) if positive.size else 0,
                    "median_occupied": float(np.median(positive)) if positive.size else 0.0,
                    "p10_size_typical_observation": _weighted_p10_group_size(sizes),
                    "max_occupied": int(positive.max()) if positive.size else 0,
                    "ties_axis0": int(np.count_nonzero(width0 == 0)),
                    "ties_axis1": int(np.count_nonzero(width1 == 0)),
                    "max_width_axis0": float(width0.max()) if width0.size else 0.0,
                    "max_width_axis1": float(width1.max()) if width1.size else 0.0,
                    "p95_width_axis0": float(np.percentile(width0, 95)) if width0.size else 0.0,
                    "p95_width_axis1": float(np.percentile(width1, 95)) if width1.size else 0.0,
                    "max_observed_cell_diameter": float(np.hypot(width0.max(), width1.max()))
                    if width0.size and width1.size else 0.0,
                }
                for threshold in thresholds:
                    row[f"coverage_ge_{threshold}_pct"] = (
                        round(100.0 * sizes[sizes >= threshold].sum() / valid, 4)
                        if valid else 0.0
                    )
                if corr_every and n == corr_bins and (index - 1) % corr_every == 0:
                    residual = np.full((*a0.shape[1:], 2), np.nan, dtype=float)
                    flat = residual.reshape((-1, 2))
                    for group in grouped.groups:
                        flat[group.flat_indices] = group.increments - group.increments.mean(axis=0)
                    for lag in lags:
                        for direction, axis in (("row", 0), ("col", 1)):
                            for component in (0, 1):
                                row[f"resid_corr_lag{lag}_{direction}_component{component}"] = (
                                    _spatial_correlation(residual, lag, axis, component)
                                )
                writer.writerow(row)
                agg = summary_stats[n]
                agg["days"] += 1
                agg["occupied"].append(row["occupied"])
                agg["empty"].append(row["empty"])
                agg["ties"] += int(row["ties_axis0"] + row["ties_axis1"] > 0)
                agg["p10_typical"].append(row["p10_size_typical_observation"])
                agg["diameters"].append(row["max_observed_cell_diameter"])
                for threshold in thresholds:
                    agg["coverage"][threshold].append(row[f"coverage_ge_{threshold}_pct"])
                for field in corr_fields:
                    if field in row:
                        agg["corr"][field].append(row[field])
            if index == 1 or index % progress_every == 0 or index == total_days:
                print(f"[quantile_diagnostics] {label}: {index}/{total_days} transitions; "
                      f"last date={day}", flush=True)
                handle.flush()

    by_bin = []
    for n in bins:
        stat = summary_stats[n]
        result = {
            "period": label, "N": n, "days": stat["days"],
            "mean_occupied": _mean_finite(stat["occupied"]),
            "mean_empty": _mean_finite(stat["empty"]),
            "days_with_ties_pct": round(100.0 * stat["ties"] / stat["days"], 4),
            "mean_p10_size_typical_observation": _mean_finite(stat["p10_typical"]),
            "max_observed_cell_diameter_p95": round(float(np.percentile(stat["diameters"], 95)), 6),
        }
        for threshold in thresholds:
            values = stat["coverage"][threshold]
            result[f"mean_coverage_ge_{threshold}_pct"] = _mean_finite(values)
            result[f"min_coverage_ge_{threshold}_pct"] = min(values)
            result[f"p05_coverage_ge_{threshold}_pct"] = round(float(np.percentile(values, 5)), 4)
        for field in corr_fields:
            result[f"mean_{field}"] = _mean_finite(stat["corr"][field])
        by_bin.append(result)
    with (directory / "by_bin_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(by_bin[0]))
        writer.writeheader()
        writer.writerows(by_bin)
    (directory / "config.json").write_text(json.dumps({
        "period": label,
        "shape": list(a0.shape),
        "water_pixels": water_count,
        "start_year_first_observation_assumption": start_year,
        "first_transition": first_transition,
        "stop_transition_exclusive": stop,
        "every": every,
        "bins": list(bins), "thresholds": list(thresholds),
        "corr_every": corr_every, "corr_bins": corr_bins, "corr_lags_pixels": list(lags),
        "grouping": "Coefficients.common.prepare_pair_transition",
        "date_warning": "Dates assume contiguous daily observations beginning on 1 January of start_year.",
        "statistics_warning": "Spatial correlations are descriptive, not proof of any mixing condition.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "period": label,
        "transitions_analyzed": total_days,
        "output_directory": str(directory),
        "daily_summary": str(directory / "daily_summary.csv"),
        "by_bin_summary": str(directory / "by_bin_summary.csv"),
        "config": str(directory / "config.json"),
    }


def run_quantile_diagnostics(
    files_path_prefix,
    *,
    names=("sensible", "latent"),
    periods=((1979, 1989), (1989, 1999), (1999, 2009), (2009, 2019), (2019, 2026)),
    mask=None,
    bins=(8, 12, 15, 20),
    thresholds=(2, 10, 20, 30, 50),
    first_transition=1,
    stop_transition=None,
    every=1,
    corr_every=30,
    corr_bins=15,
    corr_lags=(1, 4, 8),
    progress_every=365,
    out_dir=None,
) -> dict:
    """Run the diagnostics across selected period files in repository DATA.

    Does not concatenate periods. Accordingly cross-boundary transitions
    (e.g. 1988-12-31 -> 1989-01-01, if split) are not analysed. The data
    of each period are loaded with Data_processing.io.load_field as in main.py.
    """
    root = Path(files_path_prefix)
    output_root = (Path(out_dir) if out_dir is not None
                   else root / "Functional" / "Diagnostics" / "quantile_cells")
    output_root.mkdir(parents=True, exist_ok=True)
    selected = tuple((int(start), int(end)) for start, end in periods)
    if not selected:
        raise ValueError("At least one period is required")
    if len(names) != 2:
        raise ValueError("Exactly two variable names required")
    reports = {}
    combined = []
    for start_year, end_year in selected:
        print(f"[quantile_diagnostics] Period {start_year}-{end_year}: loading "
              f"{names[0]}, {names[1]}", flush=True)
        first = load_field(root, names[0], start_year, end_year)
        second = load_field(root, names[1], start_year, end_year)
        if first.shape != second.shape:
            raise ValueError(f"Period {start_year}-{end_year}: source shapes differ")
        target = output_root / f"{start_year}-{end_year}"
        result = diagnose_quantile_period(
            first, second, mask, target,
            start_year=start_year, end_year=end_year,
            bins=bins, thresholds=thresholds,
            first_transition=first_transition, stop_transition=stop_transition,
            every=every, corr_every=corr_every, corr_bins=corr_bins,
            corr_lags=corr_lags, progress_every=progress_every,
        )
        reports[f"{start_year}-{end_year}"] = result
        with (target / "by_bin_summary.csv").open(encoding="utf-8", newline="") as handle:
            combined.extend(csv.DictReader(handle))
        del first, second
    aggregate_path = output_root / "all_periods_summary.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(f"[quantile_diagnostics] Saved combined summary: {aggregate_path}", flush=True)
    return {"periods": reports, "all_periods_summary": str(aggregate_path)}


def diagnose_quantile_period_1d(
    array,
    mask,
    out_dir,
    *,
    variable: str,
    start_year: int,
    end_year: int,
    bins: Iterable[int] = (20, 50, 100, 144, 200, 250),
    thresholds: Iterable[int] = (2, 10, 20, 30, 50),
    first_transition: int = 1,
    stop_transition: int | None = None,
    every: int = 1,
    corr_every: int = 30,
    corr_bins: int = 144,
    corr_lags: Iterable[int] = (1, 4, 8),
    progress_every: int = 365,
) -> dict:
    """Diagnose empirical 1D quantile groups for one scalar field.

    Uses ``Coefficients.common.prepare_transition`` (the identical grouping
    code used by scalar NP and SP). Does not estimate coefficients or alter
    observations. Every record corresponds to X[t-1] -> X[t], with t >= 1.
    ``stop_transition`` is an exclusive endpoint index.

    Actual group sizes matter with repeated state values: equal-width-by-rank
    occupancy is not assumed. Correlations are *descriptive*, not a mixing test.
    """
    from Coefficients.common import prepare_transition

    bins = _positive_unique(bins, "bins")
    thresholds = tuple(sorted(_positive_unique(thresholds, "thresholds")))
    lags = tuple(sorted(_positive_unique(corr_lags, "corr_lags"))) if corr_every else ()
    if first_transition < 1 or every < 1 or corr_every < 0 or progress_every < 1:
        raise ValueError("first_transition, every, progress_every >= 1; corr_every >= 0")
    if corr_every and corr_bins not in bins:
        raise ValueError("corr_bins must be in bins when corr_every > 0")
    if not variable or not isinstance(variable, str):
        raise ValueError("variable must be a nonempty name")
    field = coerce_series(array)
    water = load_water_mask(mask, field.shape[1:])
    n_water = int(water.sum())
    if not n_water:
        raise ValueError("The water mask has no valid pixels")
    stop = len(field) if stop_transition is None else min(len(field), int(stop_transition))
    if first_transition >= stop:
        raise ValueError(f"No transitions: first_transition={first_transition}, stop={stop}")
    period = f"{start_year}-{end_year}"
    first_day = date(int(start_year), 1, 1)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    corr_fields = [f"resid_corr_lag{lag}_{direction}" for lag in lags
                   for direction in ("row", "col")]
    fields = [
        "period", "variable", "date", "transition_t", "N", "valid",
        "pct_valid_water", "occupied", "empty", "singletons",
        "min_occupied", "median_occupied", "p10_size_typical_observation",
        "max_occupied", "zero_width_intervals", "min_positive_interval_width",
        "median_interval_width", "p95_interval_width", "max_interval_width",
        "distinct_state_values", "duplicate_observation_pct", "max_state_multiplicity",
    ]
    fields.extend(f"coverage_ge_{th}_pct" for th in thresholds)
    fields.extend(corr_fields)
    agg = defaultdict(lambda: {
        "days": 0, "occupied": [], "empty": [], "ties": 0,
        "p10_typical": [], "max_width": [], "duplicates": [],
        "coverage": defaultdict(list), "corr": defaultdict(list),
    })
    total_days = len(range(first_transition, stop, every))
    with (output / "daily_summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for day_number, t in enumerate(range(first_transition, stop, every), start=1):
            day = (first_day + timedelta(days=t)).isoformat()
            # Exactly the same finite-pair and water-mask selection as in
            # ``prepare_transition``. This count also lets us diagnose ties.
            valid_positions = (water & np.isfinite(field[t - 1])
                               & np.isfinite(field[t]))
            prev_candidates = np.asarray(field[t - 1][valid_positions], dtype=float)
            curr_candidates = np.asarray(field[t][valid_positions], dtype=float)
            prev_values = prev_candidates[np.isfinite(curr_candidates - prev_candidates)]
            unique_values, counts = np.unique(prev_values, return_counts=True)
            duplicate_pct = (100.0 * (len(prev_values) - len(unique_values))
                             / len(prev_values) if len(prev_values) else 0.0)
            max_multiplicity = int(counts.max()) if len(counts) else 0
            for n in bins:
                grouped = prepare_transition(field[t - 1], field[t], n, water)
                sizes = np.zeros(n, dtype=np.int64)
                for group in grouped.groups:
                    sizes[group.bin_id] = len(group.increments)
                valid = int(grouped.n_valid_pairs)
                if int(sizes.sum()) != valid:
                    raise AssertionError("Scalar group sizes do not sum to valid observations")
                if valid != int(len(prev_values)):
                    raise AssertionError("Validity selection differs from prepare_transition")
                occupied = sizes[sizes > 0]
                widths = np.diff(np.asarray(grouped.edges, dtype=float))
                positive_widths = widths[widths > 0]
                row = {
                    "period": period, "variable": variable, "date": day,
                    "transition_t": t, "N": n, "valid": valid,
                    "pct_valid_water": round(100.0 * valid / n_water, 4),
                    "occupied": int(len(occupied)),
                    "empty": int((sizes == 0).sum()),
                    "singletons": int((sizes == 1).sum()),
                    "min_occupied": int(occupied.min()) if occupied.size else 0,
                    "median_occupied": float(np.median(occupied)) if occupied.size else 0.0,
                    "p10_size_typical_observation": _weighted_p10_group_size(sizes),
                    "max_occupied": int(occupied.max()) if occupied.size else 0,
                    "zero_width_intervals": int(np.count_nonzero(widths == 0)),
                    "min_positive_interval_width": float(positive_widths.min())
                    if positive_widths.size else 0.0,
                    "median_interval_width": float(np.median(widths)) if widths.size else 0.0,
                    "p95_interval_width": float(np.percentile(widths, 95)) if widths.size else 0.0,
                    "max_interval_width": float(widths.max()) if widths.size else 0.0,
                    "distinct_state_values": int(len(unique_values)),
                    "duplicate_observation_pct": round(duplicate_pct, 6),
                    "max_state_multiplicity": max_multiplicity,
                }
                for th in thresholds:
                    row[f"coverage_ge_{th}_pct"] = (
                        round(100.0 * int(sizes[sizes >= th].sum()) / valid, 4)
                        if valid else 0.0
                    )
                if corr_every and n == corr_bins and (day_number - 1) % corr_every == 0:
                    residual = np.full(field.shape[1:], np.nan, dtype=float)
                    flat_residual = residual.ravel()
                    for group in grouped.groups:
                        flat_residual[group.flat_indices] = (
                            group.increments - group.increments.mean()
                        )
                    # Existing 2D helper expects a component axis.
                    residual_3d = residual[..., None]
                    for lag in lags:
                        for direction, axis in (("row", 0), ("col", 1)):
                            row[f"resid_corr_lag{lag}_{direction}"] = (
                                _spatial_correlation(residual_3d, lag, axis, 0)
                            )
                writer.writerow(row)
                stat = agg[n]
                stat["days"] += 1
                stat["occupied"].append(row["occupied"])
                stat["empty"].append(row["empty"])
                stat["ties"] += int(row["zero_width_intervals"] > 0)
                stat["p10_typical"].append(row["p10_size_typical_observation"])
                stat["max_width"].append(row["max_interval_width"])
                stat["duplicates"].append(row["duplicate_observation_pct"])
                for th in thresholds:
                    stat["coverage"][th].append(row[f"coverage_ge_{th}_pct"])
                for key in corr_fields:
                    if key in row:
                        stat["corr"][key].append(row[key])
            if day_number == 1 or day_number % progress_every == 0 or day_number == total_days:
                print(f"[quantile_diagnostics_1d] {variable} {period}: "
                      f"{day_number}/{total_days} transitions; last date={day}", flush=True)
                fp.flush()

    summaries = []
    for n in bins:
        stat = agg[n]
        item = {
            "period": period, "variable": variable, "N": n,
            "days": stat["days"],
            "mean_occupied": _mean_finite(stat["occupied"]),
            "mean_empty": _mean_finite(stat["empty"]),
            "days_with_zero_width_intervals_pct": round(
                100.0 * stat["ties"] / stat["days"], 4),
            "mean_p10_size_typical_observation": _mean_finite(stat["p10_typical"]),
            "mean_duplicate_observation_pct": _mean_finite(stat["duplicates"]),
            "p95_max_interval_width": round(float(np.percentile(stat["max_width"], 95)), 6),
        }
        for th in thresholds:
            values = stat["coverage"][th]
            item[f"mean_coverage_ge_{th}_pct"] = _mean_finite(values)
            item[f"min_coverage_ge_{th}_pct"] = min(values)
            item[f"p05_coverage_ge_{th}_pct"] = round(float(np.percentile(values, 5)), 4)
        for key in corr_fields:
            item[f"mean_{key}"] = _mean_finite(stat["corr"][key])
        summaries.append(item)
    with (output / "by_bin_summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / "config.json").write_text(json.dumps({
        "dimension": 1,
        "variable": variable,
        "period": period,
        "shape": list(field.shape),
        "water_pixels": n_water,
        "first_transition": first_transition,
        "stop_transition_exclusive": stop,
        "every": every,
        "bins": list(bins),
        "thresholds": list(thresholds),
        "corr_every": corr_every,
        "corr_bins": corr_bins,
        "corr_lags_pixels": list(lags),
        "grouping": "Coefficients.common.prepare_transition",
        "date_warning": "Dates assume daily observations from 1 January of start_year.",
        "statistics_warning": "Spatial correlations are descriptive, not proof of mixing.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "period": period,
        "variable": variable,
        "transitions_analyzed": total_days,
        "output_directory": str(output),
        "daily_summary": str(output / "daily_summary.csv"),
        "by_bin_summary": str(output / "by_bin_summary.csv"),
        "config": str(output / "config.json"),
    }


def run_quantile_diagnostics_1d(
    files_path_prefix,
    *,
    variable: str = "sensible",
    periods=((1979, 1989), (1989, 1999), (1999, 2009),
             (2009, 2019), (2019, 2026)),
    mask=None,
    bins=(20, 50, 100, 144, 200, 250),
    thresholds=(2, 10, 20, 30, 50),
    first_transition=1,
    stop_transition=None,
    every=1,
    corr_every=30,
    corr_bins=144,
    corr_lags=(1, 4, 8),
    progress_every=365,
    out_dir=None,
) -> dict:
    """Run scalar diagnostics over requested periods in the same main.py style.

    Store sensible and latent separately; never overwrite a different variable.
    Processes one loaded period at a time. No cross-period transition analysis.
    """
    if not isinstance(variable, str) or not variable:
        raise ValueError("variable must be a nonempty string")
    selected = tuple((int(start), int(end)) for start, end in periods)
    if not selected:
        raise ValueError("At least one period is required")
    root = Path(files_path_prefix)
    output_root = (Path(out_dir) if out_dir is not None
                   else root / "Functional" / "Diagnostics" / "quantile_cells_1d" / variable)
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    combined = []
    for start_year, end_year in selected:
        print(f"[quantile_diagnostics_1d] Loading {variable}: "
              f"{start_year}-{end_year}", flush=True)
        array = load_field(root, variable, start_year, end_year)
        target = output_root / f"{start_year}-{end_year}"
        result = diagnose_quantile_period_1d(
            array, mask, target, variable=variable,
            start_year=start_year, end_year=end_year,
            bins=bins, thresholds=thresholds,
            first_transition=first_transition, stop_transition=stop_transition,
            every=every, corr_every=corr_every, corr_bins=corr_bins,
            corr_lags=corr_lags, progress_every=progress_every,
        )
        results[f"{start_year}-{end_year}"] = result
        with (target / "by_bin_summary.csv").open(encoding="utf-8", newline="") as fp:
            combined.extend(csv.DictReader(fp))
        del array
    aggregate = output_root / "all_periods_summary.csv"
    with aggregate.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(f"[quantile_diagnostics_1d] Saved {aggregate}", flush=True)
    return {"variable": variable, "periods": results, "all_periods_summary": str(aggregate)}
