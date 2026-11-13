"""Randomized-hour and matched-size spatial-thinning diagnostics for ERA5.

One fixed 24-hour transition X[d,h] -> X[d+1,h], with NO manual clock shift.
For each date and replicate: a single random permutation of the 24 hours
produces nested subsamples of size 4, 8, and 12 (configurable). A regular
spatial-thinning mask with one offset per b x b block is compared against a
uniformly random ocean-pixel mask of the SAME size. The reference is a pooled
24-hour fit on all ocean pixels, NOT a known true conditional moment.

The source uses the existing HourlySource, including its sequential GRIB
streaming/cache. Where possible, completed GRIB day caches made by the earlier
quantile_hourly task are reused without rescanning multi-GB inputs.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from Coefficients.common import _bin_ids, load_water_mask
from Coefficients.quantile_hourly import (HourlySource, VARIABLE_ALIASES,
                                          _daily_average, _resolve_paths,
                                          _stability_at_fixed_states)
from Coefficients.grib_stream_cache import GRIB_SUFFIXES


OUTPUT_FIELDS = [
    "date", "replicate", "hours_used", "selected_hours", "spatial_step",
    "spatial_strategy", "spatial_points", "nominal_pairs", "dimension",
    "variable", "N", "valid", "occupied", "empty", "singletons",
    "median_occupied", "p10_size_typical_observation", "coverage_ge_30_pct",
    "zero_width_axis0", "zero_width_axis1", "anchor_points",
    "matched_points", "matched_pct", "rms_mean_vs_hourly24",
    "rms_cov_vs_hourly24",
]


def _positive_unique(values, name):
    result = tuple(dict.fromkeys(int(x) for x in values))
    if not result or any(x <= 0 for x in result):
        raise ValueError(f"{name}: expected positive integers")
    return result


def _days(first, last, stride, limit):
    if isinstance(first, str):
        first = date.fromisoformat(first)
    if isinstance(last, str):
        last = date.fromisoformat(last)
    if last < first:
        raise ValueError("last_date is before first_date")
    out = [first + timedelta(days=i) for i in range(0, (last-first).days+1, stride)]
    return out[:limit] if limit is not None else out


def _spatial_choices(water, step, rng):
    """Two masks with exactly the same number of ocean pixels, if step > 1.

    grid: one common random phase modulo step in both axes (minimum spacing
    step along each grid coordinate). random: the same count, uniformly selected
    without replacement from all available ocean pixels.
    """
    if step == 1:
        return (("grid", np.asarray(water, dtype=bool)),)
    oy, ox = (int(rng.integers(0, step)), int(rng.integers(0, step)))
    grid = np.zeros(water.shape, dtype=bool)
    grid[oy::step, ox::step] = True
    grid &= water
    count = int(grid.sum())
    pool = np.flatnonzero(water.ravel())
    random_mask = np.zeros(water.size, dtype=bool)
    if count:
        random_mask[rng.choice(pool, size=count, replace=False)] = True
    return (("grid", grid), ("random", random_mask.reshape(water.shape)))


def _group_fit(states, increments, dim, bins):
    """Exact quantile-bin labels from common._bin_ids; pooled moments via bincount.

    The original estimator uses population (1/m) cell covariance. Its actual
    bin convention, including tie handling, is preserved via _bin_ids.
    """
    states = np.asarray(states, dtype=np.float64).reshape(-1, dim)
    increments = np.asarray(increments, dtype=np.float64).reshape(-1, dim)
    valid = np.all(np.isfinite(states) & np.isfinite(increments), axis=1)
    states, increments = states[valid], increments[valid]
    n = len(states)
    nb = bins ** dim
    edges = []
    labels = np.zeros(n, dtype=np.int64)
    if n:
        for k in range(dim):
            ids, bound = _bin_ids(states[:, k], bins)
            labels = labels * bins + ids
            edges.append(np.asarray(bound, dtype=float))
        sizes = np.bincount(labels, minlength=nb)
    else:
        sizes = np.zeros(nb, dtype=np.int64)
    mean = np.zeros((nb, dim), dtype=float)
    cov = np.zeros((nb, dim, dim), dtype=float)
    if n:
        for k in range(dim):
            sums = np.bincount(labels, weights=increments[:, k], minlength=nb)
            np.divide(sums, sizes, out=mean[:, k], where=sizes > 0)
        centered = increments - mean[labels]
        for k in range(dim):
            for j in range(k, dim):
                sums = np.bincount(labels, weights=centered[:, k]*centered[:, j],
                                   minlength=nb)
                np.divide(sums, sizes, out=cov[:, k, j], where=sizes > 0)
                cov[:, j, k] = cov[:, k, j]
    positive = np.sort(sizes[sizes > 0])
    if positive.size:
        cdf = np.cumsum(positive)
        tenth = int(positive[np.searchsorted(cdf, 0.1*cdf[-1], side="left")])
    else:
        tenth = 0
    coverage = float(100.0 * sizes[sizes >= 30].sum() / n) if n else 0.0
    row = {
        "valid": n,
        "occupied": int(positive.size),
        "empty": int((sizes == 0).sum()),
        "singletons": int((sizes == 1).sum()),
        "median_occupied": float(np.median(positive)) if positive.size else 0.0,
        "p10_size_typical_observation": tenth,
        "coverage_ge_30_pct": coverage,
        "zero_width_axis0": int(np.count_nonzero(np.diff(edges[0]) == 0)) if n else None,
        "zero_width_axis1": int(np.count_nonzero(np.diff(edges[1]) == 0)) if n and dim == 2 else None,
    }
    return row, {"edges": edges, "mean": mean, "cov": cov, "sizes": sizes,
                 "dim": dim, "N": bins}


def _pairs(before, after, hours, pixels, dimension, variable):
    """Return states and 24h increments for selected hours and ocean pixels."""
    if dimension == 1:
        p = before[variable][list(hours)].reshape(len(hours), -1)[:, pixels]
        q = after[variable][list(hours)].reshape(len(hours), -1)[:, pixels]
        return p.reshape(-1, 1), (q-p).reshape(-1, 1)
    p0 = before["sensible"][list(hours)].reshape(len(hours), -1)[:, pixels]
    p1 = before["latent"][list(hours)].reshape(len(hours), -1)[:, pixels]
    q0 = after["sensible"][list(hours)].reshape(len(hours), -1)[:, pixels]
    q1 = after["latent"][list(hours)].reshape(len(hours), -1)[:, pixels]
    return np.stack((p0, p1), axis=-1).reshape(-1, 2), \
           np.stack((q0-p0, q1-p1), axis=-1).reshape(-1, 2)


def _signature(*, paths, first, last, stride, seed, reps, hour_counts,
               spatial_steps, bins_1d, bins_2d, min_group, shape, source_stride,
               npy_start, npy_variable_names, water):
    return {
        "version": 1,
        "sources": [{"path": str(p), "size": p.stat().st_size,
                     "mtime_ns": p.stat().st_mtime_ns} for p in paths],
        "first_date": first.isoformat(), "last_date": last.isoformat(),
        "day_stride": stride, "seed": seed,
        "replicates": reps, "hour_counts": list(hour_counts),
        "spatial_steps": list(spatial_steps),
        "bins_1d": list(bins_1d), "bins_2d": list(bins_2d),
        "min_group": min_group, "grid_shape": list(shape),
        "requested_source_stride": str(source_stride),
        "npy_start": str(npy_start) if npy_start is not None else None,
        "npy_variable_names": dict(npy_variable_names or {}),
        "water_count": int(np.count_nonzero(water)),
        "water_mask_sha256": hashlib.sha256(np.asarray(water, dtype=np.uint8).tobytes()).hexdigest(),
    }


def _cache_matches(cache_dir, paths, required_days, shape, stride):
    """Use a previous *completed* cache only if provenance and coverage match."""
    manifest = Path(cache_dir) / "complete_manifest.json"
    if not manifest.is_file():
        return False
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        sig = meta["signature"]
        expected = [{"path": str(p), "size": p.stat().st_size,
                     "mtime_ns": p.stat().st_mtime_ns} for p in paths]
        if sig["sources"] != expected or sig["grid_shape"] != list(shape):
            return False
        if stride != "auto" and int(stride) != int(meta["spatial_stride"]):
            return False
        if not set(VARIABLE_ALIASES).issubset(meta["complete_days"]):
            return False
        for var in VARIABLE_ALIASES:
            days = set(meta["complete_days"][var])
            if not all(d.isoformat() in days and
                       (Path(cache_dir) / var / f"{d.isoformat()}.npy").is_file()
                       for d in required_days):
                return False
        return True
    except (KeyError, ValueError, OSError, TypeError):
        return False


class _ExistingDayCache:
    """Read-only source from a previously completed, provenance-checked cache."""
    def __init__(self, path, paths, grid_shape):
        self.path = Path(path)
        self.source_files = [str(p) for p in paths]
        self.grid_shape = grid_shape
        meta = json.loads((self.path / "complete_manifest.json").read_text(encoding="utf-8"))
        self.source_shape = tuple(meta["source_shape"])
        self.spatial_stride = int(meta["spatial_stride"])
        self.available = {var: set(meta["complete_days"][var]) for var in VARIABLE_ALIASES}

    def has_complete_day(self, day):
        return all(day.isoformat() in self.available[var] for var in VARIABLE_ALIASES)

    def day(self, day):
        result = {}
        for var in VARIABLE_ALIASES:
            path = self.path / var / f"{day.isoformat()}.npy"
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if data.shape != (24, *self.grid_shape):
                raise ValueError(f"Cache shape mismatch: {path}: {data.shape}")
            result[var] = data
        return result

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@contextmanager
def _open_source(root, paths, required_days, grid_shape, stride, cache_dir,
                 npy_start=None, npy_variable_names=None):
    paths = _resolve_paths(root, paths)
    gribs = [p for p in paths if p.suffix.lower() in GRIB_SUFFIXES]
    if gribs and len(gribs) == len(paths):
        candidates = ([Path(cache_dir)] if cache_dir is not None else [])
        if cache_dir is None:
            candidates += sorted((Path(root) / "Functional" / "Diagnostics" /
                                  "quantile_hourly").glob("*/_grib_day_cache"))
        found = next((p for p in dict.fromkeys(candidates)
                      if _cache_matches(p, gribs, required_days, grid_shape, stride)), None)
        if found is not None:
            print(f"[quantile_resampling] Reusing existing GRIB day cache: {found}", flush=True)
            with _ExistingDayCache(found, gribs, grid_shape) as source:
                yield source
            return
    fallback = (Path(cache_dir) if cache_dir is not None else
                Path(root)/"Functional"/"Diagnostics"/"quantile_resampling"/"_grib_day_cache")
    with HourlySource(root=root, paths=paths, spatial_shape=grid_shape,
                      stride=stride, required_days=required_days,
                      npy_start=npy_start, npy_variable_names=npy_variable_names,
                      grib_cache_dir=fallback) as source:
        yield source


def _summarize(rows, group_fields):
    group = defaultdict(list)
    for r in rows:
        group[tuple(r[k] for k in group_fields)].append(r)
    out = []
    for key, records in sorted(group.items()):
        rec = dict(zip(group_fields, key))
        rec["rows"] = len(records)
        rec["days"] = len({r["date"] for r in records})
        for name in ("spatial_points", "valid", "occupied", "empty",
                     "median_occupied", "p10_size_typical_observation", "coverage_ge_30_pct",
                     "matched_pct", "rms_mean_vs_hourly24", "rms_cov_vs_hourly24"):
            values = np.asarray([float(r[name]) for r in records if r.get(name) not in ("", None)], float)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            rec[f"mean_{name}"] = float(values.mean())
            rec[f"std_{name}"] = float(values.std(ddof=1)) if len(values) > 1 else None
            rec[f"p05_{name}"] = float(np.percentile(values, 5))
            rec[f"p95_{name}"] = float(np.percentile(values, 95))
        out.append(rec)
    return out


def _save_rows(path, rows, fields):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_completed(output_file, signature_file, signature, dates, expected_per_day, resume):
    """Keep only complete, unique days; never duplicate partial CSV days."""
    if not resume or not output_file.exists():
        return set(), []
    if not signature_file.exists():
        raise ValueError("Existing CSV has no run_signature.json; select a new output directory")
    old = json.loads(signature_file.read_text(encoding="utf-8"))
    if old != signature:
        raise ValueError("Resampling settings or input files changed. "
                         "Choose a new output_dir or set resample_resume=False.")
    with output_file.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != OUTPUT_FIELDS:
            raise ValueError("Existing resampling CSV uses an incompatible schema")
        all_rows = list(reader)
    by_day = defaultdict(list)
    for row in all_rows:
        by_day[row["date"]].append(row)
    needed = {d.isoformat() for d in dates}
    completed = set()
    retained = []
    for name, dayrows in by_day.items():
        keys = [(r["replicate"], r["hours_used"], r["spatial_step"],
                 r["spatial_strategy"], r["dimension"], r["variable"], r["N"])
                for r in dayrows]
        if name in needed and len(keys) == expected_per_day and len(set(keys)) == expected_per_day:
            completed.add(name)
            retained.extend(dayrows)
    return completed, retained


def _paired_comparison(rows):
    """Compare equal-count grid and random spatial choices, same day/hour draw."""
    bykey = defaultdict(dict)
    for r in rows:
        if int(r["spatial_step"]) == 1:
            continue
        key = (r["date"], r["replicate"], r["hours_used"],
               r["spatial_step"], r["dimension"], r["variable"], r["N"])
        bykey[key][r["spatial_strategy"]] = r
    columns = ["date", "replicate", "hours_used", "spatial_step", "dimension",
               "variable", "N", "spatial_points", "grid_rms_mean", "random_rms_mean",
               "grid_minus_random_rms_mean", "grid_rms_cov", "random_rms_cov",
               "grid_minus_random_rms_cov"]
    result = []
    for key, pair in bykey.items():
        if "grid" not in pair or "random" not in pair:
            continue
        a, b = pair["grid"], pair["random"]
        rec = dict(zip(columns[:7], key))
        if a["spatial_points"] != b["spatial_points"]:
            raise AssertionError("Matched spatial masks contain different pixel counts")
        rec["spatial_points"] = a["spatial_points"]
        for field, field_a, field_b, diff_field in [
            ("rms_mean_vs_hourly24", "grid_rms_mean", "random_rms_mean", "grid_minus_random_rms_mean"),
            ("rms_cov_vs_hourly24", "grid_rms_cov", "random_rms_cov", "grid_minus_random_rms_cov")
        ]:
            va, vb = a.get(field), b.get(field)
            rec[field_a], rec[field_b] = va, vb
            try:
                rec[diff_field] = float(va)-float(vb)
            except (TypeError, ValueError):
                rec[diff_field] = None
        result.append(rec)
    return columns, result


def run_quantile_hourly_resampling(
    files_path_prefix, *, hourly_files, mask,
    start_year=1979, end_year=1989, first_date=None, last_date=None,
    day_stride=7, max_days=None, hours=(4, 8, 12), replicates=3,
    spatial_steps=(1, 2, 4), bins_1d=(144,), bins_2d=(15, 20),
    random_seed=1979, compare_min_group_size=30,
    hourly_spatial_stride="auto", hourly_npy_start=None,
    hourly_npy_variable_names=None, grib_cache_dir=None,
    output_dir=None, resume=True, progress_every=10,
):
    """Run a reproducible factorial randomized-hour/spatial-thinning experiment.

    All comparisons use fixed 24-hour transitions and the SAME daily-mean4
    state query points. Reference: fit to all 24 hours/all ocean pixels, not
    ground truth. The random and grid spatial masks have exactly equal size.
    """
    hours = _positive_unique(hours, "hours")
    steps = _positive_unique(spatial_steps, "spatial_steps")
    bins1 = _positive_unique(bins_1d, "bins_1d")
    bins2 = _positive_unique(bins_2d, "bins_2d")
    if any(h > 24 for h in hours):
        raise ValueError("Hour subset sizes must be <= 24")
    if int(replicates) < 1 or int(day_stride) < 1 or int(progress_every) < 1:
        raise ValueError("replicates, day_stride, progress_every must be >= 1")
    if int(compare_min_group_size) < 1:
        raise ValueError("compare_min_group_size must be positive")
    if max_days is not None and int(max_days) < 1:
        raise ValueError("max_days must be None or positive")
    if int(start_year) >= int(end_year):
        raise ValueError("end_year is exclusive and must be > start_year")
    root = Path(files_path_prefix)
    out = (Path(output_dir) if output_dir is not None else
           root / "Functional" / "Diagnostics" / "quantile_resampling" /
           f"{start_year}-{end_year}")
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    shape = tuple(np.asarray(mask).shape) if mask is not None else (161, 181)
    if len(shape) != 2:
        raise ValueError("Expected 2D ocean mask")
    water = load_water_mask(mask, shape)
    if not water.any():
        raise ValueError("Ocean mask has no valid pixels")
    start = date.fromisoformat(first_date) if isinstance(first_date, str) else (
        first_date or date(start_year, 1, 1))
    end = date.fromisoformat(last_date) if isinstance(last_date, str) else (
        last_date or date(end_year, 1, 1)-timedelta(days=1))
    dates = _days(start, end, int(day_stride), max_days)
    required = set(dates) | {d + timedelta(days=1) for d in dates}
    paths = _resolve_paths(root, hourly_files)
    signature = _signature(paths=paths, first=start, last=end, stride=day_stride,
                           seed=random_seed, reps=replicates,
                           hour_counts=hours, spatial_steps=steps,
                           bins_1d=bins1, bins_2d=bins2,
                           min_group=compare_min_group_size,
                           shape=shape, source_stride=hourly_spatial_stride,
                           npy_start=hourly_npy_start,
                           npy_variable_names=hourly_npy_variable_names,
                           water=water)
    csv_path = out / "resampling_daily.csv"
    signature_path = out / "run_signature.json"
    n_fits = 2 * len(bins1) + len(bins2)
    spatial_variants = sum(1 if x == 1 else 2 for x in steps)
    expected_per_day = int(replicates) * len(hours) * spatial_variants * n_fits
    completed, old_rows = _read_completed(csv_path, signature_path, signature, dates,
                                          expected_per_day, bool(resume))
    # Persist the exact run configuration before writing any daily CSV.
    signature_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    if completed:
        print(f"[quantile_resampling] Resuming {len(completed)} complete days", flush=True)
    # Drop any partially written tail before appending (atomic replacement).
    temporary = csv_path.with_suffix(".tmp")
    _save_rows(temporary, old_rows, OUTPUT_FIELDS)
    temporary.replace(csv_path)
    processed = set(completed)
    skipped = []
    all_ocean_pixels = np.flatnonzero(water.ravel())
    configs = [(1, "sensible", n) for n in bins1] + \
              [(1, "latent", n) for n in bins1] + \
              [(2, "sensible_latent", n) for n in bins2]
    with _open_source(root, paths, required, shape, hourly_spatial_stride,
                      grib_cache_dir, hourly_npy_start,
                      hourly_npy_variable_names) as source, \
         csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        for position, day in enumerate(dates, start=1):
            if day.isoformat() in processed:
                continue
            if not source.has_complete_day(day) or not source.has_complete_day(day+timedelta(days=1)):
                skipped.append(day.isoformat())
                continue
            before, after = source.day(day), source.day(day+timedelta(days=1))
            anchor = {var: _daily_average(before[var]) for var in VARIABLE_ALIASES}
            query = {
                (1, "sensible"): anchor["sensible"].ravel()[all_ocean_pixels].reshape(-1, 1),
                (1, "latent"): anchor["latent"].ravel()[all_ocean_pixels].reshape(-1, 1),
                (2, "sensible_latent"): np.stack(
                    (anchor["sensible"].ravel()[all_ocean_pixels],
                     anchor["latent"].ravel()[all_ocean_pixels]), axis=-1),
            }
            full_states = {}
            full_increments = {}
            reference = {}
            for dim, var, n in configs:
                pair_key = (dim, var)
                if pair_key not in full_states:
                    full_states[pair_key], full_increments[pair_key] = _pairs(
                        before, after, tuple(range(24)), all_ocean_pixels, dim, var)
                _, ref_fit = _group_fit(full_states[pair_key], full_increments[pair_key], dim, n)
                reference[dim, var, n] = ref_fit
            row_count = 0
            for rep in range(int(replicates)):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [int(random_seed), (day-date(1970,1,1)).days, rep]))
                hour_order = rng.permutation(24)
                choices = {b: _spatial_choices(water, b, rng) for b in steps}
                for b in steps:
                    for strategy, spatial_mask in choices[b]:
                        pixels = np.flatnonzero(spatial_mask.ravel())
                        for count in hours:
                            subset = tuple(sorted(int(h) for h in hour_order[:count]))
                            point_sets = {}
                            for dim, var, n in configs:
                                pair_key = (dim, var)
                                if pair_key not in point_sets:
                                    point_sets[pair_key] = _pairs(
                                        before, after, subset, pixels, dim, var)
                                states, increments = point_sets[pair_key]
                                stats, fit = _group_fit(states, increments, dim, n)
                                comparison = _stability_at_fixed_states(
                                    fit, reference[dim, var, n], query[pair_key],
                                    min_size=int(compare_min_group_size))
                                row = {
                                    "date": day.isoformat(), "replicate": rep,
                                    "hours_used": count,
                                    "selected_hours": " ".join(f"{h:02d}" for h in subset),
                                    "spatial_step": b, "spatial_strategy": strategy,
                                    "spatial_points": len(pixels),
                                    "nominal_pairs": len(pixels)*count,
                                    "dimension": dim, "variable": var, "N": n,
                                    **stats, **comparison,
                                }
                                writer.writerow(row)
                                row_count += 1
            if row_count != expected_per_day:
                raise AssertionError(f"Expected {expected_per_day} rows/day, got {row_count}")
            handle.flush()
            processed.add(day.isoformat())
            if position == 1 or position % progress_every == 0 or position == len(dates):
                print(f"[quantile_resampling] {position}/{len(dates)} dates, "
                      f"day={day}, saved={len(processed)}, rows/day={row_count}", flush=True)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary_fields = ("hours_used", "spatial_step", "spatial_strategy",
                      "dimension", "variable", "N")
    summaries = _summarize(rows, summary_fields)
    if summaries:
        keys = list(dict.fromkeys(k for r in summaries for k in r))
        _save_rows(out / "resampling_summary.csv", summaries, keys)
    day_summary_fields = ("date", "hours_used", "spatial_step", "spatial_strategy",
                          "dimension", "variable", "N")
    per_day = _summarize(rows, day_summary_fields)
    if per_day:
        day_columns = list(dict.fromkeys(k for r in per_day for k in r))
        _save_rows(out / "within_day_replicate_spread.csv", per_day, day_columns)
    # Each within-day standard deviation uses only randomized replicates for
    # the same date, hours, spatial scheme and quantile N. This isolates
    # random selection variability from across-season variability.
    paired_columns, paired = _paired_comparison(rows)
    _save_rows(out / "spatial_grid_vs_random_daily.csv", paired, paired_columns)
    # The paired rows have different columns; summarise the actual signed contrasts.
    grouped = defaultdict(list)
    for r in paired:
        grouped[tuple(r[x] for x in ("hours_used", "spatial_step", "dimension", "variable", "N"))].append(r)
    paired_summary = []
    for key, recs in sorted(grouped.items()):
        item = dict(zip(("hours_used", "spatial_step", "dimension", "variable", "N"), key))
        item["pairs"] = len(recs)
        for metric in ("grid_minus_random_rms_mean", "grid_minus_random_rms_cov"):
            data = np.asarray([r[metric] for r in recs if r[metric] is not None], float)
            data = data[np.isfinite(data)]
            item[f"mean_{metric}"] = float(data.mean()) if len(data) else None
            item[f"p05_{metric}"] = float(np.percentile(data,5)) if len(data) else None
            item[f"p95_{metric}"] = float(np.percentile(data,95)) if len(data) else None
            item[f"pct_grid_lower_{metric}"] = float(100*np.mean(data < 0)) if len(data) else None
        paired_summary.append(item)
    if paired_summary:
        _save_rows(out/"spatial_grid_vs_random_summary.csv",paired_summary,list(paired_summary[0]))
    config = {
        "experiment": "nested random hourly subsamples and matched-size spatial thinning",
        "source_paths": [str(p) for p in paths],
        "first_date": start.isoformat(), "last_date": end.isoformat(),
        "requested_dates": len(dates), "processed_dates": len(processed),
        "skipped_incomplete": skipped,
        "seed": int(random_seed), "replicates": int(replicates),
        "hour_counts": list(hours), "spatial_steps": list(steps),
        "dimensions_bins": {"1D": list(bins1), "2D": list(bins2)},
        "matched_spatial_samples": "grid phase vs uniform random ocean pixels, equal count",
        "nested_hours": True,
        "time_shift_hours": 0, "transition_lag_hours": 24,
        "anchor": "mean of clock hours 00,06,12,18; same states in every comparison",
        "reference": "hourly_24 on all ocean pixels (NOT ground truth)",
        "valid_group_minimum": int(compare_min_group_size),
        "caution": "Finite-sample descriptive diagnostic; overlapping samples, "
                   "spatial-temporal dependence and diurnal structure remain.",
    }
    (out/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"output_dir": str(out), "processed_dates": len(processed),
            "skipped_dates": skipped, "daily": str(csv_path),
            "summary": str(out/"resampling_summary.csv")}
