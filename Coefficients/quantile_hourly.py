"""Hourly ERA5 quantile diagnostics for Ocean_components.

No manual time-label shifting. For GRIB, the source timestamp is its
valid_time (forecast reference time + step, when needed). A UTC day uses
the validity-date hours 00, 01, ..., 23. The daily_mean4 baseline is the mean of the FOUR
values labelled 00, 06, 12, 18 UTC (not four independent observations).

For hourly_L, each of L selected hours supplies a distinct state/increment
pair X[d,h] -> X[d+1,h], i.e. a 24-hour transition at fixed clock hour.
Hourly pooled and daily-averaged experiments therefore estimate different
conditional-moment functionals, which must not be silently conflated.

Inputs: one or more NetCDF (.nc/.nc4), GRIB (.grib/.grb/.grib2) and/or NumPy
(.npy, T,H,W) files, potentially with both variables in one dataset. For NPY,
provide an hourly_first_timestamp and a separate path for each variable.
NetCDF uses xarray; GRIB uses the optional cfgrib/eccodes engine. Data are
loaded one UTC day at a time, not for a complete decade.

Results: per-transition CSV, per-mode/N summary CSV, and config JSON.
This is a *descriptive* finite-sample diagnostic; spatial/temporal residual
correlations are not conditional-mixing proofs or estimates of effective n.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from contextlib import ExitStack
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np

from Coefficients.common import _bin_ids, load_water_mask
from Coefficients.quantile_diagnostics import _weighted_p10_group_size

VARIABLE_ALIASES = {
    "sensible": ("avg_ishf", "msshf", "mean_surface_sensible_heat_flux", "sensible"),
    "latent": ("avg_slhtf", "mslhf", "mean_surface_latent_heat_flux", "latent"),
}
MODE_HOURS = {
    "daily_mean4": (0, 6, 12, 18),
    "hourly_4": (0, 6, 12, 18),
    "hourly_8": tuple(range(0, 24, 3)),
    "hourly_12": tuple(range(0, 24, 2)),
    "hourly_24": tuple(range(24)),
}


def _as_hour(value):
    """Return numpy datetime64[h]; reject non-exact clock-hour timestamps."""
    try:
        dt = np.datetime64(value).astype("datetime64[s]")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported timestamp: {value!r}") from exc
    if np.isnat(dt):
        raise ValueError(f"Missing/NaT timestamp: {value!r}")
    if dt.astype("datetime64[h]").astype("datetime64[s]") != dt:
        raise ValueError(f"Non-hourly timestamp: {value!r}")
    return dt.astype("datetime64[h]")


def _timestamp_key(value) -> int:
    return int(_as_hour(value).astype(np.int64))


def _day_hour_key(day: date, hour: int) -> int:
    return _timestamp_key(f"{day.isoformat()}T{hour:02d}:00")


def _resolve_paths(root, paths):
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    resolved = []
    for item in paths:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = Path(root) / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing hourly ERA5 source: {path}")
        if path.suffix.lower() == ".zip":
            raise ValueError(f"Extract ZIP before running the diagnostic: {path}")
        resolved.append(path.resolve())
    if not resolved:
        raise ValueError("hourly_files must list at least one source file")
    return list(dict.fromkeys(resolved))


def _infer_stride(source_shape, target_shape, requested):
    if requested == "auto":
        if tuple(source_shape) == tuple(target_shape):
            return 1
        sy, sx = source_shape
        ty, tx = target_shape
        if ty < 2 or tx < 2:
            raise ValueError("Cannot infer stride for a spatial grid with dimension <2")
        ry = (sy - 1) / (ty - 1)
        rx = (sx - 1) / (tx - 1)
        if ry.is_integer() and rx.is_integer() and ry == rx and ry >= 1:
            step = int(ry)
        else:
            raise ValueError(f"Source grid {source_shape} is incompatible with water mask "
                             f"{target_shape}; specify hourly_spatial_stride explicitly")
    else:
        step = int(requested)
        if step < 1:
            raise ValueError("hourly_spatial_stride must be >= 1")
    if tuple(((source_shape[0]-1)//step+1, (source_shape[1]-1)//step+1)) != tuple(target_shape):
        raise ValueError(f"Stride {step} gives {(source_shape[0]-1)//step+1}x{(source_shape[1]-1)//step+1} "
                         f"pixels, expected {target_shape}")
    return step


class HourlySource:
    """Lazy, time-indexed source across optional multiple NetCDF/GRIB/NPY files."""

    def __init__(self, *, root, paths, spatial_shape, stride="auto",
                 npy_start=None, npy_variable_names=None):
        self._stack = ExitStack()
        self.datasets = []
        self.lookups = {var: {} for var in VARIABLE_ALIASES}
        self.source_files = []
        self.grid_shape = tuple(spatial_shape)
        self.source_shape = None
        self.spatial_stride = None
        self._coords = None
        npy_variable_names = npy_variable_names or {}
        try:
            for path in _resolve_paths(root, paths):
                self.source_files.append(str(path))
                suffix = path.suffix.lower()
                if suffix == ".npy":
                    if npy_start is None:
                        raise ValueError("hourly_npy_start is required for NPY hourly input")
                    var = npy_variable_names.get(path.name)
                    if var is None:
                        var = next((v for v, aliases in VARIABLE_ALIASES.items()
                                    if any(alias.lower() in path.stem.lower() for alias in aliases)), None)
                    if var not in VARIABLE_ALIASES:
                        raise ValueError(f"Cannot identify NPY variable for {path}; "
                                         "set hourly_npy_variable_names")
                    raw = np.load(path, mmap_mode="r", allow_pickle=False)
                    if raw.ndim != 3:
                        raise ValueError(f"NPY input must be (T,H,W), got {raw.shape}")
                    time_start = _timestamp_key(npy_start)
                    stamps = list(range(time_start, time_start + len(raw)))
                    self._register(var, raw, stamps, file_label=str(path), stride=stride)
                elif suffix in (".nc", ".nc4", ".netcdf", ".grib", ".grb", ".grib2"):
                    try:
                        import xarray as xr
                    except ImportError as exc:
                        raise ImportError("Install xarray for ERA5 NetCDF/GRIB diagnostics") from exc
                    engine = "cfgrib" if suffix in (".grib", ".grb", ".grib2") else None
                    try:
                        kwargs = {"engine": engine} if engine else {}
                        if engine:
                            kwargs["backend_kwargs"] = {"indexpath": ""}
                        ds = xr.open_dataset(path, **kwargs)
                    except Exception as exc:
                        raise RuntimeError(f"Cannot open {path}. GRIB requires cfgrib+eccodes; "
                                           "NetCDF requires an xarray-supported backend. "
                                           "For heterogeneous GRIB files, split fields by variable."
                                           ) from exc
                    self._stack.callback(ds.close)
                    found_any = False
                    for var, aliases in VARIABLE_ALIASES.items():
                        da = self._select_variable(ds, aliases)
                        if da is None:
                            continue
                        found_any = True
                        da, times, coords, selectors = self._normalize_array(da)
                        self._register(var, da, times, file_label=str(path),
                                       stride=stride, coords=coords,
                                       selectors=selectors)
                    if not found_any:
                        raise ValueError(f"ERA5 variable names not found in {path}: "
                                         f"{list(ds.data_vars)}. Expected {VARIABLE_ALIASES}")
                else:
                    raise ValueError(f"Unsupported hourly input extension {suffix} in {path}")
            for var, by_time in self.lookups.items():
                if not by_time:
                    raise ValueError(f"No hourly input records for {var}; files={self.source_files}")
        except BaseException:
            self.close()
            raise

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _select_variable(ds, aliases):
        low = {name.lower(): name for name in ds.data_vars}
        for alias in aliases:
            if alias.lower() in low:
                return ds[low[alias.lower()]]
        for da in ds.data_vars.values():
            short = str(da.attrs.get("GRIB_shortName", "")).lower()
            if short in {item.lower() for item in aliases}:
                return da
        return None

    @staticmethod
    def _normalize_array(da):
        """Index ERA5 analysis or forecast GRIB fields by their VALID UTC hour.

        Handles (time, lat, lon) and (time, step, lat, lon), including a
        non-singleton ``step``. A forecast ``time`` is its reference time,
        whereas ``valid_time`` labels the observation used in our experiment.
        No day/hour shift is imposed by this program.

        Keep GRIB data lazy: only coordinates are enumerated here; values
        are read in ``day`` for the requested day, not stacked into RAM.
        """
        time_dim = next((d for d in ("time", "datetime", "valid_time") if d in da.dims), None)
        if time_dim is None and "valid_time" in da.coords:
            possible = [d for d in da.coords["valid_time"].dims if d != "step"]
            time_dim = possible[0] if len(possible) == 1 else None
        if time_dim is None:
            raise ValueError(f"Cannot identify time dimension in {da.dims}")

        # A real GRIB step may have 2, 12, 24, ... values; never select
        # step=0 or step=1 without an explicit physical reason.
        step_dim = "step" if "step" in da.dims else None
        candidates = {time_dim, "latitude", "longitude", "lat", "lon",
                      "y", "x", "grid_latitude", "grid_longitude"}
        if step_dim is not None:
            candidates.add(step_dim)
        for d in tuple(da.dims):
            if d not in candidates and da.sizes[d] == 1:
                da = da.isel({d: 0}, drop=True)
        other = [d for d in da.dims if d not in (time_dim, step_dim)]
        if len(other) != 2:
            raise ValueError(
                f"Cannot identify the spatial grid in {da.dims}; "
                "only singleton non-spatial extra dimensions can be removed automatically"
            )
        lat_names = {"latitude", "lat", "y", "grid_latitude"}
        lon_names = {"longitude", "lon", "x", "grid_longitude"}
        latdim = next((d for d in other if d.lower() in lat_names), other[0])
        londim = next((d for d in other if d.lower() in lon_names and d != latdim), None)
        if londim is None:
            londim = next(d for d in other if d != latdim)
        coords = tuple(np.asarray(da.coords[d].values) if d in da.coords else None
                       for d in (latdim, londim))

        if step_dim is None:
            da = da.transpose(time_dim, latdim, londim)
            valid = da.coords.get("valid_time")
            if valid is None or valid.dims != (time_dim,):
                valid = da.coords.get(time_dim)
            if valid is None or valid.dims != (time_dim,):
                raise ValueError("Missing one-dimensional UTC time/valid_time coordinate")
            timestamps = [_timestamp_key(t) for t in valid.values]
            return da, timestamps, coords, None

        da = da.transpose(time_dim, step_dim, latdim, londim)
        nt, ns = da.sizes[time_dim], da.sizes[step_dim]
        valid = da.coords.get("valid_time")
        if valid is not None and set(valid.dims) == {time_dim, step_dim}:
            time_values = np.asarray(valid.transpose(time_dim, step_dim).values)
        elif valid is not None and valid.dims == (time_dim,) and ns == 1:
            time_values = np.asarray(valid.values)[:, None]
        else:
            # cfgrib should supply valid_time. If not, the CF definition is
            # reference time + forecast lead time (not an arbitrary shift).
            reference = da.coords.get(time_dim)
            lead = da.coords.get(step_dim)
            if reference is None or lead is None:
                raise ValueError("GRIB has a step dimension but no valid_time "
                                 "or time+step coordinates")
            times = np.asarray(reference.values)
            steps = np.asarray(lead.values)
            if not np.issubdtype(times.dtype, np.datetime64):
                raise ValueError("GRIB time coordinate must contain datetimes")
            if not np.issubdtype(steps.dtype, np.timedelta64):
                raise ValueError("GRIB step has no decoded timedelta units; "
                                 "provide a valid_time coordinate")
            time_values = times[:, None] + steps[None, :]
        if time_values.shape != (nt, ns):
            raise ValueError(f"valid_time shape {time_values.shape} is not {(nt, ns)}")
        timestamps = [_timestamp_key(t) for t in time_values.reshape(-1)]
        selectors = [(i, j) for i in range(nt) for j in range(ns)]
        return da, timestamps, coords, selectors

    def _register(self, var, field, timestamps, *, file_label, stride,
                  coords=None, selectors=None):
        shape = tuple(field.shape[-2:])
        step = _infer_stride(shape, self.grid_shape, stride)
        if self.source_shape is None:
            self.source_shape, self.spatial_stride = shape, step
            self._coords = coords
        else:
            if self.source_shape != shape or self.spatial_stride != step:
                raise ValueError(f"Grid mismatch in {file_label}: {shape}")
            if coords is not None and self._coords is not None:
                for old, new in zip(self._coords, coords):
                    if old is not None and new is not None and not np.allclose(old, new, atol=1e-6):
                        raise ValueError("Incompatible latitude/longitude coordinates across files")
        if selectors is None:
            if field.ndim != 3:
                raise ValueError("Three-dimensional (time,lat,lon) field expected")
            selectors = list(range(field.shape[0]))
        elif field.ndim != 4:
            raise ValueError("Four-dimensional (time,step,lat,lon) field expected")
        if len(timestamps) != len(selectors):
            raise ValueError("Timestamp count differs from the number of time/step records")
        entry_id = len(self.datasets)
        self.datasets.append((field, step))
        for idx, stamp in zip(selectors, timestamps):
            if stamp in self.lookups[var]:
                raise ValueError(
                    f"Duplicate valid UTC timestamp for {var} in {file_label}: "
                    f"{np.datetime64(stamp, 'h')}. Multiple forecast (time,step) "
                    "records refer to the same clock hour; select a forecast cycle "
                    "explicitly instead of averaging overlapping forecasts."
                )
            self.lookups[var][stamp] = (entry_id, idx)

    def has_complete_day(self, day):
        return all(_day_hour_key(day, h) in self.lookups[var]
                   for var in VARIABLE_ALIASES for h in range(24))

    def day(self, day):
        """Read one UTC day of each variable, selecting by *valid_time*.

        On four-dimensional GRIB data, each requested clock hour maps to an
        explicit (reference_time_index, forecast_step_index) pair. We access
        step slices lazily instead of materialising the time x step cube.
        """
        result = {}
        for var in VARIABLE_ALIASES:
            entries = [self.lookups[var][_day_hour_key(day, h)] for h in range(24)]
            out = np.empty((24, *self.grid_shape), dtype=np.float64)
            selections = defaultdict(list)
            for hour, (source_idx, selector) in enumerate(entries):
                if isinstance(selector, tuple):
                    time_idx, step_idx = selector
                    selections[(source_idx, step_idx)].append((hour, time_idx))
                else:
                    selections[(source_idx, None)].append((hour, selector))
            for (source_idx, step_idx), hour_indices in selections.items():
                field, stride = self.datasets[source_idx]
                hours = [pair[0] for pair in hour_indices]
                time_indices = [pair[1] for pair in hour_indices]
                if hasattr(field, "isel"):
                    if step_idx is None:
                        selection = {
                            field.dims[0]: time_indices,
                            field.dims[1]: slice(None, None, stride),
                            field.dims[2]: slice(None, None, stride),
                        }
                    else:
                        selection = {
                            field.dims[0]: time_indices,
                            field.dims[1]: step_idx,
                            field.dims[2]: slice(None, None, stride),
                            field.dims[3]: slice(None, None, stride),
                        }
                    block = np.asarray(field.isel(selection).values, dtype=np.float64)
                else:
                    block = np.asarray(field[time_indices, ::stride, ::stride],
                                       dtype=np.float64)
                if block.shape != (len(hours), *self.grid_shape):
                    raise AssertionError(
                        f"Invalid sampled {var} field shape {block.shape}, "
                        f"expected {(len(hours), *self.grid_shape)}"
                    )
                out[hours] = block
            result[var] = out
        return result


def _group_for_mode(prev, curr, *, dim, n_bins, water, return_fit=False):
    """Exact bin assignment of common._bin_ids, with vectorized group moments."""
    if dim == 1:
        if prev.shape != curr.shape or prev.ndim != 3:
            raise ValueError("Expected scalar (hours,H,W) fields")
        is_valid = np.isfinite(prev) & np.isfinite(curr) & np.isfinite(curr - prev)
        is_valid &= water[None, :, :]
        state = prev[is_valid].reshape(-1, 1)
        increments = (curr - prev)[is_valid].reshape(-1, 1)
    elif dim == 2:
        if prev.shape != curr.shape or prev.ndim != 4 or prev.shape[-1] != 2:
            raise ValueError("Expected bivariate (hours,H,W,2) fields")
        is_valid = np.all(np.isfinite(prev) & np.isfinite(curr), axis=-1)
        increments_full = curr - prev
        is_valid &= np.all(np.isfinite(increments_full), axis=-1)
        is_valid &= water[None, :, :]
        state = prev[is_valid]
        increments = increments_full[is_valid]
    else:
        raise ValueError("dim must be 1 or 2")
    n = int(np.count_nonzero(is_valid))
    bins_total = n_bins ** dim
    sizes = np.zeros(bins_total, dtype=np.int64)
    edges = []
    assignments = np.empty(n, dtype=np.int64)
    if n:
        for k in range(dim):
            ids, e = _bin_ids(state[:, k], n_bins)
            edges.append(np.asarray(e, dtype=float))
            assignments[:] = ids if k == 0 else assignments * n_bins + ids
        sizes = np.bincount(assignments, minlength=bins_total)
    positive = sizes[sizes > 0]
    row = {
        "valid": n, "occupied": int(len(positive)),
        "empty": int(bins_total - len(positive)),
        "singletons": int(np.count_nonzero(sizes == 1)),
        "min_occupied": int(positive.min()) if len(positive) else 0,
        "median_occupied": float(np.median(positive)) if len(positive) else 0.0,
        "p10_size_typical_observation": _weighted_p10_group_size(sizes),
        "max_occupied": int(positive.max()) if len(positive) else 0,
    }
    for k in range(dim):
        widths = np.diff(edges[k]) if n else np.empty(0)
        row[f"zero_width_axis{k}"] = int(np.count_nonzero(widths == 0))
        row[f"max_width_axis{k}"] = float(widths.max()) if len(widths) else np.nan
    mean = np.zeros((bins_total, dim), dtype=float)
    residual = np.full((*is_valid.shape, dim), np.nan, dtype=float)
    groupids = np.full(is_valid.shape, -1, dtype=np.int64)
    cov = np.zeros((bins_total, dim, dim), dtype=float)
    if n:
        for k in range(dim):
            means = np.bincount(assignments, weights=increments[:, k], minlength=bins_total)
            np.divide(means, sizes, out=mean[:, k], where=sizes > 0)
        error = increments - mean[assignments]
        residual[is_valid] = error
        groupids[is_valid] = assignments
        if return_fit:
            for k in range(dim):
                for j in range(k, dim):
                    numer = np.bincount(assignments, weights=error[:, k] * error[:, j],
                                        minlength=bins_total)
                    np.divide(numer, sizes, out=cov[:, k, j], where=sizes > 0)
                    cov[:, j, k] = cov[:, k, j]
    if return_fit:
        return row, sizes, residual, groupids, {
            "edges": edges, "mean": mean, "cov": cov, "sizes": sizes, "dim": dim,
            "N": n_bins,
        }
    return row, sizes, residual, groupids


def _same_group_corr(residual, groups, *, axis, lag, component):
    """Pearson correlation for residuals whose two points share one quantile bin."""
    if lag < 1 or lag >= residual.shape[axis]:
        return (np.nan, 0)
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    lo[axis], hi[axis] = slice(None, -lag), slice(lag, None)
    x = residual[(*lo, component)]
    y = residual[(*hi, component)]
    bx = groups[tuple(lo)]
    by = groups[tuple(hi)]
    ok = np.isfinite(x) & np.isfinite(y) & (bx >= 0) & (bx == by)
    count = int(ok.sum())
    if count < 30:
        return (np.nan, count)
    a, b = x[ok], y[ok]
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return (float(np.dot(a, b) / denom) if denom > 0 else np.nan, count)


def _daily_average(values, hours=(0, 6, 12, 18)):
    """Arithmetic mean, exact labels, complete cases only, no date correction."""
    sub = values[list(hours)]
    valid = np.all(np.isfinite(sub), axis=0)
    out = np.full(values.shape[1:], np.nan)
    out[valid] = sub[:, valid].mean(axis=0)
    return out


def _stability_at_fixed_states(model, reference, states, *, min_size):
    """Compare conditional moments at identical state points, reference=hourly_24.

    The reference is NOT ground truth and queries are not a held-out sample.
    No model is fitted or interpolated beyond the existing quantile-bin rule.
    """
    valid = np.all(np.isfinite(states), axis=1)
    z = states[valid]
    if not len(z):
        return {"anchor_points": 0, "matched_points": 0,
                "matched_pct": 0.0, "rms_mean_vs_hourly24": np.nan,
                "rms_cov_vs_hourly24": np.nan}
    out = []
    for fit in (model, reference):
        if not fit["edges"]:
            return {"anchor_points": len(z), "matched_points": 0,
                    "matched_pct": 0.0, "rms_mean_vs_hourly24": np.nan,
                    "rms_cov_vs_hourly24": np.nan}
        index = np.zeros(len(z), dtype=np.int64)
        for k in range(fit["dim"]):
            edges = fit["edges"][k]
            digit = np.clip(np.searchsorted(edges, z[:, k], side="right") - 1,
                            0, fit["N"] - 1)
            index = index * fit["N"] + digit
        out.append((fit, index))
    a, ia = out[0]
    b, ib = out[1]
    selected = (a["sizes"][ia] >= min_size) & (b["sizes"][ib] >= min_size)
    count = int(selected.sum())
    if not count:
        return {"anchor_points": len(z), "matched_points": 0,
                "matched_pct": 0.0, "rms_mean_vs_hourly24": np.nan,
                "rms_cov_vs_hourly24": np.nan}
    diff_mean = a["mean"][ia[selected]] - b["mean"][ib[selected]]
    diff_cov = a["cov"][ia[selected]] - b["cov"][ib[selected]]
    return {
        "anchor_points": int(len(z)), "matched_points": count,
        "matched_pct": round(100 * count / len(z), 5),
        "rms_mean_vs_hourly24": float(np.sqrt(np.mean(np.sum(diff_mean ** 2, axis=1)))),
        "rms_cov_vs_hourly24": float(np.sqrt(np.mean(np.sum(diff_cov ** 2, axis=(1, 2))))),
    }


def _pct(data, threshold):
    count = data.sum()
    return round(100.0 * data[data >= threshold].sum() / count, 5) if count else 0.0


class _SummaryAccumulator:
    """Streaming group summaries; retain percentiles only for coverage metrics."""

    def __init__(self, key_fields):
        self.key_fields = key_fields
        self.group = defaultdict(lambda: defaultdict(lambda: [0.0, 0, float("inf"), []]))
        self.days = defaultdict(int)

    def add(self, row):
        key = tuple(row[field] for field in self.key_fields)
        self.days[key] += 1
        for field, value in row.items():
            if field in ("date", "day_index") or field in self.key_fields:
                continue
            if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                item = self.group[key][field]
                item[0] += float(value)
                item[1] += 1
                item[2] = min(item[2], float(value))
                if field.startswith("coverage_ge_"):
                    item[3].append(float(value))

    def write(self, path):
        result = []
        for key, metrics in sorted(self.group.items()):
            rec = dict(zip(self.key_fields, key))
            rec["days"] = self.days[key]
            for field, (total, count, minimum, samples) in metrics.items():
                if count:
                    rec[f"mean_{field}"] = round(total / count, 6)
                    if samples:
                        rec[f"min_{field}"] = round(minimum, 5)
                        rec[f"p05_{field}"] = round(float(np.percentile(samples, 5)), 5)
            result.append(rec)
        if result:
            columns = sorted(set().union(*(r.keys() for r in result)))
            with Path(path).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(result)
        return result


def run_hourly_quantile_diagnostics(
    files_path_prefix, *,
    hourly_files,
    mask,
    start_year=1979,
    end_year=1989,
    first_date=None,
    last_date=None,
    day_stride=1,
    max_days=None,
    modes=("daily_mean4", "hourly_4", "hourly_8", "hourly_12", "hourly_24"),
    bins_1d=(20, 50, 100, 144, 200, 250),
    bins_2d=(8, 12, 15, 20),
    thresholds=(2, 10, 20, 30, 50),
    corr_every=30,
    corr_bin_1d=144,
    corr_bin_2d=15,
    spatial_lags=(1, 4, 8),
    temporal_lags=(1, 2, 3, 6),
    hourly_spatial_stride="auto",
    hourly_npy_start=None,
    hourly_npy_variable_names=None,
    output_dir=None,
    daily_reference=False,
    compare_min_group_size=30,
    progress_every=30,
):
    """One-task 1D sensible, 1D latent and 2D diagnostic, daily vs hourly.

    End year is EXCLUSIVE, like Ocean_components periods. Per-day observation
    windows require all 24 clock hours on both dates; missing pairs are skipped
    and reported. Sampling is deliberately restricted to available complete
    UTC dates and never shifts timestamps.

    ``hourly_files`` accepts one or several GRIB, NetCDF or NPY sources. NPY
    uses separate files per variable and requires ``hourly_npy_start``.
    """
    modes = tuple(dict.fromkeys(modes))
    if not modes or any(m not in MODE_HOURS for m in modes):
        raise ValueError(f"hourly_modes must be subset of {tuple(MODE_HOURS)}")
    bins_1d = tuple(dict.fromkeys(int(x) for x in bins_1d))
    bins_2d = tuple(dict.fromkeys(int(x) for x in bins_2d))
    thresholds = tuple(sorted(set(int(x) for x in thresholds)))
    spatial_lags = tuple(sorted(set(int(x) for x in spatial_lags)))
    temporal_lags = tuple(sorted(set(int(x) for x in temporal_lags)))
    if not bins_1d or not bins_2d or any(n < 1 for n in bins_1d + bins_2d):
        raise ValueError("All bin counts must be positive")
    if compare_min_group_size < 2:
        raise ValueError("compare_min_group_size must be >= 2")
    if not thresholds or any(n < 1 for n in thresholds):
        raise ValueError("Thresholds must contain positive integers")
    if any(n < 1 for n in spatial_lags + temporal_lags):
        raise ValueError("Spatial and temporal lags must be positive")
    if day_stride < 1 or (max_days is not None and max_days < 1):
        raise ValueError("day_stride >=1, max_days must be None or >=1")
    if corr_every < 0 or progress_every < 1:
        raise ValueError("corr_every >=0 and progress_every >=1")
    if corr_every and (corr_bin_1d not in bins_1d or corr_bin_2d not in bins_2d):
        raise ValueError("Correlation bin settings must occur in the corresponding bin lists")
    if int(start_year) >= int(end_year):
        raise ValueError("start_year must be less than exclusive end_year")
    root = Path(files_path_prefix)
    output = (Path(output_dir) if output_dir is not None else root / "Functional" / "Diagnostics" /
              "quantile_hourly" / f"{start_year}-{end_year}")
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    shape = (161, 181) if mask is None else np.asarray(mask).shape
    if len(shape) != 2:
        raise ValueError("Provide a 2D water mask for hourly ERA5 fields")
    water = load_water_mask(mask, shape)
    if not water.any():
        raise ValueError("Empty water mask")
    date_from = date.fromisoformat(first_date) if isinstance(first_date, str) else (
        first_date if first_date is not None else date(int(start_year), 1, 1))
    date_stop = date.fromisoformat(last_date) if isinstance(last_date, str) else (
        last_date if last_date is not None else date(int(end_year), 1, 1) - timedelta(days=1))
    if date_stop < date_from:
        raise ValueError("Empty date interval")
    filename = output / "daily_summary.csv"
    skipped = []
    stability_written = 0
    stability_accumulator = _SummaryAccumulator(("mode", "dimension", "variable", "N"))
    summary_accumulator = _SummaryAccumulator(("mode", "dimension", "variable", "N"))
    result_rows = 0
    corr_fields = []
    for lag in spatial_lags:
        for direction in ("row", "col"):
            for var in ("component0", "component1"):
                corr_fields.extend((f"corr_space_{direction}{lag}_{var}",
                                    f"pairs_space_{direction}{lag}_{var}"))
    for lag in temporal_lags:
        for var in ("component0", "component1"):
            corr_fields.extend((f"corr_time_h{lag}_{var}",
                                f"pairs_time_h{lag}_{var}"))
    columns = [
        "date", "day_index", "mode", "hours_used", "dimension", "variable", "N",
        "valid", "valid_spatial_per_hour_pct", "occupied", "empty", "singletons",
        "min_occupied", "median_occupied", "p10_size_typical_observation",
        "max_occupied", "zero_width_axis0", "max_width_axis0", "zero_width_axis1",
        "max_width_axis1",
    ] + [f"coverage_ge_{x}_pct" for x in thresholds] + corr_fields
    periods = [(date_from + timedelta(days=i))
               for i in range(0, (date_stop - date_from).days + 1, day_stride)]
    if max_days is not None:
        periods = periods[:int(max_days)]
    with HourlySource(root=root, paths=hourly_files, spatial_shape=shape,
                      stride=hourly_spatial_stride, npy_start=hourly_npy_start,
                      npy_variable_names=hourly_npy_variable_names) as source:
        previous_ref = None
        if daily_reference:
            from Data_processing.io import load_field
            try:
                previous_ref = {var: load_field(root, var, start_year, end_year)
                                for var in VARIABLE_ALIASES}
            except FileNotFoundError as exc:
                raise FileNotFoundError("daily_reference=True but existing daily series missing") from exc
        compare_rows = []
        stability_columns = ["date", "mode", "dimension", "variable", "N",
                             "anchor_points", "matched_points", "matched_pct",
                             "rms_mean_vs_hourly24", "rms_cov_vs_hourly24"]
        with (filename.open("w", newline="", encoding="utf-8") as handle,
              (output / "moment_stability_daily.csv").open(
                  "w", newline="", encoding="utf-8") as stability_handle):
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            stability_writer = csv.DictWriter(stability_handle,
                                               fieldnames=stability_columns)
            stability_writer.writeheader()
            for count, day in enumerate(periods, start=1):
                after = day + timedelta(days=1)
                if not source.has_complete_day(day) or not source.has_complete_day(after):
                    skipped.append(day.isoformat())
                    continue
                before = source.day(day)
                following = source.day(after)
                if previous_ref is not None:
                    offset = (day - date(int(start_year), 1, 1)).days
                    if offset < len(previous_ref["sensible"]):
                        rec = {"date": day.isoformat()}
                        for var in VARIABLE_ALIASES:
                            reconstructed = _daily_average(before[var])
                            historic = np.asarray(previous_ref[var][offset], dtype=float)
                            finite = water & np.isfinite(reconstructed) & np.isfinite(historic)
                            diff = reconstructed[finite] - historic[finite]
                            rec[f"{var}_MAE_vs_historic_daily"] = (
                                float(np.mean(np.abs(diff))) if len(diff) else np.nan)
                            rec[f"{var}_max_abs_vs_historic_daily"] = (
                                float(np.max(np.abs(diff))) if len(diff) else np.nan)
                        compare_rows.append(rec)
                anchor = {var: _daily_average(before[var]) for var in VARIABLE_ALIASES}
                fits_for_day = {}
                for mode in modes:
                    hours = MODE_HOURS[mode]
                    if mode == "daily_mean4":
                        p = {var: anchor[var][None] for var in VARIABLE_ALIASES}
                        q = {var: _daily_average(following[var])[None] for var in VARIABLE_ALIASES}
                    else:
                        p = {var: before[var][list(hours)] for var in VARIABLE_ALIASES}
                        q = {var: following[var][list(hours)] for var in VARIABLE_ALIASES}
                    for dim, variable in ((1, "sensible"), (1, "latent"), (2, "sensible_latent")):
                        if dim == 1:
                            prev, curr = p[variable], q[variable]
                            ns = bins_1d
                        else:
                            prev = np.stack((p["sensible"], p["latent"]), axis=-1)
                            curr = np.stack((q["sensible"], q["latent"]), axis=-1)
                            ns = bins_2d
                        for n in ns:
                            base, sizes, residual, gids, fitted = _group_for_mode(
                                prev, curr, dim=dim, n_bins=n, water=water,
                                return_fit=True)
                            fits_for_day[(mode, dim, variable, n)] = fitted
                            line = {"date": day.isoformat(), "day_index": count,
                                    "mode": mode, "hours_used": len(hours) if mode != "daily_mean4" else 1,
                                    "dimension": dim, "variable": variable, "N": n,
                                    **base,
                                    "valid_spatial_per_hour_pct": round(
                                        100*base["valid"]/(water.sum()*(len(hours) if mode!="daily_mean4" else 1)), 5),
                                    }
                            for th in thresholds:
                                line[f"coverage_ge_{th}_pct"] = _pct(sizes, th)
                            if corr_every and (count-1) % corr_every == 0 and n == (
                                corr_bin_1d if dim == 1 else corr_bin_2d):
                                for component in range(dim):
                                    for lag in spatial_lags:
                                        for direction, axis in (("row", 1), ("col", 2)):
                                            corr, pairs = _same_group_corr(
                                                residual, gids, axis=axis, lag=lag,
                                                component=component)
                                            line[f"corr_space_{direction}{lag}_component{component}"] = corr
                                            line[f"pairs_space_{direction}{lag}_component{component}"] = pairs
                                    if mode != "daily_mean4":
                                        step_hours = 24 // len(hours)
                                        for lag_hours in temporal_lags:
                                            if lag_hours % step_hours:
                                                continue  # Hour separation is absent in this mode.
                                            corr, pairs = _same_group_corr(
                                                residual, gids, axis=0,
                                                lag=lag_hours // step_hours,
                                                component=component)
                                            line[f"corr_time_h{lag_hours}_component{component}"] = corr
                                            line[f"pairs_time_h{lag_hours}_component{component}"] = pairs
                            writer.writerow(line)
                            summary_accumulator.add(line)
                            result_rows += 1
                if "hourly_24" in modes:
                    for (mode, dim, variable, n), fit in fits_for_day.items():
                        if mode == "hourly_24":
                            continue
                        reference = fits_for_day[("hourly_24", dim, variable, n)]
                        if dim == 1:
                            states = anchor[variable][water].reshape(-1, 1)
                        else:
                            states = np.stack((anchor["sensible"][water],
                                               anchor["latent"][water]), axis=-1)
                        item = {
                            "date": day.isoformat(), "mode": mode,
                            "dimension": dim, "variable": variable, "N": n,
                            **_stability_at_fixed_states(
                                fit, reference, states, min_size=compare_min_group_size)
                        }
                        stability_writer.writerow(item)
                        stability_written += 1
                        stability_accumulator.add(item)
                if count == 1 or count % progress_every == 0 or count == len(periods):
                    print(f"[quantile_hourly] {count}/{len(periods)} days, "
                          f"day={day}, modes={len(modes)}, rows={result_rows}", flush=True)
                    handle.flush()
        result = summary_accumulator.write(output / "by_mode_summary.csv")
        if stability_written:
            stability_accumulator.write(output / "moment_stability_summary.csv")
        if compare_rows:
            with (output / "daily_reference_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(compare_rows[0]))
                writer.writeheader()
                writer.writerows(compare_rows)
        config = {
            "start_year": start_year, "end_year_exclusive": end_year,
            "first_date": date_from.isoformat(), "last_date": date_stop.isoformat(),
            "requested_days": len(periods), "processed_days": len(periods)-len(skipped),
            "skipped_incomplete_days": skipped,
            "hourly_source_files": source.source_files,
            "original_grid_shape": source.source_shape,
            "selected_grid_shape": shape,
            "spatial_stride": source.spatial_stride,
            "modes_hours": {m: list(MODE_HOURS[m]) for m in modes},
            "daily_mean4_divisor": 4,
            "timestamp_shift_hours": 0,
            "grib_time_coordinate": "valid_time (or reference time + decoded step if absent)",
            "grib_multi_step_handling": "all steps indexed by validity hour; duplicate validity rejected",
            "transition_lag_hours": 24,
            "bins_1d": list(bins_1d), "bins_2d": list(bins_2d),
            "thresholds": list(thresholds), "corr_every": corr_every,
            "spatial_lags_in_pixels": list(spatial_lags),
            "temporal_lags_in_clock_hours": list(temporal_lags),
            "daily_reference": bool(daily_reference),
            "moment_stability_reference": "hourly_24 (not true conditional moments)",
            "moment_stability_query_states": "four-hour daily mean state at same grid pixels",
            "moment_stability_min_group_size": compare_min_group_size,
            "caution": "Raw hourly pooling and 4-snapshot daily mean target different conditional laws. "
                       "These correlations are descriptive, not an effective-sample-size proof.",
        }
        (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        return {"output_dir": str(output), "processed_days": config["processed_days"],
                "skipped_days": len(skipped), "by_mode_summary": result,
                "daily_summary": str(filename)}
