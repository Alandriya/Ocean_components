"""Reproducible numerical data behind article figures 1--3.

Revives pre-refactor ``get_isolines`` / ``plot_areas`` / ``plot_areas_map``
without their obsolete videos paths. No model is refitted here. The reference
PDF is either the classical stationary FP solution or the cached, explicitly
labelled semiparametric hybrid PDF. All spatial statistics use ERA5 observations.

Scientific note: the full-period stationary-model coordinate and the raw ERA5
flux coordinate need not coincide. A spatial deviation from a model mean is a
*diagnostic* comparison, not a theorem about the original SDE's stationary law.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

from Data_processing.io import field_candidates, load_field, read_ocean_mask

DEFAULT_PAIR = ("sensible", "latent")
DEFAULT_REGIONS_PERIOD = (1979, 1989)  # Figure 2 in the manuscript
DEFAULT_REFERENCE_PERIOD = (1979, 2026)  # Figure 1 in the manuscript
DEFAULT_DATE = "1979-01-01"  # Figure 3 in the manuscript
DEFAULT_HYBRID_REFERENCE = (1979, 2026)
DEFAULT_LON_RANGE = (-90.0, 0.0)
DEFAULT_LAT_RANGE = (80.0, 0.0)  # array row 0 corresponds to northern boundary


def _signature(path):
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _source(root, name, period):
    candidates = field_candidates(root, name, *period)
    file = next((path for path in candidates if path.is_file()), None)
    if file is None:
        raise FileNotFoundError(f"ERA5 source for {name} {period} not found; checked: {candidates}")
    return file


def stationary_density(root, variable, period, *, names=DEFAULT_PAIR,
                       density_source="auto", hybrid_reference=DEFAULT_HYBRID_REFERENCE):
    """Read the *existing* density; never recompute extrapolated tails or GPD.

    density_source='auto': prefer a named cached hybrid; else classical FP.
    density_source='hybrid'/'fp': require the requested type explicitly.
    """
    root = Path(root)
    first, last = map(int, period)
    if density_source not in ("auto", "hybrid", "fp"):
        raise ValueError("density_source must be 'auto', 'hybrid' or 'fp'")
    base = root / "Functional" / "Stationary"
    hybrid = (base / "Hybrid" /
              f"{variable}_{first}-{last}_gpd_reference_"
              f"{hybrid_reference[0]}-{hybrid_reference[1]}.npz")
    fp = base / f"{names[0]}-{names[1]}_{first}-{last}.npz"
    if density_source == "hybrid":
        if not hybrid.is_file():
            raise FileNotFoundError(f"Hybrid PDF not available: {hybrid}")
        path, kind, key = hybrid, "hybrid", "pdf"
    elif density_source == "fp":
        if not fp.is_file():
            raise FileNotFoundError(f"Stationary FP PDF not available: {fp}")
        path, kind, key = fp, "fp", f"pdf_{variable}"
    elif hybrid.is_file():
        path, kind, key = hybrid, "hybrid", "pdf"
    elif fp.is_file():
        path, kind, key = fp, "fp", f"pdf_{variable}"
    else:
        raise FileNotFoundError(f"Neither hybrid ({hybrid}) nor FP ({fp}) PDF exists")
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"Missing {key} in {path}: available={data.files}")
        grid = np.asarray(data["grid"], dtype=np.float64)
        pdf = np.asarray(data[key], dtype=np.float64)
    if grid.ndim != 1 or pdf.shape != grid.shape or len(grid) < 4:
        raise ValueError(f"Invalid stationary grid/PDF in {path}")
    if not np.all(np.diff(grid) > 0) or not np.isfinite(grid).all():
        raise ValueError(f"Stationary coordinate grid is not increasing: {path}")
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    if np.min(pdf) < -1e-12:
        raise ValueError(f"Negative stationary PDF values in {path}")
    pdf = np.maximum(pdf, 0.0)
    area = float(np.trapezoid(pdf, grid))
    if not 0 < area < np.inf:
        raise ValueError(f"Non-normalizable PDF in {path}")
    pdf /= area
    mean = float(np.trapezoid(grid * pdf, grid))
    std = float(np.sqrt(max(0.0, np.trapezoid((grid - mean)**2 * pdf, grid))))
    return {"grid": grid, "pdf": pdf, "mean": mean, "std": std,
            "kind": kind, "path": path, "signature": _signature(path)}


def equal_probability_regions(grid, pdf, n_regions=5):
    """Interpolate quantiles of the normalized continuous density (not bins)."""
    grid = np.asarray(grid, dtype=np.float64)
    pdf = np.asarray(pdf, dtype=np.float64)
    if grid.ndim != 1 or grid.shape != pdf.shape or n_regions < 2:
        raise ValueError("Invalid grid/PDF or number of regions")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("Grid must be strictly increasing")
    cdf = np.r_[0., cumulative_trapezoid(np.maximum(pdf, 0), grid)]
    if cdf[-1] <= 0 or not np.isfinite(cdf[-1]):
        raise ValueError("PDF has zero or non-finite mass")
    cdf /= cdf[-1]
    internal = np.interp(np.arange(1, n_regions) / n_regions, cdf, grid)
    if np.any(np.diff(internal) <= 0):
        raise ValueError("Quantile boundaries are not strictly increasing")
    masses = np.diff(np.interp(np.r_[grid[0], internal, grid[-1]], grid, cdf))
    return internal, masses


def classify_map(values, thresholds, water_mask):
    """Classes 1...5, land/non-finite = 0; q_{k-1}<x<=q_k."""
    values = np.asarray(values, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    water_mask = np.asarray(water_mask, dtype=bool)
    if thresholds.ndim != 1 or np.any(np.diff(thresholds) <= 0):
        raise ValueError("Thresholds must be strictly increasing")
    if values.shape != water_mask.shape:
        raise ValueError(f"Water mask {water_mask.shape} differs from field {values.shape}")
    result = np.zeros(values.shape, dtype=np.uint8)
    good = water_mask & np.isfinite(values)
    result[good] = (1 + np.searchsorted(thresholds, values[good], side="left")).astype(np.uint8)
    return result


def _spatial_statistics(field, ocean, *, chunk_days, thresholds=None,
                        compute_frequencies=False):
    """Compute true per-pixel temporal mean using bounded time slices."""
    if chunk_days < 1:
        raise ValueError("chunk_days must be positive")
    shape = field.shape[-2:]
    if ocean.shape != shape:
        raise ValueError(f"Ocean mask {ocean.shape} != ERA5 grid {shape}")
    sums = np.zeros(shape, np.float64)
    counts = np.zeros(shape, np.int64)
    hit = np.zeros((5, *shape), np.int64) if compute_frequencies else None
    for left in range(0, len(field), chunk_days):
        right = min(len(field), left + chunk_days)
        batch = np.asarray(field[left:right], dtype=np.float64)
        valid = np.isfinite(batch) & ocean[None, ...]
        sums += np.where(valid, batch, 0).sum(axis=0)
        counts += np.sum(valid, axis=0)
        if hit is not None:
            cat = np.searchsorted(thresholds, batch, side="left")
            for k in range(5):
                hit[k] += ((cat == k) & valid).sum(axis=0)
        if left == 0 or right == len(field) or right // chunk_days % 60 == 0:
            print(f"[article] ERA5 spatial statistics {right}/{len(field)} days", flush=True)
    mean = np.full(shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=mean, where=counts > 0)
    if hit is None:
        return mean, counts, None
    frequencies = np.full((5, *shape), np.nan, dtype=np.float64)
    np.divide(hit, counts[None], out=frequencies, where=counts[None] > 0)
    return mean, counts, frequencies


def _cache_file(root, name, period, region_period, date_str, pdf_kind, frequencies):
    root = Path(root)
    fragment = "_freq" if frequencies else ""
    return (root / "Functional" / "Article" / f"{period[0]}-{period[1]}" /
            f"{name}_regions_{region_period[0]}-{region_period[1]}_"
            f"{date_str}_{pdf_kind}{fragment}.npz")


def prepare_article_diagnostics(files_path_prefix, names=DEFAULT_PAIR,
                                start_year=1979, end_year=2026, *,
                                regions_period=DEFAULT_REGIONS_PERIOD,
                                map_date=DEFAULT_DATE, density_source="auto",
                                hybrid_reference=DEFAULT_HYBRID_REFERENCE,
                                chunk_days=32, compute_frequencies=False,
                                overwrite=False):
    """Make cached arrays for the manuscript's geographical Figures 1--3.

    Fig 1: Delta X = ERA5 time mean over [start_year,end_year] minus its model
    reference mean over the *same* period. Fig 2 uses the reference PDF and
    quintiles from regions_period. Fig 3 classifies the specified daily ERA5
    map (1979-01-01 by default), using those exact Fig-2 quintiles.

    A separate file is produced per variable. Loaded arrays remain read-only.
    """
    root = Path(files_path_prefix)
    names = tuple(dict.fromkeys(names))
    period = (int(start_year), int(end_year))
    regions_period = tuple(int(x) for x in regions_period)
    map_day = date.fromisoformat(str(map_date))
    if not names:
        raise ValueError("At least one variable required")
    if not (period[0] <= map_day.year <= period[1]):
        raise ValueError(f"map_date {map_day} not in data period {period}")
    outputs = {}
    for name in names:
        full = stationary_density(root, name, period, names=DEFAULT_PAIR,
                                  density_source=density_source,
                                  hybrid_reference=hybrid_reference)
        regions = stationary_density(root, name, regions_period,
                                     names=DEFAULT_PAIR,
                                     density_source=density_source,
                                     hybrid_reference=hybrid_reference)
        thresholds, masses = equal_probability_regions(regions["grid"], regions["pdf"])
        era5_file = _source(root, name, period)
        offset = (map_day - date(period[0], 1, 1)).days
        cache = _cache_file(root, name, period, regions_period, map_day.isoformat(),
                            f"{full['kind']}-{regions['kind']}", compute_frequencies)
        metadata = {
            "version": 2, "variable": name, "period": list(period),
            "regions_period": list(regions_period), "map_date": map_day.isoformat(),
            "density_source_requested": density_source,
            "isoline_pdf": {"source": full["kind"], "signature": full["signature"]},
            "regions_pdf": {"source": regions["kind"], "signature": regions["signature"]},
            "era5_file": _signature(era5_file),
            "chunk_days": int(chunk_days), "compute_frequencies": bool(compute_frequencies),
            "quantile_rule": "q[k] = F^-1(k/5); q[k-1] < X <= q[k]",
            "coordinate_caution": (
                "ERA5 spatial fields are raw heat fluxes. The model's effective "
                "stationary-coordinate definition can differ. These are reference-relative "
                "diagnostics; the hybrid is not the original stationary SDE solution."),
        }
        meta_path = cache.with_suffix(".json")
        if cache.is_file() and meta_path.is_file() and not overwrite:
            if _read_json(meta_path) == metadata:
                print(f"[article] Reuse {name}: {cache}", flush=True)
                outputs[name] = cache
                continue
        field = load_field(root, name, *period)
        if not (0 <= offset < len(field)):
            raise IndexError(f"Date {map_day} not in observed {name} field of {len(field)} days")
        ocean = read_ocean_mask(root, field.shape[-2:])
        mean_map, count_map, frequencies = _spatial_statistics(
            field, ocean, chunk_days=chunk_days, thresholds=thresholds,
            compute_frequencies=compute_frequencies)
        snapshot = np.asarray(field[offset], dtype=np.float64)
        class_map = classify_map(snapshot, thresholds, ocean)
        anomaly = np.where(ocean, mean_map - full["mean"], np.nan)
        standardized = anomaly / max(full["std"], 1e-12)
        output = {
            "ocean_mask": ocean, "mean_map": mean_map, "counts": count_map,
            "anomaly": anomaly, "standardized_anomaly": standardized,
            "class_map": class_map, "snapshot": snapshot,
            "grid": regions["grid"], "pdf": regions["pdf"],
            "thresholds": thresholds, "region_masses": masses,
            "model_mean": np.asarray(full["mean"]),
            "model_std": np.asarray(full["std"]),
        }
        if frequencies is not None:
            output["frequencies"] = frequencies
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **output)
        _save_json(meta_path, metadata)
        outputs[name] = cache
        print(f"[article] {name}: source={full['kind']}, mu={full['mean']:.4f}, "
              f"sigma={full['std']:.4f}, q={thresholds.round(3)}, "
              f"quintile masses={masses.round(5)}, output={cache}", flush=True)
    return outputs
