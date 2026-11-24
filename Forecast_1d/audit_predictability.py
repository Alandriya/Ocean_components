"""Audit ERA5 feature alignment and intrinsic short-lag signal without training.

Run from repository root:
    python -m Forecast_1d.audit_predictability

Reads existing Forecast_1d 15-point feature arrays, works on the train segment
only, and writes a JSON report. No neural checkpoints or datasets are changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PREFIX = ("sensible", "latent", "sst", "sp", "u10", "v10", "t2m", "d2m")
LEADS = (1, 2, 3, 7, 14, 30)
LAGS = (-3, -2, -1, 0, 1, 2, 3)


def _calendar_index(dates: np.ndarray) -> np.ndarray:
    dates = np.asarray(dates).astype("datetime64[D]")
    months = (dates.astype("datetime64[M]").astype(np.int64) % 12).astype(int)
    day0 = (dates - dates.astype("datetime64[M]").astype("datetime64[D]")).astype(int)
    offsets = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    return offsets[months] + day0


def _climatology(train_values: np.ndarray, train_doy: np.ndarray, radius=15) -> np.ndarray:
    if np.any(~np.isfinite(train_values)):
        raise ValueError("Nonfinite training values; investigate source arrays first")
    totals = np.bincount(train_doy, weights=train_values, minlength=366)
    counts = np.bincount(train_doy, minlength=366).astype(float)
    sums_smooth = np.zeros(366)
    counts_smooth = np.zeros(366)
    for offset in range(-radius, radius + 1):
        sums_smooth += np.roll(totals, offset)
        counts_smooth += np.roll(counts, offset)
    return np.divide(sums_smooth, counts_smooth,
                     out=np.full(366, float(np.mean(train_values))), where=counts_smooth > 0)


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return None
    a, b = a[ok], b[ok]
    a -= a.mean()
    b -= b.mean()
    norm = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / norm) if norm > 1e-15 else None


def _lag_corr(target: np.ndarray, proxy: np.ndarray, lag: int) -> float | None:
    """Correlation between target[t] and proxy[t + lag]."""
    if lag > 0:
        return _corr(target[:-lag], proxy[lag:])
    if lag < 0:
        return _corr(target[-lag:], proxy[:lag])
    return _corr(target, proxy)


def _vapour_pressure(temp_kelvin: np.ndarray) -> np.ndarray:
    """Approximate water saturation vapor pressure [Pa] (diagnostic only)."""
    c = np.asarray(temp_kelvin, dtype=np.float64) - 273.15
    return 611.2 * np.exp(17.67 * c / (c + 243.5))


def _specific_humidity(vapour_pressure: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    return 0.622 * vapour_pressure / (pressure - 0.378 * vapour_pressure)


def _physical_proxies(x: np.ndarray) -> dict[str, np.ndarray]:
    """Diagnostic same-day bulk proxies, NOT future-available forecasts."""
    sst, sp, u10, v10, t2m, d2m = [x[:, j] for j in range(2, 8)]
    speed = np.hypot(u10, v10)
    dry_air = _specific_humidity(_vapour_pressure(d2m), sp)
    sea_sat = _specific_humidity(0.98 * _vapour_pressure(sst), sp)
    return {"sensible_proxy": speed * (sst - t2m),
            "latent_proxy": speed * (sea_sat - dry_air)}


def _detect_unusual_units(x: np.ndarray) -> list[str]:
    warnings = []
    for name, values, lo, hi in (
        ("sst[K]", x[:, 2], 240., 325.),
        ("surface_pressure[Pa]", x[:, 3], 5e4, 1.15e5),
        ("t2m[K]", x[:, 6], 200., 330.),
        ("d2m[K]", x[:, 7], 180., 330.),
    ):
        bad = int(np.count_nonzero(~np.isfinite(values) | (values < lo) | (values > hi)))
        if bad:
            warnings.append(f"{name}: {bad} of {len(values)} train observations outside broad sanity range {lo}..{hi}")
    return warnings


def audit(flux_dir: Path, features_dir: Path, output: Path, *, val_days=730, test_days=365):
    flux_dir, features_dir = Path(flux_dir), Path(features_dir)
    old_dates = np.load(flux_dir / "dates.npy", allow_pickle=False).astype("datetime64[D]")
    old_flux = np.load(flux_dir / "centers.npy", mmap_mode="r", allow_pickle=False)
    new_dates = np.load(features_dir / "dates.npy", allow_pickle=False).astype("datetime64[D]")
    feat = np.load(features_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    target = np.load(features_dir / "targets.npy", mmap_mode="r", allow_pickle=False)
    manifest = json.loads((features_dir / "manifest.json").read_text(encoding="utf-8"))
    names = tuple(manifest.get("feature_names", ()))
    if names[:8] != PREFIX or len(names) != 20:
        raise ValueError(f"Unexpected feature order: {names}")
    if feat.ndim != 3 or feat.shape[-1] != 20 or target.shape != feat.shape[:2] + (2,):
        raise ValueError("Unexpected feature/target tensor shapes")
    if (old_flux.shape[0] < feat.shape[0] or old_flux.shape[1:] != target.shape[1:] or
            not np.array_equal(new_dates, old_dates[:len(new_dates)])):
        raise ValueError("Feature and original flux dates / point dimensions do not align")
    if not np.array_equal(target, old_flux[:len(new_dates)], equal_nan=True):
        raise ValueError("Feature targets are shifted/different from original flux series")
    if not np.array_equal(feat[..., :2], target, equal_nan=True):
        raise ValueError("First two feature channels do not match targets")
    if np.any(np.diff(new_dates) != np.timedelta64(1, 'D')):
        raise ValueError("Feature calendar has missing/duplicate dates")
    train_end = len(old_dates) - val_days - test_days
    if train_end < 400 or len(new_dates) <= train_end:
        raise ValueError("Unusable original train / feature date boundaries")
    points = json.loads((features_dir / "points.json").read_text(encoding="utf-8"))["points"]
    if len(points) != feat.shape[1]:
        raise ValueError("Site count mismatch")
    old_points = json.loads((flux_dir / "points.json").read_text(encoding="utf-8"))["points"]
    for old, new in zip(old_points, points):
        for key in ("id", "grid_y", "grid_x", "lat_deg", "lon_deg"):
            if old.get(key) != new.get(key):
                raise ValueError(f"Point order/coordinates differ: {old.get('id')} vs {new.get('id')} [{key}]")
    doy = _calendar_index(new_dates[:train_end])
    result = {"source": str(features_dir), "first_date": str(new_dates[0]),
              "last_date": str(new_dates[-1]), "train_end_exclusive": int(train_end),
              "train_last_date": str(new_dates[train_end - 1]),
              "n_points": int(feat.shape[1]), "n_features": int(feat.shape[2]),
              "exact_flux_target_alignment": True,
              "lag_convention": "lag=k means correlating target[t] with same-day proxy[t+k]",
              "warnings": [], "points": {}}
    sources = manifest.get("source_metadata", {})
    for variable in ("sst", "sp"):
        if any("legacy" in str(src.get("date_source", "")).lower()
               for src in sources.get(variable, [])):
            result["warnings"].append(
                f"{variable}: some arrays were accepted by legacy date inference; "
                "grid orientation and time stamps cannot be independently proven from this dataset")
    acf_all = {"sensible": {str(h): [] for h in LEADS},
               "latent": {str(h): [] for h in LEADS}}
    for pi, point in enumerate(points):
        x = np.asarray(feat[:train_end, pi, :8], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite train features at {point['id']}")
        warnings = _detect_unusual_units(x)
        proxies = _physical_proxies(x)
        component_report = {}
        for ci, cname in enumerate(("sensible", "latent")):
            y = x[:, ci]
            clim = _climatology(y, doy)
            anomalous = y - clim[doy]
            variance = float(np.var(y))
            seasonal_fraction = float(1 - np.var(anomalous) / variance) if variance > 1e-15 else None
            acf = {str(h): _corr(anomalous[:-h], anomalous[h:]) for h in LEADS}
            for h, r in acf.items():
                if r is not None:
                    acf_all[cname][h].append(r)
            p_name = "sensible_proxy" if cname == "sensible" else "latent_proxy"
            p = proxies[p_name]
            p_clim = _climatology(p, doy)
            p_anom = p - p_clim[doy]
            lag_corrs = {str(lag): _lag_corr(anomalous, p_anom, lag) for lag in LAGS}
            valid_lags = {lag: corr for lag, corr in lag_corrs.items() if corr is not None}
            strongest = max(valid_lags, key=lambda k: abs(valid_lags[k])) if valid_lags else None
            component_report[cname] = {
                "train_mean": float(np.mean(y)), "train_std": float(np.std(y)),
                "seasonal_variance_fraction_approx": seasonal_fraction,
                "anomaly_autocorrelation": acf,
                "bulk_proxy_lag_correlations": lag_corrs,
                "max_abs_corr_lag": int(strongest) if strongest is not None else None,
                "same_day_proxy_correlation": lag_corrs["0"],
            }
        result["points"][point["id"]] = {"lat_deg": point.get("lat_deg"),
                                            "lon_deg": point.get("lon_deg"),
                                            "channel_diagnostics": component_report,
                                            "warnings": warnings}
    result["median_anomaly_autocorrelation_by_horizon"] = {
        name: {h: float(np.median(vals)) if vals else None for h, vals in by_h.items()}
        for name, by_h in acf_all.items()
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8")
    print("[OK] flux targets, dates and point order checked", flush=True)
    print(f"[TRAIN] {train_end} days through {new_dates[train_end-1]}", flush=True)
    print("[ACF medians]", result["median_anomaly_autocorrelation_by_horizon"], flush=True)
    for warning in result["warnings"]:
        print("[WARNING]", warning, flush=True)
    print(f"[SAVED] {output}", flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--val-days", type=int, default=730)
    p.add_argument("--test-days", type=int, default=365)
    args = p.parse_args(argv)
    if args.data_root is None:
        from project_paths import DATA_ROOT
        root = Path(DATA_ROOT)
    else:
        root = args.data_root
    ds = args.dataset_dir or root / "DATA" / "Forecast_1d" / "atlantic15_native_1979-2026"
    feat = args.features_dir or ds / "features_era5"
    output = args.output or root / "Forecast" / "Results" / "Forecast_1d" / "audits" / "predictability_audit.json"
    audit(ds, feat, output, val_days=args.val_days, test_days=args.test_days)


if __name__ == "__main__":
    main()
