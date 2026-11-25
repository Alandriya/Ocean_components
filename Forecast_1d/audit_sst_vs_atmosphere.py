"""Hindsight decomposition: future SST versus future atmospheric forcing.

Run from repository root:
    python -m Forecast_1d.audit_sst_vs_atmosphere

All ``future_*`` cases use ACTUAL FUTURE ERA5 and are diagnostic oracles,
NOT operational forecasts. Historical train dates alone fit seasonal cycles,
AR(1) and bulk proxy-to-flux regressions. Identical test support in all cases.

Requires Forecast_1d/audit_future_weather.py from the preceding experiment.
Only NumPy plus the standard library. Does not change original data/checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_future_weather import (
    NAMES, calendar_index, fit_regression, fit_seasonal, physical_proxies, scores,
)

# Input physical variables are [sst, sp, u10, v10, t2m, d2m].
MET_NAMES = ("sst", "sp", "u10", "v10", "t2m", "d2m")
SCENARIOS = {
    "no_future_met": (),
    "future_sst_only": ("sst",),
    "future_air_thermodynamics_only": ("sp", "t2m", "d2m"),
    "future_wind_only": ("u10", "v10"),
    "future_sst_and_air_thermodynamics": ("sst", "sp", "t2m", "d2m"),
    "future_atmosphere_only": ("sp", "u10", "v10", "t2m", "d2m"),
    "future_all": MET_NAMES,
}
GROUPS = {"days_1_3": (1, 3), "days_1_7": (1, 7),
          "days_8_14": (8, 14), "days_1_14": (1, 14)}


def _validate_and_load(data_root: Path, *, val_days, test_days, in_len, horizon):
    base = Path(data_root) / "DATA" / "Forecast_1d" / "atlantic15_native_1979-2026"
    new_dir = base / "features_era5"
    old_dates = np.load(base / "dates.npy", allow_pickle=False).astype("datetime64[D]")
    old_flux = np.load(base / "centers.npy", mmap_mode="r", allow_pickle=False)
    dates = np.load(new_dir / "dates.npy", allow_pickle=False).astype("datetime64[D]")
    feat = np.load(new_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(new_dir / "targets.npy", mmap_mode="r", allow_pickle=False)
    manifest = json.loads((new_dir / "manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest["feature_names"][:8]) != NAMES:
        raise ValueError("Feature names/order inconsistent with expected sensible,latent,sst,sp,u10,v10,t2m,d2m")
    if not (feat.ndim == 3 and feat.shape[1] == 15 and feat.shape[-1] >= 8
            and targets.shape == (len(dates), 15, 2)
            and old_flux.shape[1:] == (15, 2)
            and feat.shape[0] == len(dates) and len(dates) <= len(old_dates)
            and np.array_equal(dates, old_dates[:len(dates)])
            and np.array_equal(targets, old_flux[:len(dates)], equal_nan=True)
            and np.array_equal(feat[..., :2], targets, equal_nan=True)):
        raise ValueError("Misaligned datasets, dates, flux values or points")
    if len(dates) < 3 or not np.all(np.diff(dates) == np.timedelta64(1, "D")):
        raise ValueError("Date sequence must be consecutive")
    train_end = len(old_dates) - val_days - test_days
    test_start = len(old_dates) - test_days
    if not (train_end >= 730 and train_end < test_start < len(dates)):
        raise ValueError("Invalid chronological train/val/test boundaries")
    if min(in_len, horizon) < 1:
        raise ValueError("History and forecast horizon must be positive")
    origins = np.arange(max(test_start - 1, in_len - 1), len(dates) - horizon, dtype=np.int64)
    if len(origins) == 0:
        raise ValueError("No test origins")
    return new_dir, dates, np.asarray(targets, dtype=np.float64), feat, train_end, origins


def evaluate(data_root: Path, output: Path, *, val_days=730, test_days=365,
             in_len=60, horizon=14, radius=15):
    new_dir, dates, flux, features, train_end, origins = _validate_and_load(
        Path(data_root), val_days=val_days, test_days=test_days,
        in_len=in_len, horizon=horizon)
    met = np.asarray(features[..., 2:8], dtype=np.float64)
    if not np.isfinite(met).all() or not np.isfinite(flux).all():
        raise ValueError("Nonfinite fields; inspect assembled data")
    # Assert reasonable ERA5 units and consistency with previous bulk audit.
    proxy = physical_proxies(features)
    day = calendar_index(dates)
    clim_y, phi_y = fit_seasonal(flux, dates, train_end, radius)
    clim_proxy, _ = fit_seasonal(proxy, dates, train_end, radius)
    clim_met, phi_met = fit_seasonal(met, dates, train_end, radius)
    intercept, beta = fit_regression(
        flux[:train_end] - clim_y[day[:train_end]],
        proxy[:train_end] - clim_proxy[day[:train_end]],
    )

    leads = np.arange(1, horizon + 1)
    forecast_indices = origins[:, None] + leads[None, :]
    truth = flux[forecast_indices]
    climatology_y_future = clim_y[day[forecast_indices]]
    climatology_proxy_future = clim_proxy[day[forecast_indices]]
    # All histories end at origin, with no future ERA5 used in this forecast.
    driver_anomalies_now = met[origins] - clim_met[day[origins]]
    met_ar = (clim_met[day[forecast_indices]]
              + driver_anomalies_now[:, None, :, :]
              * phi_met[None, None, :, :] ** leads[None, :, None, None])
    flux_ar = (climatology_y_future
               + (flux[origins] - clim_y[day[origins]])[:, None, :, :]
               * phi_y[None, None, :, :] ** leads[None, :, None, None])

    met_true_future = met[forecast_indices]
    names = {"seasonal_ar1": flux_ar}
    for scenario, known in SCENARIOS.items():
        mixture = met_ar.copy()
        for field_name in known:
            j = MET_NAMES.index(field_name)
            mixture[..., j] = met_true_future[..., j]
        # physical_proxies expects sensible/latent plus six physical channels.
        padded_features = np.concatenate(
            [np.zeros((*mixture.shape[:-1], 2), dtype=np.float64), mixture], axis=-1
        )
        mixed_proxy = physical_proxies(padded_features)
        names[scenario] = (
            climatology_y_future + intercept[None, None, :, :]
            + beta[None, None, :, :] *
            (mixed_proxy - climatology_proxy_future)
        )

    # Full oracle must be identical to preceding experiment (same fit, truth, dates).
    original_oracle = (climatology_y_future + intercept[None, None, :, :]
                       + beta[None, None, :, :] *
                       (proxy[forecast_indices] - climatology_proxy_future))
    if not np.allclose(names["future_all"], original_oracle, rtol=1e-12, atol=1e-10):
        raise AssertionError("Full-oracle reconstruction changed")

    def grouped_scores(xslice=slice(None), channel=slice(None)):
        return {key: scores(pred[:, xslice, :, channel],
                            truth[:, xslice, :, channel],
                            flux_ar[:, xslice, :, channel])
                for key, pred in names.items()}

    result = {
        "warning": "ALL future_* experiments use ACTUAL FUTURE ERA5: non-operational oracles; not forecast scores, neither are effects additive.",
        "reference": "Seasonal AR(1) targets, train-only fitted; original dataset split dates preserved.",
        "method": "Train-only seasonal+AR1 of each raw meteorological field; fit per-point one-proxy OLS to flux anomalies on train. Substitute selected true future ERA5 fields at identical test dates.",
        "met_input_order": list(MET_NAMES),
        "scenario_future_truth_fields": {key: list(known) for key, known in SCENARIOS.items()},
        "source": str(new_dir),
        "train_end_exclusive": int(train_end),
        "train_last_date": str(dates[train_end - 1]),
        "test_first_target_date": str(dates[origins[0] + 1]),
        "test_last_target_date": str(dates[origins[-1] + horizon]),
        "n_test_origins": int(len(origins)), "n_points": 15,
        "horizon": horizon,
        "overall": grouped_scores(),
        "horizon_groups": {}, "by_channel": {},
    }
    for group, (lo, hi) in GROUPS.items():
        if lo <= horizon:
            result["horizon_groups"][group] = grouped_scores(slice(lo-1, min(hi, horizon)))
    for ci, channel_name in enumerate(("sensible", "latent")):
        result["by_channel"][channel_name] = grouped_scores(channel=slice(ci, ci+1))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(f"[audit] train through {result['train_last_date']}; test {result['test_first_target_date']}..{result['test_last_target_date']}", flush=True)
    for name, row in result["overall"].items():
        print(f"  {name:<40} RMSE={row['rmse']:.3f}  MSE skill={100*row['mse_skill_vs_seasonal_ar1']:+.2f}%", flush=True)
    print(f"[saved] {output}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.data_root is None:
        from project_paths import DATA_ROOT
        root = Path(DATA_ROOT)
    else:
        root = args.data_root
    output = args.output or (root / "Forecast" / "Results" / "Forecast_1d" /
                             "audits" / "sst_vs_atmosphere_oracle.json")
    evaluate(root, output)


if __name__ == "__main__":
    main()
