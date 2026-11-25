"""Diagnostic of forecast information missing from future ERA5 fields.

Run from project root: python -m Forecast_1d.audit_future_weather

Compares on exactly the same test origins:
  seasonal_ar1   -- operational benchmark, history-only flux SAR(1)
  proxy_ar1      -- operational toy forecast: trained physical proxy AR(1)
  future_oracle  -- NON-OPERATIONAL hindsight: true future SST/wind/T/q

Fits calendar cycles, AR1, and proxy-to-flux calibration using TRAIN ONLY.
Oracle uses future reanalysis values; NEVER report its skill as a forecast.
No data, training artifacts, or checkpoints are modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ("sensible", "latent", "sst", "sp", "u10", "v10", "t2m", "d2m")


def calendar_index(dates):
    dates = np.asarray(dates, dtype="datetime64[D]")
    yr = dates.astype("datetime64[Y]")
    mo = dates.astype("datetime64[M]")
    m = (mo - yr.astype("datetime64[M]")).astype(int)
    d = (dates - mo.astype("datetime64[D]")).astype(int)
    return np.array([0,31,60,91,121,152,182,213,244,274,305,335])[m] + d


def fit_seasonal(values, dates, train_end, radius=15):
    """Same train-only calendar smoothing and zero-intercept AR1 as Forecast_1d.models.seasonal."""
    x = np.asarray(values[:train_end], dtype=np.float64)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite train array [T, P, 2]")
    bins = calendar_index(dates[:train_end])
    sums = np.zeros((366, *x.shape[1:]), dtype=np.float64)
    counts = np.zeros_like(sums)
    np.add.at(sums, bins, x)
    np.add.at(counts, bins, 1.0)
    ss = sum(np.roll(sums, k, axis=0) for k in range(-radius, radius + 1))
    cc = sum(np.roll(counts, k, axis=0) for k in range(-radius, radius + 1))
    total = counts.sum(axis=0)
    if (total <= 0).any():
        raise ValueError("Some point/channel lacks observations")
    fallback = sums.sum(axis=0) / total
    clim = np.divide(ss, cc, out=np.broadcast_to(fallback, sums.shape).copy(), where=cc > 0)
    anomalies = x - clim[bins]
    previous, next_day = anomalies[:-1], anomalies[1:]
    denom = (previous * previous).sum(axis=0)
    phi = np.divide((previous * next_day).sum(axis=0), denom,
                    out=np.zeros_like(denom), where=denom > 1e-12)
    return clim, np.clip(phi, -0.999, 0.999)


def physical_proxies(features):
    """Return [T,P,2]. Flux sign handled by a TRAIN-only linear regression."""
    a = np.asarray(features[..., 2:8], dtype=np.float64)
    sst, pressure, u10, v10, air, dew = [a[..., i] for i in range(6)]
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite ERA5 fields. Investigate gaps before this diagnostic")
    if not (np.all((sst > 240) & (sst < 325))
            and np.all((pressure > 50000) & (pressure < 115000))
            and np.all((air > 200) & (air < 330))
            and np.all((dew > 180) & (dew < 330))):
        raise ValueError("Unexpected ERA5 units or implausible met values")
    def vapour(t):
        c = t - 273.15
        return 611.2 * np.exp(17.67*c / (c+243.5))
    def specific(e):
        return 0.622*e/(pressure-0.378*e)
    speed = np.hypot(u10, v10)
    return np.stack([speed*(sst-air),
                     speed*(specific(0.98*vapour(sst))-specific(vapour(dew)))], axis=-1)


def fit_regression(flux_anomaly, proxy_anomaly):
    """Per-point, per-target TRAIN-only OLS: y = intercept + beta * proxy."""
    y = np.asarray(flux_anomaly, dtype=np.float64)
    z = np.asarray(proxy_anomaly, dtype=np.float64)
    if y.shape != z.shape or y.ndim != 3 or not np.isfinite(y).all() or not np.isfinite(z).all():
        raise ValueError("Unexpected or nonfinite train anomalies")
    ymu, zmu = y.mean(axis=0), z.mean(axis=0)
    yp, zp = y - ymu, z - zmu
    den = np.sum(zp*zp, axis=0)
    if np.any(den <= 1e-12):
        raise ValueError("Degenerate physical proxy on train")
    beta = np.sum(zp*yp, axis=0) / den
    intercept = ymu - beta*zmu
    return intercept, beta


def scores(pred, truth, reference):
    if not (pred.shape == truth.shape == reference.shape):
        raise ValueError("Incompatible prediction shapes")
    valid = np.isfinite(pred) & np.isfinite(truth) & np.isfinite(reference)
    if not valid.all():
        raise ValueError("Nonfinite evaluation values; report should use common finite mask")
    def one(a,b):
        return float(np.mean((a-b)**2))
    mse = one(pred,truth)
    baseline = one(reference,truth)
    return dict(rmse=float(np.sqrt(mse)), mse=mse,
                mse_skill_vs_seasonal_ar1=float(1-mse/baseline) if baseline>0 else None)


def audit(data_root, output, *, val_days=730, test_days=365, in_len=60,
          horizon=14, radius=15):
    root = Path(data_root)
    base = root / "DATA" / "Forecast_1d" / "atlantic15_native_1979-2026"
    directory = base / "features_era5"
    old_dates = np.load(base / "dates.npy", allow_pickle=False).astype("datetime64[D]")
    flux = np.load(base / "centers.npy", mmap_mode="r", allow_pickle=False)
    dates = np.load(directory / "dates.npy", allow_pickle=False).astype("datetime64[D]")
    features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest["feature_names"][:8]) != NAMES:
        raise ValueError("Unexpected feature order: " + repr(manifest["feature_names"][:8]))
    if (features.ndim != 3 or features.shape[2] < 8 or flux.shape[1:] != (features.shape[1],2)
            or dates.shape != (features.shape[0],) or len(dates) > len(old_dates)
            or not np.array_equal(dates, old_dates[:len(dates)])):
        raise ValueError("Misaligned point/feature/flux array dimensions or dates")
    targets = np.load(directory / "targets.npy", mmap_mode="r", allow_pickle=False)
    if (targets.shape != (len(dates), features.shape[1], 2)
            or not np.array_equal(targets, flux[:len(dates)], equal_nan=True)
            or not np.array_equal(features[..., :2], targets, equal_nan=True)):
        raise ValueError("Feature targets do not exactly agree with original flux series")
    if not np.all(np.diff(dates) == np.timedelta64(1, "D")):
        raise ValueError("Missing or repeated calendar dates")

    train_end = len(old_dates)-val_days-test_days
    test_start = len(old_dates)-test_days
    if train_end < 365*2 or test_start >= len(dates)-horizon:
        raise ValueError("Not enough train or test dates")
    origins = np.arange(max(test_start-1, in_len-1), len(dates)-horizon, dtype=np.int64)
    if origins.size < 1:
        raise ValueError("No valid common test origins")
    # Local dates and dates of targets; no training or fitting on test data.
    day = calendar_index(dates)
    fl = np.asarray(flux[:len(dates)], dtype=np.float64)
    if not np.isfinite(fl).all():
        raise ValueError("Nonfinite target flux data")
    proxy = physical_proxies(features)
    clim_y, phi_y = fit_seasonal(fl, dates, train_end, radius)
    clim_p, phi_p = fit_seasonal(proxy, dates, train_end, radius)
    intercept, beta = fit_regression(
        fl[:train_end]-clim_y[day[:train_end]],
        proxy[:train_end]-clim_p[day[:train_end]])
    lead = np.arange(1,horizon+1)
    future_idx = origins[:,None]+lead[None,:]  # [origins,horizon]
    # All following prediction arrays are [origins,horizon,points,2].
    c_y = clim_y[day[future_idx]]
    c_p = clim_p[day[future_idx]]
    p_future = proxy[future_idx]  # ORACLE ONLY -- FUTURE OBSERVED ERA5!
    p_origin_anomaly = proxy[origins]-clim_p[day[origins]]
    p_ar = c_p + p_origin_anomaly[:,None,:,:]*phi_p[None,None,:,:]**lead[None,:,None,None]
    y_ar = c_y + (fl[origins]-clim_y[day[origins]])[:,None,:,:] * phi_y[None,None,:,:]**lead[None,:,None,None]
    y_proxy_ar = c_y + intercept[None,None,:,:]+beta[None,None,:,:]*(p_ar-c_p)
    y_oracle = c_y + intercept[None,None,:,:]+beta[None,None,:,:]*(p_future-c_p)
    truth = fl[future_idx]
    # Score each on identical data support; horizon aggregation includes every point/channel/origin.
    summary = {
        "warning": "FUTURE_ORACLE uses FUTURE ACTUAL ERA5; it is hindsight reconstruction, NOT an operational forecast or achievable skill bound.",
        "method": "train-only seasonal anomalies; per-point one-proxy OLS; compare actual future proxy against AR(1)-extrapolated proxy",
        "source": str(directory), "train_end_exclusive":int(train_end),
        "train_last_date":str(dates[train_end-1]),
        "test_first_target_date":str(dates[origins[0]+1]),
        "test_last_target_date":str(dates[origins[-1]+horizon]),
        "n_test_origins":int(len(origins)), "n_points":int(features.shape[1]),
        "horizon":int(horizon), "all_horizons":{}, "by_horizon":{},
        "ols_coefficients": {"intercept":intercept.tolist(),"slope":beta.tolist()},
    }
    for name, prediction in (("seasonal_ar1",y_ar),
                             ("proxy_ar1",y_proxy_ar),
                             ("future_oracle",y_oracle)):
        summary["all_horizons"][name] = scores(prediction,truth,y_ar)
    for h in (1,2,3,7,10,14):
        if h>horizon: continue
        sl=slice(h-1,h)
        summary["by_horizon"][str(h)]={name:scores(pred[:,sl],truth[:,sl],y_ar[:,sl])
             for name,pred in (("seasonal_ar1",y_ar),
                               ("proxy_ar1",y_proxy_ar),
                               ("future_oracle",y_oracle))}
    summary["by_channel"] = {}
    for i, name in enumerate(("sensible","latent")):
        summary["by_channel"][name] = {label:scores(pred[...,i:i+1],truth[...,i:i+1],y_ar[...,i:i+1])
            for label,pred in (("seasonal_ar1",y_ar),
                               ("proxy_ar1",y_proxy_ar),
                               ("future_oracle",y_oracle))}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print("[OK] train ends",summary["train_last_date"],"test dates",
          summary["test_first_target_date"],"through",summary["test_last_target_date"])
    for name,score in summary["all_horizons"].items():
        print(f"[{name}] RMSE={score['rmse']:.3f} skill={score['mse_skill_vs_seasonal_ar1']}")
    print("[SAVED]",output)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=None)
    parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args(argv)
    if args.data_root is None:
        from project_paths import DATA_ROOT
        root=Path(DATA_ROOT)
    else:
        root=args.data_root
    destination=args.output or root/"Forecast"/"Results"/"Forecast_1d"/"audits"/"future_weather_oracle.json"
    audit(root,destination)


if __name__=="__main__":
    main()
