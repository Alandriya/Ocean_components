"""Independent diagnostics for forecast predictability, future-weather information, oracle drivers, SST versus atmosphere and saved training histories."""

from __future__ import annotations

# --- Audit Future Weather ---
# Diagnostic of forecast information missing from future ERA5 fields.
#
# Run from project root: python -m Forecast_1d.audit future-weather
#
# Compares on exactly the same test origins:
#   seasonal_ar1   -- operational benchmark, history-only flux SAR(1)
#   proxy_ar1      -- operational toy forecast: trained physical proxy AR(1)
#   future_oracle  -- NON-OPERATIONAL hindsight: true future SST/wind/T/q
#
# Fits calendar cycles, AR1, and proxy-to-flux calibration using TRAIN ONLY.
# Oracle uses future reanalysis values; NEVER report its skill as a forecast.
# No data, training artifacts, or checkpoints are modified.

import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ("sensible", "latent", "sst", "sp", "u10", "v10", "t2m", "d2m")


def calendar_index(dates):
    """Map dates to day-of-year indices on a common leap-year calendar."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    yr = dates.astype("datetime64[Y]")
    mo = dates.astype("datetime64[M]")
    m = (mo - yr.astype("datetime64[M]")).astype(int)
    d = (dates - mo.astype("datetime64[D]")).astype(int)
    return np.array([0,31,60,91,121,152,182,213,244,274,305,335])[m] + d


def fit_seasonal(values, dates, train_end, radius=15):
    """Same train-only calendar smoothing and zero-intercept AR1 as Forecast_1d.models.baselines."""
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
        """Evaluate saturation vapor pressure from absolute temperature."""
        c = t - 273.15
        return 611.2 * np.exp(17.67*c / (c+243.5))
    def specific(e):
        """Calculate specific humidity from vapor pressure and ambient pressure."""
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
    """Compute forecast skill metrics for aligned prediction, truth and reference arrays."""
    if not (pred.shape == truth.shape == reference.shape):
        raise ValueError("Incompatible prediction shapes")
    valid = np.isfinite(pred) & np.isfinite(truth) & np.isfinite(reference)
    if not valid.all():
        raise ValueError("Nonfinite evaluation values; report should use common finite mask")
    def one(a,b):
        """Compute one for scores from ``a``, ``b``."""
        return float(np.mean((a-b)**2))
    mse = one(pred,truth)
    baseline = one(reference,truth)
    return dict(rmse=float(np.sqrt(mse)), mse=mse,
                mse_skill_vs_seasonal_ar1=float(1-mse/baseline) if baseline>0 else None)


def audit_future_weather(data_root, output, *, val_days=730, test_days=365, in_len=60,
          horizon=14, radius=15):
    """Compare operational history-only forecasts with a future-weather hindsight oracle."""
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


def main_future_weather(argv=None):
    """Run the future weather workflow using its configured options."""
    parser=argparse.ArgumentParser(description='Diagnostic of forecast information missing from future ERA5 fields.\n\nRun from project root: python -m Forecast_1d.audit future-weather\n\nCompares on exactly the same test origins:\n  seasonal_ar1   -- operational benchmark, history-only flux SAR(1)\n  proxy_ar1      -- operational toy forecast: trained physical proxy AR(1)\n  future_oracle  -- NON-OPERATIONAL hindsight: true future SST/wind/T/q\n\nFits calendar cycles, AR1, and proxy-to-flux calibration using TRAIN ONLY.\nOracle uses future reanalysis values; NEVER report its skill as a forecast.\nNo data, training artifacts, or checkpoints are modified.')
    parser.add_argument("--data-root",type=Path,default=None)
    parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args(argv)
    if args.data_root is None:
        from project_paths import DATA_ROOT
        root=Path(DATA_ROOT)
    else:
        root=args.data_root
    destination=args.output or root/"Forecast"/"Results"/"Forecast_1d"/"audits"/"future_weather_oracle.json"
    audit_future_weather(root,destination)


# --- Audit Oracle Drivers ---
# Which FUTURE ERA5 drivers explain flux skill: wind or gradients?
#
# Run from the Ocean_components repository root:
#     python -m Forecast_1d.audit oracle-drivers
#
# This is a NON-OPERATIONAL, hindsight-only diagnostic. Any scenario named
# "future_*" uses actual ERA5 values after forecast issue time; it is NOT
# predictive skill and must not be compared as an operational forecasting method.
#
# Uses the shared physical-proxy helpers in this module. Uses only NumPy and project_paths.py. Does not alter source arrays.





ORACLE_SCENARIOS = ("seasonal_ar1", "no_future_drivers", "future_wind_only",
             "future_gradients_only", "future_all")
ORACLE_GROUPS = {"days_1_3": (1,3), "days_1_7": (1,7),
          "days_8_14": (8,14), "days_1_14": (1,14)}


def _drivers(features):
    """Return [time,point,3] = 10m speed, SST-T2m, qs(SST)-qa(Td2m,sp).

    Moisture formulas are IDENTICAL to the shared physical_proxies helper.
    """
    a = np.asarray(features[..., 2:8], dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite ERA5 physical variables")
    sst, pressure, u10, v10, t2m, d2m = [a[..., i] for i in range(6)]
    if not (np.all((sst > 240) & (sst < 325)) and
            np.all((pressure > 50000) & (pressure < 115000)) and
            np.all((t2m > 200) & (t2m < 330)) and
            np.all((d2m > 180) & (d2m < 330))):
        raise ValueError("Unexpected SST, pressure, air temperature or dewpoint values")

    def vapour(temp_kelvin):
        """Evaluate saturation vapor pressure from absolute temperature."""
        c = temp_kelvin - 273.15
        return 611.2 * np.exp(17.67*c / (c+243.5))

    def q(e):
        """Convert vapor pressure to specific humidity."""
        return 0.622 * e / (pressure - 0.378*e)

    speed = np.hypot(u10, v10)
    grad_t = sst - t2m
    grad_q = q(0.98*vapour(sst)) - q(vapour(d2m))
    components = np.stack([speed, grad_t, grad_q], axis=-1)
    reconstructed = _proxies_from_drivers(components)
    if not np.allclose(reconstructed, physical_proxies(features), rtol=1e-10, atol=1e-12):
        raise AssertionError("Driver decomposition differs from previous oracle")
    return components


def _proxies_from_drivers(driver):
    """Construct physical heat-flux proxies from the supplied meteorological drivers."""
    return np.stack([driver[..., 0]*driver[..., 1],
                     driver[..., 0]*driver[..., 2]], axis=-1)


def _median_anomaly_acf(anomaly, lag):
    """Compute median autocorrelation across the anomaly series at the specified lag."""
    if lag >= len(anomaly):
        return [None] * anomaly.shape[-1]
    result=[]
    for c in range(anomaly.shape[-1]):
        corr=[]
        for p in range(anomaly.shape[1]):
            x = anomaly[:-lag,p,c]
            y = anomaly[lag:,p,c]
            if np.std(x)>1e-12 and np.std(y)>1e-12:
                corr.append(float(np.corrcoef(x,y)[0,1]))
        result.append(float(np.median(corr)) if corr else None)
    return result


def evaluate_oracle_drivers(data_root: Path, output: Path, *, val_days=730, test_days=365,
             in_len=60, horizon=14, radius=15):
    """Measure the effect of individual future drivers using a controlled hindsight experiment."""
    base=Path(data_root)/"DATA"/"Forecast_1d"/"atlantic15_native_1979-2026"
    mult=base/"features_era5"
    old_dates=np.load(base/"dates.npy",allow_pickle=False).astype("datetime64[D]")
    original=np.load(base/"centers.npy",mmap_mode="r",allow_pickle=False)
    dates=np.load(mult/"dates.npy",allow_pickle=False).astype("datetime64[D]")
    features=np.load(mult/"features.npy",mmap_mode="r",allow_pickle=False)
    target=np.load(mult/"targets.npy",mmap_mode="r",allow_pickle=False)
    manifest=json.loads((mult/"manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest["feature_names"][:8])!=NAMES:
        raise ValueError("Unexpected physical feature names/order")
    if (features.ndim!=3 or features.shape[2]<8 or
        features.shape[1]!=15 or target.shape!=(len(dates),15,2) or
        original.shape[1:]!=(15,2) or len(dates)!=features.shape[0] or
        len(dates)>len(old_dates) or
        not np.array_equal(dates,old_dates[:len(dates)]) or
        not np.array_equal(target,original[:len(dates)],equal_nan=True) or
        not np.array_equal(features[...,:2],target,equal_nan=True)):
        raise ValueError("Misaligned target, features or date dimensions")
    if not np.all(np.diff(dates)==np.timedelta64(1,"D")):
        raise ValueError("Non-consecutive dates")
    train_end=len(old_dates)-val_days-test_days
    test_start=len(old_dates)-test_days
    if train_end<366*2 or not (train_end<test_start<len(dates)):
        raise ValueError("Invalid train/test boundaries")
    origins=np.arange(max(test_start-1,in_len-1),len(dates)-horizon,dtype=np.int64)
    if not len(origins): raise ValueError("No common test origins")
    day=calendar_index(dates)
    f=np.asarray(target,dtype=np.float64)
    if not np.isfinite(f).all(): raise ValueError("Nonfinite fluxes")
    driver=_drivers(features)
    proxy=_proxies_from_drivers(driver)
    climate_f,phi_f=fit_seasonal(f,dates,train_end,radius)
    climate_p,_=fit_seasonal(proxy,dates,train_end,radius)
    climate_d,phi_d=fit_seasonal(driver,dates,train_end,radius)
    intercept,beta=fit_regression(f[:train_end]-climate_f[day[:train_end]],
                                   proxy[:train_end]-climate_p[day[:train_end]])
    leads=np.arange(1,horizon+1)
    idx=origins[:,None]+leads[None,:]
    truth=f[idx]
    clim_future_f=climate_f[day[idx]]
    clim_future_p=climate_p[day[idx]]
    observed_future=driver[idx]
    # Fully causal reference for drivers: seasonal mean + trained AR(1) anomaly
    current_anomaly=driver[origins]-climate_d[day[origins]]
    driver_ar=(climate_d[day[idx]] + current_anomaly[:,None,:,:]*
               phi_d[None,None,:,:]**leads[None,:,None,None])
    y_ar=clim_future_f+(f[origins]-climate_f[day[origins]])[:,None,:,:]*phi_f[None,None,:,:]**leads[None,:,None,None]
    mixtures={
        "no_future_drivers": driver_ar,
        "future_wind_only": np.concatenate([observed_future[...,:1],driver_ar[...,1:]],axis=-1),
        "future_gradients_only": np.concatenate([driver_ar[...,:1],observed_future[...,1:]],axis=-1),
        "future_all": observed_future,
    }
    forecasts={"seasonal_ar1":y_ar}
    for name,mixed in mixtures.items():
        proxies=_proxies_from_drivers(mixed)
        forecasts[name]=clim_future_f+intercept[None,None,:,:]+beta[None,None,:,:]*(proxies-clim_future_p)
    # Guarantee full oracle reconstruction equals the previously-defined oracle formula.
    previous_oracle=(clim_future_f+intercept[None,None,:,:]+beta[None,None,:,:]*
                     (physical_proxies(features)[idx]-clim_future_p))
    if not np.allclose(forecasts["future_all"],previous_oracle,rtol=1e-10,atol=1e-8):
        raise AssertionError("Full oracle not equal to earlier audit")
    result={
        "warning":"Non-operational diagnostic. All future_* scenarios use true FUTURE ERA5. Not forecast skill. Partial-oracle effects are not additive.",
        "training":"Only train dates used to fit all climatologies, AR(1), and flux-proxy OLS.",
        "source":str(mult),
        "train_last_date":str(dates[train_end-1]),
        "test_first_target_date":str(dates[origins[0]+1]),
        "test_last_target_date":str(dates[origins[-1]+horizon]),
        "n_test_origins":len(origins),"n_points":15,"horizon":horizon,
        "driver_order":["wind_speed_10m","sst_minus_t2m_K","qs_minus_qa_kg_per_kg"],
        "method":"Train-only seasonal+AR1 for 3 bulk drivers; train-only one-proxy OLS per flux and point. Replace either future wind or future temperature/humidity gradients with ERA5 truth; fixed test support.",
        "overall":{name:scores(pred,truth,y_ar) for name,pred in forecasts.items()},
        "horizon_groups":{},"by_channel":{},
        "driver_train_anomaly_autocorrelation_median":{},
    }
    for k,(lo,hi) in ORACLE_GROUPS.items():
        if lo>horizon:continue
        idxslice=slice(lo-1,min(hi,horizon))
        result["horizon_groups"][k]={name:scores(pred[:,idxslice],truth[:,idxslice],y_ar[:,idxslice]) for name,pred in forecasts.items()}
    for ci,cname in enumerate(("sensible","latent")):
        result["by_channel"][cname]={name:scores(pred[...,ci:ci+1],truth[...,ci:ci+1],y_ar[...,ci:ci+1]) for name,pred in forecasts.items()}
    a=driver[:train_end]-climate_d[day[:train_end]]
    for lag in (1,3,7,14):
        result["driver_train_anomaly_autocorrelation_median"][str(lag)]=dict(zip(result["driver_order"],_median_anomaly_acf(a,lag)))
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print("[test]",result["test_first_target_date"],"..",result["test_last_target_date"])
    for name,m in result["overall"].items():
        print(f"[{name}] RMSE={m['rmse']:.3f} skill={m['mse_skill_vs_seasonal_ar1']:+.2%}")
    print("[saved]",output)
    return result


def main_oracle_drivers(argv=None):
    """Run the oracle drivers workflow using its configured options."""
    p=argparse.ArgumentParser(description='Which FUTURE ERA5 drivers explain flux skill: wind or gradients?\n\nRun from the Ocean_components repository root:\n    python -m Forecast_1d.audit oracle-drivers\n\nThis is a NON-OPERATIONAL, hindsight-only diagnostic. Any scenario named\n"future_*" uses actual ERA5 values after forecast issue time; it is NOT\npredictive skill and must not be compared as an operational forecasting method.\n\nUses the shared physical-proxy helpers in this module. Uses only NumPy and project_paths.py. Does not alter source arrays.')
    p.add_argument("--data-root",type=Path,default=None)
    p.add_argument("--output",type=Path,default=None)
    args=p.parse_args(argv)
    if args.data_root is None:
        from project_paths import DATA_ROOT
        root=Path(DATA_ROOT)
    else:root=args.data_root
    path=args.output or root/"Forecast"/"Results"/"Forecast_1d"/"audits"/"future_weather_drivers.json"
    evaluate_oracle_drivers(root,path)


# --- Audit Predictability ---
# Audit ERA5 feature alignment and intrinsic short-lag signal without training.
#
# Run from repository root:
#     python -m Forecast_1d.audit predictability
#
# Reads existing Forecast_1d 15-point feature arrays, works on the train segment
# only, and writes a JSON report. No neural checkpoints or datasets are changed.



PREFIX = ("sensible", "latent", "sst", "sp", "u10", "v10", "t2m", "d2m")
LEADS = (1, 2, 3, 7, 14, 30)
LAGS = (-3, -2, -1, 0, 1, 2, 3)


def _calendar_index(dates: np.ndarray) -> np.ndarray:
    """Map dates to a common 366-day calendar."""
    dates = np.asarray(dates).astype("datetime64[D]")
    months = (dates.astype("datetime64[M]").astype(np.int64) % 12).astype(int)
    day0 = (dates - dates.astype("datetime64[M]").astype("datetime64[D]")).astype(int)
    offsets = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    return offsets[months] + day0


def _climatology(train_values: np.ndarray, train_doy: np.ndarray, radius=15) -> np.ndarray:
    """Estimate the calendar climatology using training observations only."""
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
    """Compute the correlation of aligned finite values."""
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
    """Convert vapor pressure and total pressure to specific humidity."""
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
    """Flag meteorological arrays with implausible value ranges."""
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


def audit_predictability(flux_dir: Path, features_dir: Path, output: Path, *, val_days=730, test_days=365):
    """Measure correlations, persistence and physical-proxy predictability on the specified data."""
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


def main_predictability(argv=None):
    """Run the predictability workflow using its configured options."""
    p = argparse.ArgumentParser(description='Audit ERA5 feature alignment and intrinsic short-lag signal without training.\n\nRun from repository root:\n    python -m Forecast_1d.audit predictability\n\nReads existing Forecast_1d 15-point feature arrays, works on the train segment\nonly, and writes a JSON report. No neural checkpoints or datasets are changed.')
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
    audit_predictability(ds, feat, output, val_days=args.val_days, test_days=args.test_days)


# --- Audit Sst Vs Atmosphere ---
# Hindsight decomposition: future SST versus future atmospheric forcing.
#
# Run from repository root:
#     python -m Forecast_1d.audit sst-vs-atmosphere
#
# All ``future_*`` cases use ACTUAL FUTURE ERA5 and are diagnostic oracles,
# NOT operational forecasts. Historical train dates alone fit seasonal cycles,
# AR(1) and bulk proxy-to-flux regressions. Identical test support in all cases.
#
# Uses the shared physical-proxy helpers in this module.
# Only NumPy plus the standard library. Does not change original data/checkpoints.





# Input physical variables are [sst, sp, u10, v10, t2m, d2m].
MET_NAMES = ("sst", "sp", "u10", "v10", "t2m", "d2m")
SST_SCENARIOS = {
    "no_future_met": (),
    "future_sst_only": ("sst",),
    "future_air_thermodynamics_only": ("sp", "t2m", "d2m"),
    "future_wind_only": ("u10", "v10"),
    "future_sst_and_air_thermodynamics": ("sst", "sp", "t2m", "d2m"),
    "future_atmosphere_only": ("sp", "u10", "v10", "t2m", "d2m"),
    "future_all": MET_NAMES,
}
SST_GROUPS = {"days_1_3": (1, 3), "days_1_7": (1, 7),
          "days_8_14": (8, 14), "days_1_14": (1, 14)}


def _validate_and_load(data_root: Path, *, val_days, test_days, in_len, horizon):
    """Validate and load for the audit workflow."""
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


def evaluate_sst_vs_atmosphere(data_root: Path, output: Path, *, val_days=730, test_days=365,
             in_len=60, horizon=14, radius=15):
    """Compare SST and atmospheric-driver information in a hindsight forecast audit."""
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
    for scenario, known in SST_SCENARIOS.items():
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
        """Evaluate forecast metrics for the selected lead and channel group."""
        return {key: scores(pred[:, xslice, :, channel],
                            truth[:, xslice, :, channel],
                            flux_ar[:, xslice, :, channel])
                for key, pred in names.items()}

    result = {
        "warning": "ALL future_* experiments use ACTUAL FUTURE ERA5: non-operational oracles; not forecast scores, neither are effects additive.",
        "reference": "Seasonal AR(1) targets, train-only fitted; original dataset split dates preserved.",
        "method": "Train-only seasonal+AR1 of each raw meteorological field; fit per-point one-proxy OLS to flux anomalies on train. Substitute selected true future ERA5 fields at identical test dates.",
        "met_input_order": list(MET_NAMES),
        "scenario_future_truth_fields": {key: list(known) for key, known in SST_SCENARIOS.items()},
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
    for group, (lo, hi) in SST_GROUPS.items():
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


def main_sst_vs_atmosphere(argv=None):
    """Run the SST vs atmosphere workflow using its configured options."""
    parser = argparse.ArgumentParser(description='Hindsight decomposition: future SST versus future atmospheric forcing.\n\nRun from repository root:\n    python -m Forecast_1d.audit sst-vs-atmosphere\n\nAll ``future_*`` cases use ACTUAL FUTURE ERA5 and are diagnostic oracles,\nNOT operational forecasts. Historical train dates alone fit seasonal cycles,\nAR(1) and bulk proxy-to-flux regressions. Identical test support in all cases.\n\nUses the shared physical-proxy helpers in this module.\nOnly NumPy plus the standard library. Does not change original data/checkpoints.')
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
    evaluate_sst_vs_atmosphere(root, output)


# --- Audit Training ---
# Inspect saved Forecast_1d training traces without third-party dependencies.
# Reads only; writes a JSON report at the explicitly requested output path.

import math


def finite(value):
    """Check whether a value is a finite number."""
    return isinstance(value, (float, int)) and math.isfinite(value)


def audit_training(root):
    """Collect point-specific training histories and compare recorded validation losses."""
    records = []
    warnings = []
    for meta_path in sorted(root.glob('*/P*/train_metadata.json')):
        path = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            history = json.loads((path / 'history.json').read_text(encoding='utf-8'))
            if not isinstance(history, list) or not history:
                raise ValueError('Empty or unsupported history')
            valid = [r for r in history if finite(r.get('val_loss')) and finite(r.get('train_loss'))]
            if not valid:
                raise ValueError('No finite train/validation pairs')
            best = min(valid, key=lambda r: r['val_loss'])
            saved = next((r for r in valid if r['epoch'] == meta.get('best_epoch')), None)
            first, last = valid[0], valid[-1]
            records.append({
                'model': meta.get('model', path.parent.name),
                'point_id': meta.get('point_id', path.name),
                'input_length': meta.get('input_length'),
                'horizon': meta.get('horizon'),
                'dataset_fingerprint_sha256': meta.get('dataset_fingerprint_sha256'),
                'seed': meta.get('seed'), 'parameter_count': meta.get('parameter_count'),
                'model_config': meta.get('model_config'),
                'training_config': meta.get('training_config'),
                'selection_metric': meta.get('selection_metric', 'normalized'),
                'best_selection_loss': meta.get('best_selection_loss'),
                'best_validation_sar_skill': meta.get('best_validation_sar_skill'),
                'stop_reason': meta.get('stop_reason'),
                'epochs_logged': len(history), 'finite_epochs': len(valid),
                'first_train_loss': first['train_loss'], 'first_val_loss': first['val_loss'],
                'last_epoch': last['epoch'], 'last_train_loss': last['train_loss'],
                'last_val_loss': last['val_loss'], 'best_observed_epoch': best['epoch'],
                'best_observed_val_loss': best['val_loss'],
                'saved_best_epoch': meta.get('best_epoch'),
                'saved_best_validation_loss_normalized': meta.get('best_validation_loss_normalized'),
                'train_loss_at_saved_best': saved['train_loss'] if saved else None,
                'val_loss_at_saved_best': saved['val_loss'] if saved else None,
                'last_minus_best_val_loss': last['val_loss'] - best['val_loss'],
                'epochs_after_saved_best': last['epoch'] - saved['epoch'] if saved else None,
                'history': history,
            })
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            warnings.append({'path': str(path), 'error': str(exc)})
    return {'results_dir': str(root), 'record_count': len(records), 'warnings': warnings,
            'note': 'train_loss and val_loss are normalized MSE. Saved epoch may differ from normalized MSE minimum if SAR selection is used; older jobs may lack stop_reason.',
            'records': records}


def main_training(argv=None):
    """Run the training workflow using its configured options."""
    parser = argparse.ArgumentParser(description='Inspect saved Forecast_1d training traces without third-party dependencies.\nReads only; writes a JSON report at the explicitly requested output path.')
    parser.add_argument('--results-dir', type=Path, required=True, help='Existing h14 result directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.results_dir.resolve()
    if not root.is_dir():
        parser.error(f'Results directory not found: {root}')
    report = audit_training(root)
    if not report['record_count']:
        parser.error(f'No readable point-specific training histories: {report["warnings"]}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Saved {report["record_count"]} training traces to {args.output}; warnings={len(report["warnings"])}')


def main(argv=None):
    """Dispatch one audit; pass all remaining options to its original CLI."""
    import sys
    commands = {
        "future-weather": main_future_weather,
        "oracle-drivers": main_oracle_drivers,
        "predictability": main_predictability,
        "sst-vs-atmosphere": main_sst_vs_atmosphere,
        "training": main_training,
    }
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", choices=commands)
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return
    selected = parser.parse_args(args[:1])
    return commands[selected.audit](args[1:])


if __name__ == "__main__":
    main()
