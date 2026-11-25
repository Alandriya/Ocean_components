"""Which FUTURE ERA5 drivers explain flux skill: wind or gradients?

Run from the Ocean_components repository root:
    python -m Forecast_1d.audit_oracle_drivers

This is a NON-OPERATIONAL, hindsight-only diagnostic. Any scenario named
"future_*" uses actual ERA5 values after forecast issue time; it is NOT
predictive skill and must not be compared as an operational forecasting method.

Requires the existing Forecast_1d/audit_future_weather.py (the preceding
experiment). Uses only NumPy and project_paths.py. Does not alter source arrays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_future_weather import (NAMES, calendar_index, fit_seasonal,
                                   fit_regression, physical_proxies, scores)

SCENARIOS = ("seasonal_ar1", "no_future_drivers", "future_wind_only",
             "future_gradients_only", "future_all")
GROUPS = {"days_1_3": (1,3), "days_1_7": (1,7),
          "days_8_14": (8,14), "days_1_14": (1,14)}


def _drivers(features):
    """Return [time,point,3] = 10m speed, SST-T2m, qs(SST)-qa(Td2m,sp).

    Moisture formulas are IDENTICAL to audit_future_weather.physical_proxies.
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
        c = temp_kelvin - 273.15
        return 611.2 * np.exp(17.67*c / (c+243.5))

    def q(e):
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
    return np.stack([driver[..., 0]*driver[..., 1],
                     driver[..., 0]*driver[..., 2]], axis=-1)


def _median_anomaly_acf(anomaly, lag):
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


def evaluate(data_root: Path, output: Path, *, val_days=730, test_days=365,
             in_len=60, horizon=14, radius=15):
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
    for k,(lo,hi) in GROUPS.items():
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


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,default=None)
    p.add_argument("--output",type=Path,default=None)
    args=p.parse_args(argv)
    if args.data_root is None:
        from project_paths import DATA_ROOT
        root=Path(DATA_ROOT)
    else:root=args.data_root
    path=args.output or root/"Forecast"/"Results"/"Forecast_1d"/"audits"/"future_weather_drivers.json"
    evaluate(root,path)


if __name__=="__main__":
    main()
