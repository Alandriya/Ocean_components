"""History-only persistence, mean, seasonal climatology and seasonal autoregression forecasting benchmarks."""

from __future__ import annotations

# --- Baselines ---
# Training-free benchmarks using only observed history at forecast origin.

import numpy as np


def persistence(values: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed day for each forecast lead."""
    last = np.asarray(values[origins], dtype=np.float64)
    return np.repeat(last[..., None], horizon, axis=-1)


def input_mean(values: np.ndarray, origins: np.ndarray, horizon: int,
               in_len: int = 30) -> np.ndarray:
    """Forecast with the mean of the last in_len available observations."""
    output = np.full((len(origins), *values.shape[1:], horizon), np.nan, dtype=np.float64)
    for j, origin in enumerate(origins):
        left = int(origin) - in_len + 1
        if left < 0:
            raise ValueError("Origin has insufficient history")
        window = np.asarray(values[left:int(origin) + 1], dtype=np.float64)
        finite = np.isfinite(window)
        n = finite.sum(axis=0)
        result = np.full(n.shape, np.nan, dtype=np.float64)
        np.divide(np.where(finite, window, 0).sum(axis=0), n,
                  out=result, where=n > 0)
        output[j] = result[..., None]
    return output


# --- Seasonal ---
# Train-only seasonal climatology and autoregression of seasonal anomalies.



def calendar_index(dates):
    """Map month/day onto leap-year 2000 (March 1 never shifts between years)."""
    y = dates.astype('datetime64[Y]')
    m = dates.astype('datetime64[M]')
    month = (m - y.astype('datetime64[M]')).astype(int)
    day = (dates.astype('datetime64[D]') - m.astype('datetime64[D]')).astype(int)
    offsets = np.array([0,31,60,91,121,152,182,213,244,274,305,335])
    return offsets[month] + day


def fit_seasonal(values, dates, train_end, radius=15):
    """Pool observed training values in a circular +/- radius calendar window."""
    if not 0 <= radius < 183:
        raise ValueError('Invalid climatology radius')
    x = np.asarray(values[:train_end], dtype=np.float64)
    day = calendar_index(dates[:train_end])
    sums = np.zeros((366,*x.shape[1:]))
    counts = np.zeros_like(sums)
    valid = np.isfinite(x)
    np.add.at(sums, day, np.where(valid,x,0.0))
    np.add.at(counts, day, valid)
    smoothed_sums = sum(np.roll(sums,k,axis=0) for k in range(-radius,radius+1))
    smoothed_counts = sum(np.roll(counts,k,axis=0) for k in range(-radius,radius+1))
    # Short synthetic records may lack some months; use train-only overall means.
    total = counts.sum(axis=0)
    if np.any(total == 0):
        raise ValueError('No finite training observations for a channel')
    fallback = sums.sum(axis=0)/total
    clim = np.divide(smoothed_sums,smoothed_counts,out=np.broadcast_to(fallback,sums.shape).copy(),where=smoothed_counts>0)
    anomaly = x - clim[day]
    a,b = anomaly[:-1],anomaly[1:]
    valid = np.isfinite(a)&np.isfinite(b)
    if np.any(valid.sum(axis=0)<2):
        raise ValueError('Insufficient seasonal AR pairs')
    a,b = np.where(valid,a,0),np.where(valid,b,0)
    den = (a*a).sum(axis=0)
    phi = np.divide((a*b).sum(axis=0),den,out=np.zeros_like(den),where=den>1e-12)
    return clim,np.clip(phi,-0.999,0.999)


def forecast_seasonal(values, dates, origins, horizon, climatology, phi=None):
    """Use the known future calendar, never future observations."""
    day = calendar_index(dates)
    prediction = np.stack([climatology[day[origins+h]] for h in range(1,horizon+1)],axis=-1)
    if phi is not None:
        anomaly = np.asarray(values[origins],dtype=np.float64) - climatology[day[origins]]
        prediction += anomaly[...,None] * phi[...,None]**np.arange(1,horizon+1)
    return prediction
