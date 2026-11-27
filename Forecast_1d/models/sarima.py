"""Configurable SARIMA forecasting, with a NumPy AR(1) fast path.

When order=(1,0,0), seasonal_order=(0,0,0,0), trend='c', the conditional
least-squares AR(1) backend needs only NumPy. Other specifications use
statsmodels SARIMAX, fitted on the training interval only. At evaluation,
parameters are fixed and state updates use observations available by the
current forecast origin; no future target is provided to its own forecast.
"""
from __future__ import annotations

import numpy as np


def parse_order(value, count: int) -> tuple[int, ...]:
    """Parse an autoregressive model order specification."""
    if isinstance(value, (list, tuple)):
        parts = tuple(int(v) for v in value)
    else:
        parts = tuple(int(v.strip()) for v in str(value).split(","))
    if len(parts) != count or any(p < 0 for p in parts):
        raise ValueError(f"Expected {count} nonnegative integers, got {value}")
    return parts


def fast_ar1_supported(order, seasonal_order, trend):
    """Check whether the requested model supports the fast AR(1) implementation."""
    return tuple(order) == (1, 0, 0) and tuple(seasonal_order) == (0, 0, 0, 0) and trend == "c"


def fit_ar1(values: np.ndarray, fit_start: int, fit_end: int) -> np.ndarray:
    """Return per-point/channel [intercept, phi, residual_variance, pair_count]."""
    series = np.asarray(values[fit_start:fit_end], dtype=np.float64)
    if series.ndim != 3 or series.shape[1] < 1 or series.shape[2] != 2:
        raise ValueError("Expected a [T,P,2] center series")
    result = np.full((values.shape[1], 2, 4), np.nan, dtype=np.float64)
    for point in range(values.shape[1]):
        for channel in range(2):
            x, y = series[:-1, point, channel], series[1:, point, channel]
            valid = np.isfinite(x) & np.isfinite(y)
            n = int(valid.sum())
            if n < 40:
                raise ValueError(f"AR(1) has fewer than 40 valid pairs: P{point+1}, {channel}")
            x, y = x[valid], y[valid]
            xm, ym = float(x.mean()), float(y.mean())
            ss = float(np.sum((x - xm) ** 2))
            if ss <= 1e-16:
                phi = 0.0
            else:
                phi = float(np.dot(x - xm, y - ym) / ss)
            intercept = ym - phi * xm
            sigma2 = float(np.mean((y - intercept - phi * x) ** 2))
            result[point, channel] = intercept, phi, sigma2, n
    return result


def forecast_ar1(values: np.ndarray, origins: np.ndarray, horizon: int,
                 params: np.ndarray) -> np.ndarray:
    """Generate the autoregressive AR(1) forecast from the available history."""
    if params.shape != (values.shape[1], 2, 4):
        raise ValueError(f"Invalid AR(1) parameter shape: {params.shape}")
    current = np.asarray(values[origins], dtype=np.float64)
    forecast = np.full((len(origins), values.shape[1], 2, horizon), np.nan, dtype=np.float64)
    for h in range(horizon):
        current = params[None, :, :, 0] + params[None, :, :, 1] * current
        forecast[..., h] = current
    return forecast


def _sarimax_model(endog, order, seasonal_order, trend):
    """Compute sarimax model for SARIMA from ``endog``, ``order``, ``seasonal_order``, ``trend``."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise RuntimeError(
            "Custom SARIMA requires statsmodels. Install it in the Python "
            "environment or use the default (1,0,0) NumPy AR(1) specification."
        ) from exc
    return SARIMAX(endog, order=tuple(order), seasonal_order=tuple(seasonal_order),
                   trend=trend, enforce_stationarity=True,
                   enforce_invertibility=True)


def fit_sarimax(values, fit_start, fit_end, order, seasonal_order, trend, maxiter=200):
    """Estimate SARIMAX parameters independently for each location/channel."""
    result = []
    for point in range(values.shape[1]):
        row = []
        for channel in range(2):
            series = np.asarray(values[fit_start:fit_end, point, channel], dtype=np.float64)
            if np.isfinite(series).sum() < 40:
                raise ValueError(f"Too few finite data for P{point+1} channel {channel}")
            model = _sarimax_model(series, order, seasonal_order, trend)
            fitted = model.fit(disp=False, maxiter=maxiter)
            if not fitted.mle_retvals.get("converged", True):
                raise RuntimeError(f"SARIMAX optimization did not converge: P{point+1}, {channel}")
            row.append(np.asarray(fitted.params, dtype=np.float64))
            print(f"[fit] P{point+1} channel={channel}: {fitted.params}", flush=True)
        result.append(row)
    return np.stack([np.stack(row) for row in result])


def forecast_sarimax(values, origins, horizon, params, fit_start, fit_end,
                     order, seasonal_order, trend):
    """Rolling-origin conditional forecasts with fixed train-fitted parameters."""
    origins = np.asarray(origins, dtype=np.int64)
    if len(origins) == 0:
        raise ValueError("No forecast origins")
    if np.any(np.diff(origins) <= 0) or origins[0] < fit_end - 1:
        raise ValueError("Origins must be ordered and not precede fitted interval")
    output = np.full((len(origins), values.shape[1], 2, horizon), np.nan, dtype=np.float64)
    for point in range(values.shape[1]):
        for channel in range(2):
            history = np.asarray(values[fit_start:fit_end, point, channel], dtype=np.float64)
            model = _sarimax_model(history, order, seasonal_order, trend)
            state = model.filter(params[point, channel])
            next_to_process = fit_end
            for j, origin in enumerate(origins):
                if origin >= next_to_process:
                    observed = np.asarray(values[next_to_process:origin+1, point, channel],
                                          dtype=np.float64)
                    state = state.extend(observed)
                    next_to_process = int(origin) + 1
                output[j, point, channel] = np.asarray(
                    state.forecast(steps=horizon), dtype=np.float64)
            print(f"[forecast] SARIMA P{point+1} channel={channel} completed", flush=True)
    return output
