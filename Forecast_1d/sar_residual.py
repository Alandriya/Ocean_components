"""Train-only Seasonal AR(1) residual targets with expanding out-of-fold fits.

This module uses only NumPy and the existing seasonal-statistics implementation.
At every OOF fold, the SAR fit ends before the first predicted target date;
validation and test predictions use the full training-only fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models.seasonal import calendar_index, fit_seasonal, forecast_seasonal


DEFAULT_OOF_FOLDS = 4
DEFAULT_BURN_FRACTION = 0.33


@dataclass
class ResidualTraining:
    x_train: np.ndarray             # [N,history,2], normalized anomaly inputs
    y_train: np.ndarray             # [N,horizon,2], normalized SAR corrections
    x_val: np.ndarray               # [V,history,2], normalized anomaly inputs
    y_val: np.ndarray               # [V,horizon,2], normalized SAR corrections
    input_mean: np.ndarray          # [2], fitted on train-only anomalies
    input_scale: np.ndarray         # [2]
    residual_scale: np.ndarray      # [2], OOF correction std, no centering
    fold_edges: tuple[int, ...]
    train_origins: np.ndarray
    val_origins: np.ndarray


def fitted_sar(values: np.ndarray, dates: np.ndarray, end: int, radius: int):
    """Fit climatology and anomaly persistence strictly before `end`."""
    if not 0 <= radius < 183:
        raise ValueError('climatology radius must be in 0..182')
    climate, phi = fit_seasonal(values[:, None, :], dates, end, radius=radius)
    return climate, phi


def sar_prediction(values: np.ndarray, dates: np.ndarray, origins: np.ndarray,
                   horizon: int, climate: np.ndarray, phi: np.ndarray):
    """Shape [N,H,2], computed without future observed flux values."""
    return forecast_seasonal(
        values[:, None, :], dates, origins, horizon, climate, phi
    )[:, 0, :, :].transpose(0, 2, 1)


def anomaly_inputs(values: np.ndarray, dates: np.ndarray, origins: np.ndarray,
                   in_len: int, climate: np.ndarray):
    """Historical, train-fitted anomalies [N,L,2]; no future values."""
    dates_index = calendar_index(dates)
    climate_daily = climate[:, 0, :][dates_index]
    rows = np.stack([
        np.asarray(values[int(t)-in_len+1:int(t)+1], dtype=np.float64)
        - climate_daily[int(t)-in_len+1:int(t)+1]
        for t in origins
    ])
    return rows


def future_targets(values: np.ndarray, origins: np.ndarray, horizon: int):
    return np.stack([
        np.asarray(values[int(t)+1:int(t)+horizon+1], dtype=np.float64)
        for t in origins
    ])


def fold_edges(train_end: int, in_len: int, horizon: int, n_folds: int):
    """Reserve the initial third of train for the first SAR fit."""
    if n_folds < 1:
        raise ValueError('OOF fold count must be positive')
    first = max(in_len + horizon + 10, int(train_end * DEFAULT_BURN_FRACTION))
    if train_end - first < n_folds * (horizon + 2):
        raise ValueError('Training period too short for OOF SAR residual folds')
    edges = np.linspace(first, train_end, n_folds + 1, dtype=np.int64)
    if np.any(np.diff(edges) < horizon + 2):
        raise ValueError('OOF fold width must exceed the forecast horizon')
    return tuple(int(x) for x in edges)


def prepare_residual_training(values: np.ndarray, dates: np.ndarray,
                              train_end: int, val_origins: np.ndarray,
                              in_len: int, horizon: int, radius: int = 15,
                              n_folds: int = DEFAULT_OOF_FOLDS) -> ResidualTraining:
    """Build OOF train targets and full-train-fit validation targets.

    The SAR model never sees the target observations used to supervise an OOF
    correction. Train inputs are anomaly histories relative to each fold's
    past-only climatology. Mean/scale are fitted on train-only full-fit anomalies.
    """
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 2 or series.shape[1] != 2:
        raise ValueError(f'Expected [T,2] flux series; got {series.shape}')
    if len(dates) != len(series) or train_end <= 0 or train_end > len(series):
        raise ValueError('Invalid date count or training boundary')
    val_origins = np.asarray(val_origins, dtype=np.int64)
    if not len(val_origins) or np.any(val_origins < train_end - 1):
        raise ValueError('Validation origins must begin at the train/validation boundary')
    if np.any(val_origins + horizon >= len(series)):
        raise ValueError('Validation forecast targets exceed the dataset')

    full_climate, full_phi = fitted_sar(series, dates, train_end, radius)
    full_anomalies = series[:train_end] - full_climate[:, 0, :][calendar_index(dates[:train_end])]
    finite = np.where(np.isfinite(full_anomalies), full_anomalies, np.nan)
    mean = np.nanmean(finite, axis=0)
    scale = np.nanstd(finite, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Training anomalies lack finite values')
    scale = np.where(scale < 1e-8, 1.0, scale)

    fold_boundaries = fold_edges(train_end, in_len, horizon, n_folds)
    x_parts, target_parts, origin_parts = [], [], []
    for start, stop in zip(fold_boundaries[:-1], fold_boundaries[1:]):
        # First target is at `start`, last target strictly before `stop`.
        origin = np.arange(max(start-1, in_len-1), stop-horizon, dtype=np.int64)
        if not len(origin):
            continue
        climate, phi = fitted_sar(series, dates, start, radius)
        historical = anomaly_inputs(series, dates, origin, in_len, climate)
        base = sar_prediction(series, dates, origin, horizon, climate, phi)
        residual = future_targets(series, origin, horizon) - base
        # Never impute or silently truncate a nonfinite correction target.
        good = np.isfinite(residual).all(axis=(1, 2))
        if not np.any(good):
            continue
        x_parts.append(historical[good])
        target_parts.append(residual[good])
        origin_parts.append(origin[good])
    if not x_parts:
        raise ValueError('No complete OOF SAR-residual training windows')
    train_x = np.concatenate(x_parts)
    train_residual = np.concatenate(target_parts)
    train_origin = np.concatenate(origin_parts)
    # Zero correction is exactly the SAR baseline: scale, but do not center y.
    residual_scale = np.std(train_residual, axis=(0, 1))
    if not np.isfinite(residual_scale).all():
        raise ValueError('OOF residual scales are nonfinite')
    residual_scale = np.where(residual_scale < 1e-8, 1.0, residual_scale)

    validation_x = anomaly_inputs(series, dates, val_origins, in_len, full_climate)
    validation_residual = (future_targets(series, val_origins, horizon)
                           - sar_prediction(series, dates, val_origins, horizon,
                                            full_climate, full_phi))
    if not np.isfinite(validation_residual).all():
        raise ValueError('SAR-residual validation requires complete finite targets and origin observations')
    # Inputs may contain missing past values, imputed using training-only means.
    train_x = np.where(np.isfinite(train_x), train_x, mean)
    validation_x = np.where(np.isfinite(validation_x), validation_x, mean)
    return ResidualTraining(
        x_train=((train_x - mean) / scale).astype(np.float32),
        y_train=(train_residual / residual_scale).astype(np.float32),
        x_val=((validation_x - mean) / scale).astype(np.float32),
        y_val=(validation_residual / residual_scale).astype(np.float32),
        input_mean=mean, input_scale=scale,
        residual_scale=residual_scale,
        fold_edges=fold_boundaries,
        train_origins=train_origin, val_origins=val_origins
    )


def zero_output_head(model, name: str):
    """Guarantee an initial identity correction for the three Level-1 networks."""
    import torch.nn as nn
    if name == 'mlp':
        head = model.network[-1]
    elif name in ('tcn', 'gru'):
        head = model.head
    else:
        raise ValueError('SAR-residual currently supports only mlp, tcn, gru')
    if not isinstance(head, nn.Linear):
        raise TypeError(f'Unexpected final layer of {name}: {type(head).__name__}')
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
