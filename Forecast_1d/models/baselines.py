"""Training-free benchmarks using only observed history at forecast origin."""
from __future__ import annotations

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
