"""Comparable forecast scores for each lead, point, and physical variable."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import CHANNELS


def _scores(pred, target, mask=None):
    finite = np.isfinite(pred) & np.isfinite(target)
    if mask is not None:
        finite &= mask
    n = int(finite.sum())
    if n == 0:
        return {"n": 0, "rmse": None, "mae": None, "bias": None, "mse": None}
    delta = np.asarray(pred - target, dtype=np.float64)[finite]
    mse = float(np.mean(delta * delta))
    return {"n": n, "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(delta))), "bias": float(delta.mean()),
            "mse": mse}


def score_predictions(pred, target, *, baseline=None, shared_mask=None):
    """Compute scores using a shared finite mask across models, if supplied."""
    if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1:3] != (10, 2):
        raise ValueError("Forecast and targets must share [origins,10,2,horizon]")
    common_mask = np.isfinite(target) & np.isfinite(pred)
    if shared_mask is not None:
        if shared_mask.shape != pred.shape:
            raise ValueError("Shared evaluation mask has incorrect shape")
        common_mask &= shared_mask
    if baseline is not None:
        if baseline.shape != pred.shape:
            raise ValueError("Baseline shape mismatch")
        common_mask &= np.isfinite(baseline)
    horizon = pred.shape[-1]
    overall = {}
    for h in range(horizon):
        at_h = _scores(pred[..., h], target[..., h], common_mask[..., h])
        channels = {
            name: _scores(pred[:, :, c, h], target[:, :, c, h], common_mask[:, :, c, h])
            for c, name in enumerate(CHANNELS)
        }
        points = {
            f"P{p+1}": {
                name: _scores(pred[:, p, c, h], target[:, p, c, h], common_mask[:, p, c, h])
                for c, name in enumerate(CHANNELS)
            }
            for p in range(10)
        }
        result = {"joint": at_h, "channels": channels, "points": points}
        if baseline is not None:
            ref = _scores(baseline[..., h], target[..., h], common_mask[..., h])
            result["persistence_mse_skill"] = (
                float(1 - at_h["mse"] / ref["mse"])
                if ref["mse"] not in (None, 0) else None
            )
        overall[str(h+1)] = result
    return overall


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
