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
    return {
        "n": n, "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(delta))),
        "bias": float(delta.mean()), "mse": mse,
    }


def _add_skill(score, reference):
    score["persistence_mse_skill"] = (
        float(1.0 - score["mse"] / reference["mse"])
        if score["mse"] is not None and reference["mse"] not in (None, 0)
        else None
    )
    return score


def score_predictions(pred, target, *, baseline=None, shared_mask=None,
                      point_ids=None):
    """Score [origins, selected_points, two_channels, horizon] forecasts."""
    if (pred.shape != target.shape or pred.ndim != 4 or
            pred.shape[2] != 2):
        raise ValueError(
            "Forecast and targets must share [origins,points,2_channels,horizon]"
        )
    n_points = pred.shape[1]
    point_ids = list(point_ids or [f"P{i+1}" for i in range(n_points)])
    if len(point_ids) != n_points:
        raise ValueError("point_ids length does not match forecast point dimension")
    common_mask = np.isfinite(target) & np.isfinite(pred)
    if shared_mask is not None:
        if shared_mask.shape != pred.shape:
            raise ValueError("Shared evaluation mask has incorrect shape")
        common_mask &= shared_mask
    if baseline is not None:
        if baseline.shape != pred.shape:
            raise ValueError("Baseline shape mismatch")
        common_mask &= np.isfinite(baseline)

    overall = {}
    for h in range(pred.shape[-1]):
        joint = _scores(pred[..., h], target[..., h], common_mask[..., h])
        channels = {}
        for c, channel_name in enumerate(CHANNELS):
            value = _scores(
                pred[:, :, c, h], target[:, :, c, h],
                common_mask[:, :, c, h]
            )
            if baseline is not None:
                reference = _scores(
                    baseline[:, :, c, h], target[:, :, c, h],
                    common_mask[:, :, c, h]
                )
                _add_skill(value, reference)
            channels[channel_name] = value

        points = {}
        for p, point_id in enumerate(point_ids):
            points[point_id] = {}
            for c, channel_name in enumerate(CHANNELS):
                value = _scores(
                    pred[:, p, c, h], target[:, p, c, h],
                    common_mask[:, p, c, h]
                )
                if baseline is not None:
                    reference = _scores(
                        baseline[:, p, c, h], target[:, p, c, h],
                        common_mask[:, p, c, h]
                    )
                    _add_skill(value, reference)
                points[point_id][channel_name] = value

        result = {"joint": joint, "channels": channels, "points": points}
        if baseline is not None:
            reference = _scores(
                baseline[..., h], target[..., h], common_mask[..., h]
            )
            _add_skill(joint, reference)
            result["persistence_mse_skill"] = joint["persistence_mse_skill"]
        overall[str(h + 1)] = result
    return overall


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8"
    )
