"""Opt-in access to aligned ERA5-augmented Forecast_1d history.

Does NOT modify the legacy PointDataset or existing checkpoints.
Use causal window(point, origin, in_len, out_len) to prevent target leakage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class PointHistoryFeatures:
    """Container for point history features state and associated operations."""
    def __init__(self, directory):
        """Initialize Point History Features with ``directory``."""
        self.path = Path(directory).expanduser().resolve()
        self.features = np.load(self.path / "features.npy", mmap_mode="r", allow_pickle=False)
        self.targets = np.load(self.path / "targets.npy", mmap_mode="r", allow_pickle=False)
        self.dates = np.load(self.path / "dates.npy", allow_pickle=False).astype("datetime64[D]")
        manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.feature_names = tuple(manifest["feature_names"])
        self.target_names = tuple(manifest["target_names"])
        points = json.loads((self.path / "points.json").read_text(encoding="utf-8"))["points"]
        self.point_ids = tuple(p["id"] for p in points)
        if self.features.ndim != 3:
            raise ValueError("features.npy must be [T,P,F]")
        t, p, f = self.features.shape
        if (self.targets.shape != (t,p,2) or len(self.dates) != t or len(self.feature_names) != f
                or len(self.point_ids) != p):
            raise ValueError("features, targets, dates or metadata have incompatible shapes")
        if not np.array_equal(self.features[..., :2], self.targets, equal_nan=True):
            raise ValueError("First two feature channels must equal sensible/latent targets")
        if len(self.dates) > 1 and not np.all(np.diff(self.dates) == np.timedelta64(1, "D")):
            raise ValueError("Gaps in feature dates")
        self.n_dates, self.n_points, self.n_features = t, p, f

    def point_index(self, point: int | str) -> int:
        """Resolve the index of a configured observation point."""
        if isinstance(point, str):
            if point not in self.point_ids:
                raise KeyError(f"Unknown point {point}; available {self.point_ids}")
            return self.point_ids.index(point)
        if not 0 <= int(point) < self.n_points:
            raise IndexError(f"Point index out of range: {point}")
        return int(point)

    def window(self, point: int | str, origin: int, in_len: int = 60, out_len: int = 14):
        """Input dates <= origin; output targets from origin+1 through origin+H.

        A returned input array has shape [in_len, 20], a target [out_len, 2].
        No future ERA5 feature values are ever returned by this method.
        """
        if min(in_len, out_len) < 1 or origin + out_len >= self.n_dates or origin - in_len + 1 < 0:
            raise ValueError("Invalid origin/history/horizon for available dates")
        p = self.point_index(point)
        x = np.asarray(self.features[origin-in_len+1:origin+1, p], dtype=np.float32)
        y = np.asarray(self.targets[origin+1:origin+out_len+1, p], dtype=np.float32)
        return x, y

    def training_statistics(self, train_end: int):
        """Return per-feature (point-specific) mean/std computed only on train.

        train_end is an exclusive array index from the experiment's chronological
        split; it must NOT include validation/test dates. The returned arrays
        are [P,F] and can be used for (x-mean)/std broadcasting.
        """
        if not 1 < train_end <= self.n_dates:
            raise ValueError("Invalid train_end")
        a = np.asarray(self.features[:train_end], dtype=np.float64)
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite training features; inspect the assembled dataset")
        mu = a.mean(axis=0)
        sigma = a.std(axis=0)
        return mu.astype(np.float32), np.where(sigma > 1e-8, sigma, 1.0).astype(np.float32)
