"""Validated access to the extracted ten-point daily flux dataset.

All forecast splits are chronological. Evaluation origins may use past observed
values as inputs, but every forecast target lies inside its requested split.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .experiment import DEFAULT_HORIZON

DEFAULT_SUBDIR = "atlantic10_native_1979-2026"
CHANNELS = ("sensible", "latent")


def data_root(override=None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    from project_paths import DATA_ROOT
    return Path(DATA_ROOT).expanduser().resolve()


def dataset_dir(override=None, root_override=None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    return data_root(root_override) / "DATA" / "Forecast_1d" / DEFAULT_SUBDIR


def results_dir(override=None, root_override=None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    return data_root(root_override) / "Forecast" / "Results" / "Forecast_1d"


@dataclass(frozen=True)
class Splits:
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    in_len: int
    out_len: int

    def slice_for(self, split: str):
        if split == "val":
            return self.val_start, self.val_end
        if split == "test":
            return self.test_start, self.test_end
        raise ValueError(f"Unsupported split: {split}")

    def origins(self, split: str) -> np.ndarray:
        start, end = self.slice_for(split)
        # Every horizon target must be inside this split.
        first_origin = max(start - 1, self.in_len - 1)
        return np.arange(first_origin, end - self.out_len, dtype=np.int64)


def split_dates(n_dates: int, *, val_days: int = 730, test_days: int = 365,
                in_len: int = 30, out_len: int = DEFAULT_HORIZON) -> Splits:
    if min(val_days, test_days, in_len, out_len) < 1:
        raise ValueError("Split lengths and history/horizon must be positive")
    train_end = n_dates - val_days - test_days
    if train_end <= in_len + out_len + 10:
        raise ValueError(f"Not enough train observations: {train_end}")
    if min(val_days, test_days) < out_len:
        raise ValueError("Evaluation splits must be at least as long as the horizon")
    return Splits(train_end, train_end, n_dates - test_days,
                  n_dates - test_days, n_dates, in_len, out_len)


class PointDataset:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        centers_path = self.path / "centers.npy"
        dates_path = self.path / "dates.npy"
        if not centers_path.is_file() or not dates_path.is_file():
            raise FileNotFoundError(f"Expected centers.npy and dates.npy in {self.path}")
        self.values = np.load(centers_path, mmap_mode="r", allow_pickle=False)
        self.dates = np.load(dates_path, allow_pickle=False)
        if self.values.ndim != 3 or self.values.shape[1:] != (10, 2):
            raise ValueError(f"Expected centers [T,10,2]; got {self.values.shape}")
        if self.values.dtype.kind not in "fi":
            raise TypeError("centers must contain numeric values")
        if self.dates.shape != (self.values.shape[0],):
            raise ValueError("Date count does not match center series length")
        if self.dates.dtype.kind != "M":
            raise TypeError("dates.npy must contain numpy datetime64 values")
        self.dates = self.dates.astype("datetime64[D]")
        if not np.all(np.diff(self.dates).astype("timedelta64[D]").astype(int) == 1):
            raise ValueError("Expected consecutive daily timestamps with no gaps")
        points_file = self.path / "points.json"
        if not points_file.is_file():
            raise FileNotFoundError(f"Missing points.json in {self.path}")
        metadata = json.loads(points_file.read_text(encoding="utf-8"))
        self.points = metadata.get("points", [])
        if [p["id"] for p in self.points] != [f"P{i}" for i in range(1, 11)]:
            raise ValueError("Point IDs/order do not match P1..P10")
        self.n_dates = self.values.shape[0]

    def fingerprint(self) -> str:
        """Bind checkpoints to exact extracted center values and dates."""
        hasher = hashlib.sha256()
        for item in (self.values, self.dates):
            hasher.update(str(item.shape).encode("utf-8"))
            hasher.update(np.asarray(item).tobytes(order="C"))
        hasher.update(json.dumps(self.points, sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()

    def get_splits(self, val_days=730, test_days=365, in_len=30, out_len=DEFAULT_HORIZON):
        return split_dates(self.n_dates, val_days=val_days, test_days=test_days,
                           in_len=in_len, out_len=out_len)
