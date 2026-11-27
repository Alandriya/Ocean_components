"""Shared, strictly causal multivariate Forecast_1d experiment helpers.

Does not modify legacy PointDataset or its checkpoints.  Both train_features and
 test_features use the same original *calendar* split boundaries, even if the
ERA5-augmented data was trimmed by one or more final days.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import PointDataset, dataset_dir, results_dir
from .feature_data import PointHistoryFeatures
from Forecast_1d.models.baselines import calendar_index, fit_seasonal
from .experiment import DEFAULT_HORIZON

MODEL_NAMES = ("mlp", "tcn", "gru", "nbeats", "nhits", "tide")
FEATURE_SETS = ("flux", "era5", "all")
TARGET_MODES = ("anomaly", "raw")


def features_dir(override=None, flux_dir=None) -> Path:
    """Compute features dir for feature experiment from ``override``, ``flux_dir``."""
    return (Path(override).expanduser().resolve() if override is not None
            else (dataset_dir(flux_dir) / "features_era5"))


def checksum_features(folder: Path) -> str:
    """Hash feature content, date index, ordered sites and feature metadata."""
    digest = hashlib.sha256()
    for name in ("features.npy", "targets.npy", "dates.npy", "points.json", "manifest.json"):
        path = folder / name
        digest.update(name.encode("utf-8"))
        with path.open("rb") as stream:
            while True:
                block = stream.read(2 ** 20)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TimeSplits:
    """Container for time splits state and associated operations."""
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    in_len: int
    out_len: int

    def origins(self, split: str) -> np.ndarray:
        """Return eligible forecast origins within the selected split."""
        if split == "train":
            first, last = self.in_len - 1, self.train_end
        elif split == "val":
            first, last = max(self.in_len - 1, self.val_start - 1), self.val_end
        elif split == "test":
            first, last = max(self.in_len - 1, self.test_start - 1), self.test_end
        else:
            raise ValueError(f"Unknown split {split!r}")
        # Forecast dates origin+1..origin+out_len must all lie in split.
        return np.arange(first, last - self.out_len, dtype=np.int64)


class FeatureExperiment:
    """Ensure original flux and ERA5 augmented values share the same daily index."""
    def __init__(self, *, features_directory=None, flux_directory=None,
                 in_len=60, out_len=DEFAULT_HORIZON, val_days=730, test_days=365):
        """Initialize Feature Experiment with ``features_directory``, ``flux_directory``, ``in_len``, ``out_len``, ``val_days``, ``test_days``."""
        self.flux = PointDataset(dataset_dir(flux_directory))
        self.features = PointHistoryFeatures(features_dir(features_directory, flux_directory))
        f, b = self.features, self.flux
        if f.n_points != b.n_points or f.point_ids != b.point_ids:
            raise ValueError("Feature point IDs/order differ from original flux dataset")
        feature_sites = json.loads((f.path / "points.json").read_text(encoding="utf-8"))["points"]
        for site_new, site_old in zip(feature_sites, b.points):
            for field in ("id", "grid_y", "grid_x", "lat_deg", "lon_deg"):
                if field in site_new or field in site_old:
                    if site_new.get(field) != site_old.get(field):
                        raise ValueError(f"Feature site mismatch for {site_new.get('id')}: {field}")
        if f.n_dates > b.n_dates or not np.array_equal(f.dates, b.dates[:f.n_dates]):
            raise ValueError("ERA5 feature dates are not an exact prefix of the original daily flux dates")
        if not np.array_equal(f.targets, b.values[:f.n_dates], equal_nan=True):
            raise ValueError("ERA5 targets differ from original flux centers. Refuse to train on shifted data")
        expected_prefix = ("sensible", "latent", "sst", "sp", "u10", "v10", "t2m", "d2m")
        if f.feature_names[:len(expected_prefix)] != expected_prefix or f.n_features != 20:
            raise ValueError("Expected 20 channels starting with sensible/latent/sst/sp/u10/v10/t2m/d2m")
        s = b.get_splits(val_days=val_days, test_days=test_days,
                         in_len=in_len, out_len=out_len)
        if f.n_dates < s.val_end:
            raise ValueError("ERA5 history ends before the original validation period")
        self.split = TimeSplits(s.train_end, s.val_start, s.val_end,
                                s.test_start, min(s.test_end, f.n_dates), in_len, out_len)
        if len(self.split.origins("test")) == 0:
            raise ValueError("ERA5 history is too short for any full test forecast")
        self.hash = checksum_features(f.path)
        self.n_dates = f.n_dates
        self.point_ids = f.point_ids

    def selected_names(self, feature_set: str):
        """Return the names of selected features."""
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {feature_set}")
        n = {"flux": 2, "era5": 8, "all": 20}[feature_set]
        return tuple(self.features.feature_names[:n])

    def selected_indices(self, feature_set: str):
        """Return the indices of selected features."""
        return np.arange(len(self.selected_names(feature_set)), dtype=np.int64)


def base_folder(*, results_override, target_mode, feature_set, run_id, out_len):
    """Compute base folder for feature experiment from ``results_override``, ``target_mode``, ``feature_set``, ``run_id``, ``out_len``."""
    if any(ch in run_id for ch in ("/", "\\", "..")) or not run_id.strip():
        raise ValueError("run-id must be a simple nonempty directory name")
    if target_mode not in TARGET_MODES or feature_set not in FEATURE_SETS:
        raise ValueError("Invalid target-mode or feature-set")
    root = results_dir(results_override)
    return root / f"h{out_len}" / target_mode / "multivar" / feature_set / run_id


def set_seed(seed):
    """Seed the random number generators used by the experiment."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_device(choice):
    """Resolve the PyTorch device requested for the experiment."""
    import torch
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot find a GPU")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PointTransform:
    """Training-only climatology and separate x/y normalization per location."""
    def __init__(self, exp: FeatureExperiment, point_idx: int, feature_set: str,
                 target_mode: str, climatology_radius: int):
        """Initialize Point Transform with ``exp``, ``point_idx``, ``feature_set``, ``target_mode``, ``climatology_radius``."""
        self.exp = exp
        self.point_idx = int(point_idx)
        self.feature_names = exp.selected_names(feature_set)
        indices = exp.selected_indices(feature_set)
        split = exp.split
        # First choose the point, then the ordered channel subset.
        self.x_raw = np.asarray(exp.features.features[:, point_idx, :][:, indices], dtype=np.float64)
        self.y_raw = np.asarray(exp.features.targets[:, point_idx, :], dtype=np.float64)
        if self.x_raw.shape != (exp.n_dates, len(indices)):
            raise ValueError("Invalid feature orientation")
        self.mode = target_mode
        self.climatology_radius = climatology_radius
        self.climatology = None
        self.doy = calendar_index(exp.features.dates)
        self.x_work = self.x_raw.copy()
        self.y_work = self.y_raw.copy()
        if target_mode == "anomaly":
            if not 0 <= climatology_radius < 183:
                raise ValueError("climatology_radius must be in [0,182]")
            climate, _ = fit_seasonal(exp.features.targets[:, point_idx:point_idx+1, :],
                                      exp.features.dates, split.train_end,
                                      radius=climatology_radius)
            self.climatology = climate[:, 0, :]
            daily = self.climatology[self.doy]
            self.x_work[:, :2] -= daily
            self.y_work -= daily
        elif target_mode != "raw":
            raise ValueError("Unsupported target mode")
        if not np.isfinite(self.x_work).all() or not np.isfinite(self.y_work).all():
            raise ValueError(f"Nonfinite values at {exp.point_ids[point_idx]}; no silent imputation")
        self.x_mean = self.x_work[:split.train_end].mean(axis=0)
        self.x_scale = self.x_work[:split.train_end].std(axis=0)
        self.y_mean = self.y_work[:split.train_end].mean(axis=0)
        self.y_scale = self.y_work[:split.train_end].std(axis=0)
        self.x_scale = np.where(self.x_scale < 1e-8, 1., self.x_scale)
        self.y_scale = np.where(self.y_scale < 1e-8, 1., self.y_scale)
        self.x_norm = ((self.x_work - self.x_mean) / self.x_scale).astype(np.float32)
        self.y_norm = ((self.y_work - self.y_mean) / self.y_scale).astype(np.float32)

    def windows(self, origins: np.ndarray, in_len: int, out_len: int):
        """Construct history and target windows for the supplied forecast origins."""
        origins = np.asarray(origins, dtype=np.int64)
        if origins.size == 0 or origins.min() < in_len-1 or origins.max()+out_len >= self.exp.n_dates:
            raise ValueError("Invalid forecast origins for feature history")
        x_ind = origins[:, None] - np.arange(in_len-1, -1, -1, dtype=np.int64)[None, :]
        y_ind = origins[:, None] + np.arange(1, out_len+1, dtype=np.int64)[None, :]
        return self.x_norm[x_ind], self.y_norm[y_ind]

    def unscale(self, normalized_predictions: np.ndarray, origins: np.ndarray):
        """Predictions: [n_origins,horizon,2], return original flux units."""
        pred = np.asarray(normalized_predictions, dtype=np.float64)
        y = pred * self.y_scale[None, None, :] + self.y_mean[None, None, :]
        if self.mode == "anomaly":
            future = np.asarray(origins)[:, None] + np.arange(1, y.shape[1]+1)[None, :]
            y += self.climatology[self.doy[future]]
        return y

    def metadata(self):
        """Return metadata describing the current dataset or experiment."""
        return {
            "feature_names": list(self.feature_names),
            "target_mode": self.mode,
            "climatology_radius": self.climatology_radius if self.mode == "anomaly" else None,
            "x_mean": self.x_mean.tolist(), "x_scale": self.x_scale.tolist(),
            "y_mean": self.y_mean.tolist(), "y_scale": self.y_scale.tolist(),
        }

    def validate_metadata(self, saved):
        """Validate saved metadata against the current configuration."""
        a = self.metadata()
        for key in ("feature_names", "target_mode", "climatology_radius"):
            if saved[key] != a[key]:
                raise ValueError(f"Transform mismatch: {key}")
        for key in ("x_mean", "x_scale", "y_mean", "y_scale"):
            if not np.allclose(saved[key], a[key], rtol=1e-8, atol=1e-8):
                raise ValueError(f"Training-only normalization mismatch: {key}")


def save_json(path, obj):
    """Serialize the supplied result to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n",encoding="utf-8")
