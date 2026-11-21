"""Forecast horizon and artifact layout shared by training and CPU-only reporting."""
from pathlib import Path

DEFAULT_HORIZON = 14
LEGACY_HORIZON = 3
DEFAULT_POINTS = tuple(f"P{i}" for i in range(1, 16))
LEGACY_POINTS = DEFAULT_POINTS[:10]
DEFAULT_SUBDIR = "atlantic15_native_1979-2026"
DEFAULT_RESULTS_SUBDIR = "atlantic15"


def select_points(available, requested=None):
    """Default to every point in the loaded dataset, not a fixed subset."""
    chosen = tuple(available if requested is None else requested)
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("Select a nonempty list of unique point IDs")
    if not set(chosen).issubset(available):
        raise ValueError(f"Unknown point IDs: {sorted(set(chosen) - set(available))}")
    return chosen


def horizon_root(base, horizon=DEFAULT_HORIZON):
    """Keep original 3-day artifacts intact; isolate each other horizon."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")
    base = Path(base)
    return base if horizon == LEGACY_HORIZON else base / f"h{horizon}"


TARGET_MODES = ("raw", "anomaly")
DEFAULT_TARGET_MODE = "raw"


def target_root(base, horizon=DEFAULT_HORIZON, target_mode=DEFAULT_TARGET_MODE):
    """Return an experiment root; raw paths are backward-compatible."""
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    root = horizon_root(base, horizon)
    return root if target_mode == "raw" else root / "anomaly"
