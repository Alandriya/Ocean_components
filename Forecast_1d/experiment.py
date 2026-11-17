"""Forecast horizon and artifact layout shared by training and CPU-only reporting."""
from pathlib import Path

DEFAULT_HORIZON = 14
LEGACY_HORIZON = 3


def horizon_root(base, horizon=DEFAULT_HORIZON):
    """Keep original 3-day artifacts intact; isolate each other horizon."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")
    base = Path(base)
    return base if horizon == LEGACY_HORIZON else base / f"h{horizon}"
