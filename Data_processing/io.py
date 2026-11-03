"""Load time-aligned ocean fields and masks from supported historical layouts."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

GRID_SHAPE = (161, 181)
BASE_DATE = date(1979, 1, 1)
VARIABLE_DIRS = {
    "sensible": "Fluxes", "latent": "Fluxes", "Flux": "Fluxes",
    "SST": "SST", "Pressure": "Pressure",
}


def read_ocean_mask(files_path_prefix, spatial_shape=GRID_SHAPE):
    """Read a binary ocean mask, where True denotes an ocean grid cell.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory containing DATA/mask or mask.
    spatial_shape : tuple of int
        Spatial grid dimensions (height, width).

    Returns
    -------
    numpy.ndarray
        Boolean mask with shape ``spatial_shape``.
    """
    root = Path(files_path_prefix)
    path = next((p for p in (root / "DATA" / "mask", root / "mask") if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing ocean mask under {root}")
    size = int(np.prod(spatial_shape))
    raw = path.read_bytes()
    if len(raw) < size:
        raise ValueError(f"Mask is shorter than {size} bytes: {path}")
    return np.frombuffer(raw[:size], dtype=np.bool_).reshape(spatial_shape).copy()


def field_candidates(root, name, start_year, end_year):
    """Return supported input paths without assuming a single naming convention.

    Parameters
    ----------
    root : str or Path
        Root directory of the oceanographic data.
    name : str
        Variable name: sensible, latent, Flux, SST, or Pressure.
    start_year, end_year : int
        Half-open period identifiers used in input filenames.

    Returns
    -------
    list of Path
        Candidate paths in priority order.
    """
    if name not in VARIABLE_DIRS:
        raise ValueError(f"Unknown variable: {name}")
    folder = VARIABLE_DIRS[name]
    period = f"{start_year}-{end_year}"
    aliases = ("PRESS", "Pressure") if name == "Pressure" else (name,)
    return [Path(root) / base / folder / pattern.format(name=alias, period=period)
            for base in ("DATA", "") for alias in aliases
            for pattern in ("{name}_{period}_grouped.npy", "{name}_grouped_{period}.npy")]


def load_field(files_path_prefix, name, start_year, end_year, spatial_shape=GRID_SHAPE):
    """Return a named field as (time, height, width) without copying unnecessarily.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root directory.
    name : str
        Variable identifier.
    start_year, end_year : int
        Period used by existing array filenames.
    spatial_shape : tuple of int
        Expected spatial grid size.

    Returns
    -------
    numpy.ndarray
        Chronologically ordered field with shape (T, H, W).
    """
    from Coefficients.common import coerce_series

    candidates = field_candidates(files_path_prefix, name, start_year, end_year)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("Missing %s (%s-%s); checked %s" %
                                (name, start_year, end_year, ", ".join(map(str, candidates))))
    return coerce_series(np.load(path, mmap_mode="r", allow_pickle=False), spatial_shape)


def load_spectral_field(files_path_prefix, name, start_year, end_year, spatial_shape=GRID_SHAPE):
    """Load a field as (space, time) for the existing spectral estimators.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root directory.
    name : str
        Variable identifier.
    start_year, end_year : int
        Requested period.
    spatial_shape : tuple of int
        Grid shape.

    Returns
    -------
    numpy.ndarray
        Field with shape (H * W, T).
    """
    return load_field(files_path_prefix, name, start_year, end_year, spatial_shape).reshape(
        -1, *spatial_shape).reshape((-1, int(np.prod(spatial_shape)))).T


def start_label(start_year):
    """Return the one-based observation index of January 1 in start_year.

    Parameters
    ----------
    start_year : int
        Calendar year.

    Returns
    -------
    int
        One-based index relative to 1979-01-01.
    """
    return (date(start_year, 1, 1) - BASE_DATE).days + 1


def dates_from_labels(first_label, count, convention="end"):
    """Generate calendar dates for coefficient or spectral transition labels.

    Parameters
    ----------
    first_label : int
        First absolute one-based transition label.
    count : int
        Number of consecutive transitions.
    convention : {'end', 'start'}
        The A/B estimator uses the endpoint; spectra use the start point.

    Returns
    -------
    list of datetime.date
        One date per transition.
    """
    if convention not in ("end", "start") or first_label < 1 or count < 0:
        raise ValueError("Invalid labeling convention, label, or count")
    return [BASE_DATE + timedelta(days=first_label + i - 1) for i in range(count)]
