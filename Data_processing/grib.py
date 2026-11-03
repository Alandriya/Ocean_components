"""Explicit GRIB conversion and yearly splitting with no import-time operations."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np


def convert_grib(files_path_prefix, grib_file, variable, output_file, *, mask=None,
                 spatial_stride=2):
    """Read a GRIB variable, optionally downsample it, and save (space, time).

    Parameters
    ----------
    files_path_prefix : str or Path
        Root data directory.
    grib_file : str or Path
        GRIB filename relative to the root unless absolute.
    variable : str
        Variable short name in the dataset, for example sshf or sst.
    output_file : str or Path
        Destination NPY filename under the root unless absolute.
    mask : array_like, optional
        Boolean ocean mask in spatial coordinates.
    spatial_stride : int
        Subsampling factor for spatial dimensions.

    Returns
    -------
    pathlib.Path
        Saved NPY file path.
    """
    import xarray as xr

    root = Path(files_path_prefix)
    source = Path(grib_file)
    target = Path(output_file)
    source = source if source.is_absolute() else root / source
    target = target if target.is_absolute() else root / target
    with xr.open_dataset(source, engine="cfgrib") as dataset:
        if variable not in dataset:
            raise KeyError(f"GRIB dataset has no variable {variable}")
        data = np.asarray(dataset[variable].data)
        if data.ndim < 3:
            raise ValueError("Expected time and two spatial dimensions")
        data = data[..., ::spatial_stride, ::spatial_stride]
        data = data.reshape((-1, *data.shape[-2:])).reshape((-1, int(np.prod(data.shape[-2:])))).T
        if mask is not None:
            ocean = np.asarray(mask, dtype=bool).reshape(-1)
            if ocean.size != data.shape[0]:
                raise ValueError("Mask does not match downsampled GRIB grid")
            data[~ocean] = np.nan
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, data)
    return target


def split_years(files_path_prefix, source_file, variable, year_ranges, *, samples_per_day=4):
    """Split a flattened multi-year time series into annual files.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root data directory.
    source_file : str or Path
        Input NPY with shape (space, time).
    variable : str
        Stem used in output filenames.
    year_ranges : sequence of (date, date)
        Chronologically adjacent, half-open intervals.
    samples_per_day : int
        Number of temporal samples stored for each day.

    Returns
    -------
    list of Path
        Output files in the DATA directory.
    """
    root = Path(files_path_prefix)
    path = Path(source_file)
    data = np.load(path if path.is_absolute() else root / path, mmap_mode="r")
    if data.ndim != 2:
        raise ValueError("Expected array shape (space, time)")
    outputs = []
    offset = 0
    for first, last in year_ranges:
        length = (last - first).days * samples_per_day
        if length <= 0 or offset + length > data.shape[1]:
            raise ValueError("Invalid year interval or insufficient input data")
        target = root / "DATA" / f"{variable}_{first.year}-{last.year}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, data[:, offset:offset + length])
        offset += length
        outputs.append(target)
    if offset != data.shape[1]:
        raise ValueError("Unused input samples remain after splitting year ranges")
    return outputs
