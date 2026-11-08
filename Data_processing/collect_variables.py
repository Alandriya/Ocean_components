"""Join decade-long ocean observations into a single time-first NPY array.

This function is independent of numerical coefficient estimation.  To make it
available via the existing ``Data_processing.data_processing`` API, append this
line to the original data_processing.py (without replacing its other code):

    from Data_processing.collect_variables import collect_variable_periods
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from Data_processing.io import GRID_SHAPE, VARIABLE_DIRS, field_candidates


DEFAULT_PERIODS = (
    (1979, 1989),
    (1989, 1999),
    (1999, 2009),
    (2009, 2019),
    (2019, 2026),
)


def _time_first_view(array: np.ndarray, spatial_shape: tuple[int, int]) -> np.ndarray:
    """Return a time-first view supporting the project's historical NPY layouts."""
    height, width = spatial_shape
    n_cells = height * width
    if array.ndim == 3 and array.shape[1:] == spatial_shape:
        return array
    if array.ndim == 2:
        if array.shape == (n_cells, n_cells):
            raise ValueError(f"Ambiguous spatial/time orientation: {array.shape}")
        if array.shape[0] == n_cells:
            return array.T.reshape(array.shape[1], height, width)
        if array.shape[1] == n_cells:
            return array.reshape(array.shape[0], height, width)
    raise ValueError(
        f"Unsupported array shape {array.shape}; expected (T,{height},{width}), "
        f"({n_cells},T), or (T,{n_cells})"
    )


def collect_variable_periods(
    files_path_prefix: str | Path,
    name: str,
    *,
    periods: tuple[tuple[int, int], ...] = DEFAULT_PERIODS,
    chunk_days: int = 32,
    overwrite: bool = False,
    include_final_year: bool = True,
    spatial_shape: tuple[int, int] = GRID_SHAPE,
) -> Path:
    """Assemble chronological observations of one variable from decade files.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root, e.g. ``D:/Nastya/Data/OceanFull/``.
    name : str
        One of ``sensible``, ``latent``, ``Flux``, ``SST``, ``Pressure``.
    periods : tuple of (start_year, end_year)
        Consecutive half-open source periods; defaults to 1979--2026.
    chunk_days : int
        Number of days written in one block, limiting memory use.
    overwrite : bool
        Replace an existing collected file only when explicitly requested.
    include_final_year : bool
        If True, the last source file may also contain days of its labeled
        ``end_year`` (up to December 31, inclusive). This supports ongoing
        data in ``2019-2026`` without truncation. Set False to require the
        traditional half-open ``[start_year, end_year)`` convention. Only the
        final source is affected; preceding periods must be exact.
    spatial_shape : tuple[int, int]
        Spatial grid; default 161 x 181.

    Returns
    -------
    pathlib.Path
        Path to ``DATA/<folder>/<name>_1979-2026_grouped.npy`` for default periods.

    Notes
    -----
    The result has shape ``(days, height, width)``. Source layouts may be
    ``(days,height,width)``, ``(height*width,days)``, or ``(days,height*width)``.
    Nonfinal periods must contain the exact number of calendar days; the final
    period may stop at the last available observation. By default the final
    source may also include dates within its named end year (e.g. 2026).
    No input arrays are modified or deleted. The collector does not use or need
    A/B coefficients. Dates are inferred from contiguous daily observations.
    """
    root = Path(files_path_prefix)
    if name not in VARIABLE_DIRS:
        raise ValueError(f"Unknown variable {name!r}; choose from {tuple(VARIABLE_DIRS)}")
    if not isinstance(chunk_days, int) or chunk_days < 1:
        raise ValueError("chunk_days must be a positive integer")
    if len(spatial_shape) != 2 or any(not isinstance(n, int) or n < 1 for n in spatial_shape):
        raise ValueError("spatial_shape must have two positive integer dimensions")
    if not isinstance(include_final_year, bool):
        raise TypeError("include_final_year must be True or False")
    if not periods:
        raise ValueError("No source periods specified")

    periods = tuple((int(start), int(end)) for start, end in periods)
    for index, (start, end) in enumerate(periods):
        if end <= start or (index > 0 and start != periods[index - 1][1]):
            raise ValueError(f"Periods must be positive and adjacent: {periods}")

    destination = (root / "DATA" / VARIABLE_DIRS[name] /
                   f"{name}_{periods[0][0]}-{periods[-1][1]}_grouped.npy")

    # Read headers and memory-mapped sources first, to fail before writing.
    sources = []
    for index, (start, end) in enumerate(periods):
        candidates = field_candidates(root, name, start, end)
        source_path = next((p for p in candidates if p.is_file()), None)
        if source_path is None:
            raise FileNotFoundError(
                f"Missing source for {name} {start}-{end}. Checked: " +
                ", ".join(str(candidate) for candidate in candidates)
            )
        raw = np.load(source_path, mmap_mode="r", allow_pickle=False)
        if raw.dtype.kind not in "biuf":
            raise TypeError(f"Non-numeric source dtype {raw.dtype}: {source_path}")
        view = _time_first_view(raw, spatial_shape)
        actual_days = int(view.shape[0])
        calendar_days = (date(end, 1, 1) - date(start, 1, 1)).days
        is_final = (index == len(periods) - 1)
        # A final source named 2019-2026 may already contain 2026 observations.
        # Reject neither those days nor silently truncate the source file.
        max_days = ((date(end + 1, 1, 1) - date(start, 1, 1)).days
                    if is_final and include_final_year else calendar_days)
        if not is_final and actual_days != calendar_days:
            raise ValueError(
                f"{source_path}: {actual_days} days, expected {calendar_days}. "
                "Cannot merge with a possible date gap/overlap."
            )
        if actual_days < 1 or actual_days > max_days:
            raise ValueError(
                f"{source_path}: {actual_days} days outside the allowed "
                f"range 1..{max_days}. The final year is included only "
                f"when include_final_year=True."
            )
        sources.append((source_path, raw, view, actual_days))

    n_days = sum(record[3] for record in sources)
    dtype = np.result_type(*(record[2].dtype for record in sources))
    output_shape = (n_days, *spatial_shape)
    last_start = periods[-1][0]
    last_date = date(last_start, 1, 1) + timedelta(days=sources[-1][3] - 1)

    if destination.exists() and not overwrite:
        old = np.load(destination, mmap_mode="r", allow_pickle=False)
        if old.shape != output_shape or old.dtype != dtype:
            raise ValueError(
                f"Existing file {destination} has shape {old.shape}, dtype {old.dtype}; "
                f"expected {output_shape}, {dtype}. Pass overwrite=True to replace."
            )
        print(f"Reusing existing {name}: {destination} ({n_days} days)", flush=True)
        del old
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Collecting {name}: {n_days} days, {dtype}, {output_shape}", flush=True)
    temp_path: Path | None = None
    output = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{name}_merge_", suffix=".npy", dir=destination.parent
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        output = np.lib.format.open_memmap(
            temp_path, mode="w+", dtype=dtype, shape=output_shape
        )
        offset = 0
        for (start, end), (source_path, raw, view, length) in zip(periods, sources):
            for first in range(0, length, chunk_days):
                last = min(length, first + chunk_days)
                output[offset + first:offset + last] = view[first:last]
            offset += length
            print(f"  {start}-{end}: {length} days <- {source_path.name}", flush=True)
        if offset != n_days:
            raise AssertionError("Incorrect number of copied days")
        output.flush()
        del output
        output = None
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        # Release views held by the open source memmaps before removing temp.
        if output is not None:
            del output
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    metadata = {
        "variable": name,
        "requested_period": f"{periods[0][0]}-{periods[-1][1]}",
        "include_final_year": include_final_year,
        "first_date": date(periods[0][0], 1, 1).isoformat(),
        "last_available_date": last_date.isoformat(),
        "n_days": n_days,
        "shape": list(output_shape),
        "dtype": str(dtype),
        "layout": "time,latitude,longitude",
        "input_periods": [
            {"start_year": start, "end_year": end,
             "days": info[3], "file": str(info[0])}
            for (start, end), info in zip(periods, sources)
        ],
    }
    metadata_path = destination.with_suffix(".json")
    metadata_temp = metadata_path.with_name("." + metadata_path.name + ".tmp")
    try:
        metadata_temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(metadata_temp, metadata_path)
    finally:
        metadata_temp.unlink(missing_ok=True)
    print(f"Saved {destination}; last date {last_date}", flush=True)
    return destination
