"""Stream ERA5 GRIB messages into daily NumPy arrays without loading the full file.

The default output layout is (space, day), matching Ocean_components.
Each UTC date is averaged over four requested validity times (00, 06, 12, 18).
Accumulated heat fluxes are kept in their original J m-2 units unless explicitly
converted to W m-2; a selected ERA5 hourly accumulation represents one hour,
not the six-hour gap between selected observations.

Only regular latitude-longitude grids are supported. Input file order is
irrelevant, and a metadata preflight rejects duplicate or missing timestamps.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


VARIABLES = {
    "avg_ishf": ("Fluxes", "sensible"),
    "avg_ilhf": ("Fluxes", "latent"),
    "sst": ("SST", "SST"),
    "sp": ("Pressure", "Pressure"),

}
FLUX_VARIABLES = frozenset(("avg_ishf", "avg_ilhf"))


def _eccodes():
    """Import ecCodes only when the GRIB reader is actually used."""
    try:
        import eccodes
    except ImportError as exc:
        raise ImportError("Install the GRIB decoder with: python -m pip install eccodes") from exc
    return eccodes


def _get(codes, handle, key, default=None):
    """Return one GRIB key, supplying the default when the key is absent."""
    try:
        return codes.codes_get(handle, key)
    except Exception:
        return default


def _timestamp(codes, handle):
    """Extract the UTC validity timestamp rather than the forecast initial time."""
    day_number = _get(codes, handle, "validityDate")
    hhmm = _get(codes, handle, "validityTime")
    if day_number is None or hhmm is None:
        # Forecast fields can carry a nonzero lead time. Never silently use
        # their initialization time when the validity timestamp is unavailable.
        lead = _get(codes, handle, "step")
        if lead not in (None, 0, "0"):
            raise ValueError("Forecast message has no validityDate/validityTime")
        day_number = _get(codes, handle, "dataDate")
        hhmm = _get(codes, handle, "dataTime")
    if day_number is None or hhmm is None:
        raise ValueError("GRIB message has no usable validity timestamp")
    hour_minute = int(hhmm)
    if hour_minute < 0 or hour_minute > 2359 or hour_minute % 100 >= 60:
        raise ValueError(f"Invalid GRIB validityTime: {hhmm}")
    return datetime.strptime(f"{int(day_number):08d}{hour_minute:04d}", "%Y%m%d%H%M")


def _open_inputs(paths: Iterable[str | Path]):
    """Validate source paths and reject unextracted ZIP and NetCDF files."""
    result = []
    for item in paths:
        source = Path(item).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("rb") as handle:
            signature = handle.read(4)
        if signature[:2] == b"PK":
            raise ValueError(f"{source} is a ZIP archive. Extract the GRIB files first.")
        if signature != b"GRIB":
            raise ValueError(f"{source} is not a GRIB stream (file signature: {signature!r})")
        result.append(source)
    if not result:
        raise ValueError("Provide at least one GRIB file")
    return list(dict.fromkeys(result))


def _iter_messages(codes, sources, *, headers_only: bool):
    """Yield GRIB handles and reliably release each message after processing."""
    for source in sources:
        with source.open("rb") as handle:
            index = 0
            while True:
                grib = codes.codes_grib_new_from_file(handle, headers_only=headers_only)
                if grib is None:
                    break
                index += 1
                try:
                    yield source, index, grib
                finally:
                    codes.codes_release(grib)


def _inspect(sources, variables, hours, first_date, last_date,
             *, skip_incomplete_last_day=True):
    """Preflight coverage; optionally discard only the last incomplete UTC day.

    Parameters
    ----------
    sources : iterable of Path
        GRIB files whose message headers must be inspected.
    variables : iterable of str
        ERA5 short names selected for the conversion.
    hours : iterable of int
        Expected UTC validity hours per date.
    first_date, last_date : date or None
        Inclusive date filters.
    skip_incomplete_last_day : bool
        Omit a partially sampled final date only; fail on interior gaps.

    Returns
    -------
    tuple
        Complete dates, GRIB field specifications, and the omitted date
        description (or None).
    """
    codes = _eccodes()
    observations = {name: {} for name in variables}
    specifications = {}
    counted = 0
    print("ERA5 preflight: scanning GRIB headers without decoding full fields...", flush=True)
    for source, index, grib in _iter_messages(codes, sources, headers_only=True):
        name = str(_get(codes, grib, "shortName", ""))
        if name not in observations:
            continue
        timestamp = _timestamp(codes, grib)
        day = timestamp.date()
        if (first_date and day < first_date) or (last_date and day > last_date):
            continue
        if timestamp.minute != 0 or timestamp.hour not in hours:
            continue
        level_type = str(_get(codes, grib, "typeOfLevel", "surface"))
        if level_type not in ("surface", "meanSea"):
            raise ValueError(f"Unexpected level type for {name}: {level_type}")
        if _get(codes, grib, "number", 0) not in (0, "0"):
            raise ValueError("Ensemble members detected; select deterministic reanalysis only")
        grid_type = str(_get(codes, grib, "gridType", ""))
        if grid_type not in ("regular_ll", "regular_gg"):
            raise ValueError(f"Unsupported gridType={grid_type!r}; expected regular_ll")
        if grid_type == "regular_gg":
            raise ValueError("Gaussian grids must be regridded before processing")
        grid = (
            int(_get(codes, grib, "Ni", 0)),
            int(_get(codes, grib, "Nj", 0)),
            float(_get(codes, grib, "latitudeOfFirstGridPointInDegrees", 999)),
            float(_get(codes, grib, "longitudeOfFirstGridPointInDegrees", 999)),
            float(_get(codes, grib, "latitudeOfLastGridPointInDegrees", 999)),
            float(_get(codes, grib, "longitudeOfLastGridPointInDegrees", 999)),
        )
        if grid[0] < 2 or grid[1] < 2:
            raise ValueError(f"Invalid Ni/Nj for {name}: {grid[:2]}")
        spec = (grid, str(_get(codes, grib, "units", "")),
                str(_get(codes, grib, "stepType", "")))
        if name in specifications and spec != specifications[name]:
            raise ValueError(f"Inconsistent GRIB grid/units/stepType in {source} for {name}")
        specifications[name] = spec
        selected = observations[name].setdefault(day, set())
        if timestamp.hour in selected:
            raise ValueError(
                f"Duplicate record for {name} at {timestamp:%Y-%m-%d %H:%M} UTC "
                f"(source={source.name}, message={index})"
            )
        selected.add(timestamp.hour)
        counted += 1
        if counted % 4000 == 0:
            print(f"ERA5 preflight: indexed {counted:,} selected messages", flush=True)
    if not counted:
        raise ValueError("No requested variables/times were found in the GRIB files")
    for name in variables:
        if not observations[name]:
            raise ValueError(f"Variable {name} is missing. GRIB shortName must match exactly")
    all_days = sorted(set().union(*(set(days) for days in observations.values())))
    required_hours = set(hours)
    skipped_last_day = None
    if skip_incomplete_last_day:
        candidate = all_days[-1]
        missing_last = {
            name: sorted(required_hours - observations[name].get(candidate, set()))
            for name in variables
        }
        missing_last = {name: missing for name, missing in missing_last.items() if missing}
        if missing_last:
            if len(all_days) == 1:
                raise ValueError(
                    f"No complete UTC dates: last day {candidate} has missing UTC hours "
                    f"{missing_last}"
                )
            all_days.pop()
            skipped_last_day = {
                "date": candidate.isoformat(),
                "missing_utc_hours": missing_last,
            }
            print(
                f"ERA5 preflight: skipping incomplete last date {candidate}; "
                f"missing UTC hours {missing_last}", flush=True
            )
    for name, by_day in observations.items():
        for day in all_days:
            missing = required_hours - by_day.get(day, set())
            if missing:
                raise ValueError(f"{name}, {day}: missing UTC hours {sorted(missing)}")
    for before, after in zip(all_days, all_days[1:]):
        if (after - before).days != 1:
            raise ValueError(f"Missing calendar dates between {before} and {after}")
    grid0 = specifications[variables[0]][0]
    if any(specifications[name][0] != grid0 for name in variables):
        raise ValueError("Variables use different spatial grids; regrid them before averaging")
    print(f"ERA5 preflight passed: {len(all_days):,} complete days, {counted:,} messages", flush=True)
    return all_days, specifications, skipped_last_day


def _spatial_selection(codes, grib, grid, stride):
    """Determine the original grid and a safe 0.5-degree spatial subsampling."""
    ni, nj = grid[:2]
    lat = np.asarray(codes.codes_get_array(grib, "latitudes"), dtype=np.float64)
    lon = np.asarray(codes.codes_get_array(grib, "longitudes"), dtype=np.float64)
    if lat.size != ni * nj or lon.size != ni * nj:
        raise ValueError("Unexpected reduced or irregular spatial grid")
    lat = lat.reshape(nj, ni)
    lon = lon.reshape(nj, ni)
    if not np.allclose(lat, lat[:, :1], rtol=0, atol=1e-5):
        raise ValueError("Latitude must be constant within each grid row")
    if not np.allclose(lon, lon[:1, :], rtol=0, atol=1e-5):
        raise ValueError("Longitude must be constant within each grid column")
    dlat = float(np.median(np.abs(np.diff(lat[:, 0]))))
    dlon = float(np.median(np.abs(np.diff(lon[0, :]))))
    if stride == "auto":
        if np.isclose(dlat, 0.25, atol=1e-5) and np.isclose(dlon, 0.25, atol=1e-5):
            chosen = 2
        elif np.isclose(dlat, 0.5, atol=1e-5) and np.isclose(dlon, 0.5, atol=1e-5):
            chosen = 1
        else:
            raise ValueError(
                f"Grid spacing is {dlat:.5g} x {dlon:.5g} degrees; "
                "choose an explicit spatial_stride or regrid externally"
            )
    else:
        chosen = int(stride)
        if chosen < 1:
            raise ValueError("spatial_stride must be positive")
    return (nj, ni), chosen, lat[::chosen, ::chosen], lon[::chosen, ::chosen]


def _mask_array(mask, root, grid_shape):
    """Load an optional boolean ocean mask with the exact output grid shape."""
    if mask is None:
        return None
    if isinstance(mask, (str, Path)):
        path = Path(mask)
        if not path.is_absolute():
            path = root / path
        if path.suffix.lower() == ".npy":
            arr = np.load(path, allow_pickle=False)
        else:
            arr = np.fromfile(path, dtype=np.bool_)
    else:
        arr = np.asarray(mask)
    if arr.size != int(np.prod(grid_shape)):
        raise ValueError(f"Ocean mask has {arr.size} values; expected {grid_shape}")
    return np.asarray(arr, dtype=bool).reshape(-1)


def _write_space_time(source_time_space, destination, *, days, mask, factor, dtype,
                      overwrite, block_pixels):
    """Transpose a disk-backed time-space array in bounded pixel blocks."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.stem}_", suffix=".npy", delete=False
        ) as file:
            temp_path = Path(file.name)
        n_days, n_pixels = source_time_space.shape
        output = np.lib.format.open_memmap(
            temp_path, mode="w+", dtype=np.dtype(dtype), shape=(n_pixels, n_days)
        )
        for start in range(0, n_pixels, block_pixels):
            stop = min(start + block_pixels, n_pixels)
            piece = source_time_space[:, start:stop].T / float(days)
            if factor != 1.0:
                piece = piece / factor
            if mask is not None:
                piece = np.where(mask[start:stop, None], piece, np.nan)
            output[start:stop, :] = piece
        output.flush()
        del output
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _make_flux_sum(sensible_path, latent_path, flux_path, *, overwrite, dtype,
                   block_pixels=256):
    """Add sensible and latent heat-flux arrays without loading both at once."""
    sensible = np.load(sensible_path, mmap_mode="r", allow_pickle=False)
    latent = np.load(latent_path, mmap_mode="r", allow_pickle=False)
    if sensible.shape != latent.shape:
        raise ValueError("Sensible and latent output shapes differ")
    flux_path.parent.mkdir(parents=True, exist_ok=True)
    if flux_path.exists() and not overwrite:
        raise FileExistsError(flux_path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=flux_path.parent, prefix=".Flux_", suffix=".npy", delete=False
        ) as output:
            temporary = Path(output.name)
        flux = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=sensible.shape)
        for start in range(0, sensible.shape[0], block_pixels):
            stop = min(start + block_pixels, sensible.shape[0])
            flux[start:stop] = sensible[start:stop] + latent[start:stop]
        flux.flush()
        del flux
        os.replace(temporary, flux_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def convert_era5_grib(
    grib_files: Iterable[str | Path],
    output_root: str | Path,
    *,
    variables: Iterable[str] | None = None,
    variable: str | None = None,
    period_label: str = "2019-2027",
    hours: Iterable[int] = (0, 6, 12, 18),
    start_date: str | None = None,
    end_date: str | None = None,
    spatial_stride: int | str = "auto",
    mask=None,
    flux_units: str = "native",
    output_dtype: str = "float32",
    sum_dtype: str = "float64",
    overwrite: bool = False,
    derive_flux: bool = True,
    skip_incomplete_last_day: bool = True,
    block_pixels: int = 256,
) -> dict:
    """Convert one or many large GRIB streams to aligned daily ERA5 arrays.

    Parameters
    ----------
    grib_files : iterable of str or Path
        GRIB files containing arbitrary interleavings of ERA5 fields and times.
    output_root : str or Path
        Data root containing the DATA/ directory used by Ocean_components.
    variables : iterable of str, optional
        GRIB short names from sshf, slhf, sst, sp. Kept for the existing
        multi-variable API; omitted values retain its historical defaults.
    variable : str, optional
        Process a single ERA5 short name, independently of other variables.
        Cannot be combined with variables.
    period_label : str
        Exact period label used in the output filenames, e.g. 2019-2026.
    hours : iterable of int
        Required UTC validity hours for each daily mean, normally 0,6,12,18.
    start_date, end_date : str, optional
        Inclusive ISO date limits. No limits means the full available interval.
    spatial_stride : int or 'auto'
        'auto' subsamples native ERA5 0.25-degree grids by two and keeps
        already regridded 0.5-degree grids unchanged.
    mask : ndarray or path, optional
        Boolean ocean mask; land cells are stored as NaN. Can be the raw
        161x181 mask file used by the historical project.
    flux_units : {'native', 'W_m-2'}
        Keep original accumulated J m-2 heat fluxes, or divide the ERA5
        hourly accumulation by 3600 seconds. No sign inversion is applied.
    output_dtype, sum_dtype : str
        NPY dtype for final arrays and on-disk daily accumulation arrays.
    overwrite : bool
        Permit replacing destination files; disabled by default.
    derive_flux : bool
        Write Flux = sensible + latent if both components are requested.
    skip_incomplete_last_day : bool
        If the final UTC calendar date lacks any of the four required hours,
        omit that entire date. Incomplete interior dates still raise ValueError.
        Set False to require complete data for the final date as well.
    block_pixels : int
        Number of spatial locations handled per transpose/write block.

    Returns
    -------
    dict
        Output paths, grid shape, actual date coverage, units, and counts.

    Raises
    ------
    ValueError
        On missing/duplicate observations, unsupported grids or unit metadata.

    Notes
    -----
    This computes the mean of the four *selected* observations, not the mean
    of all 24 hours. For accumulated ERA5 fluxes the 00 UTC validity time
    describes the hour ending at midnight, part of the preceding calendar day.
    """
    if variable is not None and variables is not None:
        raise ValueError("Choose either variable or variables, not both")
    if variable is not None:
        names = (variable,)
    elif variables is None:
        names = tuple(VARIABLES)
    elif isinstance(variables, str):
        names = (variables,)
    else:
        names = tuple(dict.fromkeys(variables))
    if not names or any(name not in VARIABLES for name in names):
        raise ValueError(f"variable names must be from {tuple(VARIABLES)}")
    desired_hours = tuple(sorted(set(int(h) for h in hours)))
    if len(desired_hours) != 4 or any(h < 0 or h > 23 for h in desired_hours):
        raise ValueError("Exactly four distinct UTC hours are required")
    if flux_units not in ("native", "W_m-2"):
        raise ValueError("flux_units must be 'native' or 'W_m-2'")
    if output_dtype not in ("float32", "float64") or sum_dtype not in ("float32", "float64"):
        raise ValueError("Supported floating point dtypes are float32 and float64")
    if block_pixels < 1:
        raise ValueError("block_pixels must be positive")
    if not period_label or any(char in period_label for char in "/\\"):
        raise ValueError("period_label must not contain directory separators")
    if isinstance(grib_files, (str, Path)):
        grib_files = [grib_files]
    sources = _open_inputs(grib_files)
    root = Path(output_root).expanduser().resolve()
    first = date.fromisoformat(start_date) if start_date else None
    last = date.fromisoformat(end_date) if end_date else None
    if first and last and last < first:
        raise ValueError("end_date must not precede start_date")
    days, specs, skipped_last_day = _inspect(
        sources, names, desired_hours, first, last,
        skip_incomplete_last_day=skip_incomplete_last_day,
    )
    if derive_flux and not set(FLUX_VARIABLES).issubset(names):
        derive_flux = False
    paths = {
        name: root / "DATA" / VARIABLES[name][0] / f"{VARIABLES[name][1]}_{period_label}_grouped.npy"
        for name in names
    }
    flux_path = root / "DATA" / "Fluxes" / f"Flux_{period_label}_grouped.npy"
    # Per-variable sidecars allow independent conversions with the same period.
    # Keep the historical shared sidecar naming for explicit batch runs.
    sidecar_label = f"{period_label}_{names[0]}" if len(names) == 1 else period_label
    metadata_path = root / "DATA" / "ERA5_metadata" / f"ERA5_{sidecar_label}_metadata.json"
    date_path = root / "DATA" / "ERA5_metadata" / f"ERA5_{sidecar_label}_dates.npy"
    lat_path = root / "DATA" / "ERA5_metadata" / f"ERA5_{sidecar_label}_latitudes.npy"
    lon_path = root / "DATA" / "ERA5_metadata" / f"ERA5_{sidecar_label}_longitudes.npy"
    destinations = [*paths.values(), date_path, metadata_path, lat_path, lon_path]
    if derive_flux:
        destinations.append(flux_path)
    if not overwrite:
        existing = [str(p) for p in destinations if p.exists()]
        if existing:
            raise FileExistsError("Existing outputs require overwrite=True: " + ", ".join(existing))

    root.mkdir(parents=True, exist_ok=True)
    codes = _eccodes()
    n_days = len(days)
    day_index = {value: index for index, value in enumerate(days)}
    with tempfile.TemporaryDirectory(prefix=".era5_daily_", dir=root if root.exists() else None) as work:
        temporary_root = Path(work)
        accumulators = {}
        grid_shape = None
        chosen_stride = None
        latitudes = longitudes = ocean = None
        counts = {name: np.zeros(n_days, dtype=np.uint8) for name in names}
        decoded = 0
        print("ERA5 processing: streaming GRIB values to disk-backed daily sums...", flush=True)
        for source, index, grib in _iter_messages(codes, sources, headers_only=False):
            name = str(_get(codes, grib, "shortName", ""))
            if name not in paths:
                continue
            timestamp = _timestamp(codes, grib)
            day = timestamp.date()
            if day not in day_index or timestamp.minute or timestamp.hour not in desired_hours:
                continue
            grid = specs[name][0]
            if grid_shape is None:
                original_shape, chosen_stride, latitudes, longitudes = _spatial_selection(
                    codes, grib, grid, spatial_stride
                )
                grid_shape = latitudes.shape
                ocean = _mask_array(mask, root, grid_shape)
                n_pixels = int(np.prod(grid_shape))
                for key in names:
                    accumulators[key] = np.lib.format.open_memmap(
                        temporary_root / f"{key}_sums.npy", mode="w+",
                        dtype=sum_dtype, shape=(n_days, n_pixels)
                    )
            else:
                if (int(_get(codes, grib, "Ni", 0)), int(_get(codes, grib, "Nj", 0))) != grid[:2]:
                    raise ValueError(f"Grid geometry changed in {source} for {name}")
            flat = np.asarray(codes.codes_get_array(grib, "values"), dtype=np.float64)
            if bool(_get(codes, grib, "bitmapPresent", 0)):
                missing = _get(codes, grib, "missingValue")
                if missing is not None:
                    flat[flat == float(missing)] = np.nan
            if flat.size != int(np.prod(original_shape)):
                raise ValueError("Values size differs from GRIB spatial grid")
            values = flat.reshape(original_shape)[::chosen_stride, ::chosen_stride].reshape(-1)
            i = day_index[day]
            if counts[name][i] == 0:
                accumulators[name][i, :] = values
            else:
                accumulators[name][i, :] += values
            counts[name][i] += 1
            decoded += 1
            if decoded % 4000 == 0:
                print(f"ERA5 processing: decoded {decoded:,} selected messages", flush=True)
        for name, count in counts.items():
            if not np.all(count == len(desired_hours)):
                raise RuntimeError(f"Second-pass count mismatch for {name}: {count}")
        if grid_shape is None:
            raise ValueError("No eligible GRIB messages decoded")
        for name in names:
            print(f"ERA5 output: writing transposed daily array for {name}", flush=True)
            unit = specs[name][1]
            step_type = specs[name][2]
            if name in FLUX_VARIABLES and flux_units == "W_m-2":
                if "J" not in unit or step_type not in ("accum", "accumulation"):
                    raise ValueError(
                        f"Cannot convert {name} to W m-2: GRIB units={unit!r}, stepType={step_type!r}"
                    )
            accumulators[name].flush()
            factor = 3600.0 if name in FLUX_VARIABLES and flux_units == "W_m-2" else 1.0
            _write_space_time(
                accumulators[name], paths[name], days=len(desired_hours), mask=ocean,
                factor=factor, dtype=output_dtype, overwrite=overwrite,
                block_pixels=block_pixels,
            )
            del accumulators[name]
        if derive_flux:
            _make_flux_sum(paths["sshf"], paths["slhf"], flux_path,
                           overwrite=overwrite, dtype=output_dtype,
                           block_pixels=block_pixels)
        date_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(date_path, np.asarray([day.isoformat() for day in days], dtype="datetime64[D]"))
        np.save(lat_path, latitudes)
        np.save(lon_path, longitudes)
        report = {
            "period_label": period_label,
            "actual_first_date": days[0].isoformat(),
            "actual_last_date": days[-1].isoformat(),
            "days": n_days,
            "skipped_incomplete_last_day": skipped_last_day,
            "samples_per_day": len(desired_hours),
            "utc_hours": list(desired_hours),
            "grid_shape": list(grid_shape),
            "spatial_stride": chosen_stride,
            "layout": "space_time",
            "flattening": "GRIB grid row-major after optional spatial_stride",
            "dtype": output_dtype,
            "flux_units": "W m-2" if flux_units == "W_m-2" else "native GRIB J m-2",
            "timestamp_policy": "UTC validity date, including midnight accumulation on the validity date",
            "variables": {name: {"path": str(paths[name]), "units": specs[name][1],
                                 "step_type": specs[name][2]} for name in names},
            "flux_sum": str(flux_path) if derive_flux else None,
            "dates_file": str(date_path),
            "latitudes_file": str(lat_path),
            "longitudes_file": str(lon_path),
            "input_files": [str(item) for item in sources],
            "metadata_file": str(metadata_path),
        }
        metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"ERA5 completed: {n_days:,} days, output {grid_shape}", flush=True)
        return report


def convert_grib(files_path_prefix, grib_file, variable, output_file, *, mask=None,
                 spatial_stride="auto", period_label="2019-2026", flux_units="native",
                 overwrite=False):
    """Convert one GRIB variable to a requested daily NPY file.

    Parameters
    ----------
    files_path_prefix : str or Path
        Source GRIB and destination NPY root directory.
    grib_file : str or Path
        Source GRIB path, absolute or relative to files_path_prefix.
    variable : str
        GRIB short name: sshf, slhf, sst, or sp.
    output_file : str or Path
        Requested output file path, absolute or relative to files_path_prefix.
    mask : array_like or path, optional
        Ocean mask applied after sampling and daily averaging.
    spatial_stride : int or str
        Geographic stride, default auto for 0.25 to 0.5 degree.
    period_label : str
        Output period label.
    flux_units : str
        Native J m-2 or converted W m-2.
    overwrite : bool
        Whether to replace an existing NPY file.

    Returns
    -------
    Path
        Path of the requested daily NPY file.
    """
    root = Path(files_path_prefix)
    source = Path(grib_file)
    source = source if source.is_absolute() else root / source
    desired = Path(output_file)
    desired = desired if desired.is_absolute() else root / desired
    result = convert_era5_grib([source], root, variable=variable,
                               period_label=period_label, mask=mask,
                               spatial_stride=spatial_stride, flux_units=flux_units,
                               overwrite=overwrite, derive_flux=False)
    generated = Path(result["variables"][variable]["path"])
    if generated != desired:
        if desired.exists() and not overwrite:
            raise FileExistsError(desired)
        desired.parent.mkdir(parents=True, exist_ok=True)
        os.replace(generated, desired)
        metadata_file = Path(result["metadata_file"])
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        metadata["variables"][variable]["path"] = str(desired.resolve())
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return desired


def split_years(files_path_prefix, source_file, variable, year_ranges, *, samples_per_day=1):
    """Split an existing space-time NPY file by explicit calendar intervals.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data directory containing the NPY input.
    source_file : str or Path
        Existing NPY file with shape (space, time).
    variable : str
        Output filename prefix.
    year_ranges : iterable of (date, date)
        Chronologically consecutive, half-open time intervals.
    samples_per_day : int
        One for new daily NPY arrays; four for older unaveraged arrays.

    Returns
    -------
    list[Path]
        Saved per-period arrays.
    """
    root = Path(files_path_prefix)
    path = Path(source_file)
    data = np.load(path if path.is_absolute() else root / path,
                   mmap_mode="r", allow_pickle=False)
    if data.ndim != 2 or samples_per_day < 1:
        raise ValueError("Expected 2D space-time input and samples_per_day >= 1")
    offset = 0
    outputs = []
    for left, right in year_ranges:
        length = (right - left).days * samples_per_day
        if length < 1 or offset + length > data.shape[1]:
            raise ValueError("Invalid period or incomplete input data")
        target = root / "DATA" / f"{variable}_{left.year}-{right.year}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, data[:, offset:offset + length])
        outputs.append(target)
        offset += length
    if offset != data.shape[1]:
        raise ValueError("Some input samples were not assigned to periods")
    return outputs


def main():
    """Run one selected ERA5 variable or an explicit multi-variable batch."""
    parser = argparse.ArgumentParser(description="Stream ERA5 GRIB to daily NPY arrays")
    parser.add_argument("--grib", nargs="+", required=True, type=Path,
                        help="One or more GRIB files containing the requested variable")
    parser.add_argument("--root", required=True, type=Path,
                        help="OceanFull root containing the DATA directory")
    parser.add_argument("--period", default="2019-2026")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--variable", choices=list(VARIABLES),
                        help="Single ERA5 short name, e.g. sshf, slhf, sst or sp")
    choice.add_argument("--variables", nargs="+", choices=list(VARIABLES),
                        help="Explicit multi-variable mode for combined GRIB collections")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--hours", nargs=4, type=int, default=(0, 6, 12, 18))
    parser.add_argument("--spatial-stride", default="auto")
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--flux-units", choices=("native", "W_m-2"), default="native")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-derived-flux", action="store_true")
    parser.add_argument(
        "--strict-last-day", action="store_true",
        help="Raise an error instead of skipping an incomplete final UTC date",
    )
    args = parser.parse_args()
    stride = args.spatial_stride if args.spatial_stride == "auto" else int(args.spatial_stride)
    result = convert_era5_grib(
        args.grib, args.root, variable=args.variable, variables=args.variables,
        period_label=args.period, hours=args.hours,
        start_date=args.start_date, end_date=args.end_date,
        spatial_stride=stride, mask=args.mask, flux_units=args.flux_units,
        output_dtype=args.dtype, overwrite=args.overwrite,
        derive_flux=not args.no_derived_flux,
        skip_incomplete_last_day=not args.strict_last_day,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
