"""Streaming ERA5 GRIB -> daily NumPy arrays for Ocean_components.

Public entry points: convert_era5_grib, convert_grib, split_years.
Outputs are (space, day), using ERA5 validity timestamps and the same
four-hour sampling convention (00, 06, 12, 18 UTC) as the original parser.

Supports original variables and 10 m wind / 2 m temperature and dewpoint.
Requires `numpy` and (when actually reading GRIB) `eccodes`.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import hashlib

VARIABLES = {
    "avg_ishf": ("Fluxes", "sensible"),
    "avg_slhtf": ("Fluxes", "latent"),
    "sst": ("SST", "SST"),
    "sp": ("Pressure", "Pressure"),
    "10u": ("Atmosphere", "u10"),
    "10v": ("Atmosphere", "v10"),
    "2t": ("Atmosphere", "t2m"),
    "2d": ("Atmosphere", "d2m"),
}
DEFAULT_VARIABLES = ("avg_ishf", "avg_slhtf", "sst", "sp")
ALIASES = {"u10": "10u", "v10": "10v", "t2m": "2t", "d2m": "2d"}
VARIABLE_ALIASES = ALIASES
HEIGHT_LEVELS = {"10u": 10, "10v": 10, "2t": 2, "2d": 2}
FLUX_VARIABLES = frozenset(("avg_ishf", "avg_slhtf"))


def _canonical_name(value):
    return ALIASES.get(str(value), str(value))


def _eccodes():
    try:
        import eccodes
    except ImportError as exc:
        raise ImportError("Install ERA5 GRIB decoder: python -m pip install eccodes") from exc
    return eccodes


def _get(codes, message, key, default=None):
    try:
        return codes.codes_get(message, key)
    except Exception:
        return default


def _timestamp(codes, message):
    ymd = _get(codes, message, "validityDate")
    hm = _get(codes, message, "validityTime")
    if ymd is None or hm is None:
        step = _get(codes, message, "step")
        if step not in (None, 0, "0"):
            raise ValueError("Forecast message without validityDate/validityTime")
        ymd = _get(codes, message, "dataDate")
        hm = _get(codes, message, "dataTime")
    if ymd is None or hm is None:
        raise ValueError("GRIB message has no usable UTC validity time")
    hm = int(hm)
    if not 0 <= hm <= 2359 or hm % 100 >= 60:
        raise ValueError(f"Invalid validityTime: {hm}")
    return datetime.strptime(f"{int(ymd):08d}{hm:04d}", "%Y%m%d%H%M")


def _check_level(codes, message, name):
    """Accept explicit 2/10 m levels and legacy GRIB surface-level encoding.

    GRIB1 often stores dedicated 10u/10v/2t/2d parameters at surface level 0.
    Verify the ECMWF paramId before accepting this alternative representation.
    """
    kind = str(_get(codes, message, "typeOfLevel", "surface"))
    if name in HEIGHT_LEVELS:
        value = _get(codes, message, "level")
        try:
            level = int(value)
        except (ValueError, TypeError):
            level = None

        # GRIB2 style: height stored explicitly in the level metadata.
        if kind == "heightAboveGround" and level == HEIGHT_LEVELS[name]:
            return

        # GRIB1 style: height encoded in the meteorological parameter itself.
        parameter_ids = {"10u": 165, "10v": 166, "2t": 167, "2d": 168}
        param_id = _get(codes, message, "paramId")
        try:
            param_id = int(param_id)
        except (ValueError, TypeError):
            param_id = None
        if kind == "surface" and level == 0 and param_id == parameter_ids[name]:
            return

        raise ValueError(
            f"Wrong GRIB level for {name}: type={kind!r}, level={value!r}, "
            f"paramId={param_id!r}; expected heightAboveGround at "
            f"{HEIGHT_LEVELS[name]} metres, or surface level 0 with "
            f"paramId={parameter_ids[name]}"
        )
    elif kind not in ("surface", "meanSea"):
        raise ValueError(f"Unexpected level type for {name}: {kind}")


def _open_inputs(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    found = []
    for item in paths:
        source = Path(item).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("rb") as inp:
            header = inp.read(4)
        if header[:2] == b"PK":
            raise ValueError(f"{source}: ZIP archive; extract the GRIB first")
        if header != b"GRIB":
            raise ValueError(f"{source}: expected GRIB header, got {header!r}")
        if source not in found:
            found.append(source)
    if not found:
        raise ValueError("No GRIB input files supplied")
    return found


def _iter_messages(codes, sources, *, headers_only):
    for source in sources:
        with source.open("rb") as reader:
            number = 0
            while True:
                message = codes.codes_grib_new_from_file(reader, headers_only=headers_only)
                if message is None:
                    break
                number += 1
                try:
                    yield source, number, message
                finally:
                    codes.codes_release(message)


def _grid_spec(codes, message):
    kind = str(_get(codes, message, "gridType", ""))
    if kind != "regular_ll":
        raise ValueError(f"Unsupported gridType {kind!r}; use a regular lat/lon grid")
    grid = (
        int(_get(codes, message, "Ni", 0)),
        int(_get(codes, message, "Nj", 0)),
        float(_get(codes, message, "latitudeOfFirstGridPointInDegrees", 999.0)),
        float(_get(codes, message, "longitudeOfFirstGridPointInDegrees", 999.0)),
        float(_get(codes, message, "latitudeOfLastGridPointInDegrees", 999.0)),
        float(_get(codes, message, "longitudeOfLastGridPointInDegrees", 999.0)),
    )
    if min(grid[:2]) < 2 or max(map(abs, grid[2:])) > 720:
        raise ValueError(f"Invalid GRIB grid metadata {grid}")
    return grid


def _inspect(sources, variables, hours, first_date, last_date, *, skip_incomplete_last_day=True, duplicate_keys=None):
    codes = _eccodes()
    by_variable = {name: {} for name in variables}
    specifications = {}
    n = 0
    print("ERA5 preflight: scanning GRIB headers...", flush=True)
    for source, index, message in _iter_messages(codes, sources, headers_only=True):
        name = _canonical_name(_get(codes, message, "shortName", ""))
        if name not in by_variable:
            continue
        ts = _timestamp(codes, message)
        day = ts.date()
        if (first_date and day < first_date) or (last_date and day > last_date):
            continue
        if ts.minute != 0 or ts.hour not in hours:
            continue
        _check_level(codes, message, name)
        if _get(codes, message, "number", 0) not in (0, "0"):
            raise ValueError("Ensemble members are not supported")
        spec = (
            _grid_spec(codes, message),
            str(_get(codes, message, "units", "")),
            str(_get(codes, message, "stepType", "")),
        )
        if name in specifications and spec != specifications[name]:
            raise ValueError(f"Inconsistent grid, unit, or step type for {name} in {source}")
        specifications[name] = spec
        registered = by_variable[name].setdefault(day, set())




        if ts.hour in registered:
            if duplicate_keys is not None:
                duplicate_keys.add((name, ts))
            continue

        registered.add(ts.hour)
        n += 1
        if n % 4000 == 0:
            print(f"ERA5 preflight: {n:,} selected messages", flush=True)
    if not n:
        raise ValueError("No requested variables/times in GRIB input")
    for name, observations in by_variable.items():
        if not observations:
            raise ValueError(f"Missing GRIB variable: {name}")
    all_days = sorted(set().union(*(set(group) for group in by_variable.values())))
    required = set(hours)
    skipped = None
    if skip_incomplete_last_day:
        end = all_days[-1]
        missing = {name: sorted(required - group.get(end, set())) for name, group in by_variable.items()}
        missing = {name: gaps for name, gaps in missing.items() if gaps}
        if missing:
            if len(all_days) == 1:
                raise ValueError(f"No complete dates: {end} missing {missing}")
            all_days.pop()
            skipped = {"date": end.isoformat(), "missing_utc_hours": missing}
            print(f"ERA5 preflight: skipped incomplete last day {end}", flush=True)
    for name, group in by_variable.items():
        for day in all_days:
            missing = required - group.get(day, set())
            if missing:
                raise ValueError(f"{name} {day}: missing UTC hours {sorted(missing)}")
    for left, right in zip(all_days, all_days[1:]):
        if (right - left).days != 1:
            raise ValueError(f"Missing calendar day between {left} and {right}")
    first_grid = specifications[variables[0]][0]
    if any(specifications[name][0] != first_grid for name in variables):
        raise ValueError("Requested fields have different spatial grids")
    print(f"ERA5 preflight passed: {len(all_days):,} complete days", flush=True)
    return all_days, specifications, skipped


def _spatial_selection(codes, message, grid, stride):
    ni, nj = grid[:2]
    lat = np.asarray(codes.codes_get_array(message, "latitudes"), dtype=np.float64).reshape(nj, ni)
    lon = np.asarray(codes.codes_get_array(message, "longitudes"), dtype=np.float64).reshape(nj, ni)
    if not np.allclose(lat, lat[:, :1], atol=1e-5, rtol=0):
        raise ValueError("Latitude is not constant along rows")
    if not np.allclose(lon, lon[:1, :], atol=1e-5, rtol=0):
        raise ValueError("Longitude is not constant along columns")
    dy = float(np.median(np.abs(np.diff(lat[:, 0]))))
    dx = float(np.median(np.abs(np.diff(lon[0, :]))))
    if stride == "auto":
        if np.isclose(dx, 0.25, atol=1e-5) and np.isclose(dy, 0.25, atol=1e-5):
            chosen = 2
        elif np.isclose(dx, 0.5, atol=1e-5) and np.isclose(dy, 0.5, atol=1e-5):
            chosen = 1
        else:
            raise ValueError(f"Unsupported grid spacing ({dy}, {dx}); specify spatial_stride")
    else:
        chosen = int(stride)
        if chosen < 1:
            raise ValueError("spatial_stride must be positive")
    return (nj, ni), chosen, lat[::chosen, ::chosen], lon[::chosen, ::chosen]


def _mask_array(mask, root, grid_shape):
    if mask is None:
        return None
    if isinstance(mask, (str, Path)):
        p = Path(mask)
        if not p.is_absolute():
            p = root / p
        if p.suffix.lower() == ".npy":
            m = np.load(p, allow_pickle=False)
        else:
            m = np.fromfile(p, dtype=np.bool_)
    else:
        m = np.asarray(mask)
    if m.size != int(np.prod(grid_shape)):
        raise ValueError(f"Mask size {m.size} != output grid {grid_shape}")
    return np.asarray(m, dtype=bool).reshape(-1)


def _atomic_memmap_path(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, prefix=f".{output_path.stem}_", suffix=".npy", delete=False,
    ) as tmp:
        return Path(tmp.name)


def _write_space_time(sums, output_path, *, days, mask, factor, dtype, overwrite, block_pixels):
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    tmp = _atomic_memmap_path(output_path)
    try:
        n_days, n_pixels = sums.shape
        dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=(n_pixels, n_days))
        for start in range(0, n_pixels, block_pixels):
            stop = min(start + block_pixels, n_pixels)
            block = sums[:, start:stop].T / days / factor
            if mask is not None:
                block = np.where(mask[start:stop, None], block, np.nan)
            dst[start:stop] = block
        dst.flush()
        del dst
        os.replace(tmp, output_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _make_flux_sum(sensible_path, latent_path, flux_path, *, overwrite, dtype, block_pixels=256):
    sensible = np.load(sensible_path, mmap_mode="r", allow_pickle=False)
    latent = np.load(latent_path, mmap_mode="r", allow_pickle=False)
    if sensible.shape != latent.shape:
        raise ValueError("Flux component shapes do not match")
    if flux_path.exists() and not overwrite:
        raise FileExistsError(flux_path)
    tmp = _atomic_memmap_path(flux_path)
    try:
        dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=sensible.shape)
        for start in range(0, sensible.shape[0], block_pixels):
            stop = min(sensible.shape[0], start + block_pixels)
            dst[start:stop] = sensible[start:stop] + latent[start:stop]
        dst.flush()
        del dst
        os.replace(tmp, flux_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def convert_era5_grib(
    grib_files: Iterable[str | Path], output_root: str | Path, *,
    variables: Iterable[str] | None = None, variable: str | None = None,
    period_label: str = "2019-2027", hours: Iterable[int] = (0, 6, 12, 18),
    start_date: str | None = None, end_date: str | None = None,
    spatial_stride: int | str = "auto", mask=None, flux_units: str = "native",
    output_dtype: str = "float32", sum_dtype: str = "float64",
    overwrite: bool = False, derive_flux: bool = True,
    skip_incomplete_last_day: bool = True, block_pixels: int = 256,
) -> dict:
    """Convert one or more ERA5 GRIB files, streaming to aligned daily NPY arrays.

    `period_label` only names files; actual date coverage is saved separately.
    Default daily values average four sampled UTC times; not all 24 hours.
    """
    if variable is not None and variables is not None:
        raise ValueError("Select either variable or variables")
    if variable is not None:
        names = (_canonical_name(variable),)
    elif variables is None:
        names = DEFAULT_VARIABLES
    elif isinstance(variables, str):
        names = (_canonical_name(variables),)
    else:
        names = tuple(dict.fromkeys(_canonical_name(name) for name in variables))
    if not names or any(name not in VARIABLES for name in names):
        raise ValueError(f"Unknown variable. Expected {tuple(VARIABLES)} or {tuple(ALIASES)}")
    desired_hours = tuple(sorted(set(int(h) for h in hours)))
    if len(desired_hours) != 4 or any(not 0 <= h < 24 for h in desired_hours):
        raise ValueError("Exactly four distinct UTC observation hours are required")
    if flux_units not in ("native", "W_m-2"):
        raise ValueError("flux_units must be 'native' or 'W_m-2'")
    if output_dtype not in ("float32", "float64") or sum_dtype not in ("float32", "float64"):
        raise ValueError("Supported dtypes: float32, float64")
    if block_pixels <= 0:
        raise ValueError("block_pixels must be positive")
    if not period_label or any(c in period_label for c in "/\\"):
        raise ValueError("Invalid period_label")
    sources = _open_inputs(grib_files)
    root = Path(output_root).expanduser().resolve()
    left = date.fromisoformat(start_date) if start_date else None
    right = date.fromisoformat(end_date) if end_date else None
    if left and right and right < left:
        raise ValueError("end_date precedes start_date")
    duplicate_keys = set()
    days, specs, skipped_last = _inspect(
        sources, names, desired_hours, left, right,
        skip_incomplete_last_day=skip_incomplete_last_day,
        duplicate_keys=duplicate_keys,
    )
    if duplicate_keys:
        print(f"ERA5 preflight: {len(duplicate_keys):,} duplicated variable/timestamp keys; "
              "comparing field values during decoding...", flush=True)
    n_days = len(days)
    day_index = {day: i for i, day in enumerate(days)}
    paths = {
        name: root / "DATA" / VARIABLES[name][0] / f"{VARIABLES[name][1]}_{period_label}_grouped.npy"
        for name in names
    }
    derive_flux = derive_flux and FLUX_VARIABLES.issubset(names)
    flux_path = root / "DATA" / "Fluxes" / f"Flux_{period_label}_grouped.npy"
    sidecar_label = f"{period_label}_{names[0]}" if len(names) == 1 else period_label
    metadata_dir = root / "DATA" / "ERA5_metadata"
    metadata_path = metadata_dir / f"ERA5_{sidecar_label}_metadata.json"
    dates_path = metadata_dir / f"ERA5_{sidecar_label}_dates.npy"
    lats_path = metadata_dir / f"ERA5_{sidecar_label}_latitudes.npy"
    lons_path = metadata_dir / f"ERA5_{sidecar_label}_longitudes.npy"
    destinations = [*paths.values(), metadata_path, dates_path, lats_path, lons_path]
    if derive_flux:
        destinations.append(flux_path)
    if not overwrite:
        existing = [str(p) for p in destinations if p.exists()]
        if existing:
            raise FileExistsError("Outputs already exist; use overwrite=True: " + ", ".join(existing))
    root.mkdir(parents=True, exist_ok=True)
    codes = _eccodes()
    with tempfile.TemporaryDirectory(prefix=".era5_daily_", dir=root) as work:
        working = Path(work)
        accumulators = {}
        counts = {name: np.zeros(n_days, dtype=np.uint8) for name in names}
        grid_shape = None
        stride = None
        original_shape = None
        latitude = longitude = ocean_mask = None
        duplicate_hashes = {}
        skipped_duplicates = 0
        decoded = 0
        print("ERA5: decoding selected GRIB fields to disk-backed daily sums...", flush=True)
        for source, index, message in _iter_messages(codes, sources, headers_only=False):
            name = _canonical_name(_get(codes, message, "shortName", ""))
            if name not in paths:
                continue
            ts = _timestamp(codes, message)
            day = ts.date()
            if day not in day_index or ts.minute or ts.hour not in desired_hours:
                continue
            if grid_shape is None:
                original_shape, stride, latitude, longitude = _spatial_selection(
                    codes, message, specs[name][0], spatial_stride,
                )
                grid_shape = latitude.shape
                ocean_mask = _mask_array(mask, root, grid_shape)
                count_pixels = int(np.prod(grid_shape))
                for key in names:
                    accumulators[key] = np.lib.format.open_memmap(
                        working / f"{key}_sum.npy", mode="w+", dtype=sum_dtype,
                        shape=(n_days, count_pixels),
                    )
            elif _grid_spec(codes, message) != specs[name][0]:
                raise ValueError(f"Grid changed while decoding {name}, {source.name} message {index}")
            flat = np.asarray(codes.codes_get_array(message, "values"), dtype=np.float64)
            if flat.size != np.prod(original_shape):
                raise ValueError(f"GRIB field size does not match grid for {name}")
            if bool(_get(codes, message, "bitmapPresent", 0)):
                missing = _get(codes, message, "missingValue")
                if missing is not None:
                    flat[flat == float(missing)] = np.nan
            # SHA-256 of all unpacked grid values (before spatial stride).
            # Only timestamps reported as duplicated by preflight are hashed.
            duplicate_key = (name, ts)
            if duplicate_key in duplicate_keys:
                digest = hashlib.sha256(flat.tobytes()).digest()
                original_digest = duplicate_hashes.get(duplicate_key)
                if original_digest is None:
                    duplicate_hashes[duplicate_key] = digest
                elif digest == original_digest:
                    skipped_duplicates += 1
                    continue
                else:
                    raise ValueError(
                        f"Conflicting duplicate {name} at {ts:%Y-%m-%d %H:%M} UTC: "
                        f"different grid values in {source.name}, GRIB message {index}. "
                        "Cannot safely discard this observation."
                    )
            field = flat.reshape(original_shape)[::stride, ::stride].reshape(-1)
            i = day_index[day]
            if counts[name][i] == 0:
                accumulators[name][i] = field
            else:
                accumulators[name][i] += field
            counts[name][i] += 1
            decoded += 1
            if decoded % 4000 == 0:
                print(f"ERA5: decoded {decoded:,} fields", flush=True)
        if skipped_duplicates:
            print(f"ERA5: skipped {skipped_duplicates:,} byte-identical decoded "
                  "duplicate field(s)", flush=True)
        if grid_shape is None:
            raise ValueError("No matching GRIB values decoded")
        for name in names:
            if not np.all(counts[name] == len(desired_hours)):
                raise RuntimeError(f"Header/data scan mismatch for {name}: {counts[name]}")
            unit = specs[name][1]
            step_type = specs[name][2]
            if name in FLUX_VARIABLES and flux_units == "W_m-2":
                if "J" not in unit or step_type not in ("accum", "accumulation"):
                    raise ValueError(f"Cannot convert flux {name} units={unit}, stepType={step_type}")
            accumulators[name].flush()
            factor = 3600.0 if name in FLUX_VARIABLES and flux_units == "W_m-2" else 1.0
            print(f"ERA5: writing {paths[name].name}", flush=True)
            _write_space_time(
                accumulators[name], paths[name], days=len(desired_hours),
                mask=ocean_mask, factor=factor, dtype=output_dtype,
                overwrite=overwrite, block_pixels=block_pixels,
            )
            del accumulators[name]
        if derive_flux:
            _make_flux_sum(
                paths["avg_ishf"], paths["avg_slhtf"], flux_path,
                overwrite=overwrite, dtype=output_dtype, block_pixels=block_pixels,
            )
        metadata_dir.mkdir(parents=True, exist_ok=True)
        np.save(dates_path, np.asarray([str(d) for d in days], dtype="datetime64[D]"))
        np.save(lats_path, latitude)
        np.save(lons_path, longitude)
        report = {
            "period_label": period_label,
            "actual_first_date": days[0].isoformat(),
            "actual_last_date": days[-1].isoformat(),
            "days": n_days,
            "skipped_incomplete_last_day": skipped_last,
            "samples_per_day": len(desired_hours),
            "utc_hours": list(desired_hours),
            "grid_shape": list(grid_shape),
            "spatial_stride": stride,
            "layout": "space_time",
            "flattening": "GRIB grid row-major after optional spatial_stride",
            "dtype": output_dtype,
            "flux_units": (
                "W m-2" if flux_units == "W_m-2" else "native GRIB J m-2"
            ) if FLUX_VARIABLES.intersection(names) else None,
            "timestamp_policy": "UTC validity date",
            "variables": {
                name: {
                    "path": str(paths[name]), "units": specs[name][1],
                    "step_type": specs[name][2], "level_m": HEIGHT_LEVELS.get(name),
                } for name in names
            },
            "flux_sum": str(flux_path) if derive_flux else None,
            "dates_file": str(dates_path),
            "latitudes_file": str(lats_path),
            "longitudes_file": str(lons_path),
            "input_files": [str(path) for path in sources],
            "metadata_file": str(metadata_path),
        }
        metadata_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"ERA5 finished: {n_days:,} days, grid={grid_shape}", flush=True)
        return report


def convert_grib(files_path_prefix, grib_file, variable, output_file, *, mask=None,
                 spatial_stride="auto", period_label="2019-2026", flux_units="native",
                 overwrite=False):
    root = Path(files_path_prefix)
    inp = Path(grib_file)
    if not inp.is_absolute():
        inp = root / inp
    desired = Path(output_file)
    if not desired.is_absolute():
        desired = root / desired
    canonical = _canonical_name(variable)
    result = convert_era5_grib(
        [inp], root, variable=canonical, period_label=period_label, mask=mask,
        spatial_stride=spatial_stride, flux_units=flux_units,
        overwrite=overwrite, derive_flux=False,
    )
    generated = Path(result["variables"][canonical]["path"])
    if generated != desired:
        if desired.exists() and not overwrite:
            raise FileExistsError(desired)
        desired.parent.mkdir(parents=True, exist_ok=True)
        os.replace(generated, desired)
        path = Path(result["metadata_file"])
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["variables"][canonical]["path"] = str(desired.resolve())
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return desired


def split_years(files_path_prefix, source_file, variable, year_ranges, *, samples_per_day=1):
    root = Path(files_path_prefix)
    inp = Path(source_file)
    arr = np.load(inp if inp.is_absolute() else root / inp, mmap_mode="r", allow_pickle=False)
    if arr.ndim != 2 or samples_per_day < 1:
        raise ValueError("Expected 2D space/time NPY and samples_per_day >= 1")
    offset = 0
    outputs = []
    for begin, end in year_ranges:
        n = (end - begin).days * samples_per_day
        if n < 1 or offset + n > arr.shape[1]:
            raise ValueError("Invalid period or incomplete NPY")
        dest = root / "DATA" / f"{variable}_{begin.year}-{end.year}.npy"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, arr[:, offset: offset+n])
        outputs.append(dest)
        offset += n
    if offset != arr.shape[1]:
        raise ValueError("Not all samples were assigned to periods")
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Stream ERA5 GRIB to daily NumPy arrays")
    parser.add_argument("--grib", nargs="+", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--period", default="2019-2027")
    group = parser.add_mutually_exclusive_group(required=True)
    choices = [*VARIABLES, *ALIASES]
    group.add_argument("--variable", choices=choices)
    group.add_argument("--variables", nargs="+", choices=choices)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--hours", nargs=4, type=int, default=(0, 6, 12, 18))
    parser.add_argument("--spatial-stride", default="auto")
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--flux-units", choices=("native", "W_m-2"), default="native")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-derived-flux", action="store_true")
    parser.add_argument("--strict-last-day", action="store_true")
    args = parser.parse_args()
    step = args.spatial_stride if args.spatial_stride == "auto" else int(args.spatial_stride)
    result = convert_era5_grib(
        args.grib, args.root, variable=args.variable, variables=args.variables,
        period_label=args.period, hours=args.hours, start_date=args.start_date,
        end_date=args.end_date, spatial_stride=step, mask=args.mask,
        flux_units=args.flux_units, output_dtype=args.dtype,
        overwrite=args.overwrite, derive_flux=not args.no_derived_flux,
        skip_incomplete_last_day=not args.strict_last_day,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
