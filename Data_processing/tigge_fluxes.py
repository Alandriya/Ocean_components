"""TIGGE accumulated sensible/latent heat -> DAILY MEAN heat flux, W m^-2.

This processing lives in Data_processing, separately from the historical ERA5
GRIB converter. The native TIGGE fluxes (sshf=146, slhf=147) are accumulated
since +0h in J m^-2. Daily means are differences of successive cumulative
steps divided by 86400 seconds; this is *not* a GRIB parameter renaming.

By default, day 1 is the UTC calendar day of the 00 UTC issue:
  issue day: (+0h,+24h], next: (+24h,+48h], ..., day 14: (+312h,+336h].
  +360h is also downloaded for --day-offset 1 (following day as day 1).

ECMWF docs:
 https://confluence.ecmwf.int/spaces/TIGGE/pages/51721815/Surface+sensible+heat+flux
 https://confluence.ecmwf.int/spaces/TIGGE/pages/51721809/Surface+latent+heat+flux
 https://confluence.ecmwf.int/spaces/TIGGE/pages/51721961/Rules+for+data+encoding+and+exchange
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

PARAM_IDS = {146: "sensible", 147: "latent"}
SHORT_NAMES = {"sensible": "sshf", "latent": "slhf"}
FIELDS = ("sensible", "latent")
OUT_UNITS = ("W m-2", "W m-2")
CUMULATIVE_HOURS = tuple(range(24, 361, 24))  # 15 steps
SECONDS_PER_DAY = 86400.0
GRID_SHAPE = (321, 361)
ERA5_GRID_SHAPE = (161, 181)


def cumulative_to_daily_mean(accumulated: np.ndarray, day_offset: int = 0) -> np.ndarray:
    """Convert cumulative E(+24), ..., E(+360) [J m^-2] to 14 UTC daily means.

    Parameters
    ----------
    accumulated : np.ndarray
        Shape [15, ...]; cumulative energy at +24,+48,...,+360 h.
    day_offset : int
        0 -> issue-day through issue+13 (first difference E24 - E0).
        1 -> issue+1 through issue+14 (first difference E48 - E24).

    Returns
    -------
    np.ndarray
        Shape [14, ...], W m^-2; no sign reversal.
    """
    if day_offset not in (0, 1):
        raise ValueError("day_offset must be 0 or 1")
    energy = np.asarray(accumulated, dtype=np.float64)
    if energy.ndim < 1 or energy.shape[0] != len(CUMULATIVE_HOURS):
        raise ValueError(f"Expected 15 cumulative steps +24..+360 h, got shape {energy.shape}")
    if not np.isfinite(energy).all():
        raise ValueError("Non-finite accumulated energy")
    # Introduce the physical E(+0h) = 0, explicitly defined by TIGGE.
    boundaries = np.concatenate((np.zeros_like(energy[:1]), energy), axis=0)
    means = np.diff(boundaries, axis=0) / SECONDS_PER_DAY  # 15 intervals
    start = day_offset
    return means[start:start+14].astype(np.float32)


def _sha256(path: Path) -> str:
    """Calculate a file SHA-256 checksum."""
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for part in iter(lambda: inp.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _codes_get(codes: Any, message: Any, key: str, default=None):
    """Retrieve an ecCodes metadata value, or return ``default`` if missing."""
    try:
        return codes.codes_get(message, key)
    except Exception:
        return default


def _unit_ok(unit: str) -> bool:
    """Both W m^-2 s and J m^-2 represent integrated heat, not a mean rate."""
    text = str(unit).lower().replace(" ", "").replace("**", "").replace("^", "")
    return text in ("wm-2s", "jm-2", "w/m2*s", "j/m2", "wm-2s-0")


def _parse_reference_time(codes, msg) -> datetime:
    """Parse the 00 UTC forecast reference timestamp from GRIB metadata."""
    date_val = _codes_get(codes, msg, "dataDate")
    clock_val = _codes_get(codes, msg, "dataTime")
    if date_val is None or clock_val is None:
        raise ValueError("Missing GRIB initialization dataDate/dataTime")
    naive = datetime.strptime(f"{int(date_val):08d}{int(clock_val):04d}", "%Y%m%d%H%M")
    return naive.replace(tzinfo=timezone.utc)


def _check_valid_time(codes, msg, ref: datetime, lead_h: int):
    """Validate that GRIB validity time matches initialization plus lead."""
    date_val = _codes_get(codes, msg, "validityDate")
    clock_val = _codes_get(codes, msg, "validityTime")
    if date_val is None or clock_val is None:
        raise ValueError("Missing GRIB valid time; cannot prove accumulation endpoint")
    valid = datetime.strptime(f"{int(date_val):08d}{int(clock_val):04d}", "%Y%m%d%H%M")
    valid = valid.replace(tzinfo=timezone.utc)
    if valid != ref + timedelta(hours=lead_h):
        raise ValueError(f"Validity mismatch: {valid} != {ref} + {lead_h} hours")


def _grid_setup(codes, message, points: list[dict]):
    """Validate the 0.25-degree study grid and map 15 sites to native cells."""
    if str(_codes_get(codes, message, "gridType", "")) != "regular_ll":
        raise ValueError("TIGGE flux processing requires regular_ll grid")
    ni = int(_codes_get(codes, message, "Ni", 0))
    nj = int(_codes_get(codes, message, "Nj", 0))
    if (nj, ni) != GRID_SHAPE:
        raise ValueError(f"Unexpected grid {nj}x{ni}; expected 321x361 (0.25-degree study area)")
    lat = np.asarray(codes.codes_get_array(message, "latitudes"), dtype=np.float64).reshape(nj, ni)
    lon = np.asarray(codes.codes_get_array(message, "longitudes"), dtype=np.float64).reshape(nj, ni)
    lon = (lon + 180.0) % 360.0 - 180.0
    lat_axis = np.linspace(80., 0., nj)
    lon_axis = np.linspace(-90., 0., ni)
    if not (np.allclose(lat[:, 0], lat_axis, atol=1e-4)
            and np.allclose(lon[0, :], lon_axis, atol=1e-4)
            and np.allclose(lat, lat_axis[:, None], atol=1e-4)
            and np.allclose(lon, lon_axis[None, :], atol=1e-4)):
        raise ValueError("GRIB scanning/layout does not match N->S and W->E study grid")
    offsets = []
    for p in points:
        plat, plon = float(p["lat_deg"]), float(p["lon_deg"])
        r, c = int(round((80.0 - plat) * 4)), int(round((plon + 90.0) * 4))
        if not (0 <= r < nj and 0 <= c < ni and
                abs(lat[r, c] - plat) < 1e-5 and abs(lon[r, c] - plon) < 1e-5):
            raise ValueError(f"Point {p['id']} is outside forecast grid or off its nodes")
        if "grid_y" in p and "grid_x" in p:
            if r != 2*int(p["grid_y"]) or c != 2*int(p["grid_x"]):
                raise ValueError(f"Point {p['id']} has inconsistent ERA5 grid index")
        offsets.append(r*ni+c)
    return offsets, (ni, nj,
                     float(_codes_get(codes, message, "latitudeOfFirstGridPointInDegrees")),
                     float(_codes_get(codes, message, "longitudeOfFirstGridPointInDegrees")),
                     float(_codes_get(codes, message, "latitudeOfLastGridPointInDegrees")),
                     float(_codes_get(codes, message, "longitudeOfLastGridPointInDegrees")))


def convert_flux_grib(
    grib: Path, points_json: Path, out_dir: Path, *,
    day_offset: int = 0, full_grid: bool = False, overwrite: bool = False,
):
    """Decode TIGGE fluxes from a combined GRIB or original flux-only GRIB.

    Meteorological GRIB records and extra 6h accumulation endpoints are skipped;
    all accumulated flux values at +24..+360h are required.
    """
    if day_offset not in (0, 1):
        raise ValueError("day_offset must be 0 or 1")
    try:
        import eccodes
    except ImportError as exc:
        raise SystemExit("Missing ecCodes. Install: python -m pip install eccodes") from exc

    grib = Path(grib).expanduser().resolve()
    points_json = Path(points_json).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if not grib.is_file():
        raise FileNotFoundError(grib)
    points = json.loads(points_json.read_text(encoding="utf-8"))["points"]
    if len(points) != 15 or len({x["id"] for x in points}) != 15:
        raise ValueError("Expected 15 distinct points from points.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    offset_label = "issue_day" if day_offset == 0 else "next_day"
    cum_points = np.full((len(CUMULATIVE_HOURS), len(points), 2), np.nan, dtype=np.float64)
    cum_grid = (np.full((len(CUMULATIVE_HOURS), *ERA5_GRID_SHAPE, 2), np.nan, dtype=np.float64)
                if full_grid else None)
    ref_time = None
    offsets = None
    grid_spec = None
    present = set()
    count = 0
    selected_count = 0
    with grib.open("rb") as inp:
        while True:
            msg = eccodes.codes_grib_new_from_file(inp)
            if msg is None:
                break
            count += 1
            try:
                if int(_codes_get(eccodes, msg, "edition", 0)) != 2:
                    raise ValueError(f"Record {count} is not GRIB2")
                processed = _codes_get(eccodes, msg, "typeOfProcessedData", None)
                if processed is None or int(processed) != 3:
                    raise ValueError(f"Record {count}: not ECMWF TIGGE control forecast (type=3), got {processed!r}")
                centre = str(_codes_get(eccodes, msg, "centre", "")).lower()
                if centre not in ("98", "ecmf", "ecmwf"):
                    raise ValueError(f"Record {count}: wrong forecast centre {centre!r}")
                pid = int(_codes_get(eccodes, msg, "paramId", -1))
                if pid not in PARAM_IDS:
                    # One combined ECDS request also includes met fields.
                    if pid in (134, 165, 166, 167, 168):
                        continue
                    raise ValueError(f"Unexpected paramId {pid} in {grib.name} record {count}")
                field = PARAM_IDS[pid]
                short = str(_codes_get(eccodes, msg, "shortName", ""))
                if short != SHORT_NAMES[field]:
                    raise ValueError(f"Inconsistent GRIB paramId/shortName {pid}/{short!r}")
                if str(_codes_get(eccodes, msg, "typeOfLevel", "")) != "surface":
                    raise ValueError(f"{field}: unexpected GRIB level")
                if str(_codes_get(eccodes, msg, "stepType", "")) != "accum":
                    raise ValueError(f"{field}: expected accumulated, not mean/instantaneous GRIB")
                if int(_codes_get(eccodes, msg, "startStep", -1)) != 0:
                    raise ValueError(f"{field}: cumulative flux must start at +0h")
                unit = str(_codes_get(eccodes, msg, "units", ""))
                if not _unit_ok(unit):
                    raise ValueError(f"{field}: unexpected cumulative units {unit!r}; expected J/m2")
                hour = int(_codes_get(eccodes, msg, "endStep", -1))
                if hour not in CUMULATIVE_HOURS:
                    # A unified request selects the common 6h steps for all
                    # variables; 6, 12, ... may be present as additional flux
                    # accumulations. They are irrelevant to UTC daily means.
                    if 0 <= hour <= 360 and hour % 6 == 0:
                        continue
                    raise ValueError(f"{field}: unexpected accumulation endpoint +{hour}h")
                selected_count += 1
                key = (field, hour)
                if key in present:
                    raise ValueError(f"Duplicate GRIB field {key}")
                present.add(key)

                reference = _parse_reference_time(eccodes, msg)
                if reference.hour != 0 or reference.minute != 0:
                    raise ValueError("Only 00 UTC issues are supported")
                if ref_time is None:
                    ref_time = reference
                if reference != ref_time:
                    raise ValueError("GRIB contains mixed forecast issue times")
                _check_valid_time(eccodes, msg, reference, hour)
                if offsets is None:
                    offsets, grid_spec = _grid_setup(eccodes, msg, points)
                else:
                    probe = (int(_codes_get(eccodes, msg, "Ni", -1)),
                             int(_codes_get(eccodes, msg, "Nj", -1)),
                             float(_codes_get(eccodes, msg, "latitudeOfFirstGridPointInDegrees", -999)),
                             float(_codes_get(eccodes, msg, "longitudeOfFirstGridPointInDegrees", -999)),
                             float(_codes_get(eccodes, msg, "latitudeOfLastGridPointInDegrees", -999)),
                             float(_codes_get(eccodes, msg, "longitudeOfLastGridPointInDegrees", -999)))
                    if not np.allclose(probe, grid_spec):
                        raise ValueError("GRIB contains inconsistent spatial grids")

                values = np.asarray(eccodes.codes_get_values(msg), dtype=np.float64)
                if values.shape != (GRID_SHAPE[0] * GRID_SHAPE[1],):
                    raise ValueError(f"Unexpected grid values size {values.shape}")
                j = CUMULATIVE_HOURS.index(hour)
                k = FIELDS.index(field)
                selected = values[offsets]
                if not np.isfinite(selected).all():
                    raise ValueError(f"Masked/missing value at ocean points in {key}")
                cum_points[j, :, k] = selected
                if full_grid:
                    subgrid = values.reshape(GRID_SHAPE)[::2, ::2]
                    if not np.isfinite(subgrid).all():
                        raise ValueError(f"Nonfinite values on ERA5-size subgrid in {key}")
                    cum_grid[j, :, :, k] = subgrid
            finally:
                eccodes.codes_release(msg)

    expected = {(field, hour) for field in FIELDS for hour in CUMULATIVE_HOURS}
    if present != expected or selected_count != 30 or ref_time is None:
        raise ValueError(f"Incomplete TIGGE flux forecast: {selected_count}/30 daily-boundary "
                         f"records selected from {count}; missing={sorted(expected - present)}")
    means_points = cumulative_to_daily_mean(cum_points, day_offset)
    if np.abs(means_points).max() > 5000:
        raise ValueError("Implausible mean heat flux > 5000 W/m2; check GRIB accumulation and units")
    means_grid = cumulative_to_daily_mean(cum_grid, day_offset) if full_grid else None
    target_dates = np.array([(ref_time + timedelta(days=d+day_offset)).date().isoformat()
                             for d in range(14)], dtype="datetime64[D]")
    bounds = np.asarray([(24*(d+day_offset), 24*(d+day_offset+1)) for d in range(14)],
                        dtype=np.int32)
    stem = f"tigge_ecmwf_control_{ref_time:%Y%m%d_%H%M}_fluxes_mean_{offset_label}"
    path_points = out_dir / (stem + "_points.npz")
    path_grid = out_dir / (stem + "_era5grid_daily.npz") if full_grid else None
    path_meta = out_dir / (stem + "_metadata.json")
    outputs = [path_points, path_meta] + ([path_grid] if path_grid else [])
    if not overwrite and any(x.exists() for x in outputs):
        raise FileExistsError(f"Output exists (no overwrite): {[str(x) for x in outputs if x.exists()]}")

    np.savez_compressed(
        path_points,
        flux_daily_mean=means_points,
        target_dates=target_dates,
        interval_hours=bounds,
        fields=np.asarray(FIELDS),
        units=np.asarray(OUT_UNITS),
        point_ids=np.asarray([p["id"] for p in points]),
        point_lat_deg=np.asarray([p["lat_deg"] for p in points], dtype=np.float32),
        point_lon_deg=np.asarray([p["lon_deg"] for p in points], dtype=np.float32),
        initialization_utc=np.asarray(ref_time.isoformat().replace("+00:00", "Z")),
        day_offset=np.asarray(day_offset),
    )
    if full_grid:
        np.savez_compressed(
            path_grid,
            flux_daily_mean=means_grid, target_dates=target_dates,
            interval_hours=bounds, fields=np.asarray(FIELDS), units=np.asarray(OUT_UNITS),
            lat_deg=np.linspace(80., 0., 161, dtype=np.float32),
            lon_deg=np.linspace(-90., 0., 181, dtype=np.float32),
        )
    metadata = {
        "source_file": grib.name, "source_sha256": _sha256(grib),
        "forecast_issue_utc": ref_time.isoformat(),
        "day_offset": day_offset, "target_dates": target_dates.astype(str).tolist(),
        "integration_intervals_hours": bounds.tolist(),
        "grib_parameters": {"sensible": 146, "latent": 147},
        "source_units": "J m^-2 (W m^-2 s)", "target_units": "W m^-2",
        "formula": "daily mean = (cumulative_energy_at_end - cumulative_energy_at_start) / 86400",
        "sign_convention": "ECMWF positive downwards; unchanged",
        "n_messages": count, "selected_flux_messages": selected_count,
        "point_shape": list(means_points.shape),
        "points_file": points_json.name,
        "grid_shape": list(GRID_SHAPE),
        "era5_grid_shape": list(ERA5_GRID_SHAPE) if full_grid else None,
        "note": "This is a forecast, not ERA5; same units do not imply identical values.",
    }
    path_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {ref_time.isoformat()} {offset_label}: {means_points.shape} W/m2 -> {path_points}")
    return path_points, path_meta, path_grid


def main(argv: list[str] | None = None) -> int:
    """Run the TIGGE fluxes workflow using its configured options."""
    p = argparse.ArgumentParser(description="TIGGE accumulated sshf/slhf -> daily mean W/m2 for Atlantic 15 points")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--grib", type=Path, help="A single *_all.grib or *_fluxes.grib")
    group.add_argument("--raw-root", type=Path, help="Process combined and legacy flux GRIBs recursively")
    p.add_argument("--points-json", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--day-offset", type=int, choices=(0, 1), default=0,
                   help="0=UTC day of 00z issue (recommended); 1=following UTC day")
    p.add_argument("--full-grid", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args(argv)
    sources = ([args.grib] if args.grib else sorted(set(
        args.raw_root.rglob("*_all.grib")
    ) | set(args.raw_root.rglob("*_fluxes.grib"))))
    if not sources:
        p.error("No *_fluxes.grib files found")
    done = skipped = failed = 0
    for i, source in enumerate(sources, 1):
        stem = source.name.removesuffix(".grib")
        stamp = (stem.removeprefix("tigge_ecmwf_control_")
                 .removesuffix("_fluxes").removesuffix("_all"))
        label = "issue_day" if args.day_offset == 0 else "next_day"
        output_name = f"tigge_ecmwf_control_{stamp}_fluxes_mean_{label}_points.npz"
        output_base = output_name.removesuffix("_points.npz")
        expected_outputs = [args.out_dir / output_name,
                            args.out_dir / (output_base + "_metadata.json")]
        if args.full_grid:
            expected_outputs.append(args.out_dir / (output_base + "_era5grid_daily.npz"))
        if all(path.exists() for path in expected_outputs) and not args.overwrite:
            print(f"[{i}/{len(sources)}] SKIP complete {output_name}")
            skipped += 1
            continue
        try:
            convert_flux_grib(source, args.points_json, args.out_dir,
                              day_offset=args.day_offset, full_grid=args.full_grid,
                              overwrite=args.overwrite)
            done += 1
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(sources)}] ERROR {source}: {exc}")
            if not args.continue_on_error:
                raise
    print(f"Converted={done}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
