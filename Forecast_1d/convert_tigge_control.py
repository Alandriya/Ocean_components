"""Validate ECMWF TIGGE GRIB2 and export 6h/daily forecasts for Atlantic 15 points.

Dependencies: numpy, rasterio (GDAL GRIB driver).
No modification of historic ERA5 arrays or neural-network models.

Usage:
  python -m Forecast_1d.convert_tigge_control --grib FILE.grib --points-json DATASET/points.json --out-dir OUTPUT [--full-grid]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

FIELDS = ("u10", "v10", "t2m", "d2m", "sp")
UNITS = ("m/s", "m/s", "K", "K", "Pa")
# GDAL's GRIB 2 raster band labels, including the level.
GRIB_CODES = {
    ("UGRD", "10-HTGL"): "u10",
    ("VGRD", "10-HTGL"): "v10",
    ("TMP", "2-HTGL"): "t2m",
    ("DPT", "2-HTGL"): "d2m",
    ("PRES", "0-SFC"): "sp",
}
EXPECTED_HOURS = tuple(range(0, 361, 6))


def _sha256(path: Path) -> str:
    """Calculate a file SHA-256 checksum."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _as_k(value, unit: str, field: str):
    """Convert GDAL-decoded 2 m temperature/dew-point fields to kelvin.

    Other parameters are returned unchanged; unexpected temperature units raise
    ``ValueError`` rather than silently corrupting meteorological inputs."""
    if field not in ("t2m", "d2m"):
        return value
    if unit == "[C]":
        return value + 273.15
    if unit == "[K]":
        return value
    raise ValueError(f"Unexpected temperature unit for {field}: {unit}")


def convert(grib: Path, points_json: Path, out_dir: Path, full_grid: bool = False, day_offset: int = 0):
    """Convert TIGGE control meteorology to daily and six-hourly arrays.

    Accept either an original meteorology-only GRIB or a combined seven-variable
    forecast GRIB. Extract the five atmospheric parameters without interpreting
    accumulated heat-flux messages as instantaneous meteorological fields.
    The ``day_offset`` determines how target dates align with the 00 UTC issue."""
    if day_offset not in (0, 1):
        raise ValueError("day_offset must be 0 (issue day) or 1 (following day)")
    try:
        import rasterio
    except ImportError as exc:
        raise SystemExit("Missing rasterio. Install with: python -m pip install rasterio numpy") from exc

    grib = Path(grib).expanduser().resolve()
    points_json = Path(points_json).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sites = json.loads(points_json.read_text(encoding="utf-8"))["points"]
    if len(sites) != 15 or len({s["id"] for s in sites}) != 15:
        raise ValueError("Expected exactly 15 distinct points from the original dataset points.json")

    with rasterio.Env(GRIB_NORMALIZE_UNITS="YES"):
        with rasterio.open(grib) as ds:
            if ds.driver != "GRIB":
                raise ValueError(f"Expected GRIB, received {ds.driver}")
            # The original ERA5 grid covers 0..80 N and -90..0 E at 0.5-degree spacing.
            if (ds.height, ds.width) != (321, 361) or not np.allclose(ds.res, (0.25, 0.25)):
                raise ValueError(f"Unexpected TIGGE grid {ds.height}x{ds.width}, resolution {ds.res}; expected 321x361 at 0.25°")
            if not np.allclose(ds.xy(0, 0), (-90., 80.), atol=1e-5) or not np.allclose(ds.xy(ds.height-1, ds.width-1), (0., 0.), atol=1e-5):
                raise ValueError("GRIB grid corners do not match original Atlantic domain")
            # Unified GRIB files additionally contain accumulated sshf/slhf.
            # The five mandatory meteorological series are validated below.
            if ds.count < len(EXPECTED_HOURS) * len(FIELDS):
                raise ValueError(f"Expected at least 305 fields, got {ds.count}")

            locations = []
            for p in sites:
                lat, lon = float(p["lat_deg"]), float(p["lon_deg"])
                r, c = ds.index(lon, lat)
                if not (0 <= r < ds.height and 0 <= c < ds.width):
                    raise ValueError(f"Point outside GRIB domain: {p['id']} ({lat}, {lon})")
                x, y = ds.xy(r, c)
                if abs(x-lon) > 1e-5 or abs(y-lat) > 1e-5:
                    raise ValueError(f"Point {p['id']} is not on a forecast grid node: ({lat},{lon}), nearest ({y},{x})")
                if "grid_y" in p and "grid_x" in p:
                    old_r, old_c = int(p["grid_y"]), int(p["grid_x"])
                    if r != 2*old_r or c != 2*old_c:
                        raise ValueError(f"Point {p['id']} ERA5 index does not match TIGGE grid: ({r},{c}) vs twice ({old_r},{old_c})")
                locations.append((r, c))

            bandmap = {}
            ref_times = set()
            for bandno in range(1, ds.count + 1):
                info = ds.tags(bandno)
                key = (info.get("GRIB_ELEMENT"), info.get("GRIB_SHORT_NAME"))
                if key not in GRIB_CODES:
                    # Skip accumulated flux bands present in the combined GRIB.
                    if key in (("SHTFL", "0-SFC"), ("LHTFL", "0-SFC")):
                        continue
                    raise ValueError(f"Unexpected field {key} in band {bandno}")
                if "CENTER=98(ECMWF)" not in info.get("GRIB_IDS", "") or "TYPE=3(Control_forecast)" not in info.get("GRIB_IDS", ""):
                    raise ValueError(f"Band {bandno} is not ECMWF TIGGE control forecast")
                field = GRIB_CODES[key]
                seconds = int(info["GRIB_FORECAST_SECONDS"])
                if seconds % 3600:
                    raise ValueError(f"Band {bandno} step is not an integer number of hours")
                hour = seconds // 3600
                ts_ref = int(info["GRIB_REF_TIME"])
                ts_valid = int(info["GRIB_VALID_TIME"])
                if ts_valid != ts_ref + seconds:
                    raise ValueError(f"Band {bandno} reference+lead time does not equal valid time")
                ref_times.add(ts_ref)
                idx = (hour, field)
                if idx in bandmap:
                    raise ValueError(f"Duplicate hour/field {idx}")
                bandmap[idx] = (bandno, info["GRIB_UNIT"])
            if len(ref_times) != 1:
                raise ValueError(f"GRIB file contains multiple initialization times: {ref_times}")
            for hour in EXPECTED_HOURS:
                for field in FIELDS:
                    if (hour, field) not in bandmap:
                        raise ValueError(f"Missing {field} at +{hour}h")

            reference = datetime.fromtimestamp(next(iter(ref_times)), tz=timezone.utc)
            if reference.hour != 0 or reference.minute != 0:
                raise ValueError("This converter's day grouping requires a 00 UTC forecast initialization")

            point_6h = np.empty((len(EXPECTED_HOURS), len(sites), len(FIELDS)), dtype=np.float32)
            grid_6h = np.empty((len(EXPECTED_HOURS), 161, 181, len(FIELDS)), dtype=np.float32) if full_grid else None
            for ih, hour in enumerate(EXPECTED_HOURS):
                for iv, field in enumerate(FIELDS):
                    bandno, unit = bandmap[(hour, field)]
                    raster = ds.read(bandno, masked=True)
                    vals = np.asarray(raster.data)
                    if np.any(raster.mask) and any(bool(np.asarray(raster.mask)[r, c]) for r, c in locations):
                        raise ValueError(f"Masked point in {field}+{hour}h")
                    picked = np.array([vals[r, c] for r, c in locations], dtype=np.float64)
                    picked = _as_k(picked, unit, field)
                    if not np.isfinite(picked).all():
                        raise ValueError(f"Non-finite point data in {field}+{hour}h")
                    point_6h[ih, :, iv] = picked
                    if full_grid:
                        subset = np.ma.asarray(raster)[::2, ::2]
                        if np.ma.is_masked(subset) and np.ma.count_masked(subset):
                            raise ValueError(f"Masked values on 0.5° grid in {field}+{hour}h")
                        grid_6h[ih, :, :, iv] = _as_k(np.asarray(subset, dtype=np.float64), unit, field)

    # Issue 00 UTC: day_offset=0 -> issue-day [00,06,12,18] UTC.
    # day_offset=1 reproduces the previous converter's next-day convention.
    daily_steps = np.array([[24*d, 24*d+6, 24*d+12, 24*d+18]
                            for d in range(day_offset, day_offset + 14)], dtype=np.int32)
    daily_indices = daily_steps // 6
    point_daily = point_6h[daily_indices].mean(axis=1).astype(np.float32)
    dates = np.array([(reference + timedelta(days=d)).date().isoformat()
                      for d in range(day_offset, day_offset + 14)], dtype="datetime64[D]")
    valid_time = np.array([np.datetime64(reference.replace(tzinfo=None), "h") + np.timedelta64(h, "h") for h in EXPECTED_HOURS])
    stamp = reference.strftime("%Y%m%d_%H%M")
    # Keep legacy filename for day_offset=1; avoid silently overwriting old NPZ
    # when creating the scientifically better aligned issue-day convention.
    name = f"tigge_ecmwf_control_{stamp}" + ("_issue_day" if day_offset == 0 else "")
    points_path = out_dir / f"{name}_points.npz"
    np.savez_compressed(
        points_path,
        forecast_6h=point_6h, forecast_daily=point_daily,
        forecast_hours=np.asarray(EXPECTED_HOURS, dtype=np.int32),
        valid_times_utc=valid_time, target_dates=dates,
        point_ids=np.array([p["id"] for p in sites]),
        point_lat_deg=np.array([p["lat_deg"] for p in sites], dtype=np.float32),
        point_lon_deg=np.array([p["lon_deg"] for p in sites], dtype=np.float32),
        fields=np.array(FIELDS), units=np.array(UNITS),
        initialization_utc=np.array(reference.isoformat().replace("+00:00", "Z")),
        day_offset=np.array(day_offset),
    )
    meta = {
        "source_grib_filename": grib.name,
        "source_grib_sha256": _sha256(grib),
        "initialization_utc": reference.isoformat(),
        "forecast_type": "ECMWF TIGGE control forecast (NOT ERA5)",
        "horizon": "14 UTC days starting at issue day + day_offset",
        "day_offset": day_offset,
        "point_forecast_6h_shape": list(point_6h.shape),
        "point_forecast_daily_shape": list(point_daily.shape),
        "point_6h_hours": list(EXPECTED_HOURS),
        "daily_sampling_hours": daily_steps.tolist(),
        "target_dates": dates.astype(str).tolist(),
        "fields": list(FIELDS),
        "units": list(UNITS),
        "point_ids": [p["id"] for p in sites],
        "points_file": points_json.name,
        "grid": {"latitudes": "80..0 by -0.25", "longitudes": "-90..0 by +0.25", "shape": [321,361]},
        "daily_mean_definition": "mean of 00,06,12,18 UTC instantaneous samples for each UTC target date",
        "note": "forecast_6h retains 6h data; nonlinear bulk flux proxies should be computed per 6h step, then daily averaged. Do NOT use future ERA5 observations as forecast input.",
    }
    grid_path = None
    if full_grid:
        grid_daily = grid_6h[daily_indices].mean(axis=1).astype(np.float32)
        grid_path = out_dir / f"{name}_era5grid_daily.npz"
        np.savez_compressed(
            grid_path, forecast_daily=grid_daily,
            target_dates=dates, fields=np.array(FIELDS), units=np.array(UNITS),
            lat_deg=np.linspace(80.0, 0.0, 161, dtype=np.float32),
            lon_deg=np.linspace(-90.0, 0.0, 181, dtype=np.float32),
        )
        meta["full_grid_daily_shape"] = list(grid_daily.shape)
    meta_path = out_dir / f"{name}_metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Input GRIB validated:", grib)
    print("Initialization:", reference.isoformat(), "| 5 met fields x 61 steps | 15 points | day_offset=", day_offset)
    print("Daily:", point_daily.shape, str(dates[0]), "..", str(dates[-1]))
    print("Output:", points_path, "\nMetadata:", meta_path)
    if grid_path:
        print("Full ERA5-resolution grid:", grid_path)
    return points_path, meta_path, grid_path


def main():
    """Run the convert TIGGE control workflow using its configured options."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grib", required=True, type=Path, help="Input ECMWF TIGGE control GRIB file")
    p.add_argument("--points-json", required=True, type=Path, help="Existing dataset points.json with 15 ordered sites")
    p.add_argument("--out-dir", required=True, type=Path, help="Directory for output forecasts")
    p.add_argument("--full-grid", action="store_true", help="Additionally save full Atlantic region daily data at original ERA5 0.5-degree grid")
    p.add_argument("--day-offset", type=int, choices=(0, 1), default=0,
                   help="0=issue UTC day (default); 1=following UTC day (legacy)")
    args = p.parse_args()
    convert(args.grib, args.points_json, args.out_dir, args.full_grid, args.day_offset)


if __name__ == "__main__":
    main()
