"""Download ECMWF TIGGE control forecasts from ECDS, resumably, one issue at a time.

Two *separate* GRIB2 files are kept for each issue so already downloaded
meteorology (5 instantaneous variables x 61 steps) is never re-downloaded when
adding the accumulated heat fluxes (2 variables x 15 steps).

    python -m Forecast_1d.download_tigge_ecds --start 2026-10-01 --end 2026-10-01 --only fluxes --execute

Docs: https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download
      https://confluence.ecmwf.int/spaces/TIGGE/pages/40109884/Parameters
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterator

DATASET = "tigge-forecasts"
API_URL = "https://ecds.ecmwf.int/api"
MET_VARIABLES = (
    "10_m_u_component_of_wind",
    "10_m_v_component_of_wind",
    "2_m_temperature",
    "2_m_dewpoint_temperature",
    "surface_pressure",
)
FLUX_VARIABLES = (
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
)
MET_HOURS = tuple(range(0, 361, 6))  # instantaneous forecast +0..+360 h
# Accumulated from forecast initialization. Steps 24..336 allow issue-day
# UTC means; +360 additionally allows the following 14 UTC days.
FLUX_HOURS = tuple(range(24, 361, 24))
EXPECTED_FIELDS = {"met": len(MET_VARIABLES) * len(MET_HOURS),
                   "fluxes": len(FLUX_VARIABLES) * len(FLUX_HOURS)}


def default_outdir() -> Path:
    """Compute default outdir for download TIGGE ECDS."""
    if os.name == "nt":
        return Path(r"D:\Nastya\Data\OceanFull\DATA\Forecast_1d\TIGGE\raw")
    return Path.cwd() / "tigge_raw"


def make_request(
    issued: date,
    area: tuple[float, float, float, float],
    grid: str,
    group: str = "met",
) -> dict:
    """Construct request for the download TIGGE ECDS workflow."""
    north, west, south, east = area
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 360):
        raise ValueError("Area must be [NORTH, WEST, SOUTH, EAST] in degrees")
    if group not in EXPECTED_FIELDS:
        raise ValueError(f"Unknown group {group!r}; choose 'met' or 'fluxes'")
    variables = MET_VARIABLES if group == "met" else FLUX_VARIABLES
    hours = MET_HOURS if group == "met" else FLUX_HOURS
    return {
        "origin": "ecmwf",
        "forecast_type": "control_forecast",
        "level_type": "single_level",
        "variable": list(variables),
        "year": [f"{issued.year:04d}"],
        "month": [f"{issued.month:02d}"],
        "day": [f"{issued.day:02d}"],
        "time": ["00:00"],
        "leadtime_hour": [str(hour) for hour in hours],
        "area": [north, west, south, east],
        "grid": grid,
        "data_format": "grib",
    }


def issue_dates(start: date, end: date, stride: int = 1) -> Iterator[date]:
    """Compute issue dates for download TIGGE ECDS from ``start``, ``end``, ``stride``."""
    if end < start or stride < 1:
        raise ValueError("End date must be >= start date and --every must be >= 1")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=stride)


def filename_for(issued: date, root: Path, group: str = "met") -> Path:
    """Compute filename for for download TIGGE ECDS from ``issued``, ``root``, ``group``."""
    if group not in EXPECTED_FIELDS:
        raise ValueError(f"Unknown group {group!r}")
    stem = f"tigge_ecmwf_control_{issued:%Y%m%d}_0000"
    suffix = "" if group == "met" else "_fluxes"
    return root / f"{issued.year:04d}" / f"{issued.month:02d}" / f"{stem}{suffix}.grib"


def validate_grib2(path: Path, expected_fields: int = EXPECTED_FIELDS["met"]) -> tuple[int, int]:
    """Validate concatenated GRIB2 message boundaries and message count.

    This is a structural check, not proof of exact GRIB parameter IDs/lead times.
    Inspect those using Data_processing.tigge_fluxes or the met converter.
    """
    count = 0
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        total = stream.tell()
        stream.seek(0)
        while stream.tell() < total:
            start = stream.tell()
            header = stream.read(16)
            if len(header) != 16 or header[:4] != b"GRIB" or header[7] != 2:
                raise ValueError(f"Invalid GRIB2 header at byte {start}, message {count+1}")
            size = int.from_bytes(header[8:16], "big")
            if size < 20 or start + size > total:
                raise ValueError(f"Invalid/truncated message length {size} at byte {start}")
            stream.seek(start + size - 4)
            if stream.read(4) != b"7777":
                raise ValueError(f"Missing GRIB2 trailer at message {count+1}")
            count += 1
        if stream.tell() != total or count != expected_fields:
            raise ValueError(f"Found {count} messages, expected {expected_fields}; bytes={total}")
    return count, total


def append_log(root: Path, item: dict) -> None:
    """Compute append log for download TIGGE ECDS from ``root``, ``item``."""
    root.mkdir(parents=True, exist_ok=True)
    record = dict(item)
    record["at_utc"] = datetime.now(timezone.utc).isoformat()
    with (root / "download_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_client():
    """Construct client for the download TIGGE ECDS workflow."""
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit("Install cdsapi: python -m pip install -U cdsapi") from exc
    token = os.environ.get("ECDS_API_KEY")
    if token:
        return cdsapi.Client(url=API_URL, key=token)
    return cdsapi.Client(url=API_URL)


def run(argv: list[str] | None = None) -> int:
    """Run the download TIGGE ECDS workflow using its configured options."""
    p = argparse.ArgumentParser(description="Resumable ECDS TIGGE ECMWF control + heat flux download")
    p.add_argument("--start", default="2025-10-31", help="First forecast issue date YYYY-MM-DD")
    p.add_argument("--end", default="2026-10-16", help="Last forecast issue date YYYY-MM-DD")
    p.add_argument("--only", choices=("both", "met", "fluxes"), default="both",
                   help="Download met, fluxes or both (default); existing valid files are skipped")
    p.add_argument("--every", type=int, default=1, metavar="N")
    p.add_argument("--max-dates", type=int, default=0)
    p.add_argument("--out", type=Path, default=default_outdir())
    p.add_argument("--area", type=float, nargs=4, metavar=("NORTH", "WEST", "SOUTH", "EAST"),
                   default=[80.0, -90.0, 0.0, 0.0])
    p.add_argument("--grid", default="0.25/0.25")
    p.add_argument("--execute", action="store_true", help="Submit downloads; otherwise dry-run")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args(argv)
    try:
        dates = list(issue_dates(date.fromisoformat(args.start), date.fromisoformat(args.end), args.every))
        if args.max_dates < 0 or args.retries < 0:
            p.error("--max-dates and --retries must be nonnegative")
        if args.max_dates:
            dates = dates[:args.max_dates]
        groups = ("met", "fluxes") if args.only == "both" else (args.only,)
        requests = {g: make_request(dates[0], tuple(args.area), args.grid, g) for g in groups}
    except ValueError as exc:
        p.error(str(exc))

    print(f"Dataset={DATASET} | issue=00 UTC | {len(dates)} issue dates: {dates[0]} .. {dates[-1]}")
    print(f"Groups={', '.join(groups)} | met={EXPECTED_FIELDS['met']} fields | fluxes={EXPECTED_FIELDS['fluxes']} fields")
    print(f"Region={args.area} | grid={args.grid} | output={args.out}")
    for group, request in requests.items():
        print(f"Example {group} request:")
        print(json.dumps(request, ensure_ascii=False, indent=2))
    if not args.execute:
        print("DRY RUN. Add --execute to submit ECDS requests.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    client = make_client()
    downloaded = skipped = failed = 0
    for index, issued in enumerate(dates, 1):
        for group in groups:
            target = filename_for(issued, args.out, group)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file():
                try:
                    count, size = validate_grib2(target, EXPECTED_FIELDS[group])
                except (OSError, ValueError) as exc:
                    print(f"[{index}/{len(dates)}] INVALID {group}: {target.name}: {exc}")
                    damaged = target.with_name(target.name + ".invalid")
                    if damaged.exists():
                        damaged = target.with_name(target.name + f".invalid.{int(time.time())}")
                    target.rename(damaged)
                else:
                    print(f"[{index}/{len(dates)}] SKIP {group}: {target.name} ({count} fields, {size:,} bytes)")
                    skipped += 1
                    continue
            working = target.with_name(target.name + ".downloading")
            request = make_request(issued, tuple(args.area), args.grid, group)
            last_error = None
            for attempt in range(1, args.retries + 2):
                try:
                    if working.exists():
                        working.unlink()
                    print(f"[{index}/{len(dates)}] REQUEST {issued} {group}, attempt {attempt}", flush=True)
                    client.retrieve(DATASET, request, str(working))
                    count, size = validate_grib2(working, EXPECTED_FIELDS[group])
                    working.replace(target)
                    append_log(args.out, {"date": issued.isoformat(), "group": group,
                                          "status": "ok", "file": str(target), "messages": count,
                                          "bytes": size, "request": request})
                    print(f"[{index}/{len(dates)}] DONE {group}: {size:,} bytes", flush=True)
                    downloaded += 1
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"[{index}/{len(dates)}] ERROR {issued} {group}, attempt {attempt}: {exc}",
                          file=sys.stderr, flush=True)
                    if working.exists():
                        working.unlink()
                    if attempt <= args.retries:
                        time.sleep(min(30 * (2 ** (attempt - 1)), 120))
            if last_error is not None:
                failed += 1
                append_log(args.out, {"date": issued.isoformat(), "group": group,
                                      "status": "failed", "error": repr(last_error), "request": request})
                if not args.continue_on_error:
                    print(f"Summary: downloaded={downloaded}, skipped={skipped}, failed={failed}")
                    return 1
    print(f"Summary: downloaded={downloaded}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
