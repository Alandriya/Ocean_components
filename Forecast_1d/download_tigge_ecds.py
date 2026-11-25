"""Download historical ECMWF TIGGE control forecasts from ECDS, one issue per GRIB.

No project imports needed. Requires cdsapi only in --execute mode.
Default is a dry run. Data credentials are read by cdsapi from ~/.cdsapirc,
or from the ECDS_API_KEY environment variable; never hardcode secrets.

Official dataset: https://ecds.ecmwf.int/datasets/tigge-forecasts
Official API:     https://ecds.ecmwf.int/how-to-api
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
VARIABLES = (
    "10_m_u_component_of_wind",
    "10_m_v_component_of_wind",
    "2_m_temperature",
    "2_m_dewpoint_temperature",
    "surface_pressure",
)
LEADTIME_HOURS = tuple(range(0, 361, 6))
EXPECTED_FIELDS = len(VARIABLES) * len(LEADTIME_HOURS)  # 305


def default_outdir() -> Path:
    if os.name == "nt":
        return Path(r"D:\Nastya\Data\OceanFull\DATA\Forecast_1d\TIGGE\raw")
    return Path.cwd() / "tigge_raw"


def make_request(issued: date, area: tuple[float, float, float, float], grid: str):
    north, west, south, east = area
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 360):
        raise ValueError("Area must be [North West South East] in geographic degrees")
    return {
        "origin": "ecmwf",
        "forecast_type": "control_forecast",
        "level_type": "single_level",
        "variable": list(VARIABLES),
        "year": [f"{issued.year:04d}"],
        "month": [f"{issued.month:02d}"],
        "day": [f"{issued.day:02d}"],
        "time": ["00:00"],
        "leadtime_hour": [str(hour) for hour in LEADTIME_HOURS],
        "area": [north, west, south, east],
        "grid": grid,
        "data_format": "grib",
    }


def issue_dates(start: date, end: date, stride: int = 1) -> Iterator[date]:
    if end < start:
        raise ValueError("End date precedes start date")
    if stride < 1:
        raise ValueError("Stride must be a positive number of days")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=stride)


def filename_for(issued: date, root: Path) -> Path:
    return root / f"{issued.year:04d}" / f"{issued.month:02d}" / (
        f"tigge_ecmwf_control_{issued:%Y%m%d}_0000.grib"
    )


def validate_grib2(path: Path, expected_fields: int = EXPECTED_FIELDS) -> tuple[int, int]:
    """Check every GRIB2 record has a valid byte length and final 7777 marker.

    Structural check, not a complete check of variable IDs/coordinates; run the
    existing convert_tigge_control.py for full scientific validation.
    """
    n = 0
    with path.open("rb") as stream:
        stream.seek(0, 2)
        total = stream.tell()
        stream.seek(0)
        while stream.tell() < total:
            header = stream.read(16)
            if len(header) != 16 or header[:4] != b"GRIB" or header[7] != 2:
                raise ValueError(f"Invalid GRIB2 header at message {n+1}")
            size = int.from_bytes(header[8:16], "big")
            if size < 20 or stream.tell() + size - 16 > total:
                raise ValueError(f"Bad or truncated GRIB2 message {n+1}: {size} bytes")
            stream.seek(size - 20, 1)
            if stream.read(4) != b"7777":
                raise ValueError(f"Missing GRIB2 message trailer at record {n+1}")
            n += 1
        if stream.tell() != total or n != expected_fields:
            raise ValueError(
                f"Unexpected GRIB2 size/count: {n} messages, expected {expected_fields}"
            )
    return n, total


def append_log(root: Path, item: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    item = dict(item)
    item["at_utc"] = datetime.now(timezone.utc).isoformat()
    with (root / "download_log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def make_client():
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit(
            "Missing cdsapi. Install: C:\\Python\\python.exe -m pip install -U cdsapi"
        ) from exc
    token = os.environ.get("ECDS_API_KEY")
    # cdsapi reads ~/.cdsapirc when token is absent; URL forces ECDS, not CDS.
    if token:
        return cdsapi.Client(url=API_URL, key=token)
    return cdsapi.Client(url=API_URL)


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ECDS TIGGE 00 UTC control, 5 fields, 0..360h")
    p.add_argument("--start", default="2025-10-30", help="First forecast issue date YYYY-MM-DD")
    p.add_argument("--end", default="2026-10-15", help="Last forecast issue date YYYY-MM-DD")
    p.add_argument("--every", type=int, default=1, metavar="N", help="One issue every N days (default 1)")
    p.add_argument("--max-dates", type=int, default=0, help="Limit to first N dates; 0=all")
    p.add_argument("--out", type=Path, default=default_outdir(), help="Output directory")
    p.add_argument(
        "--area", type=float, nargs=4, metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=[80.0, -90.0, 0.0, 0.0],
        help="Area [N W S E], defaults to the 80N..0N, 90W..0E study map",
    )
    p.add_argument("--grid", default="0.25/0.25", help="Output regular grid resolution")
    p.add_argument("--execute", action="store_true", help="Actually submit API requests; default only prints a dry run")
    p.add_argument("--retries", type=int, default=2, help="Retries after first API failure (default 2)")
    p.add_argument("--continue-on-error", action="store_true", help="Continue to next date on failure")
    args = p.parse_args(argv)
    try:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        dates = list(issue_dates(start, end, args.every))
        if args.max_dates < 0 or args.retries < 0:
            p.error("--max-dates and --retries must be nonnegative")
        if args.max_dates:
            dates = dates[:args.max_dates]
        if not dates:
            p.error("There are no dates to download")
        template = make_request(dates[0], tuple(args.area), args.grid)
    except ValueError as exc:
        p.error(str(exc))

    print(f"Dataset: {DATASET}, ECMWF control, issue 00 UTC")
    print(f"Dates: {dates[0].isoformat()} .. {dates[-1].isoformat()}, total {len(dates)}")
    print(f"Variables: {', '.join(VARIABLES)}")
    print(f"Steps: 0..360 hours every 6h; {EXPECTED_FIELDS} GRIB2 messages/day")
    print(f"Area [N W S E]: {args.area}; grid: {args.grid}; output: {args.out}")
    print("Representative ECDS request:")
    print(json.dumps(template, ensure_ascii=False, indent=2))
    if not args.execute:
        print("DRY RUN ONLY. Add --execute to download. Use --max-dates 2 for a first pilot.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    client = make_client()
    skipped = downloaded = failed = 0
    for index, issued in enumerate(dates, 1):
        target = filename_for(issued, args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                count, size = validate_grib2(target)
            except (OSError, ValueError) as exc:
                print(f"[{index}/{len(dates)}] Invalid existing {target.name}: {exc}; re-fetching")
                # Preserve questionable existing data rather than quietly overwriting.
                damaged = target.with_name(target.name + ".invalid")
                if damaged.exists():
                    damaged = target.with_name(target.name + f".invalid.{int(time.time())}")
                target.rename(damaged)
            else:
                print(f"[{index}/{len(dates)}] SKIP {target.name} ({count} fields, {size:,} bytes)")
                skipped += 1
                continue
        working = target.with_name(target.name + ".downloading")
        request = make_request(issued, tuple(args.area), args.grid)
        last_error = None
        for attempt in range(1, args.retries + 2):
            try:
                if working.exists():
                    working.unlink()
                print(f"[{index}/{len(dates)}] REQUEST {issued} (attempt {attempt})", flush=True)
                client.retrieve(DATASET, request, str(working))
                n, size = validate_grib2(working)
                working.replace(target)
                append_log(args.out, {"date": issued.isoformat(), "status": "ok", "file": str(target), "messages": n, "bytes": size, "request": request})
                print(f"[{index}/{len(dates)}] DONE {target.name}: {n} GRIB fields, {size:,} bytes", flush=True)
                downloaded += 1
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(f"[{index}/{len(dates)}] ERROR {issued} attempt {attempt}: {exc}", file=sys.stderr, flush=True)
                if working.exists():
                    working.unlink()
                if attempt <= args.retries:
                    pause = min(30 * (2 ** (attempt - 1)), 120)
                    print(f"Retrying after {pause} seconds...", flush=True)
                    time.sleep(pause)
        if last_error is not None:
            failed += 1
            append_log(args.out, {"date": issued.isoformat(), "status": "failed", "error": repr(last_error), "request": request})
            if not args.continue_on_error:
                print("Stopped after failed request. Re-run to resume from completed files.", file=sys.stderr)
                print(f"Summary: downloaded={downloaded}, skipped={skipped}, failed={failed}")
                return 1
    print(f"Summary: downloaded={downloaded}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
