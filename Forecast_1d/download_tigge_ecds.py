"""Resumable ECMWF TIGGE control download with server-side asynchronous jobs.

Modes:
  submit  - submit jobs and store IDs; do not wait for tape retrieval
  collect - check submitted IDs and download only completed jobs
  sync    - legacy blocking cdsapi workflow (backwards-compatible default)

Uses the same GRIB file names, ECDS request dictionaries and expected field counts
as the original downloader. The JSON registry is written atomically under --out.

Official client: https://github.com/ecmwf/ecmwf-datastores-client
Dataset: https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download
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
    "10_m_u_component_of_wind", "10_m_v_component_of_wind",
    "2_m_temperature", "2_m_dewpoint_temperature", "surface_pressure",
)
FLUX_VARIABLES = ("surface_sensible_heat_flux", "surface_latent_heat_flux")
MET_HOURS = tuple(range(0, 361, 6))
FLUX_HOURS = tuple(range(24, 361, 24))
EXPECTED_FIELDS = {"met": len(MET_VARIABLES) * len(MET_HOURS),
                   "fluxes": len(FLUX_VARIABLES) * len(FLUX_HOURS)}
REGISTRY_NAME = "download_jobs.json"


def default_outdir() -> Path:
    if os.name == "nt":
        return Path(r"D:\Nastya\Data\OceanFull\DATA\Forecast_1d\TIGGE\raw")
    return Path.cwd() / "tigge_raw"


def make_request(issued: date, area: tuple[float, float, float, float],
                 grid: str, group: str = "met") -> dict:
    north, west, south, east = area
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 360):
        raise ValueError("Area must be [NORTH, WEST, SOUTH, EAST] in degrees")
    if group not in EXPECTED_FIELDS:
        raise ValueError(f"Unknown group {group!r}")
    variables = MET_VARIABLES if group == "met" else FLUX_VARIABLES
    hours = MET_HOURS if group == "met" else FLUX_HOURS
    return {
        "origin": "ecmwf", "forecast_type": "control_forecast",
        "level_type": "single_level", "variable": list(variables),
        "year": [f"{issued.year:04d}"], "month": [f"{issued.month:02d}"],
        "day": [f"{issued.day:02d}"], "time": ["00:00"],
        "leadtime_hour": [str(hour) for hour in hours],
        "area": [north, west, south, east], "grid": grid,
        "data_format": "grib",
    }


def issue_dates(start: date, end: date, stride: int = 1) -> Iterator[date]:
    if end < start or stride < 1:
        raise ValueError("End date must be >= start date and --every must be >= 1")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=stride)


def filename_for(issued: date, root: Path, group: str = "met") -> Path:
    if group not in EXPECTED_FIELDS:
        raise ValueError(f"Unknown group {group!r}")
    stem = f"tigge_ecmwf_control_{issued:%Y%m%d}_0000"
    return root / f"{issued.year:04d}" / f"{issued.month:02d}" / (
        f"{stem}{'_fluxes' if group == 'fluxes' else ''}.grib")


def validate_grib2(path: Path, expected_fields: int = EXPECTED_FIELDS["met"]) -> tuple[int, int]:
    """Validate GRIB2 boundaries/count; field IDs and dates require decoder-level checks."""
    count = 0
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        total = stream.tell()
        stream.seek(0)
        while stream.tell() < total:
            begin = stream.tell()
            header = stream.read(16)
            if len(header) != 16 or header[:4] != b"GRIB" or header[7] != 2:
                raise ValueError(f"Bad GRIB2 header at byte {begin}")
            length = int.from_bytes(header[8:16], "big")
            if length < 20 or begin + length > total:
                raise ValueError(f"Invalid/truncated GRIB2 message length {length} at {begin}")
            stream.seek(begin + length - 4)
            if stream.read(4) != b"7777":
                raise ValueError(f"Missing GRIB2 trailer at {begin}")
            count += 1
        if stream.tell() != total or count != expected_fields:
            raise ValueError(f"Found {count} messages, expected {expected_fields}; bytes={total}")
    return count, total


def append_log(root: Path, item: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    obj = dict(item, at_utc=datetime.now(timezone.utc).isoformat())
    with (root / "download_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _read_registry(root: Path) -> dict:
    path = root / REGISTRY_NAME
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Bad registry file {path}; expected JSON object")
    return obj


def _write_registry(root: Path, registry: dict) -> None:
    """Atomic state update; never leave a partially written registry on Windows."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / REGISTRY_NAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _job_key(issued: date, group: str) -> str:
    return f"{issued.isoformat()}:{group}"


def _credentials_from_cdsapirc() -> str | None:
    """Reuse the token already configured for the original cdsapi downloader."""
    cfg = Path.home() / ".cdsapirc"
    if not cfg.is_file():
        return None
    for row in cfg.read_text(encoding="utf-8").splitlines():
        key, _, value = row.partition(":")
        if key.strip() == "key":
            return value.strip().strip('"\'')
    return None


def make_async_client():
    try:
        from ecmwf.datastores import Client
    except ImportError as exc:
        raise SystemExit(
            "Install the official async client: C:\\Python\\python.exe -m pip install -U ecmwf-datastores-client"
        ) from exc
    token = os.environ.get("ECMWF_DATASTORES_KEY") or os.environ.get("ECDS_API_KEY") or _credentials_from_cdsapirc()
    if token:
        return Client(url=API_URL, key=token)
    # Client can read ~/.ecmwfdatastoresrc with key, if present.
    return Client(url=API_URL)


def make_sync_client():
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit("Install cdsapi: C:\\Python\\python.exe -m pip install -U cdsapi") from exc
    token = os.environ.get("ECDS_API_KEY")
    return cdsapi.Client(url=API_URL, key=token) if token else cdsapi.Client(url=API_URL)


def _is_valid_local(target: Path, group: str) -> bool:
    if not target.is_file():
        return False
    try:
        validate_grib2(target, EXPECTED_FIELDS[group])
        return True
    except (OSError, ValueError):
        return False


def _handle_bad_local(target: Path, group: str) -> None:
    if target.is_file() and not _is_valid_local(target, group):
        bad = target.with_name(target.name + ".invalid")
        if bad.exists():
            bad = target.with_name(target.name + f".invalid.{int(time.time())}")
        target.rename(bad)
        print(f"Renamed invalid GRIB to {bad}")


def _retrieve_sync(client, request, target, group, issued, root, retries: int):
    target.parent.mkdir(parents=True, exist_ok=True)
    working = target.with_name(target.name + ".downloading")
    for attempt in range(1, retries + 2):
        try:
            working.unlink(missing_ok=True)
            client.retrieve(DATASET, request, str(working))
            count, size = validate_grib2(working, EXPECTED_FIELDS[group])
            working.replace(target)
            append_log(root, {"date": issued.isoformat(), "group": group, "status": "ok",
                              "file": str(target), "messages": count, "bytes": size})
            return True
        except Exception as exc:
            working.unlink(missing_ok=True)
            print(f"ERROR {issued} {group} attempt {attempt}: {exc}", file=sys.stderr)
            append_log(root, {"date": issued.isoformat(), "group": group,
                              "status": "error", "error": str(exc)})
            if attempt <= retries:
                time.sleep(min(30 * 2 ** (attempt - 1), 120))
    return False


def _count_active_server_jobs(client) -> int:
    """Count all accepted/running ECDS jobs on this account (not just this script).

    Fail closed on any API/listing error: never submit without knowing how many
    jobs already occupy server-side slots, including jobs from the web portal.
    """
    jobs = client.get_jobs(limit=100, sortby="-created", status=["accepted", "running"])
    ids = set()
    pages = 0
    while jobs is not None:
        pages += 1
        if pages > 10000:
            raise RuntimeError("Unexpectedly large/looping ECDS jobs pagination; submission halted")
        ids.update(str(job_id) for job_id in jobs.request_ids)
        jobs = jobs.next
    return len(ids)


def _submit_async(client, root, dates, groups, area, grid,
                  max_submits: int, max_active: int) -> int:
    registry = _read_registry(root)
    submitted = skipped = 0
    # The maximum includes manually submitted jobs and submissions from previous runs.
    # An API failure means we cannot assert compliance; stop rather than guessing.
    try:
        active = _count_active_server_jobs(client)
    except Exception as exc:
        print(f"Cannot verify account-wide active ECDS jobs: {exc}", file=sys.stderr)
        print("No new jobs were submitted. Inspect My Requests and retry later.", file=sys.stderr)
        return 1
    print(f"Server-side active jobs (accepted/running): {active}; permitted by this run: {max_active}")
    if active >= max_active:
        print("Submission paused. Use --mode collect to retrieve ready jobs; "
              "run --mode submit again after an active job finishes.")
        return 0

    for issued in dates:
        for group in groups:
            key = _job_key(issued, group)
            target = filename_for(issued, root, group)
            if _is_valid_local(target, group):
                skipped += 1
                print(f"SKIP local {key}")
                continue
            _handle_bad_local(target, group)
            # Never silently resubmit *any* job with a known request_id,
            # including failed and rejected ones. Review those manually.
            if key in registry and registry[key].get("request_id"):
                skipped += 1
                print(f"SKIP previously submitted {key}: {registry[key]['request_id']}")
                continue
            if submitted >= max_submits or active >= max_active:
                print(f"Submission paused: submitted={submitted}/{max_submits}; "
                      f"active={active}/{max_active}. No remaining dates were submitted.")
                print(f"Summary: submitted={submitted}, skipped={skipped}")
                return 0

            request = make_request(issued, area, grid, group)
            try:
                remote = client.submit(DATASET, request)
                job_id = str(remote.request_id)
                registry[key] = {
                    "request_id": job_id, "date": issued.isoformat(), "group": group,
                    "file": str(target), "request": request, "status": str(remote.status),
                    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _write_registry(root, registry)  # atomic, before attempting any other job
                append_log(root, {"date": issued.isoformat(), "group": group,
                                  "status": "submitted", "request_id": job_id})
                submitted += 1
                active += 1
                print(f"SUBMITTED {key}: {job_id}", flush=True)
            except Exception as exc:
                print(f"ERROR submitting {key}: {exc}", file=sys.stderr)
                print("Do NOT retry submission blindly: check My Requests first, "
                      "because the server may have accepted the job before disconnection.",
                      file=sys.stderr)
                append_log(root, {"date": issued.isoformat(), "group": group,
                                  "status": "submit_error", "error": str(exc)})
                return 1
    print(f"Summary: submitted={submitted}, skipped={skipped}")
    return 0


def _collect_async(client, root, dates, groups) -> int:
    registry = _read_registry(root)
    ready = pending = failed = downloaded = 0
    keys = {_job_key(d, group) for d in dates for group in groups}
    filtered = [(k, item) for k, item in registry.items() if k in keys and item.get("request_id")]
    if not filtered:
        print(f"No submitted jobs in {root / REGISTRY_NAME} for requested dates and groups")
        return 0
    for key, item in filtered:
        group = item["group"]
        target = filename_for(date.fromisoformat(item["date"]), root, group)
        if _is_valid_local(target, group):
            item["status"] = "downloaded"
            print(f"SKIP local {key}")
            continue
        _handle_bad_local(target, group)
        try:
            remote = client.get_remote(item["request_id"])
            status = str(remote.status)
            item["status"] = status
            _write_registry(root, registry)
            if status in ("failed", "rejected", "dismissed"):
                print(f"FAILED server job {key}: {status}")
                failed += 1
                continue
            if not remote.results_ready:
                pending += 1
                print(f"PENDING {key}: {status}")
                continue
            ready += 1
            target.parent.mkdir(parents=True, exist_ok=True)
            working = target.with_name(target.name + ".downloading")
            working.unlink(missing_ok=True)
            # The remote already completed. No waiting on MARS tapes here.
            remote.download(str(working))
            count, size = validate_grib2(working, EXPECTED_FIELDS[group])
            working.replace(target)
            item["status"] = "downloaded"
            _write_registry(root, registry)
            append_log(root, {"date": item["date"], "group": group, "status": "ok",
                              "request_id": item["request_id"], "file": str(target),
                              "messages": count, "bytes": size})
            downloaded += 1
            print(f"DOWNLOADED {key}: {size:,} bytes", flush=True)
        except Exception as exc:
            print(f"ERROR collecting {key}: {exc}", file=sys.stderr)
            append_log(root, {"date": item["date"], "group": group,
                              "status": "collect_error", "error": str(exc)})
            failed += 1
            _write_registry(root, registry)
    print(f"Summary: jobs={len(filtered)} ready={ready} downloaded={downloaded} "
          f"pending={pending} errors={failed}")
    return 1 if failed else 0


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumable TIGGE control forecasts and heat fluxes")
    parser.add_argument("--start", default="2025-10-31", help="First forecast issue YYYY-MM-DD")
    parser.add_argument("--end", default="2026-10-16", help="Last forecast issue YYYY-MM-DD")
    parser.add_argument("--only", choices=("both", "met", "fluxes"), default="both")
    parser.add_argument("--mode", choices=("sync", "submit", "collect"), default="sync",
                        help="sync=old blocking; submit=send requests only; collect=fetch ready jobs")
    parser.add_argument("--max-submits", type=int, default=1,
                        help="For submit: maximum NEW jobs in this run (default 1)")
    parser.add_argument("--max-active", type=int, default=1,
                        help="Maximum server-side jobs accepted/running on this account; "
                             "default 1 per ECMWF sequential-submission advice, max 3")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-dates", type=int, default=0)
    parser.add_argument("--out", type=Path, default=default_outdir())
    parser.add_argument("--area", type=float, nargs=4,
                        metavar=("NORTH", "WEST", "SOUTH", "EAST"), default=[80., -90., 0., 0.])
    parser.add_argument("--grid", default="0.25/0.25")
    parser.add_argument("--execute", action="store_true", help="Otherwise show dry-run")
    parser.add_argument("--retries", type=int, default=2, help="Only affects sync mode")
    parser.add_argument("--continue-on-error", action="store_true", help="Only affects sync mode")
    args = parser.parse_args(argv)
    if args.max_submits < 1 or args.max_dates < 0 or args.retries < 0:
        parser.error("--max-submits must be >=1; --max-dates and --retries must be nonnegative")
    if not 1 <= args.max_active <= 3:
        parser.error("--max-active must be in 1..3 (ECMWF recommends sequential requests)")
    try:
        dates = list(issue_dates(date.fromisoformat(args.start), date.fromisoformat(args.end), args.every))
        groups = ("met", "fluxes") if args.only == "both" else (args.only,)
        if args.max_dates:
            dates = dates[:args.max_dates]
        _ = make_request(dates[0], tuple(args.area), args.grid, groups[0])
    except (ValueError, IndexError) as exc:
        parser.error(str(exc))
    print(f"Dataset={DATASET}, mode={args.mode}; {len(dates)} issue dates "
          f"({dates[0]} to {dates[-1]}), groups={groups}")
    print(f"Output={args.out}; area={args.area}; grid={args.grid}")
    if not args.execute:
        for group in groups:
            print(f"Example {group}: " + json.dumps(
                make_request(dates[0], tuple(args.area), args.grid, group), ensure_ascii=False))
        print(f"DRY RUN: max-active={args.max_active}, max-new-submits={args.max_submits}; add --execute")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode in ("submit", "collect"):
        client = make_async_client()
        if args.mode == "submit":
            return _submit_async(client, args.out, dates, groups,
                                 tuple(args.area), args.grid,
                                 args.max_submits, args.max_active)
        return _collect_async(client, args.out, dates, groups)
    client = make_sync_client()
    succeeded = skipped = errors = 0
    for index, issued in enumerate(dates, 1):
        for group in groups:
            target = filename_for(issued, args.out, group)
            if _is_valid_local(target, group):
                skipped += 1
                print(f"[{index}/{len(dates)}] SKIP {target}")
                continue
            _handle_bad_local(target, group)
            print(f"[{index}/{len(dates)}] REQUEST {issued} {group}", flush=True)
            if _retrieve_sync(client, make_request(issued, tuple(args.area), args.grid, group),
                              target, group, issued, args.out, args.retries):
                succeeded += 1
            else:
                errors += 1
                if not args.continue_on_error:
                    print(f"Summary: downloaded={succeeded}, skipped={skipped}, errors={errors}")
                    return 1
    print(f"Summary: downloaded={succeeded}, skipped={skipped}, errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(run())
