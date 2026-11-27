"""Download ECMWF TIGGE as ONE server job per forecast issue (7 variables).

The single request contains five meteorological fields and two accumulated fluxes.
All lead times +0..+360 hours by 6 h are requested. A completed GRIB can include
fluxes at all 6h steps or only a subset; validation explicitly requires all five
met fields at every 6h step and both fluxes at +24,...,+360 h.

Modes: submit (default, nonblocking), collect, status, cancel, sync (blocking).
The job registry is stored atomically as download_jobs.json under --out.
No second ECDS request for heat fluxes is ever submitted.

ECMWF docs: https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download
             https://ecmwf.github.io/ecmwf-datastores-client/
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
ALL_VARIABLES = MET_VARIABLES + FLUX_VARIABLES
MIN_REQUIRED_FIELDS = len(MET_VARIABLES) * len(MET_HOURS) + len(FLUX_VARIABLES) * len(FLUX_HOURS)
MAX_POSSIBLE_FIELDS = len(ALL_VARIABLES) * len(MET_HOURS)
# GDAL labels; levels disambiguate surface pressure from other pressure fields.
GRIB_FIELDS = {
    ("UGRD", "10-HTGL"): "u10",
    ("VGRD", "10-HTGL"): "v10",
    ("TMP", "2-HTGL"): "t2m",
    ("DPT", "2-HTGL"): "d2m",
    ("PRES", "0-SFC"): "sp",
    ("SHTFL", "0-SFC"): "sshf",
    ("LHTFL", "0-SFC"): "slhf",
}
MET_KEYS = frozenset(("u10", "v10", "t2m", "d2m", "sp"))
FLUX_KEYS = frozenset(("sshf", "slhf"))
REGISTRY_NAME = "download_jobs.json"


def default_outdir() -> Path:
    """Return the TIGGE raw-data directory for the current operating system."""
    if os.name == "nt":
        return Path(r"D:\Nastya\Data\OceanFull\DATA\Forecast_1d\TIGGE\raw")
    return Path.cwd() / "tigge_raw"


def make_request(issued: date, area: tuple[float, float, float, float],
                 grid: str) -> dict:
    """Exactly ONE combined request per issue, covering all seven variables."""
    north, west, south, east = area
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 360):
        raise ValueError("Area must be [NORTH, WEST, SOUTH, EAST] in degrees")
    return {
        "origin": "ecmwf", "forecast_type": "control_forecast",
        "level_type": "single_level", "variable": list(ALL_VARIABLES),
        "year": [f"{issued.year:04d}"], "month": [f"{issued.month:02d}"],
        "day": [f"{issued.day:02d}"], "time": ["00:00"],
        "leadtime_hour": [str(hour) for hour in MET_HOURS],
        "area": [north, west, south, east], "grid": grid,
        "data_format": "grib",
    }


def issue_dates(start: date, end: date, stride: int = 1) -> Iterator[date]:
    """Yield issue dates inclusively in ``stride``-day increments.

    Parameters
    ----------
    start, end : datetime.date
        First and last ECMWF initialization dates (inclusive).
    stride : int
        Spacing between initialized forecasts in calendar days.

    Yields
    ------
    datetime.date
        Dates of 00 UTC forecast initializations."""
    if end < start or stride < 1:
        raise ValueError("End date must be >= start date and --every must be >= 1")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=stride)


def filename_for(issued: date, root: Path) -> Path:
    """New name, intentionally distinct from preexisting met-only GRIBs."""
    stem = f"tigge_ecmwf_control_{issued:%Y%m%d}_0000_all.grib"
    return root / f"{issued.year:04d}" / f"{issued.month:02d}" / stem


def validate_grib2(path: Path, expected_issue: date | None = None) -> tuple[int, int]:
    """Strictly validate the combined product, not just its total field count.

    Flux step GRIB_FORECAST_SECONDS may be zero for accumulated fields (GDAL).
    Therefore lead times are read from GRIB_VALID_TIME - GRIB_REF_TIME.
    """
    path = Path(path)
    count = 0
    with path.open("rb") as stream:
        stream.seek(0, 2)
        total = stream.tell()
        stream.seek(0)
        while stream.tell() < total:
            pos = stream.tell()
            header = stream.read(16)
            if len(header) != 16 or header[:4] != b"GRIB" or header[7] != 2:
                raise ValueError(f"Bad GRIB2 header at byte {pos}")
            length = int.from_bytes(header[8:16], "big")
            if length < 20 or pos + length > total:
                raise ValueError(f"Invalid/truncated GRIB2 message length {length} at {pos}")
            stream.seek(pos + length - 4)
            if stream.read(4) != b"7777":
                raise ValueError(f"Missing GRIB2 trailer at byte {pos}")
            count += 1
        if stream.tell() != total or not MIN_REQUIRED_FIELDS <= count <= MAX_POSSIBLE_FIELDS:
            raise ValueError(
                f"Found {count} GRIB messages, expected {MIN_REQUIRED_FIELDS}.."
                f"{MAX_POSSIBLE_FIELDS}; bytes={total}")

    # This validation is essential: a plausible message count does not prove
    # presence of all desired field/lead-time combinations. No pixel reads.
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError(
            "Install rasterio for GRIB field validation: "
            r"C:\Python\python.exe -m pip install rasterio") from exc

    with rasterio.Env(GRIB_NORMALIZE_UNITS="YES"):
        with rasterio.open(path) as ds:
            if ds.driver != "GRIB" or ds.count != count:
                raise ValueError("GDAL did not decode all GRIB2 messages")
            if (ds.height, ds.width) != (321, 361) or not all(
                abs(a - b) < 1e-7 for a, b in zip(ds.res, (0.25, 0.25))
            ):
                raise ValueError(f"Wrong study grid {ds.height}x{ds.width}, {ds.res}")
            required = {(field, hour) for field in MET_KEYS for hour in MET_HOURS}
            required |= {(field, hour) for field in FLUX_KEYS for hour in FLUX_HOURS}
            seen: set[tuple[str, int]] = set()
            reference_times: set[int] = set()
            for i in range(1, ds.count + 1):
                info = ds.tags(i)
                kind = GRIB_FIELDS.get((info.get("GRIB_ELEMENT"), info.get("GRIB_SHORT_NAME")))
                if kind is None:
                    raise ValueError(f"Unexpected field at GRIB band {i}: "
                                     f"{info.get('GRIB_ELEMENT')} {info.get('GRIB_SHORT_NAME')}")
                ids = info.get("GRIB_IDS", "")
                if "CENTER=98(ECMWF)" not in ids or "TYPE=3(Control_forecast)" not in ids:
                    raise ValueError(f"Non-ECMWF-control field at GRIB band {i}")
                ref = int(info["GRIB_REF_TIME"])
                valid = int(info["GRIB_VALID_TIME"])
                diff = valid - ref
                if diff < 0 or diff % 3600:
                    raise ValueError(f"Bad validity time at GRIB band {i}")
                hour = diff // 3600
                if hour not in MET_HOURS:
                    raise ValueError(f"Unexpected lead time +{hour} h at GRIB band {i}")
                reference_times.add(ref)
                key = (kind, hour)
                if key in seen:
                    raise ValueError(f"Duplicate field {key}")
                seen.add(key)
            if len(reference_times) != 1:
                raise ValueError(f"Mixed initialization times: {reference_times}")
            ref_stamp = datetime.fromtimestamp(next(iter(reference_times)), tz=timezone.utc)
            if ref_stamp.hour != 0 or ref_stamp.minute != 0:
                raise ValueError(f"Non-00 UTC issue: {ref_stamp}")
            if expected_issue is not None and ref_stamp.date() != expected_issue:
                raise ValueError(f"Wrong initialization date: {ref_stamp.date()} != {expected_issue}")
            missing = required - seen
            if missing:
                raise ValueError(f"Missing {len(missing)} essential forecast fields: "
                                 f"{sorted(missing)[:12]}")
            if count != len(seen):
                raise ValueError("Repeated GRIB messages detected")
    return count, total


def append_log(root: Path, item: dict) -> None:
    """Append one event to the persistent newline-delimited JSON audit log."""
    root.mkdir(parents=True, exist_ok=True)
    obj = dict(item, at_utc=datetime.now(timezone.utc).isoformat())
    with (root / "download_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _read_registry(root: Path) -> dict:
    """Read job metadata from ``download_jobs.json`` without modifying it.

    Returns an empty dictionary when the registry does not yet exist."""
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


def _job_key(issued: date) -> str:
    """Build the registry key for a seven-variable forecast issue."""
    return f"{issued.isoformat()}:combined"


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
    """Create an authenticated nonblocking ECMWF Data Stores client.

    Credentials come from the environment or the existing ``.cdsapirc``.
    No remote request is submitted by constructing this client."""
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
    """Create a cdsapi client for the legacy blocking download mode."""
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit("Install cdsapi: C:\\Python\\python.exe -m pip install -U cdsapi") from exc
    token = os.environ.get("ECDS_API_KEY")
    return cdsapi.Client(url=API_URL, key=token) if token else cdsapi.Client(url=API_URL)


def _is_valid_local(target: Path, issued: date) -> bool:
    """Check whether an existing combined GRIB belongs to this issue and is complete."""
    if not target.is_file():
        return False
    try:
        validate_grib2(target, issued)
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def _handle_bad_local(target: Path, issued: date) -> None:
    """Move an invalid local GRIB aside before writing a new result."""
    if target.is_file() and not _is_valid_local(target, issued):
        bad = target.with_name(target.name + ".invalid")
        if bad.exists():
            bad = target.with_name(target.name + f".invalid.{int(time.time())}")
        target.rename(bad)
        print(f"Renamed invalid GRIB to {bad}")


def _retrieve_sync(client, request, target, issued, root, retries: int):
    """Submit and download a combined request via the blocking cdsapi interface.

    Retry only after a definite error; interrupted requests may continue remotely."""
    target.parent.mkdir(parents=True, exist_ok=True)
    working = target.with_name(target.name + ".downloading")
    for attempt in range(1, retries + 2):
        try:
            working.unlink(missing_ok=True)
            client.retrieve(DATASET, request, str(working))
            count, size = validate_grib2(working, issued)
            working.replace(target)
            append_log(root, {"date": issued.isoformat(), "group": "combined", "status": "ok",
                              "file": str(target), "messages": count, "bytes": size})
            return True
        except Exception as exc:
            working.unlink(missing_ok=True)
            print(f"ERROR {issued} combined attempt {attempt}: {exc}", file=sys.stderr)
            append_log(root, {"date": issued.isoformat(), "group": "combined",
                              "status": "error", "error": str(exc)})
            if attempt <= retries:
                time.sleep(min(30 * 2 ** (attempt - 1), 120))
    return False


def _have_legacy_pair(root: Path, issued: date) -> bool:
    """Avoid re-downloading dates already held as old TWO local outputs."""
    folder = root / f"{issued.year:04d}" / f"{issued.month:02d}"
    stem = f"tigge_ecmwf_control_{issued:%Y%m%d}_0000"
    met = folder / (stem + ".grib")
    flux = folder / (stem + "_fluxes.grib")
    if not (met.is_file() and flux.is_file()):
        return False
    try:
        # Legacy message counts only, original converters verify actual variables.
        return _validate_legacy_count(met, 305) and _validate_legacy_count(flux, 30)
    except (OSError, ValueError):
        return False


def _validate_legacy_count(path: Path, count_expected: int) -> bool:
    """Verify the exact number of intact GRIB2 messages in an older file."""
    count = 0
    with path.open("rb") as handle:
        handle.seek(0, 2)
        total = handle.tell()
        handle.seek(0)
        while handle.tell() < total:
            here = handle.tell()
            hdr = handle.read(16)
            if len(hdr) < 16 or hdr[:4] != b"GRIB" or hdr[7] != 2:
                raise ValueError("Invalid GRIB2 header")
            size = int.from_bytes(hdr[8:16], "big")
            if size < 20 or here + size > total:
                raise ValueError("Invalid GRIB2 message size")
            handle.seek(here + size - 4)
            if handle.read(4) != b"7777":
                raise ValueError("Invalid GRIB2 trailer")
            count += 1
    return count == count_expected


# ECDS only lists accepted/running/successful/failed/rejected jobs in its documented
# job filters. A cancelled request may instead be dismissed or removed (HTTP 404).
TERMINAL_UNUSABLE_STATUSES = frozenset({
    "failed", "rejected", "cancelled", "canceled", "dismissed",
    "aborted", "deleted", "expired", "not_found",
})


def _is_http_404(exc: Exception) -> bool:
    """Recognize a missing remote request without mistaking network failure for deletion.

    Parameters
    ----------
    exc : Exception
        Exception raised by ``Client.get_remote``; e.g. an HTTP error with
        ``status_code`` or ``response.status_code``.

    Returns
    -------
    bool
        True only when the exception explicitly carries HTTP status 404.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) == 404
    except (TypeError, ValueError):
        return False


def _read_remote_status(client, request_id: str) -> str:
    """Read the current ECDS request status, treating only HTTP 404 as deletion.

    Other errors are intentionally propagated: connection failure, timeouts or
    authentication problems must never trigger duplicate submissions.
    """
    try:
        remote = client.get_remote(request_id)
    except Exception as exc:
        if _is_http_404(exc):
            return "deleted"
        raise
    return str(remote.status).strip().lower().rsplit(".", 1)[-1]


def _refresh_job_status(client, root: Path, registry: dict,
                        key: str) -> str:
    """Refresh a saved job using its remote ID and persist the new status.

    The saved request ID is kept even when an ECMWF web cancellation removed
    the remote job, so the historical audit trail remains intact.
    """
    item = registry[key]
    previous = str(item.get("status", "unknown")).lower()
    status = _read_remote_status(client, str(item["request_id"]))
    if status != previous:
        item["status"] = status
        item["last_checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_registry(root, registry)
        append_log(root, {"status": "remote_status", "job_key": key,
                          "request_id": item["request_id"],
                          "previous": previous, "remote_status": status})
    return status


def _legacy_jobs_for(client, root: Path, registry: dict, issued: date) -> list[str]:
    """Return only usable, still-existing legacy jobs for an issue date.

    Both legacy groups are checked on the ECDS server; successful jobs remain
    blocking because their results may still be downloadable. Cancelled, failed
    or deleted jobs stop blocking submission of the combined seven-field job.
    """
    blocking = []
    for group in ("met", "fluxes"):
        key = f"{issued.isoformat()}:{group}"
        item = registry.get(key)
        if not item or not item.get("request_id"):
            continue
        if str(item.get("status", "")).lower() in TERMINAL_UNUSABLE_STATUSES:
            continue
        status = _refresh_job_status(client, root, registry, key)
        if status in TERMINAL_UNUSABLE_STATUSES:
            print(f"RELEASE {issued} old {group}: {item['request_id']} -> {status}")
            continue
        blocking.append(f"{group}: {item['request_id']} ({status})")
    return blocking


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


def _submit_async(client, root: Path, dates, area, grid,
                  max_submits: int, max_active: int) -> int:
    """Send up to ``max_submits`` new seven-variable forecast jobs.

    Only jobs with an actual usable server status block a date. Historical
    legacy IDs are refreshed using the ECDS API before deciding to skip. HTTP
    404 denotes a removed job; all other communication errors stop submission
    rather than risking duplication. Jobs and status changes are persisted."""
    registry = _read_registry(root)
    submitted = skipped = 0
    try:
        active = _count_active_server_jobs(client)
    except Exception as exc:
        print(f"Cannot check ECDS active jobs: {exc}", file=sys.stderr)
        print("No new requests submitted. Verify Your Requests on the website.", file=sys.stderr)
        return 1
    print(f"Active jobs across this ECDS account: {active}; configured limit: {max_active}")
    if active >= max_active:
        print("No free slot. Existing jobs continue on ECMWF servers.")
        return 0

    for issued in dates:
        key = _job_key(issued)
        target = filename_for(issued, root)
        if _is_valid_local(target, issued) or _have_legacy_pair(root, issued):
            skipped += 1
            print(f"SKIP already downloaded {issued}")
            continue
        _handle_bad_local(target, issued)
        if submitted >= max_submits or active >= max_active:
            break
        try:
            # An ID saved locally is historical evidence, NOT evidence that its
            # job is still running. The user may have cancelled it in My Requests.
            prior = registry.get(key)
            if prior and prior.get("request_id"):
                current = _refresh_job_status(client, root, registry, key)
                if current not in TERMINAL_UNUSABLE_STATUSES:
                    skipped += 1
                    print(f"SKIP previously submitted {key}: {prior['request_id']} ({current}); "
                          "use --mode collect when successful")
                    continue
                print(f"RETRY {key}: previous combined job {prior['request_id']} -> {current}")
            legacy = _legacy_jobs_for(client, root, registry, issued)
        except Exception as exc:
            print(f"Cannot verify remote jobs for {issued}: {exc}", file=sys.stderr)
            print("Nothing new was submitted for this date. Check ECDS before retrying; "
                  "a network problem is NOT proof of cancellation.", file=sys.stderr)
            return 1
        if legacy:
            print(f"SKIP {issued}: older met/flux jobs still available: {', '.join(legacy)}. "
                  "Collect or cancel those jobs first; no duplicate request is made.")
            skipped += 1
            continue
        request = make_request(issued, area, grid)
        try:
            remote = client.submit(DATASET, request)
            job_id = str(remote.request_id)
            prior_item = registry.get(key, {})
            history = list(prior_item.get("history", []))
            if prior_item.get("request_id"):
                history.append({"request_id": prior_item["request_id"],
                                "status": prior_item.get("status"),
                                "submitted_at_utc": prior_item.get("submitted_at_utc")})
            registry[key] = {
                "request_id": job_id, "date": issued.isoformat(), "group": "combined",
                "file": str(target), "request": request, "status": str(remote.status),
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                "history": history,
            }
            _write_registry(root, registry)
            append_log(root, {"date": issued.isoformat(), "group": "combined",
                              "status": "submitted", "request_id": job_id})
            active += 1
            submitted += 1
            print(f"SUBMITTED all 7 fields, {issued}: {job_id}", flush=True)
        except Exception as exc:
            print(f"ERROR submitting {issued}: {exc}", file=sys.stderr)
            print("Do NOT retry blindly; check My Requests: server may have accepted it.",
                  file=sys.stderr)
            append_log(root, {"date": issued.isoformat(), "group": "combined",
                              "status": "submit_error", "error": str(exc)})
            return 1
    print(f"Summary: combined jobs submitted={submitted}; dates skipped={skipped}")
    return 0


def _collect_async(client, root: Path, dates) -> int:
    """Download ready seven-variable forecasts for selected issue dates.

    Uses saved remote IDs, verifies the combined GRIB, and atomically moves a
    validated download into the expected output path."""
    registry = _read_registry(root)
    keys = {_job_key(d) for d in dates}
    items = [(k, v) for k, v in registry.items()
             if k in keys and v.get("request_id")]
    if not items:
        print("No combined requests in registry for selected dates. "
              "If submitted through old script, use My Requests to download results.")
        return 0
    downloaded = pending = failed = 0
    for key, item in items:
        issued = date.fromisoformat(item["date"])
        target = filename_for(issued, root)
        if _is_valid_local(target, issued):
            item["status"] = "downloaded"
            _write_registry(root, registry)
            print(f"SKIP local {key}")
            continue
        if str(item.get("status", "")).lower() in TERMINAL_UNUSABLE_STATUSES:
            print(f"SKIP terminal request {key}: {item['status']}; next submit can retry")
            continue
        _handle_bad_local(target, issued)
        try:
            status = _refresh_job_status(client, root, registry, key)
            if status in TERMINAL_UNUSABLE_STATUSES:
                print(f"SKIP terminal request {key}: {status}; next submit can retry")
                continue
            remote = client.get_remote(item["request_id"])
            if not remote.results_ready:
                print(f"PENDING {key}: {status}")
                pending += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            working = target.with_name(target.name + ".downloading")
            working.unlink(missing_ok=True)
            remote.download(str(working))
            count, size = validate_grib2(working, issued)
            working.replace(target)
            item["status"] = "downloaded"
            _write_registry(root, registry)
            append_log(root, {"date": issued.isoformat(), "group": "combined",
                              "status": "ok", "request_id": item["request_id"],
                              "file": str(target), "messages": count, "bytes": size})
            downloaded += 1
            print(f"DOWNLOADED combined {issued}: {count} messages, {size:,} bytes", flush=True)
        except Exception as exc:
            print(f"ERROR collecting {key}: {exc}", file=sys.stderr)
            append_log(root, {"date": issued.isoformat(), "group": "combined",
                              "status": "collect_error", "error": str(exc)})
            failed += 1
    print(f"Summary: downloaded={downloaded}, pending={pending}, errors={failed}")
    return 1 if failed else 0


def _status_async(client) -> int:
    """Print current ECDS job IDs and statuses without modifying them."""
    jobs = client.get_jobs(limit=100, sortby="-created",
                           status=["accepted", "running", "successful", "failed", "rejected"])
    print("ECDS jobs, newest first (all datasets on this account):")
    count = 0
    while jobs is not None:
        for job_id in jobs.request_ids:
            remote = client.get_remote(str(job_id))
            print(f"  {job_id}  {remote.status}")
            count += 1
            if count >= 20:
                return 0
        jobs = jobs.next
    return 0


def _cancel_async(client, root: Path, request_id: str) -> int:
    """Cancel one still-active remote ECDS request with an explicit ID.

    A successful cancellation request is also recorded in the local registry."""
    remote = client.get_remote(request_id)
    status = str(remote.status)
    if status not in ("accepted", "running"):
        print(f"Not cancelling {request_id}: status={status}; "
              "only accepted/running jobs are eligible in this command.")
        return 1
    # Official Data Stores Client.delete(*request_ids) endpoint.
    client.delete(request_id)
    registry = _read_registry(root)
    for item in registry.values():
        if item.get("request_id") == request_id:
            item["status"] = "cancelled"
            item["cancelled_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_registry(root, registry)
    append_log(root, {"status": "cancelled", "request_id": request_id})
    print(f"Cancellation requested for {request_id}; check My Requests for final status.")
    return 0


def run(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch submit, collect, status, cancel or sync.

    Return an operating-system exit code (zero on success)."""
    parser = argparse.ArgumentParser(description="One ECMWF TIGGE request per day with all 7 fields")
    parser.add_argument("--start", default="2025-10-31")
    parser.add_argument("--end", default="2026-10-16")
    parser.add_argument("--mode", choices=("submit", "collect", "status", "cancel", "sync"),
                        default="submit", help="Default: nonblocking submit")
    parser.add_argument("--max-submits", type=int, default=1)
    parser.add_argument("--max-active", type=int, default=1,
                        help="Maximum accepted/running requests on account, up to 3")
    parser.add_argument("--request-id", help="Specific remote request ID for --mode cancel")
    parser.add_argument("--confirm-cancel", action="store_true",
                        help="Explicitly allow remote cancellation for --mode cancel")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-dates", type=int, default=0)
    parser.add_argument("--out", type=Path, default=default_outdir())
    parser.add_argument("--area", type=float, nargs=4,
                        metavar=("NORTH", "WEST", "SOUTH", "EAST"), default=[80., -90., 0., 0.])
    parser.add_argument("--grid", default="0.25/0.25")
    parser.add_argument("--execute", action="store_true", help="Otherwise dry-run")
    parser.add_argument("--retries", type=int, default=0, help="Only for --mode sync")
    parser.add_argument("--continue-on-error", action="store_true", help="Only for --mode sync")
    args = parser.parse_args(argv)
    if args.max_submits < 1 or args.max_dates < 0 or args.retries < 0:
        parser.error("--max-submits must be >=1; --max-dates and --retries must be >=0")
    if not 1 <= args.max_active <= 3:
        parser.error("--max-active must be in 1..3; sequential submission recommended")
    if args.mode == "cancel":
        if not args.request_id:
            parser.error("--mode cancel requires --request-id")
        if not args.confirm_cancel:
            parser.error("--mode cancel requires --confirm-cancel")
        if not args.execute:
            print(f"DRY RUN: would cancel remote ECDS request {args.request_id}")
            return 0
        return _cancel_async(make_async_client(), args.out, args.request_id)
    if args.mode == "status":
        if not args.execute:
            print("DRY RUN: add --execute to query ECDS jobs")
            return 0
        return _status_async(make_async_client())

    try:
        dates = list(issue_dates(date.fromisoformat(args.start), date.fromisoformat(args.end), args.every))
        if args.max_dates:
            dates = dates[:args.max_dates]
        _ = make_request(dates[0], tuple(args.area), args.grid)
    except (ValueError, IndexError) as exc:
        parser.error(str(exc))
    print(f"Dataset={DATASET}; mode={args.mode}; issue 00 UTC; "
          f"{len(dates)} dates ({dates[0]} .. {dates[-1]})")
    print(f"ONE request per date: five met + two heat fluxes; lead times +0..+360 h by 6 h")
    print(f"Area={args.area}, grid={args.grid}, output={args.out}")
    if not args.execute:
        print("Example combined request:")
        print(json.dumps(make_request(dates[0], tuple(args.area), args.grid), ensure_ascii=False,
                         indent=2))
        print("DRY RUN: add --execute to submit/collect")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode in ("submit", "collect"):
        client = make_async_client()
        if args.mode == "submit":
            return _submit_async(client, args.out, dates, tuple(args.area), args.grid,
                                 args.max_submits, args.max_active)
        return _collect_async(client, args.out, dates)

    client = make_sync_client()
    succeeded = skipped = errors = 0
    for i, issued in enumerate(dates, 1):
        target = filename_for(issued, args.out)
        if _is_valid_local(target, issued) or _have_legacy_pair(args.out, issued):
            print(f"[{i}/{len(dates)}] SKIP existing {issued}")
            skipped += 1
            continue
        _handle_bad_local(target, issued)
        print(f"[{i}/{len(dates)}] REQUEST all seven fields {issued}")
        if _retrieve_sync(client, make_request(issued, tuple(args.area), args.grid),
                          target, issued, args.out, args.retries):
            succeeded += 1
        else:
            errors += 1
            if not args.continue_on_error:
                return 1
    print(f"Summary: downloaded={succeeded}, skipped={skipped}, errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(run())
