"""Download selected GEFSv12 reforecast flux messages via NOAA HTTPS byte ranges.

The GEFSv12 retrospective archive stores *one variable per GRIB2 file* and one
forecast lead per GRIB2 message. A companion ``.grib2.idx`` index gives starting
byte offsets for each message. This module retrieves only the 3-hour or 6-hour
means needed for requested UTC days, without AWS CLI, curl, or external Python
packages.

Pilot (Windows PowerShell, from the repository root)::

    C:\\Python\\python.exe -m Forecast_1d.download_gefs_reforecast --date 2017-03-14 --days 1
    C:\\Python\\python.exe -m Forecast_1d.download_gefs_reforecast --date 2017-03-14 --days 1 --execute

Default mode is inspection-only. GRIB2 messages are kept unaltered, including the
native NOAA heat-flux sign convention, in concatenated, valid GRIB2 files.
Nothing here is ERA5-aligned yet; daily means and model-target harmonization
must be performed in a separate, independently validated conversion step.

Official NOAA data specification:
https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE_URL = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
VARIABLES = ("shtfl_sfc", "lhtfl_sfc")
VAR_SHORT_NAMES = {"shtfl_sfc": "SHTFL", "lhtfl_sfc": "LHTFL"}
MAX_FORECAST_HOUR = 384
TIMEOUT_SECONDS = 45


def default_outdir() -> Path:
    """Return the existing local data-root convention, or a portable fallback."""
    if os.name == "nt":
        return Path(r"D:\Nastya\Data\OceanFull\DATA\Forecast_1d\GEFS\raw")
    return Path.cwd() / "GEFS_raw"


def parse_days(expression: str) -> list[int]:
    """Parse comma-separated one-based UTC forecast days (e.g. ``1,3,8-14``)."""
    values: set[int] = set()
    for part in expression.split(","):
        part = part.strip()
        if not part:
            raise ValueError("Empty day in --days")
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if not match:
            raise ValueError(f"Invalid day range: {part!r}")
        first = int(match.group(1))
        last = int(match.group(2) or match.group(1))
        if not (1 <= first <= last <= 14):
            raise ValueError("Requested forecast days must be within 1..14")
        values.update(range(first, last + 1))
    return sorted(values)


def forecast_hours(day: int) -> tuple[int, ...]:
    """Return endpoints of all interval means covering forecast UTC day 1..14.

    0.25-degree GEFSv12 fields are 3-hourly through +240 h; the 0.5-degree
    continuation is 6-hourly after +240 h. ``day`` is one-based, with day 1
    representing UTC [issue 00Z, issue+24h].
    """
    if not 1 <= day <= 14:
        raise ValueError("day must be 1..14")
    end = day * 24
    stride = 3 if day <= 10 else 6
    return tuple(range(end - 24 + stride, end + 1, stride))


def grib_urls(issue: date, variable: str, member: str, segment: str) -> tuple[str, str]:
    """Construct the NOAA public HTTPS GRIB2 and sidecar IDX URLs."""
    if variable not in VARIABLES:
        raise ValueError(f"Unsupported flux variable: {variable}")
    if not re.fullmatch(r"(?:c00|p(?:0[1-9]|10))", member):
        raise ValueError("member must be c00, p01..p10 (weekly members may be missing)")
    if segment not in ("Days:1-10", "Days:10-16"):
        raise ValueError(f"Unsupported segment {segment}")
    issue_tag = f"{issue:%Y%m%d}00"
    path = f"{issue.year:04d}/{issue_tag}/{member}/{quote(segment, safe='')}/{variable}_{issue_tag}_{member}.grib2"
    uri = f"{BASE_URL}/{path}"
    return uri, uri + ".idx"


def request_text(url: str, timeout: int = TIMEOUT_SECONDS) -> str:
    """Read a UTF-8/ASCII NOAA index, without any third-party dependency."""
    req = Request(url, headers={"User-Agent": "OceanComponents-GEFSv12-pilot/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Expected HTTP 200 for IDX, got {resp.status}: {url}")
        raw = resp.read()
    return raw.decode("utf-8-sig")


def remote_size(url: str, timeout: int = TIMEOUT_SECONDS) -> int:
    """Read total GRIB2 length using HEAD; needed for the last IDX record."""
    req = Request(url, method="HEAD", headers={"User-Agent": "OceanComponents-GEFSv12-pilot/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Expected HTTP 200 for HEAD, got {resp.status}: {url}")
        length = resp.headers.get("Content-Length")
    if length is None or int(length) <= 0:
        raise ValueError(f"Missing/invalid HTTP Content-Length: {url}")
    return int(length)


def parse_idx(text: str, file_size: int) -> list[dict]:
    """Return NOAA index rows enriched with exact inclusive HTTP byte ranges.

    Index rows have fields ``message_number:byte_offset:d=YYYYMMDDHH:...``.
    The next message start is the prior message end + 1; the last message is
    bounded by the object's HTTP Content-Length. The forecast *endpoint* is
    taken from the ``... hour ...`` range text (e.g. ``0-3 hour ave fcst``).
    """
    rows = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = raw.strip().split(":")
        if len(parts) < 6 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"Unexpected NOAA IDX record: {raw!r}")
        hour_matches = re.findall(r"(?:(\d+)\s*-\s*)?(\d+)\s+hour", raw, flags=re.IGNORECASE)
        if not hour_matches:
            raise ValueError(f"No forecast hour found in IDX row: {raw!r}")
        first, last = hour_matches[-1]
        lead_h = int(last)
        avg_start = int(first) if first else None
        rows.append({"number": int(parts[0]), "start": int(parts[1]),
                     "lead_h": lead_h, "avg_start": avg_start,
                     "parameter": parts[3].strip(), "raw": raw.strip()})
    if not rows or rows[0]["start"] != 0:
        raise ValueError("IDX must begin at GRIB byte offset zero")
    for i, item in enumerate(rows):
        end_exclusive = rows[i + 1]["start"] if i + 1 < len(rows) else file_size
        if item["start"] >= end_exclusive or end_exclusive > file_size:
            raise ValueError(f"Bad IDX offsets around message {item['number']}")
        if i and item["number"] != rows[i - 1]["number"] + 1:
            raise ValueError("Non-sequential IDX message numbers")
        item["end"] = end_exclusive - 1
        item["size"] = end_exclusive - item["start"]
    return rows


def selection_for_days(rows: list[dict], days: Iterable[int], variable: str, segment: str) -> list[dict]:
    """Select all interval-mean GRIB messages for the requested UTC days.

    Fails closed if a lead hour is absent, ambiguous, or belongs to the wrong
    variable. For fluxes, accepts only records explicitly marked as averages.
    """
    target_short_name = VAR_SHORT_NAMES[variable]
    wanted: set[int] = set()
    for day in days:
        if (day <= 10) != (segment == "Days:1-10"):
            continue
        wanted.update(forecast_hours(day))
    selected = []
    for hour in sorted(wanted):
        candidates = [r for r in rows if r["lead_h"] == hour and r["parameter"].upper() == target_short_name]
        if len(candidates) != 1:
            raise ValueError(f"{variable}, {segment}, +{hour}h: expected 1 record, found {len(candidates)}")
        item = candidates[0]
        if not re.search(r"\b(?:ave|avg|average)\b", item["raw"], re.IGNORECASE):
            raise ValueError(f"Expected time-averaged flux in {item['raw']!r}")
        selected.append(item)
    return selected


def plan_download(issue: date, member: str, days: list[int], variables: tuple[str, ...]) -> list[dict]:
    """Inspect indices for selected days and estimate exact network byte volume."""
    specs = []
    for variable in variables:
        for segment in ("Days:1-10", "Days:10-16"):
            subdays = [d for d in days if (d <= 10) == (segment == "Days:1-10")]
            if not subdays:
                continue
            url, idx_url = grib_urls(issue, variable, member, segment)
            index = request_text(idx_url)
            size = remote_size(url)
            rows = parse_idx(index, size)
            selected = selection_for_days(rows, subdays, variable, segment)
            if not selected:
                raise ValueError(f"No messages found for {variable}/{segment}")
            specs.append({"variable": variable, "segment": segment, "url": url,
                          "idx_url": idx_url, "source_bytes": size,
                          "rows": selected, "download_bytes": sum(r["size"] for r in selected)})
    return specs


def validate_concatenated_grib2(path: Path, expected_messages: int) -> tuple[int, int]:
    """Check message boundaries, edition and ``7777`` trailer without ecCodes."""
    count = 0
    size = path.stat().st_size
    with path.open("rb") as f:
        while f.tell() < size:
            offset = f.tell()
            hdr = f.read(16)
            if len(hdr) != 16 or hdr[:4] != b"GRIB" or hdr[7] != 2:
                raise ValueError(f"Not GRIB2 at byte {offset}: {path}")
            n = int.from_bytes(hdr[8:16], "big")
            if n < 20 or offset + n > size:
                raise ValueError(f"Truncated GRIB2 message at byte {offset}: {path}")
            f.seek(offset + n - 4)
            if f.read(4) != b"7777":
                raise ValueError(f"Missing GRIB end marker at byte {offset}: {path}")
            count += 1
    if count != expected_messages:
        raise ValueError(f"Expected {expected_messages} messages, got {count}: {path}")
    return count, size


def append_range(output, url: str, start: int, end: int, timeout: int = TIMEOUT_SECONDS) -> int:
    """Stream exactly one selected GRIB message using HTTP 206 Partial Content.

    A server returning HTTP 200 instead of 206 is rejected so that a partial
    request can never accidentally download a huge whole-GRIB file.
    """
    req = Request(url, headers={"Range": f"bytes={start}-{end}",
                                "User-Agent": "OceanComponents-GEFSv12-pilot/1.0"})
    with urlopen(req, timeout=timeout) as response:
        if response.status != 206:
            raise RuntimeError(f"Server ignored Range ({start}-{end}), HTTP {response.status}")
        expected_header = f"bytes {start}-{end}/"
        actual_header = response.headers.get("Content-Range", "")
        if not actual_header.startswith(expected_header):
            raise RuntimeError(f"Wrong Content-Range: {actual_header!r}, expected {expected_header!r}")
        n = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            n += len(block)
    expected = end - start + 1
    if n != expected:
        raise IOError(f"Short range transfer: {n} bytes, expected {expected}")
    return n


def run_download(specs: list[dict], output_root: Path, issue: date, member: str,
                 *, retries: int = 2) -> list[Path]:
    """Save each variable/segment as an atomic concatenated GRIB2 and JSON inventory."""
    issue_tag = f"{issue:%Y%m%d}00"
    folder = output_root / str(issue.year) / issue_tag / member
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in specs:
        segment_slug = "days01-10" if spec["segment"] == "Days:1-10" else "days11-16"
        # Selected forecast hours uniquely identify a file; avoids treating
        # a day-1 pilot as a completed 14-day data set.
        hours = [r["lead_h"] for r in spec["rows"]]
        tag = f"h{hours[0]:03d}-{hours[-1]:03d}_n{len(hours):02d}"
        stem = f"{spec['variable']}_{issue_tag}_{member}_{segment_slug}_{tag}"
        target = folder / f"{stem}.grib2"
        metadata = folder / f"{stem}.json"
        if target.is_file() and metadata.is_file():
            try:
                validate_concatenated_grib2(target, len(spec["rows"]))
                previous = json.loads(metadata.read_text(encoding="utf-8"))
                if (previous["source_url"] == spec["url"] and
                        previous["hours"] == hours and
                        target.stat().st_size == spec["download_bytes"]):
                    print(f"SKIP valid {target}")
                    paths.append(target)
                    continue
            except (ValueError, OSError, KeyError, json.JSONDecodeError):
                pass
        temp = target.with_name(target.name + ".part")
        success = False
        for attempt in range(retries + 1):
            try:
                temp.unlink(missing_ok=True)
                with temp.open("wb") as stream:
                    for row in spec["rows"]:
                        append_range(stream, spec["url"], row["start"], row["end"])
                validate_concatenated_grib2(temp, len(spec["rows"]))
                if temp.stat().st_size != spec["download_bytes"]:
                    raise IOError("Downloaded size differs from index estimate")
                temp.replace(target)
                meta = {"source_url": spec["url"], "index_url": spec["idx_url"],
                        "issue_utc": issue_tag, "member": member, "segment": spec["segment"],
                        "variable": spec["variable"], "hours": hours, "size_bytes": target.stat().st_size,
                        "rows": [{"hour": r["lead_h"], "byte_start": r["start"],
                                  "byte_end": r["end"], "index_row": r["raw"]}
                                 for r in spec["rows"]]}
                temporary_meta = metadata.with_suffix(".json.part")
                temporary_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary_meta.replace(metadata)
                print(f"SAVED {target} ({len(hours)} fields, {target.stat().st_size:,} bytes)")
                paths.append(target)
                success = True
                break
            except (OSError, IOError, HTTPError, URLError, ValueError, RuntimeError) as e:
                temp.unlink(missing_ok=True)
                print(f"ERROR {spec['variable']} {spec['segment']}, attempt {attempt+1}: {e}", file=sys.stderr)
                if attempt < retries:
                    time.sleep(min(2 ** attempt * 2, 8))
        if not success:
            raise RuntimeError(f"Unable to download {spec['variable']} {spec['segment']} after {retries+1} tries")
    return paths


def main(argv: list[str] | None = None) -> int:
    """Provide a safe, inspection-first Windows-compatible GEFSv12 pilot CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2017-03-14", help="GEFSv12 issue date YYYY-MM-DD, 2000..2019")
    parser.add_argument("--member", default="c00", help="Ensemble member, default control c00")
    parser.add_argument("--days", default="1", help="UTC forecast days, e.g. 1 or 1-14 (default 1)")
    parser.add_argument("--variables", nargs="+", choices=VARIABLES, default=list(VARIABLES))
    parser.add_argument("--out", type=Path, default=default_outdir())
    parser.add_argument("--execute", action="store_true", help="Actually download selected GRIB messages")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        issued = date.fromisoformat(args.date)
        if not (date(2000, 1, 1) <= issued <= date(2019, 12, 31)):
            raise ValueError("GEFSv12 reforecast period is 2000-01-01 through 2019-12-31")
        days = parse_days(args.days)
        variables = tuple(dict.fromkeys(args.variables))
        if args.retries < 0:
            raise ValueError("--retries must be >= 0")
        grib_urls(issued, variables[0], args.member, "Days:1-10")
    except ValueError as e:
        parser.error(str(e))
    print(f"NOAA GEFSv12 reforecast, issue={issued} 00Z, member={args.member}, days={days}")
    print(f"Variables={variables}; output={args.out}")
    specs = plan_download(issued, args.member, days, variables)
    total = sum(s["download_bytes"] for s in specs)
    for s in specs:
        print(f"  {s['variable']:10s} {s['segment']:11s}: "
              f"{len(s['rows']):2d} fields, {s['download_bytes'] / 1024**2:.2f} MiB "
              f"(full file {s['source_bytes'] / 1024**2:.2f} MiB)")
    print(f"TOTAL selected: {total:,} bytes ({total / 1024**2:.2f} MiB)")
    if not args.execute:
        print("INSPECTION ONLY. Add --execute to download via HTTPS Range requests.")
        return 0
    run_download(specs, args.out, issued, args.member, retries=args.retries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
