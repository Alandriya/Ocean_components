"""Sequential, bounded-memory ERA5 GRIB day cache for hourly quantile diagnostics.

Read concatenated GRIB1/GRIB2 messages from their raw byte stream (read only).
Decode each message in memory with ecCodes; NEVER seek into the source GRIB.
Only requested UTC days are cached as small per-day NumPy files. A validity
hour is taken from the GRIB validityDate/validityTime metadata. No time shift.

Caches are written beneath Functional/Diagnostics and reused only after a
completed scan whose source file sizes/mtimes and requested days match.
Partial scans are discarded on retry. Each cached day has all 24 hourly fields
for one variable, downsampled to the exact ocean-mask grid.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path

import numpy as np

CACHE_VERSION = 1
GRIB_SUFFIXES = frozenset((".grib", ".grb", ".grib2"))


def _raw_messages(path):
    """Yield whole GRIB1/2 records without calling seek() on the source file."""
    with Path(path).open("rb", buffering=1024 * 1024) as stream:
        number = 0
        while True:
            header = stream.read(16)
            if not header:
                return
            if len(header) != 16 or header[:4] != b"GRIB":
                raise ValueError(
                    f"{path}: invalid GRIB record after message {number}; "
                    "expected consecutive GRIB1/GRIB2 messages. "
                    "Check that the downloaded file is complete and not a ZIP archive."
                )
            edition = header[7]
            if edition == 1:
                total = int.from_bytes(header[4:7], "big")
            elif edition == 2:
                total = int.from_bytes(header[8:16], "big")
            else:
                raise ValueError(f"{path}: unsupported GRIB edition {edition}")
            if not 16 <= total <= 128 * 1024 * 1024:
                raise ValueError(
                    f"{path}: suspicious GRIB message length {total} after record {number}. "
                    "Extended-length GRIB1 or nonstandard containers require another decoder."
                )
            remainder = stream.read(total - 16)
            if len(remainder) != total - 16:
                raise EOFError(
                    f"{path}: truncated GRIB message {number + 1}: "
                    f"expected {total} bytes, found {16 + len(remainder)}"
                )
            message = header + remainder
            if message[-4:] != b"7777":
                raise ValueError(f"{path}: GRIB message {number + 1} has no 7777 terminator")
            number += 1
            yield number, message


def _get(codes, handle, key, default=None):
    try:
        return codes.codes_get(handle, key)
    except Exception:
        return default


def _valid_day_hour(codes, handle):
    d = _get(codes, handle, "validityDate")
    t = _get(codes, handle, "validityTime")
    if d is None or t is None:
        # A forecast lead time must never be mistaken for a clock hour.
        step = _get(codes, handle, "step")
        if step not in (None, 0, "0"):
            raise ValueError("GRIB forecast has no validityDate/validityTime")
        d = _get(codes, handle, "dataDate")
        t = _get(codes, handle, "dataTime")
    if d is None or t is None:
        raise ValueError("GRIB message lacks a valid timestamp")
    tt = int(t)
    if tt < 0 or tt > 2359 or tt % 100 != 0:
        raise ValueError(f"Non-hourly GRIB validityTime: {t}")
    stamp = datetime.strptime(f"{int(d):08d}{tt:04d}", "%Y%m%d%H%M")
    return stamp.date(), stamp.hour


class _DayWriters:
    """LRU of writable day arrays; each file is a few MB, never a huge map."""

    def __init__(self, directory, grid_shape, max_open=6):
        self.directory = Path(directory)
        self.grid_shape = tuple(grid_shape)
        self.open = OrderedDict()
        self.max_open = max_open

    def filename(self, variable, day):
        return self.directory / variable / f"{day.isoformat()}.npy"

    def write(self, variable, day, hour, pixels):
        key = (variable, day)
        arr = self.open.pop(key, None)
        if arr is None:
            path = self.filename(variable, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                arr = np.load(path, mmap_mode="r+", allow_pickle=False)
            else:
                arr = np.lib.format.open_memmap(
                    path, mode="w+", dtype="float32", shape=(24, *self.grid_shape)
                )
                arr[:] = np.nan
        arr[hour] = pixels
        self.open[key] = arr
        while len(self.open) > self.max_open:
            _, old = self.open.popitem(last=False)
            old.flush()
            del old

    def close(self):
        while self.open:
            _, arr = self.open.popitem()
            arr.flush()
            del arr


def prepare_grib_day_cache(*, paths, days, cache_dir, grid_shape,
                           spatial_stride, aliases, progress_every=15000):
    """Create/reuse day-wise caches for requested dates, returning metadata.

    `days` includes both state and endpoint dates of every selected transition.
    Returns (paths_by_variable_and_day, source_shape, spatial_stride).
    Files with insufficient hours are not registered; they are reported as
    incomplete by the calling diagnostic. Any duplicate UTC hour is an error.
    """
    paths = tuple(Path(p).resolve() for p in paths)
    required = tuple(sorted({d if isinstance(d, date) else date.fromisoformat(str(d))
                             for d in days}))
    if not required:
        raise ValueError("No dates were requested for GRIB conversion")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "complete_manifest.json"
    fingerprints = [{"path": str(p), "size": p.stat().st_size,
                     "mtime_ns": p.stat().st_mtime_ns} for p in paths]
    signature = {
        "cache_version": CACHE_VERSION,
        "sources": fingerprints,
        "requested_days": [d.isoformat() for d in required],
        "grid_shape": list(grid_shape),
        "requested_spatial_stride": str(spatial_stride),
        "variables": {k: list(v) for k, v in aliases.items()},
    }
    if manifest_path.exists():
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            if metadata.get("signature") == signature:
                complete = {
                    var: {date.fromisoformat(day): cache / var / f"{day}.npy"
                          for day in metadata["complete_days"][var]}
                    for var in aliases
                }
                if all(p.is_file() for by_day in complete.values() for p in by_day.values()):
                    print(f"[grib_stream] Reusing completed cache: {cache}", flush=True)
                    return (complete, tuple(metadata["source_shape"]),
                            int(metadata["spatial_stride"]))
        except (ValueError, KeyError, OSError, TypeError):
            pass
    # A manifest is only saved after a successful scan. Rebuild incomplete data.
    manifest_path.unlink(missing_ok=True)
    for var in aliases:
        directory = cache / var
        if directory.exists():
            for stale in directory.glob("*.npy"):
                stale.unlink()
    writers = _DayWriters(cache, grid_shape)
    wanted = set(required)
    low_aliases = {alias.lower(): var for var, names in aliases.items() for alias in names}
    presence = {var: {day: np.zeros(24, dtype=bool) for day in required}
                for var in aliases}
    original_shape = None
    selected_stride = None
    n_messages = 0
    n_extracted = 0
    try:
        try:
            import eccodes
        except ImportError as exc:
            raise ImportError("Sequential GRIB cache requires: pip install eccodes") from exc
        for path in paths:
            print(f"[grib_stream] Sequential scan (no source seek): {path}", flush=True)
            for number, payload in _raw_messages(path):
                n_messages += 1
                handle = eccodes.codes_new_from_message(payload)
                try:
                    short = str(_get(eccodes, handle, "shortName", "")).lower()
                    var = low_aliases.get(short)
                    if var is None:
                        # Some files encode a different shortName but preserve
                        # their requested parameter aliases in paramId metadata.
                        continue
                    day, hour = _valid_day_hour(eccodes, handle)
                    if day not in wanted:
                        continue
                    if presence[var][day][hour]:
                        raise ValueError(
                            f"Duplicate {var} at {day} {hour:02d}:00 across GRIB files. "
                            "Do not silently average different forecast cycles."
                        )
                    grid_type = str(_get(eccodes, handle, "gridType", ""))
                    if grid_type != "regular_ll":
                        raise ValueError(f"Expected regular_ll GRIB grid, found {grid_type!r}")
                    ni = int(_get(eccodes, handle, "Ni", 0))
                    nj = int(_get(eccodes, handle, "Nj", 0))
                    shape = (nj, ni)
                    if ni < 2 or nj < 2:
                        raise ValueError(f"Invalid GRIB grid {shape}")
                    if original_shape is None:
                        original_shape = shape
                        if spatial_stride == "auto":
                            if shape == tuple(grid_shape):
                                selected_stride = 1
                            else:
                                ratios = ((shape[0] - 1) / (grid_shape[0] - 1),
                                          (shape[1] - 1) / (grid_shape[1] - 1))
                                if ratios[0] != ratios[1] or not ratios[0].is_integer():
                                    raise ValueError(f"GRIB grid {shape} does not match target {grid_shape}")
                                selected_stride = int(ratios[0])
                        else:
                            selected_stride = int(spatial_stride)
                        reduced = ((nj - 1) // selected_stride + 1,
                                   (ni - 1) // selected_stride + 1)
                        if selected_stride < 1 or reduced != tuple(grid_shape):
                            raise ValueError(f"GRIB stride {selected_stride} gives {reduced}, "
                                             f"but target grid is {grid_shape}")
                    elif shape != original_shape:
                        raise ValueError(f"Source grid changed from {original_shape} to {shape}")
                    values = np.asarray(eccodes.codes_get_array(handle, "values"),
                                        dtype=np.float64)
                    if values.size != ni * nj:
                        raise ValueError(f"Decoded {values.size} values, expected {ni*nj}")
                    values = values.reshape(nj, ni)[::selected_stride, ::selected_stride]
                    missing = _get(eccodes, handle, "missingValue")
                    if missing is not None:
                        values = np.where(values == float(missing), np.nan, values)
                    writers.write(var, day, hour, values.astype(np.float32))
                    presence[var][day][hour] = True
                    n_extracted += 1
                finally:
                    eccodes.codes_release(handle)
                    if progress_every and n_messages % progress_every == 0:
                        print(f"[grib_stream] {n_messages:,} GRIB messages scanned, "
                              f"{n_extracted:,} hours cached", flush=True)
        if original_shape is None:
            raise ValueError("No requested GRIB messages for requested variables/dates")
    finally:
        writers.close()

    complete = {var: {} for var in aliases}
    missing_counts = {}
    for var in aliases:
        incomplete = 0
        for day, bitmap in presence[var].items():
            path = writers.filename(var, day)
            if bitmap.all():
                complete[var][day] = path
            else:
                incomplete += 1
                path.unlink(missing_ok=True)
        missing_counts[var] = incomplete
    if not all(complete[var] for var in aliases):
        raise ValueError(f"Some variables have no complete 24-hour days: {missing_counts}")
    metadata = {
        "signature": signature,
        "source_shape": list(original_shape),
        "spatial_stride": selected_stride,
        "complete_days": {var: [d.isoformat() for d in complete[var]] for var in aliases},
        "incomplete_day_counts": missing_counts,
        "read_messages": n_messages,
        "cached_hourly_fields": n_extracted,
    }
    temp = manifest_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(manifest_path)
    print(f"[grib_stream] Done: {n_messages:,} messages; {n_extracted:,} "
          f"hourly fields; cache={cache}", flush=True)
    return complete, original_shape, selected_stride
