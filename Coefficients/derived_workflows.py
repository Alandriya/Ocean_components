"""Rolling C/F/FS from continuous A/B, with crash-safe progress checkpoints."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

from Coefficients.coefficients_cf import compute_c, compute_f, compute_fs
from Coefficients.common import load_water_mask

_SCHEMA = 1


def _stamp(path):
    """Build a file metadata stamp for cache validation."""
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _source_stamp(path):
    """Record the input source metadata used to validate derived outputs."""
    return {"path": str(Path(path).resolve()), **_stamp(path)}


def _sync_dir(path):
    # POSIX directory fsync makes rename durable; Windows does not support it.
    """Synchronize directory metadata after publishing an output file."""
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _atomic_json(path, value):
    """Write JSON to a temporary file and atomically publish it."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _sync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def _exclusive_lock(path):
    """OS lock released on process exit, even after interruption; never unlink it."""
    with path.open("a+b") as f:
        if f.seek(0, os.SEEK_END) == 0:
            f.write(b"\0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another derived_cf job is using {path.parent}") from exc
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another derived_cf job is using {path.parent}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _duration(seconds):
    """Format or report the elapsed calculation time."""
    if not np.isfinite(seconds):
        return "--"
    s = max(0, int(seconds))
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    minutes, s = divmod(s, 60)
    return (f"{days}d " if days else "") + f"{hours:02}:{minutes:02}:{s:02}"


def _validate_array(path, shape, dtype):
    """Validate the stored array against the expected layout."""
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if arr.shape != tuple(shape) or arr.dtype != np.dtype(dtype):
            raise ValueError(f"Unexpected output shape/dtype: {path}: {arr.shape}, {arr.dtype}")
    finally:
        del arr


def _flush(target, path):
    """Flush pending array or file writes."""
    target.flush()
    with path.open("r+b") as f:
        os.fsync(f.fileno())


def _progress(tag, key, n, total, start_label, initial, started, completed_before, grand_total):
    """Compute progress for derived workflows from ``tag``, ``key``, ``n``, ``total``, ``start_label``."""
    elapsed = time.monotonic() - started
    rate = (n - initial) / elapsed if elapsed > 0 and n > initial else 0.0
    eta = (total - n) / rate if rate > 0 else float("nan")
    label = start_label + n - 1 if n else "--"
    print(f"{tag}[{key}] {n:,}/{total:,} ({100*n/total:5.1f}%) | "
          f"last label={label} | {rate:.2f} windows/s | elapsed={_duration(elapsed)} | "
          f"ETA(stage)={_duration(eta)} | total={completed_before+n:,}/{grand_total:,} "
          f"({100*(completed_before+n)/grand_total:.1f}%)", flush=True)


def _calculate(key, a, b, width, step, mask):
    # Preserve exactly the previous window, missing-data rules and formulas.
    """Calculate the requested derived coefficient field."""
    aw = np.asarray(a[step:step + width], dtype=float)
    bw = np.asarray(b[step:step + width], dtype=float)
    if key == "C":
        return compute_c(aw, bw, mask)
    valid = (np.all(np.isfinite(aw), axis=(0, 1)) &
             np.all(np.isfinite(bw), axis=(0, 1)))
    average_a, average_b = aw.mean(axis=0), bw.mean(axis=0)
    average_a[:, ~valid] = np.nan
    average_b[:, ~valid] = np.nan
    return (compute_f(average_a, average_b, mask) if key == "F"
            else compute_fs(average_a, average_b, mask))


def _process_stage(*, key, width, a, b, mask, first_label, parent, input_identity,
                   overwrite, resume, checkpoint_every, progress_seconds,
                   tag, completed_before, grand_total, check_sources):
    """Run one stage of the derived coefficient workflow."""
    first = first_label + width - 1
    last = first_label + len(a) - 1
    total = len(a) - width + 1
    shape = (total, 2, *a.shape[2:]) if key in ("C", "FS") else (total, *a.shape[2:])
    output = parent / f"{key}_{first}-{last}.npy"
    partial = output.with_name(f".{output.name}.partial")
    progress = output.with_name(f".{output.name}.progress.json")
    signature = dict(input_identity, coefficient=key, window=width,
                     shape=list(shape), dtype=np.dtype(float).str,
                     first_label=first, last_label=last)
    state = None
    if resume and progress.exists():
        try:
            state = json.loads(progress.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"Unreadable checkpoint {progress}; use resume=False to restart explicitly") from exc
        if not isinstance(state, dict) or state.get("schema") != _SCHEMA or state.get("signature") != signature:
            raise RuntimeError(f"Checkpoint inputs/settings changed: {progress}. "
                               "Use resume=False, overwrite=True to explicitly restart.")
        n = state.get("next_index")
        if type(n) is not int or not 0 <= n <= total:
            raise ValueError(f"Invalid checkpoint index in {progress}: {n}")
        if state.get("status") not in ("running", "ready", "complete"):
            raise ValueError(f"Invalid checkpoint state: {progress}")
        if state["status"] in ("ready", "complete"):
            if n != total:
                raise ValueError(f"Final checkpoint has incomplete count: {progress}")
            # A crash after os.replace but before writing 'complete' is recoverable.
            published = not partial.exists() and output.is_file()
            if published and _stamp(output) == state.get("final_stamp"):
                _validate_array(output, shape, float)
                if state["status"] != "complete":
                    state["status"] = "complete"
                    _atomic_json(progress, state)
                print(f"{tag}[{key}] SKIP verified complete: {total:,}/{total:,}, {output.name}", flush=True)
                return {"path": str(output), "first_label": first, "last_label": last}
            if state["status"] == "complete" or not partial.is_file():
                raise RuntimeError(f"Completed output is missing or changed: {output}; "
                                   "use resume=False, overwrite=True to rebuild.")
            if _stamp(partial) != state.get("final_stamp"):
                raise RuntimeError(f"Ready checkpoint data changed: {partial}")
        if not partial.is_file():
            raise RuntimeError(f"Missing checkpoint array: {partial}; use resume=False to restart")
        _validate_array(partial, shape, float)
        print(f"{tag}[{key}] RESUME: {n:,}/{total:,} committed; next label={first+n}", flush=True)
        target = np.lib.format.open_memmap(partial, mode="r+")
    else:
        if not resume and (output.exists() or partial.exists() or progress.exists()) and not overwrite:
            raise FileExistsError(f"Restart would replace existing results: {output}; set overwrite=True")
        if output.exists() and not overwrite:
            raise FileExistsError(f"Unverified legacy array {output}; no completion checkpoint. "
                                  "Set overwrite=True to rebuild once, without deleting the old file first.")
        if partial.exists() and resume:
            # A crash before the initial checkpoint leaves an untracked array.
            # No committed data exist; only recreate with explicit overwrite permission.
            if not overwrite:
                raise FileExistsError(f"Untracked partial array: {partial}; set overwrite=True or restart")
            print(f"{tag}[{key}] Ignoring untracked partial file; restarting at index 0", flush=True)
        if output.exists():
            print(f"{tag}[{key}] Existing result is unverified or restart requested; "
                  "keeping old file until replacement is complete", flush=True)
        if not resume:
            progress.unlink(missing_ok=True)
        target = np.lib.format.open_memmap(partial, mode="w+", dtype=float, shape=shape)
        n = 0
        state = {"schema": _SCHEMA, "signature": signature, "next_index": 0,
                 "status": "running"}
        _flush(target, partial)
        _atomic_json(progress, state)
        print(f"{tag}[{key}] START: {total:,} windows; labels={first}..{last}; width={width}", flush=True)

    initial = n
    started = time.monotonic()
    last_checkpoint = started
    committed = n

    def checkpoint():
        """Persist the progress of the current calculation."""
        nonlocal committed, last_checkpoint
        _flush(target, partial)
        # If a process stops after a data flush but before the metadata rename,
        # the uncommitted suffix is simply computed again on the next run.
        state.update(next_index=n, status="running")
        state.pop("final_stamp", None)
        _atomic_json(progress, state)
        committed = n
        last_checkpoint = time.monotonic()

    try:
        while n < total:
            target[n] = _calculate(key, a, b, width, n, mask)
            n += 1
            if (n == total or n == initial + 1 or n - committed >= checkpoint_every or
                    time.monotonic() - last_checkpoint >= progress_seconds):
                checkpoint()
                _progress(tag, key, n, total, first, initial, started, completed_before, grand_total)
        check_sources()
        _flush(target, partial)
    except BaseException:
        # Save all fully calculated windows on a normal exception or Ctrl+C.
        # A second interruption/power loss still leaves the previous checkpoint.
        try:
            checkpoint()
            print(f"{tag}[{key}] STOPPED: saved {n:,}/{total:,}; "
                  f"next label={first+n}; restart the same task to resume", flush=True)
        except BaseException as exc:
            print(f"{tag}[{key}] Could not save final checkpoint: {exc}; "
                  "the previous committed checkpoint is retained", flush=True)
        raise
    finally:
        # No views of target leave this function. Releasing the mmap before rename
        # is important on Windows (which cannot replace open mapped files).
        del target

    # Record the final timestamp only after the memory mapping is closed.
    # On Windows, closing a write handle can update the modification time.
    state.update(next_index=total, status="ready", final_stamp=_stamp(partial))
    _atomic_json(progress, state)
    os.replace(partial, output)
    _sync_dir(output.parent)
    state["status"] = "complete"
    _atomic_json(progress, state)
    print(f"{tag}[{key}] COMPLETE: {total:,} windows | run time={_duration(time.monotonic()-started)} | {output}", flush=True)
    return {"path": str(output), "first_label": first, "last_label": last}


def run_continuous_cf(files_path_prefix, names=("sensible", "latent"), *,
                      n_bins_2d=12, method="np", correlation_window=14,
                      mean_width=7, mask=None, overwrite=False, resume=True,
                      checkpoint_every=50, progress_seconds=30.0):
    """Calculate the existing rolling C/F/FS statistics with restart support.

    The output paths and endpoint labels are identical to the original workflow.
    With resume=True (default), valid partial *and completed* checkpoints take
    priority over overwrite. overwrite=True permits replacing an old unverified
    array at successful completion. To intentionally recalculate verified output,
    pass resume=False, overwrite=True. Changed sources/settings are rejected
    rather than mixing old and new windows. Checkpoints are durable every
    checkpoint_every windows or progress_seconds seconds, whichever comes first.
    """
    if correlation_window < 3 or mean_width < 1:
        raise ValueError("Invalid window sizes")
    if method not in ("np", "sp"):
        raise ValueError("method must be 'np' or 'sp'")
    if checkpoint_every < 1 or not np.isfinite(progress_seconds) or progress_seconds <= 0:
        raise ValueError("checkpoint_every and progress_seconds must be positive")
    folder = "Nonparametric_2d" if method == "np" else "Semiparametric_2d"
    parent = (Path(files_path_prefix) / "Coefficients" / "continuous" /
              f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}" / folder)
    manifest_path = Path(files_path_prefix) / "Coefficients" / "continuous" / (
        f"manifest_{names[0]}_{names[1]}_{n_bins_2d}.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if method not in manifest["methods"]:
        raise ValueError(f"Method {method} not present in the continuous coefficient series")
    pa, pb = (Path(manifest["methods"][method][key]) for key in ("A", "B"))
    source_stamps = {"A": _source_stamp(pa), "B": _source_stamp(pb)}
    a = np.load(pa, mmap_mode="r", allow_pickle=False)
    b = np.load(pb, mmap_mode="r", allow_pickle=False)
    if (a.ndim != 4 or b.ndim != 4 or a.shape[1] != 2 or b.shape[1] != 4 or
            len(a) != len(b) or a.shape[2:] != b.shape[2:]):
        raise ValueError("Expected aligned A=(T,2,H,W) and B=(T,4,H,W)")
    if len(a) < max(correlation_window, mean_width):
        raise ValueError("Insufficient number of days for the requested windows")
    first_label = int(manifest["first_label"])
    water = load_water_mask(mask, a.shape[2:])
    identity = {"sources": source_stamps, "mask_sha256": hashlib.sha256(water.tobytes()).hexdigest(),
                "method": method, "names": list(names), "n_bins_2d": int(n_bins_2d),
                "source_first_label": first_label, "source_shape_A": list(a.shape),
                "source_shape_B": list(b.shape)}

    def check_sources():
        """Validate the availability of required input sources."""
        if {"A": _source_stamp(pa), "B": _source_stamp(pb)} != source_stamps:
            raise RuntimeError("A/B changed during derived_cf; restart with resume=False after inputs stabilize")

    parent.mkdir(parents=True, exist_ok=True)
    tag = f"[derived_cf][{names[0]}-{names[1]}][{method}][pid={os.getpid()}]"
    stages = (("C", correlation_window), ("F", mean_width), ("FS", mean_width))
    grand_total = sum(len(a) - width + 1 for _, width in stages)
    result = {}
    completed_before = 0
    started = time.monotonic()
    with _exclusive_lock(parent / ".derived_cf.lock"):
        print(f"{tag} START workflow | A/B transitions={len(a):,} | C width={correlation_window} | "
              f"F/FS width={mean_width} | total={grand_total:,} windows | resume={resume}", flush=True)
        print(f"{tag} Output: {parent}", flush=True)
        for key, width in stages:
            check_sources()
            result[key] = _process_stage(key=key, width=width, a=a, b=b, mask=water,
                first_label=first_label, parent=parent, input_identity=identity,
                overwrite=overwrite, resume=resume, checkpoint_every=checkpoint_every,
                progress_seconds=progress_seconds, tag=tag, completed_before=completed_before,
                grand_total=grand_total, check_sources=check_sources)
            completed_before += len(a) - width + 1
        print(f"{tag} ALL COMPLETE: {completed_before:,}/{grand_total:,}; "
              f"total run time={_duration(time.monotonic()-started)}", flush=True)
    return result
