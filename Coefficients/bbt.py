"""Chunked calculation of B @ B.T from assembled two-dimensional diffusion maps.

The Ocean_components channel convention is [B00, B11, B01, B10].
An assembled input B array has shape (time, 4, height, width).
The output retains the same channel order for the covariance B @ B.T.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np


def _validate_b_array(array: np.ndarray) -> None:
    """Validate the time-first, four-channel diffusion array layout.

    Parameters
    ----------
    array : numpy.ndarray
        Input with shape ``(T, 4, H, W)``.
    """
    if array.ndim != 4 or array.shape[1] != 4:
        raise ValueError(
            "B must have shape (time, 4, height, width) "
            "with channels [B00, B11, B01, B10]; "
            f"received {array.shape}"
        )
    if any(dimension < 1 for dimension in (array.shape[0], *array.shape[2:])):
        raise ValueError("B must contain at least one time step and a nonempty spatial grid")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"Expected floating-point B arrays, received {array.dtype}")


def BBT(
    b_file: str | Path,
    output_file: str | Path | None = None,
    *,
    chunk_days: int = 32,
    overwrite: bool = False,
    output_dtype: str | np.dtype | None = None,
    show_progress: bool = True,
) -> Path:
    """Save the time-resolved matrix product B @ B.T without loading all B.

    Parameters
    ----------
    b_file : str or pathlib.Path
        Existing, uncompressed assembled ``B_*.npy`` file. Expected array
        shape is ``(T, 4, H, W)`` with channels ``[B00, B11, B01, B10]``.
    output_file : str or pathlib.Path, optional
        Output filename. By default, the leading ``B_`` in the input filename
        is replaced by ``BBT_`` in the same directory.
    chunk_days : int, default=32
        Maximum number of time steps processed at once. Lower this if memory
        is constrained; increase it for potentially faster sequential IO.
    overwrite : bool, default=False
        Recompute an existing output. A valid cached output is reused only
        when it is at least as recent as the input file.
    output_dtype : floating-point dtype, optional
        Storage dtype, defaulting to the dtype of the input B array.
    show_progress : bool, default=True
        Print processing progress and the destination path.

    Returns
    -------
    pathlib.Path
        Path of the saved covariance array with shape ``(T, 4, H, W)``.

    Notes
    -----
    For B = [[b00,b01],[b10,b11]], the resulting channels are:

        [b00**2 + b01**2, b10**2 + b11**2,
         b00*b10 + b01*b11, b00*b10 + b01*b11].

    The product is a *covariance/amplitude-product* map; no square-root or
    additional time-step scaling is applied. The source B is not modified.
    Results are written to a temporary ``.npy`` file and atomically renamed
    after all chunks are successfully processed.
    """
    source_path = Path(b_file).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".npy":
        raise ValueError("BBT requires an uncompressed .npy input file")
    if not isinstance(chunk_days, int) or isinstance(chunk_days, bool) or chunk_days < 1:
        raise ValueError("chunk_days must be a positive integer")

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    _validate_b_array(source)
    dtype = np.dtype(source.dtype if output_dtype is None else output_dtype)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError("output_dtype must be a floating-point dtype")

    if output_file is None:
        if not source_path.name.startswith("B_"):
            raise ValueError("Specify output_file for a source not named B_*.npy")
        destination = source_path.with_name("BBT_" + source_path.name[2:])
    else:
        destination = Path(output_file).expanduser()

    if destination.resolve() == source_path.resolve():
        raise ValueError("Input and output paths must differ; refusing to overwrite B")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        try:
            cached = np.load(destination, mmap_mode="r", allow_pickle=False)
            valid = cached.shape == source.shape and cached.dtype == dtype
            del cached
        except (OSError, ValueError, TypeError, EOFError):
            valid = False
        if not valid:
            raise ValueError(
                f"The existing BBT file is invalid: {destination}. "
                "Use overwrite=True to rebuild it."
            )
        if destination.stat().st_mtime_ns < source_path.stat().st_mtime_ns:
            raise ValueError(
                f"The existing BBT file is older than B: {destination}. "
                "Use overwrite=True to recompute the product."
            )
        if show_progress:
            print(f"BBT: reusing existing array: {destination}", flush=True)
        del source
        return destination

    temp_path = None
    started = time.perf_counter()
    n_days = source.shape[0]
    update_interval = max(chunk_days, (n_days + 9) // 10)
    next_update = update_interval

    try:
        fd, filename = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.stem}_", suffix=".npy"
        )
        os.close(fd)
        temp_path = Path(filename)
        result = np.lib.format.open_memmap(
            temp_path, mode="w+", dtype=dtype, shape=source.shape
        )

        # Reuse one scratch channel for every chunk to reduce allocations.
        scratch_buffer = np.empty(
            (min(n_days, chunk_days), *source.shape[2:]), dtype=dtype
        )
        for start in range(0, n_days, chunk_days):
            stop = min(n_days, start + chunk_days)
            b = source[start:stop]
            out = result[start:stop]
            scratch = scratch_buffer[:stop - start]

            # [0, 0] = B00*B00 + B01*B01
            np.multiply(b[:, 0], b[:, 0], out=out[:, 0])
            np.multiply(b[:, 2], b[:, 2], out=scratch)
            np.add(out[:, 0], scratch, out=out[:, 0])

            # [1, 1] = B10*B10 + B11*B11
            np.multiply(b[:, 3], b[:, 3], out=out[:, 1])
            np.multiply(b[:, 1], b[:, 1], out=scratch)
            np.add(out[:, 1], scratch, out=out[:, 1])

            # [0, 1] = [1, 0] = B00*B10 + B01*B11
            np.multiply(b[:, 0], b[:, 3], out=out[:, 2])
            np.multiply(b[:, 2], b[:, 1], out=scratch)
            np.add(out[:, 2], scratch, out=out[:, 2])
            out[:, 3] = out[:, 2]

            if show_progress and (stop >= next_update or stop == n_days):
                elapsed = time.perf_counter() - started
                print(
                    f"BBT: {stop}/{n_days} days ({100 * stop / n_days:.1f}%), "
                    f"elapsed {elapsed:.1f} s",
                    flush=True,
                )
                next_update = stop + update_interval

        result.flush()
        # Release all mmap views before renaming; this matters on Windows.
        del out, b, scratch, scratch_buffer, result, source
        os.replace(temp_path, destination)
        temp_path = None
        if show_progress:
            print(f"BBT saved: {destination}", flush=True)
        return destination
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def bbt(*args, **kwargs) -> Path:
    """Lowercase alias for :func:`BBT`, with identical parameters."""
    return BBT(*args, **kwargs)
