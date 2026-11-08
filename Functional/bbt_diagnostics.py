"""Chunked diagnostics for two-dimensional diffusion matrices and BBT arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _validate_array_layout(array: np.ndarray, name: str) -> None:
    """Validate a time-first four-channel matrix-field array.

    Parameters
    ----------
    array : numpy.ndarray
        Array expected to have shape ``(T, 4, H, W)``.
    name : str
        Array name used in error messages.

    Raises
    ------
    ValueError
        If the array does not have the required dimensions or channel count.
    TypeError
        If the array is not floating-point.
    """
    if array.ndim != 4 or array.shape[1] != 4:
        raise ValueError(
            f"{name} must have shape (time, 4, height, width); received {array.shape}"
        )
    if any(size < 1 for size in (array.shape[0], array.shape[2], array.shape[3])):
        raise ValueError(f"{name} must have nonempty time and spatial dimensions")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be floating-point; received {array.dtype}")


def _expected_bbt(b: np.ndarray) -> np.ndarray:
    """Calculate B @ B.T for channel order [B00, B11, B01, B10]."""
    b = np.asarray(b, dtype=np.float64)
    result = np.empty(b.shape, dtype=np.float64)
    result[:, 0] = b[:, 0] ** 2 + b[:, 2] ** 2
    result[:, 1] = b[:, 3] ** 2 + b[:, 1] ** 2
    cross = b[:, 0] * b[:, 3] + b[:, 2] * b[:, 1]
    result[:, 2] = cross
    result[:, 3] = cross
    return result


def validate_bbt_files(
    b_file: str | Path,
    bbt_file: str | Path,
    *,
    chunk_days: int = 32,
    comparison_rtol: float = 1e-5,
    comparison_atol: float = 1e-8,
    psd_rtol: float = 1e-7,
    psd_atol: float = 1e-10,
    report_file: str | Path | None = None,
    show_progress: bool = True,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Validate a saved BBT array against B and test its matrix properties.

    Parameters
    ----------
    b_file : str or pathlib.Path
        Assembled B array with shape ``(T, 4, H, W)`` and channel order
        ``[B00, B11, B01, B10]``.
    bbt_file : str or pathlib.Path
        Saved BBT array with the same shape and channel order
        ``[D00, D11, D01, D10]``.
    chunk_days : int, default=32
        Number of time steps processed per chunk.
    comparison_rtol, comparison_atol : float
        Relative and absolute tolerances for comparison with independently
        reconstructed B @ B.T.
    psd_rtol, psd_atol : float
        Relative and absolute tolerances for positive-semidefinite checks.
    report_file : str or pathlib.Path, optional
        JSON report destination. By default, a report is written beside BBT.
    show_progress : bool, default=True
        Whether to print progress and the final diagnostic summary.
    raise_on_failure : bool, default=False
        Raise ``ValueError`` if any numerical validation check fails.

    Returns
    -------
    dict
        Diagnostic metrics, validation flags, and input paths.

    Notes
    -----
    The stored channels are interpreted as ``[D00, D11, D01, D10]``.
    Symmetry and positive semidefiniteness are tested only where all four
    matrix entries are finite. The independent BBT comparison also checks
    matching finite-value masks, so missing data are not silently ignored.
    The input files are opened read-only and are never modified.
    """
    b_path = Path(b_file).expanduser()
    d_path = Path(bbt_file).expanduser()
    if not b_path.is_file():
        raise FileNotFoundError(b_path)
    if not d_path.is_file():
        raise FileNotFoundError(d_path)
    if not isinstance(chunk_days, int) or isinstance(chunk_days, bool) or chunk_days < 1:
        raise ValueError("chunk_days must be a positive integer")
    tolerances = (comparison_rtol, comparison_atol, psd_rtol, psd_atol)
    if any(not np.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("All numerical tolerances must be finite and nonnegative")

    b_array = np.load(b_path, mmap_mode="r", allow_pickle=False)
    d_array = np.load(d_path, mmap_mode="r", allow_pickle=False)
    _validate_array_layout(b_array, "B")
    _validate_array_layout(d_array, "BBT")
    if b_array.shape != d_array.shape:
        raise ValueError(f"B and BBT shapes differ: {b_array.shape} != {d_array.shape}")

    total_cells = int(np.prod((b_array.shape[0], *b_array.shape[2:])))
    finite_matrix_count = 0
    nonfinite_bbt_matrix_count = 0
    nonfinite_b_matrix_count = 0
    symmetry_violations = 0
    psd_violations = 0
    negative_diagonal_count = 0
    finite_mask_mismatches = 0
    comparison_mismatches = 0
    max_symmetry_error = 0.0
    max_abs_error = 0.0
    max_relative_error = 0.0
    min_eigenvalue = float("inf")
    max_abs_matrix_entry = 0.0

    for start in range(0, b_array.shape[0], chunk_days):
        stop = min(b_array.shape[0], start + chunk_days)
        b = np.asarray(b_array[start:stop], dtype=np.float64)
        d = np.asarray(d_array[start:stop], dtype=np.float64)
        expected = _expected_bbt(b)

        b_finite = np.all(np.isfinite(b), axis=1)
        d_finite = np.all(np.isfinite(d), axis=1)
        expected_finite = np.all(np.isfinite(expected), axis=1)
        nonfinite_b_matrix_count += int(np.count_nonzero(~b_finite))
        nonfinite_bbt_matrix_count += int(np.count_nonzero(~d_finite))
        finite_mask_mismatches += int(np.count_nonzero(d_finite != expected_finite))

        comparable = np.isfinite(d) & np.isfinite(expected)
        if np.any(comparable):
            difference = np.abs(d[comparable] - expected[comparable])
            max_abs_error = max(max_abs_error, float(np.max(difference)))
            denominator = np.abs(expected[comparable]) + comparison_atol
            relative = difference / denominator
            max_relative_error = max(max_relative_error, float(np.max(relative)))
            comparison_mismatches += int(np.count_nonzero(~np.isclose(
                d[comparable], expected[comparable],
                rtol=comparison_rtol, atol=comparison_atol,
            )))

        valid = d_finite
        if not np.any(valid):
            if show_progress:
                print(f"BBT diagnostics: checked {stop}/{b_array.shape[0]} time steps", flush=True)
            continue

        d00 = d[:, 0][valid]
        d11 = d[:, 1][valid]
        d01 = d[:, 2][valid]
        d10 = d[:, 3][valid]
        finite_matrix_count += len(d00)
        symmetry_error = np.abs(d01 - d10)
        if symmetry_error.size:
            max_symmetry_error = max(max_symmetry_error, float(np.max(symmetry_error)))
            symmetry_scale = np.maximum(1.0, np.maximum(np.abs(d01), np.abs(d10)))
            symmetry_violations += int(np.count_nonzero(
                symmetry_error > psd_atol + psd_rtol * symmetry_scale
            ))

        negative_diagonal_count += int(np.count_nonzero(d00 < -(psd_atol + psd_rtol * np.maximum(1.0, np.abs(d00)))))
        negative_diagonal_count += int(np.count_nonzero(d11 < -(psd_atol + psd_rtol * np.maximum(1.0, np.abs(d11)))))
        scale = np.maximum(1.0, np.maximum.reduce((np.abs(d00), np.abs(d11), np.abs(d01), np.abs(d10))))
        matrices = np.empty((len(d00), 2, 2), dtype=np.float64)
        matrices[:, 0, 0] = d00
        matrices[:, 1, 1] = d11
        off_diagonal = 0.5 * (d01 + d10)
        matrices[:, 0, 1] = off_diagonal
        matrices[:, 1, 0] = off_diagonal
        eigenvalues = np.linalg.eigvalsh(matrices)
        local_minimum = float(np.min(eigenvalues[:, 0]))
        min_eigenvalue = min(min_eigenvalue, local_minimum)
        psd_violations += int(np.count_nonzero(eigenvalues[:, 0] < -(psd_atol + psd_rtol * scale)))
        max_abs_matrix_entry = max(
            max_abs_matrix_entry,
            float(np.max(np.maximum.reduce((np.abs(d00), np.abs(d11), np.abs(d01), np.abs(d10))))),
        )

        if show_progress:
            print(f"BBT diagnostics: checked {stop}/{b_array.shape[0]} time steps", flush=True)

    if min_eigenvalue == float("inf"):
        min_eigenvalue = float("nan")

    comparison_ok = finite_mask_mismatches == 0 and comparison_mismatches == 0
    symmetry_ok = symmetry_violations == 0
    psd_ok = (finite_matrix_count > 0 and psd_violations == 0
              and negative_diagonal_count == 0)
    report: dict[str, Any] = {
        "B_file": str(b_path.resolve()),
        "BBT_file": str(d_path.resolve()),
        "shape": list(b_array.shape),
        "B_dtype": str(b_array.dtype),
        "BBT_dtype": str(d_array.dtype),
        "chunk_days": chunk_days,
        "tolerances": {
            "comparison_rtol": comparison_rtol,
            "comparison_atol": comparison_atol,
            "psd_rtol": psd_rtol,
            "psd_atol": psd_atol,
        },
        "counts": {
            "total_spatiotemporal_cells": total_cells,
            "finite_matrices_in_BBT": finite_matrix_count,
            "nonfinite_matrices_in_B": nonfinite_b_matrix_count,
            "nonfinite_matrices_in_BBT": nonfinite_bbt_matrix_count,
            "finite_mask_mismatches": finite_mask_mismatches,
            "comparison_mismatches": comparison_mismatches,
            "symmetry_violations": symmetry_violations,
            "positive_semidefinite_violations": psd_violations,
            "negative_diagonal_entries": negative_diagonal_count,
        },
        "metrics": {
            "maximum_absolute_BBT_reconstruction_error": max_abs_error,
            "maximum_relative_BBT_reconstruction_error": max_relative_error,
            "maximum_absolute_symmetry_error": max_symmetry_error,
            "minimum_eigenvalue": min_eigenvalue,
            "maximum_absolute_matrix_entry": max_abs_matrix_entry,
        },
        "checks": {
            "matches_independent_B_times_BT": comparison_ok,
            "symmetric_within_tolerance": symmetry_ok,
            "positive_semidefinite_within_tolerance": psd_ok,
        },
    }
    report["checks"]["all_checks_passed"] = all(report["checks"].values())

    destination = (
        Path(report_file).expanduser()
        if report_file is not None
        else d_path.with_name(f"{d_path.stem}_diagnostics.json")
    )
    report["report_file"] = str(destination.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")

    if show_progress:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True), flush=True)
    if raise_on_failure and not report["checks"]["all_checks_passed"]:
        raise ValueError(f"BBT validation failed; see {destination}")
    return report
