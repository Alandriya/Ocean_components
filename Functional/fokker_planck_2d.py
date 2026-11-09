"""Prepare state-space coefficients for the two-dimensional Fokker-Planck equation.

This module does not solve the PDE. It reconstructs the drift vector A(x1, x2)
and diffusion tensor D(x1, x2) = B B.T on a fixed rectangular state-space grid
from the already assembled daily coefficient maps.

Conventions
-----------
- State coordinates are ordered exactly as ``names``.
- A has channels [A0, A1].
- BBT has channels [D00, D11, D01, D10].
- Daily coefficients for transition k are conditioned on the previous state,
  therefore A[k] and BBT[k] are paired with observations X[k], not X[k + 1].
- Spatial block averaging is optional and uses only mutually valid ocean cells.
- No interpolation or hole filling is performed at this stage.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np


def _validate_settings(bins, quantile_bounds, min_count, chunk_days,
                       spatial_block_size, sample_size, psd_rtol, psd_atol):
    """Validate phase-space reconstruction settings."""
    if (not isinstance(bins, (tuple, list)) or len(bins) != 2
            or any(not isinstance(v, (int, np.integer)) or v < 4 for v in bins)):
        raise ValueError("bins must contain two integers >= 4")
    q_low, q_high = map(float, quantile_bounds)
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("quantile_bounds must satisfy 0 <= low < high <= 1")
    if not isinstance(min_count, (int, np.integer)) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    if not isinstance(chunk_days, (int, np.integer)) or chunk_days < 1:
        raise ValueError("chunk_days must be a positive integer")
    if (not isinstance(spatial_block_size, (int, np.integer))
            or spatial_block_size < 1):
        raise ValueError("spatial_block_size must be a positive integer")
    if not isinstance(sample_size, (int, np.integer)) or sample_size < 1000:
        raise ValueError("sample_size must be at least 1000")
    if not (np.isfinite(psd_rtol) and psd_rtol >= 0
            and np.isfinite(psd_atol) and psd_atol >= 0):
        raise ValueError("PSD tolerances must be finite and nonnegative")


def _source_signature(path):
    """Return a stable metadata signature for an input file."""
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _resolve_sources(root, names, start_year, end_year, method):
    """Locate observations, drift and BBT files for one complete period."""
    from Data_processing.io import field_candidates, load_field

    if method not in ("np", "sp"):
        raise ValueError("method must be 'np' or 'sp'")
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("Exactly two different state variables are required")

    period = f"{start_year}-{end_year}"
    pair = f"{names[0]}_{names[1]}"
    coefficient_dir = Path(root) / "Coefficients" / period / pair
    a_path = coefficient_dir / f"A_{period}_{method}.npy"
    bbt_path = coefficient_dir / f"BBT_{period}_{method}.npy"

    if not a_path.is_file():
        raise FileNotFoundError(f"Missing assembled drift file: {a_path}")
    if not bbt_path.is_file():
        raise FileNotFoundError(
            f"Missing assembled BBT file: {bbt_path}. "
            "Run the bbt task before preparing the Fokker-Planck grid."
        )

    field_paths = []
    fields = []
    for name in names:
        candidates = field_candidates(root, name, start_year, end_year)
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"Missing full-period field {name}; checked: "
                + ", ".join(map(str, candidates))
            )
        field_paths.append(path)
        fields.append(load_field(root, name, start_year, end_year))

    a = np.load(a_path, mmap_mode="r", allow_pickle=False)
    bbt = np.load(bbt_path, mmap_mode="r", allow_pickle=False)

    if (a.ndim != 4 or a.shape[1] != 2
            or bbt.ndim != 4 or bbt.shape[1] != 4):
        raise ValueError(
            f"Expected A=(T,2,H,W) and BBT=(T,4,H,W); "
            f"received A={a.shape}, BBT={bbt.shape}"
        )
    if a.shape[0] != bbt.shape[0] or a.shape[2:] != bbt.shape[2:]:
        raise ValueError(
            f"A and BBT are not aligned: A={a.shape}, BBT={bbt.shape}"
        )
    if any(field.shape[1:] != a.shape[2:] for field in fields):
        raise ValueError(
            "Observation and coefficient spatial grids do not match"
        )

    # A[k] and BBT[k] describe X[k] -> X[k+1], hence observations need T+1 rows.
    n_transitions = int(a.shape[0])
    if any(len(field) < n_transitions + 1 for field in fields):
        raise ValueError(
            f"Need at least {n_transitions + 1} observations for "
            f"{n_transitions} transitions; got {[len(field) for field in fields]}"
        )

    signatures = {
        names[0]: _source_signature(field_paths[0]),
        names[1]: _source_signature(field_paths[1]),
        "A": _source_signature(a_path),
        "BBT": _source_signature(bbt_path),
    }
    return tuple(fields), a, bbt, signatures


def _water_mask(root, shape):
    """Load the project ocean mask, or fail instead of silently using land."""
    from Data_processing.io import read_ocean_mask
    from Coefficients.common import load_water_mask

    return load_water_mask(read_ocean_mask(root), shape)


def _block_mean(values, valid, block_size):
    """Average one time-first field over square ocean-only spatial blocks."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or values.ndim != 3:
        raise ValueError("values and valid must share shape (T,H,W)")

    if block_size == 1:
        return np.where(valid, values, np.nan)

    t, h, w = values.shape
    hb = h // block_size
    wb = w // block_size
    if hb < 1 or wb < 1:
        raise ValueError("spatial_block_size exceeds the spatial grid")

    hcut = hb * block_size
    wcut = wb * block_size
    reshaped = values[:, :hcut, :wcut].reshape(
        t, hb, block_size, wb, block_size
    )
    good = valid[:, :hcut, :wcut].reshape(
        t, hb, block_size, wb, block_size
    )
    sums = np.where(good, reshaped, 0.0).sum(axis=(2, 4))
    counts = good.sum(axis=(2, 4))
    return np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _sample_state_domain(fields, mask, n_transitions, *,
                         chunk_days, block_size, sample_size, seed):
    """Estimate robust state-space limits from deterministic block-state samples."""
    rng = np.random.default_rng(seed)
    n_chunks = (n_transitions + chunk_days - 1) // chunk_days
    per_chunk = max(1, (sample_size + n_chunks - 1) // n_chunks)
    selected = []

    for start in range(0, n_transitions, chunk_days):
        stop = min(n_transitions, start + chunk_days)
        x0 = np.asarray(fields[0][start:stop], dtype=np.float64)
        x1 = np.asarray(fields[1][start:stop], dtype=np.float64)
        valid = (
            mask[None, :, :]
            & np.isfinite(x0)
            & np.isfinite(x1)
        )
        x0b = _block_mean(x0, valid, block_size)
        x1b = _block_mean(x1, valid, block_size)
        good = np.isfinite(x0b) & np.isfinite(x1b)
        if not np.any(good):
            continue
        pairs = np.column_stack((x0b[good], x1b[good]))
        take = min(per_chunk, len(pairs))
        indices = rng.choice(len(pairs), size=take, replace=False)
        selected.append(pairs[indices])

    if not selected:
        raise ValueError("No finite paired state samples were found")
    sample = np.concatenate(selected, axis=0)
    if len(sample) > sample_size:
        indices = rng.choice(len(sample), size=sample_size, replace=False)
        sample = sample[indices]
    return sample


def _accumulate(values, joint_ids, n_cells):
    """Return one weighted bincount on flattened phase-space cells."""
    return np.bincount(
        joint_ids,
        weights=np.asarray(values, dtype=np.float64),
        minlength=n_cells,
    ).astype(np.float64, copy=False)


def _phase_eigenvalues(diffusion):
    """Return ordered eigenvalues of symmetric 2x2 diffusion matrices."""
    d00 = diffusion[0]
    d11 = diffusion[1]
    d01 = 0.5 * (diffusion[2] + diffusion[3])
    half_trace = 0.5 * (d00 + d11)
    radius = np.sqrt(np.maximum(
        0.0,
        (0.5 * (d00 - d11)) ** 2 + d01 ** 2,
    ))
    return np.stack((half_trace - radius, half_trace + radius), axis=0)


def _output_paths(root, names, start_year, end_year, method, bins):
    """Return NPZ and JSON destinations for one phase-space reconstruction."""
    folder = Path(root) / "Functional" / "FokkerPlanck2D"
    stem = (
        f"{names[0]}-{names[1]}_{start_year}-{end_year}_{method}_"
        f"{int(bins[0])}x{int(bins[1])}"
    )
    return folder / f"{stem}.npz", folder / f"{stem}.json"


def prepare_phase_space_coefficients(
        files_path_prefix,
        start_year,
        end_year,
        *,
        names=("sensible", "latent"),
        method="sp",
        bins=(80, 80),
        quantile_bounds=(0.005, 0.995),
        min_count=100,
        chunk_days=32,
        spatial_block_size=5,
        sample_size=500_000,
        random_seed=1979,
        psd_rtol=1e-7,
        psd_atol=1e-10,
        overwrite=False):
    """Reconstruct A(x1,x2) and D(x1,x2)=BBT on a uniform phase-space grid.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory containing observations and assembled coefficients.
    start_year, end_year : int
        Complete observation period.
    names : tuple[str, str]
        Ordered state variables matching coefficient channels.
    method : {'np', 'sp'}
        Coefficient estimator whose assembled A and BBT arrays are used.
    bins : tuple[int, int]
        Number of uniform physical-state cells along both coordinates.
    quantile_bounds : tuple[float, float]
        Robust marginal quantiles defining the outer rectangular domain.
        Samples outside the rectangle are counted and excluded from the grid.
    min_count : int
        Minimum number of block-state samples required to retain a grid cell.
    chunk_days : int
        Number of coefficient transitions read in one disk chunk.
    spatial_block_size : int
        Ocean-only square spatial averaging width. Use 1 to disable averaging.
    sample_size : int
        Maximum deterministic sample used only to estimate robust domain bounds.
    random_seed : int
        Seed for reproducible domain sampling.
    psd_rtol, psd_atol : float
        Tolerances for the post-aggregation positive-semidefinite check.
    overwrite : bool
        Rebuild an existing output even when its metadata match.

    Returns
    -------
    pathlib.Path
        Saved NPZ containing the phase-space grid and coefficient fields.

    Notes
    -----
    This is a preprocessing step for the 2D Fokker-Planck solver. It deliberately
    leaves under-populated cells as NaN and does not interpolate them.

    Daily coefficient row k was estimated from the transition X[k] -> X[k+1]
    using groups defined by X[k]. Consequently, coefficient row k is paired
    with observation row k here. Pairing it with X[k+1] would shift the state
    dependence by one day.
    """
    bins = tuple(int(v) for v in bins)
    _validate_settings(
        bins, quantile_bounds, min_count, chunk_days,
        spatial_block_size, sample_size, psd_rtol, psd_atol,
    )

    root = Path(files_path_prefix)
    fields, a, bbt, sources = _resolve_sources(
        root, names, start_year, end_year, method
    )
    n_transitions = int(a.shape[0])
    mask = _water_mask(root, a.shape[2:])

    output_path, metadata_path = _output_paths(
        root, names, start_year, end_year, method, bins
    )
    settings = {
        "names": list(names),
        "period": f"{start_year}-{end_year}",
        "method": method,
        "bins": list(bins),
        "quantile_bounds": [float(v) for v in quantile_bounds],
        "min_count": int(min_count),
        "chunk_days": int(chunk_days),
        "spatial_block_size": int(spatial_block_size),
        "sample_size": int(sample_size),
        "random_seed": int(random_seed),
        "psd_rtol": float(psd_rtol),
        "psd_atol": float(psd_atol),
        "sources": sources,
    }

    if output_path.is_file() and metadata_path.is_file() and not overwrite:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("settings") == settings:
            print(f"Reusing phase-space coefficients: {output_path}", flush=True)
            return output_path
        raise FileExistsError(
            f"Existing phase-space grid has different settings: {output_path}. "
            "Set overwrite=True to rebuild it."
        )

    print(
        f"Fokker-Planck 2D preprocessing: {n_transitions} transitions, "
        f"grid={bins[0]}x{bins[1]}, block={spatial_block_size}; "
        "estimating state-space bounds",
        flush=True,
    )
    sample = _sample_state_domain(
        fields,
        mask,
        n_transitions,
        chunk_days=chunk_days,
        block_size=spatial_block_size,
        sample_size=sample_size,
        seed=random_seed,
    )
    q_low, q_high = map(float, quantile_bounds)
    lower = np.quantile(sample, q_low, axis=0)
    upper = np.quantile(sample, q_high, axis=0)
    del sample

    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
            and np.all(lower < upper)):
        raise ValueError(
            f"Invalid sampled phase-space bounds: lower={lower}, upper={upper}"
        )

    x_edges = np.linspace(lower[0], upper[0], bins[0] + 1)
    y_edges = np.linspace(lower[1], upper[1], bins[1] + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    n_cells = bins[0] * bins[1]

    counts = np.zeros(n_cells, dtype=np.int64)
    sums = np.zeros((6, n_cells), dtype=np.float64)
    total_valid = 0
    total_inside = 0

    print(
        "Aggregating drift and BBT on the fixed phase-space grid",
        flush=True,
    )
    for start in range(0, n_transitions, chunk_days):
        stop = min(n_transitions, start + chunk_days)

        # Coefficient row k is paired with the PREVIOUS state X[k].
        x0 = np.asarray(fields[0][start:stop], dtype=np.float64)
        x1 = np.asarray(fields[1][start:stop], dtype=np.float64)
        drift = np.asarray(a[start:stop], dtype=np.float64)
        diffusion = np.asarray(bbt[start:stop], dtype=np.float64)

        valid = (
            mask[None, :, :]
            & np.isfinite(x0)
            & np.isfinite(x1)
            & np.all(np.isfinite(drift), axis=1)
            & np.all(np.isfinite(diffusion), axis=1)
        )

        x0b = _block_mean(x0, valid, spatial_block_size)
        x1b = _block_mean(x1, valid, spatial_block_size)
        a0b = _block_mean(drift[:, 0], valid, spatial_block_size)
        a1b = _block_mean(drift[:, 1], valid, spatial_block_size)
        d00b = _block_mean(diffusion[:, 0], valid, spatial_block_size)
        d11b = _block_mean(diffusion[:, 1], valid, spatial_block_size)
        d01b = _block_mean(diffusion[:, 2], valid, spatial_block_size)
        d10b = _block_mean(diffusion[:, 3], valid, spatial_block_size)

        good = (
            np.isfinite(x0b)
            & np.isfinite(x1b)
            & np.isfinite(a0b)
            & np.isfinite(a1b)
            & np.isfinite(d00b)
            & np.isfinite(d11b)
            & np.isfinite(d01b)
            & np.isfinite(d10b)
        )
        total_valid += int(np.count_nonzero(good))
        if not np.any(good):
            continue

        sx = x0b[good]
        sy = x1b[good]
        channels = (
            a0b[good], a1b[good],
            d00b[good], d11b[good], d01b[good], d10b[good],
        )

        inside = (
            (sx >= x_edges[0]) & (sx <= x_edges[-1])
            & (sy >= y_edges[0]) & (sy <= y_edges[-1])
        )
        if not np.any(inside):
            continue

        sx = sx[inside]
        sy = sy[inside]
        channels = tuple(values[inside] for values in channels)
        total_inside += int(len(sx))

        ix = np.searchsorted(x_edges, sx, side="right") - 1
        iy = np.searchsorted(y_edges, sy, side="right") - 1
        ix = np.clip(ix, 0, bins[0] - 1)
        iy = np.clip(iy, 0, bins[1] - 1)
        joint = ix * bins[1] + iy

        counts += np.bincount(joint, minlength=n_cells).astype(np.int64)
        for channel, values in enumerate(channels):
            sums[channel] += _accumulate(values, joint, n_cells)

        print(
            f"  transitions {start + 1}-{stop}/{n_transitions}",
            end="\r" if stop < n_transitions else "\n",
            flush=True,
        )

    occupancy = counts.reshape(bins)
    reliable = occupancy >= min_count

    means = np.full((6, n_cells), np.nan, dtype=np.float64)
    positive = counts > 0
    means[:, positive] = (
        sums[:, positive] / counts[positive][None, :]
    )
    means = means.reshape((6, *bins))

    drift_grid = means[:2]
    diffusion_grid = means[2:]

    # The validated source BBT is symmetric. Average the two stored off-diagonal
    # channels after aggregation to retain exact symmetry in the phase grid.
    symmetry_error_before = np.abs(diffusion_grid[2] - diffusion_grid[3])
    off_diagonal = 0.5 * (diffusion_grid[2] + diffusion_grid[3])
    diffusion_grid[2] = off_diagonal
    diffusion_grid[3] = off_diagonal

    drift_grid[:, ~reliable] = np.nan
    diffusion_grid[:, ~reliable] = np.nan

    eigenvalues = _phase_eigenvalues(diffusion_grid)
    scale = np.maximum.reduce((
        np.ones(bins, dtype=np.float64),
        np.abs(diffusion_grid[0]),
        np.abs(diffusion_grid[1]),
        np.abs(diffusion_grid[2]),
    ))
    tolerance = psd_atol + psd_rtol * scale
    psd_violations = reliable & (eigenvalues[0] < -tolerance)
    determinant = (
        diffusion_grid[0] * diffusion_grid[1]
        - diffusion_grid[2] ** 2
    )

    empirical_mass = occupancy.astype(np.float64)
    empirical_mass /= max(1.0, float(empirical_mass.sum()))
    dx = np.diff(x_edges)[:, None]
    dy = np.diff(y_edges)[None, :]
    empirical_density = empirical_mass / (dx * dy)

    diagnostics = {
        "n_transitions": n_transitions,
        "spatial_shape": list(a.shape[2:]),
        "water_cells": int(mask.sum()),
        "total_block_samples_with_finite_coefficients": int(total_valid),
        "samples_inside_phase_domain": int(total_inside),
        "samples_outside_phase_domain": int(total_valid - total_inside),
        "occupied_cells": int(np.count_nonzero(occupancy)),
        "reliable_cells": int(np.count_nonzero(reliable)),
        "total_cells": int(occupancy.size),
        "reliable_fraction": float(np.mean(reliable)),
        "maximum_occupancy": int(occupancy.max(initial=0)),
        "minimum_reliable_occupancy": (
            int(occupancy[reliable].min()) if np.any(reliable) else None
        ),
        "maximum_pre_symmetrization_error": (
            float(np.nanmax(symmetry_error_before))
            if np.any(np.isfinite(symmetry_error_before)) else None
        ),
        "minimum_phase_grid_eigenvalue": (
            float(np.nanmin(eigenvalues[0]))
            if np.any(np.isfinite(eigenvalues[0])) else None
        ),
        "psd_violations": int(np.count_nonzero(psd_violations)),
        "negative_diagonal_cells": int(np.count_nonzero(
            reliable & (
                (diffusion_grid[0] < -psd_atol)
                | (diffusion_grid[1] < -psd_atol)
            )
        )),
    }

    if diagnostics["reliable_cells"] == 0:
        raise ValueError(
            "No phase-space cells satisfy min_count; reduce min_count or bins"
        )
    if diagnostics["psd_violations"]:
        raise ValueError(
            f"Aggregated phase-space diffusion has "
            f"{diagnostics['psd_violations']} PSD violations"
        )

    first_state_date = date(start_year, 1, 1)
    last_state_date = first_state_date + timedelta(days=n_transitions - 1)
    last_observation_date = first_state_date + timedelta(days=n_transitions)

    metadata = {
        "settings": settings,
        "state_grid": {
            "x_name": names[0],
            "y_name": names[1],
            "x_bounds": [float(x_edges[0]), float(x_edges[-1])],
            "y_bounds": [float(y_edges[0]), float(y_edges[-1])],
            "uniform_physical_grid": True,
            "under_populated_cells_are_nan": True,
            "interpolation_performed": False,
        },
        "channel_conventions": {
            "drift": ["A0", "A1"],
            "diffusion": ["D00", "D11", "D01", "D10"],
            "diffusion_definition": "D = B @ B.T",
        },
        "time_alignment": {
            "coefficient_row_k_conditions_on": "observation_row_k",
            "transition": "observation_row_k -> observation_row_k_plus_1",
            "first_state_date": first_state_date.isoformat(),
            "last_state_date": last_state_date.isoformat(),
            "last_observation_date": last_observation_date.isoformat(),
        },
        "diagnostics": diagnostics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}_",
            suffix=".npz",
            delete=False,
        ) as handle:
            temp = Path(handle.name)
        np.savez_compressed(
            temp,
            x_edges=x_edges,
            y_edges=y_edges,
            x_centers=x_centers,
            y_centers=y_centers,
            drift=drift_grid,
            diffusion=diffusion_grid,
            occupancy=occupancy,
            reliable_mask=reliable,
            diffusion_eigenvalues=eigenvalues,
            diffusion_determinant=determinant,
            empirical_mass=empirical_mass,
            empirical_density=empirical_density,
        )
        os.replace(temp, output_path)
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved phase-space coefficients: {output_path}", flush=True)
    print(f"Saved phase-space diagnostics: {metadata_path}", flush=True)
    print(json.dumps(diagnostics, indent=2), flush=True)
    return output_path
