"""Stationary density estimation, fitted diffusion laws, full-period aggregation and spatial stationary diagnostics."""

from __future__ import annotations

# --- Stationary ---
# Stationary Ito-SDE density fitting without any plotting side effects.

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit


def _shifted_log_variance(x, *, x0, x1, c1, c2, c3, c4):
    """One shared fitted log(b²) law for BOTH plots and stationary density.

    This is the shifted-coordinate model currently used in the project plots.
    It is NOT the unshifted |x| law from the analytical manuscript.
    """
    x = np.asarray(x, dtype=np.float64)
    left_level = c2 * np.sqrt(abs(x1 - x0)) + c3
    return np.where(
        x < x1,
        c1 * np.abs(x - x1) + left_level,
        np.where(x < x0,
                 c2 * np.sqrt(np.abs(x - x0)) + c3,
                 c4 * np.sqrt(np.abs(x - x0)) + c3),
    )


def _fit_log_diffusion(quantiles, log_diffusion, variable):
    """Fit the historical three-branch model for ``log(b^2(x))``.

    The regression uses the same shifted-coordinate piecewise law as
    ``Functional.func_estimation.model_logb2``:

        x < x1:         c1*abs(x - x1) + dL
        x1 <= x < x0:   c2*sqrt(abs(x - x0)) + c3
        x >= x0:        c4*sqrt(abs(x - x0)) + c3

    where ``dL = c2*sqrt(abs(x1 - x0)) + c3`` ensures continuity at ``x1``.
    The central and right branches are continuous at ``x0`` because both are
    equal to ``c3`` there.
    """
    x = np.asarray(quantiles, dtype=np.float64)
    y = np.asarray(log_diffusion, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 8:
        raise ValueError("Expected equally sized quantile and log-diffusion vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Quantile and log-diffusion inputs must be finite")
    if variable not in ("sensible", "latent"):
        raise ValueError("variable must be sensible or latent")

    x0 = float(x[np.argmin(y)])
    x1 = -50.0 if variable == "sensible" else -40.0
    if not x1 < x0:
        raise ValueError(
            f"Cannot join three diffusion regions: x1={x1}, x0={x0}; "
            "check the estimated diffusion minimum"
        )

    center = (x1 <= x) & (x < x0)
    left = ((x < x1) if variable == "sensible" else
            ((x < x1) & ((x < -250.0) | (x > -45.0))))
    right = x >= x0
    if min(int(center.sum()), int(left.sum()), int(right.sum())) < 2:
        raise ValueError("Insufficient historical quantile data in a diffusion fitting region")

    # Center branch in shifted coordinates relative to x0.
    basis_center = np.column_stack((np.sqrt(np.abs(x[center] - x0)),
                                    np.ones(int(center.sum()))))
    c2, c3 = np.linalg.lstsq(basis_center, y[center], rcond=None)[0]

    # Left branch in shifted coordinates relative to x1, anchored at x1.
    center_at_x1 = c2 * np.sqrt(abs(x1 - x0)) + c3
    left_basis = np.abs(x[left] - x1)
    if float(left_basis @ left_basis) <= 0.0:
        raise ValueError("Degenerate left diffusion regression region")
    c1 = float((left_basis @ (y[left] - center_at_x1)) /
               (left_basis @ left_basis))

    # Right branch in shifted coordinates relative to x0, anchored at x0.
    right_basis = np.sqrt(np.abs(x[right] - x0))
    if float(right_basis @ right_basis) <= 0.0:
        raise ValueError("Degenerate right diffusion regression region")
    c4 = float((right_basis @ (y[right] - c3)) /
               (right_basis @ right_basis))

    prediction = _shifted_log_variance(
        x, x0=x0, x1=x1, c1=c1, c2=c2, c3=c3, c4=c4)
    left_at_join = c1 * abs(x1 - x1) + center_at_x1
    center_at_left_join = c2 * np.sqrt(abs(x1 - x0)) + c3
    center_at_right_join = c2 * np.sqrt(abs(x0 - x0)) + c3
    right_at_join = c4 * np.sqrt(abs(x0 - x0)) + c3
    if not (np.isclose(left_at_join, center_at_left_join, atol=1e-10, rtol=1e-10)
            and np.isclose(center_at_right_join, right_at_join,
                           atol=1e-10, rtol=1e-10)):
        raise AssertionError("Piecewise diffusion branches are discontinuous")

    return {
        "x0": x0,
        "x1": x1,
        "c1": float(c1),
        "c2": float(c2),
        "c3": float(c3),
        "c4": float(c4),
        "rmse_log_b": float(np.sqrt(np.mean((prediction - y) ** 2))),
    }


def _stationary_integration_constants(params):
    """Build integration constants for either sign of the junction ``x0``.

    Parameters
    ----------
    params : sequence
        Parameters ``[x0, x1, drift, xmin, xmax, c1, c2, c3, c4, z]``.

    Returns
    -------
    list of float
        Four continuity-preserving integration constants.

    Notes
    -----
    The historical ``count_constants`` implementation assumes ``x0 < 0``.
    The positive-junction branch follows the corresponding formula in the
    stationary-density derivation and avoids evaluating ``Psi_minus(x0)``
    at a positive ``x0``.
    """
    from Functional.func_estimation import (
        count_constants, Phi, Psi_minus, Psi_plus,
    )

    x0, x1, k, _xmin, _xmax, c1, c2, c3, c4, _z = params
    if x0 < 0:
        return list(count_constants(params)[-4:])

    d_l = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
    d_c = c3
    d_r = c3 + (c2 - c4) * np.sqrt(abs(x0))
    alpha_l = 2.0 * np.exp(-d_l)
    alpha_c = 2.0 * np.exp(-d_c)
    alpha_r = 2.0 * np.exp(-d_r)

    phi_x1 = Phi(x1, c1, k)
    c_left = -alpha_l * phi_x1
    c_center_left = (c_left + alpha_l * phi_x1
                     - alpha_c * Psi_minus(x1, c2, k))
    c_center_right = (c_center_left + alpha_c * Psi_minus(0.0, c2, k)
                      - alpha_c * Psi_plus(0.0, c2, k))
    c_right = (c_center_right + alpha_c * Psi_plus(x0, c2, k)
               - alpha_r * Psi_plus(x0, c4, k))
    return [c_left, c_center_left, c_center_right, c_right]


def _stationary_density(grid, params, variable):
    """Compute stationary density numerically for the *fitted shifted* b².

    The zero-probability-current identity is

        log p(x) = const + integral[2*a(u)*exp(-g(u)) du] - g(x),

    where g = log(b²) is EXACTLY ``_shifted_log_variance``.

    IMPORTANT: the old ``Functional.func_estimation.I/log_b2`` evaluate the
    different UNshifted |x| model from the manuscript. Combining those with
    the current shifted graphical fit produced inconsistent stationary PDFs.
    They must not be used here without first refitting the manuscript model.

    Following the existing visualization convention, only the *integral*
    is continued as a constant beyond ``xmax`` for sensible or below
    ``xmin`` for latent. Diffusion g is never clipped, so PDF tails can decay.
    """
    if variable not in ("sensible", "latent"):
        raise ValueError("variable must be 'sensible' or 'latent'")
    grid = np.asarray(grid, dtype=np.float64)
    if (grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all()
            or not np.all(np.diff(grid) > 0)):
        raise ValueError("grid must be finite, 1D, and strictly increasing")
    x0, x1, k, xmin, xmax, c1, c2, c3, c4, _z = params
    k = np.asarray(k, dtype=np.float64)
    if k.shape != (5,) or not np.isfinite(k).all():
        raise ValueError("Drift must contain five finite ascending polynomial coefficients")
    if not np.isfinite([x0, x1, xmin, xmax, c1, c2, c3, c4]).all():
        raise ValueError("Non-finite stationary fit parameters")
    if not (xmin < xmax and x1 < x0):
        raise ValueError("Invalid fit interval or diffusion junctions")

    # Use a finer grid than the output grid to integrate the fitted Ito drift.
    # Include the junctions and fit boundaries exactly (not only interpolated).
    dense = np.linspace(grid[0], grid[-1], max(30000, 12 * grid.size))
    anchors = np.array((x0, x1, xmin, xmax), dtype=np.float64)
    anchors = anchors[(anchors >= grid[0]) & (anchors <= grid[-1])]
    dense = np.unique(np.concatenate((dense, anchors)))
    log_b2_dense = _shifted_log_variance(
        dense, x0=x0, x1=x1, c1=c1, c2=c2, c3=c3, c4=c4)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        integrand = (2.0 * np.polynomial.polynomial.polyval(dense, k)
                     * np.exp(-log_b2_dense))
    if not np.all(np.isfinite(integrand)):
        raise ValueError(
            f"{variable}: non-finite 2*a/b² on the integration interval; "
            "inspect fitted diffusion tails and polynomial drift"
        )
    integral = cumulative_trapezoid(integrand, dense, initial=0.0)
    if not np.all(np.isfinite(integral)):
        raise ValueError(f"{variable}: stationary drift integral is not finite")

    integral_x = (np.minimum(grid, xmax) if variable == "sensible"
                  else np.maximum(grid, xmin))
    log_values = (np.interp(integral_x, dense, integral)
                  - _shifted_log_variance(
                      grid, x0=x0, x1=x1, c1=c1, c2=c2, c3=c3, c4=c4))
    if not np.all(np.isfinite(log_values)):
        raise ValueError(f"{variable}: non-finite stationary log density")
    scaled = np.exp(log_values - np.max(log_values))
    normalization = float(np.trapezoid(scaled, grid))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError(f"{variable}: invalid stationary normalization")
    density = scaled / normalization
    if not np.isclose(np.trapezoid(density, grid), 1.0, atol=1e-8, rtol=1e-8):
        raise ValueError(f"{variable}: stationary density normalization failed")
    return density


def _window_probability_mass(grid, density, lower, upper):
    """Integrate density within the historical plot window.

    Parameters
    ----------
    grid, density : numpy.ndarray
        Increasing evaluation grid and normalized density.
    lower, upper : float
        Historical plotting boundaries.

    Returns
    -------
    float
        Probability mass in the selected window.
    """
    left = max(float(lower), float(grid[0]))
    right = min(float(upper), float(grid[-1]))
    if left >= right:
        return 0.0
    interior = (grid > left) & (grid < right)
    points = np.concatenate(([left], grid[interior], [right]))
    values = np.interp(points, grid, density)
    return float(np.trapezoid(values, points))




def _load_historical_grouped(root, names, first, last, b_type):
    """Load precomputed historical quantile fits when all files exist.

    Parameters
    ----------
    root : pathlib.Path
        Data root.
    names : tuple of str
        Names of the two heat fluxes.
    first, last : int
        One-based endpoint coefficient labels.
    b_type : str
        Saved historical diffusion regression type.

    Returns
    -------
    dict or None
        Maps each variable name to (quantiles, drift, log diffusion).
        Returns None when any required historical file is absent.
    """
    folder = root / "Functional" / f"{names[0]}-{names[1]}"
    keys = ("quantiles1", "quantiles2", "a_grouped",
            f"b_grouped_log_{b_type}")
    paths = [folder / f"{first}-{last}_{key}.npy" for key in keys]
    if not all(path.is_file() for path in paths):
        return None
    q0, q1, a, log_b = [np.load(path, allow_pickle=False) for path in paths]
    if a.shape[0] != 2 or log_b.shape[0] != 2:
        raise ValueError(f"Invalid saved historical grouped coefficients: {paths[2]}")
    return {names[0]: (np.asarray(q0), np.asarray(a[0]), np.asarray(log_b[0])),
            names[1]: (np.asarray(q1), np.asarray(a[1]), np.asarray(log_b[1]))}


def _mean_in_spatial_blocks(values, valid, block_size):
    """Compute ocean-only spatial block means without reallocating all years.

    Parameters
    ----------
    values : numpy.ndarray
        Array of shape (chunk_days, H, W).
    valid : numpy.ndarray
        Boolean per-element validity mask of the same shape.
    block_size : int
        Width of square averaging blocks; 1 disables averaging.

    Returns
    -------
    numpy.ndarray
        Array of shape (chunk_days, H//block_size, W//block_size).
    """
    if block_size == 1:
        return np.where(valid, values, np.nan)
    t, h, w = values.shape
    height, width = h // block_size, w // block_size
    if not height or not width:
        raise ValueError("spatial_block_size exceeds the input grid size")
    hcut, wcut = height * block_size, width * block_size
    v = np.asarray(values[:, :hcut, :wcut], dtype=np.float64).reshape(
        t, height, block_size, width, block_size)
    good = valid[:, :hcut, :wcut].reshape(t, height, block_size, width, block_size)
    sums = np.where(good, v, 0.0).sum(axis=(2, 4))
    counts = good.sum(axis=(2, 4))
    return np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0)


def _assembled_sources(root, names, start_year, end_year, method):
    """Locate memmapped full-period observations and A/B or BBT coefficients.

    Parameters
    ----------
    root : pathlib.Path
        Data root.
    names : tuple of str
        Flux names in the coefficient matrix channel order.
    start_year, end_year : int
        Period labels used in filenames.
    method : {'np', 'sp'}
        Coefficient estimation method.

    Returns
    -------
    tuple
        Two time-first data arrays, A and B/BBT arrays, source information,
        and their matched transition count.
    """
    from Data_processing.io import field_candidates, load_field

    period = f"{start_year}-{end_year}"
    parent = root / "Coefficients" / period / f"{names[0]}_{names[1]}"
    a_path = parent / f"A_{period}_{method}.npy"
    bbt_path = parent / f"BBT_{period}_{method}.npy"
    b_path = parent / f"B_{period}_{method}.npy"
    if not a_path.is_file() or not (bbt_path.is_file() or b_path.is_file()):
        raise FileNotFoundError(
            f"Full-period coefficients not found: {a_path} and "
            f"either {bbt_path} or {b_path}. Run bridge_coefficients first."
        )
    # Prefer the precomputed covariance only if it is not older than B.
    use_bbt = bbt_path.is_file() and (
        not b_path.is_file() or bbt_path.stat().st_mtime_ns >= b_path.stat().st_mtime_ns
    )
    if bbt_path.is_file() and not use_bbt:
        print(f"WARNING: {bbt_path.name} is older than B; "
              "recomputing marginal variances from B channels", flush=True)
    diffusion_path = bbt_path if use_bbt else b_path
    diffusion_kind = "BBT" if use_bbt else "B"
    field_paths = []
    for name in names:
        found = next((p for p in field_candidates(root, name, start_year, end_year)
                      if p.is_file()), None)
        if found is None:
            raise FileNotFoundError(
                f"Cannot find full-period {name} data ({period}); "
                f"checked: {field_candidates(root, name, start_year, end_year)}"
            )
        field_paths.append(found)
    fields = tuple(load_field(root, name, start_year, end_year) for name in names)
    a = np.load(a_path, mmap_mode="r", allow_pickle=False)
    d = np.load(diffusion_path, mmap_mode="r", allow_pickle=False)
    if (a.ndim != 4 or d.ndim != 4 or a.shape[1] != 2 or d.shape[1] != 4
            or a.shape[2:] != d.shape[2:] or a.shape[0] != d.shape[0]):
        raise ValueError(f"Expected A=(T,2,H,W), B/BBT=(T,4,H,W); got {a.shape}, {d.shape}")
    if (fields[0].shape[1:] != a.shape[2:] or fields[1].shape[1:] != a.shape[2:]):
        raise ValueError("Full-period data and coefficient spatial grids differ")
    if len(fields[0]) < len(a) + 1 or len(fields[1]) < len(a) + 1:
        raise ValueError(
            f"Time alignment failed: A has {len(a)} endpoint transitions, "
            f"but observations have lengths {len(fields[0])}, {len(fields[1])}; "
            "the first coefficient must correspond to the SECOND observation"
        )
    # Unlike the old grouped script, x is taken from the endpoint date of A_t.
    # We intentionally use observations[1:T+1] with coefficients[0:T].
    details = {
        "method": method,
        "diffusion_kind": diffusion_kind,
        "source_paths": [str(x) for x in (*field_paths, a_path, diffusion_path)],
        "source_stats": [[int(p.stat().st_size), int(p.stat().st_mtime_ns)]
                         for p in (*field_paths, a_path, diffusion_path)],
        "n_transitions": int(len(a)),
    }
    # Use the same label convention as the coefficients subsystem.
    from Data_processing.io import start_label
    details["first_label"] = start_label(start_year) + 1
    details["last_label"] = start_label(start_year) + len(a)
    if len(fields[0]) > len(a) + 1 or len(fields[1]) > len(a) + 1:
        print(f"NOTE: using {len(a) + 1} aligned observation dates; "
              "additional observations are outside the assembled coefficient range", flush=True)
    return fields, a, d, details


def _stream_stationary_chunks(fields, a, diffusion, details, mask,
                               chunk_days, block_size):
    """Yield aligned per-flux (state, drift, diffusion-variance) blocks.

    Parameters
    ----------
    fields : tuple of numpy.ndarray
        Time-first sensible and latent observations.
    a : numpy.ndarray
        Drift array with shape (T,2,H,W).
    diffusion : numpy.ndarray
        Diffusion B or product BBT with shape (T,4,H,W).
    details : dict
        Source metadata including ``diffusion_kind``.
    mask : numpy.ndarray
        Boolean water mask (H,W).
    chunk_days : int
        Number of transitions in a disk-read chunk.
    block_size : int
        Optional square spatial averaging width.

    Yields
    ------
    tuple of tuple
        Each flux yields (state, drift, diffusion variance), all flattened.
    """
    n = len(a)
    for start in range(0, n, chunk_days):
        stop = min(n, start + chunk_days)
        drift = np.asarray(a[start:stop], dtype=np.float64)
        diff = np.asarray(diffusion[start:stop], dtype=np.float64)
        if details["diffusion_kind"] == "BBT":
            variances = (diff[:, 0], diff[:, 1])
        else:
            # Stored B-channel order is [B00,B11,B01,B10].
            variances = (diff[:, 0] ** 2 + diff[:, 2] ** 2,
                         diff[:, 1] ** 2 + diff[:, 3] ** 2)
        results = []
        for channel, field in enumerate(fields):
            observation = np.asarray(field[start + 1:stop + 1], dtype=np.float64)
            var = variances[channel]
            mean_a = drift[:, channel]
            valid = (np.isfinite(observation) & np.isfinite(mean_a)
                     & np.isfinite(var) & (var > 0) & mask[None, :, :])
            state = _mean_in_spatial_blocks(observation - mean_a, valid, block_size)
            a_mean = _mean_in_spatial_blocks(mean_a, valid, block_size)
            var_mean = _mean_in_spatial_blocks(var, valid, block_size)
            results.append((state.ravel(), a_mean.ravel(), var_mean.ravel()))
        yield tuple(results)


def _group_full_period(root, names, start_year, end_year, method,
                       *, bins, trim_bins, chunk_days, block_size,
                       sample_size, refresh_grouped):
    """Create grouped stationary inputs from assembled multiyear A/B arrays.

    Parameters
    ----------
    root : pathlib.Path
        Root containing data and assembled coefficients.
    names : tuple of str
        Two flux names.
    start_year, end_year : int
        Full-period labels.
    method : {'np', 'sp'}
        Coefficient estimator to read.
    bins : int
        Number of approximate equal-mass quantile groups.
    trim_bins : int
        Number of sparse tail groups to remove at each side.
    chunk_days : int
        Number of days read in one pass.
    block_size : int
        Spatial averaging block width.
    sample_size : int
        Target number of spatially averaged observations sampled per flux.
    refresh_grouped : bool
        Whether cached grouped parameters should be rebuilt.

    Returns
    -------
    tuple
        Mapping of flux names to fitted quantile vectors, and cache metadata.

    Notes
    -----
    Quantile edges are estimated from a deterministic uniform sample within
    time chunks. Drift and diffusion group means are computed in a second
    pass from ALL finite spatial-block observations, without sampling.
    """
    from Data_processing.io import read_ocean_mask
    if method not in ("np", "sp"):
        raise ValueError("method must be 'np' or 'sp'")
    if chunk_days < 1 or block_size < 1 or bins <= 2 * trim_bins + 8:
        raise ValueError("Invalid chunk_days, block_size, or quantile bin count")
    if sample_size < 100:
        raise ValueError("sample_size must be at least 100")
    fields, a, diffusion, details = _assembled_sources(root, names, start_year, end_year, method)
    h, w = a.shape[2:]
    try:
        mask = read_ocean_mask(root, (h, w))
    except FileNotFoundError:
        mask = np.ones((h, w), dtype=bool)
    count = int(details["n_transitions"])
    cache_folder = root / "Functional" / f"{names[0]}-{names[1]}"
    cache_folder.mkdir(parents=True, exist_ok=True)
    cache = (cache_folder / f"grouped_{start_year}-{end_year}_{method}_"
             f"{details['first_label']}-{details['last_label']}.npz")
    meta_path = cache.with_suffix(".json")
    config = {
        "bins": bins, "trim_bins": trim_bins, "chunk_days": chunk_days,
        "block_size": block_size, "sample_size": sample_size,
        "sources": details,
    }
    if not refresh_grouped and cache.exists() and meta_path.exists():
        saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved_meta.get("config") == config:
            with np.load(cache, allow_pickle=False) as grouped:
                fit_inputs = {
                    name: tuple(np.array(grouped[f"{key}_{name}"], copy=True)
                                for key in ("quantiles", "drift", "log_variance"))
                    for name in names
                }
            print(f"Reusing grouped stationary inputs: {cache}", flush=True)
            return fit_inputs, {"cache": str(cache), "sources": details,
                                "grouping": config}

    samples = [[], []]
    n_chunks = (count + chunk_days - 1) // chunk_days
    sample_per_chunk = max(1, (sample_size + n_chunks - 1) // n_chunks)
    rng = np.random.default_rng(1979)
    print(f"Preparing grouped stationary parameters from {count} transitions "
          f"({method}, {details['diffusion_kind']}); pass 1/2", flush=True)
    for pairs in _stream_stationary_chunks(fields, a, diffusion, details,
                                           mask, chunk_days, block_size):
        for i, (state, drift, variance) in enumerate(pairs):
            clean = state[np.isfinite(state)]
            if clean.size:
                take = min(sample_per_chunk, clean.size)
                indices = rng.choice(clean.size, take, replace=False)
                samples[i].append(clean[indices])

    edges = []
    for name, selected in zip(names, samples):
        if not selected:
            raise ValueError(f"No finite stationarity data for {name}")
        values = np.concatenate(selected)
        boundaries = np.quantile(values, np.linspace(0, 1, bins + 1))
        boundaries = np.unique(boundaries)
        if len(boundaries) < 2 * trim_bins + 10:
            raise ValueError(f"Too few distinct quantile values for {name}")
        edges.append(boundaries)
    del samples

    sums = []
    for limits in edges:
        length = len(limits) - 1
        sums.append({key: np.zeros(length, dtype=np.float64)
                     for key in ("counts", "drift", "variance")})

    print("Grouping all valid transitions; pass 2/2", flush=True)
    for pairs in _stream_stationary_chunks(fields, a, diffusion, details,
                                           mask, chunk_days, block_size):
        for i, (state, drift, variance) in enumerate(pairs):
            good = np.isfinite(state) & np.isfinite(drift) & np.isfinite(variance) & (variance > 0)
            if not good.any():
                continue
            x, aa, vv = state[good], drift[good], variance[good]
            group_index = np.searchsorted(edges[i], x, side="right") - 1
            group_index = np.clip(group_index, 0, len(edges[i]) - 2)
            n = len(edges[i]) - 1
            sums[i]["counts"] += np.bincount(group_index, minlength=n)
            sums[i]["drift"] += np.bincount(group_index, weights=aa, minlength=n)
            sums[i]["variance"] += np.bincount(group_index, weights=vv, minlength=n)

    fit_inputs = {}
    store = {}
    for i, name in enumerate(names):
        limits = edges[i]
        mids = 0.5 * (limits[:-1] + limits[1:])
        keep = np.arange(len(mids))
        chosen = (keep >= trim_bins) & (keep < len(mids) - trim_bins)
        chosen &= sums[i]["counts"] > 0
        xx = mids[chosen]
        drift = np.divide(sums[i]["drift"][chosen], sums[i]["counts"][chosen])
        variance = np.divide(sums[i]["variance"][chosen], sums[i]["counts"][chosen])
        if xx.size < 8 or not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            raise ValueError(f"Invalid grouped drift/diffusion for {name}; check input B")
        fit_inputs[name] = (xx, drift, np.log(variance))
        for key, data in zip(("quantiles", "drift", "log_variance"), fit_inputs[name]):
            store[f"{key}_{name}"] = data
        print(f"{name}: {len(xx)} retained groups, "
              f"{int(sums[i]['counts'].sum()):,} valid samples", flush=True)

    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_folder, prefix=".stationary_",
                                         suffix=".npz", delete=False) as handle:
            temp = Path(handle.name)
        np.savez_compressed(temp, **store)
        os.replace(temp, cache)
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    meta_path.write_text(json.dumps({"config": config}, indent=2), encoding="utf-8")
    print(f"Saved grouped stationary inputs: {cache}", flush=True)
    return fit_inputs, {"cache": str(cache), "sources": details,
                        "grouping": config}

def estimate_stationary_characteristics(files_path_prefix, start_year, end_year,
                                        *, names=("sensible", "latent"),
                                        b_type="simple", grid_points=2500,
                                        method="sp", quantile_bins=260,
                                        trim_bins=5, chunk_days=16,
                                        spatial_block_size=5, sample_size=400000,
                                        refresh_grouped=False):
    """Fit stationary distributions from historical or assembled coefficients.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root containing the ocean fields and coefficients.
    start_year, end_year : int
        Requested period labels, including the full 1979--2026 period.
    names : tuple of str
        Ordered names of the two fluxes, matching coefficient channels.
    b_type : str
        Historical grouped diffusion fit label, normally ``simple``.
    grid_points : int
        Number of stationary density evaluation points.
    method : {'np', 'sp'}
        Method used for assembled, time-first coefficient arrays.
    quantile_bins : int
        Number of groups for multiyear automatic preprocessing.
    trim_bins : int
        Number of tail bins removed from each side.
    chunk_days : int
        Number of transitions loaded from memmapped arrays at once.
    spatial_block_size : int
        Size of ocean-only square spatial averaging cells; 1 disables it.
    sample_size : int
        Target sampled observations used for estimating quantile edges.
    refresh_grouped : bool
        Rebuild existing grouped cache even when source file statistics match.

    Returns
    -------
    pathlib.Path
        Saved stationary densities in the historical plotting NPZ layout.

    Notes
    -----
    For the full period the first A/B row is for 1979-01-01 -> 1979-01-02.
    Thus full-period input fluxes are indexed from 1, while A/B start at 0.
    BBT is preferred if available; otherwise its marginal variances are
    computed from the four channels of B without an intermediate giant file.
    """
    from Data_processing.io import start_label

    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("Exactly two different flux names are required")
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    if method not in ("np", "sp"):
        raise ValueError("method must be np or sp")
    root = Path(files_path_prefix)
    period = (start_year, end_year)
    historical = {
        (1979, 1989): 3653,
        (1989, 1999): 7305,
        (1999, 2009): 10958,
        (2009, 2019): 14610,
        (2019, 2026): 17106,
    }
    first = start_label(start_year)
    inputs = None
    provenance = {}

    # Retain the previously cached decade workflow unchanged. The full
    # 1979--2026 period instead reads newly assembled NP/SP A/B arrays.
    if period in historical:
        inputs = _load_historical_grouped(root, names, first, historical[period], b_type)
        if inputs is not None:
            provenance = {"input_type": "legacy_grouped", "first_label": first,
                          "last_label": historical[period]}

    if inputs is None:
        if b_type != "simple":
            raise ValueError(
                "Assembled A/B preprocessing supports b_type='simple' only; "
                "the eigen diffusion source requires a separate numerical workflow"
            )
        inputs, provenance = _group_full_period(
            root, names, start_year, end_year, method,
            bins=quantile_bins, trim_bins=trim_bins, chunk_days=chunk_days,
            block_size=spatial_block_size, sample_size=sample_size,
            refresh_grouped=refresh_grouped,
        )
        provenance["input_type"] = "assembled_coefficients"
        provenance["state_construction"] = (
            "endpoint_observation_minus_local_drift_A_then_spatial_block_mean"
        )

    grid = np.linspace(-2000, 1500, grid_points)
    output = {"grid": grid}
    meta = {"pair": list(names), "period": f"{start_year}-{end_year}",
            "method": method, "provenance": provenance, "fits": {}}

    print(f"Stationary fit: {start_year}-{end_year}, "
          f"variables: {', '.join(names)}, method={method}", flush=True)
    for name in names:
        print(f"Fitting stationary parameters for {name}...", flush=True)
        x, drift, log_variance = inputs[name]
        params = _fit_log_diffusion(x, log_variance, name)
        coeffs = list(np.polynomial.polynomial.polyfit(x, drift, 4))
        drift_prediction = np.polynomial.polynomial.polyval(x, coeffs)
        drift_rmse = float(np.sqrt(np.mean((drift_prediction - drift) ** 2)))
        fitted = [params["x0"], params["x1"], coeffs,
                  float(np.min(x)), float(np.max(x)),
                  *[params[key] for key in ("c1", "c2", "c3", "c4")], 1]
        density = _stationary_density(grid, fitted, name)
        plot_bounds = ((-500.0, 500.0) if name == "sensible"
                       else (-800.0, 300.0) if name == "latent" else None)
        plot_mass = (_window_probability_mass(grid, density, *plot_bounds)
                     if plot_bounds is not None else None)
        mean = float(np.trapezoid(grid * density, grid))
        variance = float(np.trapezoid((grid - mean) ** 2 * density, grid))
        output[f"pdf_{name}"] = density
        meta["fits"][name] = {
            **params,
            "drift_polynomial_ascending": coeffs,
            "drift_rmse": drift_rmse,
            "drift_rmse_definition": "RMSE of quartic A(x) over retained grouped nodes",
            "mean": mean,
            "mean_definition": (
                "Mean under the normalized stationary PDF in the fitted x-coordinate; "
                "NOT the mean of ERA5 observations or a weighted decadal average"
            ),
            "variance": variance,
            "standard_deviation": float(np.sqrt(max(variance, 0.0))),
            "diffusion_model": "shifted_x0_x1_piecewise_log_variance",
            "stationary_integration_method": "numeric_cumulative_trapezoid_of_2a_over_b2",
            "tail_continuation": "drift integral only; log b2 uses physical x",
            "density_integral": float(np.trapezoid(density, grid)),
            "reliable_regression_interval": [float(fitted[3]), float(fitted[4])],
            "drift_integral_continuation": (
                "right at xmax" if name == "sensible" else "left at xmin"
            ),
            "density_mode": float(grid[int(np.argmax(density))]),
            "plot_window": list(plot_bounds) if plot_bounds is not None else None,
            "probability_mass_in_plot_window": plot_mass,
        }
        print(f"Stationary parameters for {name} ({start_year}-{end_year}):", flush=True)
        print(json.dumps(meta["fits"][name], indent=2, ensure_ascii=False), flush=True)
        if plot_mass is not None and plot_mass < 0.95:
            print(f"WARNING: {name}: only {plot_mass:.2%} of probability mass "
                  f"lies inside the plotting interval {plot_bounds}", flush=True)

    target = root / "Functional" / "Stationary"
    target.mkdir(parents=True, exist_ok=True)
    result = target / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    np.savez_compressed(result, **output)
    metadata_path = result.with_suffix(".json")
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Stationary densities saved to: {result}", flush=True)
    print(f"Stationary parameters saved to: {metadata_path}", flush=True)
    return result


# --- Stationary Diagnostics ---
# Stationary-reference spatial diagnostics from existing PDFs and ERA5.
#
# No model is refitted here. Spatial deviations are diagnostic comparisons
# because the model coordinate may differ from the raw ERA5 flux coordinate.

from datetime import date


def field_candidates(*args, **kwargs):
    """Resolve field candidates lazily for the spatial diagnostic workflow."""
    from Data_processing.io import field_candidates as resolve_candidates
    return resolve_candidates(*args, **kwargs)


def load_field(*args, **kwargs):
    """Load a diagnostic field only when the spatial calculation requests it."""
    from Data_processing.io import load_field as read_field
    return read_field(*args, **kwargs)


def read_ocean_mask(*args, **kwargs):
    """Load the spatial ocean mask only when diagnostics need it."""
    from Data_processing.io import read_ocean_mask as read_mask
    return read_mask(*args, **kwargs)

DEFAULT_PAIR = ("sensible", "latent")
DEFAULT_REGIONS_PERIOD = (1979, 1989)
DEFAULT_REFERENCE_PERIOD = (1979, 2026)
DEFAULT_DATE = "1979-01-01"
DEFAULT_HYBRID_REFERENCE = (1979, 2026)
DEFAULT_LON_RANGE = (-90.0, 0.0)
DEFAULT_LAT_RANGE = (80.0, 0.0)  # array row 0 corresponds to northern boundary


def _signature(path):
    """Build a signature for validating cached inputs or results."""
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def _read_json(path):
    """Read JSON for the stationary workflow."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path, data):
    """Save JSON for the stationary workflow."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _source(root, name, period):
    """Compute source for stationary from ``root``, ``name``, ``period``."""
    candidates = field_candidates(root, name, *period)
    file = next((path for path in candidates if path.is_file()), None)
    if file is None:
        raise FileNotFoundError(f"ERA5 source for {name} {period} not found; checked: {candidates}")
    return file


def stationary_density(root, variable, period, *, names=DEFAULT_PAIR,
                       density_source="auto", hybrid_reference=DEFAULT_HYBRID_REFERENCE):
    """Read the *existing* density; never recompute extrapolated tails or GPD.

    density_source='auto': prefer a named cached hybrid; else classical FP.
    density_source='hybrid'/'fp': require the requested type explicitly.
    """
    root = Path(root)
    first, last = map(int, period)
    if density_source not in ("auto", "hybrid", "fp"):
        raise ValueError("density_source must be 'auto', 'hybrid' or 'fp'")
    base = root / "Functional" / "Stationary"
    hybrid = (base / "Hybrid" /
              f"{variable}_{first}-{last}_gpd_reference_"
              f"{hybrid_reference[0]}-{hybrid_reference[1]}.npz")
    fp = base / f"{names[0]}-{names[1]}_{first}-{last}.npz"
    if density_source == "hybrid":
        if not hybrid.is_file():
            raise FileNotFoundError(f"Hybrid PDF not available: {hybrid}")
        path, kind, key = hybrid, "hybrid", "pdf"
    elif density_source == "fp":
        if not fp.is_file():
            raise FileNotFoundError(f"Stationary FP PDF not available: {fp}")
        path, kind, key = fp, "fp", f"pdf_{variable}"
    elif hybrid.is_file():
        path, kind, key = hybrid, "hybrid", "pdf"
    elif fp.is_file():
        path, kind, key = fp, "fp", f"pdf_{variable}"
    else:
        raise FileNotFoundError(f"Neither hybrid ({hybrid}) nor FP ({fp}) PDF exists")
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"Missing {key} in {path}: available={data.files}")
        grid = np.asarray(data["grid"], dtype=np.float64)
        pdf = np.asarray(data[key], dtype=np.float64)
    if grid.ndim != 1 or pdf.shape != grid.shape or len(grid) < 4:
        raise ValueError(f"Invalid stationary grid/PDF in {path}")
    if not np.all(np.diff(grid) > 0) or not np.isfinite(grid).all():
        raise ValueError(f"Stationary coordinate grid is not increasing: {path}")
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    if np.min(pdf) < -1e-12:
        raise ValueError(f"Negative stationary PDF values in {path}")
    pdf = np.maximum(pdf, 0.0)
    area = float(np.trapezoid(pdf, grid))
    if not 0 < area < np.inf:
        raise ValueError(f"Non-normalizable PDF in {path}")
    pdf /= area
    mean = float(np.trapezoid(grid * pdf, grid))
    std = float(np.sqrt(max(0.0, np.trapezoid((grid - mean)**2 * pdf, grid))))
    return {"grid": grid, "pdf": pdf, "mean": mean, "std": std,
            "kind": kind, "path": path, "signature": _signature(path)}


def equal_probability_regions(grid, pdf, n_regions=5):
    """Interpolate quantiles of the normalized continuous density (not bins)."""
    grid = np.asarray(grid, dtype=np.float64)
    pdf = np.asarray(pdf, dtype=np.float64)
    if grid.ndim != 1 or grid.shape != pdf.shape or n_regions < 2:
        raise ValueError("Invalid grid/PDF or number of regions")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("Grid must be strictly increasing")
    cdf = np.r_[0., cumulative_trapezoid(np.maximum(pdf, 0), grid)]
    if cdf[-1] <= 0 or not np.isfinite(cdf[-1]):
        raise ValueError("PDF has zero or non-finite mass")
    cdf /= cdf[-1]
    internal = np.interp(np.arange(1, n_regions) / n_regions, cdf, grid)
    if np.any(np.diff(internal) <= 0):
        raise ValueError("Quantile boundaries are not strictly increasing")
    masses = np.diff(np.interp(np.r_[grid[0], internal, grid[-1]], grid, cdf))
    return internal, masses


def classify_map(values, thresholds, water_mask):
    """Classes 1...5, land/non-finite = 0; q_{k-1}<x<=q_k."""
    values = np.asarray(values, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    water_mask = np.asarray(water_mask, dtype=bool)
    if thresholds.ndim != 1 or np.any(np.diff(thresholds) <= 0):
        raise ValueError("Thresholds must be strictly increasing")
    if values.shape != water_mask.shape:
        raise ValueError(f"Water mask {water_mask.shape} differs from field {values.shape}")
    result = np.zeros(values.shape, dtype=np.uint8)
    good = water_mask & np.isfinite(values)
    result[good] = (1 + np.searchsorted(thresholds, values[good], side="left")).astype(np.uint8)
    return result


def _spatial_statistics(field, ocean, *, chunk_days, thresholds=None,
                        compute_frequencies=False):
    """Compute true per-pixel temporal mean using bounded time slices."""
    if chunk_days < 1:
        raise ValueError("chunk_days must be positive")
    shape = field.shape[-2:]
    if ocean.shape != shape:
        raise ValueError(f"Ocean mask {ocean.shape} != ERA5 grid {shape}")
    sums = np.zeros(shape, np.float64)
    counts = np.zeros(shape, np.int64)
    hit = np.zeros((5, *shape), np.int64) if compute_frequencies else None
    for left in range(0, len(field), chunk_days):
        right = min(len(field), left + chunk_days)
        batch = np.asarray(field[left:right], dtype=np.float64)
        valid = np.isfinite(batch) & ocean[None, ...]
        sums += np.where(valid, batch, 0).sum(axis=0)
        counts += np.sum(valid, axis=0)
        if hit is not None:
            cat = np.searchsorted(thresholds, batch, side="left")
            for k in range(5):
                hit[k] += ((cat == k) & valid).sum(axis=0)
        if left == 0 or right == len(field) or right // chunk_days % 60 == 0:
            print(f"[stationary spatial] ERA5 spatial statistics {right}/{len(field)} days", flush=True)
    mean = np.full(shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=mean, where=counts > 0)
    if hit is None:
        return mean, counts, None
    frequencies = np.full((5, *shape), np.nan, dtype=np.float64)
    np.divide(hit, counts[None], out=frequencies, where=counts[None] > 0)
    return mean, counts, frequencies


def _cache_file(root, name, period, region_period, date_str, pdf_kind, frequencies, *, legacy=False):
    """Compute cache file for stationary from ``root``, ``name``, ``period``, ``region_period``, ``date_str``."""
    root = Path(root)
    fragment = "_freq" if frequencies else ""
    base = root / "Functional"
    base = base / "Article" if legacy else base / "Stationary" / "Spatial"
    return (base / f"{period[0]}-{period[1]}" /
            f"{name}_regions_{region_period[0]}-{region_period[1]}_"
            f"{date_str}_{pdf_kind}{fragment}.npz")


def prepare_stationary_diagnostics(files_path_prefix, names=DEFAULT_PAIR,
                                start_year=1979, end_year=2026, *,
                                regions_period=DEFAULT_REGIONS_PERIOD,
                                map_date=DEFAULT_DATE, density_source="auto",
                                hybrid_reference=DEFAULT_HYBRID_REFERENCE,
                                chunk_days=32, compute_frequencies=False,
                                overwrite=False, cache_only=False):
    """Prepare reference-relative anomalies, PDF quintiles and daily classes.

    The time mean and model reference use the same period. Quintiles use
    regions_period; the selected daily map uses exactly these thresholds.
    cache_only=True validates and loads existing cache paths without reading
    the large ERA5 arrays; otherwise an absent or stale cache is calculated.
    Legacy caches are accepted only with matching metadata.
    """
    if cache_only and overwrite:
        raise ValueError("cache_only and overwrite are mutually exclusive")
    root = Path(files_path_prefix)
    names = tuple(dict.fromkeys(names))
    period = (int(start_year), int(end_year))
    regions_period = tuple(int(x) for x in regions_period)
    map_day = date.fromisoformat(str(map_date))
    if not names:
        raise ValueError("At least one variable required")
    if not (period[0] <= map_day.year <= period[1]):
        raise ValueError(f"map_date {map_day} not in data period {period}")
    outputs = {}
    for name in names:
        full = stationary_density(root, name, period, names=DEFAULT_PAIR,
                                  density_source=density_source,
                                  hybrid_reference=hybrid_reference)
        regions = stationary_density(root, name, regions_period,
                                     names=DEFAULT_PAIR,
                                     density_source=density_source,
                                     hybrid_reference=hybrid_reference)
        thresholds, masses = equal_probability_regions(regions["grid"], regions["pdf"])
        era5_file = _source(root, name, period)
        offset = (map_day - date(period[0], 1, 1)).days
        cache = _cache_file(root, name, period, regions_period, map_day.isoformat(),
                            f"{full['kind']}-{regions['kind']}", compute_frequencies)
        metadata = {
            "version": 2, "variable": name, "period": list(period),
            "regions_period": list(regions_period), "map_date": map_day.isoformat(),
            "density_source_requested": density_source,
            "isoline_pdf": {"source": full["kind"], "signature": full["signature"]},
            "regions_pdf": {"source": regions["kind"], "signature": regions["signature"]},
            "era5_file": _signature(era5_file),
            "chunk_days": int(chunk_days), "compute_frequencies": bool(compute_frequencies),
            "quantile_rule": "q[k] = F^-1(k/5); q[k-1] < X <= q[k]",
            "coordinate_caution": (
                "ERA5 spatial fields are raw heat fluxes. The model's effective "
                "stationary-coordinate definition can differ. These are reference-relative "
                "diagnostics; the hybrid is not the original stationary SDE solution."),
        }
        meta_path = cache.with_suffix(".json")
        legacy_cache = _cache_file(root, name, period, regions_period, map_day.isoformat(),
                                   f"{full['kind']}-{regions['kind']}", compute_frequencies,
                                   legacy=True)
        if not overwrite:
            for candidate in (cache, legacy_cache):
                candidate_meta = candidate.with_suffix(".json")
                if candidate.is_file() and candidate_meta.is_file() and _read_json(candidate_meta) == metadata:
                    print(f"[stationary spatial] Reuse {name}: {candidate}", flush=True)
                    outputs[name] = candidate
                    break
            if name in outputs:
                continue
        if cache_only:
            raise FileNotFoundError(
                f"Matching stationary spatial cache not found: {cache}. "
                "Run main.py with task='stationary_spatial' and identical settings first.")
        field = load_field(root, name, *period)
        if not (0 <= offset < len(field)):
            raise IndexError(f"Date {map_day} not in observed {name} field of {len(field)} days")
        ocean = read_ocean_mask(root, field.shape[-2:])
        mean_map, count_map, frequencies = _spatial_statistics(
            field, ocean, chunk_days=chunk_days, thresholds=thresholds,
            compute_frequencies=compute_frequencies)
        snapshot = np.asarray(field[offset], dtype=np.float64)
        class_map = classify_map(snapshot, thresholds, ocean)
        anomaly = np.where(ocean, mean_map - full["mean"], np.nan)
        standardized = anomaly / max(full["std"], 1e-12)
        output = {
            "ocean_mask": ocean, "mean_map": mean_map, "counts": count_map,
            "anomaly": anomaly, "standardized_anomaly": standardized,
            "class_map": class_map, "snapshot": snapshot,
            "grid": regions["grid"], "pdf": regions["pdf"],
            "thresholds": thresholds, "region_masses": masses,
            "model_mean": np.asarray(full["mean"]),
            "model_std": np.asarray(full["std"]),
        }
        if frequencies is not None:
            output["frequencies"] = frequencies
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **output)
        _save_json(meta_path, metadata)
        outputs[name] = cache
        print(f"[stationary spatial] {name}: source={full['kind']}, mu={full['mean']:.4f}, "
              f"sigma={full['std']:.4f}, q={thresholds.round(3)}, "
              f"quintile masses={masses.round(5)}, output={cache}", flush=True)
    return outputs
