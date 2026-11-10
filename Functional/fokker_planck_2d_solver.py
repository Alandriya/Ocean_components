"""Stationary two-dimensional Fokker-Planck solver on a phase-space grid.

The stationary Fokker-Planck equation is solved through a locally consistent
continuous-time Markov-chain approximation of the Ito diffusion generator.

For the SDE

    dX = A(X) dt + B(X) dW,          D(X) = B(X) B(X).T,

the backward generator is

    L f = A_i d_i f + 0.5 D_ij d_ij f.

At every retained grid cell this module constructs nonnegative jump rates whose
first jump moment equals A and whose second jump moment equals D. The stationary
probability vector then satisfies

    Q.T @ pi = 0,    sum(pi) = 1,

where Q is the Markov generator. This is the discrete adjoint stationary
Fokker-Planck problem.

A wide diagonal stencil is used because a nearest-neighbor 9-point monotone
stencil is not sufficient for strongly correlated anisotropic diffusion.

Boundary convention
-------------------
The computational domain is the largest 4-connected reliable phase-space
component. Small enclosed holes may be filled from nearby reliable coefficients.
Jumps leaving the domain are censored (suppressed). Thus no probability can
leave the discrete domain; this is a conservative no-outflow boundary
approximation.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.spatial import cKDTree


def _source_signature(path):
    """Return file metadata used to identify the phase-grid input."""
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_phase_grid(path):
    """Load and validate a prepared two-dimensional phase-space coefficient grid."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Phase-space coefficient file not found: {path}")

    required = {
        "x_edges", "y_edges", "x_centers", "y_centers",
        "drift", "diffusion", "occupancy", "reliable_mask",
        "empirical_mass",
    }
    with np.load(path, allow_pickle=False) as source:
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"Missing phase-grid arrays: {sorted(missing)}")
        arrays = {key: np.array(source[key], copy=True) for key in required}

    x_edges = arrays["x_edges"]
    y_edges = arrays["y_edges"]
    x_centers = arrays["x_centers"]
    y_centers = arrays["y_centers"]
    drift = arrays["drift"]
    diffusion = arrays["diffusion"]
    occupancy = arrays["occupancy"]
    reliable = arrays["reliable_mask"].astype(bool, copy=False)
    empirical_mass = arrays["empirical_mass"]

    shape = (len(x_centers), len(y_centers))
    if x_edges.shape != (shape[0] + 1,) or y_edges.shape != (shape[1] + 1,):
        raise ValueError("Edge and center arrays are inconsistent")
    if drift.shape != (2, *shape):
        raise ValueError(f"Expected drift shape (2,{shape[0]},{shape[1]})")
    if diffusion.shape != (4, *shape):
        raise ValueError(f"Expected diffusion shape (4,{shape[0]},{shape[1]})")
    if occupancy.shape != shape or reliable.shape != shape or empirical_mass.shape != shape:
        raise ValueError("Phase-grid scalar arrays do not share the center-grid shape")
    if not (np.all(np.diff(x_edges) > 0) and np.all(np.diff(y_edges) > 0)):
        raise ValueError("Phase-space edges must be strictly increasing")

    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    if not (np.allclose(dx, dx[0], rtol=1e-12, atol=1e-12)
            and np.allclose(dy, dy[0], rtol=1e-12, atol=1e-12)):
        raise ValueError("The wide-stencil solver requires a uniform physical grid")

    return {
        **arrays,
        "reliable_mask": reliable,
        "path": path,
        "dx": float(dx[0]),
        "dy": float(dy[0]),
    }


def _largest_reliable_component(reliable):
    """Return the largest 4-connected reliable component and component statistics."""
    structure = ndimage.generate_binary_structure(2, 1)
    labels, count = ndimage.label(reliable, structure=structure)
    if count == 0:
        raise ValueError("The phase grid contains no reliable cells")
    sizes = np.bincount(labels.ravel())[1:]
    main_label = int(np.argmax(sizes)) + 1
    main = labels == main_label
    return main, {
        "n_reliable_components_4_connected": int(count),
        "component_sizes_descending": [
            int(v) for v in np.sort(sizes)[::-1].tolist()
        ],
        "largest_component_cells": int(main.sum()),
        "excluded_reliable_cells": int(reliable.sum() - main.sum()),
    }


def _small_enclosed_holes(main_component, max_hole_cells):
    """Return enclosed holes no larger than ``max_hole_cells``."""
    if max_hole_cells < 0:
        raise ValueError("max_hole_cells must be nonnegative")
    filled_all = ndimage.binary_fill_holes(main_component)
    holes_all = filled_all & ~main_component
    labels, count = ndimage.label(
        holes_all,
        structure=ndimage.generate_binary_structure(2, 1),
    )
    selected = np.zeros_like(main_component, dtype=bool)
    sizes = []
    for label in range(1, count + 1):
        region = labels == label
        size = int(region.sum())
        sizes.append(size)
        if size <= max_hole_cells:
            selected |= region
    return selected, sorted(sizes, reverse=True)


def _interpolate_enclosed_holes(drift, diffusion, main_component, holes,
                                x_centers, y_centers, neighbors):
    """Fill enclosed holes by positive weighted averages of reliable coefficients.

    Positive convex combinations of symmetric positive-definite diffusion
    matrices remain positive definite, so this interpolation does not destroy
    the diffusion-tensor structure.
    """
    drift = np.asarray(drift, dtype=np.float64).copy()
    diffusion = np.asarray(diffusion, dtype=np.float64).copy()
    if not np.any(holes):
        return drift, diffusion

    source_indices = np.argwhere(main_component)
    target_indices = np.argwhere(holes)
    source_points = np.column_stack((
        x_centers[source_indices[:, 0]],
        y_centers[source_indices[:, 1]],
    ))
    target_points = np.column_stack((
        x_centers[target_indices[:, 0]],
        y_centers[target_indices[:, 1]],
    ))
    tree = cKDTree(source_points)
    k = min(max(1, int(neighbors)), len(source_indices))
    distances, local_indices = tree.query(target_points, k=k)

    if k == 1:
        distances = distances[:, None]
        local_indices = local_indices[:, None]

    weights = 1.0 / np.maximum(distances, 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)

    for target_row, (i, j) in enumerate(target_indices):
        neighbors_ij = source_indices[local_indices[target_row]]
        for channel in range(2):
            values = drift[
                channel,
                neighbors_ij[:, 0],
                neighbors_ij[:, 1],
            ]
            drift[channel, i, j] = float(weights[target_row] @ values)

        # Average the symmetric 2x2 matrices with identical positive weights.
        d00 = diffusion[0, neighbors_ij[:, 0], neighbors_ij[:, 1]]
        d11 = diffusion[1, neighbors_ij[:, 0], neighbors_ij[:, 1]]
        d01 = 0.5 * (
            diffusion[2, neighbors_ij[:, 0], neighbors_ij[:, 1]]
            + diffusion[3, neighbors_ij[:, 0], neighbors_ij[:, 1]]
        )
        diffusion[0, i, j] = float(weights[target_row] @ d00)
        diffusion[1, i, j] = float(weights[target_row] @ d11)
        cross = float(weights[target_row] @ d01)
        diffusion[2, i, j] = cross
        diffusion[3, i, j] = cross

    return drift, diffusion


def _minimum_eigenvalue(diffusion, mask):
    """Return the minimum eigenvalue of symmetric 2x2 matrices on ``mask``."""
    d00 = diffusion[0][mask]
    d11 = diffusion[1][mask]
    d01 = 0.5 * (diffusion[2][mask] + diffusion[3][mask])
    half_trace = 0.5 * (d00 + d11)
    radius = np.sqrt(
        np.maximum(0.0, (0.5 * (d00 - d11)) ** 2 + d01 ** 2)
    )
    return float(np.min(half_trace - radius))


def _candidate_directions(dx, dy, max_radius):
    """Create positive-slope lattice directions ordered only by later scoring."""
    if not isinstance(max_radius, (int, np.integer)) or max_radius < 1:
        raise ValueError("max_stencil_radius must be a positive integer")
    result = []
    for m in range(1, int(max_radius) + 1):
        for n in range(1, int(max_radius) + 1):
            slope = (n * dy) / (m * dx)
            result.append((float(slope), int(m), int(n)))
    return tuple(result)


def _choose_diffusion_direction(a0, a1, d00, d11, d01, dx, dy,
                                candidates, tolerance):
    """Choose a nonnegative wide-stencil decomposition of one local tensor.

    The diffusion tensor is decomposed into x-axis, y-axis and one signed
    diagonal rank-one contribution. Drift is represented by biasing the two
    axis jump pairs without changing their total second moments.
    """
    cross = abs(float(d01))
    if d00 <= 0 or d11 <= 0 or d00 * d11 - d01 * d01 <= 0:
        raise ValueError("Diffusion matrix must be positive definite")

    if cross <= tolerance:
        if (d00 + tolerance < abs(a0) * dx
                or d11 + tolerance < abs(a1) * dy):
            raise ValueError(
                "Axis diffusion is too small to represent drift with "
                "nonnegative locally consistent rates"
            )
        return {
            "m": 0,
            "n": 0,
            "slope": None,
            "residual_x": float(d00),
            "residual_y": float(d11),
        }

    lower = cross / d00
    upper = d11 / cross
    target = math.sqrt(d11 / d00)
    feasible = []

    for slope, m, n in candidates:
        if slope < lower - tolerance or slope > upper + tolerance:
            continue
        residual_x = d00 - cross / slope
        residual_y = d11 - cross * slope
        if (residual_x + tolerance < abs(a0) * dx
                or residual_y + tolerance < abs(a1) * dy):
            continue
        feasible.append((
            max(m, n),
            m * m + n * n,
            abs(math.log(slope / target)),
            slope,
            m,
            n,
            residual_x,
            residual_y,
        ))

    if not feasible:
        raise ValueError(
            "No nonnegative locally consistent stencil direction is available; "
            "increase max_stencil_radius"
        )

    _, _, _, slope, m, n, residual_x, residual_y = min(feasible)
    return {
        "m": int(m),
        "n": int(n),
        "slope": float(slope),
        "residual_x": float(residual_x),
        "residual_y": float(residual_y),
    }


def _local_jump_rates(a0, a1, d00, d11, d01, dx, dy,
                      candidates, tolerance):
    """Return locally consistent nonnegative jumps before boundary censoring."""
    choice = _choose_diffusion_direction(
        a0, a1, d00, d11, d01, dx, dy,
        candidates, tolerance,
    )
    residual_x = choice["residual_x"]
    residual_y = choice["residual_y"]

    base_x = residual_x / (2.0 * dx * dx)
    base_y = residual_y / (2.0 * dy * dy)
    jumps = [
        (1, 0, base_x + a0 / (2.0 * dx)),
        (-1, 0, base_x - a0 / (2.0 * dx)),
        (0, 1, base_y + a1 / (2.0 * dy)),
        (0, -1, base_y - a1 / (2.0 * dy)),
    ]

    m = choice["m"]
    n = choice["n"]
    if m and n:
        cross = abs(float(d01))
        diagonal_rate = cross / (2.0 * (m * dx) * (n * dy))
        sign = 1 if d01 >= 0 else -1
        jumps.extend([
            (m, sign * n, diagonal_rate),
            (-m, -sign * n, diagonal_rate),
        ])

    cleaned = []
    for di, dj, rate in jumps:
        if rate < -tolerance:
            raise ValueError(f"Negative transition rate {rate}")
        cleaned.append((di, dj, max(0.0, float(rate))))

    return cleaned, choice


def _jump_moments(jumps, dx, dy):
    """Return first and second moments per unit time for a jump-rate list."""
    first = np.zeros(2, dtype=np.float64)
    second = np.zeros((2, 2), dtype=np.float64)
    for di, dj, rate in jumps:
        delta = np.array([di * dx, dj * dy], dtype=np.float64)
        first += rate * delta
        second += rate * np.outer(delta, delta)
    return first, second


def _build_generator(drift, diffusion, domain, empirical_mass,
                     dx, dy, max_stencil_radius, rate_tolerance):
    """Build a conservative sparse Markov generator on the selected domain."""
    candidates = _candidate_directions(dx, dy, max_stencil_radius)
    coordinates = np.argwhere(domain)
    n_states = len(coordinates)
    index = np.full(domain.shape, -1, dtype=np.int64)
    index[domain] = np.arange(n_states, dtype=np.int64)

    rows = []
    cols = []
    values = []
    stencil_radius = np.zeros(domain.shape, dtype=np.int16)
    censored_fraction = np.full(domain.shape, np.nan, dtype=np.float64)

    max_first_error = 0.0
    max_second_error = 0.0
    total_proposed_rate = 0.0
    total_censored_rate = 0.0
    empirical_weighted_censored = 0.0
    radius_counts = Counter()

    domain_mass = float(empirical_mass[domain].sum())
    empirical_weights = (
        empirical_mass / domain_mass
        if domain_mass > 0
        else np.zeros_like(empirical_mass, dtype=np.float64)
    )

    for state, (i, j) in enumerate(coordinates):
        a0 = float(drift[0, i, j])
        a1 = float(drift[1, i, j])
        d00 = float(diffusion[0, i, j])
        d11 = float(diffusion[1, i, j])
        d01 = 0.5 * float(diffusion[2, i, j] + diffusion[3, i, j])
        if not np.all(np.isfinite([a0, a1, d00, d11, d01])):
            raise ValueError(f"Non-finite coefficient inside solver domain at {(i, j)}")

        jumps, choice = _local_jump_rates(
            a0, a1, d00, d11, d01,
            dx, dy, candidates, rate_tolerance,
        )
        radius = max(choice["m"], choice["n"])
        stencil_radius[i, j] = radius
        radius_counts[int(radius)] += 1

        first, second = _jump_moments(jumps, dx, dy)
        target_first = np.array([a0, a1], dtype=np.float64)
        target_second = np.array(
            [[d00, d01], [d01, d11]],
            dtype=np.float64,
        )
        max_first_error = max(
            max_first_error,
            float(np.max(np.abs(first - target_first))),
        )
        max_second_error = max(
            max_second_error,
            float(np.max(np.abs(second - target_second))),
        )

        row_out = 0.0
        proposed = 0.0
        censored = 0.0
        for di, dj, rate in jumps:
            if rate <= 0:
                continue
            proposed += rate
            ni = i + di
            nj = j + dj
            if (0 <= ni < domain.shape[0]
                    and 0 <= nj < domain.shape[1]
                    and domain[ni, nj]):
                rows.append(state)
                cols.append(int(index[ni, nj]))
                values.append(rate)
                row_out += rate
            else:
                censored += rate

        total_proposed_rate += proposed
        total_censored_rate += censored
        fraction = censored / proposed if proposed > 0 else 0.0
        censored_fraction[i, j] = fraction
        empirical_weighted_censored += empirical_weights[i, j] * fraction

        rows.append(state)
        cols.append(state)
        values.append(-row_out)

    generator = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(n_states, n_states),
        dtype=np.float64,
    )
    generator.sum_duplicates()

    row_sums = np.asarray(generator.sum(axis=1)).ravel()
    off_diagonal = generator.copy().tolil()
    off_diagonal.setdiag(0.0)
    off_diagonal = off_diagonal.tocsr()
    off_diagonal.eliminate_zeros()

    min_off_diagonal = (
        float(off_diagonal.data.min())
        if off_diagonal.nnz
        else 0.0
    )
    if min_off_diagonal < -rate_tolerance:
        raise AssertionError("Generator contains negative off-diagonal rates")

    adjacency = off_diagonal.copy()
    adjacency.data[:] = 1.0
    strong_count, _ = connected_components(
        adjacency,
        directed=True,
        connection="strong",
    )

    diagnostics = {
        "states": int(n_states),
        "generator_nonzeros": int(generator.nnz),
        "maximum_absolute_row_sum": float(np.max(np.abs(row_sums))),
        "minimum_off_diagonal_rate": min_off_diagonal,
        "strongly_connected_components": int(strong_count),
        "maximum_first_moment_reconstruction_error": max_first_error,
        "maximum_second_moment_reconstruction_error": max_second_error,
        "total_censored_rate_fraction": (
            float(total_censored_rate / total_proposed_rate)
            if total_proposed_rate > 0 else 0.0
        ),
        "empirical_mass_weighted_censored_rate_fraction": float(
            empirical_weighted_censored
        ),
        "cells_with_any_censored_rate": int(
            np.count_nonzero((censored_fraction > 0) & domain)
        ),
        "empirical_mass_in_cells_with_any_censored_rate": float(
            empirical_mass[(censored_fraction > 0) & domain].sum()
            / domain_mass
        ) if domain_mass > 0 else None,
        "stencil_radius_counts": {
            str(radius): int(count)
            for radius, count in sorted(radius_counts.items())
        },
        "maximum_stencil_radius_used": int(max(radius_counts, default=0)),
    }
    return generator, coordinates, stencil_radius, censored_fraction, diagnostics


def _solve_stationary_mass(generator, negative_tolerance):
    """Solve Q.T pi = 0 with unit normalization."""
    n_states = generator.shape[0]
    if n_states < 2:
        raise ValueError("At least two solver states are required")

    system = generator.T.tolil(copy=True)
    rhs = np.zeros(n_states, dtype=np.float64)

    # Replacing one dependent stationary equation by normalization produces a
    # square nonsingular system for an irreducible finite Markov chain.
    normalization_row = 0
    system[normalization_row, :] = np.ones(n_states, dtype=np.float64)
    rhs[normalization_row] = 1.0
    system = system.tocsr()

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        probability = np.asarray(spsolve(system, rhs), dtype=np.float64)

    if probability.shape != (n_states,) or not np.all(np.isfinite(probability)):
        raise ValueError("Stationary linear system returned non-finite values")
    if float(probability.min()) < -negative_tolerance:
        raise ValueError(
            f"Stationary mass contains a negative value {probability.min()}"
        )

    probability = np.maximum(probability, 0.0)
    total = float(probability.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Stationary probability cannot be normalized")
    probability /= total

    residual = generator.T @ probability
    return probability, {
        "probability_sum": float(probability.sum()),
        "minimum_probability": float(probability.min()),
        "maximum_probability": float(probability.max()),
        "maximum_absolute_stationary_residual": float(
            np.max(np.abs(residual))
        ),
        "l1_stationary_residual": float(np.sum(np.abs(residual))),
    }


def _distribution_moments(mass, domain, x_centers, y_centers):
    """Return mean, standard deviation and correlation for one grid mass."""
    coordinates = np.argwhere(domain)
    weights = mass[domain].astype(np.float64)
    weights /= weights.sum()
    xx = x_centers[coordinates[:, 0]]
    yy = y_centers[coordinates[:, 1]]
    mean_x = float(weights @ xx)
    mean_y = float(weights @ yy)
    var_x = float(weights @ ((xx - mean_x) ** 2))
    var_y = float(weights @ ((yy - mean_y) ** 2))
    covariance = float(weights @ ((xx - mean_x) * (yy - mean_y)))
    correlation = covariance / math.sqrt(var_x * var_y)
    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "std_x": math.sqrt(max(var_x, 0.0)),
        "std_y": math.sqrt(max(var_y, 0.0)),
        "correlation": float(correlation),
    }


def _comparison_metrics(stationary, empirical, domain):
    """Compare two probability masses conditioned on the solver domain."""
    p = stationary[domain].astype(np.float64)
    q = empirical[domain].astype(np.float64)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    def kl(left, right):
        keep = left > 0
        return float(np.sum(left[keep] * np.log(left[keep] / right[keep])))

    return {
        "total_variation_distance": float(0.5 * np.sum(np.abs(p - q))),
        "hellinger_distance": float(
            math.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
        ),
        "jensen_shannon_divergence_nats": float(
            0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)
        ),
    }


def _phase_file(root, names, start_year, end_year, method, bins):
    """Return the conventional prepared phase-grid path."""
    return (
        Path(root)
        / "Functional"
        / "FokkerPlanck2D"
        / (
            f"{names[0]}-{names[1]}_{start_year}-{end_year}_{method}_"
            f"{int(bins[0])}x{int(bins[1])}.npz"
        )
    )


def solve_stationary_fokker_planck_2d(
        files_path_prefix,
        start_year,
        end_year,
        *,
        names=("sensible", "latent"),
        method="sp",
        bins=(80, 80),
        phase_file=None,
        max_stencil_radius=7,
        max_hole_cells=16,
        hole_neighbors=4,
        rate_tolerance=1e-10,
        negative_tolerance=1e-12,
        overwrite=False):
    """Solve the stationary 2D Fokker-Planck problem on a prepared phase grid.

    Parameters
    ----------
    files_path_prefix : str or Path
        Project data root.
    start_year, end_year : int
        Period labels used by the prepared phase grid.
    names : tuple[str, str]
        Ordered state coordinates.
    method : {'np', 'sp'}
        Coefficient-estimation method.
    bins : tuple[int, int]
        Prepared phase-grid dimensions.
    phase_file : str or Path, optional
        Explicit prepared NPZ. If omitted, the conventional project path is used.
    max_stencil_radius : int
        Maximum integer lattice radius searched for a monotone diagonal stencil.
    max_hole_cells : int
        Maximum size of each enclosed unreliable hole allowed to be filled.
    hole_neighbors : int
        Number of nearby reliable cells used in convex coefficient interpolation.
    rate_tolerance : float
        Floating-point tolerance for local rate nonnegativity.
    negative_tolerance : float
        Allowed roundoff negativity in the stationary linear solve.
    overwrite : bool
        Recompute even when the stationary result already exists.

    Returns
    -------
    pathlib.Path
        Saved NPZ containing stationary mass, density and solver diagnostics.

    Notes
    -----
    The method is a Markov-chain approximation of the diffusion generator.
    Away from the boundary, every local stencil exactly reproduces the supplied
    drift vector and diffusion tensor to floating-point tolerance. Suppressing
    jumps that leave the selected connected domain imposes a discrete no-outflow
    boundary while preserving total probability and nonnegativity.
    """
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("Exactly two different state variables are required")
    if method not in ("np", "sp"):
        raise ValueError("method must be 'np' or 'sp'")
    if hole_neighbors < 1:
        raise ValueError("hole_neighbors must be positive")
    if rate_tolerance < 0 or negative_tolerance < 0:
        raise ValueError("Numerical tolerances must be nonnegative")

    root = Path(files_path_prefix)
    phase_path = (
        Path(phase_file)
        if phase_file is not None
        else _phase_file(root, names, start_year, end_year, method, bins)
    )
    grid = _load_phase_grid(phase_path)

    output_dir = root / "Functional" / "FokkerPlanck2D" / "Stationary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{phase_path.stem}_stationary_wide_stencil.npz"
    report = output.with_suffix(".json")
    if output.is_file() and report.is_file() and not overwrite:
        print(f"Reusing stationary 2D Fokker-Planck solution: {output}", flush=True)
        return output

    reliable = grid["reliable_mask"]
    main, component_diagnostics = _largest_reliable_component(reliable)
    holes, all_hole_sizes = _small_enclosed_holes(main, max_hole_cells)
    domain = main | holes

    drift, diffusion = _interpolate_enclosed_holes(
        grid["drift"],
        grid["diffusion"],
        main,
        holes,
        grid["x_centers"],
        grid["y_centers"],
        hole_neighbors,
    )
    # Enforce exact stored symmetry before constructing rates.
    cross = 0.5 * (diffusion[2] + diffusion[3])
    diffusion[2] = cross
    diffusion[3] = cross

    minimum_eigenvalue = _minimum_eigenvalue(diffusion, domain)
    if minimum_eigenvalue <= 0:
        raise ValueError(
            f"Solver-domain diffusion is not positive definite: "
            f"minimum eigenvalue={minimum_eigenvalue}"
        )

    generator, coordinates, stencil_radius, censored_fraction, generator_diag = (
        _build_generator(
            drift,
            diffusion,
            domain,
            grid["empirical_mass"],
            grid["dx"],
            grid["dy"],
            max_stencil_radius,
            rate_tolerance,
        )
    )
    if generator_diag["strongly_connected_components"] != 1:
        raise ValueError(
            "Discrete Markov generator is not strongly connected; "
            "the stationary distribution would not be unique"
        )

    probability, solve_diag = _solve_stationary_mass(
        generator,
        negative_tolerance,
    )

    stationary_mass = np.zeros(domain.shape, dtype=np.float64)
    stationary_mass[domain] = probability
    cell_area = grid["dx"] * grid["dy"]
    stationary_density = stationary_mass / cell_area

    empirical_conditioned = np.zeros(domain.shape, dtype=np.float64)
    empirical_domain_total = float(grid["empirical_mass"][domain].sum())
    if empirical_domain_total <= 0:
        raise ValueError("No empirical probability mass lies in the solver domain")
    empirical_conditioned[domain] = (
        grid["empirical_mass"][domain] / empirical_domain_total
    )
    empirical_density_conditioned = empirical_conditioned / cell_area

    solver_drift = np.full_like(drift, np.nan)
    solver_diffusion = np.full_like(diffusion, np.nan)
    solver_drift[:, domain] = drift[:, domain]
    solver_diffusion[:, domain] = diffusion[:, domain]

    comparison = _comparison_metrics(
        stationary_mass,
        empirical_conditioned,
        domain,
    )
    stationary_moments = _distribution_moments(
        stationary_mass,
        domain,
        grid["x_centers"],
        grid["y_centers"],
    )
    empirical_moments = _distribution_moments(
        empirical_conditioned,
        domain,
        grid["x_centers"],
        grid["y_centers"],
    )

    phase_report = phase_path.with_suffix(".json")
    phase_metadata = None
    if phase_report.is_file():
        phase_metadata = json.loads(
            phase_report.read_text(encoding="utf-8")
        )

    metadata = {
        "method": "wide_stencil_continuous_time_markov_chain",
        "equation": (
            "stationary adjoint problem Q.T @ pi = 0 approximating "
            "-d_i(A_i p) + 0.5 d_i d_j(D_ij p) = 0"
        ),
        "phase_grid": _source_signature(phase_path),
        "settings": {
            "names": list(names),
            "period": f"{start_year}-{end_year}",
            "coefficient_method": method,
            "bins": [int(v) for v in bins],
            "max_stencil_radius": int(max_stencil_radius),
            "max_hole_cells": int(max_hole_cells),
            "hole_neighbors": int(hole_neighbors),
            "rate_tolerance": float(rate_tolerance),
            "negative_tolerance": float(negative_tolerance),
            "boundary": "censored no-outflow jumps",
        },
        "domain": {
            **component_diagnostics,
            "all_enclosed_hole_sizes_descending": [
                int(v) for v in all_hole_sizes
            ],
            "filled_hole_cells": int(holes.sum()),
            "solver_cells": int(domain.sum()),
            "phase_grid_empirical_mass_in_solver_domain": empirical_domain_total,
            "minimum_solver_domain_diffusion_eigenvalue": minimum_eigenvalue,
        },
        "generator": generator_diag,
        "linear_solve": solve_diag,
        "comparison_with_empirical_mass_conditioned_on_solver_domain": comparison,
        "stationary_moments": stationary_moments,
        "empirical_moments_conditioned_on_solver_domain": empirical_moments,
    }

    if phase_metadata is not None:
        diagnostics = phase_metadata.get("diagnostics", {})
        inside = diagnostics.get("samples_inside_phase_domain")
        total = diagnostics.get("total_block_samples_with_finite_coefficients")
        if inside is not None and total:
            metadata["domain"]["fraction_of_original_finite_samples_in_phase_grid"] = (
                float(inside) / float(total)
            )
            metadata["domain"]["approx_fraction_of_original_finite_samples_in_solver_domain"] = (
                float(inside) / float(total) * empirical_domain_total
            )

    np.savez_compressed(
        output,
        x_edges=grid["x_edges"],
        y_edges=grid["y_edges"],
        x_centers=grid["x_centers"],
        y_centers=grid["y_centers"],
        domain_mask=domain,
        original_main_reliable_component=main,
        filled_holes=holes,
        stationary_mass=stationary_mass,
        stationary_density=stationary_density,
        empirical_mass_conditioned=empirical_conditioned,
        empirical_density_conditioned=empirical_density_conditioned,
        drift_solver=solver_drift,
        diffusion_solver=solver_diffusion,
        stencil_radius=stencil_radius,
        censored_rate_fraction=censored_fraction,
    )
    report.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved stationary 2D Fokker-Planck solution: {output}", flush=True)
    print(f"Saved stationary 2D Fokker-Planck diagnostics: {report}", flush=True)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    return output
