"""Copula densities, marginal constraints, joint quantile classes, diagnostic metrics and period-level estimation workflows."""

from __future__ import annotations

# --- Copula Calculations ---

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kendalltau, norm, rankdata
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import t
from scipy.stats import beta

def normalize_1d_density(x, density):
    """Normalize the one-dimensional density on its supplied grid."""
    x = np.asarray(x, dtype=float)
    density = np.maximum(np.asarray(density, dtype=float), 0.0)
    normalization = np.trapezoid(density, x)
    if normalization <= 0:
        raise RuntimeError("Density normalization is zero.")
    return density / normalization


def calculate_cdf_from_density(x, density):
    """Integrate the density to obtain a cumulative distribution on its grid."""
    density = normalize_1d_density(x, density)
    cdf = cumulative_trapezoid(density, x, initial=0.0)
    return cdf / cdf[-1]


def estimate_kendall_dependence(data1_array, data2_array, sample_size=300000, random_seed=12345):
    """Estimate Kendall rank dependence from paired observations."""
    data1_array = np.asarray(data1_array)
    data2_array = np.asarray(data2_array)
    if data1_array.shape != data2_array.shape:
        raise ValueError("data1_array and data2_array must have the same shape.")

    rng = np.random.default_rng(random_seed)
    data1_flat = data1_array.ravel()
    data2_flat = data2_array.ravel()
    total_size = len(data1_flat)
    sampled_x1, sampled_x2 = [], []
    collected = 0

    while collected < sample_size:
        needed = sample_size - collected
        batch_size = max(2 * needed, 10000)
        indices = rng.integers(0, total_size, size=batch_size)
        x1 = data1_flat[indices]
        x2 = data2_flat[indices]
        valid = np.isfinite(x1) & np.isfinite(x2)
        x1 = x1[valid]
        x2 = x2[valid]
        take = min(needed, len(x1))
        sampled_x1.append(x1[:take])
        sampled_x2.append(x2[:take])
        collected += take

    sampled_x1 = np.concatenate(sampled_x1)
    sampled_x2 = np.concatenate(sampled_x2)
    tau, p_value = kendalltau(sampled_x1, sampled_x2)
    rho = np.sin(0.5 * np.pi * tau)

    print(f"Kendall tau: {tau}")
    print(f"Gaussian copula rho: {rho}")
    print(f"p-value: {p_value}")
    print(f"sample size: {len(sampled_x1)}")

    return {"tau": tau, "rho": rho, "p_value": p_value, "sampled_x1": sampled_x1, "sampled_x2": sampled_x2}


def gaussian_copula_density(u, v, rho, epsilon=1e-10):
    """Evaluate Gaussian copula density at the supplied probability coordinates."""
    u = np.clip(u, epsilon, 1.0 - epsilon)
    v = np.clip(v, epsilon, 1.0 - epsilon)
    z1 = norm.ppf(u)
    z2 = norm.ppf(v)
    denominator = 1.0 - rho ** 2
    exponent = (2.0 * rho * z1 * z2 - rho ** 2 * (z1 ** 2 + z2 ** 2)) / (2.0 * denominator)
    return np.exp(exponent) / np.sqrt(denominator)


def build_gaussian_copula_stationary_density(x1, p1, x2, p2, rho):
    """Combine Gaussian copula density with the specified stationary marginals."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    p1 = normalize_1d_density(x1, p1)
    p2 = normalize_1d_density(x2, p2)
    cdf1 = calculate_cdf_from_density(x1, p1)
    cdf2 = calculate_cdf_from_density(x2, p2)
    u, v = np.meshgrid(cdf1, cdf2)
    copula_density = gaussian_copula_density(u, v, rho)
    p1_mesh, p2_mesh = np.meshgrid(p1, p2)
    joint_density = copula_density * p1_mesh * p2_mesh
    normalization = np.trapezoid(np.trapezoid(joint_density, x1, axis=1), x2)
    joint_density /= normalization
    return {"joint_density": joint_density, "p1": p1, "p2": p2, "cdf1": cdf1, "cdf2": cdf2, "rho": rho}


def calculate_copula_marginals(joint_density, x1, x2):
    """Integrate a joint density to recover its marginal densities."""
    marginal_x1 = np.trapezoid(joint_density, x2, axis=0)
    marginal_x2 = np.trapezoid(joint_density, x1, axis=1)
    return marginal_x1, marginal_x2


def calculate_joint_density_moments(joint_density, x1, x2):
    """Compute moments of the supplied joint density."""
    x1_mesh, x2_mesh = np.meshgrid(x1, x2)
    mean_x1 = np.trapezoid(np.trapezoid(x1_mesh * joint_density, x1, axis=1), x2)
    mean_x2 = np.trapezoid(np.trapezoid(x2_mesh * joint_density, x1, axis=1), x2)
    var_x1 = np.trapezoid(np.trapezoid((x1_mesh - mean_x1) ** 2 * joint_density, x1, axis=1), x2)
    var_x2 = np.trapezoid(np.trapezoid((x2_mesh - mean_x2) ** 2 * joint_density, x1, axis=1), x2)
    covariance = np.trapezoid(np.trapezoid((x1_mesh - mean_x1) * (x2_mesh - mean_x2) * joint_density, x1, axis=1), x2)
    correlation = covariance / np.sqrt(var_x1 * var_x2)
    return {"mean_x1": mean_x1, "mean_x2": mean_x2, "sd_x1": np.sqrt(var_x1), "sd_x2": np.sqrt(var_x2), "covariance": covariance, "correlation": correlation}


def calculate_marginal_errors(joint_density, x1, p1, x2, p2):
    """Measure discrepancies between reconstructed and target marginal densities."""
    p1 = normalize_1d_density(x1, p1)
    p2 = normalize_1d_density(x2, p2)
    marginal_x1, marginal_x2 = calculate_copula_marginals(joint_density, x1, x2)
    error_x1 = np.trapezoid(np.abs(marginal_x1 - p1), x1)
    error_x2 = np.trapezoid(np.abs(marginal_x2 - p2), x2)
    return error_x1, error_x2


def print_copula_summary(dependence, moments, error_x1, error_x2):
    """Print diagnostics of the estimated copula."""
    print(f"Kendall tau: {dependence['tau']}")
    print(f"Gaussian copula rho: {dependence['rho']}")
    print(f"Resulting Pearson correlation: {moments['correlation']}")
    print(f"L1 marginal error X1: {error_x1}")
    print(f"L1 marginal error X2: {error_x2}")

def build_empirical_copula_density(sampled_x1, sampled_x2, bins=60):
    """Estimate an empirical copula density from paired observations."""
    sampled_x1 = np.asarray(sampled_x1, dtype=float)
    sampled_x2 = np.asarray(sampled_x2, dtype=float)
    valid = np.isfinite(sampled_x1) & np.isfinite(sampled_x2)
    sampled_x1 = sampled_x1[valid]
    sampled_x2 = sampled_x2[valid]

    n = len(sampled_x1)
    u = rankdata(sampled_x1, method="average") / (n + 1.0)
    v = rankdata(sampled_x2, method="average") / (n + 1.0)

    edges = np.linspace(0.0, 1.0, bins + 1)
    density, _, _ = np.histogram2d(v, u, bins=[edges, edges], density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return {"u": u, "v": v, "density": density, "centers": centers, "edges": edges}


def build_gaussian_copula_density_grid(rho, centers):
    """Evaluate the Gaussian copula over a probability grid."""
    u_mesh, v_mesh = np.meshgrid(centers, centers)
    density = gaussian_copula_density(u_mesh, v_mesh, rho)

    du = centers[1] - centers[0]
    density /= np.sum(density) * du ** 2

    return density


def compare_copula_densities(empirical_density, gaussian_density, centers):
    """Compare empirical and model copula densities on a common grid."""
    du = centers[1] - centers[0]
    empirical_probability = empirical_density * du ** 2
    gaussian_probability = gaussian_density * du ** 2

    empirical_probability /= empirical_probability.sum()
    gaussian_probability /= gaussian_probability.sum()

    tv = 0.5 * np.sum(np.abs(empirical_probability - gaussian_probability))
    hellinger = np.sqrt(0.5 * np.sum((np.sqrt(empirical_probability) - np.sqrt(gaussian_probability)) ** 2))

    return {"total_variation": tv, "hellinger": hellinger}

def student_t_copula_density(u, v, rho, df, epsilon=1e-8):
    """Evaluate Student-t copula density at the requested probability coordinates."""
    u = np.clip(u, epsilon, 1.0 - epsilon)
    v = np.clip(v, epsilon, 1.0 - epsilon)

    z1 = t.ppf(u, df)
    z2 = t.ppf(v, df)
    denominator = 1.0 - rho ** 2

    log_joint = gammaln((df + 2.0) / 2.0) - gammaln(df / 2.0)
    log_joint -= np.log(df * np.pi) + 0.5 * np.log(denominator)
    log_joint -= (df + 2.0) / 2.0 * np.log1p((z1 ** 2 - 2.0 * rho * z1 * z2 + z2 ** 2) / (df * denominator))

    log_marginal1 = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
    log_marginal1 -= 0.5 * np.log(df * np.pi)
    log_marginal1 -= (df + 1.0) / 2.0 * np.log1p(z1 ** 2 / df)

    log_marginal2 = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
    log_marginal2 -= 0.5 * np.log(df * np.pi)
    log_marginal2 -= (df + 1.0) / 2.0 * np.log1p(z2 ** 2 / df)

    return np.exp(log_joint - log_marginal1 - log_marginal2)


def fit_student_t_copula(u, v, rho, df_min=2.01, df_max=100.0):
    """Fit the Student-t copula parameters from paired rank observations."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    valid = np.isfinite(u) & np.isfinite(v)
    u = u[valid]
    v = v[valid]

    def negative_log_likelihood(df):
        """Evaluate the negative log likelihood for the candidate copula parameters."""
        density = student_t_copula_density(u, v, rho, df)
        return -np.sum(np.log(np.maximum(density, 1e-300)))

    result = minimize_scalar(negative_log_likelihood, bounds=(df_min, df_max), method="bounded")

    if not result.success:
        raise RuntimeError("Student t-copula optimization failed.")

    print(f"Student t-copula rho: {rho}")
    print(f"Student t-copula df: {result.x}")
    print(f"Negative log-likelihood: {result.fun}")

    return {"rho": rho, "df": result.x, "negative_log_likelihood": result.fun}


def build_student_t_copula_density_grid(rho, df, centers):
    """Evaluate the fitted Student-t copula on a probability grid."""
    u_mesh, v_mesh = np.meshgrid(centers, centers)
    density = student_t_copula_density(u_mesh, v_mesh, rho, df)
    du = centers[1] - centers[0]
    density /= np.sum(density) * du ** 2
    return density


def calculate_student_t_tail_dependence(rho, df):
    """Calculate the Student-t copula tail dependence coefficient."""
    value = 2.0 * t.cdf(-np.sqrt((df + 1.0) * (1.0 - rho) / (1.0 + rho)), df + 1.0)
    return value

def build_checkerboard_copula(u, v, bins=50, tolerance=1e-12, maximum_iterations=10000):
    """Construct a piecewise-constant checkerboard copula from paired ranks."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    valid = np.isfinite(u) & np.isfinite(v)
    u = np.clip(u[valid], 0.0, 1.0)
    v = np.clip(v[valid], 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _, _ = np.histogram2d(v, u, bins=[edges, edges])

    probability_raw = counts / counts.sum()
    probability = probability_raw.copy()
    target = np.full(bins, 1.0 / bins)

    if np.any(probability.sum(axis=0) == 0) or np.any(probability.sum(axis=1) == 0):
        raise RuntimeError("Checkerboard contains an empty marginal bin.")

    error = np.inf

    for iteration in range(maximum_iterations):
        probability *= (target / probability.sum(axis=1))[:, None]
        probability *= (target / probability.sum(axis=0))[None, :]

        row_error = np.max(np.abs(probability.sum(axis=1) - target))
        column_error = np.max(np.abs(probability.sum(axis=0) - target))
        error = max(row_error, column_error)

        if error < tolerance:
            break

    density = probability * bins ** 2
    raw_density = probability_raw * bins ** 2
    correction_tv = 0.5 * np.sum(np.abs(probability - probability_raw))

    print(f"Checkerboard bins: {bins} x {bins}")
    print(f"Checkerboard iterations: {iteration + 1}")
    print(f"Checkerboard marginal error: {error}")
    print(f"TV introduced by marginal balancing: {correction_tv}")

    return {
        "density": density,
        "raw_density": raw_density,
        "probability": probability,
        "centers": centers,
        "edges": edges,
        "bins": bins,
        "balancing_error": error,
        "balancing_tv": correction_tv,
    }


def trapezoid_weights(x):
    """Compute trapezoidal quadrature weights for the given grid."""
    x = np.asarray(x, dtype=float)
    weights = np.empty(len(x))
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def balance_joint_density_marginals(joint_density, x1, p1, x2, p2, tolerance=1e-12, maximum_iterations=10000):
    """Adjust joint density weights to match the requested marginal densities."""
    w1 = trapezoid_weights(x1)
    w2 = trapezoid_weights(x2)

    target1 = p1 * w1
    target2 = p2 * w2
    target1 /= target1.sum()
    target2 /= target2.sum()

    joint_probability = joint_density * w2[:, None] * w1[None, :]
    joint_probability /= joint_probability.sum()

    error = np.inf

    for iteration in range(maximum_iterations):
        row_sum = joint_probability.sum(axis=1)
        row_factor = np.divide(target2, row_sum, out=np.ones_like(target2), where=row_sum > 0)
        joint_probability *= row_factor[:, None]

        column_sum = joint_probability.sum(axis=0)
        column_factor = np.divide(target1, column_sum, out=np.ones_like(target1), where=column_sum > 0)
        joint_probability *= column_factor[None, :]

        row_error = np.max(np.abs(joint_probability.sum(axis=1) - target2))
        column_error = np.max(np.abs(joint_probability.sum(axis=0) - target1))
        error = max(row_error, column_error)

        if error < tolerance:
            break

    joint_density = joint_probability / (w2[:, None] * w1[None, :])

    print(f"Physical marginal balancing iterations: {iteration + 1}")
    print(f"Physical marginal balancing error: {error}")

    return joint_density


def build_checkerboard_stationary_density(x1, p1, x2, p2, checkerboard, enforce_marginals=True):
    """Combine checkerboard copula density and stationary marginals."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    p1 = normalize_1d_density(x1, p1)
    p2 = normalize_1d_density(x2, p2)

    cdf1 = calculate_cdf_from_density(x1, p1)
    cdf2 = calculate_cdf_from_density(x2, p2)

    edges = checkerboard["edges"]
    copula_density = checkerboard["density"]
    bins = checkerboard["bins"]

    index1 = np.searchsorted(edges, cdf1, side="right") - 1
    index2 = np.searchsorted(edges, cdf2, side="right") - 1
    index1 = np.clip(index1, 0, bins - 1)
    index2 = np.clip(index2, 0, bins - 1)

    copula_on_grid = copula_density[np.ix_(index2, index1)]
    joint_density = copula_on_grid * p2[:, None] * p1[None, :]

    normalization = np.trapezoid(np.trapezoid(joint_density, x1, axis=1), x2)
    joint_density /= normalization

    if enforce_marginals:
        joint_density = balance_joint_density_marginals(joint_density, x1, p1, x2, p2)

    marginal_x1, marginal_x2 = calculate_copula_marginals(joint_density, x1, x2)
    error_x1 = np.trapezoid(np.abs(marginal_x1 - p1), x1)
    error_x2 = np.trapezoid(np.abs(marginal_x2 - p2), x2)
    moments = calculate_joint_density_moments(joint_density, x1, x2)

    print(f"Checkerboard stationary L1 marginal error X1: {error_x1}")
    print(f"Checkerboard stationary L1 marginal error X2: {error_x2}")
    print(f"Checkerboard stationary Pearson correlation: {moments['correlation']}")
    print(f"Checkerboard stationary covariance: {moments['covariance']}")

    return {
        "joint_density": joint_density,
        "copula_on_grid": copula_on_grid,
        "marginal_x1": marginal_x1,
        "marginal_x2": marginal_x2,
        "p1": p1,
        "p2": p2,
        "cdf1": cdf1,
        "cdf2": cdf2,
        "moments": moments,
        "error_x1": error_x1,
        "error_x2": error_x2,
    }

def centers_to_edges(centers):
    """Convert bin centers to the corresponding bin edges."""
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def build_empirical_2d_density(data1_array, data2_array, x1, x2, chunk_size=2_000_000):
    """Estimate an empirical two-dimensional density on the requested grid."""
    x_edges = centers_to_edges(x1)
    y_edges = centers_to_edges(x2)
    counts = np.zeros((len(x2), len(x1)), dtype=float)

    data1_flat = np.asarray(data1_array).ravel()
    data2_flat = np.asarray(data2_array).ravel()

    if data1_flat.shape != data2_flat.shape:
        raise ValueError("data1_array and data2_array must have the same shape.")

    for start in range(0, len(data1_flat), chunk_size):
        end = min(start + chunk_size, len(data1_flat))
        x1_chunk = data1_flat[start:end]
        x2_chunk = data2_flat[start:end]
        valid = np.isfinite(x1_chunk) & np.isfinite(x2_chunk)

        chunk_counts, _, _ = np.histogram2d(
            x2_chunk[valid],
            x1_chunk[valid],
            bins=[y_edges, x_edges],
        )
        counts += chunk_counts

    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    area = dy[:, None] * dx[None, :]

    retained = counts.sum()
    density = counts / retained / area

    print(f"Empirical 2D observations inside grid: {int(retained)}")
    print(f"Empirical 2D density normalization: {np.sum(density * area)}")

    return density

def evaluate_bernstein_copula(u, v, probability):
    """Evaluate the Bernstein copula at specified probability coordinates."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    probability = np.asarray(probability, dtype=float)

    m = probability.shape[0]
    if probability.shape != (m, m):
        raise ValueError("Probability matrix must be square.")

    indices = np.arange(1, m + 1)
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    v = np.clip(v, 1e-10, 1.0 - 1e-10)

    basis_u = beta.pdf(u[None, :], indices[:, None], (m + 1 - indices)[:, None])
    basis_v = beta.pdf(v[None, :], indices[:, None], (m + 1 - indices)[:, None])

    return basis_v.T @ probability @ basis_u


def build_bernstein_copula_grid(checkerboard, grid_size=200):
    """Construct the Bernstein copula density grid."""
    probability = checkerboard["probability"]
    grid = np.linspace(0.005, 0.995, grid_size)
    density = evaluate_bernstein_copula(grid, grid, probability)

    du = grid[1] - grid[0]
    normalization = np.sum(density) * du ** 2

    print(f"Bernstein copula degree: {probability.shape[0]}")
    print(f"Bernstein grid normalization: {normalization}")

    return {"grid": grid, "density": density, "probability": probability}


def build_bernstein_stationary_density(x1, p1, x2, p2, checkerboard):
    """Combine Bernstein copula density with stationary marginals."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    p1 = normalize_1d_density(x1, p1)
    p2 = normalize_1d_density(x2, p2)

    cdf1 = calculate_cdf_from_density(x1, p1)
    cdf2 = calculate_cdf_from_density(x2, p2)

    copula_density = evaluate_bernstein_copula(cdf1, cdf2, checkerboard["probability"])
    joint_density = copula_density * p2[:, None] * p1[None, :]

    normalization = np.trapezoid(np.trapezoid(joint_density, x1, axis=1), x2)
    joint_density /= normalization

    marginal_x1, marginal_x2 = calculate_copula_marginals(joint_density, x1, x2)
    error_x1 = np.trapezoid(np.abs(marginal_x1 - p1), x1)
    error_x2 = np.trapezoid(np.abs(marginal_x2 - p2), x2)
    moments = calculate_joint_density_moments(joint_density, x1, x2)

    print(f"Bernstein stationary normalization: {normalization}")
    print(f"Bernstein L1 marginal error X1: {error_x1}")
    print(f"Bernstein L1 marginal error X2: {error_x2}")
    print(f"Bernstein Pearson correlation: {moments['correlation']}")
    print(f"Bernstein covariance: {moments['covariance']}")

    return {
        "joint_density": joint_density,
        "copula_density": copula_density,
        "marginal_x1": marginal_x1,
        "marginal_x2": marginal_x2,
        "p1": p1,
        "p2": p2,
        "cdf1": cdf1,
        "cdf2": cdf2,
        "moments": moments,
        "error_x1": error_x1,
        "error_x2": error_x2,
    }


# --- Copula Quintiles ---
# Joint quantile-class probabilities from cached copula fits (no ERA5 reload).
#
# The repository stores densities with rows v (latent), columns u (sensible).
# This module exports matrices with rows u, columns v, as in the manuscript.

import csv
import json
from pathlib import Path

from scipy.integrate import quad
from scipy.stats import beta, norm, t


def _edges(k):
    """Compute edges for copula from ``k``."""
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError('classes must be an integer >= 2')
    return np.linspace(0., 1., k + 1)


def _check_matrix(p, k, *, uniform=True, atol=2e-8):
    """Validate a joint probability matrix."""
    p = np.asarray(p, dtype=float)
    if p.shape != (k, k) or not np.isfinite(p).all() or p.min() < -atol:
        raise ValueError('Invalid rectangle probabilities')
    p = np.maximum(p, 0.)  # round-off only; never rebalance a fitted model
    if not np.isclose(p.sum(), 1., atol=atol, rtol=0.):
        raise ValueError(f'Probability mass {p.sum()} != 1')
    if uniform and (not np.allclose(p.sum(0), 1/k, atol=atol, rtol=0.) or
                    not np.allclose(p.sum(1), 1/k, atol=atol, rtol=0.)):
        raise ValueError('Copula does not have uniform class marginals')
    return p


def _cell_masses(density, centers, *, atol=1e-9):
    """Read the uniform histogram/checkerboard convention in copula_workflows."""
    density = np.asarray(density, dtype=float)
    centers = np.asarray(centers, dtype=float)
    m = centers.size
    if centers.ndim != 1 or m < 1 or density.shape != (m, m):
        raise ValueError('Unexpected density/centers shapes')
    if not np.allclose(centers, (np.arange(m) + .5)/m, atol=atol, rtol=0.):
        raise ValueError('Expected midpoint grid on [0,1]')
    p = density.T / m**2  # transpose: repository v,u -> manuscript u,v
    if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1., atol=atol, rtol=0.):
        raise ValueError('Invalid saved cell masses')
    return p


def histogram_classes(cell_mass, k=5):
    """Exact integration of a piecewise constant density, even for m % k != 0."""
    p = np.asarray(cell_mass, dtype=float)
    m = p.shape[0]
    if p.shape != (m, m):
        raise ValueError('Expected square mass matrix')
    edges, fine = _edges(k), np.linspace(0., 1., m+1)
    overlap = np.maximum(0., np.minimum(edges[1:, None], fine[None, 1:]) -
                         np.maximum(edges[:-1, None], fine[None, :-1])) * m
    return overlap @ p @ overlap.T


def bernstein_classes(cell_mass, k=5):
    """Integrate normalized beta bases exactly via beta CDF differences."""
    p = np.asarray(cell_mass, dtype=float)
    m = p.shape[0]
    if p.shape != (m, m):
        raise ValueError('Expected square mass matrix')
    i = np.arange(1, m+1)
    b = np.diff(beta.cdf(_edges(k)[:, None], i[None, :], (m+1-i)[None, :]), axis=0)
    return b @ p @ b.T


def elliptical_classes(rho, k=5, *, df=None):
    """Integrate Gaussian or Student-t rectangles via 1D conditional CDF.

    With df=None, conditional normal; otherwise conditional t_(df+1).
    Integration over probability u avoids arbitrary cutoffs near 0 and 1.
    """
    rho = float(rho)
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError('Expected |rho| < 1')
    if df is not None and (not np.isfinite(df) or df <= 0):
        raise ValueError('Student degrees of freedom must be positive')
    e = _edges(k)
    z = norm.ppf(e) if df is None else t.ppf(e, df)
    scale = np.sqrt(1-rho*rho)
    p = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            def conditional(u):
                """Compute conditional for copula from ``u``."""
                x = norm.ppf(u) if df is None else t.ppf(u, df)
                s = scale if df is None else scale*np.sqrt((df+x*x)/(df+1))
                lo, hi = (z[j]-rho*x)/s, (z[j+1]-rho*x)/s
                dist = norm if df is None else t(df+1)
                # Avoid subtracting two values close to one in the right tail.
                return float(dist.sf(lo)-dist.sf(hi) if lo > 0 else dist.cdf(hi)-dist.cdf(lo))
            p[i,j] = quad(conditional, e[i], e[i+1], epsabs=1e-11,
                          epsrel=1e-9, limit=300)[0]
    return _check_matrix(p, k)


def _metrics(p, empirical):
    """Compute metrics for copula from ``p``, ``empirical``."""
    k = len(p)
    return dict(p11=float(p[0,0]), pkk=float(p[-1,-1]), p1k=float(p[0,-1]),
                pk1=float(p[-1,0]), same_class=float(np.trace(p)),
                tv_independence=float(.5*np.abs(p-1/k**2).sum()),
                tv_empirical=float(.5*np.abs(p-empirical).sum()),
                row_sums=p.sum(axis=1).tolist(), column_sums=p.sum(axis=0).tolist())


def _trapz(y, x, axis=-1):
    # NumPy 1.x and 2.x compatibility.
    """Integrate sampled values by the trapezoidal rule."""
    fn = getattr(np, 'trapezoid', None)
    return (fn or np.trapz)(y, x, axis=axis)


def _moments(x, p):
    """Compute the moments used in the current diagnostic."""
    x, p = np.asarray(x, float), np.asarray(p, float)
    if (x.ndim != 1 or p.shape != x.shape or len(x) < 2 or
            not np.isfinite(x).all() or not np.isfinite(p).all() or
            np.any(np.diff(x) <= 0) or np.any(p < 0)):
        raise ValueError('Invalid marginal grid/density')
    mass = float(_trapz(p, x))
    if mass <= 0:
        raise ValueError('Zero marginal mass')
    p = p/mass
    mu = float(_trapz(x*p, x))
    var = float(_trapz((x-mu)**2*p, x))
    return dict(mass=mass, mean=mu, variance=var, sd=float(np.sqrt(var)),
                grid_min=float(x[0]), grid_max=float(x[-1]), nodes=int(len(x)))


def audit_marginal_moments(root, names, period, data, *, hybrid_reference=(1979, 2026)):
    """Compare FP, GPD hybrid, interpolated FP and integrated cached joint margins."""
    root = Path(root)
    first, last = period
    base = root/'Functional'/'Stationary'
    fp = base/f'{names[0]}-{names[1]}_{first}-{last}.npz'
    out = {'fp_source': str(fp), 'fp': {}, 'fp_on_joint_grid': {},
           'hybrid': {}, 'joint': {}, 'missing': []}
    if fp.is_file():
        with np.load(fp, allow_pickle=False) as f:
            for i, name in enumerate(names, 1):
                x, pdf = f['grid'], f[f'pdf_{name}']
                out['fp'][name] = _moments(x, pdf)
                gx = data[f'x{i}']
                out['fp_on_joint_grid'][name] = _moments(gx, np.interp(gx, x, pdf))
    else:
        out['missing'].append(str(fp))
    for name in names:
        h = base/'Hybrid'/f'{name}_{first}-{last}_gpd_reference_{hybrid_reference[0]}-{hybrid_reference[1]}.npz'
        if h.is_file():
            with np.load(h, allow_pickle=False) as f:
                out['hybrid'][name] = dict(source=str(h), **_moments(f['grid'], f['pdf']))
        else:
            out['missing'].append(str(h))
    for model in ('gaussian', 'checkerboard', 'bernstein'):
        key = f'joint_{model}'
        if key not in data.files:
            continue
        x, y, pdf = data['x1'], data['x2'], data[key]
        if pdf.shape != (len(y), len(x)):
            raise ValueError(f'Invalid joint grid: {key}')
        px, py = _trapz(pdf, y, axis=0), _trapz(pdf, x, axis=1)
        out['joint'][model] = {names[0]: _moments(x, px), names[1]: _moments(y, py)}
    return out


_NAMES_RU = {'empirical': 'Эмпирическая', 'gaussian': 'Гауссова',
             'student_t': 'Стьюдента', 'checkerboard': 'Шахматная', 'bernstein': 'Бернштейна'}


def _tex_results(path, matrices, summaries, period, k):
    """A directly insertable article fragment; all displayed values in probabilities."""
    def fmt(x):
        """Format the supplied value for diagnostic output."""
        return '$'+f'{x:.4f}'.replace('.', '{,}')+'$'
    lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
             rf'\caption{{Матрицы совместных вероятностей за {period[0]}--{period[1]} гг. Строки соответствуют классам явного, столбцы --- скрытого потока.}}',
             r'\label{tab:joint_quintile_matrix}', r'\setlength{\tabcolsep}{4pt}']
    # All matrices rather than just the preferred one. Keep floats modest for k=5.
    for model in ('empirical', 'checkerboard', 'bernstein'):
        lines += [r'\par\smallskip\textit{'+_NAMES_RU[model]+r'}\par\smallskip',
                  r'\begin{tabular}{@{}l'+'r'*k+r'@{}}', r'\toprule',
                  '$i/j$ & '+' & '.join(map(str,range(1,k+1)))+r' \\', r'\midrule']
        for i, row in enumerate(matrices[model], 1):
            lines += [str(i)+' & '+' & '.join(fmt(x) for x in row)+r' \\']
        lines += [r'\bottomrule', r'\end{tabular}']
    lines += [r'\end{table}', r'\begin{table}[htbp]', r'\centering', r'\small',
              rf'\caption{{Характеристики совместных классов за {period[0]}--{period[1]} гг. $D_{{\mathrm{{TV}}}}^{{({k})}}$ --- расстояние до эмпирической матрицы.}}',
              r'\label{tab:joint_quintile_summary}', r'\setlength{\tabcolsep}{4pt}',
              r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
              rf'Модель & $\pi_{{11}}$ & $\pi_{{{k}{k}}}$ & $\pi_{{1{k}}}$ & $\pi_{{{k}1}}$ & $\sum_i\pi_{{ii}}$ & $D_{{\mathrm{{TV}}}}^{{({k})}}$ \\', r'\midrule']
    for model, s in summaries.items():
        lines += [_NAMES_RU[model]+' & '+' & '.join(fmt(s[z]) for z in ('p11','pkk','p1k','pk1','same_class','tv_empirical'))+r' \\']
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def estimate_joint_quantile_classes(files_path_prefix, start_year, end_year, *,
                                    names=('sensible','latent'), classes=5,
                                    hybrid_reference=(1979,2026), out_dir=None):
    """Read the NPZ/JSON created by task='copulas'; export CSV, NPZ, JSON and TeX."""
    names = tuple(names)
    if len(names) != 2:
        raise ValueError('Expected two variables')
    _edges(classes)
    period = (int(start_year), int(end_year))
    root = Path(files_path_prefix)
    tag = f'{names[0]}-{names[1]}_{period[0]}-{period[1]}'
    source = root/'Functional'/'Copulas'/f'{tag}.npz'
    meta = source.with_suffix('.json')
    if not source.is_file() or not meta.is_file():
        raise FileNotFoundError(f"Run main.py task='copulas' for {period} first: {source}, {meta}")
    fit = json.loads(meta.read_text(encoding='utf-8'))
    if tuple(fit['pair']) != names or fit['period'] != f'{period[0]}-{period[1]}':
        raise ValueError('Saved copula identity does not match requested pair/period')
    with np.load(source, allow_pickle=False) as data:
        empirical_mass = _cell_masses(data['empirical'], data['centers'])
        checker_mass = _cell_masses(data['checkerboard'], data['checkerboard_centers'])
        # Checkerboard balancing is the basis of BOTH nonparametric models.
        _check_matrix(checker_mass, len(checker_mass))
        matrices = {'empirical': _check_matrix(histogram_classes(empirical_mass, classes), classes, uniform=False),
                    'gaussian': elliptical_classes(fit['gaussian_rho'], classes),
                    'student_t': elliptical_classes(fit['student_t_rho'], classes, df=fit['student_t_df']),
                    'checkerboard': _check_matrix(histogram_classes(checker_mass, classes), classes),
                    'bernstein': _check_matrix(bernstein_classes(checker_mass, classes), classes)}
        audit = audit_marginal_moments(root, names, period, data, hybrid_reference=hybrid_reference)
    summaries = {model: _metrics(p, matrices['empirical']) for model,p in matrices.items()}
    target = Path(out_dir) if out_dir is not None else root/'Functional'/'Copulas'/'Quintiles'/f'{tag}_k{classes}'
    target.mkdir(parents=True, exist_ok=True)
    for model, p in matrices.items():
        with (target/f'{model}.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['row_class']+list(range(1,classes+1)))
            for i, row in enumerate(p,1):
                w.writerow([i]+list(row))
    np.savez_compressed(target/'matrices.npz', edges=_edges(classes), **matrices)
    payload = dict(pair=list(names), period=list(period), classes=classes, source=str(source),
                   source_json=str(meta), axes={'rows':names[0], 'columns':names[1]},
                   empirical='integrated saved rank histogram (not direct physical-class counts)',
                   empirical_bins=len(empirical_mass), checkerboard_bins=len(checker_mass),
                   gaussian_rho=fit['gaussian_rho'], student_t_rho=fit['student_t_rho'],
                   student_t_df=fit['student_t_df'], independence_probability=1/classes**2,
                   matrices={m:p.tolist() for m,p in matrices.items()}, metrics=summaries,
                   cached_joint_moments=fit.get('moments', {}), marginal_audit=audit)
    (target/'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if names == ('sensible','latent') and classes == 5:
        _tex_results(target/'joint_quintiles_results.tex', matrices, summaries, period, classes)
    for model, s in summaries.items():
        print(f"{model:12s} P11={s['p11']:.6f} Pkk={s['pkk']:.6f} "
              f"same={s['same_class']:.6f} TV_emp={s['tv_empirical']:.6f}")
    print(f'Joint quantile results: {target}')
    return target


# --- Copula Workflows ---
# Calculate empirical and parametric copulas independently from rendering.



from Data_processing.io import load_field


def estimate_copula_characteristics(files_path_prefix, start_year, end_year,
                                     *, names=("sensible", "latent"),
                                     empirical_bins=60, checkerboard_bins=50,
                                     sample_size=300000, random_seed=12345,
                                     joint_grid_points=400):
    """Estimate and store all supported copulas and dependence metrics.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory containing original fields and stationary density fits.
    start_year, end_year : int
        Requested data period.
    names : tuple of str
        Variables of the joint distribution.
    empirical_bins : int
        Binning for the empirical copula and Gaussian/Student comparisons.
    checkerboard_bins : int
        Binning for checkerboard and Bernstein constructions.
    sample_size : int
        Deterministic valid paired sample size.
    random_seed : int
        Random generator seed.
    joint_grid_points : int
        Resolution for each coordinate of joint stationary density grids.

    Returns
    -------
    pathlib.Path
        NPZ file containing copula and stationary joint density grids.
    """

    root = Path(files_path_prefix)
    fields = [load_field(root, name, start_year, end_year) for name in names]
    stationary = root / "Functional" / "Stationary" / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    if not stationary.is_file():
        raise FileNotFoundError(f"Calculate stationary densities first: {stationary}")
    with np.load(stationary) as data:
        source_grid = data["grid"]
        grids = [np.linspace(source_grid[0], source_grid[-1], joint_grid_points)] * 2
        densities = [np.interp(grids[i], source_grid, data[f"pdf_{name}"])
                     for i, name in enumerate(names)]
    dependence = estimate_kendall_dependence(*fields, sample_size=sample_size, random_seed=random_seed)
    empirical = build_empirical_copula_density(dependence["sampled_x1"],
                                               dependence["sampled_x2"], bins=empirical_bins)
    centers = empirical["centers"]
    gaussian = build_gaussian_copula_density_grid(dependence["rho"], centers)
    student_fit = fit_student_t_copula(empirical["u"], empirical["v"], dependence["rho"])
    student = build_student_t_copula_density_grid(student_fit["rho"], student_fit["df"], centers)
    checkerboard = build_checkerboard_copula(empirical["u"], empirical["v"], bins=checkerboard_bins)
    bernstein = build_bernstein_copula_grid(checkerboard, grid_size=200)
    gaussian_joint = build_gaussian_copula_stationary_density(*(
        grids[0], densities[0], grids[1], densities[1], dependence["rho"]))
    checkerboard_joint = build_checkerboard_stationary_density(
        grids[0], densities[0], grids[1], densities[1], checkerboard)
    bernstein_joint = build_bernstein_stationary_density(
        grids[0], densities[0], grids[1], densities[1], checkerboard)
    payload = {"centers": centers, "empirical": empirical["density"],
               "gaussian": gaussian, "student_t": student,
               "checkerboard_centers": checkerboard["centers"],
               "checkerboard": checkerboard["density"],
               "bernstein_grid": bernstein["grid"], "bernstein": bernstein["density"],
               "x1": grids[0], "x2": grids[1],
               "joint_gaussian": gaussian_joint["joint_density"],
               "joint_checkerboard": checkerboard_joint["joint_density"],
               "joint_bernstein": bernstein_joint["joint_density"]}
    summary = {"pair": list(names), "period": f"{start_year}-{end_year}",
               "kendall_tau": float(dependence["tau"]), "gaussian_rho": float(dependence["rho"]),
               "student_t_rho": float(student_fit["rho"]),
               "student_t_df": float(student_fit["df"]),
               "tail_dependence": calculate_student_t_tail_dependence(student_fit["rho"], student_fit["df"]),
               "gaussian_distance": compare_copula_densities(empirical["density"], gaussian, centers),
               "student_t_distance": compare_copula_densities(empirical["density"], student, centers),
               "moments": {"gaussian": calculate_joint_density_moments(gaussian_joint["joint_density"], *grids),
                           "checkerboard": checkerboard_joint["moments"],
                           "bernstein": bernstein_joint["moments"]}}
    output = root / "Functional" / "Copulas"
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    np.savez_compressed(target, **payload)
    target.with_suffix(".json").write_text(json.dumps(summary, indent=2,
                                                      default=lambda x: float(x) if isinstance(x, np.number) else str(x)),
                                           encoding="utf-8")
    return target
