"""Class-free Bernstein-beta approximation of a 2-D stationary density.

Inputs are compatible with the earlier Fokker-Planck code:
    quantiles1, quantiles2, stationary_density_direct, fp_system,
    files_path_prefix, data1_name, data2_name.

The fitted analytical density is

p(x1,x2) = 1/(d1*d2) * sum_ij w_ij beta_{i,m}(u) beta_{j,n}(v),

u=(x1-L1)/d1, v=(x2-L2)/d2, w_ij>=0, sum_ij w_ij=1.
Thus non-negativity and normalization are guaranteed analytically.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize
from scipy.special import betainc, gammaln


def coordinates_to_edges(coordinates: np.ndarray, mesh_size: int, name: str) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {coordinates.shape}.")
    if coordinates.size == mesh_size + 1:
        edges = coordinates.copy()
    elif coordinates.size == mesh_size:
        if mesh_size < 2:
            raise ValueError(f"At least two {name} centers are required.")
        edges = np.empty(mesh_size + 1, dtype=float)
        edges[1:-1] = 0.5 * (coordinates[:-1] + coordinates[1:])
        edges[0] = coordinates[0] - 0.5 * (coordinates[1] - coordinates[0])
        edges[-1] = coordinates[-1] + 0.5 * (coordinates[-1] - coordinates[-2])
    else:
        raise ValueError(
            f"{name} has length {coordinates.size}; expected {mesh_size} centers "
            f"or {mesh_size + 1} edges."
        )
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must contain finite, strictly increasing values.")
    return edges


def cell_centers(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def prepare_density_grid(
    quantiles1: np.ndarray,
    quantiles2: np.ndarray,
    density_mesh: np.ndarray,
) -> dict:
    density = np.asarray(density_mesh, dtype=float)
    if density.ndim != 2:
        raise ValueError(f"density_mesh must be 2-D, got {density.shape}.")
    if np.any(~np.isfinite(density)) or np.any(density < 0):
        raise ValueError("density_mesh must be finite and non-negative.")

    ny, nx = density.shape
    x1_edges = coordinates_to_edges(quantiles1, nx, "quantiles1")
    x2_edges = coordinates_to_edges(quantiles2, ny, "quantiles2")
    x1_centers = cell_centers(x1_edges)
    x2_centers = cell_centers(x2_edges)
    cell_area = np.outer(np.diff(x2_edges), np.diff(x1_edges))

    total = float(np.sum(density * cell_area))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("The numerical density has invalid total probability.")

    density = density / total
    probability_mass = density * cell_area
    return {
        "density": density,
        "probability_mass": probability_mass,
        "cell_area": cell_area,
        "x1_edges": x1_edges,
        "x2_edges": x2_edges,
        "x1_centers": x1_centers,
        "x2_centers": x2_centers,
        "domain": (
            float(x1_edges[0]), float(x1_edges[-1]),
            float(x2_edges[0]), float(x2_edges[-1]),
        ),
        "shape": (ny, nx),
        "original_total_probability": total,
    }


def normalized_beta_basis_density(u: np.ndarray, degree: int) -> np.ndarray:
    """Columns are Beta(i+1, degree-i+1) densities on [0,1]."""
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or np.any((u < 0) | (u > 1)):
        raise ValueError("u must be a one-dimensional array inside [0,1].")

    i = np.arange(degree + 1, dtype=float)
    log_c = (
        np.log(degree + 1.0)
        + gammaln(degree + 1.0)
        - gammaln(i + 1.0)
        - gammaln(degree - i + 1.0)
    )
    tiny = np.finfo(float).tiny
    us = np.clip(u, tiny, 1.0 - tiny)
    basis = np.empty((u.size, degree + 1), dtype=float)
    for k in range(degree + 1):
        basis[:, k] = np.exp(
            log_c[k] + k * np.log(us) + (degree - k) * np.log1p(-us)
        )
    at_zero = u == 0.0
    at_one = u == 1.0
    if np.any(at_zero):
        basis[at_zero, :] = 0.0
        basis[at_zero, 0] = degree + 1.0
    if np.any(at_one):
        basis[at_one, :] = 0.0
        basis[at_one, -1] = degree + 1.0
    return basis


def normalized_beta_basis_cell_mass(edges: np.ndarray, degree: int) -> np.ndarray:
    """Exact basis probability in each interval between normalized edges."""
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or np.any((edges < 0) | (edges > 1)):
        raise ValueError("Normalized edges must be one-dimensional and inside [0,1].")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Normalized edges must be strictly increasing.")
    i = np.arange(degree + 1)[None, :]
    alpha = i + 1
    beta_par = degree - i + 1
    return (
        betainc(alpha, beta_par, edges[1:, None])
        - betainc(alpha, beta_par, edges[:-1, None])
    )


def build_bernstein_basis_matrices(
    quantiles1: np.ndarray,
    quantiles2: np.ndarray,
    density_shape: tuple[int, int],
    degree1: int,
    degree2: int,
) -> dict:
    ny, nx = density_shape
    x1_edges = coordinates_to_edges(quantiles1, nx, "quantiles1")
    x2_edges = coordinates_to_edges(quantiles2, ny, "quantiles2")
    x1_centers = cell_centers(x1_edges)
    x2_centers = cell_centers(x2_edges)

    l1, u1 = float(x1_edges[0]), float(x1_edges[-1])
    l2, u2 = float(x2_edges[0]), float(x2_edges[-1])
    d1, d2 = u1 - l1, u2 - l2

    uc = (x1_centers - l1) / d1
    vc = (x2_centers - l2) / d2
    ue = (x1_edges - l1) / d1
    ve = (x2_edges - l2) / d2

    b1 = normalized_beta_basis_density(uc, degree1) / d1
    b2 = normalized_beta_basis_density(vc, degree2) / d2
    m1 = normalized_beta_basis_cell_mass(ue, degree1)
    m2 = normalized_beta_basis_cell_mass(ve, degree2)

    # Weight matrix shape: (degree1+1, degree2+1); second index varies fastest.
    density_design = np.einsum("xi,yj->yxij", b1, b2, optimize=True).reshape(
        ny * nx, (degree1 + 1) * (degree2 + 1), order="C"
    )
    cell_mass_design = np.einsum("xi,yj->yxij", m1, m2, optimize=True).reshape(
        ny * nx, (degree1 + 1) * (degree2 + 1), order="C"
    )
    if not np.allclose(cell_mass_design.sum(axis=0), 1.0, rtol=1e-11, atol=1e-12):
        raise RuntimeError("The exact cell-mass basis is not normalized.")

    return {
        "density_design": density_design,
        "cell_mass_design": cell_mass_design,
        "degree1": degree1,
        "degree2": degree2,
        "domain": (l1, u1, l2, u2),
        "shape": (ny, nx),
        "number_of_weights": (degree1 + 1) * (degree2 + 1),
        "x1_edges": x1_edges,
        "x2_edges": x2_edges,
        "x1_centers": x1_centers,
        "x2_centers": x2_centers,
    }


def numerical_density_moments(
    density_mesh: np.ndarray,
    cell_area: np.ndarray,
    x1_centers: np.ndarray,
    x2_centers: np.ndarray,
) -> dict:
    x1_grid, x2_grid = np.meshgrid(x1_centers, x2_centers)
    mass = np.asarray(density_mesh) * np.asarray(cell_area)
    mass = mass / mass.sum()
    mean1 = float(np.sum(x1_grid * mass))
    mean2 = float(np.sum(x2_grid * mass))
    second1 = float(np.sum(x1_grid**2 * mass))
    second2 = float(np.sum(x2_grid**2 * mass))
    cross = float(np.sum(x1_grid * x2_grid * mass))
    var1 = second1 - mean1**2
    var2 = second2 - mean2**2
    cov = cross - mean1 * mean2
    corr = cov / np.sqrt(var1 * var2) if var1 > 0 and var2 > 0 else np.nan
    return {
        "mean_x1": mean1,
        "mean_x2": mean2,
        "second_x1": second1,
        "second_x2": second2,
        "cross": cross,
        "variance_x1": var1,
        "variance_x2": var2,
        "standard_deviation_x1": np.sqrt(max(var1, 0.0)),
        "standard_deviation_x2": np.sqrt(max(var2, 0.0)),
        "covariance_x1_x2": cov,
        "correlation_x1_x2": corr,
    }


def build_analytic_moment_vectors(
    degree1: int,
    degree2: int,
    domain: tuple[float, float, float, float],
) -> dict:
    l1, u1, l2, u2 = domain
    d1, d2 = u1 - l1, u2 - l2
    i = np.arange(degree1 + 1, dtype=float)
    j = np.arange(degree2 + 1, dtype=float)
    eu = (i + 1.0) / (degree1 + 2.0)
    ev = (j + 1.0) / (degree2 + 2.0)
    eu2 = (i + 1.0) * (i + 2.0) / ((degree1 + 2.0) * (degree1 + 3.0))
    ev2 = (j + 1.0) * (j + 2.0) / ((degree2 + 2.0) * (degree2 + 3.0))
    ex1 = l1 + d1 * eu
    ex2 = l2 + d2 * ev
    ex12 = l1**2 + 2.0 * l1 * d1 * eu + d1**2 * eu2
    ex22 = l2**2 + 2.0 * l2 * d2 * ev + d2**2 * ev2
    shape = (degree1 + 1, degree2 + 1)
    return {
        "mean_x1": np.broadcast_to(ex1[:, None], shape).ravel(order="C"),
        "mean_x2": np.broadcast_to(ex2[None, :], shape).ravel(order="C"),
        "second_x1": np.broadcast_to(ex12[:, None], shape).ravel(order="C"),
        "second_x2": np.broadcast_to(ex22[None, :], shape).ravel(order="C"),
        "cross": (ex1[:, None] * ex2[None, :]).ravel(order="C"),
    }


def analytic_moments_from_weights(weights: np.ndarray, vectors: Mapping[str, np.ndarray]) -> dict:
    w = np.asarray(weights, dtype=float).ravel()
    mean1 = float(vectors["mean_x1"] @ w)
    mean2 = float(vectors["mean_x2"] @ w)
    second1 = float(vectors["second_x1"] @ w)
    second2 = float(vectors["second_x2"] @ w)
    cross = float(vectors["cross"] @ w)
    var1 = second1 - mean1**2
    var2 = second2 - mean2**2
    cov = cross - mean1 * mean2
    corr = cov / np.sqrt(var1 * var2) if var1 > 0 and var2 > 0 else np.nan
    return {
        "mean_x1": mean1,
        "mean_x2": mean2,
        "second_x1": second1,
        "second_x2": second2,
        "cross": cross,
        "variance_x1": var1,
        "variance_x2": var2,
        "standard_deviation_x1": np.sqrt(max(var1, 0.0)),
        "standard_deviation_x2": np.sqrt(max(var2, 0.0)),
        "covariance_x1_x2": cov,
        "correlation_x1_x2": corr,
    }


def build_second_difference_matrix(degree1: int, degree2: int) -> sp.csr_matrix:
    n1, n2 = degree1 + 1, degree2 + 1
    n = n1 * n2
    rows, cols, vals = [], [], []
    r = 0
    for i in range(1, n1 - 1):
        for j in range(n2):
            idx = [(i - 1) * n2 + j, i * n2 + j, (i + 1) * n2 + j]
            rows.extend([r, r, r]); cols.extend(idx); vals.extend([1.0, -2.0, 1.0]); r += 1
    for i in range(n1):
        for j in range(1, n2 - 1):
            idx = [i * n2 + j - 1, i * n2 + j, i * n2 + j + 1]
            rows.extend([r, r, r]); cols.extend(idx); vals.extend([1.0, -2.0, 1.0]); r += 1
    return sp.csr_matrix((vals, (rows, cols)), shape=(r, n))


def initialize_bernstein_weights(
    cell_mass_design: np.ndarray,
    target_probability_mass: np.ndarray,
    method: str = "bounded_least_squares",
) -> np.ndarray:
    n = cell_mass_design.shape[1]
    if method == "uniform":
        return np.full(n, 1.0 / n)
    if method != "bounded_least_squares":
        raise ValueError("method must be 'uniform' or 'bounded_least_squares'.")
    fit = lsq_linear(
        np.asarray(cell_mass_design),
        np.asarray(target_probability_mass).ravel(),
        bounds=(0.0, np.inf),
        lsmr_tol="auto",
        verbose=0,
    )
    w = np.maximum(np.asarray(fit.x, dtype=float), 0.0)
    return w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)


def build_exact_linear_constraints(
    number_of_weights: int,
    moment_vectors: Mapping[str, np.ndarray],
    target_moments: Mapping[str, float],
    exact_moment_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    allowed = {"mean_x1", "mean_x2", "second_x1", "second_x2", "cross"}
    rows = [np.ones(number_of_weights, dtype=float)]
    values = [1.0]
    for name in exact_moment_names:
        if name not in allowed:
            raise ValueError(f"Unsupported exact moment name: {name}.")
        rows.append(np.asarray(moment_vectors[name], dtype=float))
        values.append(float(target_moments[name]))
    return np.vstack(rows), np.asarray(values, dtype=float)


def build_fit_objective(
    density_design: np.ndarray,
    cell_mass_design: np.ndarray,
    target_density: np.ndarray,
    target_probability_mass: np.ndarray,
    cell_area: np.ndarray,
    loss_type: str,
    smoothness_matrix: sp.csr_matrix,
    lambda_smooth: float,
    fp_operator: Optional[sp.spmatrix],
    lambda_fp: float,
    moment_vectors: Mapping[str, np.ndarray],
    target_moments: Mapping[str, float],
    moment_penalty_weights: Optional[Mapping[str, float]],
    epsilon: float,
):
    phi = np.asarray(density_design, dtype=float)
    mass_design = np.asarray(cell_mass_design, dtype=float)
    p_target = np.asarray(target_density, dtype=float).ravel()
    q_target = np.asarray(target_probability_mass, dtype=float).ravel()
    area = np.asarray(cell_area, dtype=float).ravel()

    allowed_losses = {"kl_mass", "hellinger_mass", "weighted_l2_density"}
    if loss_type not in allowed_losses:
        raise ValueError(f"loss_type must be one of {sorted(allowed_losses)}.")
    if lambda_smooth < 0 or lambda_fp < 0:
        raise ValueError("Penalty weights cannot be negative.")

    density_scale = max(float(np.sum(area * p_target**2)), epsilon)
    fp_design = None
    fp_scale = 1.0
    if fp_operator is not None and lambda_fp > 0:
        fp_operator = fp_operator.tocsr()
        if fp_operator.shape[0] != phi.shape[0]:
            raise ValueError("fp_operator is incompatible with the density grid.")
        fp_design = np.asarray(fp_operator @ phi, dtype=float)
        rate = max(float(np.max(np.abs(fp_operator.diagonal()))), epsilon)
        fp_scale = max(rate**2 * float(np.sum(area * p_target**2)), epsilon)

    penalties = dict(moment_penalty_weights or {})
    scales = {
        "mean_x1": max(float(target_moments["standard_deviation_x1"]), epsilon),
        "mean_x2": max(float(target_moments["standard_deviation_x2"]), epsilon),
        "second_x1": max(abs(float(target_moments["second_x1"])), epsilon),
        "second_x2": max(abs(float(target_moments["second_x2"])), epsilon),
        "cross": max(abs(float(target_moments["cross"])), epsilon),
    }

    def objective_and_gradient(w: np.ndarray) -> tuple[float, np.ndarray]:
        w = np.asarray(w, dtype=float)
        value = 0.0
        grad = np.zeros_like(w)

        if loss_type == "weighted_l2_density":
            diff = phi @ w - p_target
            value += 0.5 * float(np.sum(area * diff**2)) / density_scale
            grad += phi.T @ (area * diff) / density_scale
        else:
            q = mass_design @ w
            qs = np.maximum(q, epsilon)
            if loss_type == "kl_mass":
                positive = q_target > 0
                value += float(np.sum(q_target[positive] * np.log(q_target[positive] / qs[positive])))
                dq = np.zeros_like(qs)
                dq[positive] = -q_target[positive] / qs[positive]
                grad += mass_design.T @ dq
            else:
                sqrt_target = np.sqrt(np.maximum(q_target, 0.0))
                sqrt_q = np.sqrt(qs)
                diff = sqrt_q - sqrt_target
                value += 0.5 * float(np.sum(diff**2))
                grad += mass_design.T @ (0.5 * (1.0 - sqrt_target / sqrt_q))

        if lambda_smooth > 0 and smoothness_matrix.shape[0] > 0:
            rough = smoothness_matrix @ w
            value += 0.5 * lambda_smooth * float(rough @ rough)
            grad += lambda_smooth * np.asarray(smoothness_matrix.T @ rough).ravel()

        if fp_design is not None:
            residual = fp_design @ w
            value += 0.5 * lambda_fp * float(np.sum(area * residual**2)) / fp_scale
            grad += lambda_fp * fp_design.T @ (area * residual) / fp_scale

        for name, weight in penalties.items():
            if weight <= 0:
                continue
            if name not in moment_vectors:
                raise ValueError(f"Unsupported moment penalty name: {name}.")
            vector = np.asarray(moment_vectors[name], dtype=float)
            diff = (float(vector @ w) - float(target_moments[name])) / scales[name]
            value += 0.5 * weight * diff**2
            grad += weight * diff * vector / scales[name]

        return float(value), grad

    return (
        lambda w: objective_and_gradient(w)[0],
        lambda w: objective_and_gradient(w)[1],
    )


def calculate_fit_metrics(
    numerical_density: np.ndarray,
    analytical_density: np.ndarray,
    numerical_probability_mass: np.ndarray,
    analytical_probability_mass: np.ndarray,
    cell_area: np.ndarray,
    epsilon: float = 1e-15,
) -> dict:
    p_num = np.asarray(numerical_density, dtype=float)
    p_app = np.asarray(analytical_density, dtype=float)
    q_num = np.asarray(numerical_probability_mass, dtype=float)
    q_app = np.asarray(analytical_probability_mass, dtype=float)
    area = np.asarray(cell_area, dtype=float)
    positive = q_num > 0
    q_safe = np.maximum(q_app, epsilon)
    return {
        "l1_density": float(np.sum(np.abs(p_app - p_num) * area)),
        "l2_density": float(np.sqrt(np.sum((p_app - p_num) ** 2 * area))),
        "total_variation": 0.5 * float(np.sum(np.abs(q_app - q_num))),
        "kl_numerical_to_analytical": float(
            np.sum(q_num[positive] * np.log(q_num[positive] / q_safe[positive]))
        ),
        "hellinger_squared": 0.5 * float(
            np.sum((np.sqrt(np.maximum(q_num, 0.0)) - np.sqrt(np.maximum(q_app, 0.0))) ** 2)
        ),
        "maximum_density_error": float(np.max(np.abs(p_app - p_num))),
    }


def calculate_fp_residual_diagnostics(density_mesh: np.ndarray, fp_system: dict) -> dict:
    density = np.asarray(density_mesh, dtype=float)
    if density.shape != tuple(fp_system["shape"]):
        raise ValueError("density_mesh shape does not match fp_system.")
    operator = fp_system["operator"].tocsr()
    area = np.asarray(fp_system["cell_area"], dtype=float)
    area_vector = area.ravel(order="C")
    vector = density.ravel(order="C")
    residual = operator @ vector
    diagonal_rate = max(float(np.max(np.abs(operator.diagonal()))), np.finfo(float).tiny)
    row_rate = max(
        float(np.max(np.asarray(np.abs(operator).sum(axis=1)).ravel())),
        np.finfo(float).tiny,
    )
    density_scale = max(float(np.max(np.abs(vector))), np.finfo(float).tiny)
    residual_inf = float(np.max(np.abs(residual)))
    residual_l1 = float(area_vector @ np.abs(residual))
    return {
        "residual_mesh": residual.reshape(density.shape, order="C"),
        "residual_inf": residual_inf,
        "residual_l1": residual_l1,
        "relative_residual_inf": residual_inf / (row_rate * density_scale),
        "relative_residual_l1": residual_l1 / diagonal_rate,
    }


def fit_bernstein_stationary_density(
    quantiles1: np.ndarray,
    quantiles2: np.ndarray,
    stationary_density_direct: np.ndarray,
    degree1: int = 10,
    degree2: int = 10,
    fp_system: Optional[dict] = None,
    loss_type: str = "kl_mass",
    lambda_smooth: float = 1e-4,
    lambda_fp: float = 0.0,
    moment_penalty_weights: Optional[Mapping[str, float]] = None,
    exact_moment_names: Sequence[str] = (),
    initialization: str = "bounded_least_squares",
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
    epsilon: float = 1e-15,
    verbose: bool = True,
) -> dict:
    """Fit a non-negative, exactly normalized Bernstein-beta density."""
    if degree1 < 0 or degree2 < 0:
        raise ValueError("Degrees must be non-negative.")

    grid = prepare_density_grid(quantiles1, quantiles2, stationary_density_direct)
    basis = build_bernstein_basis_matrices(
        quantiles1, quantiles2, grid["shape"], degree1, degree2
    )
    numerical_moments = numerical_density_moments(
        grid["density"], grid["cell_area"], grid["x1_centers"], grid["x2_centers"]
    )
    moment_vectors = build_analytic_moment_vectors(degree1, degree2, basis["domain"])
    smoothness = build_second_difference_matrix(degree1, degree2)
    initial_weights = initialize_bernstein_weights(
        basis["cell_mass_design"], grid["probability_mass"], method=initialization
    )

    fp_operator = None
    if fp_system is not None:
        if "operator" not in fp_system:
            raise ValueError("fp_system does not contain 'operator'.")
        if tuple(fp_system["shape"]) != tuple(grid["shape"]):
            raise ValueError("fp_system shape does not match the density mesh.")
        fp_operator = fp_system["operator"]

    objective, gradient = build_fit_objective(
        density_design=basis["density_design"],
        cell_mass_design=basis["cell_mass_design"],
        target_density=grid["density"],
        target_probability_mass=grid["probability_mass"],
        cell_area=grid["cell_area"],
        loss_type=loss_type,
        smoothness_matrix=smoothness,
        lambda_smooth=lambda_smooth,
        fp_operator=fp_operator,
        lambda_fp=lambda_fp,
        moment_vectors=moment_vectors,
        target_moments=numerical_moments,
        moment_penalty_weights=moment_penalty_weights,
        epsilon=epsilon,
    )

    equality_matrix, equality_values = build_exact_linear_constraints(
        basis["number_of_weights"],
        moment_vectors,
        numerical_moments,
        exact_moment_names,
    )
    constraint = LinearConstraint(equality_matrix, equality_values, equality_values)
    bounds = Bounds(
        np.zeros(basis["number_of_weights"]),
        np.ones(basis["number_of_weights"]),
    )

    optimization = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=[constraint],
        options={"maxiter": int(max_iterations), "ftol": float(tolerance), "disp": bool(verbose)},
    )

    weights = np.maximum(np.asarray(optimization.x, dtype=float), 0.0)
    if weights.sum() <= 0:
        raise RuntimeError("The optimizer returned zero total weight.")
    weights /= weights.sum()

    density_raw = (basis["density_design"] @ weights).reshape(grid["shape"], order="C")
    center_norm = float(np.sum(density_raw * grid["cell_area"]))
    density_discrete = density_raw / center_norm
    mass_mesh = (basis["cell_mass_design"] @ weights).reshape(grid["shape"], order="C")
    analytic_moments = analytic_moments_from_weights(weights, moment_vectors)
    metrics = calculate_fit_metrics(
        grid["density"], density_discrete,
        grid["probability_mass"], mass_mesh,
        grid["cell_area"], epsilon,
    )
    fp_diagnostics = (
        calculate_fp_residual_diagnostics(density_discrete, fp_system)
        if fp_system is not None else None
    )

    result = {
        "weights": weights,
        "weight_matrix": weights.reshape(degree1 + 1, degree2 + 1, order="C"),
        "degree1": degree1,
        "degree2": degree2,
        "domain": basis["domain"],
        "density_mesh_analytic": density_raw,
        "density_mesh_discrete_normalized": density_discrete,
        "probability_mass_mesh": mass_mesh,
        "numerical_density": grid["density"],
        "numerical_probability_mass": grid["probability_mass"],
        "cell_area": grid["cell_area"],
        "x1_edges": grid["x1_edges"],
        "x2_edges": grid["x2_edges"],
        "x1_centers": grid["x1_centers"],
        "x2_centers": grid["x2_centers"],
        "analytic_normalization": float(mass_mesh.sum()),
        "center_quadrature_normalization": center_norm,
        "numerical_moments": numerical_moments,
        "analytic_moments": analytic_moments,
        "metrics": metrics,
        "fp_diagnostics": fp_diagnostics,
        "optimizer": {
            "success": bool(optimization.success),
            "status": int(optimization.status),
            "message": str(optimization.message),
            "iterations": int(getattr(optimization, "nit", -1)),
            "function_evaluations": int(getattr(optimization, "nfev", -1)),
            "objective": float(optimization.fun),
            "maximum_constraint_error": float(
                np.max(np.abs(equality_matrix @ weights - equality_values))
            ),
        },
        "settings": {
            "loss_type": loss_type,
            "lambda_smooth": float(lambda_smooth),
            "lambda_fp": float(lambda_fp),
            "moment_penalty_weights": dict(moment_penalty_weights or {}),
            "exact_moment_names": list(exact_moment_names),
            "initialization": initialization,
        },
    }
    if verbose:
        print_fit_summary(result)
    return result


def evaluate_bernstein_density(
    x1: np.ndarray | float,
    x2: np.ndarray | float,
    weights: np.ndarray,
    degree1: int,
    degree2: int,
    domain: tuple[float, float, float, float],
    outside_value: float = 0.0,
) -> np.ndarray:
    """Evaluate the fitted analytical density at arbitrary broadcastable points."""
    l1, u1, l2, u2 = domain
    d1, d2 = u1 - l1, u2 - l2
    x1a, x2a = np.broadcast_arrays(np.asarray(x1, float), np.asarray(x2, float))
    output = np.full(x1a.shape, outside_value, dtype=float)
    inside = (x1a >= l1) & (x1a <= u1) & (x2a >= l2) & (x2a <= u2)
    if not np.any(inside):
        return output
    uu = ((x1a[inside] - l1) / d1).ravel()
    vv = ((x2a[inside] - l2) / d2).ravel()
    b1 = normalized_beta_basis_density(uu, degree1) / d1
    b2 = normalized_beta_basis_density(vv, degree2) / d2
    wm = np.asarray(weights, float).reshape(degree1 + 1, degree2 + 1, order="C")
    values = np.einsum("ni,ij,nj->n", b1, wm, b2, optimize=True)
    output[inside] = values
    return output


def calculate_probability_current_from_fp_system(
    density_mesh: np.ndarray,
    fp_system: dict,
) -> dict:
    """Use the same discrete face-current operators as the earlier solver."""
    density = np.asarray(density_mesh, dtype=float)
    ny, nx = fp_system["shape"]
    if density.shape != (ny, nx):
        raise ValueError(f"density_mesh has shape {density.shape}; expected {(ny, nx)}.")
    vector = density.ravel(order="C")
    j1_faces = (fp_system["current_x_operator"] @ vector).reshape(ny, nx + 1, order="C")
    j2_faces = (fp_system["current_y_operator"] @ vector).reshape(ny + 1, nx, order="C")
    j1 = 0.5 * (j1_faces[:, :-1] + j1_faces[:, 1:])
    j2 = 0.5 * (j2_faces[:-1, :] + j2_faces[1:, :])
    divergence = (
        fp_system["divergence_x_operator"] @ j1_faces.ravel(order="C")
        + fp_system["divergence_y_operator"] @ j2_faces.ravel(order="C")
    ).reshape(ny, nx, order="C")
    magnitude = np.hypot(j1, j2)
    area = np.asarray(fp_system["cell_area"], dtype=float)
    return {
        "J1": j1,
        "J2": j2,
        "J1_faces": j1_faces,
        "J2_faces": j2_faces,
        "magnitude": magnitude,
        "divergence": divergence,
        "diagnostics": {
            "maximum_abs_divergence": float(np.max(np.abs(divergence))),
            "area_weighted_divergence_l1": float(np.sum(np.abs(divergence) * area)),
            "maximum_current_magnitude": float(np.max(magnitude)),
            "mean_current_magnitude": float(np.sum(magnitude * area) / np.sum(area)),
            "left_boundary_max_abs_flux": float(np.max(np.abs(j1_faces[:, 0]))),
            "right_boundary_max_abs_flux": float(np.max(np.abs(j1_faces[:, -1]))),
            "bottom_boundary_max_abs_flux": float(np.max(np.abs(j2_faces[0, :]))),
            "top_boundary_max_abs_flux": float(np.max(np.abs(j2_faces[-1, :]))),
        },
    }


def compare_probability_currents(
    numerical_density: np.ndarray,
    analytical_density: np.ndarray,
    fp_system: dict,
) -> dict:
    numerical = calculate_probability_current_from_fp_system(numerical_density, fp_system)
    analytical = calculate_probability_current_from_fp_system(analytical_density, fp_system)
    dj1 = analytical["J1"] - numerical["J1"]
    dj2 = analytical["J2"] - numerical["J2"]
    dm = np.hypot(dj1, dj2)
    area = np.asarray(fp_system["cell_area"], dtype=float)
    denominator = max(float(np.sum(numerical["magnitude"] * area)), np.finfo(float).tiny)
    return {
        "numerical_current": numerical,
        "analytical_current": analytical,
        "difference_J1": dj1,
        "difference_J2": dj2,
        "difference_magnitude": dm,
        "area_weighted_current_l1": float(np.sum(dm * area)),
        "relative_area_weighted_current_l1": float(np.sum(dm * area) / denominator),
        "maximum_current_difference": float(np.max(dm)),
    }


def scan_bernstein_degrees(
    quantiles1: np.ndarray,
    quantiles2: np.ndarray,
    stationary_density_direct: np.ndarray,
    degree_pairs: Iterable[tuple[int, int]],
    fp_system: Optional[dict] = None,
    loss_type: str = "kl_mass",
    lambda_smooth: float = 1e-4,
    lambda_fp: float = 0.0,
    moment_penalty_weights: Optional[Mapping[str, float]] = None,
    exact_moment_names: Sequence[str] = (),
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    results, summary = [], []
    for degree1, degree2 in degree_pairs:
        if verbose:
            print(f"\nFitting degrees ({degree1}, {degree2})")
        result = fit_bernstein_stationary_density(
            quantiles1=quantiles1,
            quantiles2=quantiles2,
            stationary_density_direct=stationary_density_direct,
            degree1=degree1,
            degree2=degree2,
            fp_system=fp_system,
            loss_type=loss_type,
            lambda_smooth=lambda_smooth,
            lambda_fp=lambda_fp,
            moment_penalty_weights=moment_penalty_weights,
            exact_moment_names=exact_moment_names,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=False,
        )
        results.append(result)
        row = {
            "degree1": degree1,
            "degree2": degree2,
            "number_of_weights": (degree1 + 1) * (degree2 + 1),
            "optimizer_success": result["optimizer"]["success"],
            "objective": result["optimizer"]["objective"],
            **result["metrics"],
        }
        if result["fp_diagnostics"] is not None:
            row["fp_relative_residual_inf"] = result["fp_diagnostics"]["relative_residual_inf"]
            row["fp_residual_l1"] = result["fp_diagnostics"]["residual_l1"]
        summary.append(row)
        if verbose:
            print(
                f"TV={row['total_variation']:.6g}, "
                f"KL={row['kl_numerical_to_analytical']:.6g}, "
                f"H2={row['hellinger_squared']:.6g}"
            )
    return results, summary


def print_fit_summary(result: dict) -> None:
    print("\nBernstein-beta stationary-density fit")
    print("--------------------------------------")
    print("Degrees:", result["degree1"], result["degree2"])
    print("Weights:", result["weights"].size)
    print("Optimizer success:", result["optimizer"]["success"])
    print("Optimizer message:", result["optimizer"]["message"])
    print("Analytical normalization:", result["analytic_normalization"])
    print("Center quadrature normalization:", result["center_quadrature_normalization"])
    for name, value in result["metrics"].items():
        print(f"{name}: {value}")
    print("\nMoment comparison")
    for name in (
        "mean_x1", "mean_x2", "variance_x1", "variance_x2",
        "covariance_x1_x2", "correlation_x1_x2",
    ):
        print(
            f"{name}: numerical={result['numerical_moments'][name]}, "
            f"analytical={result['analytic_moments'][name]}"
        )
    if result["fp_diagnostics"] is not None:
        print("\nFokker-Planck residual")
        for name, value in result["fp_diagnostics"].items():
            if name != "residual_mesh":
                print(f"{name}: {value}")


def _json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def build_latex_formula_template(result: dict, data1_name: str, data2_name: str) -> str:
    l1, u1, l2, u2 = result["domain"]
    m, n = result["degree1"], result["degree2"]
    return rf"""% Bernstein-beta approximation for {data1_name} and {data2_name}
% The full matrix w_{{ij}} is stored in the accompanying CSV file.

\[
u=\frac{{x_1-({l1:.16g})}}{{({u1:.16g})-({l1:.16g})}},
\qquad
v=\frac{{x_2-({l2:.16g})}}{{({u2:.16g})-({l2:.16g})}}.
\]

\[
\beta_{{i,m}}(u)=(m+1)\binom{{m}}{{i}}u^i(1-u)^{{m-i}}.
\]

\[
p_{{{m},{n}}}(x_1,x_2)=
\frac{{1}}{{[({u1:.16g})-({l1:.16g})][({u2:.16g})-({l2:.16g})]}}
\sum_{{i=0}}^{{{m}}}\sum_{{j=0}}^{{{n}}}
w_{{ij}}\beta_{{i,{m}}}(u)\beta_{{j,{n}}}(v),
\]

\[
w_{{ij}}\ge 0,
\qquad
\sum_{{i=0}}^{{{m}}}\sum_{{j=0}}^{{{n}}}w_{{ij}}=1.
\]
"""


def save_bernstein_fit(
    files_path_prefix: str,
    result: dict,
    data1_name: str,
    data2_name: str,
    subdirectory: str = "videos/2D/bernstein_fit",
) -> dict:
    output_dir = Path(files_path_prefix) / subdirectory
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = f"{data1_name}-{data2_name}".replace(" ", "_").replace("/", "_")

    weights_path = output_dir / f"{safe}_bernstein_weights.csv"
    with weights_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["i", "j", "weight"])
        for i in range(result["degree1"] + 1):
            for j in range(result["degree2"] + 1):
                writer.writerow([i, j, result["weight_matrix"][i, j]])

    arrays_path = output_dir / f"{safe}_bernstein_fit.npz"
    np.savez_compressed(
        arrays_path,
        weights=result["weights"],
        weight_matrix=result["weight_matrix"],
        density_mesh_analytic=result["density_mesh_analytic"],
        density_mesh_discrete_normalized=result["density_mesh_discrete_normalized"],
        probability_mass_mesh=result["probability_mass_mesh"],
        numerical_density=result["numerical_density"],
        numerical_probability_mass=result["numerical_probability_mass"],
        x1_edges=result["x1_edges"],
        x2_edges=result["x2_edges"],
        x1_centers=result["x1_centers"],
        x2_centers=result["x2_centers"],
    )

    diagnostics = {
        "degree1": result["degree1"],
        "degree2": result["degree2"],
        "domain": result["domain"],
        "analytic_normalization": result["analytic_normalization"],
        "center_quadrature_normalization": result["center_quadrature_normalization"],
        "optimizer": result["optimizer"],
        "settings": result["settings"],
        "metrics": result["metrics"],
        "numerical_moments": result["numerical_moments"],
        "analytic_moments": result["analytic_moments"],
        "fp_diagnostics": (
            {k: v for k, v in result["fp_diagnostics"].items() if k != "residual_mesh"}
            if result["fp_diagnostics"] is not None else None
        ),
    }
    diagnostics_path = output_dir / f"{safe}_bernstein_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(_json_compatible(diagnostics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    formula_path = output_dir / f"{safe}_bernstein_formula.tex"
    formula_path.write_text(
        build_latex_formula_template(result, data1_name, data2_name),
        encoding="utf-8",
    )
    return {
        "weights_csv": str(weights_path),
        "arrays_npz": str(arrays_path),
        "diagnostics_json": str(diagnostics_path),
        "formula_tex": str(formula_path),
    }


def plot_bernstein_fit(
    files_path_prefix: str,
    quantiles1: np.ndarray,
    quantiles2: np.ndarray,
    stationary_density_direct: np.ndarray,
    fit_result: dict,
    data1_name: str,
    data2_name: str,
    subdirectory: str = "videos/2D/bernstein_fit",
    dpi: int = 300,
) -> None:
    import matplotlib.pyplot as plt

    output_dir = Path(files_path_prefix) / subdirectory
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = f"{data1_name}-{data2_name}".replace(" ", "_").replace("/", "_")
    numerical = prepare_density_grid(quantiles1, quantiles2, stationary_density_direct)["density"]
    analytical = fit_result["density_mesh_discrete_normalized"]
    signed = analytical - numerical
    absolute = np.abs(signed)

    specs = (
        (numerical, "Numerical stationary probability density", "Stationary probability density", f"{safe}_numerical_density.png"),
        (analytical, "Bernstein-beta analytical approximation", "Stationary probability density", f"{safe}_bernstein_density.png"),
        (signed, "Analytical minus numerical density", "Density difference", f"{safe}_signed_difference.png"),
        (absolute, "Absolute density approximation error", "Absolute density error", f"{safe}_absolute_difference.png"),
    )
    for values, title, label, filename in specs:
        fig, ax = plt.subplots(figsize=(16, 16))
        mesh = ax.pcolormesh(quantiles1, quantiles2, values, shading="auto")
        fig.colorbar(mesh, ax=ax, label=label)
        ax.set_xlabel(data1_name)
        ax.set_ylabel(data2_name)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_current_comparison(
    files_path_prefix: str,
    current_comparison: dict,
    fp_system: dict,
    data1_name: str,
    data2_name: str,
    arrow_step: int = 2,
    subdirectory: str = "videos/2D/bernstein_fit",
    dpi: int = 300,
) -> None:
    import matplotlib.pyplot as plt

    output_dir = Path(files_path_prefix) / subdirectory
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = f"{data1_name}-{data2_name}".replace(" ", "_").replace("/", "_")
    xg, yg = np.meshgrid(fp_system["x_centers"], fp_system["y_centers"])
    sl = (slice(None, None, arrow_step), slice(None, None, arrow_step))
    specs = (
        (
            current_comparison["numerical_current"]["J1"],
            current_comparison["numerical_current"]["J2"],
            current_comparison["numerical_current"]["magnitude"],
            "Numerical stationary probability current",
            f"{safe}_numerical_current.png",
        ),
        (
            current_comparison["analytical_current"]["J1"],
            current_comparison["analytical_current"]["J2"],
            current_comparison["analytical_current"]["magnitude"],
            "Current from the Bernstein-beta density",
            f"{safe}_bernstein_current.png",
        ),
        (
            current_comparison["difference_J1"],
            current_comparison["difference_J2"],
            current_comparison["difference_magnitude"],
            "Difference between analytical and numerical currents",
            f"{safe}_current_difference.png",
        ),
    )
    for j1, j2, magnitude, title, filename in specs:
        fig, ax = plt.subplots(figsize=(16, 16))
        mesh = ax.pcolormesh(
            fp_system["x_edges"], fp_system["y_edges"], magnitude, shading="auto"
        )
        fig.colorbar(mesh, ax=ax, label="Probability-current magnitude")
        ax.quiver(
            xg[sl], yg[sl], j1[sl], j2[sl],
            angles="xy", scale_units="xy", scale=None,
        )
        ax.set_xlabel(data1_name)
        ax.set_ylabel(data2_name)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def recommended_fit_example(
    quantiles1: np.ndarray,
    quantiles2: np.ndarray,
    stationary_density_direct: np.ndarray,
    fp_system: dict,
) -> dict:
    """A reasonable first run; tune penalties and degrees afterwards."""
    return fit_bernstein_stationary_density(
        quantiles1=quantiles1,
        quantiles2=quantiles2,
        stationary_density_direct=stationary_density_direct,
        degree1=10,
        degree2=10,
        fp_system=fp_system,
        loss_type="kl_mass",
        lambda_smooth=1e-4,
        lambda_fp=1e-2,
        moment_penalty_weights={
            "mean_x1": 10.0,
            "mean_x2": 10.0,
            "second_x1": 1.0,
            "second_x2": 1.0,
            "cross": 1.0,
        },
        exact_moment_names=(),
        max_iterations=5000,
        tolerance=1e-10,
        verbose=True,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Import this module and call fit_bernstein_stationary_density(...)."
    )
