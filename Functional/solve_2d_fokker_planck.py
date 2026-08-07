
"""
Class-free finite-volume solver for a stationary two-dimensional
Fokker-Planck equation with a full diffusion tensor.

State-space convention
----------------------
axis 0 -> x2
axis 1 -> x1

Input coefficient meshes:
    a1[i, j]  = a_1(x1_j, x2_i)
    a2[i, j]  = a_2(x1_j, x2_i)
    c11[i, j] = C_11(x1_j, x2_i)
    c12[i, j] = C_12(x1_j, x2_i)
    c22[i, j] = C_22(x1_j, x2_i)

The stationary PDE is

    0 = -dJ1/dx1 - dJ2/dx2,

with

    J1 = a1*p - 1/2 [d(C11*p)/dx1 + d(C12*p)/dx2]
    J2 = a2*p - 1/2 [d(C12*p)/dx1 + d(C22*p)/dx2].

Reflecting boundaries are imposed by setting the normal probability flux
to zero on every boundary face.

Dependencies:
    numpy
    scipy
    matplotlib  # only for the optional plotting helper
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _coordinates_to_edges(
    coordinates: np.ndarray,
    mesh_size: int,
    coordinate_name: str,
) -> np.ndarray:
    """
    Accept either cell edges or cell centers and return cell edges.

    If centers are supplied, the outer edges are extrapolated by half of
    the nearest center spacing.
    """
    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.ndim != 1:
        raise ValueError(
            f"{coordinate_name} must be one-dimensional, "
            f"got shape {coordinates.shape}."
        )

    if coordinates.size == mesh_size + 1:
        edges = coordinates.copy()
    elif coordinates.size == mesh_size:
        if mesh_size < 2:
            raise ValueError(
                f"At least two {coordinate_name} centers are required "
                "to reconstruct bin edges."
            )

        edges = np.empty(mesh_size + 1, dtype=float)
        edges[1:-1] = 0.5 * (
            coordinates[:-1] + coordinates[1:]
        )
        edges[0] = coordinates[0] - 0.5 * (
            coordinates[1] - coordinates[0]
        )
        edges[-1] = coordinates[-1] + 0.5 * (
            coordinates[-1] - coordinates[-2]
        )
    else:
        raise ValueError(
            f"{coordinate_name} has length {coordinates.size}. "
            f"Expected {mesh_size} centers or {mesh_size + 1} edges."
        )

    if np.any(~np.isfinite(edges)):
        raise ValueError(f"{coordinate_name} contains non-finite values.")

    if np.any(np.diff(edges) <= 0):
        raise ValueError(
            f"{coordinate_name} must be strictly increasing."
        )

    return edges


def _cell_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _validate_coefficient_meshes(
    a1_mesh: np.ndarray,
    a2_mesh: np.ndarray,
    c11_mesh: np.ndarray,
    c12_mesh: np.ndarray,
    c22_mesh: np.ndarray,
) -> tuple[np.ndarray, ...]:
    arrays = tuple(
        np.asarray(array, dtype=float)
        for array in (
            a1_mesh,
            a2_mesh,
            c11_mesh,
            c12_mesh,
            c22_mesh,
        )
    )

    shape = arrays[0].shape

    if len(shape) != 2:
        raise ValueError(
            f"Coefficient meshes must be two-dimensional, got {shape}."
        )

    names = ("a2", "C11", "C12", "C22")
    for name, array in zip(names, arrays[1:]):
        if array.shape != shape:
            raise ValueError(
                f"{name} has shape {array.shape}, expected {shape}."
            )

    finite = np.logical_and.reduce(
        [np.isfinite(array) for array in arrays]
    )

    if not np.all(finite):
        missing = int(np.size(finite) - np.count_nonzero(finite))
        raise ValueError(
            f"The coefficient meshes contain {missing} cells with NaN or "
            "infinite values. Fill/smooth them or solve only on a complete "
            "rectangular domain."
        )

    return arrays


def check_diffusion_tensor(
    c11_mesh: np.ndarray,
    c12_mesh: np.ndarray,
    c22_mesh: np.ndarray,
    relative_tolerance: float = 1e-12,
) -> dict:
    """
    Check symmetry-compatible positive definiteness of the 2x2 tensor C.

    C is represented by C11, C12, C22, with C21 = C12.
    """
    c11 = np.asarray(c11_mesh, dtype=float)
    c12 = np.asarray(c12_mesh, dtype=float)
    c22 = np.asarray(c22_mesh, dtype=float)

    if not (c11.shape == c12.shape == c22.shape):
        raise ValueError("C11, C12, and C22 must have identical shapes.")

    trace = c11 + c22
    discriminant = np.sqrt(
        np.maximum(
            (c11 - c22) ** 2 + 4.0 * c12 ** 2,
            0.0,
        )
    )

    lambda_max = 0.5 * (trace + discriminant)
    lambda_min = 0.5 * (trace - discriminant)

    scale = float(np.nanmedian(np.abs(trace)))
    floor = max(relative_tolerance * max(scale, 1.0), 0.0)

    positive_definite = (
        np.isfinite(lambda_min)
        & np.isfinite(lambda_max)
        & (lambda_min > floor)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        condition_number = lambda_max / lambda_min

    return {
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "condition_number": condition_number,
        "positive_definite_mask": positive_definite,
        "positive_definite_fraction": float(
            np.mean(positive_definite)
        ),
        "minimum_lambda": float(np.nanmin(lambda_min)),
        "maximum_condition_number": float(
            np.nanmax(condition_number[positive_definite])
        )
        if np.any(positive_definite)
        else np.inf,
    }


def _cell_derivative_x_matrix(
    x_centers: np.ndarray,
    ny: int,
) -> sp.csr_matrix:
    """
    Cell-centered x derivative.

    A simple centered difference is used internally and a one-sided
    difference at the two outer cell-center rows.
    """
    nx = x_centers.size
    n_cells = ny * nx

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    for i in range(ny):
        for j in range(nx):
            row = i * nx + j

            if nx == 1:
                continue

            if j == 0:
                spacing = x_centers[1] - x_centers[0]
                rows.extend((row, row))
                columns.extend((row, row + 1))
                values.extend((-1.0 / spacing, 1.0 / spacing))

            elif j == nx - 1:
                spacing = x_centers[-1] - x_centers[-2]
                rows.extend((row, row))
                columns.extend((row - 1, row))
                values.extend((-1.0 / spacing, 1.0 / spacing))

            else:
                spacing = x_centers[j + 1] - x_centers[j - 1]
                rows.extend((row, row))
                columns.extend((row - 1, row + 1))
                values.extend((-1.0 / spacing, 1.0 / spacing))

    return sp.csr_matrix(
        (values, (rows, columns)),
        shape=(n_cells, n_cells),
    )


def _cell_derivative_y_matrix(
    y_centers: np.ndarray,
    nx: int,
) -> sp.csr_matrix:
    """Cell-centered y derivative."""
    ny = y_centers.size
    n_cells = ny * nx

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    for i in range(ny):
        for j in range(nx):
            row = i * nx + j

            if ny == 1:
                continue

            if i == 0:
                spacing = y_centers[1] - y_centers[0]
                rows.extend((row, row))
                columns.extend((row, row + nx))
                values.extend((-1.0 / spacing, 1.0 / spacing))

            elif i == ny - 1:
                spacing = y_centers[-1] - y_centers[-2]
                rows.extend((row, row))
                columns.extend((row - nx, row))
                values.extend((-1.0 / spacing, 1.0 / spacing))

            else:
                spacing = y_centers[i + 1] - y_centers[i - 1]
                rows.extend((row, row))
                columns.extend((row - nx, row + nx))
                values.extend((-1.0 / spacing, 1.0 / spacing))

    return sp.csr_matrix(
        (values, (rows, columns)),
        shape=(n_cells, n_cells),
    )


def build_fokker_planck_operator(
    quantiles1: np.ndarray,
    quantiles2: np.ndarray,
    a1_mesh: np.ndarray,
    a2_mesh: np.ndarray,
    c11_mesh: np.ndarray,
    c12_mesh: np.ndarray,
    c22_mesh: np.ndarray,
    advection_scheme: str = "upwind",
) -> dict:
    """
    Build the sparse finite-volume Fokker-Planck operator.

    Parameters
    ----------
    quantiles1, quantiles2
        Either bin edges or bin centers for x1 and x2.

    a1_mesh, a2_mesh, c11_mesh, c12_mesh, c22_mesh
        Arrays of shape (n_x2, n_x1).

    advection_scheme
        "upwind"  : robust, but introduces some numerical diffusion.
        "central" : less numerical diffusion, but may produce oscillations
                    or negative densities if drift dominates diffusion.

    Returns
    -------
    Dictionary containing:
        operator
            Sparse matrix A satisfying dp/dt = A p.
        current_x_operator, current_y_operator
            Sparse operators mapping cell density to face currents.
        divergence_x_operator, divergence_y_operator
        x_edges, y_edges, x_centers, y_centers
        cell_area
    """
    (
        a1,
        a2,
        c11,
        c12,
        c22,
    ) = _validate_coefficient_meshes(
        a1_mesh,
        a2_mesh,
        c11_mesh,
        c12_mesh,
        c22_mesh,
    )

    if advection_scheme not in {"upwind", "central"}:
        raise ValueError(
            "advection_scheme must be 'upwind' or 'central'."
        )

    ny, nx = a1.shape
    n_cells = ny * nx

    x_edges = _coordinates_to_edges(
        quantiles1,
        nx,
        "quantiles1",
    )
    y_edges = _coordinates_to_edges(
        quantiles2,
        ny,
        "quantiles2",
    )

    x_centers = _cell_centers(x_edges)
    y_centers = _cell_centers(y_edges)

    cell_width_x = np.diff(x_edges)
    cell_width_y = np.diff(y_edges)
    cell_area = np.outer(cell_width_y, cell_width_x)

    derivative_x_cell = _cell_derivative_x_matrix(
        x_centers,
        ny,
    )
    derivative_y_cell = _cell_derivative_y_matrix(
        y_centers,
        nx,
    )

    # ---------------------------------------------------------------
    # Operators on x-normal faces.
    #
    # There are ny * (nx + 1) x-faces. Boundary rows remain zero,
    # which imposes J1 = 0 on the left and right boundaries.
    # ---------------------------------------------------------------
    n_x_faces = ny * (nx + 1)

    gradient_rows: list[int] = []
    gradient_columns: list[int] = []
    gradient_values: list[float] = []

    interpolation_rows: list[int] = []
    interpolation_columns: list[int] = []
    interpolation_values: list[float] = []

    advection_rows: list[int] = []
    advection_columns: list[int] = []
    advection_values: list[float] = []

    for i in range(ny):
        for face_j in range(1, nx):
            face_index = i * (nx + 1) + face_j
            left_index = i * nx + face_j - 1
            right_index = i * nx + face_j

            center_spacing = (
                x_centers[face_j]
                - x_centers[face_j - 1]
            )

            gradient_rows.extend(
                (face_index, face_index)
            )
            gradient_columns.extend(
                (left_index, right_index)
            )
            gradient_values.extend(
                (-1.0 / center_spacing, 1.0 / center_spacing)
            )

            interpolation_rows.extend(
                (face_index, face_index)
            )
            interpolation_columns.extend(
                (left_index, right_index)
            )
            interpolation_values.extend((0.5, 0.5))

            face_velocity = 0.5 * (
                a1[i, face_j - 1]
                + a1[i, face_j]
            )

            if advection_scheme == "upwind":
                source_index = (
                    left_index
                    if face_velocity >= 0
                    else right_index
                )

                advection_rows.append(face_index)
                advection_columns.append(source_index)
                advection_values.append(face_velocity)

            else:
                advection_rows.extend(
                    (face_index, face_index)
                )
                advection_columns.extend(
                    (left_index, right_index)
                )
                advection_values.extend(
                    (0.5 * face_velocity, 0.5 * face_velocity)
                )

    gradient_x_face = sp.csr_matrix(
        (
            gradient_values,
            (gradient_rows, gradient_columns),
        ),
        shape=(n_x_faces, n_cells),
    )

    interpolation_x_face = sp.csr_matrix(
        (
            interpolation_values,
            (interpolation_rows, interpolation_columns),
        ),
        shape=(n_x_faces, n_cells),
    )

    advection_x_face = sp.csr_matrix(
        (
            advection_values,
            (advection_rows, advection_columns),
        ),
        shape=(n_x_faces, n_cells),
    )

    # ---------------------------------------------------------------
    # Operators on y-normal faces.
    #
    # There are (ny + 1) * nx y-faces. Boundary rows remain zero,
    # imposing J2 = 0 on the bottom and top boundaries.
    # ---------------------------------------------------------------
    n_y_faces = (ny + 1) * nx

    gradient_rows = []
    gradient_columns = []
    gradient_values = []

    interpolation_rows = []
    interpolation_columns = []
    interpolation_values = []

    advection_rows = []
    advection_columns = []
    advection_values = []

    for face_i in range(1, ny):
        for j in range(nx):
            face_index = face_i * nx + j
            bottom_index = (face_i - 1) * nx + j
            top_index = face_i * nx + j

            center_spacing = (
                y_centers[face_i]
                - y_centers[face_i - 1]
            )

            gradient_rows.extend(
                (face_index, face_index)
            )
            gradient_columns.extend(
                (bottom_index, top_index)
            )
            gradient_values.extend(
                (-1.0 / center_spacing, 1.0 / center_spacing)
            )

            interpolation_rows.extend(
                (face_index, face_index)
            )
            interpolation_columns.extend(
                (bottom_index, top_index)
            )
            interpolation_values.extend((0.5, 0.5))

            face_velocity = 0.5 * (
                a2[face_i - 1, j]
                + a2[face_i, j]
            )

            if advection_scheme == "upwind":
                source_index = (
                    bottom_index
                    if face_velocity >= 0
                    else top_index
                )

                advection_rows.append(face_index)
                advection_columns.append(source_index)
                advection_values.append(face_velocity)

            else:
                advection_rows.extend(
                    (face_index, face_index)
                )
                advection_columns.extend(
                    (bottom_index, top_index)
                )
                advection_values.extend(
                    (0.5 * face_velocity, 0.5 * face_velocity)
                )

    gradient_y_face = sp.csr_matrix(
        (
            gradient_values,
            (gradient_rows, gradient_columns),
        ),
        shape=(n_y_faces, n_cells),
    )

    interpolation_y_face = sp.csr_matrix(
        (
            interpolation_values,
            (interpolation_rows, interpolation_columns),
        ),
        shape=(n_y_faces, n_cells),
    )

    advection_y_face = sp.csr_matrix(
        (
            advection_values,
            (advection_rows, advection_columns),
        ),
        shape=(n_y_faces, n_cells),
    )

    # ---------------------------------------------------------------
    # Face-to-cell divergence operators.
    # ---------------------------------------------------------------
    divergence_rows: list[int] = []
    divergence_columns: list[int] = []
    divergence_values: list[float] = []

    for i in range(ny):
        for j in range(nx):
            cell_index = i * nx + j
            left_face = i * (nx + 1) + j
            right_face = left_face + 1

            divergence_rows.extend(
                (cell_index, cell_index)
            )
            divergence_columns.extend(
                (left_face, right_face)
            )
            divergence_values.extend(
                (
                    -1.0 / cell_width_x[j],
                    1.0 / cell_width_x[j],
                )
            )

    divergence_x = sp.csr_matrix(
        (
            divergence_values,
            (divergence_rows, divergence_columns),
        ),
        shape=(n_cells, n_x_faces),
    )

    divergence_rows = []
    divergence_columns = []
    divergence_values = []

    for i in range(ny):
        for j in range(nx):
            cell_index = i * nx + j
            bottom_face = i * nx + j
            top_face = (i + 1) * nx + j

            divergence_rows.extend(
                (cell_index, cell_index)
            )
            divergence_columns.extend(
                (bottom_face, top_face)
            )
            divergence_values.extend(
                (
                    -1.0 / cell_width_y[i],
                    1.0 / cell_width_y[i],
                )
            )

    divergence_y = sp.csr_matrix(
        (
            divergence_values,
            (divergence_rows, divergence_columns),
        ),
        shape=(n_cells, n_y_faces),
    )

    diagonal_c11 = sp.diags(c11.ravel(order="C"))
    diagonal_c12 = sp.diags(c12.ravel(order="C"))
    diagonal_c22 = sp.diags(c22.ravel(order="C"))

    # J1 = a1*p - 1/2[d_x(C11*p) + d_y(C12*p)]
    current_x_operator = (
        advection_x_face
        - 0.5
        * (
            gradient_x_face @ diagonal_c11
            + interpolation_x_face
            @ derivative_y_cell
            @ diagonal_c12
        )
    ).tocsr()

    # J2 = a2*p - 1/2[d_x(C12*p) + d_y(C22*p)]
    current_y_operator = (
        advection_y_face
        - 0.5
        * (
            interpolation_y_face
            @ derivative_x_cell
            @ diagonal_c12
            + gradient_y_face @ diagonal_c22
        )
    ).tocsr()

    # dp/dt = -div(J)
    operator = -(
        divergence_x @ current_x_operator
        + divergence_y @ current_y_operator
    ).tocsr()

    # Conservative finite-volume check:
    # area^T A should be approximately zero.
    area_vector = cell_area.ravel(order="C")
    conservation_error = float(
        np.max(np.abs(area_vector @ operator))
    )

    return {
        "operator": operator,
        "current_x_operator": current_x_operator,
        "current_y_operator": current_y_operator,
        "divergence_x_operator": divergence_x,
        "divergence_y_operator": divergence_y,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "cell_area": cell_area,
        "shape": (ny, nx),
        "conservation_error": conservation_error,
        "advection_scheme": advection_scheme,
    }


def solve_stationary_density(
    fp_system: dict,
    initial_density: Optional[np.ndarray] = None,
    pseudo_time_step: Optional[float] = None,
    pseudo_time_factor: float = 50.0,
    tolerance: float = 1e-10,
    residual_tolerance: float = 1e-8,
    max_iterations: int = 20_000,
    maximum_allowed_negative_mass: float = 1e-6,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Obtain the stationary density by implicit pseudo-time relaxation.

    One iteration solves

        (I - dt*A) p_(n+1) = p_n.

    The converged solution satisfies A p = 0.

    `pseudo_time_step` is not the physical observation time step. It is a
    numerical relaxation parameter. A larger value normally converges faster,
    but can expose non-monotonicity of the discretized operator.

    Returns
    -------
    stationary_density, diagnostics
    """
    operator = fp_system["operator"].tocsr()
    cell_area = np.asarray(
        fp_system["cell_area"],
        dtype=float,
    )
    ny, nx = fp_system["shape"]
    n_cells = ny * nx

    area_vector = cell_area.ravel(order="C")

    if initial_density is None:
        density = np.ones(n_cells, dtype=float)
    else:
        initial_density = np.asarray(
            initial_density,
            dtype=float,
        )

        if initial_density.shape != (ny, nx):
            raise ValueError(
                f"initial_density has shape {initial_density.shape}, "
                f"expected {(ny, nx)}."
            )

        if np.any(~np.isfinite(initial_density)):
            raise ValueError(
                "initial_density contains non-finite values."
            )

        if np.any(initial_density < 0):
            raise ValueError(
                "initial_density must be non-negative."
            )

        density = initial_density.ravel(order="C").copy()

    mass = float(area_vector @ density)
    if mass <= 0:
        raise ValueError("Initial density has zero total mass.")

    density /= mass

    if pseudo_time_step is None:
        characteristic_rate = max(
            float(np.max(np.abs(operator.diagonal()))),
            1e-12,
        )
        pseudo_time_step = (
            pseudo_time_factor / characteristic_rate
        )

    if pseudo_time_step <= 0:
        raise ValueError("pseudo_time_step must be positive.")

    implicit_matrix = (
        sp.eye(n_cells, format="csc")
        - pseudo_time_step * operator.tocsc()
    )

    factorization = spla.splu(implicit_matrix)

    maximum_negative_mass = 0.0
    l1_change = np.inf

    for iteration in range(1, max_iterations + 1):
        new_density = factorization.solve(density)

        negative_mass = float(
            area_vector
            @ np.maximum(-new_density, 0.0)
        )
        maximum_negative_mass = max(
            maximum_negative_mass,
            negative_mass,
        )

        if negative_mass > maximum_allowed_negative_mass:
            raise RuntimeError(
                "The implicit iteration produced substantial negative "
                f"probability mass ({negative_mass:.3e}). Reduce "
                "pseudo_time_step, use the upwind scheme, smooth the "
                "coefficient fields, or refine the grid."
            )

        # Remove only numerical-scale negative values.
        new_density = np.maximum(new_density, 0.0)

        new_mass = float(area_vector @ new_density)
        if not np.isfinite(new_mass) or new_mass <= 0:
            raise RuntimeError(
                "Stationary iteration lost a finite positive total mass."
            )

        new_density /= new_mass

        l1_change = float(
            area_vector
            @ np.abs(new_density - density)
        )

        density = new_density

        if l1_change < tolerance:
            break

    density_mesh = density.reshape((ny, nx), order="C")

    residual = operator @ density
    residual_inf = float(np.max(np.abs(residual)))
    residual_l1 = float(
        area_vector @ np.abs(residual)
    )

    diagnostics = {
        "iterations": iteration,
        "converged_by_change": bool(l1_change < tolerance),
        "l1_change": l1_change,
        "residual_inf": residual_inf,
        "residual_l1": residual_l1,
        "residual_acceptable": bool(
            residual_inf < residual_tolerance
        ),
        "total_probability": float(
            area_vector @ density
        ),
        "minimum_density": float(
            np.min(density)
        ),
        "maximum_density": float(
            np.max(density)
        ),
        "pseudo_time_step": float(
            pseudo_time_step
        ),
        "maximum_negative_mass": float(
            maximum_negative_mass
        ),
        "conservation_error": float(
            fp_system["conservation_error"]
        ),
    }

    if verbose:
        for name, value in diagnostics.items():
            print(f"{name}: {value}")

    return density_mesh, diagnostics


def calculate_probability_current(
    stationary_density: np.ndarray,
    fp_system: dict,
) -> dict:
    """
    Calculate the stationary probability current using exactly the same
    discrete face-flux operators as the Fokker-Planck solver.

    Returns both face currents and cell-centered currents.
    """
    density = np.asarray(
        stationary_density,
        dtype=float,
    )

    ny, nx = fp_system["shape"]

    if density.shape != (ny, nx):
        raise ValueError(
            f"stationary_density has shape {density.shape}, "
            f"expected {(ny, nx)}."
        )

    density_vector = density.ravel(order="C")

    current_x_faces = (
        fp_system["current_x_operator"]
        @ density_vector
    ).reshape((ny, nx + 1), order="C")

    current_y_faces = (
        fp_system["current_y_operator"]
        @ density_vector
    ).reshape((ny + 1, nx), order="C")

    # Average adjacent face currents to obtain a vector at cell centers.
    current_1 = 0.5 * (
        current_x_faces[:, :-1]
        + current_x_faces[:, 1:]
    )

    current_2 = 0.5 * (
        current_y_faces[:-1, :]
        + current_y_faces[1:, :]
    )

    divergence = (
        fp_system["divergence_x_operator"]
        @ current_x_faces.ravel(order="C")
        + fp_system["divergence_y_operator"]
        @ current_y_faces.ravel(order="C")
    ).reshape((ny, nx), order="C")

    current_magnitude = np.hypot(
        current_1,
        current_2,
    )

    cell_area = fp_system["cell_area"]

    diagnostics = {
        "maximum_abs_divergence": float(
            np.max(np.abs(divergence))
        ),
        "area_weighted_divergence_l1": float(
            np.sum(np.abs(divergence) * cell_area)
        ),
        "maximum_current_magnitude": float(
            np.max(current_magnitude)
        ),
        "mean_current_magnitude": float(
            np.sum(current_magnitude * cell_area)
            / np.sum(cell_area)
        ),
        "left_boundary_max_abs_flux": float(
            np.max(np.abs(current_x_faces[:, 0]))
        ),
        "right_boundary_max_abs_flux": float(
            np.max(np.abs(current_x_faces[:, -1]))
        ),
        "bottom_boundary_max_abs_flux": float(
            np.max(np.abs(current_y_faces[0, :]))
        ),
        "top_boundary_max_abs_flux": float(
            np.max(np.abs(current_y_faces[-1, :]))
        ),
    }

    return {
        "J1": current_1,
        "J2": current_2,
        "magnitude": current_magnitude,
        "J1_faces": current_x_faces,
        "J2_faces": current_y_faces,
        "divergence": divergence,
        "diagnostics": diagnostics,
    }


def calculate_stationary_moments(
    stationary_density: np.ndarray,
    fp_system: dict,
) -> dict:
    """Calculate joint first and second moments."""
    density = np.asarray(
        stationary_density,
        dtype=float,
    )

    x1 = fp_system["x_centers"]
    x2 = fp_system["y_centers"]
    area = fp_system["cell_area"]

    x1_mesh, x2_mesh = np.meshgrid(x1, x2)

    weights = density * area
    probability = float(np.sum(weights))

    mean_x1 = float(
        np.sum(x1_mesh * weights) / probability
    )
    mean_x2 = float(
        np.sum(x2_mesh * weights) / probability
    )

    variance_x1 = float(
        np.sum((x1_mesh - mean_x1) ** 2 * weights)
        / probability
    )
    variance_x2 = float(
        np.sum((x2_mesh - mean_x2) ** 2 * weights)
        / probability
    )
    covariance = float(
        np.sum(
            (x1_mesh - mean_x1)
            * (x2_mesh - mean_x2)
            * weights
        )
        / probability
    )

    correlation = (
        covariance
        / np.sqrt(variance_x1 * variance_x2)
        if variance_x1 > 0 and variance_x2 > 0
        else np.nan
    )

    # Marginal densities. These integrate to one with the corresponding
    # one-dimensional cell widths.
    marginal_x1 = np.sum(
        density * np.diff(fp_system["y_edges"])[:, None],
        axis=0,
    )
    marginal_x2 = np.sum(
        density * np.diff(fp_system["x_edges"])[None, :],
        axis=1,
    )

    return {
        "mean_x1": mean_x1,
        "mean_x2": mean_x2,
        "variance_x1": variance_x1,
        "variance_x2": variance_x2,
        "standard_deviation_x1": np.sqrt(variance_x1),
        "standard_deviation_x2": np.sqrt(variance_x2),
        "covariance_x1_x2": covariance,
        "correlation_x1_x2": correlation,
        "marginal_x1": marginal_x1,
        "marginal_x2": marginal_x2,
    }


def save_density_and_current_plots(
    files_path_prefix: str,
    stationary_density: np.ndarray,
    current: dict,
    fp_system: dict,
    data1_name: str,
    data2_name: str,
    arrow_step: int = 2,
    normalize_current_arrows: bool = False,
    dpi: int = 300,
) -> None:
    """
    Optional plotting helper. Produces one density map and one current map.

    No custom colors are imposed.
    """
    import matplotlib.pyplot as plt

    output_directory = (
        Path(files_path_prefix)
        / "videos"
        / "2D"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_edges = fp_system["x_edges"]
    y_edges = fp_system["y_edges"]
    x_centers = fp_system["x_centers"]
    y_centers = fp_system["y_centers"]

    # Density.
    fig, ax = plt.subplots(figsize=(16, 16))

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        stationary_density,
        shading="auto",
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label="Stationary probability density",
    )

    ax.set_xlabel(data1_name)
    ax.set_ylabel(data2_name)
    ax.set_title(
        f"Stationary joint density of "
        f"{data1_name} and {data2_name}"
    )

    fig.tight_layout()
    fig.savefig(
        output_directory
        / f"{data1_name}-{data2_name}_stationary_density.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Current.
    fig, ax = plt.subplots(figsize=(16, 16))

    magnitude_mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        current["magnitude"],
        shading="auto",
    )

    fig.colorbar(
        magnitude_mesh,
        ax=ax,
        label="Probability-current magnitude",
    )

    x_grid, y_grid = np.meshgrid(
        x_centers,
        y_centers,
    )

    plot_slice = (
        slice(None, None, arrow_step),
        slice(None, None, arrow_step),
    )

    u = current["J1"][plot_slice].copy()
    v = current["J2"][plot_slice].copy()

    if normalize_current_arrows:
        magnitude = np.hypot(u, v)

        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.divide(
                u,
                magnitude,
                out=np.zeros_like(u),
                where=magnitude > 0,
            )
            v = np.divide(
                v,
                magnitude,
                out=np.zeros_like(v),
                where=magnitude > 0,
            )

    ax.quiver(
        x_grid[plot_slice],
        y_grid[plot_slice],
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=None,
    )

    ax.set_xlabel(data1_name)
    ax.set_ylabel(data2_name)
    ax.set_title(
        f"Stationary probability current for "
        f"{data1_name} and {data2_name}"
    )

    fig.tight_layout()
    fig.savefig(
        output_directory
        / f"{data1_name}-{data2_name}_stationary_current.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def solve_stationary_density_direct(
    fp_system: dict,
    negative_mass_tolerance: float = 1e-10,
    clip_tiny_negative_values: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Solve the stationary Fokker-Planck equation directly:

        A p = 0
        area.T @ p = 1

    using an augmented sparse system.

    Parameters
    ----------
    fp_system
        Output of build_fokker_planck_operator().

    negative_mass_tolerance
        Maximum accepted integrated negative probability mass.

    clip_tiny_negative_values
        Clip only numerical-scale negative values and renormalize.

    Returns
    -------
    stationary_density, diagnostics
    """
    operator = fp_system["operator"].tocsc()
    cell_area = np.asarray(
        fp_system["cell_area"],
        dtype=float,
    )

    ny, nx = fp_system["shape"]
    n_cells = ny * nx

    area_vector = cell_area.ravel(order="C")

    # Scale the normalization vector to improve conditioning.
    area_norm = np.linalg.norm(area_vector)

    if not np.isfinite(area_norm) or area_norm <= 0:
        raise ValueError("Invalid cell-area vector.")

    normalized_area = area_vector / area_norm

    normalized_area_column = sp.csc_matrix(
        normalized_area[:, np.newaxis]
    )

    normalized_area_row = sp.csc_matrix(
        normalized_area[np.newaxis, :]
    )

    zero_block = sp.csc_matrix((1, 1))

    # Augmented system:
    #
    # [ A   q ] [p     ] = [0      ]
    # [q.T  0 ] [lambda]   [1/||w||]
    #
    # where q = w / ||w|| and w contains cell areas.
    augmented_matrix = sp.bmat(
        [
            [operator, normalized_area_column],
            [normalized_area_row, zero_block],
        ],
        format="csc",
    )

    right_hand_side = np.zeros(
        n_cells + 1,
        dtype=float,
    )

    right_hand_side[-1] = 1.0 / area_norm

    solution = spla.spsolve(
        augmented_matrix,
        right_hand_side,
    )

    density_vector = np.asarray(
        solution[:-1],
        dtype=float,
    )

    lagrange_multiplier = float(solution[-1])

    if np.any(~np.isfinite(density_vector)):
        raise RuntimeError(
            "The direct sparse solve returned non-finite density values."
        )

    negative_mass_before_clipping = float(
        area_vector
        @ np.maximum(-density_vector, 0.0)
    )

    minimum_before_clipping = float(
        np.min(density_vector)
    )

    if (
        negative_mass_before_clipping
        > negative_mass_tolerance
    ):
        raise RuntimeError(
            "The direct stationary solution contains substantial negative "
            "probability mass: "
            f"{negative_mass_before_clipping:.6e}. "
            "This usually indicates a non-monotone spatial discretization, "
            "an ill-conditioned diffusion tensor, or an insufficient grid."
        )

    if clip_tiny_negative_values:
        density_vector = np.maximum(
            density_vector,
            0.0,
        )

    total_probability = float(
        area_vector @ density_vector
    )

    if not np.isfinite(total_probability) or total_probability <= 0:
        raise RuntimeError(
            "The direct solution has invalid total probability."
        )

    density_vector /= total_probability

    residual = operator @ density_vector

    residual_inf = float(
        np.max(np.abs(residual))
    )

    residual_l1 = float(
        area_vector @ np.abs(residual)
    )

    # Characteristic operator rate.
    diagonal_rate = float(
        np.max(np.abs(operator.diagonal()))
    )

    row_sum_rate = float(
        np.max(
            np.asarray(
                np.abs(operator).sum(axis=1)
            ).ravel()
        )
    )

    density_scale = float(
        np.max(np.abs(density_vector))
    )

    relative_residual_inf = (
        residual_inf
        / (
            row_sum_rate * density_scale
            + np.finfo(float).tiny
        )
    )

    relative_residual_l1 = (
        residual_l1
        / (
            diagonal_rate
            * float(area_vector @ density_vector)
            + np.finfo(float).tiny
        )
    )

    density_mesh = density_vector.reshape(
        (ny, nx),
        order="C",
    )

    diagnostics = {
        "lagrange_multiplier": lagrange_multiplier,
        "total_probability": float(
            area_vector @ density_vector
        ),
        "minimum_density_before_clipping":
            minimum_before_clipping,
        "minimum_density": float(
            np.min(density_vector)
        ),
        "maximum_density": float(
            np.max(density_vector)
        ),
        "negative_mass_before_clipping":
            negative_mass_before_clipping,
        "residual_inf": residual_inf,
        "residual_l1": residual_l1,
        "relative_residual_inf":
            relative_residual_inf,
        "relative_residual_l1":
            relative_residual_l1,
        "diagonal_rate_scale":
            diagonal_rate,
        "row_sum_rate_scale":
            row_sum_rate,
        "conservation_error": float(
            fp_system["conservation_error"]
        ),
    }

    if verbose:
        for name, value in diagnostics.items():
            print(f"{name}: {value}")

    return density_mesh, diagnostics