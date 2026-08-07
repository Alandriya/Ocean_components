import numpy as np


def _get_centers(
    coordinates: np.ndarray,
    mesh_size: int,
    coordinate_name: str,
) -> np.ndarray:
    """
    Convert bin edges to centers, or return centers unchanged.
    """
    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.ndim != 1:
        raise ValueError(
            f"{coordinate_name} must be one-dimensional, "
            f"got shape {coordinates.shape}."
        )

    if coordinates.size == mesh_size:
        return coordinates

    if coordinates.size == mesh_size + 1:
        return 0.5 * (coordinates[:-1] + coordinates[1:])

    raise ValueError(
        f"{coordinate_name} has length {coordinates.size}. "
        f"Expected {mesh_size} centers or {mesh_size + 1} edges."
    )


def check_zero_current_solution(
    quantiles1: np.ndarray,
    quantiles2: np.ndarray,
    a1_mesh: np.ndarray,
    a2_mesh: np.ndarray,
    c11_mesh: np.ndarray,
    c12_mesh: np.ndarray,
    c22_mesh: np.ndarray,
    counts_mesh: np.ndarray = None,
    min_count: int = 0,
    regularization_relative: float = 1e-10,
    determinant_relative_tolerance: float = 1e-12,
) -> dict:
    """
    Check whether a zero-current stationary density can exist.

    The state-space array convention is:

        axis 0 -> X2
        axis 1 -> X1

    Returns
    -------
    Dictionary containing:
        F1, F2
        curl_residual
        relative_curl
        determinant
        valid_mask
        numerical summary
    """
    a1 = np.asarray(a1_mesh, dtype=float)
    a2 = np.asarray(a2_mesh, dtype=float)
    c11 = np.asarray(c11_mesh, dtype=float)
    c12 = np.asarray(c12_mesh, dtype=float)
    c22 = np.asarray(c22_mesh, dtype=float)

    expected_shape = a1.shape

    for name, array in {
        "a2_mesh": a2,
        "c11_mesh": c11,
        "c12_mesh": c12,
        "c22_mesh": c22,
    }.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {array.shape}; "
                f"expected {expected_shape}."
            )

    if a1.ndim != 2:
        raise ValueError(
            f"All meshes must be two-dimensional, got {a1.shape}."
        )

    n_x2, n_x1 = a1.shape

    x1 = _get_centers(
        quantiles1,
        n_x1,
        "quantiles1",
    )
    x2 = _get_centers(
        quantiles2,
        n_x2,
        "quantiles2",
    )

    if np.any(np.diff(x1) <= 0):
        raise ValueError("X1 coordinates must be strictly increasing.")

    if np.any(np.diff(x2) <= 0):
        raise ValueError("X2 coordinates must be strictly increasing.")

    finite = (
        np.isfinite(a1)
        & np.isfinite(a2)
        & np.isfinite(c11)
        & np.isfinite(c12)
        & np.isfinite(c22)
    )

    if counts_mesh is not None:
        counts = np.asarray(counts_mesh)

        if counts.shape != expected_shape:
            raise ValueError(
                f"counts_mesh has shape {counts.shape}; "
                f"expected {expected_shape}."
            )

        finite &= counts >= min_count

    # Small regularization prevents numerical instability when C is
    # nearly singular. This changes C only very slightly.
    trace_c = c11 + c22
    typical_scale = np.nanmedian(trace_c[finite])

    if not np.isfinite(typical_scale) or typical_scale <= 0:
        raise ValueError(
            "Cannot determine a positive characteristic scale for C."
        )

    regularization = regularization_relative * typical_scale

    c11_regularized = c11 + regularization
    c22_regularized = c22 + regularization

    determinant = (
        c11_regularized * c22_regularized
        - c12 * c12
    )

    determinant_tolerance = (
        determinant_relative_tolerance
        * typical_scale**2
    )

    valid = (
        finite
        & (c11_regularized > 0)
        & (c22_regularized > 0)
        & (determinant > determinant_tolerance)
    )

    # Mask invalid regions before differentiation.
    a1_work = np.where(valid, a1, np.nan)
    a2_work = np.where(valid, a2, np.nan)

    c11_work = np.where(valid, c11_regularized, np.nan)
    c12_work = np.where(valid, c12, np.nan)
    c22_work = np.where(valid, c22_regularized, np.nan)

    edge_order = 2 if min(n_x1, n_x2) >= 3 else 1

    # div(C), first component:
    #
    # (div C)_1 = dC11/dx1 + dC12/dx2
    d_c11_dx1 = np.gradient(
        c11_work,
        x1,
        axis=1,
        edge_order=edge_order,
    )

    d_c12_dx2 = np.gradient(
        c12_work,
        x2,
        axis=0,
        edge_order=edge_order,
    )

    div_c_1 = d_c11_dx1 + d_c12_dx2

    # div(C), second component:
    #
    # (div C)_2 = dC12/dx1 + dC22/dx2
    d_c12_dx1 = np.gradient(
        c12_work,
        x1,
        axis=1,
        edge_order=edge_order,
    )

    d_c22_dx2 = np.gradient(
        c22_work,
        x2,
        axis=0,
        edge_order=edge_order,
    )

    div_c_2 = d_c12_dx1 + d_c22_dx2

    rhs1 = 2.0 * a1_work - div_c_1
    rhs2 = 2.0 * a2_work - div_c_2

    # F = inverse(C) @ rhs
    #
    # inverse(C) =
    # 1/det * [[ C22, -C12],
    #          [-C12,  C11]]
    f1 = (
        c22_work * rhs1
        - c12_work * rhs2
    ) / determinant

    f2 = (
        -c12_work * rhs1
        + c11_work * rhs2
    ) / determinant

    f1 = np.where(valid, f1, np.nan)
    f2 = np.where(valid, f2, np.nan)

    # Integrability condition:
    #
    # dF2/dx1 - dF1/dx2 = 0
    d_f2_dx1 = np.gradient(
        f2,
        x1,
        axis=1,
        edge_order=edge_order,
    )

    d_f1_dx2 = np.gradient(
        f1,
        x2,
        axis=0,
        edge_order=edge_order,
    )

    curl_residual = d_f2_dx1 - d_f1_dx2

    # Dimensionless local diagnostic.
    derivative_scale = (
        np.abs(d_f2_dx1)
        + np.abs(d_f1_dx2)
    )

    finite_scale = derivative_scale[
        np.isfinite(derivative_scale)
    ]

    scale_floor = (
        0.01 * np.nanmedian(finite_scale)
        if finite_scale.size
        else 1e-12
    )

    scale_floor = max(scale_floor, 1e-12)

    relative_curl = (
        np.abs(curl_residual)
        / (derivative_scale + scale_floor)
    )

    diagnostic_mask = (
        valid
        & np.isfinite(curl_residual)
        & np.isfinite(relative_curl)
    )

    if not np.any(diagnostic_mask):
        raise ValueError(
            "No valid cells remain for the integrability diagnostic. "
            "Check missing values and whether C is nearly singular."
        )

    absolute_values = np.abs(
        curl_residual[diagnostic_mask]
    )
    relative_values = relative_curl[diagnostic_mask]

    summary = {
        "valid_cell_fraction": float(np.mean(valid)),
        "regularization": float(regularization),
        "minimum_valid_determinant": float(
            np.nanmin(determinant[valid])
        ),
        "absolute_curl_mean": float(
            np.nanmean(absolute_values)
        ),
        "absolute_curl_rms": float(
            np.sqrt(np.nanmean(absolute_values**2))
        ),
        "absolute_curl_max": float(
            np.nanmax(absolute_values)
        ),
        "relative_curl_median": float(
            np.nanmedian(relative_values)
        ),
        "relative_curl_p90": float(
            np.nanquantile(relative_values, 0.90)
        ),
        "relative_curl_p95": float(
            np.nanquantile(relative_values, 0.95)
        ),
    }

    return {
        "x1": x1,
        "x2": x2,
        "F1": f1,
        "F2": f2,
        "div_C_1": div_c_1,
        "div_C_2": div_c_2,
        "curl_residual": curl_residual,
        "relative_curl": relative_curl,
        "determinant": determinant,
        "valid_mask": valid,
        "summary": summary,
    }