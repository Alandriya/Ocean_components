"""Stationary Ito-SDE density fitting without any plotting side effects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def _fit_log_diffusion(quantiles, log_diffusion, variable):
    """Fit the existing three-region square-root/log-diffusion model.

    Parameters
    ----------
    quantiles : array-like
        Grouped process values used as regression nodes.
    log_diffusion : array-like
        Grouped values of ``log(b^2)``.
    variable : str
        Variable name, currently ``"sensible"`` or ``"latent"``.

    Returns
    -------
    dict
        Dictionary with fitted piecewise-diffusion parameters.
    """
    from Functional.func_estimation import model_logb2

    x = np.asarray(quantiles, dtype=float)
    y = np.asarray(log_diffusion, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 8:
        raise ValueError("Expected equally sized quantile and log-diffusion vectors")
    x0 = float(x[np.argmin(y)])
    x1 = -40.0
    center = (x1 <= x) & (x < (-10.0 if variable == "sensible" else x0))
    left = ((x < x1) if variable == "sensible" else
            (x < x1) & ((x < -250) | (x > -45)))
    right = x >= x0
    if min(int(center.sum()), int(left.sum()), int(right.sum())) < 2:
        raise ValueError("Insufficient historical quantile data in a diffusion fitting region")

    def central(z, c2, c3):
        return c2 * np.sqrt(np.abs(z)) + c3

    def left_fit(z, c1):
        return c1 * np.abs(z)

    def right_fit(z, c4):
        return c4 * np.sqrt(np.abs(z))

    (c2, c3), _ = curve_fit(central, x[center] - x0, y[center], maxfev=5000)
    # The latent branch used x0 in the historical left-side fit.
    left_origin = x1 if variable == "sensible" else x0
    (c1,), _ = curve_fit(left_fit, x[left] - left_origin,
                         y[left] - central(x1 - x0, c2, c3), maxfev=5000)
    (c4,), _ = curve_fit(right_fit, x[right] - x0, y[right] - c3, maxfev=5000)
    prediction = np.array([model_logb2(v, x1, x0, c1, c2, c3, c4) for v in x])
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
    """Calculate a stationary density using article-defined tail continuation.

    Parameters
    ----------
    grid : array-like
        Strictly increasing grid of physical flux values.
    params : sequence
        Historical parameter tuple ``[x0, x1, k, xmin, xmax, c1, c2,
        c3, c4, z]``. ``xmin`` and ``xmax`` are the endpoints of retained
        regression nodes, not the plotting limits.
    variable : {'sensible', 'latent'}
        Determines which unsupported tail gets a constant continuation of
        the *drift integral* ``I``.

    Returns
    -------
    numpy.ndarray
        Density normalized by the trapezoidal rule on ``grid``.

    Notes
    -----
    The manuscript's 'One-dimensional stationary distributions' section
    defines the plotting approximation:

      sensible: I_plot(x) = I(min(x, xmax))
      latent:   I_plot(x) = I(max(x, xmin))
      both:     p_plot(x) proportional to exp(I_plot(x) - log(b^2(x)))

    Importantly, ``log_b2`` is always evaluated at the actual physical
    argument ``x``. Clipping *both* terms used to create artificial flat
    tails; extrapolating *both* created a spurious distant probability mode.
    This is the visualization-tail convention, not a new closed-form
    stationary law on the whole real line.
    """
    from Functional.func_estimation import I, log_b2

    if variable not in ('sensible', 'latent'):
        raise ValueError("variable must be 'sensible' or 'latent'")
    grid = np.asarray(grid, dtype=float)
    if (grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all()
            or not np.all(np.diff(grid) > 0)):
        raise ValueError('grid must be finite, one-dimensional and increasing')

    x0, x1, k, xmin, xmax, c1, c2, c3, c4, _z = params
    if not np.isfinite([x0, x1, xmin, xmax, c1, c2, c3, c4]).all():
        raise ValueError('Nonfinite stationary fit parameters')
    if not (xmin < xmax and x1 < x0):
        raise ValueError('Invalid regression interval or diffusion junctions')

    constants = _stationary_integration_constants(params)
    args = [c1, c2, c3, c4, x0, x1, k, *constants]

    if variable == 'sensible':
        integral_arguments = np.minimum(grid, xmax)
    else:
        integral_arguments = np.maximum(grid, xmin)

    # The original x must be used here for log_b2, not the capped argument.
    log_values = np.asarray([
        I(float(x_i), args) - log_b2(float(x), args)
        for x, x_i in zip(grid, integral_arguments)
    ], dtype=float)
    if not np.isfinite(log_values).all():
        first_bad = grid[np.flatnonzero(~np.isfinite(log_values))[0]]
        raise ValueError(
            f'{variable}: nonfinite log stationary density near x={first_bad}; '
            'check fitted diffusion parameters and integration constants'
        )

    # Subtracting the peak avoids overflow without modifying relative density.
    scaled_values = np.exp(log_values - float(np.max(log_values)))
    integral = float(np.trapezoid(scaled_values, grid))
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError(f'{variable}: stationary density cannot be normalized')
    density = scaled_values / integral
    if not np.isclose(np.trapezoid(density, grid), 1.0, atol=1e-8, rtol=1e-8):
        raise ValueError(f'{variable}: stationary density normalization failed')
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


def estimate_stationary_characteristics(files_path_prefix, start_year, end_year,
                                        *, names=("sensible", "latent"),
                                        b_type="simple", grid_points=2500):
    """Fit model parameters and estimate stationary density and moments.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root containing the existing Functional fit input arrays.
    start_year, end_year : int
        Stored fit period.
    names : tuple of str
        Ordered variable names for available 1D stationary fits.
    b_type : str
        Historical diffusion fit suffix, usually ``"simple"``.
    grid_points : int
        Number of evaluation points on the historical ``[-2000, 1500]`` interval.

    Returns
    -------
    pathlib.Path
        Path to the numerical NPZ file holding grid and normalized densities.

    Notes
    -----
    Fitting uses the same fourth-order drift and three-region diffusion
    parameterizations as the legacy stationary workflow, without producing
    figures directly.
    """
    from Data_processing.io import start_label

    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    root = Path(files_path_prefix)
    first = start_label(start_year)
    known_ends = {
        (1979, 1989): 3653,
        (1989, 1999): 7305,
        (1999, 2009): 10958,
        (2009, 2019): 14610,
        (2019, 2026): 17106,
    }
    last = known_ends.get((start_year, end_year))
    if last is None:
        raise ValueError("Unknown historical stationary fit filename period")

    prefix = root / "Functional" / f"{names[0]}-{names[1]}"
    quantiles1 = np.load(prefix / f"{first}-{last}_quantiles1.npy")
    quantiles2 = np.load(prefix / f"{first}-{last}_quantiles2.npy")
    drift = np.load(prefix / f"{first}-{last}_a_grouped.npy")
    log_b = np.load(prefix / f"{first}-{last}_b_grouped_log_{b_type}.npy")

    inputs = [quantiles1, quantiles2, drift, log_b]
    grid = np.linspace(-2000, 1500, grid_points)
    output = {"grid": grid}
    meta = {"pair": list(names), "period": f"{start_year}-{end_year}", "fits": {}}

    print(f"Stationary fit: {start_year}-{end_year}, variables: {', '.join(names)}", flush=True)

    for index, name in enumerate(names):
        print(f"Fitting stationary parameters for {name}...", flush=True)
        params = _fit_log_diffusion(inputs[index], log_b[index], name)
        coeffs = list(np.polynomial.polynomial.polyfit(inputs[index], drift[index], 4))
        fitted = [
            params["x0"],
            params["x1"],
            coeffs,
            float(np.min(inputs[index])),
            float(np.max(inputs[index])),
            *[params[key] for key in ("c1", "c2", "c3", "c4")],
            1,
        ]
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
            "mean": mean,
            "standard_deviation": float(np.sqrt(max(variance, 0.0))),
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
            print(
                f"WARNING: {name}: only {plot_mass:.2%} of the probability "
                f"mass lies inside historical plotting limits {plot_bounds}. "
                "Inspect the drift/diffusion fit before interpreting this PDF.",
                flush=True,
            )

    target = root / "Functional" / "Stationary"
    target.mkdir(parents=True, exist_ok=True)
    result = target / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    np.savez_compressed(result, **output)
    metadata_file = result.with_suffix(".json")
    metadata_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Stationary densities saved to: {result}", flush=True)
    print(f"Stationary parameters saved to: {metadata_file}", flush=True)
    return result
