"""Stationary Ito-SDE density fitting without any plotting side effects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def _fit_log_diffusion(quantiles, log_diffusion, variable):
    """Fit the existing three-region square-root/log-diffusion model."""
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
    return {"x0": x0, "x1": x1, "c1": float(c1), "c2": float(c2),
            "c3": float(c3), "c4": float(c4),
            "rmse_log_b": float(np.sqrt(np.mean((prediction - y)**2)))}


def _stationary_density(grid, params):
    """Evaluate the unchanged piecewise stationary-density formula and normalize."""
    from Functional.func_estimation import I, log_b2, count_constants

    x0, x1, k, xmin, xmax, c1, c2, c3, c4, _ = params
    constants = count_constants(params)
    # count_constants returns regional coefficients followed by integration constants.
    args = [c1, c2, c3, c4, x0, x1, k, *constants[-4:]]
    clipped = np.clip(grid, xmin, xmax)
    log_values = np.asarray([I(float(x), args) - log_b2(float(x), args) for x in clipped])
    values = np.exp(log_values - np.max(log_values))
    norm = np.trapezoid(values, grid)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Stationary probability density could not be normalized")
    return values / norm


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
        Historical diffusion fit suffix, usually 'simple'.
    grid_points : int
        Number of evaluation points on the historical [-2000, 1500] interval.

    Returns
    -------
    pathlib.Path
        Path to the numerical NPZ file holding grid and normalized densities.

    Notes
    -----
    Fitting uses the same fourth-order drift and three-region diffusion
    parameterizations as plot_ab_functional_2d, without producing figures.
    """
    from Data_processing.io import start_label

    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    root = Path(files_path_prefix)
    first = start_label(start_year)
    known_ends = {(1979, 1989): 3653, (1989, 1999): 7305,
                  (1999, 2009): 10958, (2009, 2019): 14610,
                  (2019, 2026): 17106}
    last = known_ends.get((start_year, end_year))
    if last is None:
        raise ValueError("Unknown historical stationary fit filename period")
    prefix = root / "Functional" / f"{names[0]}-{names[1]}"
    inputs = [np.load(prefix / f"{first}-{last}_{key}.npy")
              for key in ("quantiles1", "quantiles2", "a_grouped",
                          f"b_grouped_log_{b_type}")]
    quantiles1, quantiles2, drift, log_b = inputs
    grid = np.linspace(-2000, 1500, grid_points)
    output = {"grid": grid}
    meta = {"pair": list(names), "period": f"{start_year}-{end_year}", "fits": {}}
    for index, name in enumerate(names):
        params = _fit_log_diffusion(inputs[index], log_b[index], name)
        coeffs = list(np.polynomial.polynomial.polyfit(inputs[index], drift[index], 4))
        fitted = [params["x0"], params["x1"], coeffs,
                  float(np.min(inputs[index])), float(np.max(inputs[index])),
                  *[params[key] for key in ("c1", "c2", "c3", "c4")], 1]
        density = _stationary_density(grid, fitted)
        mean = float(np.trapezoid(grid * density, grid))
        variance = float(np.trapezoid((grid - mean)**2 * density, grid))
        output[f"pdf_{name}"] = density
        meta["fits"][name] = {
            **params, "drift_polynomial_ascending": coeffs, "mean": mean,
            "standard_deviation": float(np.sqrt(max(variance, 0))),
            "density_integral": float(np.trapezoid(density, grid)),
        }
    target = root / "Functional" / "Stationary"
    target.mkdir(parents=True, exist_ok=True)
    result = target / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    np.savez_compressed(result, **output)
    result.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return result
