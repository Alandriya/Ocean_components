"""Plots for fitted drift and diffusion dependencies on heat flux values."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _historical_grouped_inputs(root, names, start_year, end_year, method, b_type,
                               quantile_bins, trim_bins, chunk_days,
                               spatial_block_size, sample_size, refresh_grouped):
    """Load grouped fit inputs or rebuild them from assembled coefficients.

    Parameters
    ----------
    root : pathlib.Path
        Data root.
    names : tuple[str, str]
        Ordered pair of flux variable names.
    start_year, end_year : int
        Requested period.
    method : {'np', 'sp'}
        Coefficient method used for assembled full-period inputs.
    b_type : str
        Historical log-diffusion fit label.
    quantile_bins : int
        Number of automatic quantile groups for assembled inputs.
    trim_bins : int
        Number of tail bins removed from each side.
    chunk_days : int
        Read size for assembled inputs.
    spatial_block_size : int
        Spatial averaging block size.
    sample_size : int
        Number of sampled observations for quantile edges.
    refresh_grouped : bool
        Rebuild grouped cache even if it already exists.

    Returns
    -------
    dict
        Mapping ``name -> (x, drift, log_variance)``.
    """
    from Data_processing.io import start_label
    from Functional.stationary import _group_full_period, _load_historical_grouped

    historical = {
        (1979, 1989): 3653,
        (1989, 1999): 7305,
        (1999, 2009): 10958,
        (2009, 2019): 14610,
        (2019, 2026): 17106,
    }
    first = start_label(start_year)
    period = (start_year, end_year)

    if period in historical:
        loaded = _load_historical_grouped(root, names, first, historical[period], b_type)
        if loaded is not None:
            return loaded

    grouped, _meta = _group_full_period(
        root,
        names,
        start_year,
        end_year,
        method,
        bins=quantile_bins,
        trim_bins=trim_bins,
        chunk_days=chunk_days,
        block_size=spatial_block_size,
        sample_size=sample_size,
        refresh_grouped=refresh_grouped,
    )
    return grouped


def _signed_term(value, power):
    """Return a signed polynomial term without an accidental '+ -' sign."""
    sign = "+" if value >= 0.0 else "-"
    magnitude = f"{abs(float(value)):.3e}"
    if power == 0:
        return f"{sign} {magnitude}"
    if power == 1:
        return f"{sign} {magnitude} * x"
    return f"{sign} {magnitude} * x^{power}"


def _format_polynomial_desc(coeffs):
    """Print the historical polynomial expression, wrapped to fit one panel."""
    powers = list(reversed(range(len(coeffs))))
    terms = [_signed_term(coeffs[power], power) for power in powers]
    if not terms:
        return "0"
    first = terms[0].strip()
    if first.startswith("+ "):
        first = first[2:]
    # A short break after the quadratic term keeps the long legend entirely
    # within one subplot. The formula itself is unchanged.
    if len(terms) == 5:
        return first + " " + " ".join(terms[1:3]) + "\n" + " ".join(terms[3:])
    return first + " " + " ".join(terms[1:])


def _diffusion_branches(x, params):
    """Evaluate the *same* continuous three-region law as stationary.py.

    Returns each interval including its junction point. The orange, purple and
    red curves share the actual join values at x1 and x0 and cannot have gaps
    due to plotting different coordinate systems.
    """
    x0, x1 = float(params["x0"]), float(params["x1"])
    c1, c2 = float(params["c1"]), float(params["c2"])
    c3, c4 = float(params["c3"]), float(params["c4"])
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if not (xmin < x1 < x0 < xmax):
        raise ValueError(
            f"Diffusion joins are outside x range: x=[{xmin},{xmax}], "
            f"x1={x1}, x0={x0}"
        )

    d_l = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
    d_r = c3 + (c2 - c4) * np.sqrt(abs(x0))

    left_x = np.linspace(xmin, x1, 500, endpoint=True)
    center_x = np.linspace(x1, x0, 500, endpoint=True)
    right_x = np.linspace(x0, xmax, 500, endpoint=True)

    left_y = c1 * np.abs(left_x) + d_l
    center_y = c2 * np.sqrt(np.abs(center_x)) + c3
    right_y = c4 * np.sqrt(np.abs(right_x)) + d_r

    assert np.isclose(left_y[-1], center_y[0], rtol=1e-12, atol=1e-12)
    assert np.isclose(center_y[-1], right_y[0], rtol=1e-12, atol=1e-12)
    return (
        (left_x, left_y, "orange", f"{c1:.3e} * |x| {d_l:+.3e}"),
        (center_x, center_y, "purple", f"{c2:.3e} * sqrt(|x|) {c3:+.3e}"),
        (right_x, right_y, "red", f"{c4:.3e} * sqrt(|x|) {d_r:+.3e}"),
    )


def _place_legend(ax):
    """Place legends above plots as in the original seaborn whitegrid figures."""
    return ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        borderaxespad=0.0,
        fontsize=10,
        frameon=True,
        ncol=1,
        handlelength=2.2,
    )


def plot_functional_fits(files_path_prefix, start_year, end_year,
                         *, names=("sensible", "latent"), method="sp",
                         b_type="simple", quantile_bins=100, trim_bins=2,
                         chunk_days=128, spatial_block_size=1,
                         sample_size=200_000, refresh_grouped=False):
    """Recreate the white-background seaborn 2x2 A/log(B) fit figure.

    Uses exactly the grouped data and diffusion fitter employed by the
    stationary-distribution calculation. Supports historical saved groups and
    assembled multiyear coefficients. Parameters match main_plotter.py.

    Returns
    -------
    pathlib.Path
        Filename of the saved PNG.
    """
    import seaborn as sns
    from Functional.stationary import _fit_log_diffusion

    if len(names) != 2:
        raise ValueError("The 2x2 figure requires two variable names")
    root = Path(files_path_prefix)
    inputs = _historical_grouped_inputs(
        root, names, start_year, end_year, method, b_type,
        quantile_bins, trim_bins, chunk_days, spatial_block_size,
        sample_size, refresh_grouped,
    )

    # The legacy project used sns.set_style('whitegrid') with white axes and
    # legends above each panel. Do not use the previous gray 'ggplot' theme.
    with sns.axes_style("whitegrid"), plt.rc_context({
        "font.size": 14,
        "axes.labelsize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.edgecolor": "#c9c9c9",
        "grid.color": "#dedede",
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        try:
            for col, name in enumerate(names):
                x, drift, log_variance = inputs[name]
                x = np.asarray(x, dtype=np.float64)
                drift = np.asarray(drift, dtype=np.float64)
                log_variance = np.asarray(log_variance, dtype=np.float64)
                if (x.ndim != 1 or x.size < 8 or drift.shape != x.shape
                        or log_variance.shape != x.shape):
                    raise ValueError(f"Invalid grouped data for {name}")
                if not (np.isfinite(x).all() and np.isfinite(drift).all()
                        and np.isfinite(log_variance).all()):
                    raise ValueError(f"Non-finite grouped values for {name}")

                coeffs = np.polynomial.polynomial.polyfit(x, drift, 4)
                grid = np.linspace(float(x.min()), float(x.max()), 1500)
                drift_fit = np.polynomial.polynomial.polyval(grid, coeffs)
                drift_rmse = float(np.sqrt(np.mean(
                    (np.polynomial.polynomial.polyval(x, coeffs) - drift) ** 2)))

                ax = axes[0, col]
                ax.plot(x, drift, c="cyan", lw=1.4, label="mean")
                ax.plot(grid, drift_fit, c="red", lw=1.6,
                        label=_format_polynomial_desc(coeffs))
                ax.set_xlabel(name + " values")
                ax.set_ylabel("A")
                _place_legend(ax)

                params = _fit_log_diffusion(x, log_variance, name)
                ax = axes[1, col]
                ax.plot(x, log_variance, c="cyan", lw=1.4, label="mean")
                for xb, yb, color, label in _diffusion_branches(x, params):
                    ax.plot(xb, yb, c=color, lw=1.6, label=label)
                ax.set_xlabel(name + " values")
                # Historical terminology retained: stored y is log of variance.
                ax.set_ylabel("log(B)")
                _place_legend(ax)
                print(f"{name}: A RMSE={drift_rmse:.5g}, "
                      f"log(B) RMSE={params['rmse_log_b']:.5g}; "
                      f"joins x1={params['x1']:.5g}, x0={params['x0']:.5g}",
                      flush=True)

            # Explicit room for legends above each subplot; tight_layout()
            # would push the long formulas across adjacent panels.
            fig.subplots_adjust(
                left=0.075, right=0.965,
                bottom=0.075, top=0.84,
                wspace=0.21, hspace=0.62,
            )
            folder = root / "Plots" / "Functional" / "Coefficients"
            folder.mkdir(parents=True, exist_ok=True)
            suffix = f"_{method}" if (start_year, end_year) == (1979, 2026) else ""
            path = folder / (
                f"fit_{names[0]}_{names[1]}_{start_year}-{end_year}{suffix}.png")
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        finally:
            plt.close(fig)
    return path
