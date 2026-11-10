"""Plots for fitted drift and diffusion dependencies on heat flux values."""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np


def _historical_grouped_inputs(root, names, start_year, end_year, method, b_type,
                               quantile_bins, trim_bins, chunk_days,
                               spatial_block_size, sample_size, refresh_grouped):
    """Return saved grouped points, without silently processing millions of cells.

    Normal plotting performs only small NPZ/JSON reads. Heavy 2-pass grouping
    is allowed ONLY with an explicit ``refresh_grouped=True``.
    """
    from Data_processing.io import start_label
    from Functional.stationary import _load_historical_grouped

    historical = {
        (1979, 1989): 3653,
        (1989, 1999): 7305,
        (1999, 2009): 10958,
        (2009, 2019): 14610,
        (2019, 2026): 17106,
    }
    period = (start_year, end_year)
    if period in historical:
        loaded = _load_historical_grouped(
            root, names, start_label(start_year), historical[period], b_type
        )
        if loaded is not None:
            print("Reusing historical grouped points for "
                  f"{start_year}-{end_year}", flush=True)
            return loaded

    if refresh_grouped:
        from Functional.stationary import _group_full_period
        print("Explicit refresh_grouped=True: rebuilding grouped data. "
              "This processes the entire coefficient time series.", flush=True)
        grouped, _meta = _group_full_period(
            root, names, start_year, end_year, method,
            bins=quantile_bins, trim_bins=trim_bins, chunk_days=chunk_days,
            block_size=spatial_block_size, sample_size=sample_size,
            refresh_grouped=True,
        )
        return grouped

    # The grouped cache is small: read it directly instead of going through
    # _group_full_period(), which otherwise falls back to expensive regrouping.
    pair = f"{names[0]}-{names[1]}"
    folder = root / "Functional" / pair
    requested = {
        "bins": quantile_bins,
        "trim_bins": trim_bins,
        "chunk_days": chunk_days,
        "block_size": spatial_block_size,
        "sample_size": sample_size,
    }
    candidates = sorted(folder.glob(
        f"grouped_{start_year}-{end_year}_{method}_*.npz"
    ))
    if not candidates:
        raise FileNotFoundError(
            f"No saved grouped stationary inputs for {start_year}-{end_year} "
            f"method={method} in {folder}. Run task='stationary' in main.py "
            "once. Plotting will not regroup all raw data automatically."
        )

    # If a stationary density was already generated, prefer the exact grouped
    # file listed in its provenance. This makes the fit a reproducible check.
    stationary_meta = root / "Functional" / "Stationary" / (
        f"{pair}_{start_year}-{end_year}.json"
    )
    if stationary_meta.is_file():
        meta = json.loads(stationary_meta.read_text(encoding="utf-8"))
        provenance = meta.get("provenance", {})
        if meta.get("method") == method and provenance.get("cache"):
            preferred_name = Path(provenance["cache"]).name
            candidates.sort(key=lambda f: (f.name != preferred_name, f.name))

    mismatch_report = []
    for cache in candidates:
        meta_path = cache.with_suffix(".json")
        if not meta_path.is_file():
            mismatch_report.append(f"{cache.name}: metadata JSON missing")
            continue
        cache_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        saved = cache_metadata.get("config", {})
        different = [
            f"{key}: saved={saved.get(key)!r}, requested={value!r}"
            for key, value in requested.items()
            if saved.get(key) != value
        ]
        if different:
            mismatch_report.append(f"{cache.name}: " + "; ".join(different))
            continue
        with np.load(cache, allow_pickle=False) as stored:
            grouped = {
                name: tuple(np.asarray(stored[f"{key}_{name}"], dtype=float).copy()
                            for key in ("quantiles", "drift", "log_variance"))
                for name in names
            }
        # Source timestamps are not needed to reproduce the *saved* density;
        # they can change later while its curve and grouped points stay fixed.
        print(f"Reusing grouped stationary inputs: {cache}", flush=True)
        return grouped

    raise ValueError(
        "Grouped inputs exist, but their settings differ from main_plotter.py:\n"
        + "\n".join(mismatch_report)
        + "\nCopy the settings from the STATIONARY block in main.py, "
          "or explicitly set functional_refresh_grouped=True to recompute."
    )


def _check_against_saved_stationary(root, names, start_year, end_year,
                                    method, actual_fits):
    """Check that the saved PDF JSON has these exact fitted coefficients.

    This does NOT recalculate stationary densities. It reports if the PDF was
    produced with older code (for example sensible x1=-40 instead of -50).
    """
    path = root / "Functional" / "Stationary" / (
        f"{names[0]}-{names[1]}_{start_year}-{end_year}.json"
    )
    if not path.is_file():
        print("NOTE: no saved stationary JSON to verify against. "
              "Run task='stationary' in main.py to create it.", flush=True)
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    discrepancies = []
    # Historical grouped inputs are not specific to NP/SP. For assembled data,
    # a method mismatch means the two different fitted stochastic models are
    # being compared.
    if (saved.get("provenance", {}).get("input_type") != "legacy_grouped"
            and saved.get("method") != method):
        discrepancies.append(
            f"method: PDF={saved.get('method')}, plot={method}"
        )
    for name in names:
        fitted = actual_fits[name]
        saved_fit = saved.get("fits", {}).get(name)
        if not saved_fit:
            discrepancies.append(f"{name}: absent in saved stationary JSON")
            continue
        for key, expected in (
            ("diffusion_model", "shifted_x0_x1_piecewise_log_variance"),
            ("stationary_integration_method", "numeric_cumulative_trapezoid_of_2a_over_b2"),
        ):
            if saved_fit.get(key) != expected:
                discrepancies.append(
                    f"{name}/{key}: PDF={saved_fit.get(key)!r}, expected={expected!r}"
                )
        for key in ("x0", "x1", "c1", "c2", "c3", "c4"):
            if (key not in saved_fit or not np.isclose(
                    float(saved_fit[key]), float(fitted[key]),
                    rtol=1e-9, atol=1e-10)):
                discrepancies.append(
                    f"{name}/{key}: PDF={saved_fit.get(key)}, plot={fitted[key]}"
                )
        saved_poly = saved_fit.get("drift_polynomial_ascending")
        current_poly = fitted["drift_polynomial_ascending"]
        if (saved_poly is None or not np.allclose(
                saved_poly, current_poly, rtol=1e-9, atol=1e-12)):
            discrepancies.append(f"{name}: drift polynomial differs")

    if discrepancies:
        warning = (
            "THE SAVED STATIONARY DISTRIBUTION DOES NOT MATCH THE CURRENT "
            "FUNCTIONAL FIT:\n  " + "\n  ".join(discrepancies)
            + "\nRe-run task='stationary' in main.py with the same period "
              "and method, then replot."
        )
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
        return False
    print("VERIFIED: saved stationary density JSON matches the current "
          "fit AND uses the corresponding numerical shifted diffusion model.", flush=True)
    return True


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
    """Plot the very same shifted log-variance model as the numeric PDF."""
    from Functional.stationary import _shifted_log_variance

    x0, x1 = float(params["x0"]), float(params["x1"])
    c1, c2 = float(params["c1"]), float(params["c2"])
    c3, c4 = float(params["c3"]), float(params["c4"])
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if not (xmin < x1 < x0 < xmax):
        raise ValueError(
            f"Diffusion joins are outside x range: [{xmin}, {xmax}], "
            f"x1={x1}, x0={x0}"
        )

    branches = []
    for lower, upper, color, label in (
        (xmin, x1, "orange",
         f"{c1:.3e} * |x - ({x1:.3g})| + "
         f"{c2 * np.sqrt(abs(x1 - x0)) + c3:.3e}"),
        (x1, x0, "purple",
         f"{c2:.3e} * sqrt(|x - ({x0:.3g})|) + {c3:.3e}"),
        (x0, xmax, "red",
         f"{c4:.3e} * sqrt(|x - ({x0:.3g})|) + {c3:.3e}"),
    ):
        xs = np.linspace(lower, upper, 500, endpoint=True)
        ys = _shifted_log_variance(
            xs, x0=x0, x1=x1, c1=c1, c2=c2, c3=c3, c4=c4)
        branches.append((xs, ys, color, label))
    assert np.isclose(branches[0][1][-1], branches[1][1][0], atol=1e-12)
    assert np.isclose(branches[1][1][-1], branches[2][1][0], atol=1e-12)
    return branches


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
                         sample_size=200_000, refresh_grouped=False,
                         verify_stationary=True):
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
    actual_fits = {}

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
                actual_fits[name] = {
                    **params,
                    "drift_polynomial_ascending": coeffs.tolist(),
                }
                ax = axes[1, col]
                ax.plot(x, log_variance, c="cyan", lw=1.4, label="mean")
                for xb, yb, color, label in _diffusion_branches(x, params):
                    ax.plot(xb, yb, c=color, lw=1.6, label=label)
                ax.set_xlabel(name + " values")
                # Historical terminology retained: stored y is log of variance.
                ax.set_ylabel("log(B²)")
                _place_legend(ax)
                print(f"{name}: A RMSE={drift_rmse:.5g}, "
                      f"log(B) RMSE={params['rmse_log_b']:.5g}; "
                      f"joins x1={params['x1']:.5g}, x0={params['x0']:.5g}",
                      flush=True)

            if verify_stationary:
                _check_against_saved_stationary(
                    root, names, start_year, end_year, method, actual_fits
                )

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
