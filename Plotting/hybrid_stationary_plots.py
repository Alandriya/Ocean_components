"""Hybrid stationary-density plots and comparisons with two-dimensional Fokker–Planck marginals."""

from __future__ import annotations

# --- Hybrid Stationary Plots ---
# Histogram + stationary core / empirical-GPD transitions / GPD tails.
#
# Dedicated experimental visualization: never overwrites the classical plot.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Functional.hybrid_tails import build_hybrid_density
from Plotting.stationary_plots import (
    NAMES, _observed_histogram, _stationary_plot_config, plot_path, save_figure,
)

CORE_COLOR = "#c62828"          # red: Fokker--Planck central fit
TRANSITION_COLOR = "#f28e2b"    # orange: smooth C1 transitions
TAIL_COLOR = "#ff69b4"          # pink: GPD tails
HIST_COLOR = "#1f77b4"          # blue: transparent ERA5 histogram
HIST_EDGE_COLOR = "#2d6396"


def _colored_sections(ax, grid, pdf, info):
    """Draw joined intervals with identical endpoints and no visible gaps."""
    l = info["left"]
    r = info["right"]
    anchors = [float(grid[0]), float(l["tail_start"]), float(l["anchor"]),
               float(r["anchor"]), float(r["tail_start"]), float(grid[-1])]
    if not all(a < b for a, b in zip(anchors[:-1], anchors[1:])):
        raise ValueError(f"Hybrid sections overlap or are out of order: {anchors}")
    colors = [TAIL_COLOR, TRANSITION_COLOR, CORE_COLOR, TRANSITION_COLOR, TAIL_COLOR]
    labels = ["Empirical GPD tails", "Smooth C¹ transitions",
              "Stationary FP core", None, None]
    for left, right, color, label in zip(anchors[:-1], anchors[1:], colors, labels):
        interior = grid[(grid > left) & (grid < right)]
        xx = np.concatenate(([left], interior, [right]))
        yy = np.interp(xx, grid, pdf)
        ax.plot(xx, yy, color=color, lw=2.7, label=label, zorder=5)
    for v in anchors[1:-1]:
        ax.axvline(v, ls=":", lw=0.8, color="#777777", alpha=0.5, zorder=1)


def plot_hybrid_stationary_distributions(files_path_prefix, names, start_year,
                                         end_year, *, histogram_bins=100,
                                         histogram_chunk_days=128,
                                         refresh_histograms=False,
                                         tail_quantile=0.975,
                                         transition_quantile=0.90,
                                         reference_weight=0.70,
                                         reference_period=(1979, 2026),
                                         sample_days=1600, pixels_per_day=160,
                                         refresh_samples=False,
                                         refresh_hybrid=False,
                                         show_original=False):
    """Render diagnostic histogram and 3-color hybrid stationary density.

    Previous stationary densities are NEVER drawn. ``show_original`` is kept
    only for compatibility with older main_plotter.py configurations; it is
    intentionally ignored even when True. Original data files are unchanged.
    """
    root = Path(files_path_prefix)
    source = (root / "Functional" / "Stationary"
              / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz")
    if not source.is_file():
        raise FileNotFoundError(f"No central stationary PDF found: {source}")
    outputs = []
    # The original NPZ is still used by build_hybrid_density for the central
    # fit, but its unmodified PDF must not appear on this comparison-free plot.
    with sns.axes_style("whitegrid"), plt.rc_context({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.edgecolor": "#c9c9c9",
        "grid.color": "#dedede",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        for name in names:
            grid, hybrid_pdf, _, info, hybrid_path = build_hybrid_density(
                root, name, start_year, end_year, names=names,
                tail_quantile=tail_quantile,
                transition_quantile=transition_quantile,
                reference_weight=reference_weight,
                reference_period=reference_period,
                sample_days=sample_days, pixels_per_day=pixels_per_day,
                refresh_samples=refresh_samples, refresh_hybrid=refresh_hybrid,
            )
            config = _stationary_plot_config(name)
            hist_density, hist_edges, hist_cache = _observed_histogram(
                root, name, start_year, end_year,
                bins=histogram_bins, value_range=config["hist_range"],
                chunk_days=histogram_chunk_days, refresh=refresh_histograms,
            )
            print(f"Using observed histogram cache: {hist_cache}", flush=True)
            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            try:
                ax.stairs(hist_density, hist_edges, fill=True,
                          facecolor=HIST_COLOR, edgecolor=HIST_EDGE_COLOR,
                          lw=0.6, alpha=0.26,
                          label="Observed ERA5 histogram (display range)", zorder=1)
                _colored_sections(ax, grid, hybrid_pdf, info)
                if config["xlim"] is not None:
                    ax.set_xlim(*config["xlim"])
                ax.set_ylim(bottom=0)
                ax.set_title(NAMES.get(name, name) + " — experimental hybrid tails",
                             fontsize=15)
                ax.set_xlabel(config["xlabel_en"])
                ax.set_ylabel("Probability density")
                # Show the three requested components in the intuitive order.
                handles, labels = ax.get_legend_handles_labels()
                priority = {"Stationary FP core": 1,
                            "Smooth C¹ transitions": 2,
                            "Empirical GPD tails": 3,
                            "Observed ERA5 histogram (display range)": 0}
                ordered = sorted(range(len(labels)),
                                 key=lambda i: priority.get(labels[i], 99))
                ax.legend([handles[i] for i in ordered],
                          [labels[i] for i in ordered],
                          loc="upper right", fontsize=10)
                fig.tight_layout()
                filename = f"{name}_{start_year}-{end_year}_hybrid_eng.png"
                output = save_figure(fig, plot_path(
                    root, "Distributions", "Stationary", filename))
                print(f"Hybrid tail figure: {output}", flush=True)
                print(f"Hybrid source: {hybrid_path}", flush=True)
                outputs.append(output)
            finally:
                plt.close(fig)
    return outputs


# --- Hybrid Fp2D Marginal Overlay ---
# Overlay ERA5 histograms, 1D hybrid densities and stationary 2D FP marginals.
#
# Designed for the Alandriya/Ocean_components repository. Uses its existing
# sources and plotting helpers rather than reimplementing hybrid-tail fitting or
# 2D marginalization. Select task="hybrid_fp2d_marginal_overlay" in main_plotter.py
# and run `python main_plotter.py` from the repository root. The standalone CLI entry point is also supported.
#
# The three curves do NOT necessarily represent the same conditioning:
# * The ERA5 histogram is normalized on the selected plotting range.
# * The 1D hybrid PDF is normalized on its finite stationary output grid.
# * The 2D FP marginal is conditioned on the retained 2D solver domain.
# Furthermore, hybrid GPD tails use raw ERA5 observations, whereas the 2D
# phase-space state can use a different preprocessing convention. Therefore
# this overlay is descriptive and is not by itself a formal goodness-of-fit test.
#
# No existing figures or source PDFs are overwritten.

import argparse
import json



from Plotting.stationary_plots import (
    NAMES,
    _observed_histogram,
    _stationary_plot_config,
)
from Plotting.fokker_planck_2d_plots import (
    _load_stationary_result,
    _marginals,
    _stationary_path,
)

MARGINAL_COLOR = "#4b347c"       # Purple, visually distinct from hybrid curves
MARGINAL_LABEL = "2D stationary FP marginal (retained domain)"
HIST_LABEL = "Observed ERA5 histogram (display range)"


def _validate_hybrid(grid, density, name):
    """Check the returned hybrid arrays before plotting."""
    grid = np.asarray(grid, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or density.shape != grid.shape:
        raise ValueError(f"Invalid one-dimensional hybrid density for {name}")
    if not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
        raise ValueError(f"Hybrid grid for {name} must be finite and increasing")
    if not np.all(np.isfinite(density)) or np.min(density) < -1e-12:
        raise ValueError(f"Hybrid density for {name} must be nonnegative and finite")
    if np.trapezoid(np.maximum(density, 0), grid) <= 0:
        raise ValueError(f"Hybrid density for {name} has zero integrated mass")
    return grid, density


def _panel(ax, name, hist, hybrid, marginal):
    """Draw one heat-flux panel, retaining the repository's hybrid color code."""
    hist_density, hist_edges = hist
    grid, hybrid_pdf, hybrid_info = hybrid
    edges, marginal_pdf = marginal
    if marginal_pdf.shape != (len(edges) - 1,):
        raise ValueError(f"2D marginal for {name} has incompatible bin edges")
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"Non-increasing 2D marginal edges for {name}")

    ax.stairs(
        hist_density,
        hist_edges,
        fill=True,
        facecolor=HIST_COLOR,
        edgecolor=HIST_EDGE_COLOR,
        linewidth=0.5,
        alpha=0.26,
        label=HIST_LABEL,
        zorder=1,
    )
    # This helper segments the hybrid PDF into stationary core, smooth joins,
    # and GPD tails exactly as in the original hybrid figure code.
    _colored_sections(ax, grid, hybrid_pdf, hybrid_info)
    # Bin masses are converted to density in _marginals. A stairs plot is
    # appropriate because the reconstructed 2D marginal is piecewise constant.
    ax.stairs(
        marginal_pdf,
        edges,
        color=MARGINAL_COLOR,
        linewidth=2.4,
        linestyle="--",
        label=MARGINAL_LABEL,
        zorder=8,
    )

    config = _stationary_plot_config(name)
    if config["xlim"] is not None:
        ax.set_xlim(*config["xlim"])
    ax.set_ylim(bottom=0)
    ax.set_title(NAMES.get(name, name), fontsize=14)
    ax.set_xlabel(config["xlabel_en"], fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)

    # Keep one instance of each label and order it as the reader expects.
    order = [
        HIST_LABEL,
        "Stationary FP core",
        "Smooth C¹ transitions",
        "Empirical GPD tails",
        MARGINAL_LABEL,
    ]
    handles, labels = ax.get_legend_handles_labels()
    selected = {label: handle for handle, label in zip(handles, labels)}
    ax.legend(
        [selected[key] for key in order if key in selected],
        [key for key in order if key in selected],
        loc="upper right",
        fontsize=8.6,
        framealpha=0.92,
    )
    ax.grid(True, alpha=0.25)



def _distance_on_common_support(grid, pdf_1d, edges, pdf_2d):
    """TV/H for the normalized densities on their overlapping 1D interval.

    The hybrid density is linearly interpolated on its saved grid. The 2D
    stationary marginal is piecewise constant on its phase-space cell edges.
    Both densities are conditioned on the same coordinate interval, but this
    does NOT eliminate differences in state preprocessing or 2D conditioning.
    Absolute-value and square-root integrals are evaluated analytically on
    each segment of the union of the two grids.
    """
    grid = np.asarray(grid, dtype=float)
    pdf_1d = np.asarray(pdf_1d, dtype=float)
    edges = np.asarray(edges, dtype=float)
    pdf_2d = np.asarray(pdf_2d, dtype=float)
    if (grid.ndim != 1 or pdf_1d.shape != grid.shape or
            edges.ndim != 1 or pdf_2d.shape != (len(edges) - 1,)):
        raise ValueError("Incompatible marginal and hybrid array sizes")
    if (not np.isfinite(grid).all() or not np.isfinite(edges).all() or
            np.any(np.diff(grid) <= 0) or np.any(np.diff(edges) <= 0)):
        raise ValueError("Invalid monotonically increasing comparison grids")
    if (not np.isfinite(pdf_1d).all() or not np.isfinite(pdf_2d).all() or
            np.min(pdf_1d) < -1e-12 or np.min(pdf_2d) < -1e-12):
        raise ValueError("Comparison densities must be finite and nonnegative")
    low = max(float(grid[0]), float(edges[0]))
    high = min(float(grid[-1]), float(edges[-1]))
    if not low < high:
        raise ValueError("The two one-dimensional density supports do not overlap")
    knots = np.unique(np.concatenate((
        np.array([low, high]), grid[(grid > low) & (grid < high)],
        edges[(edges > low) & (edges < high)],
    )))
    widths = np.diff(knots)
    mid = 0.5 * (knots[:-1] + knots[1:])
    bin_idx = np.searchsorted(edges, mid, side="right") - 1
    q = np.maximum(pdf_2d[bin_idx], 0.0)
    p_start = np.maximum(np.interp(knots[:-1], grid, pdf_1d), 0.0)
    p_end = np.maximum(np.interp(knots[1:], grid, pdf_1d), 0.0)
    mass_p = float(np.sum(0.5 * (p_start + p_end) * widths))
    mass_q = float(np.sum(q * widths))
    if mass_p <= 0 or mass_q <= 0:
        raise ValueError("Both densities must have positive mass on common support")
    p_start /= mass_p
    p_end /= mass_p
    q /= mass_q

    # TV: exact integral of the absolute difference of two affine functions
    # on each segment, including a possible sign change in the interior.
    a = p_start - q
    b = p_end - q
    abs_a = np.abs(a)
    abs_b = np.abs(b)
    crossing = (a * b < 0)
    absolute_area = 0.5 * widths * (abs_a + abs_b)
    relative_zero = np.divide(abs_a, abs_a + abs_b,
                              out=np.zeros_like(abs_a), where=(abs_a + abs_b) > 0)
    absolute_area[crossing] = 0.5 * widths[crossing] * (
        abs_a[crossing] * relative_zero[crossing] +
        abs_b[crossing] * (1.0 - relative_zero[crossing]))
    tv = 0.5 * float(absolute_area.sum())

    # H^2 = 1 - integral sqrt(p q) over the normalized common interval.
    # For p affine, the integral of sqrt(p) has a stable closed form.
    sqrt_p0 = np.sqrt(p_start)
    sqrt_p1 = np.sqrt(p_end)
    denom = sqrt_p0 + sqrt_p1
    sqrt_p_integral = np.divide(
        (2.0 * widths / 3.0) * (p_start + p_end + sqrt_p0 * sqrt_p1),
        denom, out=np.zeros_like(widths), where=denom > 0,
    )
    bhattacharyya = float(np.sum(np.sqrt(q) * sqrt_p_integral))
    hellinger = float(np.sqrt(np.clip(1.0 - bhattacharyya, 0.0, 1.0)))
    if not (-1e-9 <= tv <= 1 + 1e-9):
        raise ValueError(f"Unexpected TV distance {tv}")
    return {
        "total_variation": tv,
        "hellinger": hellinger,
        "common_interval": [low, high],
        "hybrid_mass_before_restriction_normalization": mass_p,
        "marginal_mass_before_restriction_normalization": mass_q,
        "integration": "exact for piecewise-linear hybrid and piecewise-constant 2D marginal",
    }


def _write_metric_outputs(folder, year_start, year_end, method, bins, metrics):
    """Write reproducible JSON and LaTeX macro overrides for the manuscript."""
    label = f"{year_start}-{year_end}_{method}_{bins[0]}x{bins[1]}"
    json_path = folder / f"hybrid_fp2d_marginal_distances_{label}.json"
    json_path.write_text(json.dumps({
        "period": f"{year_start}-{year_end}",
        "grid": list(map(int, bins)),
        "method": method,
        "comparison": "separately renormalized on common coordinate interval",
        "limitations": "2D conditioning and state preprocessing may still differ",
        "variables": metrics,
    }, indent=2), encoding="utf-8")
    latex_path = folder / "comparison_1d_2d_metrics.tex"
    latex_lines = ["% Computed from saved NPZ arrays, NOT from PNG plots.",
                   "% Copy into manuscript figures/ directory to fill the blue table."]
    for name, prefix in (("sensible", "Sensible"), ("latent", "Latent")):
        if name in metrics:
            m = metrics[name]
            latex_lines.append(f"\\renewcommand{{\\comparison{prefix}TV}}{{{m['total_variation']:.4f}}}")
            latex_lines.append(f"\\renewcommand{{\\comparison{prefix}H}}{{{m['hellinger']:.4f}}}")
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    return json_path, latex_path

def plot_hybrid_fp2d_marginal_overlay(
    files_path_prefix,
    start_year,
    end_year,
    *,
    names=("sensible", "latent"),
    method="sp",
    bins=(80, 80),
    stationary_file=None,
    histogram_bins=100,
    histogram_chunk_days=128,
    refresh_histograms=False,
    tail_quantile=0.975,
    transition_quantile=0.90,
    reference_weight=0.70,
    reference_period=(1979, 2026),
    sample_days=1600,
    pixels_per_day=160,
    refresh_samples=False,
    refresh_hybrid=False,
    dpi=300,
):
    """Write two standalone comparisons, one per variable.

    Returns a dict with ``individual`` and ``stationary_source``.
    No 2D system is solved; a previously saved stationary 2D NPZ is required.
    The 1D hybrid source can be assembled from the existing cached 1D input
    using the same function as ``hybrid_stationary_plots.py``.
    """
    root = Path(files_path_prefix).expanduser()
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("names must contain two distinct coordinates")
    if len(bins) != 2 or min(bins) < 1:
        raise ValueError("bins must be a pair of positive integers")

    stationary_source = (
        Path(stationary_file).expanduser()
        if stationary_file is not None
        else _stationary_path(root, names, start_year, end_year, method, bins)
    )
    result = _load_stationary_result(stationary_source)
    marginals = _marginals(
        result["stationary_mass"], result["x_edges"], result["y_edges"]
    )
    phase_edges = (result["x_edges"], result["y_edges"])

    panels = []
    distance_metrics = {}
    for axis_index, name in enumerate(names):
        grid, pdf, _sections, info, cache_path = build_hybrid_density(
            root,
            name,
            start_year,
            end_year,
            names=names,
            tail_quantile=tail_quantile,
            transition_quantile=transition_quantile,
            reference_weight=reference_weight,
            reference_period=reference_period,
            sample_days=sample_days,
            pixels_per_day=pixels_per_day,
            refresh_samples=refresh_samples,
            refresh_hybrid=refresh_hybrid,
        )
        grid, pdf = _validate_hybrid(grid, pdf, name)
        config = _stationary_plot_config(name)
        if config["hist_range"] is None:
            raise ValueError(f"No histogram range configured for {name}")
        hist_density, hist_edges, histogram_cache = _observed_histogram(
            root,
            name,
            start_year,
            end_year,
            bins=histogram_bins,
            value_range=config["hist_range"],
            chunk_days=histogram_chunk_days,
            refresh=refresh_histograms,
        )
        print(f"{name}: hybrid={cache_path}; histogram={histogram_cache}")
        distance_metrics[name] = _distance_on_common_support(
            grid, pdf, phase_edges[axis_index], marginals[axis_index]
        )
        print(
            f"{name}: TV={distance_metrics[name]['total_variation']:.4f}, "
            f"H={distance_metrics[name]['hellinger']:.4f} "
            f"on {distance_metrics[name]['common_interval']}"
        )
        panels.append(
            (name, (hist_density, hist_edges), (grid, pdf, info),
             (phase_edges[axis_index], marginals[axis_index]))
        )

    output_dir = root / "Plots" / "Distributions" / "Stationary"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    note = (
        "Different normalization domains: ERA5 histogram (display range), "
        "1D hybrid (output grid), 2D marginal (retained solver domain)."
    )

    with sns.axes_style("whitegrid"), plt.rc_context({
        "font.size": 11,
        "axes.edgecolor": "#c9c9c9",
        "grid.color": "#dedede",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        for name, hist, hybrid, marginal in panels:
            fig, ax = plt.subplots(figsize=(12, 7))
            try:
                _panel(ax, name, hist, hybrid, marginal)
                fig.subplots_adjust(bottom=0.15)
                fig.text(0.5, 0.035, note, ha="center", fontsize=9,
                         color="#555555")
                output = output_dir / (
                    f"{name}_{start_year}-{end_year}_hybrid_vs_2d_"
                    f"{method}_{bins[0]}x{bins[1]}.png"
                )
                fig.savefig(output, dpi=dpi, bbox_inches="tight")
                outputs.append(output)
            finally:
                plt.close(fig)

    metric_json, metric_tex = _write_metric_outputs(
        output_dir, start_year, end_year, method, bins, distance_metrics
    )
    for path in outputs:
        print(f"Saved comparison figure: {path}")
    print(f"Computed distance report: {metric_json}")
    print(f"LaTeX macro overrides: {metric_tex}")
    return {
        "individual": outputs,
        "stationary_source": stationary_source,
        "metrics": distance_metrics,
        "metrics_json": metric_json,
        "metrics_tex": metric_tex,
    }


def main(argv=None):
    """Run the hybrid stationary plots workflow using its configured options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="Root directory of the OceanFull data archive")
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--method", default="sp")
    parser.add_argument("--bins", nargs=2, type=int, default=(80, 80),
                        metavar=("NX", "NY"))
    parser.add_argument("--stationary-file", type=Path, default=None,
                        help="Explicit 2D stationary NPZ; overrides conventional path")
    parser.add_argument("--histogram-bins", type=int, default=100)
    parser.add_argument("--histogram-chunk-days", type=int, default=128)
    parser.add_argument("--refresh-histograms", action="store_true")
    parser.add_argument("--refresh-samples", action="store_true")
    parser.add_argument("--refresh-hybrid", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    plot_hybrid_fp2d_marginal_overlay(
        args.root, args.start_year, args.end_year,
        method=args.method, bins=tuple(args.bins),
        stationary_file=args.stationary_file,
        histogram_bins=args.histogram_bins,
        histogram_chunk_days=args.histogram_chunk_days,
        refresh_histograms=args.refresh_histograms,
        refresh_samples=args.refresh_samples,
        refresh_hybrid=args.refresh_hybrid,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
