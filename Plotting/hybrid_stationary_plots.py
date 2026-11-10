"""Histogram + stationary core / empirical-GPD transitions / GPD tails.

Dedicated experimental visualization: never overwrites the classical plot.
"""

from __future__ import annotations

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
