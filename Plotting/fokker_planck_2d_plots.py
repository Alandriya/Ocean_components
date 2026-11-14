"""Visualization and grid-convergence diagnostics for stationary 2D Fokker-Planck results.

The functions in this module are plotting-only. They read already calculated
``*_stationary_wide_stencil.npz`` files and never rebuild phase-space
coefficients or solve the Fokker-Planck equation.

Two workflows are provided:

1. ``plot_fp2d_stationary_comparison`` compares the stationary Fokker-Planck
   density with the empirical joint density on one selected grid.
2. ``plot_fp2d_grid_convergence`` compares several already solved phase grids
   and writes a small JSON summary of marginal moments and pairwise distances.
"""

from __future__ import annotations

import json
import math
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def _stationary_path(root, names, start_year, end_year, method, bins):
    """Return the conventional stationary 2D Fokker-Planck result path."""
    return (
        Path(root)
        / "Functional"
        / "FokkerPlanck2D"
        / "Stationary"
        / (
            f"{names[0]}-{names[1]}_{start_year}-{end_year}_{method}_"
            f"{int(bins[0])}x{int(bins[1])}_stationary_wide_stencil.npz"
        )
    )


def _load_stationary_result(path):
    """Load and validate one saved stationary 2D Fokker-Planck solution."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Stationary 2D result not found: {path}")

    required = {
        "x_edges",
        "y_edges",
        "x_centers",
        "y_centers",
        "domain_mask",
        "stationary_mass",
        "stationary_density",
        "empirical_mass_conditioned",
        "empirical_density_conditioned",
    }
    with np.load(path, allow_pickle=False) as source:
        missing = required.difference(source.files)
        if missing:
            raise ValueError(
                f"{path.name} is missing arrays: {sorted(missing)}"
            )
        result = {
            key: np.array(source[key], copy=True)
            for key in required
        }

    nx = len(result["x_centers"])
    ny = len(result["y_centers"])
    shape = (nx, ny)
    if result["x_edges"].shape != (nx + 1,):
        raise ValueError("x_edges and x_centers are inconsistent")
    if result["y_edges"].shape != (ny + 1,):
        raise ValueError("y_edges and y_centers are inconsistent")

    for key in (
        "domain_mask",
        "stationary_mass",
        "stationary_density",
        "empirical_mass_conditioned",
        "empirical_density_conditioned",
    ):
        if result[key].shape != shape:
            raise ValueError(
                f"{key} has shape {result[key].shape}, expected {shape}"
            )

    result["domain_mask"] = result["domain_mask"].astype(bool, copy=False)
    result["path"] = path
    return result


def _marginals(mass, x_edges, y_edges):
    """Return normalized marginal probability densities along both coordinates."""
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 2:
        raise ValueError("mass must be a two-dimensional array")

    total = float(mass.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("mass must have a positive finite sum")
    mass = mass / total

    dx = np.diff(np.asarray(x_edges, dtype=np.float64))
    dy = np.diff(np.asarray(y_edges, dtype=np.float64))
    if mass.shape != (len(dx), len(dy)):
        raise ValueError("mass shape does not match grid edges")

    px = mass.sum(axis=1) / dx
    py = mass.sum(axis=0) / dy
    return px, py


def _moments(mass, x_centers, y_centers):
    """Return first two marginal moments and correlation for a grid mass."""
    mass = np.asarray(mass, dtype=np.float64)
    mass = mass / mass.sum()

    x = np.asarray(x_centers, dtype=np.float64)
    y = np.asarray(y_centers, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    mean_x = float(np.sum(mass * xx))
    mean_y = float(np.sum(mass * yy))
    var_x = float(np.sum(mass * (xx - mean_x) ** 2))
    var_y = float(np.sum(mass * (yy - mean_y) ** 2))
    covariance = float(
        np.sum(mass * (xx - mean_x) * (yy - mean_y))
    )
    correlation = covariance / math.sqrt(var_x * var_y)

    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "std_x": math.sqrt(max(var_x, 0.0)),
        "std_y": math.sqrt(max(var_y, 0.0)),
        "correlation": float(correlation),
    }


def _comparison_fields(result, relative_floor):
    """Return masked empirical/FP/difference/relative-difference fields."""
    if not np.isfinite(relative_floor) or not 0 <= relative_floor < 1:
        raise ValueError("relative_floor must satisfy 0 <= value < 1")

    domain = result["domain_mask"]
    stationary = np.asarray(result["stationary_density"], dtype=np.float64)
    empirical = np.asarray(
        result["empirical_density_conditioned"],
        dtype=np.float64,
    )

    fp = np.where(domain, stationary, np.nan)
    observed = np.where(domain, empirical, np.nan)
    difference = np.where(domain, stationary - empirical, np.nan)

    threshold = relative_floor * float(np.nanmax(empirical))
    relative_mask = domain & np.isfinite(empirical) & (empirical > threshold)
    relative = np.full_like(stationary, np.nan)
    relative[relative_mask] = (
        stationary[relative_mask] - empirical[relative_mask]
    ) / empirical[relative_mask]

    return observed, fp, difference, relative, threshold


def _save_figure(fig, path, dpi):
    """Save a figure and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fp2d_stationary_comparison(
        files_path_prefix,
        start_year,
        end_year,
        *,
        names=("sensible", "latent"),
        method="sp",
        bins=(80, 80),
        stationary_file=None,
        relative_floor=0.01,
        dpi=300):
    """Plot empirical and stationary 2D densities plus their marginals.

    Parameters
    ----------
    files_path_prefix : str or Path
        Project data root.
    start_year, end_year : int
        Period labels.
    names : tuple[str, str]
        Ordered phase-space coordinates.
    method : {'np', 'sp'}
        Coefficient-estimation method.
    bins : tuple[int, int]
        Grid resolution used by the saved stationary solution.
    stationary_file : str or Path, optional
        Explicit stationary result. If omitted, the conventional project path
        is used.
    relative_floor : float
        Relative-difference values are shown only where the empirical density
        exceeds this fraction of its maximum. This avoids meaningless ratios
        in almost-empty tails.
    dpi : int
        Output figure resolution.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Paths to the 2D comparison figure and marginal-density figure.
    """
    root = Path(files_path_prefix)
    source = (
        Path(stationary_file)
        if stationary_file is not None
        else _stationary_path(
            root, names, start_year, end_year, method, bins
        )
    )
    result = _load_stationary_result(source)

    observed, fp, difference, relative, threshold = _comparison_fields(
        result,
        relative_floor,
    )

    x_edges = result["x_edges"]
    y_edges = result["y_edges"]
    x_centers = result["x_centers"]
    y_centers = result["y_centers"]

    density_limit = float(
        max(np.nanmax(observed), np.nanmax(fp))
    )
    difference_limit = float(np.nanmax(np.abs(difference)))
    relative_limit = float(np.nanpercentile(np.abs(relative), 98))
    if not np.isfinite(relative_limit) or relative_limit <= 0:
        relative_limit = 1.0

    with plt.rc_context({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))

        panels = (
            (axes[0, 0], observed, "Empirical joint density", 0.0, density_limit),
            (axes[0, 1], fp, "Stationary Fokker-Planck density", 0.0, density_limit),
            (
                axes[1, 0],
                difference,
                "FP density - empirical density",
                -difference_limit,
                difference_limit,
            ),
            (
                axes[1, 1],
                relative,
                (
                    "(FP - empirical) / empirical\n"
                    f"shown for empirical > {relative_floor:.1%} of peak"
                ),
                -relative_limit,
                relative_limit,
            ),
        )

        for ax, values, title, vmin, vmax in panels:
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                values.T,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(mesh, ax=ax)
            ax.set_title(title)
            ax.set_xlabel(names[0])
            ax.set_ylabel(names[1])

        fig.suptitle(
            f"Stationary 2D Fokker-Planck comparison: "
            f"{start_year}-{end_year}, {bins[0]}x{bins[1]}",
            fontsize=15,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    folder = root / "Plots" / "Functional" / "FokkerPlanck2D/"
    comparison_path = _save_figure(
        fig,
        folder / (
            f"stationary_comparison_{names[0]}-{names[1]}_"
            f"{start_year}-{end_year}_{method}_{bins[0]}x{bins[1]}.png"
        ),
        dpi,
    )
    sns.set_style("whitegrid")
    stationary_px, stationary_py = _marginals(
        result["stationary_mass"],
        x_edges,
        y_edges,
    )
    empirical_px, empirical_py = _marginals(
        result["empirical_mass_conditioned"],
        x_edges,
        y_edges,
    )

    with plt.rc_context({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

        axes[0].plot(
            x_centers,
            empirical_px,
            label="Empirical",
            c='green',
        )
        axes[0].plot(
            x_centers,
            stationary_px,
            label="Stationary FP",
            c='red',
        )
        axes[0].set_title(f"{names[0]} marginal")
        axes[0].set_xlabel(names[0])
        axes[0].set_ylabel("Probability density")
        axes[0].legend()

        axes[1].plot(
            y_centers,
            empirical_py,
            label="Empirical",
            c='green',
        )
        axes[1].plot(
            y_centers,
            stationary_py,
            label="Stationary FP",
            c='red',
        )
        axes[1].set_title(f"{names[1]} marginal")
        axes[1].set_xlabel(names[1])
        axes[1].set_ylabel("Probability density")
        axes[1].legend()

        # fig.suptitle(
        #     f"Stationary marginal densities: "
        #     f"{start_year}-{end_year}, {bins[0]}x{bins[1]}",
        #     fontsize=15,
        # )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    marginals_path = _save_figure(
        fig,
        folder / (
            f"stationary_marginals_{names[0]}-{names[1]}_"
            f"{start_year}-{end_year}_{method}_{bins[0]}x{bins[1]}.png"
        ),
        dpi,
    )

    diagnostic_path = folder / (
        f"stationary_comparison_{names[0]}-{names[1]}_"
        f"{start_year}-{end_year}_{method}_{bins[0]}x{bins[1]}.json"
    )
    diagnostic_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "relative_floor": float(relative_floor),
                "relative_density_threshold": threshold,
                "stationary_moments": _moments(
                    result["stationary_mass"],
                    x_centers,
                    y_centers,
                ),
                "empirical_moments": _moments(
                    result["empirical_mass_conditioned"],
                    x_centers,
                    y_centers,
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return comparison_path, marginals_path


def _evaluate_piecewise_constant_density(result, x, y, field):
    """Evaluate one grid density at target cell centers by cell lookup."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    ix = np.searchsorted(
        result["x_edges"],
        x,
        side="right",
    ) - 1
    iy = np.searchsorted(
        result["y_edges"],
        y,
        side="right",
    ) - 1

    valid_x = (ix >= 0) & (ix < len(result["x_centers"]))
    valid_y = (iy >= 0) & (iy < len(result["y_centers"]))
    ix = np.clip(ix, 0, len(result["x_centers"]) - 1)
    iy = np.clip(iy, 0, len(result["y_centers"]) - 1)

    values = np.asarray(result[field], dtype=np.float64)[
        ix[:, None],
        iy[None, :],
    ]
    valid = valid_x[:, None] & valid_y[None, :]
    return np.where(valid, values, 0.0)


def _pairwise_density_distance(first, second, common_resolution):
    """Approximate TV and Hellinger distances on a common physical grid."""
    if common_resolution < 100:
        raise ValueError("common_resolution must be at least 100")

    x_low = max(first["x_edges"][0], second["x_edges"][0])
    x_high = min(first["x_edges"][-1], second["x_edges"][-1])
    y_low = max(first["y_edges"][0], second["y_edges"][0])
    y_high = min(first["y_edges"][-1], second["y_edges"][-1])
    if not (x_low < x_high and y_low < y_high):
        raise ValueError("The two stationary grids do not overlap")

    x_edges = np.linspace(x_low, x_high, common_resolution + 1)
    y_edges = np.linspace(y_low, y_high, common_resolution + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])

    p = _evaluate_piecewise_constant_density(
        first,
        x,
        y,
        "stationary_density",
    )
    q = _evaluate_piecewise_constant_density(
        second,
        x,
        y,
        "stationary_density",
    )
    p = p * area
    q = q * area
    p /= p.sum()
    q /= q.sum()

    tv = 0.5 * float(np.sum(np.abs(p - q)))
    hellinger = math.sqrt(
        0.5 * float(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    )
    return {
        "total_variation_distance": tv,
        "hellinger_distance": hellinger,
    }


def plot_fp2d_grid_convergence(
        files_path_prefix,
        start_year,
        end_year,
        *,
        names=("sensible", "latent"),
        method="sp",
        resolutions=(60, 80, 100),
        stationary_files=None,
        common_resolution=800,
        dpi=300):
    """Plot marginal and moment convergence for already solved phase grids.

    Parameters
    ----------
    files_path_prefix : str or Path
        Project data root.
    start_year, end_year : int
        Period labels.
    names : tuple[str, str]
        Ordered phase-space coordinates.
    method : {'np', 'sp'}
        Coefficient-estimation method.
    resolutions : tuple[int, ...]
        Square phase-grid sizes to compare.
    stationary_files : dict[int, str | Path], optional
        Explicit files keyed by resolution. Conventional project paths are
        used for missing keys.
    common_resolution : int
        Auxiliary physical grid used only for pairwise density distances.
    dpi : int
        Output figure resolution.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Paths to the convergence PNG and JSON summary.
    """
    root = Path(files_path_prefix)
    sns.set_style("whitegrid")
    stationary_files = stationary_files or {}
    results = {}

    for resolution in resolutions:
        source = stationary_files.get(
            int(resolution),
            _stationary_path(
                root,
                names,
                start_year,
                end_year,
                method,
                (resolution, resolution),
            ),
        )
        results[int(resolution)] = _load_stationary_result(source)

    ordered = sorted(results)
    summary = {
        "period": f"{start_year}-{end_year}",
        "method": method,
        "resolutions": ordered,
        "moments": {},
        "pairwise_stationary_distances": {},
    }

    with plt.rc_context({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for resolution in ordered:
            result = results[resolution]
            fp_x, fp_y = _marginals(
                result["stationary_mass"],
                result["x_edges"],
                result["y_edges"],
            )
            axes[0, 0].plot(
                result["x_centers"],
                fp_x,
                label=f"{resolution}x{resolution}",
            )
            axes[0, 1].plot(
                result["y_centers"],
                fp_y,
                label=f"{resolution}x{resolution}",
            )

            stationary_moments = _moments(
                result["stationary_mass"],
                result["x_centers"],
                result["y_centers"],
            )
            empirical_moments = _moments(
                result["empirical_mass_conditioned"],
                result["x_centers"],
                result["y_centers"],
            )
            summary["moments"][str(resolution)] = {
                "stationary": stationary_moments,
                "empirical": empirical_moments,
            }

        axes[0, 0].set_title(f"{names[0]} stationary marginal")
        axes[0, 0].set_xlabel(names[0])
        axes[0, 0].set_ylabel("Probability density")
        axes[0, 0].legend()

        axes[0, 1].set_title(f"{names[1]} stationary marginal")
        axes[0, 1].set_xlabel(names[1])
        axes[0, 1].set_ylabel("Probability density")
        axes[0, 1].legend()

        mean_x = [
            summary["moments"][str(r)]["stationary"]["mean_x"]
            for r in ordered
        ]
        mean_y = [
            summary["moments"][str(r)]["stationary"]["mean_y"]
            for r in ordered
        ]
        corr = [
            summary["moments"][str(r)]["stationary"]["correlation"]
            for r in ordered
        ]

        axes[1, 0].plot(
            ordered,
            mean_x,
            marker="o",
            label=f"mean {names[0]}",
        )
        axes[1, 0].plot(
            ordered,
            mean_y,
            marker="o",
            label=f"mean {names[1]}",
        )
        axes[1, 0].set_title("Stationary means vs grid resolution")
        axes[1, 0].set_xlabel("Grid size per axis")
        axes[1, 0].set_ylabel("Mean")
        axes[1, 0].legend()

        axes[1, 1].plot(
            ordered,
            corr,
            marker="o",
            label="Stationary correlation",
        )
        axes[1, 1].set_title("Stationary correlation vs grid resolution")
        axes[1, 1].set_xlabel("Grid size per axis")
        axes[1, 1].set_ylabel("Correlation")
        axes[1, 1].legend()

        # fig.suptitle(
        #     f"2D Fokker-Planck grid convergence: {start_year}-{end_year}",
        #     fontsize=15,
        # )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    folder = root / "Plots" / "Functional" / "FokkerPlanck2D"
    label = "-".join(map(str, ordered))
    figure_path = _save_figure(
        fig,
        folder / (
            f"grid_convergence_{names[0]}-{names[1]}_"
            f"{start_year}-{end_year}_{method}_{label}.png"
        ),
        dpi,
    )

    for first, second in zip(ordered[:-1], ordered[1:]):
        key = f"{first}x{first}_vs_{second}x{second}"
        summary["pairwise_stationary_distances"][key] = (
            _pairwise_density_distance(
                results[first],
                results[second],
                common_resolution,
            )
        )

    summary["common_distance_grid_resolution"] = int(common_resolution)
    summary_path = folder / (
        f"grid_convergence_{names[0]}-{names[1]}_"
        f"{start_year}-{end_year}_{method}_{label}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return figure_path, summary_path
