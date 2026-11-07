"""Standalone plotting helpers for one-dimensional stationary distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Data_processing.io import load_field

NAMES_RU = {
    "sensible": "Явный тепловой поток",
    "latent": "Скрытый тепловой поток",
    "Flux": "Полный тепловой поток",
    "SST": "Температура поверхности океана",
    "Pressure": "Атмосферное давление",
}


def plot_path(root: Path, *parts: str) -> Path:
    """Build a figure path under ``Plots`` and create parent folders."""
    path = root / "Plots"
    for part in parts[:-1]:
        path /= part
    path.mkdir(parents=True, exist_ok=True)
    return path / parts[-1]


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save a Matplotlib figure and return its output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def _stationary_plot_config(name: str) -> dict:
    """Return legacy plotting ranges for known heat-flux variables."""
    legacy = {
        "sensible": {
            "xlim": (-500.0, 500.0),
            "hist_range": (-500.0, 500.0),
            "xlabel_ru": "Значения явного теплового потока",
            "xlabel_en": "Sensible flux values",
        },
        "latent": {
            "xlim": (-800.0, 300.0),
            "hist_range": (-800.0, 300.0),
            "xlabel_ru": "Значения скрытого теплового потока",
            "xlabel_en": "Latent flux values",
        },
    }
    if name in legacy:
        return legacy[name]
    return {
        "xlim": None,
        "hist_range": None,
        "xlabel_ru": "Значение переменной",
        "xlabel_en": "Variable value",
    }


def plot_stationary_distributions(files_path_prefix, names, start_year, end_year,
                                  *, histogram_bins: int = 100):
    """Draw each stationary distribution on a separate seaborn-styled figure.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory with cached stationary densities and raw fields.
    names : tuple of str
        Variables to draw.
    start_year, end_year : int
        Period labels.
    histogram_bins : int, default=100
        Number of histogram bins.

    Returns
    -------
    list[pathlib.Path]
        Paths to saved figures, one file per variable.

    Notes
    -----
    The function restores the historical plotting style used in the older
    project version:

    * one variable per figure;
    * seaborn ``whitegrid`` styling;
    * historical x-axis ranges for sensible and latent heat fluxes.
    """
    root = Path(files_path_prefix)
    source = root / "Functional" / "Stationary" / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    outputs = []

    sns.set_style("whitegrid")

    with np.load(source) as densities:
        grid = np.asarray(densities["grid"], dtype=float)
        for name in names:
            config = _stationary_plot_config(name)
            field = load_field(root, name, start_year, end_year)
            clean = np.asarray(field, dtype=float)
            clean = clean[np.isfinite(clean)].ravel()

            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            ax.plot(
                grid,
                densities[f"pdf_{name}"],
                c="r",
                linewidth=2.0,
                label="Стационарная плотность",
            )
            # ax.plot(grid, densities[f"pdf_{name}"], c="r", linewidth=2.0,
            #         label="Stationary density")

            ax.hist(
                clean,
                range=config["hist_range"],
                alpha=0.5,
                bins=histogram_bins,
                density=True,
                label="Гистограмма наблюдений",
            )
            # ax.hist(clean, range=config["hist_range"], alpha=0.5,
            #         bins=histogram_bins, density=True,
            #         label="Observed histogram")

            if config["xlim"] is not None:
                ax.set_xlim(*config["xlim"])

            ax.set_title(NAMES_RU.get(name, name), fontsize=16)
            ax.set_xlabel(config["xlabel_ru"], fontsize=14)
            # ax.set_xlabel(config["xlabel_en"], fontsize=14)
            ax.set_ylabel("Плотность вероятности", fontsize=14)
            # ax.set_ylabel("Probability density", fontsize=14)
            ax.legend()
            fig.tight_layout()

            output = save_figure(
                fig,
                plot_path(
                    root,
                    "Distributions",
                    "Stationary",
                    f"{name}_{start_year}-{end_year}.png",
                ),
            )
            plt.close(fig)
            outputs.append(output)

    return outputs
