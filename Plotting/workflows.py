"""High-level plotting workflows reading precomputed oceanographic arrays."""

from __future__ import annotations

import json
import subprocess
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Data_processing.io import BASE_DATE, load_field, load_spectral_field, read_ocean_mask, start_label
from Plotting.paths import plot_path, save_figure

NAMES_RU = {"sensible": "Явный тепловой поток", "latent": "Скрытый тепловой поток",
            "Flux": "Суммарный тепловой поток", "SST": "Температура поверхности моря",
            "Pressure": "Атмосферное давление", "A": "Коэффициент A",
            "B": "Коэффициент B", "C": "Коэффициент C", "F": "Коэффициент F"}


def _draw_map(ax, data, title):
    """Draw one geographic grid with a Russian-language title and axes."""
    im = ax.imshow(np.ma.masked_invalid(data), cmap="coolwarm", origin="upper")
    ax.set_title(title)
    # ax.set_title("Spatial field")
    ax.set_xlabel("Долгота")
    # ax.set_xlabel("Longitude")
    ax.set_ylabel("Широта")
    # ax.set_ylabel("Latitude")
    return im


def plot_variables(files_path_prefix, start_year, end_year, *, day=0):
    """Create paired heat-flux and three-variable maps for one observation day.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root data directory.
    start_year, end_year : int
        Period label used by raw field filenames.
    day : int
        Zero-based local observation index.

    Returns
    -------
    list of Path
        Saved PNG paths.
    """
    root = Path(files_path_prefix)
    outputs = []
    for group, folder in (("sensible latent", "Sensible-Latent"),
                          ("Flux SST Pressure", "Flux-SST-Pressure")):
        names = group.split()
        fields = [load_field(root, name, start_year, end_year) for name in names]
        if any(day < 0 or day >= len(field) for field in fields):
            raise IndexError("Requested day is outside the available input files")
        fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False)
        for ax, name, array in zip(axes[0], names, fields):
            im = _draw_map(ax, array[day], NAMES_RU[name])
            fig.colorbar(im, ax=ax, shrink=0.8)
        timestamp = BASE_DATE + timedelta(days=start_label(start_year) - 1 + day)
        fig.suptitle(f"Поля океанографических переменных — {timestamp:%d.%m.%Y}")
        # fig.suptitle("Oceanographic variables by date")
        fig.tight_layout()
        outputs.append(save_figure(fig, plot_path(root, "Variables", folder,
                                                 f"{start_year}-{end_year}_{day:05d}.png"), dpi=200))
        plt.close(fig)
    return outputs


def plot_spectral_modes(files_path_prefix, start_year, end_year, *, names=("Flux", "SST"),
                        first_day=0, last_day=5, n_bins=250, n_modes=3):
    """Render saved spectral decompositions using the original scientific plotter.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root.
    start_year, end_year : int
        Requested period.
    names : tuple of str
        Self or mixed spectral variable pair.
    first_day, last_day : int
        Half-open local plotting interval.
    n_bins : int
        Quantile bin count used during spectral estimation.
    n_modes : int
        Number of displayed modes.

    Returns
    -------
    None
        Image files are written under Plots/Eigenvalues.
    """
    from Plotting.plot_eigenvalues import plot_eigenvalues
    from Data_processing.data_processing import scale_to_bins

    root = Path(files_path_prefix)
    series = [load_spectral_field(root, name, start_year, end_year) for name in names]
    quantiles = [scale_to_bins(series[i], n_bins)[1] for i in range(2)]
    plot_eigenvalues(str(root), n_modes, read_ocean_mask(root), first_day, last_day,
                     start_label(start_year), series[1], quantiles[1], names=names,
                     shape=(161, 181), array1=series[0], array1_quantiles=quantiles[0],
                     save_mode_maps=False)


def plot_mean_year(files_path_prefix, category, name, start_year, end_year,
                   *, day_indices=(45, 105, 165, 225, 285, 345)):
    """Display six representative calendar dates of an annual climatology.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root.
    category : str
        Numerical Mean_year category.
    name : str
        Mean-year array name.
    start_year, end_year : int
        Inclusive/exclusive period labels.
    day_indices : sequence of int
        Zero-based climatological days to display.

    Returns
    -------
    pathlib.Path
        Output figure path.
    """
    root = Path(files_path_prefix)
    data = np.load(root / "Mean_year" / category /
                   f"{name}_{start_year}-{end_year}.npy", mmap_mode="r")
    outputs = []
    if data.ndim < 3:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(len(data)), data)
        ax.set_xlabel("День среднего года")
        # ax.set_xlabel("Day of climatological year")
        ax.set_ylabel("Среднее значение")
        # ax.set_ylabel("Mean value")
        fig.suptitle(f"Средний год: {NAMES_RU.get(name, name)}")
        # fig.suptitle("Annual climatology")
        fig.tight_layout()
        outputs.append(save_figure(fig, plot_path(root, "Mean Year", category,
                                                   f"{name}_{start_year}-{end_year}.png")))
        plt.close(fig)
    else:
        channels = (data[:, channel] for channel in range(data.shape[1])) if data.ndim == 4 else (data,)
        for channel_number, channel in enumerate(channels):
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            for ax, index in zip(axes.flat, day_indices):
                im = _draw_map(ax, channel[index], f"День {index + 1}")
                fig.colorbar(im, ax=ax, shrink=0.72)
            description = (f" — компонент {channel_number + 1}" if data.ndim == 4 else "")
            fig.suptitle(f"Средний год: {NAMES_RU.get(name, name)}{description}")
            # fig.suptitle("Annual climatology")
            fig.tight_layout()
            suffix = f"_ch{channel_number+1}" if data.ndim == 4 else ""
            outputs.append(save_figure(fig, plot_path(root, "Mean Year", category,
                                                       f"{name}_{start_year}-{end_year}{suffix}.png")))
            plt.close(fig)
    return outputs


def coefficient_frames(files_path_prefix, names, n_bins_2d, method, coefficient,
                        *, start=None, stop=None, step=1):
    """Generate an ordered PNG sequence from previously calculated coefficients.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root.
    names : tuple of str
        Variable pair.
    n_bins_2d : int
        Quantile group count.
    method : {'np', 'sp'}
        Coefficient estimator.
    coefficient : {'A', 'B', 'C', 'F', 'FS'}
        Numeric array to display.
    start, stop : int, optional
        Zero-based half-open frame boundaries.
    step : int
        Sampling stride.

    Returns
    -------
    list of Path
        Saved frame paths in chronological order.
    """
    root = Path(files_path_prefix)
    folder = "Nonparametric_2d" if method == "np" else "Semiparametric_2d"
    parent = root / "Coefficients" / "continuous" / f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}" / folder
    files = sorted(parent.glob(f"{coefficient}_*-*.npy"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one collected {coefficient} array under {parent}; found {files}")
    data = np.load(files[0], mmap_mode="r")
    stop = len(data) if stop is None else min(stop, len(data))
    start = 0 if start is None else start
    frames = []
    for i in range(start, stop, step):
        channels = data[i] if data.ndim == 4 else data[i][None, ...]
        fig, axes = plt.subplots(1, len(channels), figsize=(5.5 * len(channels), 5), squeeze=False)
        for j, (ax, field) in enumerate(zip(axes[0], channels)):
            img = _draw_map(ax, field, f"{NAMES_RU.get(coefficient, coefficient)} — компонент {j+1}")
            # ax.set_title("Coefficient component")
            fig.colorbar(img, ax=ax, shrink=0.8)
        fig.tight_layout()
        saved = save_figure(fig, plot_path(root, "Coefficients", coefficient,
                                           f"{names[0]}-{names[1]}", method, f"frame_{i:05d}.png"))
        frames.append(saved)
        plt.close(fig)
    return frames


def coefficient_video(files_path_prefix, names, method, coefficient, *, fps=5):
    """Create an MP4 from a contiguous numbered series of coefficient frames.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory.
    names : tuple of str
        Variable pair.
    method : str
        NP/SP estimator name.
    coefficient : str
        Coefficient identifier.
    fps : int
        Frame rate of the resulting video.

    Returns
    -------
    pathlib.Path
        Saved MP4 under Plots/Videos/Coefficients.
    """
    root = Path(files_path_prefix)
    directory = root / "Plots" / "Coefficients" / coefficient / f"{names[0]}-{names[1]}" / method
    frames = sorted(directory.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No coefficient frames found in {directory}")
    # FFmpeg's glob pattern supports sparse frame indices produced by step > 1.
    target = plot_path(root, "Videos", "Coefficients", f"{names[0]}-{names[1]}",
                       f"{coefficient}_{method}.mp4")
    subprocess.run(["ffmpeg", "-y", "-pattern_type", "glob", "-framerate", str(fps),
                    "-i", str(directory / "frame_*.png"), "-pix_fmt", "yuv420p",
                    str(target)], check=True)
    return target


def plot_stationary_distributions(files_path_prefix, names, start_year, end_year,
                                  *, histogram_bins=100):
    """Overlay saved stationary PDFs with normalized raw-data histograms.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory.
    names : tuple of str
        Variables to compare.
    start_year, end_year : int
        Period labels.
    histogram_bins : int
        Number of bins in empirical histograms.

    Returns
    -------
    pathlib.Path
        One figure with a panel for each variable.
    """
    root = Path(files_path_prefix)
    source = root / "Functional" / "Stationary" / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    with np.load(source) as densities:
        grid = densities["grid"]
        fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 5), squeeze=False)
        for ax, name in zip(axes[0], names):
            field = load_field(root, name, start_year, end_year)
            clean = np.asarray(field)[np.isfinite(field)]
            ax.hist(clean, bins=histogram_bins, density=True, alpha=0.5,
                    label="Гистограмма наблюдений")
            # ax.hist(clean, label="Observed histogram")
            ax.plot(grid, densities[f"pdf_{name}"], linewidth=2,
                    label="Стационарная плотность")
            # ax.plot(grid, densities[f"pdf_{name}"], label="Stationary density")
            ax.set_title(NAMES_RU.get(name, name))
            ax.set_xlabel("Значение переменной")
            # ax.set_xlabel("Variable value")
            ax.set_ylabel("Плотность вероятности")
            # ax.set_ylabel("Probability density")
            ax.legend()
    fig.tight_layout()
    output = save_figure(fig, plot_path(root, "Distributions", "Stationary",
                                        f"{names[0]}-{names[1]}_{start_year}-{end_year}.png"))
    plt.close(fig)
    return output


def plot_copula_results(files_path_prefix, names, start_year, end_year):
    """Render cached copula densities, joint densities, and model diagnostics.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root directory.
    names : tuple of str
        Variable pair.
    start_year, end_year : int
        Period labels.

    Returns
    -------
    list of Path
        Saved figures for copula and joint-density comparisons.
    """
    root = Path(files_path_prefix)
    source = root / "Functional" / "Copulas" / f"{names[0]}-{names[1]}_{start_year}-{end_year}.npz"
    outputs = []
    with np.load(source) as data:
        variants = (("empirical", "Эмпирическая"), ("gaussian", "Гауссовская"),
                    ("student_t", "Стьюдента"), ("checkerboard", "Шахматная"),
                    ("bernstein", "Бернштейна"))
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for ax, (key, description) in zip(axes.flat, variants):
            im = ax.imshow(data[key], origin="lower", extent=(0, 1, 0, 1))
            ax.set_title(f"Копула {description.lower()}")
            # ax.set_title("Copula density")
            ax.set_xlabel("Ранг первой переменной")
            # ax.set_xlabel("Rank of first variable")
            ax.set_ylabel("Ранг второй переменной")
            # ax.set_ylabel("Rank of second variable")
            fig.colorbar(im, ax=ax, shrink=0.75)
        axes.flat[-1].axis("off")
        fig.tight_layout()
        outputs.append(save_figure(fig, plot_path(root, "Copulas", "Comparison",
                                                  f"{names[0]}-{names[1]}_{start_year}-{end_year}.png")))
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, key, label in zip(axes, ("joint_gaussian", "joint_checkerboard", "joint_bernstein"),
                                  ("Гауссовская", "Шахматная", "Бернштейна")):
            im = ax.contourf(data["x1"], data["x2"], data[key], levels=15)
            ax.set_title(f"Совместная плотность: {label}")
            # ax.set_title("Joint probability density")
            ax.set_xlabel(NAMES_RU.get(names[0], names[0]))
            ax.set_ylabel(NAMES_RU.get(names[1], names[1]))
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        outputs.append(save_figure(fig, plot_path(root, "Copulas", "Joint",
                                                  f"{names[0]}-{names[1]}_{start_year}-{end_year}.png")))
        plt.close(fig)
    return outputs
