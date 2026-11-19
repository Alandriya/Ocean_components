"""Map-based plotting workflows for cached oceanographic arrays."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from Data_processing.io import BASE_DATE, load_field, start_label
from Data_processing.mean_year import mean_year_parts, mean_year_paths
from Plotting.paths import plot_path


NAMES_EN = {
    "sensible": "Sensible heat flux",
    "latent": "Latent heat flux",
    "Flux": "Total heat flux",
    "SST": "Sea surface temperature",
    "Pressure": "Atmospheric pressure",
    "A": "Coefficient A",
    "B": "Coefficient B",
    "C": "Coefficient C",
    "F": "Coefficient F",
    "FS": "Coefficient FS",
}

NAMES_RU = {
    "sensible": "Явный поток",
    "latent": "Скрытый поток",
    "Flux": "Суммарный поток",
    "SST": "Температура поверхности моря",
    "Pressure": "Атмосферное давление",
    "A": "Коэффициент A",
    "B": "Коэффициент B",
    "C": "Коэффициент C",
    "F": "Коэффициент F",
    "FS": "Коэффициент FS",
}

LON_BOUNDS = (-90.0, 0.0)
LAT_BOUNDS = (0.0, 80.0)
LON_TICKS = [-90, -60, -30, 0]
LAT_TICKS = [0, 30, 60, 80]
LON_LABELS = ["90W", "60W", "30W", "0"]
LAT_LABELS = ["0", "30N", "60N", "80N"]
DIVERGING_NAMES = {"sensible", "latent", "Flux", "A", "B", "C", "F", "FS"}


def save_figure(figure, filename, *args, **kwargs):
    """Save figures without Plotting.paths automatic translation side effects."""
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(destination), *args, **kwargs)
    return destination


def _format_geo_axis(ax, *, show_x_labels=True, show_y_labels=True):
    ax.set_xlim(*LON_BOUNDS)
    ax.set_ylim(*LAT_BOUNDS)
    ax.set_xticks(LON_TICKS)
    ax.set_yticks(LAT_TICKS)
    ax.set_xticklabels(LON_LABELS if show_x_labels else [])
    ax.set_yticklabels(LAT_LABELS if show_y_labels else [])
    ax.tick_params(axis="both", labelsize=9, pad=2)
    ax.grid(True, color="#bcbcbc", alpha=0.45, linewidth=0.7)


def _data_norm(data, name):
    finite = np.asarray(data, dtype=float)[np.isfinite(data)]
    if finite.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin == vmax:
        delta = 1.0 if vmin == 0.0 else 0.01 * abs(vmin)
        vmin -= delta
        vmax += delta
    if name in DIVERGING_NAMES and vmin < 0.0 < vmax:
        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _draw_map(ax, data, title, *, cmap="coolwarm", name="",
              show_x_labels=True, show_y_labels=True):
    """Draw one geographic grid with coordinate axes and matching colorbar."""
    masked = np.ma.masked_invalid(data)
    image = ax.imshow(
        masked,
        cmap=cmap,
        norm=_data_norm(masked, name),
        origin="upper",
        extent=(*LON_BOUNDS, *LAT_BOUNDS),
        interpolation="none",
        aspect="auto",
    )
    image.cmap.set_bad("lightgreen")
    ax.set_title(title)
    _format_geo_axis(ax, show_x_labels=show_x_labels, show_y_labels=show_y_labels)
    return image


def _add_colorbar(fig, ax, image):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    return fig.colorbar(image, cax=cax)


def plot_variables(files_path_prefix, start_year, end_year, *, start=0, stop=1):
    """Create paired heat-flux and three-variable maps for requested days.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root data directory.
    start_year, end_year : int
        Period label used by raw field filenames.
    start, stop : int
        Zero-based half-open local observation interval.

    Returns
    -------
    list of Path
        Saved PNG paths.
    """
    root = Path(files_path_prefix)
    outputs = []
    if start < 0 or stop < start:
        raise ValueError("Expected 0 <= start <= stop")
    for group, folder in (("sensible latent", "Sensible-Latent"),
                          ("Flux SST Pressure", "Flux-SST-Pressure")):
        names = group.split()
        fields = [load_field(root, name, start_year, end_year) for name in names]
        if any(stop > len(field) for field in fields):
            raise IndexError("Requested day interval is outside the available input files")
        for day in range(start, stop):
            fig, axes = plt.subplots(1, len(names), figsize=(6.8 * len(names), 5.6), squeeze=False,
                                     constrained_layout=True)
            for ax, name, array in zip(axes[0], names, fields):
                image = _draw_map(ax, array[day], NAMES_RU.get(name, name), name=name,
                                  show_y_labels=(ax is axes[0][0]))
                _add_colorbar(fig, ax, image)
            timestamp = BASE_DATE + timedelta(days=start_label(start_year) - 1 + day)
            fig.suptitle(f"Поля переменных — {timestamp:%d.%m.%Y}")
            outputs.append(save_figure(
                fig,
                plot_path(root, "Variables", folder, f"{start_year}-{end_year}_{day:05d}.png"),
                dpi=200,
            ))
            plt.close(fig)
    return outputs


def plot_mean_year(files_path_prefix, category, name, start_year, end_year,
                   *, day_indices=(45, 105, 165, 225, 285, 345),
                   pair=None, method=None, n_bins_2d=None,
                   variable=None, n_bins_1d=None):
    """Display six representative calendar dates of an annual climatology."""
    root = Path(files_path_prefix)
    identity = dict(pair=pair, method=method, n_bins_2d=n_bins_2d,
                    variable=variable, n_bins_1d=n_bins_1d)
    parts = mean_year_parts(category, **identity)
    source, _ = mean_year_paths(root, category, name,
                                years=(start_year, end_year), **identity)
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing annual climatology: {source}. Run mean_year in main.py "
            "with matching period, name, pair, method and bin count."
        )
    data = np.load(source, mmap_mode="r", allow_pickle=False)
    context = ""
    if category == "Coefficients_1d":
        context = f" — {NAMES_EN.get(variable, variable)} ({method.upper()}, {n_bins_1d} bins)"
    elif category in ("Coefficients", "Derived", "Eigenvalues"):
        context = f" — {pair[0]}-{pair[1]}"
        if category != "Eigenvalues":
            context += f" ({method.upper()}, {n_bins_2d} bins)"
    outputs = []
    if data.ndim < 3:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(len(data)), data)
        ax.set_xlabel("Day of climatological year")
        ax.set_ylabel("Mean value")
        fig.suptitle(f"Annual climatology: {NAMES_EN.get(name, name)}{context}")
        fig.tight_layout()
        outputs.append(save_figure(fig, plot_path(root, "Mean Year", *parts,
                                                   f"{name}_{start_year}-{end_year}.png")))
        plt.close(fig)
    else:
        channels = (data[:, channel] for channel in range(data.shape[1])) if data.ndim == 4 else (data,)
        for channel_number, channel in enumerate(channels):
            fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
            for ax, index in zip(axes.flat, day_indices):
                image = _draw_map(ax, channel[index], f"День {index + 1}", name=name,
                                  show_y_labels=(ax in axes[:, 0]))
                _add_colorbar(fig, ax, image)
            description = (f" — component {channel_number + 1}" if data.ndim == 4 else "")
            fig.suptitle(f"Annual climatology: {NAMES_EN.get(name, name)}{context}{description}")
            suffix = f"_ch{channel_number + 1}" if data.ndim == 4 else ""
            outputs.append(save_figure(fig, plot_path(root, "Mean Year", *parts,
                                                       f"{name}_{start_year}-{end_year}{suffix}.png")))
            plt.close(fig)
    return outputs


def coefficient_frames(files_path_prefix, names, n_bins_2d, method, coefficient,
                       *, start=None, stop=None, step=1):
    """Generate an ordered PNG sequence from previously calculated coefficients."""
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
        fig, axes = plt.subplots(1, len(channels), figsize=(5.8 * len(channels), 5.2), squeeze=False,
                                 constrained_layout=True)
        for j, (ax, field) in enumerate(zip(axes[0], channels)):
            image = _draw_map(ax, field,
                              f"{NAMES_EN.get(coefficient, coefficient)} — component {j + 1}",
                              name=coefficient, show_y_labels=(j == 0))
            _add_colorbar(fig, ax, image)
        saved = save_figure(fig, plot_path(root, "Coefficients", coefficient,
                                           f"{names[0]}-{names[1]}", method,
                                           f"frame_{i:05d}.png"))
        frames.append(saved)
        plt.close(fig)
    return frames
