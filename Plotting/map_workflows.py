"""Map-based plotting workflows for cached oceanographic arrays."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
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
LON_LABELS_RU = ["90° з. д.", "60° з. д.", "30° з. д.", "0"]
LAT_LABELS_RU = ["0", "30° с. ш.", "60° с. ш.", "80° с. ш."]
DIVERGING_NAMES = {"sensible", "latent", "Flux", "A", "B", "C", "F", "FS"}
BLUE_WHITE_RED = LinearSegmentedColormap.from_list(
    "blue_white_red", ["#1f3a93", "#ffffff", "#c1121f"], N=256
)
RU_MONTHS_GENITIVE = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


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
    ax.set_xticklabels(LON_LABELS_RU if show_x_labels else [])
    ax.set_yticklabels(LAT_LABELS_RU if show_y_labels else [])
    ax.tick_params(axis="both", labelsize=9, pad=2)
    ax.grid(True, color="#bcbcbc", alpha=0.45, linewidth=0.7)


def _calendar_label(day_index: int) -> str:
    """Return a day-month label in a no-leap 365-day climatological year."""
    if not 0 <= int(day_index) < 365:
        raise ValueError(f"Expected a climatological day in 0..364, got {day_index}")
    stamp = date(2001, 1, 1) + timedelta(days=int(day_index))
    return f"{stamp.day} {RU_MONTHS_GENITIVE[stamp.month - 1]}"


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
        limit = max(abs(vmin), abs(vmax))
        return mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _map_cmap(name=""):
    """Choose a variable-specific palette for map plots."""
    if name in DIVERGING_NAMES:
        return BLUE_WHITE_RED
    if name == "SST":
        return "YlOrBr"
    if name == "Pressure":
        return "BuPu"
    return "viridis"


def _draw_map(ax, data, title, *, name="",
              show_x_labels=True, show_y_labels=True):
    """Draw one geographic grid with coordinate axes and matching colorbar."""
    masked = np.ma.masked_invalid(data)
    image = ax.imshow(
        masked,
        cmap=_map_cmap(name),
        norm=_data_norm(masked, name),
        origin="upper",
        extent=(*LON_BOUNDS, *LAT_BOUNDS),
        interpolation="none",
        aspect="auto",
    )
    image.cmap.set_bad("lightgreen")
    ax.set_title(title, fontsize=13, pad=7)
    _format_geo_axis(ax, show_x_labels=show_x_labels, show_y_labels=show_y_labels)
    return image


def _add_colorbar(fig, ax, image):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.0%", pad=0.08)
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.ax.tick_params(labelsize=9, pad=2)
    return colorbar


def _mean_year_context(category, name, *, pair=None, method=None,
                       n_bins_2d=None, variable=None, n_bins_1d=None):
    if category == "Coefficients_1d":
        return (f"{NAMES_RU.get(name, name)} — {NAMES_RU.get(variable, variable)} "
                f"({method.upper()}, {n_bins_1d} групп)")
    if category in ("Coefficients", "Derived"):
        return (f"{NAMES_RU.get(name, name)} — {NAMES_RU.get(pair[0], pair[0])}-"
                f"{NAMES_RU.get(pair[1], pair[1])} ({method.upper()}, {n_bins_2d} групп)")
    if category == "Eigenvalues":
        return (f"{NAMES_RU.get(name, name)} — {NAMES_RU.get(pair[0], pair[0])}-"
                f"{NAMES_RU.get(pair[1], pair[1])}")
    return NAMES_RU.get(name, name)


def _plot_six_maps(data_cube, name, title, *, saved_to, dpi=300):
    """Render six climatological snapshots in the lambdas/eigenvalues style."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False,
                             constrained_layout=False)
    fig.subplots_adjust(left=0.04, right=0.968, bottom=0.065, top=0.90,
                        wspace=0.17, hspace=0.14)
    day_indices = (45, 104, 165, 226, 287, 348)
    for i, (ax, index) in enumerate(zip(axes.flat, day_indices)):
        row, col = divmod(i, 3)
        image = _draw_map(
            ax,
            data_cube[index],
            _calendar_label(index),
            name=name,
            show_x_labels=(row == 1),
            show_y_labels=(col == 0),
        )
        _add_colorbar(fig, ax, image)
    fig.suptitle(title, fontsize=17, y=0.965)
    output = save_figure(fig, saved_to, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output


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
                   *, day_indices=(45, 104, 165, 226, 287, 348),
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
    title = f"Средний год: {_mean_year_context(category, name, pair=pair, method=method, n_bins_2d=n_bins_2d, variable=variable, n_bins_1d=n_bins_1d)}"
    outputs = []
    if data.ndim < 3:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(len(data)), data)
        ax.set_xlabel("День климатологического года")
        ax.set_ylabel("Среднее значение")
        fig.suptitle(title)
        fig.tight_layout()
        outputs.append(save_figure(fig, plot_path(root, "Mean Year", *parts,
                                                   f"{name}_{start_year}-{end_year}.png")))
        plt.close(fig)
    else:
        if data.ndim == 4:
            for channel_number in range(data.shape[1]):
                channel = data[:, channel_number]
                channel_title = f"{title} — компонента {channel_number + 1}"
                suffix = f"_ch{channel_number + 1}"
                outputs.append(_plot_six_maps(
                    channel,
                    name,
                    channel_title,
                    saved_to=plot_path(root, "Mean Year", *parts,
                                       f"{name}_{start_year}-{end_year}{suffix}.png"),
                ))
        else:
            outputs.append(_plot_six_maps(
                data,
                name,
                title,
                saved_to=plot_path(root, "Mean Year", *parts,
                                   f"{name}_{start_year}-{end_year}.png"),
            ))
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
