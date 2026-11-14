"""Geographical ocean maps with stable interval-wide historical color scales.

The plotting functions use the exact range of days requested by the caller,
not one day's extreme values, to construct their normalization. This makes
consecutive images suitable for assembly into videos without color flicker.
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
import numpy as np

from Data_processing.io import BASE_DATE, load_field, read_ocean_mask, start_label
from Plotting.paths import plot_path, save_figure

NAMES_RU = {
    "sensible": "Явный тепловой поток",
    "latent": "Скрытый тепловой поток",
    "Flux": "Суммарный тепловой поток",
    "SST": "Температура поверхности моря",
    "Pressure": "Атмосферное давление",
    "A": "Коэффициент A",
    "B": "Коэффициент B",
    "C": "Коэффициент C",
    "F": "Коэффициент F",
    "FS": "Коэффициент FS",
}


def _water_mask(root, shape):
    """Read ocean mask or infer land from invalid data if no mask exists."""
    try:
        return np.asarray(read_ocean_mask(root, shape), dtype=bool)
    except FileNotFoundError:
        return np.ones(shape, dtype=bool)


def _masked(field, water):
    """Mask land and invalid values so the colormap's bad color paints land."""
    array = np.asarray(field)
    if array.shape != water.shape:
        raise ValueError(f"Field shape {array.shape} != land-mask shape {water.shape}")
    return np.ma.array(array, mask=(~water | ~np.isfinite(array)))


def _extrema(series, start, stop, water, channel=None, chunk_size=24):
    """Find finite extrema across an entire half-open interval without copying it.

    Parameters
    ----------
    series : array-like
        Either (T,H,W) or (T,C,H,W) series, possibly memory mapped.
    start, stop : int
        Half-open temporal bounds.
    water : ndarray
        Ocean mask (H,W).
    channel : int or None
        Channel selection for 4D coefficient series.
    chunk_size : int
        Maximum observations scanned per block.

    Returns
    -------
    tuple[float, float]
        Minimum and maximum over valid ocean data.
    """
    low, high = np.inf, -np.inf
    for left in range(start, stop, chunk_size):
        block = series[left:min(left + chunk_size, stop)]
        if channel is not None:
            block = block[:, channel]
        values = np.asarray(block)
        valid = np.isfinite(values) & water
        if np.any(valid):
            low = min(low, float(values[valid].min()))
            high = max(high, float(values[valid].max()))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"No finite water values in requested days {start}:{stop}")
    return low, high


def _historical_palette(kind, minimum, maximum):
    """Build the original map palette and one fixed normalization.

    Parameters
    ----------
    kind : str
        Flux, sensible, latent, A, B, C, F, FS, SST or Pressure.
    minimum, maximum : float
        Extrema across the requested plotting interval, not one frame.

    Returns
    -------
    tuple
        Matplotlib colormap and normalization objects.
    """
    if kind in ("sensible", "latent", "Flux", "A"):
        cmap = LinearSegmentedColormap.from_list(
            "ocean_blue_white_red", ["#000080", "#ffffff", "#ff0000"], N=257
        )
        # Zero is pure white even for asymmetric signed data limits.
        # An epsilon-sized opposite-side range keeps TwoSlopeNorm well-defined.
        scale = max(abs(minimum), abs(maximum), 1e-12)
        vmin = minimum if minimum < 0 else -scale * 1e-6
        vmax = maximum if maximum > 0 else scale * 1e-6
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    elif kind == "B":
        cmap = plt.get_cmap("Reds").copy()
        norm = Normalize(vmin=0.0, vmax=max(0.0, maximum, 1e-12), clip=True)
    elif kind == "C":
        cmap = LinearSegmentedColormap.from_list(
            "ocean_correlation", list(zip(
                (0.0, 0.4, 0.6, 1.0),
                ("#4073ff", "#ffffff", "#ffffff", "#db4035"),
            ))
        )
        norm = Normalize(vmin=-1.0, vmax=1.0, clip=True)
    elif kind in ("F", "FS"):
        cmap = ListedColormap(["MintCream", "Aquamarine", "blue", "red", "DarkRed"])
        bounds = [0.0, 0.25, 0.5, 1.0, 1.5, max(2.0, maximum)]
        norm = BoundaryNorm(bounds, cmap.N, clip=True)
    elif kind == "SST":
        cmap = plt.get_cmap("Wistia").copy()
        norm = Normalize(vmin=minimum, vmax=maximum if maximum > minimum else minimum + 1e-12)
    elif kind == "Pressure":
        cmap = LinearSegmentedColormap.from_list(
            "pressure_skyblue_purple", ["#87ceeb", "#800080"]
        )
        norm = Normalize(vmin=minimum, vmax=maximum if maximum > minimum else minimum + 1e-12)
    else:
        cmap = plt.get_cmap("viridis").copy()
        norm = Normalize(vmin=minimum, vmax=maximum if maximum > minimum else minimum + 1e-12)
    cmap.set_bad("lightgreen", 1.0)
    return cmap, norm


def _draw_map(ax, field, water, label, palette):
    """Draw one map with Russian axes, masked land and a persistent palette."""
    cmap, norm = palette
    image = ax.imshow(_masked(field, water), origin="upper",
                      interpolation="none", cmap=cmap, norm=norm)
    ax.set_title(label)
    # ax.set_title("Spatial field")
    ax.set_xlabel("Долгота")
    # ax.set_xlabel("Longitude")
    ax.set_ylabel("Широта")
    # ax.set_ylabel("Latitude")
    return image


def _validate_interval(length, start, stop, step):
    """Validate and normalize half-open plotting boundaries."""
    if start is None:
        start = 0
    if stop is None:
        stop = length
    if not isinstance(start, int) or not isinstance(stop, int) or not isinstance(step, int):
        raise TypeError("start, stop and step must be integers")
    if not 0 <= start < stop <= length or step < 1:
        raise ValueError(f"Invalid interval {start}:{stop}:{step} for {length} observations")
    return start, stop


def plot_variables(files_path_prefix, start_year, end_year, *,
                   day=0, start=None, stop=None, step=1,
                   groups=("Sensible-Latent", "Flux-SST-Pressure")):
    """Plot flux maps and Flux/SST/Pressure maps using fixed interval colors.

    Parameters
    ----------
    files_path_prefix : str or Path
        Root data directory.
    start_year, end_year : int
        Raw-field filename period.
    day : int
        Single zero-based day, if start/stop are omitted.
    start, stop : int or None
        Half-open frame interval. The color limits use all days in this range.
    step : int
        Step between frames.
    groups : sequence[str]
        Requested groups: Sensible-Latent and/or Flux-SST-Pressure.

    Returns
    -------
    list[pathlib.Path]
        PNG outputs in chronological order.
    """
    root = Path(files_path_prefix)
    definitions = {
        "Sensible-Latent": ("sensible", "latent"),
        "Flux-SST-Pressure": ("Flux", "SST", "Pressure"),
    }
    outputs = []
    for group in groups:
        if group not in definitions:
            raise ValueError(f"Unknown variable group: {group}")
        names = definitions[group]
        fields = [load_field(root, name, start_year, end_year) for name in names]
        if len({len(field) for field in fields}) != 1:
            raise ValueError(f"Different series lengths for {group}")
        length = len(fields[0])
        first, last = _validate_interval(length, day if start is None else start,
                                         (day + 1) if stop is None and start is None else stop,
                                         step)
        water = _water_mask(root, fields[0].shape[-2:])
        color_scales = [
            _historical_palette(name, *_extrema(field, first, last, water))
            for name, field in zip(names, fields)
        ]
        for t in range(first, last, step):
            fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False)
            for ax, name, field, palette in zip(axes[0], names, fields, color_scales):
                image = _draw_map(ax, field[t], water, NAMES_RU.get(name, name), palette)
                fig.colorbar(image, ax=ax, shrink=0.8)
            stamp = BASE_DATE + timedelta(days=start_label(start_year) - 1 + t)
            fig.suptitle(f"Поля океанографических переменных — {stamp:%d.%m.%Y}")
            # fig.suptitle("Oceanographic variables by date")
            fig.tight_layout()
            path = plot_path(root, "Variables", group, f"{start_year}-{end_year}_{t:05d}.png")
            outputs.append(save_figure(fig, path, dpi=200))
            plt.close(fig)
    return outputs


def _collected_file(root, names, n_bins_2d, method, coefficient):
    """Find one canonical or historical continuous coefficient array."""
    if method not in ("np", "sp"):
        raise ValueError("method must be 'np' or 'sp'")
    root = Path(root)
    pair = f"{names[0]}_{names[1]}"
    flat_candidates = list((root / "Coefficients").glob(
        f"*-*/{pair}/{coefficient}_*-*_{method}.npy"
    ))
    if len(flat_candidates) > 1:
        # raise FileExistsError(f"More than one continuous {coefficient} series: {flat_candidates}")
        return flat_candidates[0]
    if len(flat_candidates) == 1:
        return flat_candidates[0]
    folder = "Nonparametric_2d" if method == "np" else "Semiparametric_2d"
    legacy = (root / "Coefficients" / "continuous" / f"2d_bins_{n_bins_2d}" /
              pair / folder)
    files = list(legacy.glob(f"{coefficient}_*-*.npy"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one {coefficient} continuous file in {legacy}; found {files}"
        )
    return files[0]


def _first_coefficient_label(filename, coefficient, root, n_bins_2d, names):
    """Read the first transition label from a numeric filename or manifest."""
    match = re.fullmatch(rf"{re.escape(coefficient)}_(\d+)-\d+\.npy", filename.name)
    if match:
        return int(match.group(1))
    manifest = (Path(root) / "Coefficients" / "continuous" /
                f"manifest_{names[0]}_{names[1]}_{n_bins_2d}.json")
    if manifest.is_file():
        import json
        return int(json.loads(manifest.read_text(encoding="utf-8"))["first_label"])
    year = int(filename.parent.parent.name.split("-")[0])
    return start_label(year) + 1


def coefficient_frames(files_path_prefix, names, n_bins_2d, method, coefficient,
                       *, start=None, stop=None, step=1):
    """Generate coefficient frames with one normalization per requested interval.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root.
    names : tuple[str, str]
        Variable pair.
    n_bins_2d : int
        Quantile bins used in historical filenames.
    method : {'np', 'sp'}
        Estimator name.
    coefficient : {'A', 'B', 'C', 'F', 'FS'}
        Coefficient to map.
    start, stop : int or None
        Zero-based half-open frame bounds for color calibration.
    step : int
        Number of days between frames.

    Returns
    -------
    list[pathlib.Path]
        Saved frame paths.
    """
    if coefficient not in ("A", "B", "C", "F", "FS"):
        raise ValueError(f"Unknown coefficient: {coefficient}")
    root = Path(files_path_prefix)
    filename = _collected_file(root, names, n_bins_2d, method, coefficient)
    data = np.load(filename, mmap_mode="r", allow_pickle=False)
    start, stop = _validate_interval(len(data), start, stop, step)
    if data.ndim not in (3, 4):
        raise ValueError(f"Expected (T,H,W) or (T,C,H,W), received {data.shape}")
    water = _water_mask(root, data.shape[-2:])
    channels = 1 if data.ndim == 3 else data.shape[1]
    palettes = []
    for ch in range(channels):
        extrema = _extrema(data, start, stop, water, channel=ch if data.ndim == 4 else None)
        palettes.append(_historical_palette(coefficient, *extrema))
    first_label = _first_coefficient_label(filename, coefficient, root, n_bins_2d, names)
    frames = []
    for t in range(start, stop, step):
        fields = [data[t]] if data.ndim == 3 else [data[t, i] for i in range(channels)]
        fig, axes = plt.subplots(1, channels, figsize=(5.5 * channels, 5), squeeze=False)
        for ch, (ax, field, palette) in enumerate(zip(axes[0], fields, palettes)):
            label = (f"{NAMES_RU.get(coefficient, coefficient)} — компонент {ch+1}"
                     if channels > 1 else NAMES_RU.get(coefficient, coefficient))
            image = _draw_map(ax, field, water, label, palette)
            fig.colorbar(image, ax=ax, shrink=0.8)
        stamp = BASE_DATE + timedelta(days=first_label + t - 1)
        fig.suptitle(f"{NAMES_RU.get(coefficient, coefficient)} — {stamp:%d.%m.%Y}")
        fig.tight_layout()
        path = plot_path(root, "Coefficients", coefficient,
                         f"{names[0]}-{names[1]}", method, f"frame_{t:05d}.png")
        frames.append(save_figure(fig, path))
        plt.close(fig)
    return frames


def plot_mean_year(files_path_prefix, category, name, start_year, end_year,
                   *, day_indices=(45, 105, 165, 225, 285, 345)):
    """Draw mean-year maps using one palette for all selected dates.

    Parameters
    ----------
    files_path_prefix : str or Path
        Data root.
    category : str
        Mean-year category.
    name : str
        Variable or coefficient identifier.
    start_year, end_year : int
        Period filename labels.
    day_indices : sequence[int]
        Selected zero-based climatological days.

    Returns
    -------
    list[pathlib.Path]
        Paths of generated mean-year figures.
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
        fig.tight_layout()
        path = plot_path(root, "Mean Year", category, f"{name}_{start_year}-{end_year}.png")
        outputs.append(save_figure(fig, path))
        plt.close(fig)
        return outputs
    indices = tuple(day_indices)
    if not indices or any(not 0 <= idx < len(data) for idx in indices):
        raise IndexError("Mean-year day index is outside the available series")
    if data.ndim not in (3, 4):
        raise ValueError(f"Unexpected mean-year array shape: {data.shape}")
    water = _water_mask(root, data.shape[-2:])
    n_channels = 1 if data.ndim == 3 else data.shape[1]
    for ch in range(n_channels):
        channel_series = data if data.ndim == 3 else data[:, ch]
        low = high = None
        for index in indices:
            values = np.asarray(channel_series[index])
            valid = np.isfinite(values) & water
            if not np.any(valid):
                continue
            vmin, vmax = float(values[valid].min()), float(values[valid].max())
            low = vmin if low is None else min(low, vmin)
            high = vmax if high is None else max(high, vmax)
        if low is None:
            raise ValueError("No finite values for climatological plot")
        palette = _historical_palette(name, low, high)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, index in zip(axes.flat, indices):
            ax.set_visible(True)
            image = _draw_map(ax, channel_series[index], water, f"День {index + 1}", palette)
            fig.colorbar(image, ax=ax, shrink=0.72)
        suffix = f" — компонент {ch+1}" if n_channels > 1 else ""
        fig.suptitle(f"Средний год: {NAMES_RU.get(name, name)}{suffix}")
        fig.tight_layout()
        ending = f"_ch{ch+1}" if n_channels > 1 else ""
        path = plot_path(root, "Mean Year", category,
                         f"{name}_{start_year}-{end_year}{ending}.png")
        outputs.append(save_figure(fig, path))
        plt.close(fig)
    return outputs
