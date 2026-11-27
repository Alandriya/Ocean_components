"""Map-based plotting workflows for cached oceanographic arrays.

Coefficient A frames default to a fixed, signed nonlinear colour scale,
physical-value isolines and shared limits across the requested time interval.
Other map workflows and other coefficient styles remain unchanged.
"""

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
    """Format geo axis for the map workflows workflow."""
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


def _data_norm(data, name, *, vmin=None, vmax=None):
    """Build a normalization, optionally using explicit fixed color limits."""
    if vmin is not None or vmax is not None:
        if vmin is None or vmax is None or not vmin < vmax:
            raise ValueError("Both vmin and vmax must be provided with vmin < vmax")
        if vmin < 0.0 < vmax:
            return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    finite = np.asarray(data, dtype=float)[np.isfinite(data)]
    if finite.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    if data_min == data_max:
        delta = 1.0 if data_min == 0.0 else 0.01 * abs(data_min)
        data_min -= delta
        data_max += delta
    if name in DIVERGING_NAMES and data_min < 0.0 < data_max:
        limit = max(abs(data_min), abs(data_max))
        return mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    return mcolors.Normalize(vmin=data_min, vmax=data_max)


def _map_cmap(name=""):
    """Choose a variable-specific palette for map plots."""
    if name in DIVERGING_NAMES:
        return BLUE_WHITE_RED
    if name == "SST":
        return "YlOrBr"
    if name == "Pressure":
        return "BuPu"
    return "viridis"


def _draw_map(ax, data, title, *, name="", cmap=None, vmin=None, vmax=None,
              show_x_labels=True, show_y_labels=True):
    """Draw one geographic grid with coordinate axes and matching colorbar."""
    masked = np.ma.masked_invalid(data)
    image = ax.imshow(
        masked,
        cmap=_map_cmap(name) if cmap is None else cmap,
        norm=_data_norm(masked, name, vmin=vmin, vmax=vmax),
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
    """Compute add colorbar for map workflows from ``fig``, ``ax``, ``image``."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.2%", pad=0.05)
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.ax.tick_params(labelsize=9, pad=2)
    return colorbar


def _mean_year_context(category, name, *, pair=None, method=None,
                       n_bins_2d=None, variable=None, n_bins_1d=None):
    """Compute mean year context for map workflows from ``category``, ``name``, ``pair``, ``method``, ``n_bins_2d``."""
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


def _plot_six_maps(data_cube, name, title, *, saved_to, dpi=300,
                   cmap=None, vmin=None, vmax=None):
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
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
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
                    cmap="Reds" if name == "B" else None,
                    vmin=-1.0 if name == "C" else None,
                    vmax=1.0 if name == "C" else None,
                ))
        else:
            outputs.append(_plot_six_maps(
                data,
                name,
                title,
                saved_to=plot_path(root, "Mean Year", *parts,
                                   f"{name}_{start_year}-{end_year}.png"),
                cmap="Reds" if name == "B" else None,
                vmin=-1.0 if name == "C" else None,
                vmax=1.0 if name == "C" else None,
            ))
    return outputs


# -----------------------------------------------------------------------------
# Coefficient A: interval-wide limits and sign-separated nonlinear colors.
# Functions are private helpers inside the existing plotting module.
# -----------------------------------------------------------------------------


def _coefficient_a_extrema(data, start: int, stop: int, water_mask=None,
                           chunk_days: int = 12):
    """Exact min/max for each channel over ALL selected days and ocean pixels.

    The 'step' used for frame rendering is intentionally NOT used here.
    Chunked access allows the collected .npy array to remain memory mapped.
    """
    nch = data.shape[1] if data.ndim == 4 else 1
    lo = np.full(nch, np.inf, dtype=float)
    hi = np.full(nch, -np.inf, dtype=float)
    for left in range(start, stop, chunk_days):
        arr = np.asarray(data[left:min(left + chunk_days, stop)], dtype=np.float64)
        if data.ndim == 3:
            arr = arr[:, None, :, :]
        if water_mask is not None:
            arr = np.where(water_mask[None, None, :, :], arr, np.nan)
        for j in range(nch):
            vals = arr[:, j]
            finite = vals[np.isfinite(vals)]
            if finite.size:
                lo[j] = min(lo[j], float(finite.min()))
                hi[j] = max(hi[j], float(finite.max()))
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("No finite coefficient A values in the selected interval")
    return [(float(lo[j]), float(hi[j])) for j in range(nch)]


def _coefficient_a_signed_scale(lo: float, hi: float,
                                gamma_negative: float = 0.5,
                                gamma_positive: float = 0.5):
    """Map signed physical values nonlinearly to blue/white/red display colors.

    Each side is scaled by its OWN global extreme. 0 is always white.
    Gamma < 1 allocates more color distance to small absolute coefficients.
    Both forward and inverse are supplied, so Matplotlib's colorbar is truthful.
    """
    for label, gamma in (("gamma_negative", gamma_negative),
                         ("gamma_positive", gamma_positive)):
        if not (np.isfinite(gamma) and gamma > 0):
            raise ValueError(f"{label} must be a finite positive number")
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        raise ValueError(f"Bad coefficient A range {lo}..{hi}")

    # Handle one-signed channels explicitly: zero is the white color anchor.
    if lo < 0 < hi:
        neg, pos = -lo, hi
        palette = mcolors.LinearSegmentedColormap.from_list(
            "A_signed", ["#224f93", "#ffffff", "#b52e38"], N=1025
        )

        def forward(x):
            """Apply the model transformation to the input tensors."""
            x = np.asarray(x, dtype=float)
            y = np.empty_like(x)
            mask = x < 0
            y[mask] = 0.5 * (1.0 - (np.maximum(-x[mask], 0) / neg) ** gamma_negative)
            y[~mask] = 0.5 * (1.0 + (np.maximum(x[~mask], 0) / pos) ** gamma_positive)
            return y

        def inverse(y):
            """Transform values between the data and display coordinates."""
            y = np.asarray(y, dtype=float)
            x = np.empty_like(y)
            mask = y < 0.5
            x[mask] = -neg * (np.maximum(1.0 - 2.0 * y[mask], 0)) ** (1.0 / gamma_negative)
            x[~mask] = pos * (np.maximum(2.0 * y[~mask] - 1.0, 0)) ** (1.0 / gamma_positive)
            return x

        minimum, maximum = lo, hi
    elif hi > 0:
        pos = hi
        palette = mcolors.LinearSegmentedColormap.from_list(
            "A_positive", ["#ffffff", "#b52e38"], N=513
        )

        def forward(x):
            """Apply the model transformation to the input tensors."""
            return (np.maximum(np.asarray(x, dtype=float), 0.0) / pos) ** gamma_positive

        def inverse(y):
            """Transform values between the data and display coordinates."""
            return pos * np.maximum(np.asarray(y, dtype=float), 0.0) ** (1.0 / gamma_positive)

        minimum, maximum = 0.0, hi
    elif lo < 0:
        neg = -lo
        palette = mcolors.LinearSegmentedColormap.from_list(
            "A_negative", ["#224f93", "#ffffff"], N=513
        )

        def forward(x):
            """Apply the model transformation to the input tensors."""
            return 1.0 - (np.maximum(-np.asarray(x, dtype=float), 0.0) / neg) ** gamma_negative

        def inverse(y):
            """Transform values between the data and display coordinates."""
            return -neg * np.maximum(1.0 - np.asarray(y, dtype=float), 0.0) ** (1.0 / gamma_negative)

        minimum, maximum = lo, 0.0
    else:
        palette = mcolors.LinearSegmentedColormap.from_list(
            "A_zero", ["#224f93", "#ffffff", "#b52e38"], N=1025
        )
        return mcolors.Normalize(vmin=-1.0, vmax=1.0), palette, (-1.0, 1.0)

    palette.set_bad("#dbe8cf")  # land / masked pixels: soft light green
    norm = mcolors.FuncNorm((forward, inverse), vmin=minimum, vmax=maximum, clip=True)
    return norm, palette, (minimum, maximum)


def _coefficient_a_contour_levels(lo: float, hi: float,
                                  gamma_negative: float, gamma_positive: float,
                                  levels_per_side: int):
    """Contours denote actual A values, with more levels near zero."""
    if levels_per_side < 0:
        raise ValueError("levels_per_side must be >= 0")
    if levels_per_side == 0:
        return np.empty(0, dtype=float)
    fractions = np.linspace(0.22, 0.92, levels_per_side)
    vals = []
    if lo < 0:
        vals.extend((-abs(lo) * fractions ** (1.0 / gamma_negative)).tolist())
    if hi > 0:
        vals.extend((hi * fractions ** (1.0 / gamma_positive)).tolist())
    return np.asarray(sorted(vals), dtype=float)


def _coefficient_a_smooth(field: np.ndarray, sigma: float):
    """Optional NaN-aware smoothing for presentation, not stored estimates."""
    if sigma <= 0:
        return field
    from scipy.ndimage import gaussian_filter  # imported only when requested
    good = np.isfinite(field)
    numerator = gaussian_filter(np.where(good, field, 0), sigma, mode="nearest")
    denominator = gaussian_filter(good.astype(float), sigma, mode="nearest")
    return np.where(good, numerator / np.maximum(denominator, 1e-12), np.nan)


def _coefficient_a_water_mask(root: Path, shape):
    """Use the project mask if available; otherwise rely on NaNs in coefficients."""
    mask_file = root / "DATA" / "mask"
    if not mask_file.is_file():
        mask_file = root / "mask"
    if not mask_file.is_file():
        return None
    size = int(np.prod(shape))
    raw = mask_file.read_bytes()
    if len(raw) < size:
        raise ValueError(f"Ocean mask too short: {mask_file}")
    return np.frombuffer(raw[:size], dtype=np.bool_).reshape(shape).copy()


def _coefficient_a_colorbar(fig, ax, image, norm):
    """Compute coefficient a colorbar for map workflows from ``fig``, ``ax``, ``image``, ``norm``."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.2%", pad=0.05)
    # Uniform visual spacing; tick labels are the real coefficient values.
    ticks = np.asarray(norm.inverse(np.linspace(0, 1, 7)), dtype=float)
    cb = fig.colorbar(image, cax=cax, ticks=ticks)
    cb.ax.tick_params(labelsize=8, pad=1)
    cb.formatter = plt.matplotlib.ticker.FuncFormatter(lambda value, pos: f"{value:.3g}")
    cb.update_ticks()
    return cb


def coefficient_frames(files_path_prefix, names, n_bins_2d, method, coefficient,
                       *, start=None, stop=None, step=1,
                       style="auto", gamma_negative=0.35, gamma_positive=0.35,
                       show_contours=False, contours_per_side=3, smooth_sigma=0.65):
    """Generate PNG maps for precomputed coefficient sequences.

    For coefficient A, by default use:
      - exact finite ocean min/max per channel from every day in [start, stop),
      - the same limits for all frames (even with step > 1),
      - nonlinear signed gamma colors, centered on white at zero,
      - NO coefficient isolines by default (calmer video),
      - optional real-value isolines and a highlighted zero line with
        show_contours=True or style="nonlinear_contours",
      - gamma=0.35 gives higher contrast for small absolute values.

    The original style is preserved for B/C/F/FS and with style='original'.
    """
    root = Path(files_path_prefix)
    folder = "Nonparametric_2d" if method == "np" else "Semiparametric_2d"
    parent = root / "Coefficients" / "continuous" / f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}" / folder
    files = sorted(parent.glob(f"{coefficient}_*-*.npy"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one collected {coefficient} array under {parent}; found {files}")
    data = np.load(files[0], mmap_mode="r", allow_pickle=False)
    stop = len(data) if stop is None else min(int(stop), len(data))
    start = 0 if start is None else int(start)
    if not (0 <= start < stop <= len(data)) or int(step) <= 0:
        raise ValueError(f"Bad frame interval start={start}, stop={stop}, step={step}, length={len(data)}")
    if style not in ("auto", "original", "nonlinear_contours"):
        raise ValueError("style must be 'auto', 'original' or 'nonlinear_contours'")
    alternative = coefficient == "A" and style != "original"
    draw_contours = show_contours or style == "nonlinear_contours"
    if style == "nonlinear_contours" and coefficient != "A":
        raise ValueError("Nonlinear contour style currently supports coefficient A only")

    if alternative:
        if data.ndim not in (3, 4):
            raise ValueError(f"Expected A array [T,H,W] or [T,C,H,W], got {data.shape}")
        channels_num = data.shape[1] if data.ndim == 4 else 1
        grid_shape = data.shape[-2:]
        water_mask = _coefficient_a_water_mask(root, grid_shape)
        bounds = _coefficient_a_extrema(data, start, stop, water_mask)
        styles = [_coefficient_a_signed_scale(lo, hi, gamma_negative, gamma_positive)
                  for lo, hi in bounds]
        if draw_contours:
            iso_levels = [_coefficient_a_contour_levels(lo, hi, gamma_negative,
                                                       gamma_positive, contours_per_side)
                          for lo, hi in bounds]
            lon = np.linspace(*LON_BOUNDS, grid_shape[1])
            lat = np.linspace(LAT_BOUNDS[1], LAT_BOUNDS[0], grid_shape[0])
        print(f"Coefficient A fixed bounds for [{start}, {stop}), including skipped frames:")
        for ch, (lo, hi) in enumerate(bounds):
            print(f"  component {ch + 1}: {lo:.7g} .. {hi:.7g}; gamma-={gamma_negative:g}, gamma+={gamma_positive:g}")

    frames = []
    for i in range(start, stop, step):
        channels = data[i] if data.ndim == 4 else data[i][None, ...]
        fig, axes = plt.subplots(1, len(channels), figsize=(6.55 * len(channels), 5.6), squeeze=False,
                                 constrained_layout=False)
        fig.subplots_adjust(left=0.055, right=0.972, bottom=0.075, top=0.965, wspace=0.14)
        try:
            for j, (ax, field) in enumerate(zip(axes[0], channels)):
                if not alternative:
                    image = _draw_map(ax, field,
                                      f"{NAMES_EN.get(coefficient, coefficient)} — component {j + 1}",
                                      name=coefficient, show_y_labels=(j == 0))
                    _add_colorbar(fig, ax, image)
                    continue

                norm, cmap, _ = styles[j]
                z = np.asarray(field, dtype=float)
                if water_mask is not None:
                    z = np.where(water_mask, z, np.nan)
                z = _coefficient_a_smooth(z, smooth_sigma)
                ax.set_facecolor("#dbe8cf")  # also covers fully masked patches
                image = ax.imshow(np.ma.masked_invalid(z), cmap=cmap, norm=norm,
                                  extent=(*LON_BOUNDS, *LAT_BOUNDS), origin="upper",
                                  interpolation="bilinear", aspect="auto")
                _format_geo_axis(ax, show_y_labels=(j == 0))
                # For videos and figure montages, titles take too much space and are usually added externally.
                # ax.set_title(f"Coefficient A — {names[j] if j < len(names) else 'component ' + str(j + 1)}",
                #              fontsize=12)
                valid = z[np.isfinite(z)]
                if draw_contours and valid.size and contours_per_side > 0:
                    current_levels = iso_levels[j]
                    current_levels = current_levels[(current_levels > valid.min()) & (current_levels < valid.max())]
                    if current_levels.size:
                        ax.contour(lon, lat, z, levels=current_levels,
                                   colors="#5f6770", linestyles="solid",
                                   linewidths=0.38, alpha=0.52)
                        # No contour labels in animation: their positions jump between frames.
                    if valid.min() < 0 < valid.max():
                        ax.contour(lon, lat, z, levels=[0],
                                   colors="#2f343a", linewidths=0.8, alpha=0.75)
                if water_mask is not None and water_mask.any() and not water_mask.all():
                    # This is only the coastline, not an isoline of coefficient A.
                    ax.contour(np.linspace(*LON_BOUNDS, grid_shape[1]),
                               np.linspace(LAT_BOUNDS[1], LAT_BOUNDS[0], grid_shape[0]),
                               water_mask.astype(float), levels=[0.5],
                               colors="#718579", linewidths=0.30, alpha=0.48)
                _coefficient_a_colorbar(fig, ax, image, norm)

            if alternative:
                # Optional overall title disabled by default: it reduces the useful map area in videos.
                pass
            saved = save_figure(fig, plot_path(root, "Coefficients", coefficient,
                                               f"{names[0]}-{names[1]}", method,
                                               f"frame_{i:05d}.png"))
            frames.append(saved)
        finally:
            plt.close(fig)
    return frames
