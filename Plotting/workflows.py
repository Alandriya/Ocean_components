"""High-level plotting workflows reading precomputed oceanographic arrays."""

from __future__ import annotations

import json
import subprocess
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

from Data_processing.io import BASE_DATE, load_field, load_spectral_field, read_ocean_mask, start_label
from Plotting.paths import plot_path


def save_figure(figure, filename, *args, **kwargs):
    """Save English-language figures without Plotting.paths automatic RU translation.

    Plotting.paths.save_figure always calls translate_figure and would replace
    English text back with Russian. Russian captions remain commented below.
    If Russian graphics are needed again, enable that translation explicitly.
    """
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # from Plotting.paths import translate_figure
    # translate_figure(figure)  # Russian-language option (disabled).
    figure.savefig(str(destination), *args, **kwargs)
    return destination

NAMES_EN = {"sensible": "Sensible heat flux", "latent": "Latent heat flux",
            "Flux": "Total heat flux", "SST": "Sea surface temperature",
            "Pressure": "Atmospheric pressure", "A": "Coefficient A",
            "B": "Coefficient B", "C": "Coefficient C",
            "F": "Coefficient F", "FS": "Coefficient FS"}

# Russian names, retained for switching figure labels back to Russian.
NAMES_RU = {"sensible": "Явный тепловой поток", "latent": "Скрытый тепловой поток",
            "Flux": "Суммарный тепловой поток", "SST": "Температура поверхности моря",
            "Pressure": "Атмосферное давление", "A": "Коэффициент A",
            "B": "Коэффициент B", "C": "Коэффициент C", "F": "Коэффициент F"}


def _draw_map(ax, data, title):
    """Draw one geographic grid with English labels; Russian alternatives are kept."""
    im = ax.imshow(np.ma.masked_invalid(data), cmap="coolwarm", origin="upper")
    ax.set_title(title)
    # ax.set_title("Spatial field")
    ax.set_xlabel("Longitude")
    # ax.set_xlabel("Долгота")
    ax.set_ylabel("Latitude")
    # ax.set_ylabel("Широта")
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
            im = _draw_map(ax, array[day], NAMES_EN[name])
            # im = _draw_map(ax, array[day], NAMES_RU[name])
            fig.colorbar(im, ax=ax, shrink=0.8)
        timestamp = BASE_DATE + timedelta(days=start_label(start_year) - 1 + day)
        fig.suptitle(f"Oceanographic variables — {timestamp:%d.%m.%Y}")
        # fig.suptitle(f"Поля океанографических переменных — {timestamp:%d.%m.%Y}")
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
        ax.set_xlabel("Day of climatological year")
        # ax.set_xlabel("День среднего года")
        ax.set_ylabel("Mean value")
        # ax.set_ylabel("Среднее значение")
        fig.suptitle(f"Annual climatology: {NAMES_EN.get(name, name)}")
        # fig.suptitle(f"Средний год: {NAMES_RU.get(name, name)}")
        fig.tight_layout()
        outputs.append(save_figure(fig, plot_path(root, "Mean Year", category,
                                                   f"{name}_{start_year}-{end_year}.png")))
        plt.close(fig)
    else:
        channels = (data[:, channel] for channel in range(data.shape[1])) if data.ndim == 4 else (data,)
        for channel_number, channel in enumerate(channels):
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            for ax, index in zip(axes.flat, day_indices):
                im = _draw_map(ax, channel[index], f"Day {index + 1}")
                # im = _draw_map(ax, channel[index], f"День {index + 1}")
                fig.colorbar(im, ax=ax, shrink=0.72)
            description = (f" — component {channel_number + 1}" if data.ndim == 4 else "")
            # description = (f" — компонент {channel_number + 1}" if data.ndim == 4 else "")
            fig.suptitle(f"Annual climatology: {NAMES_EN.get(name, name)}{description}")
            # fig.suptitle(f"Средний год: {NAMES_RU.get(name, name)}{description}")
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
            img = _draw_map(ax, field, f"{NAMES_EN.get(coefficient, coefficient)} — component {j+1}")
            # img = _draw_map(ax, field, f"{NAMES_RU.get(coefficient, coefficient)} — компонент {j+1}")
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
                    label="Observed histogram")
            # ax.hist(clean, bins=histogram_bins, density=True, alpha=0.5,
            #         label="Гистограмма наблюдений")
            ax.plot(grid, densities[f"pdf_{name}"], linewidth=2,
                    label="Stationary density")
            # ax.plot(grid, densities[f"pdf_{name}"], linewidth=2,
            #         label="Стационарная плотность")
            ax.set_title(NAMES_EN.get(name, name))
            # ax.set_title(NAMES_RU.get(name, name))
            ax.set_xlabel("Variable value")
            # ax.set_xlabel("Значение переменной")
            ax.set_ylabel("Probability density")
            # ax.set_ylabel("Плотность вероятности")
            ax.legend()
    fig.tight_layout()
    output = save_figure(fig, plot_path(root, "Distributions", "Stationary",
                                        f"{names[0]}-{names[1]}_{start_year}-{end_year}.png"))
    plt.close(fig)
    return output


def _resolve_array_spec(root, array_spec):
    """Load an array from .npy or .npz using syntax 'path' or 'path::key'."""
    if array_spec is None:
        return None, None
    key = None
    spec = str(array_spec)
    if '::' in spec:
        spec, key = spec.split('::', 1)
    path = Path(spec)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.exists():
        raise FileNotFoundError(f"Input array not found: {path}")
    if path.suffix.lower() == '.npy':
        array = np.load(path, mmap_mode='r')
        return np.asarray(array), path
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as npz:
            picked_key = key
            if picked_key is None:
                candidates = [name for name in npz.files if not name.startswith('__')]
                if not candidates:
                    raise KeyError(f"No arrays found in {path}")
                picked_key = candidates[0]
            if picked_key not in npz.files:
                raise KeyError(f"Key '{picked_key}' not found in {path}; available: {npz.files}")
            return np.asarray(npz[picked_key]), path
    raise ValueError(f"Unsupported array file: {path}")


def _prepare_map_field(array):
    array = np.asarray(array, dtype=float)
    if array.ndim == 2:
        field = array
    elif array.ndim == 3:
        field = np.nanmean(array, axis=0)
    elif array.ndim == 4:
        field = np.nanmean(array, axis=0)
        if field.ndim == 3:
            field = field[0]
    else:
        raise ValueError(f"Expected 2D/3D/4D array for map contours, got shape {array.shape}")
    return np.squeeze(field)


def _infer_levels(field, variable, n_levels=7):
    finite = np.asarray(field, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot infer contour levels from an all-NaN field")
    presets = {
        'sensible': np.array([-45, -30, -15, 0, 15], dtype=float),
        'latent': np.array([-120, -80, -40, 0, 40, 80], dtype=float),
    }
    if variable in presets:
        return presets[variable]
    lo, hi = np.nanpercentile(finite, [10, 90])
    if lo == hi:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_levels)


def _format_lon_labels(values):
    labels = []
    for value in values:
        if value < 0:
            labels.append(f"{abs(int(value))}W")
        elif value > 0:
            labels.append(f"{int(value)}E")
        else:
            labels.append('0')
    return labels


def _format_lat_labels(values):
    labels = []
    for value in values:
        if value == 0:
            labels.append('EQ')
        elif value > 0:
            labels.append(f"{int(value)}N")
        else:
            labels.append(f"{abs(int(value))}S")
    return labels


def plot_stationary_isolines(files_path_prefix, variable, start_year, end_year, *,
                             method1_array=None, method2_array=None,
                             method1_label='Method 1', method2_label='2D method',
                             method1_levels=None, method2_levels=None,
                             lon_bounds=(-90.0, 0.0), lat_bounds=(0.0, 80.0),
                             ocean_green='#86d883'):
    """Draw legacy-style geographic isolines for one variable.

    The task is intentionally flexible: each method can be provided either as
    an explicit .npy/.npz path or, for method 1, omitted to use the temporal
    mean of the grouped ERA5 field over the requested period. If a supplied
    array is 3D, the time mean is used automatically.
    """
    root = Path(files_path_prefix)
    outputs = []

    if method1_array is None:
        method1_source = root / 'DATA' / 'Fluxes' / f'{variable}_{start_year}-{end_year}_grouped.npy'
        if method1_source.exists():
            method1 = _prepare_map_field(np.load(method1_source, mmap_mode='r'))
        else:
            method1 = _prepare_map_field(load_field(root, variable, start_year, end_year))
            method1_source = Path(f'load_field({variable})')
    else:
        method1, method1_source = _resolve_array_spec(root, method1_array)
        method1 = _prepare_map_field(method1)

    if method2_array is None:
        raise ValueError(
            'method2_array must point to the two-dimensional estimate to compare. '            'Use syntax like "Functional/some_file.npy" or "file.npz::array_key".')
    method2, method2_source = _resolve_array_spec(root, method2_array)
    method2 = _prepare_map_field(method2)

    if method1.shape != method2.shape:
        raise ValueError(f'Method arrays must have identical 2D shapes: {method1.shape} vs {method2.shape}')

    mask = np.asarray(read_ocean_mask(root), dtype=bool)
    if mask.shape != method1.shape:
        mask = np.ones_like(method1, dtype=bool)

    lon = np.linspace(lon_bounds[0], lon_bounds[1], method1.shape[1])
    lat = np.linspace(lat_bounds[0], lat_bounds[1], method1.shape[0])
    xx, yy = np.meshgrid(lon, lat)
    field1 = np.where(mask, method1, np.nan)
    field2 = np.where(mask, method2, np.nan)

    levels1 = np.asarray(method1_levels if method1_levels is not None else _infer_levels(field1, variable), dtype=float)
    levels2 = np.asarray(method2_levels if method2_levels is not None else levels1, dtype=float)
    primary_palette = ['#ffd84d', '#f7c600', '#f0ad00', '#ff8c00', '#ff5a00', '#ff3300']
    secondary_palette = ['#6a3d9a', '#7b51a8', '#5e4fa2', '#4967b0', '#2c7fb8', '#225ea8']
    colors1 = [primary_palette[min(i, len(primary_palette)-1)] for i in range(len(levels1))]
    colors2 = [secondary_palette[min(i, len(secondary_palette)-1)] for i in range(len(levels2))]

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    try:
        # Background: ocean light gray, land light green.
        background = np.where(mask, 0.0, 1.0)
        bg_cmap = mcolors.ListedColormap(['#e6e6e6', ocean_green])
        ax.imshow(background, origin='lower', extent=(lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]),
                  cmap=bg_cmap, interpolation='nearest', aspect='auto', zorder=0)

        cs1 = ax.contour(xx, yy, field1, levels=levels1, colors=colors1, linewidths=1.6, zorder=2)
        ax.clabel(cs1, inline=True, fmt='%g', fontsize=8)
        cs2 = ax.contour(xx, yy, field2, levels=levels2, colors=colors2, linewidths=1.6, zorder=3)
        ax.clabel(cs2, inline=True, fmt='%g', fontsize=8)

        ax.set_title(f'Isolines of {NAMES_EN.get(variable, variable)}')
        # ax.set_title(f'Изолинии: {NAMES_RU.get(variable, variable)}')
        ax.set_xlim(*lon_bounds)
        ax.set_ylim(*lat_bounds)
        xticks = [-90, -60, -30, 0]
        yticks = [0, 30, 60, 80]
        ax.set_xticks(xticks)
        ax.set_xticklabels(_format_lon_labels(xticks))
        ax.set_yticks(yticks)
        ax.set_yticklabels(_format_lat_labels(yticks))
        ax.grid(True, color='#b0b0b0', alpha=0.7, linewidth=0.6)
        handles = [
            plt.Line2D([0], [0], color=colors1[min(len(colors1)-1, max(0, len(colors1)//2))], lw=2.0, label=method1_label),
            plt.Line2D([0], [0], color=colors2[min(len(colors2)-1, max(0, len(colors2)//2))], lw=2.0, label=method2_label),
        ]
        ax.legend(handles=handles, loc='lower left')
        filename = f'{variable}_isolines_compare_eng.png'
        output = save_figure(fig, plot_path(root, 'Figures', 'Isolines', filename), dpi=300)
        outputs.append(output)
        print(f"[isolines] Saved: {output}", flush=True)
        print(f"[isolines] Method 1 source: {method1_source}", flush=True)
        print(f"[isolines] Method 2 source: {method2_source}", flush=True)
    finally:
        plt.close(fig)
    return outputs


def plot_copula_results(files_path_prefix, names, start_year, end_year):
    """Save every copula figure independently, using English figure text.

    Existing copula/FP NPZ files are read-only; all statistical calculations
    have already been performed. The legacy YlOrRd palette and zoom intervals
    are retained. Russian alternatives are commented at each label.

    Returns a list of independent image paths under
    Plots/Copulas/Figures/<pair>_<period>/, with *_eng.png filenames.
    This intentionally does not create composite / subplot-grid figures.
    """
    root = Path(files_path_prefix)
    pair = f"{names[0]}-{names[1]}"
    period = f"{start_year}-{end_year}"
    source = root / "Functional" / "Copulas" / f"{pair}_{period}.npz"
    if not source.is_file():
        raise FileNotFoundError(
            f"Saved copula models not found: {source}; run task='copulas' first")

    # Filenames follow the individual scientific-figure examples in Figures(1).zip.
    folder = root / "Plots" / "Copulas" / "Figures" / f"{pair}_{period}"
    folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    cmap = "YlOrRd"  # Keep the original yellow/orange/red palette.
    display_ranges = {"sensible": (-300.0, 300.0), "latent": (-800.0, 300.0)}
    rank_models = (
        ("empirical", "centers", "Empirical copula", "copula_empirical_eng.png"),
        # RU: "Эмпирическая копула"
        ("gaussian", "centers", "Gaussian copula", "copula_gaussian_eng.png"),
        # RU: "Гауссовская копула"
        ("student_t", "centers", "Student t copula", "copula_student_t_eng.png"),
        # RU: "Копула Стьюдента"
        ("checkerboard", "checkerboard_centers", "Checkerboard copula", "copula_checkerboard_eng.png"),
        # RU: "Шахматная копула"
        ("bernstein", "bernstein_grid", "Bernstein copula", "copula_bernstein_eng.png"),
        # RU: "Копула Бернштейна"
    )
    joint_models = (
        ("joint_gaussian", "Gaussian copula stationary density", "gaussian_stationary_density_eng.png"),
        # RU: "Совместная стационарная плотность гауссовской копулы"
        ("joint_checkerboard", "Checkerboard copula stationary density", "checkerboard_stationary_density_eng.png"),
        # RU: "Совместная стационарная плотность шахматной копулы"
        ("joint_bernstein", "Bernstein copula stationary density", "bernstein_stationary_density_eng.png"),
        # RU: "Совместная стационарная плотность копулы Бернштейна"
    )

    def _cap(data_arrays, percentile):
        finite_caps = []
        for values in data_arrays:
            a = np.asarray(values, dtype=float)
            finite = a[np.isfinite(a) & (a >= 0)]
            if finite.size:
                finite_caps.append(float(np.percentile(finite, percentile)))
        cap = max(finite_caps, default=1.0)
        return cap if np.isfinite(cap) and cap > 0 else 1.0

    def _write_one(fig, name):
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        outputs.append(save_figure(fig, destination, dpi=300))
        print(f"[copula plot] Saved: {destination}", flush=True)

    def _rank_labels(ax):
        ax.set_xlabel("U = F1(X1)")
        # ax.set_xlabel("U = F1(X1), ранг первой переменной")
        ax.set_ylabel("V = F2(X2)")
        # ax.set_ylabel("V = F2(X2), ранг второй переменной")
        ax.set_xlim(0.01, 0.99)
        ax.set_ylim(0.01, 0.99)
        ax.set_aspect("equal", adjustable="box")

    def _bounds(coordinate, density, variable):
        coordinate = np.asarray(coordinate, dtype=float)
        if variable in display_ranges:
            lo, hi = display_ranges[variable]
        else:
            # Use the central 99% of the saved marginal for other variables.
            density = np.maximum(np.asarray(density, dtype=float), 0.0)
            weights = 0.5 * (density[:-1] + density[1:]) * np.diff(coordinate)
            cdf = np.concatenate(([0.0], np.cumsum(weights)))
            if cdf[-1] > 0:
                cdf /= cdf[-1]
                lo, hi = np.interp((0.005, 0.995), cdf, coordinate)
            else:
                lo, hi = coordinate[0], coordinate[-1]
        lo = max(float(lo), float(coordinate[0]))
        hi = min(float(hi), float(coordinate[-1]))
        visible = int(np.count_nonzero((coordinate >= lo) & (coordinate <= hi)))
        if lo >= hi or visible < 4:
            lo, hi = float(coordinate[0]), float(coordinate[-1])
        return lo, hi

    # Shared colour limits let the reader compare models on the same scale.
    # The 99th / 99.5th percentiles reduce distortion from singular corner cells.
    with np.load(source, allow_pickle=False) as data, plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "savefig.facecolor": "white",
    }):
        rank_arrays = {}
        for key, grid_key, title, filename in rank_models:
            xy = np.asarray(data[grid_key], dtype=float)
            pdf = np.asarray(data[key], dtype=float)
            if xy.ndim != 1 or len(xy) < 2 or np.any(np.diff(xy) <= 0):
                raise ValueError(f"{grid_key}: rank grid must be increasing")
            if pdf.shape != (len(xy), len(xy)):
                raise ValueError(f"{key}: copula density {pdf.shape} does not match rank grid {len(xy)}")
            rank_arrays[key] = (xy, pdf, title, filename)
        # Apply the common vmax only to visible plotting region: singularities
        # at nearly 0 or 1 should not wash out the meaningful central colour map.
        rank_vmax = _cap([
            pdf[np.ix_((xy >= 0.02) & (xy <= 0.98),
                        (xy >= 0.02) & (xy <= 0.98))]
            for xy, pdf, _, _ in rank_arrays.values()
        ], percentile=99.0)
        for key, (xy, pdf, title, filename) in rank_arrays.items():
            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
            try:
                image = ax.pcolormesh(xy, xy, np.ma.masked_invalid(pdf),
                                      shading="auto", cmap=cmap,
                                      vmin=0, vmax=rank_vmax)
                ax.set_title(title)
                # ax.set_title("Эмпирическая / гауссовская / Стьюдента / шахматная / Бернштейна копула")
                _rank_labels(ax)
                if key == "checkerboard":
                    ax.set_xlim(0.02, 0.98)
                    ax.set_ylim(0.02, 0.98)
                fig.colorbar(image, ax=ax, label="Copula density")
                # fig.colorbar(image, ax=ax, label="Плотность копулы")
                _write_one(fig, filename)
            finally:
                plt.close(fig)

        # Old-style normalized-level overlays: the models need not have
        # identical absolute peaks for their shapes to be comparable.
        def _plot_contours(models, title, filename):
            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
            try:
                for key, colour, linestyle, legend in models:
                    xy, pdf, _, _ = rank_arrays[key]
                    vmax = float(np.nanmax(pdf))
                    if np.isfinite(vmax) and vmax > 0:
                        ax.contour(xy, xy, pdf / vmax,
                                   levels=np.linspace(0.05, 0.90, 10),
                                   colors=colour, linestyles=linestyle,
                                   linewidths=1.3)
                    ax.plot([], [], color=colour, linestyle=linestyle,
                            linewidth=1.7, label=legend)
                _rank_labels(ax)
                ax.set_title(title)
                # ax.set_title("Сравнение контуров плотности копул")
                ax.legend(loc="upper left", fontsize=10)
                _write_one(fig, filename)
            finally:
                plt.close(fig)

        _plot_contours(
            (("empirical", "black", "-", "Empirical"),
             ("gaussian", "darkorange", "--", "Gaussian"),
             ("student_t", "forestgreen", ":", "Student t")),
            # RU legend: "Эмпирическая", "Гауссовская", "Стьюдента"
            "Empirical vs Gaussian vs Student t copula",
            "copula_models_contours_eng.png",
        )
        _plot_contours(
            (("checkerboard", "black", "-", "Checkerboard"),
             ("bernstein", "red", "--", "Bernstein")),
            # RU legend: "Шахматная", "Бернштейна"
            "Checkerboard vs Bernstein copula",
            "checkerboard_bernstein_contours_eng.png",
        )

        x1 = np.asarray(data["x1"], dtype=float)
        x2 = np.asarray(data["x2"], dtype=float)
        if x1.ndim != 1 or x2.ndim != 1:
            raise ValueError("Joint-density coordinates must be 1D")
        joint = {}
        for key, title, filename in joint_models:
            pdf = np.asarray(data[key], dtype=float)
            if pdf.shape != (len(x2), len(x1)):
                raise ValueError(f"{key}: expected {(len(x2), len(x1))}, got {pdf.shape}")
            joint[key] = (pdf, title, filename)
        ref = joint["joint_gaussian"][0]
        default_bounds_x = _bounds(x1, np.trapezoid(ref, x2, axis=0), names[0])
        default_bounds_y = _bounds(x2, np.trapezoid(ref, x1, axis=1), names[1])
        zoom_by_joint = {
            # Legacy article-style zoom used for the Bernstein sensible/latent panel.
            ("sensible", "latent", "joint_bernstein"): ((-50.0, 150.0), (-250.0, 50.0)),
            ("latent", "sensible", "joint_bernstein"): ((-250.0, 50.0), (-50.0, 150.0)),
        }
        visible_windows = {}
        for key in joint:
            bx, by = zoom_by_joint.get((names[0], names[1], key), (default_bounds_x, default_bounds_y))
            visible_windows[key] = (
                (x1 >= bx[0]) & (x1 <= bx[1]),
                (x2 >= by[0]) & (x2 <= by[1]),
            )
        vmax = _cap([
            joint[key][0][np.ix_(visible_windows[key][1], visible_windows[key][0])]
            for key in joint
        ], percentile=99.5)
        for key, (pdf, title, filename) in joint.items():
            bounds_x, bounds_y = zoom_by_joint.get((names[0], names[1], key), (default_bounds_x, default_bounds_y))
            fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
            try:
                image = ax.pcolormesh(x1, x2, np.ma.masked_invalid(pdf),
                                      shading="auto", cmap=cmap,
                                      vmin=0.0, vmax=vmax)
                ax.set_xlim(*bounds_x)
                ax.set_ylim(*bounds_y)
                ax.set_title(title)
                # ax.set_title("Стационарная совместная плотность копулы")
                ax.set_xlabel(NAMES_EN.get(names[0], names[0]))
                # ax.set_xlabel(NAMES_RU.get(names[0], names[0]))
                ax.set_ylabel(NAMES_EN.get(names[1], names[1]))
                # ax.set_ylabel(NAMES_RU.get(names[1], names[1]))
                fig.colorbar(image, ax=ax, label="Stationary joint density")
                # fig.colorbar(image, ax=ax, label="Совместная стационарная плотность")
                _write_one(fig, filename)
            finally:
                plt.close(fig)

        # Two additional figures reproduce the sample Bernstein marginals.
        # They require the classical 1D stationary PDFs already used to build
        # the saved copula models. If not present, simply omit the two figures.
        stationary = (root / "Functional" / "Stationary" /
                      f"{pair}_{period}.npz")
        if stationary.is_file():
            with np.load(stationary, allow_pickle=False) as marginal_source:
                old_grid = np.asarray(marginal_source["grid"], dtype=float)
                marginal_reference = [
                    np.interp(x, old_grid,
                              np.asarray(marginal_source[f"pdf_{name}"], dtype=float))
                    for x, name in ((x1, names[0]), (x2, names[1]))
                ]
            bernstein = joint["joint_bernstein"][0]
            derived = [np.trapezoid(bernstein, x2, axis=0),
                       np.trapezoid(bernstein, x1, axis=1)]
            for index, (x, reference, reconstructed) in enumerate(
                    zip((x1, x2), marginal_reference, derived), start=1):
                fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
                try:
                    ax.plot(x, reference, color="tab:blue", lw=2,
                            label="1D stationary")
                    # label="Одномерная стационарная плотность"
                    ax.plot(x, reconstructed, color="tab:orange", ls="--", lw=2,
                            label="Bernstein 2D marginal")
                    # label="Маргинал двумерной копулы Бернштейна"
                    ax.set_xlim(*(bounds_x if index == 1 else bounds_y))
                    ax.set_xlabel(NAMES_EN.get(names[index - 1], names[index - 1]))
                    # ax.set_xlabel(NAMES_RU.get(names[index - 1], names[index - 1]))
                    ax.set_ylabel("Probability density")
                    # ax.set_ylabel("Плотность вероятности")
                    ax.set_title(f"Bernstein copula: marginal of {names[index - 1]}")
                    # ax.set_title(f"Копула Бернштейна: маргинал {names[index - 1]}")
                    ax.legend()
                    _write_one(fig, f"bernstein_marginal_x{index}_eng.png")
                finally:
                    plt.close(fig)
        else:
            print(f"[copula plot] Stationary marginals not found: {stationary}"
                  " (skipping two marginal figures)", flush=True)
    print(f"[copula plot] Saved {len(outputs)} English individual figures from {source}", flush=True)
    return outputs
