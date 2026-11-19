"""High-level plotting workflows reading precomputed oceanographic arrays."""

from __future__ import annotations

import json
import subprocess
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
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


def _spectral_quantile_edges(series, n_bins, scratch_dir, *, chunk_elements=1_000_000):
    """Exact finite-value linear quantiles with disk-backed working storage.

    The calculation uses float64 values and np.quantile, as scale_to_bins did,
    but never creates the full scaled array, mask or bin-index array in RAM.
    Quantile order is independent of the traversal order. Follow the source
    layout to make the sequential read efficient for either time orientation.
    """
    import tempfile
    from tqdm.auto import tqdm

    n_bins = int(n_bins)
    probabilities = np.linspace(0.0, 1.0, n_bins + 1)
    with tempfile.TemporaryDirectory(prefix="quantiles-", dir=scratch_dir) as tmp:
        path = Path(tmp) / "finite_values.bin"
        count = 0
        # Slice along the slower axis to keep chunks as contiguous as possible.
        axis = int(abs(series.strides[1]) >= abs(series.strides[0]))
        width = series.shape[1 - axis]
        step = max(1, chunk_elements // max(1, width))
        starts = range(0, series.shape[axis], step)
        with path.open("wb") as stream:
            for begin in tqdm(starts, desc="Read quantile data", unit="chunk", leave=False):
                stop = begin + step
                section = series[:, begin:stop] if axis == 1 else series[begin:stop, :]
                chunk = np.asarray(section, dtype=np.float64)
                finite = chunk[np.isfinite(chunk)]
                finite.tofile(stream)
                count += finite.size
        if count == 0:
            raise ValueError("No finite values to calculate quantiles")
        # In-place partition of a writable, disposable 1D file mapping.
        # Never partition the original read-only field. OS may cache pages.
        data = np.memmap(path, dtype=np.float64, mode="r+", shape=(count,))
        try:
            edges = np.array(np.quantile(data, probabilities, overwrite_input=True), copy=True)
        finally:
            # Explicitly release the mapping before temporary-file deletion on Windows.
            data._mmap.close()
            del data
    return edges


def _cached_spectral_quantiles(root, name, start_year, end_year, n_bins, series):
    """Reuse exact edges while the source file, period and bin count agree."""
    import tempfile
    from Data_processing.io import field_candidates

    source = next((p for p in field_candidates(root, name, start_year, end_year)
                   if p.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"Missing spectral source for {name}, {start_year}-{end_year}")
    stat = source.stat()
    signature = json.dumps({"format": 1, "source": str(source.resolve()),
                            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                            "shape": list(series.shape), "dtype": str(series.dtype),
                            "n_bins": n_bins, "method": "linear-finite-float64",
                            "numpy": np.__version__}, sort_keys=True)
    folder = Path(root) / "Plots" / "Eigenvalues" / "_quantiles"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_{start_year}-{end_year}_{n_bins}.npz"
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as cache:
                edges = np.asarray(cache["edges"], dtype=float)
                if (str(cache["signature"].item()) == signature and
                        edges.shape == (n_bins + 1,) and np.all(np.isfinite(edges)) and
                        np.all(np.diff(edges) >= 0)):
                    print(f"[eigenvalues] Reusing quantile edges: {path}", flush=True)
                    return edges
        except (OSError, ValueError, KeyError, EOFError):
            pass
    print(f"[eigenvalues] Exact quantiles for {name}, {start_year}-{end_year}; "
          f"temporary disk storage up to {series.size * 8 / 1024**3:.2f} GiB", flush=True)
    edges = _spectral_quantile_edges(series, n_bins, folder)
    # A changed file must not receive a cache for the previous contents.
    current = source.stat()
    if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError(f"Spectral input changed while reading quantiles: {source}")
    # Atomic replacement prevents a partially written cache from being reused.
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, edges=edges, signature=signature)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return edges


def plot_spectral_modes(files_path_prefix, start_year, end_year, *, names=("Flux", "SST"),
                        first_day=0, last_day=5, n_bins=None, n_modes=3):
    """Render daily cached spectra with read-only fields and cached exact edges.

    start_year/end_year must describe the original spectral calculation period.
    first_day/last_day select the half-open interval to draw within that period.
    n_bins=None infers the bin count from the first complete daily spectrum.
    It does not infer the period or prove that the original quantiles agree.
    """
    from Plotting.plot_eigenvalues import plot_eigenvalues, infer_spectral_bins

    root = Path(files_path_prefix)
    if len(names) != 2 or not 0 <= first_day <= last_day:
        raise ValueError("Expected two variable names and 0 <= first_day <= last_day")
    if first_day == last_day:
        return 0
    offset = start_label(start_year)
    folder = root / "Eigenvalues" / f"{names[0]}-{names[1]}"
    saved_bins = infer_spectral_bins(folder, offset, first_day, last_day, names[0] != names[1])
    if n_bins is None:
        n_bins = saved_bins
    elif (not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, (bool, np.bool_))
          or n_bins != saved_bins):
        raise ValueError(f"{folder.name}: requested n_bins={n_bins}, saved n_bins={saved_bins}. "
                         "Use n_bins=None or the number used during calculation.")
    n_bins = int(n_bins)
    print(f"[eigenvalues] {folder.name}: n_bins={n_bins}, n_modes={n_modes}; "
          f"quantile period {start_year}-{end_year}", flush=True)

    # load_spectral_field delegates to load_field, which opens .npy in mmap_mode='r'.
    # For a self-pair, reuse the same mapping and the same cached quantiles.
    fields = {name: load_spectral_field(root, name, start_year, end_year)
              for name in dict.fromkeys(names)}
    first, second = fields[names[0]], fields[names[1]]
    if first.shape != second.shape or last_day > first.shape[1]:
        raise ValueError(f"Inconsistent spectral fields or day range: {first.shape}, {second.shape}, "
                         f"[{first_day}, {last_day})")
    quantiles = {name: _cached_spectral_quantiles(root, name, start_year, end_year, n_bins, data)
                 for name, data in fields.items()}
    return plot_eigenvalues(str(root), n_modes, read_ocean_mask(root), first_day, last_day,
                            offset, second, quantiles[names[1]], names=names,
                            shape=(161, 181), array1=first, array1_quantiles=quantiles[names[0]],
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


from Plotting.copula_plotting import plot_copula_results
