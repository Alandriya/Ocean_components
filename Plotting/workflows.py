"""High-level plotting workflows reading precomputed oceanographic arrays."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Data_processing.io import BASE_DATE, load_field, load_spectral_field, read_ocean_mask, start_label
from Plotting.paths import plot_path
from Plotting.video import frame_directory, frame_date


def save_figure(figure, filename, *args, **kwargs):
    """Save English-language figures without Plotting.paths automatic translation side effects.

    Plotting.paths.save_figure always calls translate_figure and would replace
    English text back with Russian. Russian captions remain commented below.
    If Russian graphics are needed again, enable that translation explicitly.
    """
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(destination), *args, **kwargs)
    return destination


NAMES_EN = {"sensible": "Sensible heat flux", "latent": "Latent heat flux",
            "Flux": "Total heat flux", "SST": "Sea surface temperature",
            "Pressure": "Atmospheric pressure", "A": "Coefficient A",
            "B": "Coefficient B", "C": "Coefficient C",
            "F": "Coefficient F", "FS": "Coefficient FS"}

NAMES_RU = {"sensible": "Явный тепловой поток", "latent": "Скрытый тепловой поток",
            "Flux": "Суммарный тепловой поток", "SST": "Температура поверхности моря",
            "Pressure": "Атмосферное давление", "A": "Коэффициент A",
            "B": "Коэффициент B", "C": "Коэффициент C", "F": "Коэффициент F"}


def _draw_map(ax, data, title, *, video_frames=False):
    """Draw one geographic grid with English labels; Russian alternatives are kept."""
    im = ax.imshow(np.ma.masked_invalid(data), cmap="coolwarm", origin="upper")
    if video_frames:
        ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return im


def plot_variables(files_path_prefix, start_year, end_year, *, day=0, video_frames=False):
    """Create paired heat-flux and three-variable maps for one observation day."""
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
            im = _draw_map(ax, array[day], NAMES_EN[name], video_frames=video_frames)
            fig.colorbar(im, ax=ax, shrink=0.8)
        timestamp = BASE_DATE + timedelta(days=start_label(start_year) - 1 + day)
        if video_frames:
            fig.suptitle(frame_date(timestamp))
        fig.tight_layout()
        outputs.append(save_figure(fig, plot_path(root, "Variables", folder, *frame_directory(video_frames),
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
        data = np.memmap(path, dtype=np.float64, mode="r+", shape=(count,))
        try:
            edges = np.array(np.quantile(data, probabilities, overwrite_input=True), copy=True)
        finally:
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
    current = source.stat()
    if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError(f"Spectral input changed while reading quantiles: {source}")
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, edges=edges, signature=signature)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return edges


def plot_spectral_modes(files_path_prefix, start_year, end_year, *, names=("Flux", "SST"),
                        first_day=0, last_day=5, n_bins=None, n_modes=3, video_frames=False):
    """Render daily cached spectra with read-only fields and cached exact edges."""
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
                            save_mode_maps=False, video_frames=video_frames)


def plot_mean_year(files_path_prefix, category, name, start_year, end_year,
                   *, day_indices=(45, 105, 165, 225, 285, 345), video_frames=False):
    """Display six representative calendar dates of an annual climatology."""
    root = Path(files_path_prefix)
    data = np.load(root / "Mean_year" / category /
                   f"{name}_{start_year}-{end_year}.npy", mmap_mode="r")
    outputs = []
    if data.ndim < 3:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(len(data)), data)
        ax.set_xlabel("Day of climatological year")
        ax.set_ylabel("Mean value")
        fig.suptitle(f"Annual climatology: {NAMES_EN.get(name, name)}")
        fig.tight_layout()
        outputs.append(save_figure(fig, plot_path(root, "Mean Year", category,
                                                   f"{name}_{start_year}-{end_year}.png")))
        plt.close(fig)
    else:
        channels = (data[:, channel] for channel in range(data.shape[1])) if data.ndim == 4 else (data,)
        for channel_number, channel in enumerate(channels):
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            for ax, index in zip(axes.flat, day_indices):
                im = _draw_map(ax, channel[index], f"{NAMES_EN.get(name, name)} — day {index + 1}", video_frames=video_frames)
                fig.colorbar(im, ax=ax, shrink=0.72)
            description = (f" — component {channel_number + 1}" if data.ndim == 4 else "")
            if video_frames:
                fig.suptitle(f"{start_year}–{end_year - 1}")
            fig.tight_layout()
            suffix = f"_ch{channel_number+1}" if data.ndim == 4 else ""
            outputs.append(save_figure(fig, plot_path(root, "Mean Year", category, *frame_directory(video_frames),
                                                       f"{name}_{start_year}-{end_year}{suffix}.png")))
            plt.close(fig)
    return outputs


def coefficient_frames(files_path_prefix, names, n_bins_2d, method, coefficient,
                        *, start=None, stop=None, step=1, video_frames=False):
    """Generate an ordered PNG sequence from previously calculated coefficients."""
    root = Path(files_path_prefix)
    folder = "Nonparametric_2d" if method == "np" else "Semiparametric_2d"
    parent = root / "Coefficients" / "continuous" / f"2d_bins_{n_bins_2d}" / f"{names[0]}_{names[1]}" / folder
    files = sorted(parent.glob(f"{coefficient}_*-*.npy"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one collected {coefficient} array under {parent}; found {files}")
    data = np.load(files[0], mmap_mode="r")
    first_label = int(files[0].stem.rsplit("_", 1)[-1].split("-")[0]) if video_frames else None
    stop = len(data) if stop is None else min(stop, len(data))
    start = 0 if start is None else start
    frames = []
    for i in range(start, stop, step):
        channels = data[i] if data.ndim == 4 else data[i][None, ...]
        fig, axes = plt.subplots(1, len(channels), figsize=(5.5 * len(channels), 5), squeeze=False)
        for j, (ax, field) in enumerate(zip(axes[0], channels)):
            img = _draw_map(ax, field, f"{NAMES_EN.get(coefficient, coefficient)} — component {j+1}", video_frames=video_frames)
            fig.colorbar(img, ax=ax, shrink=0.8)
        if video_frames:
            fig.suptitle(frame_date(BASE_DATE + timedelta(days=first_label + i - 1)))
        fig.tight_layout()
        saved = save_figure(fig, plot_path(root, "Coefficients", coefficient,
                                           f"{names[0]}-{names[1]}", method, *frame_directory(video_frames), f"frame_{i:05d}.png"))
        frames.append(saved)
        plt.close(fig)
    return frames


def _get_ffmpeg_executable() -> str:
    """Find an FFmpeg binary without requiring users to edit the Windows PATH.

    imageio-ffmpeg includes the binary in the pip-installed wheel on Windows.
    Use a system installation as a fallback.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        raise RuntimeError(
            "FFmpeg is not available. Install it into the same Python environment: "
            "C:\\Python\\python.exe -m pip install -U imageio-ffmpeg. "
            "Then run main_plotter.py again."
        ) from None


def coefficient_video(files_path_prefix, names, method, coefficient, *, fps=5,
                      start=None, stop=None, step=None, video_frames=False):
    """Encode cached PNG frames into H.264 MP4 with no external PATH requirement.

    The files may have non-contiguous frame numbers. If start/stop are provided,
    only PNGs with start <= frame_index < stop are included. step can optionally
    restrict frame_index to start + k * step.

    Using raw RGB frames over stdin avoids FFmpeg's OS-dependent glob behavior.
    The source PNGs are not modified and the original numerical data are untouched.
    Set video_frames=True to encode the separately stored labeled frames.
    """
    from PIL import Image

    root = Path(files_path_prefix)
    directory = root / "Plots" / "Coefficients" / coefficient / f"{names[0]}-{names[1]}" / method
    directory = directory.joinpath(*frame_directory(video_frames))
    frames = sorted(
        ((int(path.stem.removeprefix("frame_")), path) for path in directory.glob("frame_*.png")
         if path.stem.removeprefix("frame_").isdigit()),
        key=lambda item: item[0],
    )
    if start is not None:
        frames = [(number, path) for number, path in frames if number >= start]
    if stop is not None:
        frames = [(number, path) for number, path in frames if number < stop]
    if step is not None:
        if not isinstance(step, int) or step < 1:
            raise ValueError("step must be a positive integer")
        reference = 0 if start is None else start
        frames = [(number, path) for number, path in frames if (number - reference) % step == 0]
    if not frames:
        raise FileNotFoundError(f"No coefficient PNG frames matching the selected range in {directory}")
    if fps <= 0:
        raise ValueError("fps must be positive")

    target = plot_path(root, "Videos", "Coefficients", f"{names[0]}-{names[1]}",
                       f"{coefficient}_{method}{'_video_frames' if video_frames else ''}.mp4")
    ffmpeg = _get_ffmpeg_executable()
    with Image.open(frames[0][1]) as first:
        width, height = first.size

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-crf", "18",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
    ]
    print(f"[video] {len(frames)} PNG frames; {fps} fps; {width}x{height} px")
    print(f"[video] FFmpeg: {ffmpeg}")
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    try:
        for number, path in frames:
            with Image.open(path) as img:
                if img.size != (width, height):
                    raise ValueError(
                        f"Frame {path.name} has size {img.size}, expected {(width, height)}. "
                        "Clear old frames for different figure sizes and render the chosen interval again."
                    )
                rgb = img.convert("RGB")
                process.stdin.write(rgb.tobytes())
        process.stdin.close()
        errors = process.stderr.read().decode("utf-8", "replace")
        status = process.wait()
        if status != 0:
            raise RuntimeError(f"FFmpeg exited with code {status}: {errors.strip()}")
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        if target.exists():
            target.unlink()
        raise
    finally:
        if process.stderr:
            process.stderr.close()
    print(f"[video] Created: {target}")
    return target


def plot_stationary_distributions(files_path_prefix, names, start_year, end_year,
                                  *, histogram_bins=100):
    """Overlay saved stationary PDFs with normalized raw-data histograms."""
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
            ax.plot(grid, densities[f"pdf_{name}"], linewidth=2,
                    label="Stationary density")
            ax.set_xlabel("Variable value")
            ax.set_ylabel("Probability density")
            ax.set_title(NAMES_EN.get(name, name))
            ax.legend()
    fig.tight_layout()
    output = save_figure(fig, plot_path(root, "Distributions", "Stationary",
                                f"{names[0]}-{names[1]}_{start_year}-{end_year}.png"))
    plt.close(fig)
    return output


from Plotting.copula_plotting import plot_copula_results
