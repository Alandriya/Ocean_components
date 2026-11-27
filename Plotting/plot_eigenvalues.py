"""Render saved spectral modes using one set of Matplotlib artists per job."""
from __future__ import annotations

import datetime
import warnings
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from tqdm.auto import tqdm

from Eigenvalues.eigenvalues import get_bins


BASE_DATE = datetime.date(1979, 1, 1)
LON_BOUNDS = (-90.0, 0.0)
LAT_BOUNDS = (0.0, 80.0)
LON_TICKS = [-90, -60, -30, 0]
LAT_TICKS = [0, 30, 60, 80]
LON_LABELS = ["90W", "60W", "30W", "0"]
LAT_LABELS = ["0", "30N", "60N", "80N"]
BLUE_WHITE_RED = LinearSegmentedColormap.from_list(
    "blue_white_red", ["#1f3a93", "#ffffff", "#c1121f"], N=256
)


def _field_at_time(array, t, n_points):
    """Read one spatial field, not the entire memory-mapped series."""
    if array.ndim != 2 or array.shape[0] != n_points:
        raise ValueError(f"Expected ({n_points}, n_times), got {array.shape}")
    if not 0 <= t < array.shape[1]:
        raise IndexError(f"Time index {t} is out of range")
    return np.asarray(array[:, t], dtype=float)


def _fix_signs(vectors, previous=None):
    """Align eigenvector signs with the preceding time step."""
    vectors = np.asarray(vectors, dtype=float).copy()
    for k in range(vectors.shape[1]):
        column = vectors[:, k]
        pivot = column[np.argmax(np.abs(column))]
        score = pivot
        if previous is not None:
            overlap = np.dot(previous[:, k], column)
            if abs(overlap) > 0.05:
                score = overlap
        if score < 0:
            vectors[:, k] *= -1
    return vectors


def _fix_svd_signs(U, V, previous=None):
    """Align paired singular-vector signs without changing U S V.T."""
    U = np.asarray(U, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()
    for k in range(U.shape[1]):
        pivot = U[np.argmax(np.abs(U[:, k])), k]
        score = pivot
        if previous is not None:
            old_U, old_V = previous
            overlap = np.dot(old_U[:, k], U[:, k]) + np.dot(old_V[:, k], V[:, k])
            if abs(overlap) > 0.1:
                score = overlap
        if score < 0:
            U[:, k] *= -1
            V[:, k] *= -1
    return U, V


def _map_vector(vector, indices, valid, shape):
    """Project a vector indexed by quantile bins onto the grid."""
    result = np.full(indices.size, np.nan, dtype=float)
    result[valid] = vector[indices[valid]]
    return result.reshape(shape)


def _map_contribution(coefficient, left_vector, right_vector, bins1, bins2,
                      valid, shape):
    """Map one sign-invariant spectral contribution onto geography."""
    result = np.full(bins1.size, np.nan, dtype=float)
    result[valid] = coefficient * left_vector[bins1[valid]] * right_vector[bins2[valid]]
    return result.reshape(shape)


def _format_axis(ax, *, show_x_labels=True, show_y_labels=True):
    """Format axis for the plot eigenvalues workflow."""
    ax.set_xlim(*LON_BOUNDS)
    ax.set_ylim(*LAT_BOUNDS)
    ax.set_xticks(LON_TICKS)
    ax.set_yticks(LAT_TICKS)
    ax.set_xticklabels(LON_LABELS if show_x_labels else [])
    ax.set_yticklabels(LAT_LABELS if show_y_labels else [])
    ax.tick_params(axis="both", labelsize=9, pad=2)
    ax.grid(True, color="#bcbcbc", alpha=0.45, linewidth=0.7)


def _color_limits(matrix, signed=True, fixed_limit=None):
    """Compute color limits for plot eigenvalues from ``matrix``, ``signed``, ``fixed_limit``."""
    if fixed_limit is not None:
        if not np.isfinite(fixed_limit) or fixed_limit <= 0:
            raise ValueError("fixed_limit must be finite and positive")
        limit = float(fixed_limit)
    else:
        finite = matrix[np.isfinite(matrix)]
        limit = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1e-12)
    return (-limit if signed else 0.0), limit


def _map_norm(matrix, signed=True, fixed_limit=None):
    """Compute map norm for plot eigenvalues from ``matrix``, ``signed``, ``fixed_limit``."""
    vmin, vmax = _color_limits(matrix, signed=signed, fixed_limit=fixed_limit)
    if signed:
        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _map_cmap(signed=True, cmap=None):
    """Compute map cmap for plot eigenvalues from ``signed``, ``cmap``."""
    if cmap is not None:
        return cmap
    return BLUE_WHITE_RED if signed else "viridis"


def _draw_matrix(ax, matrix, title, signed=True, fixed_limit=None, *,
                 show_x_labels=True, show_y_labels=True, aspect="auto", cmap=None):
    """Create an image once; subsequent frames use _update_matrix."""
    image = ax.imshow(
        matrix,
        cmap=_map_cmap(signed, cmap=cmap),
        norm=_map_norm(matrix, signed=signed, fixed_limit=fixed_limit),
        interpolation="none",
        origin="upper",
        extent=(*LON_BOUNDS, *LAT_BOUNDS),
        aspect=aspect,
    )
    image.cmap.set_bad("lightgreen")
    ax.set_title(title, fontsize=10, pad=5)
    _format_axis(ax, show_x_labels=show_x_labels, show_y_labels=show_y_labels)
    return image


def _update_matrix(image, matrix, title, signed=True, fixed_limit=None):
    """Reuse AxesImage and its colorbar; update values, limits and caption."""
    image.set_data(matrix)
    image.set_norm(_map_norm(matrix, signed=signed, fixed_limit=fixed_limit))
    image.axes.title.set_text(title)


class _FrameFigure:
    """Own a fixed Figure, Axes, AxesImages and colorbars for a frame series."""

    def __init__(self, stack, rows, cols, shape, figsize, *, signed=True,
                 fixed_limit=None, xlabels_mode="all", ylabels_mode="all",
                 aspect="auto", cmap=None, video_frames=False):
        """Initialize Frame Figure with ``stack``, ``rows``, ``cols``, ``shape``, ``figsize``, ``signed``, ``fixed_limit`` and the remaining options."""
        self.video_frames = video_frames
        self.rows = rows
        self.cols = cols
        self.xlabels_mode = xlabels_mode
        self.ylabels_mode = ylabels_mode
        self.cmap = cmap
        self.fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False,
                                      constrained_layout=False)
        self.fig.subplots_adjust(left=0.04, right=0.968, bottom=0.065, top=0.84 if video_frames else 0.95,
                                 wspace=0.17, hspace=0.10)
        stack.callback(plt.close, self.fig)
        self.signed = signed
        self.fixed_limit = fixed_limit
        self.title = self.fig.suptitle("", fontsize=16, y=0.965) if video_frames else None
        self.images = np.empty((rows, cols), dtype=object)
        self.colorbars = np.empty((rows, cols), dtype=object)
        empty = np.zeros(shape, dtype=float)
        for row in range(rows):
            for col in range(cols):
                ax = axes[row, col]
                show_x = self._show_x_labels(row)
                show_y = self._show_y_labels(col)
                image = _draw_matrix(ax, empty, "", signed, fixed_limit,
                                     show_x_labels=show_x, show_y_labels=show_y,
                                     aspect=aspect, cmap=self.cmap)
                self.images[row, col] = image
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3.8%", pad=0.08)
                self.colorbars[row, col] = self.fig.colorbar(image, cax=cax)
                self.colorbars[row, col].ax.tick_params(labelsize=8, pad=2)

    def _show_x_labels(self, row):
        """Compute show x labels for Frame Figure from ``row``."""
        if self.xlabels_mode == "bottom":
            return row == self.rows - 1
        return True

    def _show_y_labels(self, col):
        """Compute show y labels for Frame Figure from ``col``."""
        if self.ylabels_mode == "left":
            return col == 0
        return True

    def update(self, row, col, matrix, title, *, fixed_limit=None):
        """Update the internal state using the supplied observations or values."""
        image = self.images[row, col]
        limit = self.fixed_limit if fixed_limit is None else fixed_limit
        _update_matrix(image, matrix, title if self.video_frames else "", self.signed, limit)
        self.colorbars[row, col].update_normal(image)

    def save(self, path, title):
        """Compute save for Frame Figure from ``path``, ``title``."""
        if self.title is not None:
            self.title.set_text(title)
        self.fig.savefig(path, dpi=300, **({} if self.video_frames else {"bbox_inches": "tight", "pad_inches": 0.08}))


def _spectrum_paths(folder, step, mixed):
    """Compute spectrum paths for plot eigenvalues from ``folder``, ``step``, ``mixed``."""
    stems = (("singular_values", "left_singularvectors", "right_singularvectors")
             if mixed else ("eigenvalues", "eigenvectors"))
    return tuple(folder / f"{stem}_{step}.npy" for stem in stems)


def infer_spectral_bins(folder, offset, first_day, last_day, mixed):
    """Infer the number of bins from the first complete saved daily spectrum."""
    folder = Path(folder)
    for t in range(first_day, last_day):
        paths = _spectrum_paths(folder, offset + t, mixed)
        if not all(path.is_file() for path in paths):
            continue
        U = np.load(paths[1], mmap_mode="r", allow_pickle=False)
        V = np.load(paths[2], mmap_mode="r", allow_pickle=False) if mixed else U
        if U.ndim != 2 or V.ndim != 2 or U.shape[0] != V.shape[0] or U.shape[0] < 1:
            raise ValueError(f"Invalid/inconsistent spectral-vector shapes: {paths[1]} {U.shape}, {V.shape}")
        return U.shape[0]
    raise FileNotFoundError(f"No complete daily spectra in {folder} for labels "
                            f"{offset + first_day}..{offset + last_day - 1}")


def _check_reconstruction(folder, step, mixed, N, values, U, V,
                          reconstructed, ocean):
    """Diagnose saved matrix vs geographic projection, using only the ocean."""
    stem = "Bn" if mixed else "Cn"
    matrix_path = folder / f"{stem}_{step}.npy"
    reference_path = folder / f"{stem}_map_{step}.npy"
    if not reference_path.is_file():
        return None
    if matrix_path.is_file():
        reference_matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        matrix = (U * values[:N]) @ V.T
        if reference_matrix.shape != matrix.shape:
            return ("saved matrix shape mismatch",
                    f"{matrix_path.name}: saved={reference_matrix.shape}, current={matrix.shape}")
        if not np.allclose(matrix, reference_matrix, rtol=1e-5, atol=1e-8, equal_nan=True):
            return ("saved matrix differs (rank or spectral files)",
                    f"{matrix_path.name}: check calculation n_components vs n_modes={N}, "
                    "and whether the spectrum and reconstruction are from the same run")
    elif N != 3:
        return ("reference rank not verifiable",
                f"{matrix_path.name} missing; cannot verify rank {N} against a legacy map")

    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    if reference.shape != reconstructed.shape:
        return ("saved map shape mismatch",
                f"{reference_path.name}: saved={reference.shape}, current={reconstructed.shape}")
    actual = reconstructed.reshape(-1)[ocean]
    expected = np.asarray(reference).reshape(-1)[ocean]
    if np.allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=True):
        return None
    actual_finite, expected_finite = np.isfinite(actual), np.isfinite(expected)
    common = actual_finite & expected_finite
    max_error = (f"{np.max(np.abs(actual[common] - expected[common])):.3e}"
                 if np.any(common) else "n/a")
    count = np.count_nonzero(~np.isclose(actual[common], expected[common], rtol=1e-5, atol=1e-8))
    mask_difference = np.count_nonzero(actual_finite != expected_finite)
    return ("geographic reconstruction differs",
            f"{reference_path.name}: different ocean values={count}/{np.count_nonzero(common)}, "
            f"different finite masks={mask_difference}, max abs error={max_error}; "
            "check quantile edges, source period and daily fields")


def plot_eigenvalues(
    files_path_prefix: str,
    n_lambdas: int,
    mask: np.ndarray,
    t_start: int,
    t_end: int,
    offset: int,
    array2: np.ndarray,
    array2_quantiles,
    names: tuple = ("Sensible", "Latent"),
    shape: tuple = (161, 181),
    *,
    array1: np.ndarray | None = None,
    array1_quantiles=None,
    save_mode_maps: bool = True,
    plot_contributions: bool = True,
    plot_reconstruction: bool = True,
    vector_limit: float = 1.0,
    video_frames: bool = False,
):
    """Visualize saved eigenmodes or paired singular modes one day at a time.

    Fields have shape (n_points, n_times) and may be read-only memory maps.
    offset is the one-based absolute day index from 1979-01-01. Quantile edges
    must correspond to the *calculation period*, not just the plotted days.
    Vector signs are aligned for visualization only; paired signs leave the
    modal contribution unchanged. Near-degenerate subspaces may require
    additional mode matching. Each enabled figure is created only once.
    """
    if n_lambdas < 1:
        raise ValueError("n_lambdas must be positive")
    if not np.isfinite(vector_limit) or vector_limit <= 0:
        raise ValueError("vector_limit must be finite and positive")
    if offset < 1:
        raise ValueError("offset must use one-based day numbering")
    if not 0 <= t_start <= t_end:
        raise ValueError("Expected 0 <= t_start <= t_end")

    root = Path(files_path_prefix)
    pair = f"{names[0]}-{names[1]}"
    folder = root / "Eigenvalues" / pair
    out_dir = root / "Plots" / "Eigenvalues" / pair
    if video_frames:
        out_dir = out_dir / "video_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    mixed = names[0] != names[1]
    n_points = int(np.prod(shape))
    ocean = np.asarray(mask, dtype=bool).reshape(-1)
    if ocean.size != n_points:
        raise ValueError("Mask size does not match spatial shape")
    if mixed:
        if array1 is None or array1_quantiles is None:
            raise ValueError("Mixed pairs require array1 and array1_quantiles")
    else:
        array1 = array2
        array1_quantiles = array2_quantiles
    array1 = np.asanyarray(array1)
    array2 = np.asanyarray(array2)
    if (array1.ndim != 2 or array2.ndim != 2 or array1.shape != array2.shape
            or array1.shape[0] != n_points or t_end > array1.shape[1]):
        raise ValueError(f"Incompatible spectral fields or day range: {array1.shape}, "
                         f"{array2.shape}, [{t_start}, {t_end})")

    q1 = np.asarray(array1_quantiles, dtype=float)
    q2 = np.asarray(array2_quantiles, dtype=float)
    for q in (q1, q2):
        if (q.ndim != 1 or q.size < 2 or not np.all(np.isfinite(q))
                or np.any(np.diff(q) < 0)):
            raise ValueError("Invalid quantile boundaries")

    previous = None
    processed = 0
    diagnostics = Counter()
    examples = {}
    N = n_lambdas
    rows = 2 if mixed else 1
    vectors = contributions = reconstruction = None

    with ExitStack() as stack, tqdm(range(t_start, t_end), desc=f"Eigenvalues {pair}",
                                    unit="day", dynamic_ncols=True) as progress:
        for t in progress:
            step = t + offset
            date = BASE_DATE + datetime.timedelta(days=step - 1)
            paths = _spectrum_paths(folder, step, mixed)
            missing = [p.name for p in paths if not p.exists()]
            if missing:
                diagnostics["missing spectra"] += 1
                examples.setdefault("missing spectra", f"{date}: {', '.join(missing)}")
                continue

            values = np.asarray(np.load(paths[0], mmap_mode="r", allow_pickle=False), dtype=float)
            U_raw = np.load(paths[1], mmap_mode="r", allow_pickle=False)
            V_raw = np.load(paths[2], mmap_mode="r", allow_pickle=False) if mixed else U_raw
            if values.ndim != 1:
                raise ValueError(f"{paths[0]}: spectral values must be one-dimensional")
            if U_raw.ndim != 2 or V_raw.ndim != 2:
                raise ValueError(f"{pair}, {date}: spectral vectors must be matrices")
            if U_raw.shape[0] != len(q1) - 1 or V_raw.shape[0] != len(q2) - 1:
                raise ValueError(f"Quantile dimensions disagree with {pair} spectrum on {date}: "
                                 f"saved bins=({U_raw.shape[0]}, {V_raw.shape[0]}), "
                                 f"plot bins=({len(q1)-1}, {len(q2)-1}); "
                                 "check n_bins and calculation period")
            if N > len(values) or N > U_raw.shape[1] or N > V_raw.shape[1]:
                raise ValueError(f"{pair}, {date}: requested {N} unavailable modes")
            if not np.all(np.isfinite(values)) or np.any(np.diff(values) > 1e-10):
                raise ValueError(f"{paths[0]}: spectral values must be finite and sorted")
            if mixed:
                if np.any(values < -1e-10):
                    raise ValueError("Negative singular values found")
                U, V = _fix_svd_signs(U_raw[:, :N], V_raw[:, :N], previous)
                previous = (U.copy(), V.copy())
            else:
                tolerance = 1e-10 * max(1.0, float(np.max(np.abs(values))))
                if np.any(values < -tolerance):
                    raise ValueError("The same-variable matrix is not positive semidefinite; "
                                     "check the eigenvalue calculation")
                values = np.maximum(values, 0.0)
                U = _fix_signs(U_raw[:, :N], previous)
                V = U
                previous = U.copy()
            del U_raw, V_raw

            field1 = _field_at_time(array1, t, n_points)
            field2 = _field_at_time(array2, t, n_points) if mixed else field1
            bins1 = get_bins(field1, q1)
            bins2 = get_bins(field2, q2) if mixed else bins1
            valid = ocean & np.isfinite(field1) & np.isfinite(field2) & (bins1 >= 0) & (bins2 >= 0)
            if not np.any(valid):
                diagnostics["no valid geographical points"] += 1
                examples.setdefault("no valid geographical points", str(date))
                continue

            reconstructed = np.full(n_points, np.nan, dtype=float)
            reconstructed[valid] = 0.0
            if vectors is None:
                vectors = _FrameFigure(stack, rows, N, shape, (5.6 * N, 5.0 * rows),
                                       signed=True, fixed_limit=vector_limit,
                                       xlabels_mode="bottom" if rows > 1 else "all",
                                       ylabels_mode="left", video_frames=video_frames)
                if plot_contributions:
                    contributions = _FrameFigure(stack, 1, N, shape, (5.6 * N, 5.0),
                                                 signed=mixed, ylabels_mode="left",
                                                 cmap="plasma", video_frames=video_frames)
                if plot_reconstruction:
                    reconstruction = _FrameFigure(stack, 1, N, shape, (5.6 * N, 5.0),
                                                  signed=mixed, ylabels_mode="left",
                                                  aspect="equal", video_frames=video_frames)

            rank_maps = []
            for k in range(N):
                if mixed:
                    descriptions = (f"Левый вектор {k + 1}: {names[0]}",
                                    f"Правый вектор {k + 1}: {names[1]}")
                    symbol = r"\sigma"
                else:
                    descriptions = (f"Собственный вектор {k + 1}",)
                    symbol = r"\lambda"
                left_map = _map_vector(U[:, k], bins1, valid, shape)
                value_title = rf"${symbol}_{{{k + 1}}}$" + f" = {values[k]:.3e}"
                vectors.update(0, k, left_map, descriptions[0] + "\n" + value_title)
                if mixed:
                    right_map = _map_vector(V[:, k], bins2, valid, shape)
                    vectors.update(1, k, right_map, descriptions[1] + "\n" + value_title)
                contribution = _map_contribution(values[k], U[:, k], V[:, k], bins1, bins2, valid, shape)
                reconstructed[valid] += contribution.reshape(-1)[valid]
                if reconstruction is not None:
                    # Store a snapshot of each partial sum, not the single mode.
                    rank_maps.append(reconstructed.reshape(shape).copy())
                if contributions is not None:
                    contributions.update(0, k, contribution, f"Вклад моды {k + 1}")
                if save_mode_maps:
                    np.save(folder / f"mode_contribution_{k + 1}_{step}.npy", contribution.astype(np.float32))
            reconstructed = reconstructed.reshape(shape)

            diagnostic = _check_reconstruction(folder, step, mixed, N, values, U, V,
                                               reconstructed, ocean)
            if diagnostic is not None:
                category, detail = diagnostic
                diagnostics[category] += 1
                examples.setdefault(category, f"{date}: {detail}")

            date_label = f"{date:%d.%m.%Y}"
            # Use singular modes for mixed pairs and eigenvectors for self-pairs.
            # vectors.save(out_dir / f"Lambdas_{step}.png", vector_title + "\n" + date_label)
            vectors.save(out_dir / f"Lambdas_{step}.png", date_label)
            if contributions is not None:
                # contributions.save(out_dir / f"Mode_contributions_{step}.png",
                # Show the spectral-mode contributions for the pair and date.
                contributions.save(out_dir / f"Mode_contributions_{step}.png", date_label)
            if reconstruction is not None:
                # Compare cumulative rank approximations using one daily scale.
                # Arrange from highest rank down to rank one (B_N, ..., B_1).
                common_limit = max(_color_limits(matrix, signed=mixed)[1]
                                   for matrix in rank_maps)
                symbol = "B" if mixed else "C"
                for col, rank in enumerate(range(N, 0, -1)):
                    reconstruction.update(0, col, rank_maps[rank - 1],
                                          rf"${symbol}_{{{rank}}}$",
                                          fixed_limit=common_limit)
                reconstruction.save(out_dir / f"Rank1-{N}_{step}.png" if N > 1
                                    else out_dir / f"Rank1_{step}.png", date_label)
            processed += 1
            if diagnostic is not None:
                progress.set_postfix(diagnostics=sum(diagnostics.values()), refresh=False)

    if diagnostics:
        summary = "\n".join(f"  {category}: {count} day(s); example: {examples[category]}"
                            for category, count in diagnostics.items())
        warnings.warn(f"{pair}: spectral plotting diagnostics (one summary for the run):\n{summary}",
                      RuntimeWarning, stacklevel=2)
    print(f"{pair}: completed {processed} plots", flush=True)
    return processed


def plot_mean_year(
    files_path_prefix: str,
    names: tuple = ("Sensible", "Latent"),
    start_year: int = 1979,
    end_year: int = 2024,
    mode: int = 1,
    shape: tuple = (161, 181),
    *, video_frames=False,
):
    """Plot a seasonal mean of sign-invariant modal contributions.

    Input maps are generated by plot_eigenvalues with save_mode_maps=True.
    end_year is exclusive. This is not an average of signed eigenvectors.
    """
    if mode < 1:
        raise ValueError("mode must be positive")
    root = Path(files_path_prefix)
    pair = f"{names[0]}-{names[1]}"
    folder = root / "Eigenvalues" / pair
    output_dir = root / "Plots" / "Mean_year" / "Eigenvectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dates = ((2, 15), (4, 15), (6, 15), (8, 15), (10, 15), (12, 15))
    month_names = {2: "февраля", 4: "апреля", 6: "июня", 8: "августа",
                   10: "октября", 12: "декабря"}
    averages = []
    sample_counts = []
    for month, day in selected_dates:
        samples = []
        for year in range(start_year, end_year):
            current_date = datetime.date(year, month, day)
            absolute_day = (current_date - BASE_DATE).days + 1
            path = folder / f"mode_contribution_{mode}_{absolute_day}.npy"
            if not path.exists():
                continue
            data = np.load(path)
            if data.shape != shape:
                raise ValueError(f"Unexpected shape in {path}: {data.shape}")
            samples.append(data)
        if not samples:
            raise FileNotFoundError(f"No modal contribution maps for {day} {month_names[month]} in {pair}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            average = np.nanmean(np.stack(samples, axis=0), axis=0)
        averages.append(average)
        sample_counts.append(len(samples))
    mixed = names[0] != names[1]
    values = np.stack(averages)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("All seasonal means are non-finite")
    limit = max(float(np.max(np.abs(finite))), 1e-12)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    if video_frames:
        fig.suptitle(f"{start_year}–{end_year - 1}", fontsize=18)
    for i, (month, day) in enumerate(selected_dates):
        ax = axes.flat[i]
        image = _draw_matrix(ax, averages[i], f"Вклад моды {mode}: {pair} — {day} {month_names[month]}" if video_frames else "",
                             signed=mixed, fixed_limit=limit, cmap="plasma")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.08)
        fig.colorbar(image, cax=cax)
    filename = f"modal_contribution_{pair}_{start_year}-{end_year}_mode{mode}_mean_year"
    figure_dir = output_dir / "video_frames" if video_frames else output_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{filename}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    np.save(output_dir / f"{filename}.npy", values)
    return values
