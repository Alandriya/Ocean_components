
from __future__ import annotations

import datetime
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Eigenvalues.eigenvalues import get_bins


X_TICKS = [0, 60, 120, 180]
Y_TICKS = [160, 100, 40, 0]
X_LABELS = ["90° з. д.", "60° з. д.", "30° з. д.", "0°"]
Y_LABELS = ["Экватор", "30° с. ш.", "60° с. ш.", "80° с. ш."]

BASE_DATE = datetime.date(1979, 1, 1)


def _field_at_time(array, t, n_points):
    """Extract a flattened spatial field at a given time."""
    array = np.asarray(array)

    if array.ndim != 2 or array.shape[0] != n_points:
        raise ValueError(
            f"Expected ({n_points}, n_times), got {array.shape}"
        )

    if not 0 <= t < array.shape[1]:
        raise IndexError(f"Time index {t} is out of range")

    return array[:, t]


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
    """
    Align paired singular-vector signs without changing U S V.T.
    """
    U = np.asarray(U, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()

    for k in range(U.shape[1]):
        pivot = U[np.argmax(np.abs(U[:, k])), k]
        score = pivot

        if previous is not None:
            old_U, old_V = previous

            overlap = (
                np.dot(old_U[:, k], U[:, k])
                + np.dot(old_V[:, k], V[:, k])
            )

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


def _map_contribution(
    coefficient,
    left_vector,
    right_vector,
    bins1,
    bins2,
    valid,
    shape,
):
    """
    Map one spectral contribution onto the geographical grid.

    For equal variables:
        lambda_k * e_k[i] * e_k[j]

    For mixed variables:
        sigma_k * u_k[i] * v_k[j]
    """
    result = np.full(bins1.size, np.nan, dtype=float)

    result[valid] = (
        coefficient
        * left_vector[bins1[valid]]
        * right_vector[bins2[valid]]
    )

    return result.reshape(shape)


def _format_axis(ax, shape):
    """Apply geographical coordinate labels when appropriate."""
    if tuple(shape) == (161, 181):
        ax.set_xticks(X_TICKS)
        ax.set_yticks(Y_TICKS)
        ax.set_xticklabels(X_LABELS)
        ax.set_yticklabels(Y_LABELS)


def _draw_matrix(
    ax,
    matrix,
    title,
    shape,
    signed=True,
    fixed_limit=None,
):
    """Draw a map with an appropriate color scale."""
    finite = matrix[np.isfinite(matrix)]

    if finite.size == 0:
        limit = 1.0
    else:
        limit = float(np.max(np.abs(finite)))

    limit = max(limit, 1e-12)

    if fixed_limit is not None:
        if fixed_limit <= 0:
            raise ValueError("fixed_limit must be positive")
        limit = fixed_limit

    if signed:
        image = ax.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="none",
        )
    else:
        image = ax.imshow(
            matrix,
            cmap="viridis",
            vmin=0,
            vmax=limit,
            interpolation="none",
        )

    image.cmap.set_bad("lightgreen")
    ax.set_title(title, fontsize=11)
    _format_axis(ax, shape)

    return image


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
):
    """
    Visualize eigenmodes or paired singular modes.

    Parameters
    ----------
    offset:
        One-based absolute day index relative to 1979-01-01.
        For 2019-01-01, offset is 14611.

    array2:
        Second field with shape (n_points, n_times).

    array1:
        First field, required for mixed-variable pairs.

    save_mode_maps:
        Save sign-invariant modal contributions for subsequent
        seasonal averaging.

    Notes
    -----
    The first field indexes left singular vectors.
    The second field indexes right singular vectors.

    Mode signs are aligned for visualization only. Near-degenerate
    singular subspaces may require additional mode matching.
    """
    if n_lambdas < 1:
        raise ValueError("n_lambdas must be positive")

    if vector_limit <= 0:
        raise ValueError("vector_limit must be positive")

    if offset < 1:
        raise ValueError("offset must use one-based day numbering")

    root = Path(files_path_prefix)
    pair = f"{names[0]}-{names[1]}"
    folder = root / "Eigenvalues" / pair
    out_dir = root / "videos" / "Eigenvalues" / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    mixed = names[0] != names[1]
    n_points = int(np.prod(shape))

    ocean = np.asarray(mask, dtype=bool).reshape(-1)

    if ocean.size != n_points:
        raise ValueError("Mask size does not match spatial shape")

    if mixed:
        if array1 is None or array1_quantiles is None:
            raise ValueError(
                "Mixed pairs require array1 and array1_quantiles"
            )
    else:
        array1 = array2
        array1_quantiles = array2_quantiles

    q1 = np.asarray(array1_quantiles, dtype=float)
    q2 = np.asarray(array2_quantiles, dtype=float)

    if (
        q1.ndim != 1 or q2.ndim != 1
        or q1.size < 2 or q2.size < 2
        or not np.all(np.isfinite(q1))
        or not np.all(np.isfinite(q2))
        or np.any(np.diff(q1) < 0)
        or np.any(np.diff(q2) < 0)
    ):
        raise ValueError("Invalid quantile boundaries")

    previous = None
    processed = 0

    for t in range(t_start, t_end):
        step = t + offset
        date = BASE_DATE + datetime.timedelta(days=step - 1)

        if mixed:
            filenames = (
                f"singular_values_{step}.npy",
                f"left_singularvectors_{step}.npy",
                f"right_singularvectors_{step}.npy",
            )
        else:
            filenames = (
                f"eigenvalues_{step}.npy",
                f"eigenvectors_{step}.npy",
            )

        paths = [folder / name for name in filenames]
        missing = [p.name for p in paths if not p.exists()]

        if missing:
            warnings.warn(
                f"{pair}, {date}: missing {', '.join(missing)}"
            )
            continue

        values = np.asarray(np.load(paths[0]), dtype=float)

        if mixed:
            U = np.asarray(np.load(paths[1]), dtype=float)
            V = np.asarray(np.load(paths[2]), dtype=float)
        else:
            U = np.asarray(np.load(paths[1]), dtype=float)
            V = U

        if values.ndim != 1:
            raise ValueError("Spectral values must be one-dimensional")

        if U.ndim != 2 or V.ndim != 2:
            raise ValueError("Spectral vectors must be matrices")

        if (
            U.shape[0] != len(q1) - 1
            or V.shape[0] != len(q2) - 1
        ):
            raise ValueError(
                f"Quantile dimensions disagree with {pair} spectrum"
            )

        N = n_lambdas

        if (
            N > len(values)
            or N > U.shape[1]
            or N > V.shape[1]
        ):
            raise ValueError(
                f"{pair}: requested {N} unavailable modes"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError("Non-finite spectral values")

        if np.any(np.diff(values) > 1e-10):
            raise ValueError("Spectral values are not sorted")

        if mixed:
            if np.any(values < -1e-10):
                raise ValueError("Negative singular values found")

            U, V = _fix_svd_signs(U[:, :N], V[:, :N], previous)
            previous = (U.copy(), V.copy())
        else:
            tolerance = (
                1e-10 * max(1.0, float(np.max(np.abs(values))))
            )
            if np.any(values < -tolerance):
                raise ValueError(
                    "The same-variable matrix is not positive "
                    "semidefinite; check the eigenvalue calculation"
                )

            values = np.maximum(values, 0.0)
            U = _fix_signs(U[:, :N], previous)
            V = U
            previous = U.copy()

        field1 = _field_at_time(array1, t, n_points)
        field2 = _field_at_time(array2, t, n_points)

        bins1 = get_bins(field1, q1)
        bins2 = get_bins(field2, q2)

        valid = (
            ocean
            & np.isfinite(field1)
            & np.isfinite(field2)
            & (bins1 >= 0)
            & (bins2 >= 0)
        )

        if not np.any(valid):
            warnings.warn(f"No valid geographical points for {date}")
            continue

        vector_maps = []
        contribution_maps = []

        reconstructed = np.full(n_points, np.nan, dtype=float)
        reconstructed[valid] = 0.0

        for k in range(N):
            if mixed:
                left_map = _map_vector(
                    U[:, k], bins1, valid, shape
                )
                right_map = _map_vector(
                    V[:, k], bins2, valid, shape
                )

                vector_maps.append((left_map, right_map))
            else:
                eigen_map = _map_vector(
                    U[:, k], bins1, valid, shape
                )

                vector_maps.append((eigen_map,))

            contribution = _map_contribution(
                values[k],
                U[:, k],
                V[:, k],
                bins1,
                bins2,
                valid,
                shape,
            )

            contribution_maps.append(contribution)
            reconstructed[valid] += contribution.reshape(-1)[valid]

            if save_mode_maps:
                np.save(
                    folder / f"mode_contribution_{k + 1}_{step}.npy",
                    contribution.astype(np.float32),
                )

        reconstructed = reconstructed.reshape(shape)

        # Check the reconstruction against saved numerical results.
        reference_name = (
            f"Bn_map_{step}.npy"
            if mixed else
            f"Cn_map_{step}.npy"
        )
        reference_path = folder / reference_name

        if reference_path.exists() and N == 3:
            reference = np.load(reference_path)

            if not np.allclose(
                reconstructed,
                reference,
                rtol=1e-5,
                atol=1e-8,
                equal_nan=True,
            ):
                warnings.warn(
                    f"{date}: reconstructed map differs "
                    f"from {reference_name}"
                )

        # Plot the signed eigenvectors or singular vectors.
        rows = 2 if mixed else 1

        fig, axes = plt.subplots(
            rows,
            N,
            figsize=(5.0 * N, 4.8 * rows),
            squeeze=False,
            constrained_layout=True,
        )

        fig.suptitle(
            (
                f"Сингулярные моды {pair}"
                if mixed else
                f"Собственные векторы {pair}"
            )
            + f"\n{date:%d.%m.%Y}",
            fontsize=16,
        )

        for k in range(N):
            if mixed:
                descriptions = (
                    f"Левый вектор {k + 1}: {names[0]}",
                    f"Правый вектор {k + 1}: {names[1]}",
                )
                symbol = r"\sigma"
            else:
                descriptions = (f"Собственный вектор {k + 1}",)
                symbol = r"\lambda"

            for row, matrix in enumerate(vector_maps[k]):
                title = (
                    descriptions[row]
                    + "\n"
                    + rf"${symbol}_{{{k + 1}}}$"
                    + f" = {values[k]:.3e}"
                )

                image = _draw_matrix(
                    axes[row, k],
                    matrix,
                    title,
                    shape,
                    signed=True,
                    fixed_limit=vector_limit,
                )

                fig.colorbar(
                    image,
                    ax=axes[row, k],
                    shrink=0.75,
                )

        fig.savefig(
            out_dir / f"Lambdas_{step}.png",
            dpi=300,
        )
        plt.close(fig)

        # Plot the sign-invariant contribution of each mode.
        if plot_contributions:
            fig, axes = plt.subplots(
                1, N,
                figsize=(5 * N, 4.8),
                squeeze=False,
                constrained_layout=True,
            )

            fig.suptitle(
                f"Вклады спектральных мод {pair}\n"
                f"{date:%d.%m.%Y}",
                fontsize=16,
            )

            for k in range(N):
                image = _draw_matrix(
                    axes[0, k],
                    contribution_maps[k],
                    f"Вклад моды {k + 1}",
                    shape,
                    signed=mixed,
                )

                fig.colorbar(
                    image,
                    ax=axes[0, k],
                    shrink=0.75,
                )

            fig.savefig(
                out_dir / f"Mode_contributions_{step}.png",
                dpi=300,
            )
            plt.close(fig)

        # Plot the rank-N covariance or cross-moment matrix.
        if plot_reconstruction:
            fig, ax = plt.subplots(
                figsize=(8, 6),
                constrained_layout=True,
            )

            description = (
                rf"$B_{{{N}}}$ — смешанный момент"
                if mixed else
                rf"$C_{{{N}}}$ — второй момент"
            )

            image = _draw_matrix(
                ax,
                reconstructed,
                f"{description}\n{date:%d.%m.%Y}",
                shape,
                signed=mixed,
            )

            fig.colorbar(image, ax=ax, shrink=0.8)

            fig.savefig(
                out_dir / f"Rank{N}_{step}.png",
                dpi=300,
            )
            plt.close(fig)

        processed += 1

        if processed % 50 == 0:
            print(
                f"{pair}: {processed} time steps processed",
                flush=True,
            )

    print(f"{pair}: completed {processed} plots")
    return processed


def plot_mean_year(
    files_path_prefix: str,
    names: tuple = ("Sensible", "Latent"),
    start_year: int = 1979,
    end_year: int = 2024,
    mode: int = 1,
    shape: tuple = (161, 181),
):
    """
    Plot a seasonal mean of sign-invariant modal contributions.

    The input maps must have been generated by plot_eigenvalues
    with save_mode_maps=True.

    end_year is exclusive.

    This is not an arithmetic average of signed eigenvectors.
    """
    if mode < 1:
        raise ValueError("mode must be positive")

    root = Path(files_path_prefix)
    pair = f"{names[0]}-{names[1]}"
    folder = root / "Eigenvalues" / pair

    output_dir = root / "videos" / "Mean_year" / "Eigenvectors"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_dates = (
        (2, 15), (4, 15), (6, 15),
        (8, 15), (10, 15), (12, 15),
    )

    month_names = {
        2: "февраля",
        4: "апреля",
        6: "июня",
        8: "августа",
        10: "октября",
        12: "декабря",
    }

    averages = []
    sample_counts = []

    for month, day in selected_dates:
        samples = []

        for year in range(start_year, end_year):
            current_date = datetime.date(year, month, day)
            absolute_day = (
                current_date - BASE_DATE
            ).days + 1

            path = (
                folder
                / f"mode_contribution_{mode}_{absolute_day}.npy"
            )

            if not path.exists():
                continue

            data = np.load(path)

            if data.shape != shape:
                raise ValueError(
                    f"Unexpected shape in {path}: {data.shape}"
                )

            samples.append(data)

        if not samples:
            raise FileNotFoundError(
                f"No modal contribution maps for "
                f"{day} {month_names[month]} in {pair}"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            average = np.nanmean(
                np.stack(samples, axis=0),
                axis=0,
            )

        averages.append(average)
        sample_counts.append(len(samples))

    mixed = names[0] != names[1]

    values = np.stack(averages)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        raise ValueError("All seasonal means are non-finite")

    limit = max(float(np.max(np.abs(finite))), 1e-12)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 10),
        constrained_layout=True,
    )

    fig.suptitle(
        f"Среднегодовой вклад моды {mode}: {pair}\n"
        f"{start_year}–{end_year - 1}",
        fontsize=18,
    )

    for i, (month, day) in enumerate(selected_dates):
        ax = axes.flat[i]

        image = _draw_matrix(
            ax,
            averages[i],
            (
                f"{day} {month_names[month]}\n"
                f"Число лет: {sample_counts[i]}"
            ),
            shape,
            signed=mixed,
            fixed_limit=limit,
        )

        fig.colorbar(
            image,
            ax=ax,
            shrink=0.75,
        )

    filename = (
        f"modal_contribution_{pair}_"
        f"{start_year}-{end_year}_mode{mode}_mean_year"
    )

    fig.savefig(
        output_dir / f"{filename}.png",
        dpi=300,
    )
    plt.close(fig)

    np.save(
        output_dir / f"{filename}.npy",
        values,
    )

    return values
