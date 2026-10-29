"""Quantile-conditioned second increment moments and spectral reconstructions.

Algorithm follows translation_corrected_eigenvector_article_ru.tex, section 3.3.
IMPORTANT: B_ij is a conditional second moment, not a centered covariance.
For identical fields and disjoint bins B is diagonal by construction.
"""
from __future__ import annotations

import datetime
from itertools import combinations_with_replacement
from pathlib import Path
import numpy as np

from Data_processing.data_processing import scale_to_bins, quantile_bin_indices


def get_eig(b_matrix: np.ndarray, names: tuple):
    """Symmetric eigenpairs for X-X; singular values and LEFT vectors for X-Y.

    For mixed fields the returned spectrum consists of sigma, NOT sigma**2.
    To obtain right vectors use ``np.linalg.svd`` explicitly.
    """
    b = np.asarray(b_matrix, dtype=float)
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError("b_matrix must be square")
    if not np.all(np.isfinite(b)):
        raise ValueError("b_matrix must be finite")
    if names[0] != names[1]:
        u, sigma, _ = np.linalg.svd(b, full_matrices=False)
        return sigma, u
    b = 0.5 * (b + b.T)
    values, vectors = np.linalg.eigh(b)
    order = np.argsort(values)[::-1]
    # Do not silently clip physically significant negative eigenvalues.
    return values[order], vectors[:, order]


def get_bins(values, quantiles):
    """Quantile bins (-1 for invalid values)."""
    return quantile_bin_indices(values, quantiles)


def matrix_to_map(matrix, field1, field2, quantiles1, quantiles2, spatial_shape):
    """Map B_N[i(p), j(p)] to geography, ignoring invalid observations."""
    a = np.asarray(field1).reshape(-1)
    b = np.asarray(field2).reshape(-1)
    if a.size != b.size or a.size != int(np.prod(spatial_shape)):
        raise ValueError("Fields must match spatial_shape")
    i, j = get_bins(a, quantiles1), get_bins(b, quantiles2)
    good = (i >= 0) & (j >= 0)
    result = np.full(a.size, np.nan, dtype=float)
    result[good] = matrix[i[good], j[good]]
    return result.reshape(spatial_shape)


def bin_values_to_map(bin_values, field, quantiles, spatial_shape):
    """Map a value associated with each quantile interval to geography."""
    values = np.asarray(field).reshape(-1)
    if values.size != int(np.prod(spatial_shape)):
        raise ValueError("Field must match spatial_shape")
    bins = get_bins(values, quantiles)
    good = bins >= 0
    result = np.full(values.size, np.nan)
    result[good] = np.asarray(bin_values)[bins[good]]
    return result.reshape(spatial_shape)


def _moment_matrix(x0, x1, y0, y1, edges_x, edges_y, dt):
    """Vectorized equivalent of 1/dt * mean(delta_x*delta_y) per joint cell.

    Cells with zero finite pairs are assigned 0 to preserve legacy output
    semantics. They are *unobserved*, not measured zero diffusion.
    Returns B and observation counts, enabling diagnostics.
    """
    nx, ny = len(edges_x) - 1, len(edges_y) - 1
    i, j = get_bins(x0, edges_x), get_bins(y0, edges_y)
    dx = x1 - x0
    dy = y1 - y0
    valid = (i >= 0) & (j >= 0) & np.isfinite(dx) & np.isfinite(dy)
    cells = i[valid] * ny + j[valid]
    counts = np.bincount(cells, minlength=nx * ny)
    weighted = np.bincount(cells, weights=dx[valid] * dy[valid], minlength=nx * ny)
    b = np.divide(weighted, counts * dt, out=np.zeros(nx * ny, dtype=float), where=counts != 0)
    return b.reshape(nx, ny), counts.reshape(nx, ny)


def count_eigenvalues_pair(
    files_path_prefix: str,
    array1: np.ndarray,
    array2: np.ndarray,
    array1_quantiles,
    array2_quantiles,
    n_bins: int,
    offset: int,
    names: tuple,
    spatial_shape: tuple,
    dt: float = 1.0,
    n_components: int = 3,
    overwrite: bool = False,
    save_counts: bool = False,
):
    """Save spectra/reconstructions for consecutive days of two fields.

    Fields: (n_grid_cells,n_days). ``offset`` is the ONE-based index of
    day t=0 relative to 1979-01-01 (1979-01-01 = 1, 2019-01-01 = 14611).
    Files have suffix ``offset+t`` and describe increments from day t to t+1.
    The final day of a block has no next observation and produces no file.

    Identical variables: eigh(B_sym), C_N = E_N Lambda_N E_N.T.
    Distinct variables: svd(B), B_N = U_N Sigma_N V_N.T.
    Quantiles are computed once per input period (not once per day).
    """
    x = np.asarray(array1)
    y = np.asarray(array2)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError(f"Fields must have identical (space,time) shapes: {x.shape}, {y.shape}")
    if x.shape[0] != int(np.prod(spatial_shape)) or x.shape[1] < 2:
        raise ValueError("Unexpected spatial shape or fewer than two days")
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    if not (1 <= n_components <= n_bins):
        raise ValueError("Require 1 <= n_components <= n_bins")
    if len(array1_quantiles) != n_bins + 1 or len(array2_quantiles) != n_bins + 1:
        raise ValueError("Quantile boundary lengths must equal n_bins+1")
    if len(names) != 2:
        raise ValueError("names must contain exactly two field names")
    same = names[0] == names[1]
    destination = Path(files_path_prefix) / 'Eigenvalues' / f'{names[0]}-{names[1]}'
    destination.mkdir(parents=True, exist_ok=True)

    for t in range(x.shape[1] - 1):
        idx = t + offset
        required = ('eigenvalues', 'eigenvectors', 'Cn', 'Cn_map', 'B2_map', 'B_map') if same else (
            'singular_values', 'left_singularvectors', 'right_singularvectors', 'Bn', 'Bn_map')
        if not overwrite and all((destination / f'{stem}_{idx}.npy').exists() for stem in required):
            continue
        if idx % 100 == 0:
            print(f'{names[0]}-{names[1]}: step {idx}', flush=True)
        b, counts = _moment_matrix(x[:, t], x[:, t + 1], y[:, t], y[:, t + 1],
                                   array1_quantiles, array2_quantiles, dt)
        if save_counts:
            np.save(destination / f'counts_{idx}.npy', counts)
        if same:
            b = 0.5 * (b + b.T)
            values, vectors = get_eig(b, names)
            np.save(destination / f'eigenvalues_{idx}.npy', values)
            np.save(destination / f'eigenvectors_{idx}.npy', vectors)
            c_n = (vectors[:, :n_components] * values[:n_components]) @ vectors[:, :n_components].T
            np.save(destination / f'Cn_{idx}.npy', c_n)
            np.save(destination / f'Cn_map_{idx}.npy', matrix_to_map(
                c_n, x[:, t], y[:, t], array1_quantiles, array2_quantiles, spatial_shape))
            b2 = bin_values_to_map(np.diag(c_n), x[:, t], array1_quantiles, spatial_shape)
            np.save(destination / f'B2_map_{idx}.npy', b2)
            np.save(destination / f'B_map_{idx}.npy', np.sqrt(np.clip(b2, 0.0, None)))
        else:
            u, sigma, vt = np.linalg.svd(b, full_matrices=False)
            np.save(destination / f'singular_values_{idx}.npy', sigma)
            np.save(destination / f'left_singularvectors_{idx}.npy', u)
            np.save(destination / f'right_singularvectors_{idx}.npy', vt.T)
            b_n = (u[:, :n_components] * sigma[:n_components]) @ vt[:n_components, :]
            np.save(destination / f'Bn_{idx}.npy', b_n)
            np.save(destination / f'Bn_map_{idx}.npy', matrix_to_map(
                b_n, x[:, t], y[:, t], array1_quantiles, array2_quantiles, spatial_shape))


def count_eigenvalues_all(files_path_prefix, fields, spatial_shape, offset,
                          n_bins=100, n_components=3, dt=1.0, overwrite=False,
                          save_counts=False):
    """All unordered pairs with repetition, e.g. 4 fields -> 10 pairs."""
    fields = {name: np.asarray(data) for name, data in fields.items()}
    if not fields:
        raise ValueError('No fields provided')
    if len({data.shape for data in fields.values()}) != 1:
        raise ValueError('All field shapes must match')
    quantiles = {name: scale_to_bins(data, n_bins)[1] for name, data in fields.items()}
    for left, right in combinations_with_replacement(fields, 2):
        count_eigenvalues_pair(files_path_prefix, fields[left], fields[right],
                               quantiles[left], quantiles[right], n_bins, offset,
                               (left, right), spatial_shape, dt, n_components,
                               overwrite=overwrite, save_counts=save_counts)


def count_eigenvalues_triplets(files_path_prefix: str, flux_array: np.ndarray,
                               SST_array: np.ndarray, press_array: np.ndarray,
                               spatial_shape: tuple, offset: int = 0,
                               n_bins: int = 100, n_components: int = 3,
                               overwrite: bool = False):
    """Legacy wrapper for all six distinct/self pairs of Flux, SST, Pressure."""
    count_eigenvalues_all(files_path_prefix,
                          {'Flux': flux_array, 'SST': SST_array, 'Pressure': press_array},
                          spatial_shape, offset, n_bins, n_components, overwrite=overwrite)


def count_mean_year(files_path_prefix: str,
                    start_year: int = 2009,
                    end_year: int = 2019,
                    names: tuple = ('Flux', 'Flux'),
                    mask: np.ndarray = None,
                    ):
    """
    Counts mean year for the first (already sorted by absolute values of eigenvalues) eigenvector and the
    corresponding eigenvalue
    :param files_path_prefix: path to the working directory
    :param start_year: start year of the range
    :param end_year: end year of the range (not included)
    :param names: tuple with names of the data arrays, e.g. ('Flux', 'SST')
    :param mask: boolean 1D mask with length 161*181. If true, it's ocean point, if false - land. Only ocean points are
        of interest
    :return:
    """
    height, width = 161, 181
    mean_year = np.zeros((363, height * width))
    mean_year_values = np.zeros(363)
    print(f'Pair {names[0]}-{names[1]}')

    for year in range(start_year, end_year):
        time_start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=1979, month=1, day=1)).days

        for day in range(363):
            matrix = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigen0_{day + time_start + 1}.npy')
            mean_year[day] += matrix
            mean_year[day][np.logical_not(mask)] = None

            eigenvalues = np.load(
                files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvalues_{day + time_start + 1}.npy')
            mean_year_values[day] += eigenvalues[0]

    mean_year /= (end_year - start_year)
    mean_year_values /= (end_year - start_year)
    np.save(files_path_prefix + f'Mean_year/eigenvector_{names[0]}-{names[1]}_{start_year}-{end_year}.npy', mean_year)
    np.save(files_path_prefix + f'Mean_year/eigenvalues_{names[0]}-{names[1]}_{start_year}-{end_year}.npy',
            mean_year_values)
    return


def get_trends(files_path_prefix: str,
               t_start: int,
               t_end: int,
               names: tuple = ('Flux', 'Flux')):
    """
    Counts max, min amd mean for the first eigenvector for pair names[0]-names[1] for time in range (t_start, t_end)
    :param files_path_prefix: path to the working directory
    :param t_start: absolute time for start day
    :param t_end: absolute time for end day (not included)
    :param names: tuple with names of the data arrays, e.g. ('Flux', 'SST')
    :return:
    """
    max_eigenvector = np.zeros(t_end - t_start)
    min_eigenvector = np.zeros(t_end - t_start)
    mean_eigenvector = np.zeros(t_end - t_start)
    for t in range(t_start, t_end):
        if not os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigen0_{t}.npy'):
            print(f'Missing eigen0_{t}')
            continue
        matrix = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigen0_{t}.npy')
        max_eigenvector[t - t_start] = np.nanmax(matrix)
        min_eigenvector[t - t_start] = np.nanmin(matrix)
        mean_eigenvector[t - t_start] = np.nanmean(matrix)

    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}_trends_max.npy', max_eigenvector)
    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}_trends_min.npy', min_eigenvector)
    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}_trends_mean.npy', mean_eigenvector)
    return



def reconstruct_b2_map(
    eigenvalues,
    eigenvectors,
    field_at_t,
    quantiles,
    spatial_shape,
    n_components=3,
    mask=None,
):
    """
    Reconstruct b^2 from the first N eigenpairs and map the n-bin values back to the geographical grid.
    :param eigenvalues: np.array with shape (n_bins)
    :param eigenvectors: np.array with shape (n_bins, n_bins). Eigenvectors are columns.
    :param field_at_t: Flattened geographical field at time t. Shape (n_spatial_points,).
    :param quantiles: Bin boundaries used when calculating the n_bins x n_bins matrix. Shape (n_bins + 1,).
    :param spatial_shape: np.array with 2d map size, for example (161, 181).
    :param n_components: Amount of eigenpairs retained, for example 3.
    :param mask: Boolean geographical mask with size equal to 2d map size, for example (161, 181).

    Returns
    -------
    b2_map: Reconstructed b^2 geographical map.
    b_map: Reconstructed diffusion-amplitude map sqrt(b^2).
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    field_at_t = np.asarray(field_at_t, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    # Sort by decreasing eigenvalue.
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]

    N = min(n_components, eigenvalues.size)

    lambda_n = eigenvalues[:N]
    vectors_n = eigenvectors[:, :N]

    # Full rank-N reconstruction:
    # C_N = sum_i lambda_i * e_i * e_i.T
    # b2_matrix_n = (vectors_n * lambda_n[np.newaxis, :]) @ vectors_n.T

    # Diagonal of C_N:
    # b2_bins[j] = sum_i lambda_i * e[j, i]^2
    b2_bins = np.sum(lambda_n[np.newaxis, :] * vectors_n**2, axis=1,)

    # Equivalent check:
    # np.allclose(b2_bins, np.diag(b2_matrix_n))
    # should return True.

    n_bins = len(quantiles) - 1
    bin_indices = (np.searchsorted(quantiles, field_at_t, side="right",) - 1)
    # Include values equal to the final edge.
    bin_indices[field_at_t == quantiles[-1]] = n_bins - 1

    valid = (np.isfinite(field_at_t) & (bin_indices >= 0) & (bin_indices < n_bins))

    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)

    b2_map_flat = np.full(field_at_t.size, np.nan)
    b2_map_flat[valid] = b2_bins[bin_indices[valid]]
    b2_map = b2_map_flat.reshape(spatial_shape)

    # Pointwise diffusion amplitude.
    b_map = np.sqrt(np.clip(b2_map, 0.0, None))

    return b2_map, b_map



