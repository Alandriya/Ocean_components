from __future__ import annotations
import datetime
import os
import numpy as np


def scale_to_bins(arr, bins=100):
    quantiles = list(np.nanquantile(arr, np.linspace(0, 1, bins, endpoint=False)))

    arr_scaled = np.zeros_like(arr)
    arr_scaled[np.isnan(arr)] = np.nan
    # for j in tqdm.tqdm(range(bins - 1)):
    for j in range(bins - 1):
        arr_scaled[np.where((np.logical_not(np.isnan(arr))) & (quantiles[j] <= arr) & (arr < quantiles[j + 1]))] = \
            (quantiles[j] + quantiles[j + 1]) / 2

    quantiles += [np.nanmax(arr)]

    return arr_scaled, quantiles


def get_eig(b_matrix: np.ndarray,
            names: tuple):
    """
    Counts eigenvalues for the covariances matrix B for two cases: if both the variables in the data arrays are the
    same, e.g. (Flux, Flux) and for different, e.g. (Flux, SST)
    :param b_matrix: np.array with shape (n_bins, n_bins), two-dimensional
    :param names: tuple with names of the data, e.g. ('Flux', 'SST'), ('Flux', 'Flux')
    :return:
    """
    if names[0] == names[1]:
        # Same-variable matrix.
        covariance = 0.5 * (b_matrix + b_matrix.T)
    else:
        # Left covariance for the first variable.
        covariance = b_matrix @ b_matrix.T

    # Eliminate tiny numerical asymmetry.
    covariance = 0.5 * (covariance + covariance.T)

    # eigh is intended for symmetric matrices.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort from largest eigenvalue to smallest.
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Remove small negative values caused by numerical roundoff.
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    return eigenvalues, eigenvectors


def get_bins(values, quantiles):
    """
    Convert field values to bin indices.
    Invalid values receive bin index -1.
    """
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    bins = np.searchsorted(
        quantiles,
        values,
        side="right",
    ) - 1

    # Include values exactly equal to the final bin boundary.
    bins[values == quantiles[-1]] = len(quantiles) - 2
    invalid = (~np.isfinite(values) | (bins < 0)| (bins >= len(quantiles) - 1))
    bins[invalid] = -1

    return bins

def matrix_to_map(
    matrix,
    field1,
    field2,
    quantiles1,
    quantiles2,
    spatial_shape,
):
    """
    Map an n_bins x n_bins matrix to the geographical grid: map[p] = matrix[bin1[p], bin2[p]]
    """
    bins1 = get_bins(field1, quantiles1)
    bins2 = get_bins(field2, quantiles2)

    valid = ((bins1 >= 0) & (bins2 >= 0) & np.isfinite(field1) & np.isfinite(field2))

    result = np.full(field1.shape[0], np.nan)
    result[valid] = matrix[bins1[valid], bins2[valid],]
    return result.reshape(spatial_shape)

def bin_values_to_map(
    bin_values,
    field,
    quantiles,
    spatial_shape,
):
    """
    Maps one value per bin to the geographical grid:
        map[p] = bin_values[bin[p]]
    This is used for the diagonal b^2 reconstruction.
    """
    bins = get_bins(field, quantiles)

    valid = ((bins >= 0) & np.isfinite(field))

    result = np.full(field.shape[0], np.nan)
    result[valid] = bin_values[bins[valid]]
    return result.reshape(spatial_shape)


def count_eigenvalues_pair(
    files_path_prefix: str,
    array1: np.ndarray,
    array2: np.ndarray,
    array1_quantiles: list,
    array2_quantiles: list,
    n_bins: int,
    offset: int,
    names: tuple,
    spatial_shape: tuple,
    dt: float = 1.0,
    n_components: int = 3,
):
    """
    Counts the Karhunen-Loeve decomposition for the pair of two variables, e.g. sensible and latent fluxes
    :param files_path_prefix: path to the working directory
    :param array1: array with shape (height*width, n_days): e.g. (29141, 1410)
    :param array2: array with shape (height*width, n_days): e.g. (29141, 1410)
    :param array1_quantiles: list with length = n_bins + 1 of the quantiles built by scale_to_bins function
    :param array2_quantiles: list with length = n_bins + 1 of the quantiles built by scale_to_bins function
    :param n_bins: amount of bins to divide the values of each array
    :param offset: shift of the beginning of the data arrays in days from 01.01.1979, for 01.01.2019 is 14610
    :param names: tuple with names of the data arrays, e.g. ('Flux', 'SST')
    :param spatial_shape: shape of the map, e.g. (161, 181)
    :param dt: time step
    :param n_components: amount of eigenvalues and eigenvectors to get
    :return:
    """
    if not os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}'):
        os.mkdir(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}')

    for t in range(array1.shape[1]-1):
        if (t + offset) % 100 == 0:
            print(f'Counting timestep {t + offset}')
        if os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvalues_{t + offset}.npy'):
            continue

        b_matrix = np.zeros((n_bins, n_bins))
        for i in range(0, n_bins):
            for j in range(0, n_bins):
                joint_points = np.where(
                    (array1_quantiles[i] <= array1[:, t])
                    & (array1[:, t] < array1_quantiles[i + 1])
                    & (array2_quantiles[j] <= array2[:, t])
                    & (array2[:, t] < array2_quantiles[j + 1])
                )[0]
                dx = array1[joint_points, t + 1] - array1[joint_points, t]
                dy = array2[joint_points, t + 1] - array2[joint_points, t]

                b_matrix[i, j] = np.mean(dx * dy) / dt

        b_matrix = np.nan_to_num(b_matrix)
        N = n_components
        # if same variable
        if names[0] == names[1]:
            # count eigenvalues
            b_matrix = 0.5 * (b_matrix + b_matrix.T) # A covariance matrix must be symmetric.
            eigenvalues, eigenvectors = np.linalg.eigh(b_matrix)
            # Sort from largest to smallest.
            order = np.argsort(eigenvalues)[::-1]

            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvalues_{t + offset}.npy', eigenvalues)
            np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvectors_{t + offset}.npy', eigenvectors)

            C_N = (eigenvectors[:, :N] * eigenvalues[:N]) @ eigenvectors[:, :N].T
            """
            C_N = np.zeros((eigenvectors.shape[0], eigenvectors.shape[0]))
            for k in range(N):
                eigenvalue = eigenvalues[k]
                eigenvector = eigenvectors[:, k]
            
                C_N += eigenvalue * np.outer(eigenvector, eigenvector)
            """
            np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/Cn_{t + offset}.npy', C_N)
            C_N_map = matrix_to_map(
                matrix=C_N,
                field1=array1[:, t],
                field2=array2[:, t],
                quantiles1=array1_quantiles,
                quantiles2=array2_quantiles,
                spatial_shape=spatial_shape,
            )
            np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/Cn_map_{t + offset}.npy', C_N_map)
            # --------------------------------------------------------
            # Reconstructed b^2:
            # b2_bins[j] = C_N[j, j]
            # This is the diagonal of the reconstructed covariance.
            # --------------------------------------------------------

            b2_bins = np.diag(C_N)
            b2_map = bin_values_to_map(
                bin_values=b2_bins,
                field=array1[:, t],
                quantiles=array1_quantiles,
                spatial_shape=spatial_shape,
            )
            np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B2_map_{t + offset}.npy', b2_map)
            # Diffusion amplitude b = sqrt(b^2).
            b_map = np.sqrt(np.clip(b2_map, 0.0, None))
            np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_map_{t + offset}.npy', b_map)
        else:
            # if different variables
            U, singular_values, Vt = np.linalg.svd(
                b_matrix,
                full_matrices=False,
            )
            # Rank-N cross-matrix reconstruction:
            # B_N = sum_k s_k * u_k * v_k.T
            B_N = ( U[:, :N] * singular_values[:N]) @ Vt[:N, :]
            np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/Bn_{t + offset}.npy', B_N)
            B_N_map = matrix_to_map(
                matrix=B_N,
                field1=array1[:, t],
                field2=array2[:, t],
                quantiles1=array1_quantiles,
                quantiles2=array2_quantiles,
                spatial_shape=spatial_shape,
            )
            np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/Bn_map_{t + offset}.npy', B_N_map)
    return


def count_eigenvalues_triplets(files_path_prefix: str,
                               flux_array: np.ndarray,
                               SST_array: np.ndarray,
                               press_array: np.ndarray,
                               spatial_shape: tuple,
                               offset: int = 0,
                               n_bins: int = 100,
                               ):
    """
    Counts and plots eigenvalues and eigenvectors for pairs Flux-Flux, SST-SST, Flux-SST, Flux-Pressure for time range
    offset: offset + len(data_array)
    :param files_path_prefix: path to the working directory
    :param flux_array: array with shape (height*width, n_days): e.g. (29141, 1410) with flux values
    :param SST_array: array with shape (height*width, n_days): e.g. (29141, 1410) with SST values
    :param press_array: array with shape (height*width, n_days): e.g. (29141, 1410) with pressure values
    :param spatial_shape: shape of the map, e.g. (161, 181)
    :param offset: shift of the beginning of the data arrays in days from 01.01.1979, for 01.01.2019 is 14610
    :param n_bins: amount of bins to divide the values of each array
    :return:
    """

    flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
    SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)
    press_array_grouped, quantiles_press = scale_to_bins(press_array, n_bins)

    if not os.path.exists(files_path_prefix + f'Eigenvalues'):
        os.mkdir(files_path_prefix + f'Eigenvalues')

    # flux-flux
    count_eigenvalues_pair(files_path_prefix, flux_array, flux_array, quantiles_flux, quantiles_flux, n_bins,
                           offset, ('Flux', 'Flux'), spatial_shape)

    # sst-sst
    count_eigenvalues_pair(files_path_prefix, SST_array, SST_array, quantiles_sst, quantiles_sst, n_bins,
                           offset, ('SST', 'SST'), spatial_shape)

    # press-press
    count_eigenvalues_pair(files_path_prefix, press_array, press_array, quantiles_press, quantiles_press, n_bins,
                           offset, ('Pressure', 'Pressure'), spatial_shape)

    # flux-sst
    count_eigenvalues_pair(files_path_prefix, flux_array, SST_array, quantiles_flux, quantiles_sst, n_bins,
                           offset, ('Flux', 'SST'), spatial_shape)

    # flux-pressure
    count_eigenvalues_pair(files_path_prefix, flux_array, press_array, quantiles_flux, quantiles_press, n_bins,
                           offset, ('Flux', 'Pressure'), spatial_shape)

    # sst-pressure
    count_eigenvalues_pair(files_path_prefix, SST_array, press_array, quantiles_sst, quantiles_press, n_bins,
                           offset, ('SST', 'Pressure'), spatial_shape)

    return


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
        max_eigenvector[t] = np.nanmax(matrix)
        min_eigenvector[t] = np.nanmin(matrix)
        mean_eigenvector[t] = np.nanmean(matrix)

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



