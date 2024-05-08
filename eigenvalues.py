import numpy as np
import tqdm
import os
from scipy.linalg import sqrtm
from multiprocessing import Pool
import gc
from Plotting.plot_eigenvalues import plot_eigenvalues
import datetime

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


def get_eig(B: np.ndarray,
            names: tuple):
    """
    Counts eigenvalues for the big matrix B for two cases: if both the variables in the data arrays are the same, e.g.
    (Flux, Flux) and for different, e.g. (Flux, SST)
    :param B: np.array with shape (height * width, height * width), two-dimensional
    :param names: tuple with names of the data, e.g. ('Flux', 'SST'), ('Flux', 'Flux')
    :return:
    """
    if names[0] == names[1]:
        A = B
    else:
        # print('Performing A = B*B^T', flush=True)
        A = np.dot(B, B.transpose())
        # print('Getting sqrt(A)', flush=True)
        A = sqrtm(A)

    gc.collect()
    # print('Counting eigenvalues', flush=True)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # sort by absolute value of the eigenvalues
    eigenvalues = np.real(eigenvalues)
    eigenvalues = [0 if np.isnan(e) else e for e in eigenvalues]
    positions = [x for x in range(len(eigenvalues))]
    positions = [x for _, x in reversed(sorted(zip(np.abs(eigenvalues), positions)))]

    return np.take(eigenvalues, positions), np.take(eigenvectors, positions, axis=1), positions


def count_eigenvalues_pair(files_path_prefix: str,
                           array1,
                           array2,
                           array1_quantiles,
                           array2_quantiles,
                           t: int,
                           n_bins: int,
                           offset: int,
                           names: tuple):
    if not os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}'):
        os.mkdir(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}')

    if not os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}'):
        os.mkdir(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}')

    if os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvalues_{t + offset}.npy'):
        return

    b_matrix = np.zeros((n_bins, n_bins))
    for i1 in range(0, n_bins):
        points_x1 = np.where((array1_quantiles[i1] <= array1[:, t]) & (array1[:, t] < array1_quantiles[i1 + 1]))[0]
        for j1 in range(0, n_bins):
            points_y1 = np.where((array2_quantiles[j1] <= array2[:, t]) & (array2[:, t] < array2_quantiles[j1 + 1]))[0]
            if len(points_x1) and len(points_y1):
                mean1 = np.mean(array1[points_x1, t])
                mean2 = np.mean(array2[points_y1, t])
                vec1 = array1[points_x1, t + 1] - mean1
                vec2 = array2[points_y1, t + 1] - mean2
                b_matrix[i1, j1] = np.sum(np.multiply.outer(vec1, vec2).ravel())

    b_matrix = np.nan_to_num(b_matrix)

    # count eigenvalues
    eigenvalues, eigenvectors, positions = get_eig(b_matrix, (names[0], names[1]))
    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvalues_{t + offset}.npy', eigenvalues)
    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvectors_{t + offset}.npy', eigenvectors)
    return


def count_eigenvalues_triplets(files_path_prefix: str,
                               t_start: int,
                               flux_array: np.ndarray,
                               SST_array: np.ndarray,
                               press_array: np.ndarray,
                               mask: np.ndarray,
                               offset: int = 14610,
                               n_bins: int = 100,
                               cpu_amount: int = 16,
                               ):
    """
    Counts eigenvalues for all possible pairs in triplet (Flux, SST, Pressure) for time moment (t, t+1)
    :param files_path_prefix: path to the working directory
    :param flux_array: array with shape (height*width, n_days): e.g. (29141, 1410) with flux values
    :param SST_array:
    :param press_array:
    :param offset: shift of the beginning of the data arrays in days from 01.01.1979, for 01.01.2019 is 14610
    :param n_bins: amount of bins to divide the values of each array
    :param cpu_amount: amount of CPU used for parallel run
    :return:
    """

    flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
    SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)
    press_array_grouped, quantiles_press = scale_to_bins(press_array, n_bins)

    if not os.path.exists(files_path_prefix + f'Eigenvalues'):
        os.mkdir(files_path_prefix + f'Eigenvalues')
    if not os.path.exists(files_path_prefix + f'Eigenvalues/tmp'):
        os.mkdir(files_path_prefix + f'Eigenvalues/tmp')

    # for t in tqdm.tqdm(range(t_start, flux_array.shape[1])):
    for t in range(t_start, flux_array.shape[1]-1):
        # print(f'Timestep {t}')
        # flux-flux
        count_eigenvalues_pair(files_path_prefix, flux_array, flux_array, quantiles_flux, quantiles_flux, t, n_bins,
                               offset, ('Flux', 'Flux'))

        # sst-sst
        count_eigenvalues_pair(files_path_prefix, SST_array, SST_array, quantiles_sst, quantiles_sst, t, n_bins,
                               offset, ('SST', 'SST'))

        # flux-sst
        count_eigenvalues_pair(files_path_prefix, flux_array, SST_array, quantiles_flux, quantiles_sst, t, n_bins,
                               offset, ('Flux', 'SST'))

        # flux-pressure
        count_eigenvalues_pair(files_path_prefix, flux_array, press_array, quantiles_flux, quantiles_press, t, n_bins,
                               offset, ('Flux', 'Pressure'))

    plot_eigenvalues(files_path_prefix, 3, mask, 0, flux_array.shape[1], offset, flux_array, quantiles_flux,
                     ('Flux', 'Flux'))
    plot_eigenvalues(files_path_prefix, 3, mask, 0, SST_array.shape[1], offset, SST_array, quantiles_sst,
                     ('SST', 'SST'))
    plot_eigenvalues(files_path_prefix, 3, mask, 0, SST_array.shape[1], offset, SST_array, quantiles_sst,
                     ('Flux', 'SST'))
    plot_eigenvalues(files_path_prefix, 3, mask, 0, press_array.shape[1], offset, press_array, quantiles_press,
                     ('Flux', 'Pressure'))
    return


def count_mean_year(files_path_prefix: str,
                    start_year: int = 2009,
                    end_year: int = 2019,
                    names: tuple = ('Flux', 'Flux'),
                    mask: np.ndarray = None,
                    ):
    height, width = 161, 181
    mean_year = np.zeros((365, height, width))

    for year in tqdm.tqdm(range(start_year, end_year)):
        time_start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=1979, month=1, day=1)).days

        for day in range(364):
            matrix = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigen0_{day + time_start + 1}.npy')
            mean_year[day, :, :] += matrix
            mean_year[day][np.logical_not(mask)] = None

    mean_year /= (end_year - start_year)
    np.save(files_path_prefix + f'Mean_year/Lambdas_{names[0]}-{names[1]}_{start_year}-{end_year}.npy', mean_year)
    return
