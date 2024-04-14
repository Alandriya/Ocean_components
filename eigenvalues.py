import numpy as np
import tqdm
import os
from scipy.linalg import sqrtm
from multiprocessing import Pool
import gc
from Plotting.plot_eigenvalues import plot_eigenvalues


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
        print('Performing A = B*B^T', flush=True)
        A = np.dot(B, B.transpose())
        print('Getting sqrt(A)', flush=True)
        A = sqrtm(A)

    gc.collect()
    print('Counting eigenvalues', flush=True)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # sort by absolute value of the eigenvalues
    positions = [x for x in range(len(eigenvalues))]
    positions = [x for _, x in reversed(sorted(zip(np.abs(eigenvalues), positions)))]
    return eigenvalues, eigenvectors, positions


def _count_B_i(args):
    points_x1, points_y1, i1, j1, n_bins, array1_quantiles, array2_quantiles, array1, array2, offset, \
    files_path_prefix, names = args
    tmp_sum = 0
    prob = 0
    t = 0
    if len(points_x1) and len(points_y1):
        for pair2_idx in range(n_bins * n_bins):
            i2 = pair2_idx // n_bins
            j2 = pair2_idx % n_bins
            points_x2 = np.where((array1_quantiles[i2] <= array1[:, t + 1]) &
                                 (array1[:, t + 1] < array1_quantiles[i2 + 1]))[0]
            points_x2 = np.intersect1d(points_x2, points_x1)
            points_y2 = np.where((array2_quantiles[j2] <= array2[:, t + 1]) &
                                 (array2[:, t + 1] < array1_quantiles[j2 + 1]))[0]
            points_y2 = np.intersect1d(points_y2, points_y1)
            if len(points_x2) and len(points_y2):
                prob = len(points_x2) * len(points_y2) * 1.0 / (len(points_x1) * len(points_y1))
                arr1_first_vec = [array1[x1, t] for x1 in points_x1]
                arr1_second_vec = [array1[x2, t + 1] for x2 in points_x2]
                array2_first_vec = [array2[y1, t] for y1 in points_y1]
                array2_second_vec = [array2[y2, t + 1] for y2 in points_y2]
                tmp1 = [v2 - v1 for v1 in arr1_first_vec for v2 in arr1_second_vec]
                del arr1_first_vec, arr1_second_vec
                tmp2 = [v2 - v1 for v1 in array2_first_vec for v2 in array2_second_vec]
                del array2_first_vec, array2_second_vec
                tmp_sum = 0
                for v1 in tmp1:
                    for v2 in tmp2:
                        tmp_sum += v1 * v2
                del tmp1, tmp2
        np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}/{i1}_{j1}_points_x1.npy',
                points_x1)
        np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}/{i1}_{j1}_points_y1.npy',
                points_y1)
        np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}/B_{i1}_{j1}.npy',
                [tmp_sum * prob])
        print(f'Ended i={i1}, j= {j1}, process id: {os.getpid()}', flush=True)
    return


def count_eigenvalues_pair(files_path_prefix: str,
                            i_start: int,
                             array1,
                             array2,
                             array1_quantiles,
                             array2_quantiles,
                             t: int,
                             n_bins: int,
                             offset: int,
                             names: tuple):
    height, width = 161, 181

    if not os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}'):
        os.mkdir(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}')

    if not os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}'):
        os.mkdir(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}')

    # # count B
    # for i1 in range(i_start, i_start + 10):
    #     args_list = list()
    #     for j1 in range(0, n_bins):
    #         points_x1 = np.where((array1_quantiles[i1] <= array1[:, t]) & (array1[:, t] < array1_quantiles[i1 + 1]))[0]
    #         points_y1 = np.where((array2_quantiles[j1] <= array2[:, t]) & (array2[:, t] < array2_quantiles[j1 + 1]))[0]
    #         args = [points_x1, points_y1, i1, j1, n_bins, array1_quantiles, array2_quantiles, array1[:, t:t+2],
    #                 array2[:, t:t+2], t + offset, files_path_prefix, names]
    #         args_list.append(args)
    #     with Pool(n_bins) as p:
    #         p.map(_count_B_i, args_list)
    #         p.close()
    #         p.join()
    #     gc.collect()

    if not os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}.npy'):
        # collect b_matrix
        print('Collecting B matrix', flush=True)
        b_matrix_vals = np.zeros((height * width, height * width), dtype=float)
        for i1 in range(0, n_bins):
            for j1 in range(0, n_bins):
                if os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}/{i1}_{j1}_points_x1.npy') and \
                    os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}/{i1}_{j1}_points_y1.npy'):
                    points_x1 = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}/{i1}_{j1}_points_x1.npy')
                    points_y1 = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}/{i1}_{j1}_points_y1.npy')
                    for x1 in points_x1:
                        for y1 in points_y1:
                            b_matrix_vals[x1, y1] = \
                        np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}/B_{i1}_{j1}.npy')[0]
        np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}.npy', b_matrix_vals)
    else:
        b_matrix_vals = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}.npy')

    print('B matrix ready!', flush=True)
    gc.collect()
    # count eigenvalues
    eigenvalues, eigenvectors, positions = get_eig(b_matrix_vals, (names[0], names[1]))
    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvalues_{t + offset}.npy', eigenvalues)
    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvectors_{t + offset}.npy', eigenvectors)
    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/positions_{t + offset}.npy', positions)
    return


def count_eigenvalues_triplets(files_path_prefix: str,
                               i_start: int,
                               flux_array: np.ndarray,
                               SST_array: np.ndarray,
                               press_array: np.ndarray,
                               mask: np.ndarray,
                               t: int = 0,
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
    :param t: time moment from the beginning of the array = time relative
    :param offset: shift of the beginning of the data arrays in days from 01.01.1979, for 01.01.2019 is 14610
    :param n_bins: amount of bins to divide the values of each array
    :param cpu_amount: amount of CPU used for parallel run
    :return:
    """
    flux_array = flux_array[:, t:t + 2]
    SST_array = SST_array[:, t:t + 2]
    press_array = press_array[:, t:t + 2]

    flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
    SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)
    # press_array_grouped, quantiles_press = scale_to_bins(press_array, n_bins)

    if not os.path.exists(files_path_prefix + f'Eigenvalues'):
        os.mkdir(files_path_prefix + f'Eigenvalues')
    if not os.path.exists(files_path_prefix + f'Eigenvalues/tmp'):
        os.mkdir(files_path_prefix + f'Eigenvalues/tmp')

    # flux-sst
    # if not os.path.exists(files_path_prefix + f'Eigenvalues/Flux-SST/eigenvalues_{t + offset}.npy'):
    #     count_eigenvalues_pair(files_path_prefix, flux_array, SST_array, quantiles_flux, quantiles_sst, t, n_bins,
    #                              offset, ('Flux', 'SST'))
    # plot_eigenvalues(files_path_prefix, 3, mask, t+offset, ('Flux', 'SST'))

    # flux-flux
    if not os.path.exists(files_path_prefix + f'Eigenvalues/Flux-Flux/eigenvalues_{t + offset}.npy'):
        count_eigenvalues_pair(files_path_prefix, i_start, flux_array, flux_array, quantiles_flux, quantiles_flux, t, n_bins,
                             offset, ('Flux', 'Flux'))
    plot_eigenvalues(files_path_prefix, 3, mask, t + offset, ('Flux', 'Flux'))
    return
