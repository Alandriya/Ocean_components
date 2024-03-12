import numpy as np
import tqdm
import os
from scipy.linalg import sqrtm
from multiprocessing import Pool
import gc

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
    if names[0] == names[1]:
        A = B
    else:
        A = np.dot(B, B.transpose())
        A = sqrtm(A)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # sort by absolute value of the eigenvalues
    positions = [x for x in range(len(eigenvalues))]
    positions = [x for _, x in reversed(sorted(zip(np.abs(eigenvalues), positions)))]
    return eigenvalues, eigenvectors, positions


def _count_B_ij(args:list):
    files_path_prefix, t, i1, j1, array1_first, array1_second, array1_quantiles, array2_first, array2_second, \
    array2_quantiles, names, shape = args
    print(f'My process id: {os.getpid()}, i1 = {i1}, j1 = {j1}', flush=True)
    b_matrix = np.zeros(shape, dtype=float)
    points_x1 = np.where((array1_quantiles[i1] <= array1_first) & (array1_first < array1_quantiles[i1 + 1]))[0]
    points_y1 = np.where((array2_quantiles[j1] <= array2_first) & (array2_first < array2_quantiles[j1 + 1]))[0]
    for i2 in range(len(array1_quantiles) - 1):
        points_x2 = np.where((array1_quantiles[i2] <= array1_second) & (array1_second < array1_quantiles[i2 + 1]))[0]
        for j2 in range(len(array2_quantiles) - 1):
            points_y2 = np.where((array2_quantiles[j2] <= array2_second) & (array2_second < array2_quantiles[j2 + 1]))[0]
            if len(points_x2) and len(points_y2):
                prob = len(points_x1) * len(points_y1) / (len(points_x2) * len(points_y2))
                arr1_first_vec = [array1_first[x1] for x1 in points_x1]
                arr1_second_vec = [array1_second[x2] for x2 in points_x2]
                array2_first_vec = [array2_first[y1] for y1 in points_y1]
                array2_second_vec = [array2_second[y2] for y2 in points_y2]
                tmp1 = [v2 - v1 for v1 in arr1_first_vec for v2 in arr1_second_vec]
                del arr1_first_vec, arr1_second_vec
                tmp2 = [v2 - v1 for v1 in array2_first_vec for v2 in array2_second_vec]
                del array2_first_vec, array2_second_vec
                tmp_sum = 0
                for v1 in tmp1:
                    for v2 in tmp2:
                        tmp_sum += v1*v2
                del tmp1, tmp2
                b_matrix[i2, j2] = tmp_sum * prob
                gc.collect()
            del points_y2
        del points_x2
    # save
    if not os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}'):
        os.mkdir(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}')
    if not os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t}'):
        os.mkdir(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t}')
    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t}/B_{i1}_{j1}.npy', b_matrix)
    del b_matrix
    print(f'Process {os.getpid()} finished', flush=True)
    return


def count_eigenvalues_parralel(files_path_prefix,
                           cpu_amount,
                           array1,
                           array1_quantiles,
                           array2,
                           array2_quantiles,
                           t,
                           offset,
                           names,
                           n_bins):
    height, width = 161, 181
    i_idxes = [list() for _ in range(cpu_amount)]
    j_idxes = [list() for _ in range(cpu_amount)]
    q_amount = len(array1_quantiles) - 1
    delta = (q_amount * q_amount + cpu_amount) // cpu_amount
    for k in range(cpu_amount):
        for m in range(k * delta, (k + 1) * delta):
            if m <= q_amount * q_amount:
                i_idxes[k].append(m // q_amount)
                j_idxes[k].append(m % q_amount)

    args_list = []
    for i1 in range(n_bins):
        for j1 in range(n_bins):
            if not os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t+offset}/B_{i1}_{j1}.npy'):
                print(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t}/B_{i1}_{j1}.npy')
                points_x1 = np.where((array1_quantiles[i1] <= array1[:, t]) & (array1[:, t] < array1_quantiles[i1 + 1]))[0]
                points_y1 = np.where((array2_quantiles[j1] <= array2[:, t]) & (array2[:, t] < array2_quantiles[j1 + 1]))[0]

                args = [files_path_prefix, t, i1, j1, points_x1, array1[:, t + 1], array1_quantiles, points_y1, array2[:, t + 1],
                        array2_quantiles, names, (height, width)]
                # _count_B_ij(args)
                args_list.append(args)

            if len(args_list) == cpu_amount:
                with Pool(cpu_amount) as p:
                    p.map(_count_B_ij, args_list)
                    p.close()
                    p.join()
                gc.collect()
                args_list = []

    print('All finished', flush=True)
    return


def collect_eigenvalues_pair(files_path_prefix,
                           array1_quantiles,
                           array2_quantiles,
                           t,
                           offset,
                           names):
    height, width = 161, 181
    b_matrix_vals = np.zeros((height, width, height, width), dtype=float)

    for i in range(len(array1_quantiles) - 1):
        for j in range(len(array2_quantiles) - 1):
            b_matrix_vals = np.load(
                files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t + offset}/B_{i}_{j}.npy')

    eigenvalues, eigenvectors, positions = get_eig(b_matrix_vals.reshape((height * width, height * width)),
                                                   (names[0], names[1]))
    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvalues_{t + offset}.npy', eigenvalues)
    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvectors_{t + offset}.npy', eigenvectors)
    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/positions_{t + offset}.npy', positions)
    return


def count_eigenvalues_triplets(files_path_prefix: str,
                               flux_array: np.ndarray,
                               SST_array: np.ndarray,
                               press_array: np.ndarray,
                               t: int,
                               offset: int,
                               n_bins: int = 100,
                               cpu_amount: int = 16,
                               ):
    flux_array = flux_array[:, t:t + 2]
    SST_array = SST_array[:, t:t + 2]
    press_array = press_array[:, t:t + 2]

    flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
    SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)
    press_array_grouped, quantiles_press = scale_to_bins(press_array, n_bins)

    if not os.path.exists(files_path_prefix + f'Eigenvalues'):
        os.mkdir(files_path_prefix + f'Eigenvalues')
    if not os.path.exists(files_path_prefix + f'Eigenvalues/tmp'):
        os.mkdir(files_path_prefix + f'Eigenvalues/tmp')

    for t in range(flux_array.shape[1] - 1):
        # flux-sst
        count_eigenvalues_parralel(files_path_prefix, cpu_amount, flux_array, quantiles_flux, SST_array, quantiles_sst,
                               0, t + offset, ('Flux', 'SST'), n_bins)
        collect_eigenvalues_pair(files_path_prefix, quantiles_flux, quantiles_sst, t, offset, ('Flux', 'SST'))

        # flux-press
        count_eigenvalues_parralel(files_path_prefix, cpu_amount, flux_array, quantiles_flux, press_array, quantiles_press,
                               0, t + offset, ('Flux', 'Pressure'), n_bins)
        collect_eigenvalues_pair(files_path_prefix, quantiles_flux, quantiles_press, t, offset, ('Flux', 'Pressure'))

        # sst-press
        count_eigenvalues_parralel(files_path_prefix, cpu_amount, SST_array, quantiles_sst, press_array, quantiles_press,
                               0, t + offset, ('SST', 'Pressure'), n_bins)
        collect_eigenvalues_pair(files_path_prefix, quantiles_sst, quantiles_press, t, offset, ('SST', 'Pressure'))

        # flux-flux
        count_eigenvalues_parralel(files_path_prefix, cpu_amount, flux_array, quantiles_flux, flux_array, quantiles_flux,
                               0, t + offset, ('Flux', 'Flux'), n_bins)
        collect_eigenvalues_pair(files_path_prefix, quantiles_flux, quantiles_flux, t, offset, ('Flux', 'Flux'))

        # SST-SST
        count_eigenvalues_parralel(files_path_prefix, cpu_amount, SST_array, quantiles_sst, SST_array, quantiles_sst,
                               0, t + offset, ('SST', 'SST'), n_bins)
        collect_eigenvalues_pair(files_path_prefix, quantiles_sst, quantiles_sst, t, offset, ('SST', 'SST'))

        # press-press
        count_eigenvalues_parralel(files_path_prefix, cpu_amount, press_array, quantiles_press, press_array,
                               quantiles_press,
                               0, t + offset, ('Pressure', 'Pressure'), n_bins)
        collect_eigenvalues_pair(files_path_prefix, quantiles_press, quantiles_press, t, offset, ('Pressure', 'Pressure'))
    return
