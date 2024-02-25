import numpy as np
import tqdm
import os
from scipy.linalg import sqrtm
from data_processing import scale_to_bins
from Plotting.plot_eigenvalues import plot_eigenvalues
from multiprocessing import Pool


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


def count_B(arg: list,
            ):
    print('My process id:', os.getpid())
    files_path_prefix, t, i_idxes, j_idxes, array1_first, array1_second, array1_quantiles, array2_first, array2_second, \
    array2_quantiles, names, shape = arg

    b_matrix = np.zeros(shape, dtype=float)
    # count b_matrix
    for i1 in i_idxes:
        for j1 in j_idxes:
            points_x1 = np.where((array1_quantiles[i1] <= array1_first) & (array1_first < array1_quantiles[i1 + 1]))[0]
            points_y1 = np.where((array2_quantiles[j1] <= array2_first) & (array2_first < array2_quantiles[j1 + 1]))[0]

            for i2 in range(len(array1_quantiles) - 1):
                points_x2 = np.where((array1_quantiles[i2] <= array1_second) & (array1_second < array1_quantiles[i2 + 1]))[0]
                for j2 in tqdm.tqdm(range(len(array2_quantiles) - 1)):
                    points_y2 = \
                    np.where((array2_quantiles[j2] <= array2_second) & (array2_second < array2_quantiles[j2 + 1]))[0]
                    if len(points_x2) and len(points_y2):
                        prob = len(points_x1) * len(points_y1) / (len(points_x2) * len(points_y2))
                        tmp = sum([(x2 - x1) * (y2 - y1) for x1 in points_x1 for x2 in points_x2 for y1 in points_y1 for y2 in
                                   points_y2])
                        b_matrix[i2, j2] = tmp * prob

            # save
            if not os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}'):
                os.mkdir(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}')
            if not os.path.exists(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t}'):
                os.mkdir(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t}')
            np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t}/B_{i1}_{j1}.npy', b_matrix)
    print(f'Process {os.getpid()} finished')
    return


def count_eigenvalues_pair(files_path_prefix,
                           cpu_amount,
                           array1,
                           array1_quantiles,
                           array2,
                           array2_quantiles,
                           t,
                           offset,
                           names):
    height, width = 161, 181
    b_matrix_vals = np.zeros((height, width, height, width), dtype=float)

    i_idxes = [list() for _ in range(cpu_amount)]
    j_idxes = [list() for _ in range(cpu_amount)]
    q_amount = len(array1_quantiles)-1
    delta = (q_amount * q_amount + cpu_amount) // cpu_amount
    for k in range(cpu_amount):
        for m in range(k*delta, (k+1)*delta):
            if m <= q_amount * q_amount:
                i_idxes[k].append(m // q_amount)
                j_idxes[k].append(m % q_amount)

    args = [[files_path_prefix, t+offset, i_idxes[k], j_idxes[k], array1[:, t], array1[:, t+1], array1_quantiles,
             array2[:, t], array2[:, t+1], array2_quantiles, names, (height, width)] for k in range(cpu_amount)]

    with Pool(cpu_amount) as p:
        p.map(count_B, args)
        p.close()
        p.join()

    print('All finished, counting eigenvalues')

    for i in range(len(array1_quantiles)-1):
        for j in range(len(array2_quantiles) - 1):
            b_matrix_vals = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/B_{t+offset}/B_{i}_{j}.npy')

    eigenvalues, eigenvectors, positions = get_eig(b_matrix_vals.reshape((height*width, height*width)), (names[0], names[1]))
    np.save(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvalues_{t+offset}.npy', eigenvalues)
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

    flux_array = flux_array[:, t:t+2]
    SST_array = SST_array[:, t:t+2]
    press_array = press_array[:, t:t+2]

    flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
    SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)
    press_array_grouped, quantiles_press = scale_to_bins(press_array, n_bins)

    if not os.path.exists(files_path_prefix + f'Eigenvalues'):
        os.mkdir(files_path_prefix + f'Eigenvalues')
    if not os.path.exists(files_path_prefix + f'Eigenvalues/tmp'):
        os.mkdir(files_path_prefix + f'Eigenvalues/tmp')

    for t in range(flux_array.shape[1]-1):
        # flux-sst
        count_eigenvalues_pair(files_path_prefix, cpu_amount, flux_array, quantiles_flux, SST_array, quantiles_sst,
                               0, t + offset, ('Flux', 'SST'))
        # flux-press
        count_eigenvalues_pair(files_path_prefix, cpu_amount, flux_array, quantiles_flux, press_array, quantiles_press,
                               0, t + offset, ('Flux', 'Pressure'))

        # sst-press
        count_eigenvalues_pair(files_path_prefix, cpu_amount, SST_array, quantiles_sst, press_array, quantiles_press,
                               0, t + offset, ('SST', 'Pressure'))

        # flux-flux
        count_eigenvalues_pair(files_path_prefix, cpu_amount, flux_array, quantiles_flux, flux_array, quantiles_flux,
                               0, t + offset, ('Flux', 'Flux'))

        # SST-SST
        count_eigenvalues_pair(files_path_prefix, cpu_amount, SST_array, quantiles_sst, SST_array, quantiles_sst,
                               0, t + offset, ('SST', 'SST'))

        # press-press
        count_eigenvalues_pair(files_path_prefix, cpu_amount, press_array, quantiles_press, press_array, quantiles_press,
                               0, t + offset, ('Pressure', 'Pressure'))
    return
