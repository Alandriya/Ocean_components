import numpy as np
import tqdm
import os
from scipy.linalg import sqrtm
from data_processing import scale_to_bins
from Plotting.plot_eigenvalues import plot_eigenvalues


def get_save_eig(files_path_prefix,
                 t: int,
                 pair_name: str,
                 B: np.ndarray, ):
    A1 = np.dot(B, B.conjugate())
    A1 = sqrtm(A1)
    eigenvalues, eigenvectors = np.linalg.eig(A1)
    # sort by absolute value of the eigenvalues
    positions = [x for x in range(len(eigenvalues))]
    positions = [x for _, x in reversed(sorted(zip(np.abs(eigenvalues), positions)))]
    np.save(files_path_prefix + f'Eigenvalues/tmp/{t}_{pair_name}_eval1.npy', eigenvalues)
    np.save(files_path_prefix + f'Eigenvalues/tmp/{t}_{pair_name}_positions1.npy', positions)

    # A2 = np.dot(B.conjugate(), B)
    # A2 = sqrtm(A2)
    # eigenvalues, eigenvectors = np.linalg.eig(A2)
    # # sort by absolute value of the eigenvalues
    # positions = [x for x in range(len(eigenvalues))]
    # positions = [x for _, x in sorted(zip(np.abs(eigenvalues), positions))]
    # np.save(files_path_prefix + f'Eigenvalues/{t}_{pair_name}_eval2.npy', eigenvalues)
    # np.save(files_path_prefix + f'Eigenvalues/{t}_{pair_name}_positions2.npy', positions)
    return eigenvalues, positions


def count_eigenvalues_pair(files_path_prefix,
                           n_bins,
                           array1,
                           array2,
                           values1,
                           values2,
                           t,
                           offset,
                           pair_name,
                           mask,
                           n_lambdas,
                           names):
    probs = np.zeros((n_bins, n_bins), dtype=float)
    b_matrix_vals = np.zeros((n_bins, n_bins), dtype=float)
    for i in range(len(values1)):
        x1 = values1[i]
        for j in range(len(values2)):
            x2 = values2[j]
            points_x1 = np.where(array1[:, t] == x1)[0]
            points_x2 = np.where(array2[:, t] == x2)[0]
            intersection = np.intersect1d(points_x1, points_x2)
            probs2 = np.zeros((n_bins, n_bins), dtype=float)
            for k in range(len(values2)):
                y1 = values1[k]
                for l in range(len(values2)):
                    y2 = values2[l]
                    points_y1 = np.intersect1d(np.where(array1[:, t + 1] == y1)[0], points_x1)
                    points_y2 = np.intersect1d(np.where(array2[:, t + 1] == y2)[0], points_x2)
                    intersection2 = np.intersect1d(points_y1, points_y2)
                    if len(intersection):
                        probs2[k, l] = len(intersection2) / len(intersection)
                        b_matrix_vals[i, j] += (y1 - x1) * (y2 - x2) * probs2[k, l]

    np.save(files_path_prefix + f'Eigenvalues/tmp/{t + offset}_probs_{pair_name}.npy', probs)
    np.save(files_path_prefix + f'Eigenvalues/tmp/{t + offset}_b_vals_{pair_name}.npy', b_matrix_vals)
    eigenvalues, positions = get_save_eig(files_path_prefix, t + offset, pair_name, b_matrix_vals)
    matrix_list = list()
    lambdas_list = list()
    for i in range(n_bins):
        matrix = np.zeros(array1.shape[0])
        matrix[np.logical_not(mask)] = np.nan
        x1 = values1[positions[i]]
        x2 = values2[positions[i]]
        lambda_value = eigenvalues[positions[i]]
        points_x1 = np.where(array1[:, t] == x1)[0]
        points_x2 = np.where(array2[:, t] == x2)[0]
        intersection = np.intersect1d(points_x1, points_x2)
        # print(len(intersection))
        if len(intersection):
            matrix[intersection] = lambda_value
            matrix_list.append(matrix)
            lambdas_list.append(lambda_value)
        if len(matrix_list) >= n_lambdas:
            break
    plot_eigenvalues(files_path_prefix, matrix_list, t + offset, lambdas_list, (names[0], names[1]), n_lambdas)
    return


def count_eigenvalues_triplets(files_path_prefix: str,
                               mask: np.ndarray,
                               flux_array: np.ndarray,
                               values_flux: np.ndarray,
                               SST_array: np.ndarray,
                               values_sst: np.ndarray,
                               press_array: np.ndarray,
                               values_press: np.ndarray,
                               offset: int,
                               n_bins: int = 100,
                               n_lambdas: int = 3,
                               ):
    if not os.path.exists(files_path_prefix + f'Eigenvalues'):
        os.mkdir(files_path_prefix + f'Eigenvalues')
    if not os.path.exists(files_path_prefix + f'Eigenvalues/tmp'):
        os.mkdir(files_path_prefix + f'Eigenvalues/tmp')

    for t in tqdm.tqdm(range(flux_array.shape[1]-1)):
        # flux-sst
        count_eigenvalues_pair(files_path_prefix, n_bins, flux_array, SST_array, values_flux, values_sst, t, offset,
                               'flux-sst', mask, n_lambdas, ('Flux', 'SST'))
        # flux-press
        count_eigenvalues_pair(files_path_prefix, n_bins, flux_array, press_array, values_flux, values_press, t, offset,
                               'flux-press', mask, n_lambdas, ('Flux', 'Pressure'))

        # sst-press
        count_eigenvalues_pair(files_path_prefix, n_bins, SST_array, press_array, values_sst, values_press, t, offset,
                               'sst-press', mask, n_lambdas, ('SST', 'Pressure'))


        # flux-flux
        count_eigenvalues_pair(files_path_prefix, n_bins, flux_array, flux_array, values_flux, values_flux, t, offset,
                               'flux-flux', mask, n_lambdas, ('Flux', 'Flux'))

        # SST-SST
        count_eigenvalues_pair(files_path_prefix, n_bins, SST_array, SST_array, values_sst, values_sst, t, offset,
                               'sst-sst', mask, n_lambdas, ('SST', 'SST'))

        # press-press
        count_eigenvalues_pair(files_path_prefix, n_bins, press_array, press_array, values_press, values_press, t, offset,
                               'press-press', mask, n_lambdas, ('Pressure', 'Pressure'))
    return
