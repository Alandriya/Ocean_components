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
        probs = np.zeros((n_bins, n_bins), dtype=float)
        b_matrix_vals = np.zeros((n_bins, n_bins), dtype=float)
        for i in range(len(values_flux)):
            x1 = values_flux[i]
            for j in range(len(values_sst)):
                x2 = values_sst[j]
                points_x1 = np.where(flux_array[:, t] == x1)[0]
                points_x2 = np.where(SST_array[:, t+1] == x2)[0]
                intersection = np.intersect1d(points_x1, points_x2)
                probs[i, j] = len(intersection) / len(points_x1)

                b_matrix_vals[i, j] = (x1 - x2)**2 * probs[i, j]
        np.save(files_path_prefix + f'Eigenvalues/tmp/{t + offset}_probs_flux-sst.npy', probs)
        np.save(files_path_prefix + f'Eigenvalues/tmp/{t + offset}_b_vals_flux-sst.npy', b_matrix_vals)
        eigenvalues, positions = get_save_eig(files_path_prefix, t + offset, 'flux-sst', b_matrix_vals)

        matrix_list = list()
        lambdas_list = list()
        for i in range(n_bins):
            matrix = np.zeros(flux_array.shape[0])
            matrix[np.logical_not(mask)] = np.nan
            x1 = values_flux[positions[i]]
            x2 = values_sst[positions[i]]
            lambda_value = eigenvalues[positions[i]]
            points_x1 = np.where(flux_array[:, t] == x1)[0]
            points_x2 = np.where(SST_array[:, t+1] == x2)[0]
            intersection = np.intersect1d(points_x1, points_x2)
            # print(len(intersection))
            if len(intersection):
                matrix[intersection] = lambda_value
                matrix_list.append(matrix)
                lambdas_list.append(lambda_value)
            if len(matrix_list) >= n_lambdas:
                break
        plot_eigenvalues(files_path_prefix, matrix_list, t + offset, lambdas_list, ('Flux', 'SST'), n_lambdas)

        # flux-press
        probs = np.zeros((n_bins, n_bins), dtype=float)
        b_matrix_vals = np.zeros((n_bins, n_bins), dtype=float)
        for i in range(len(values_flux)):
            x1 = values_flux[i]
            for j in range(len(values_press)):
                x2 = values_press[j]
                points_x1 = np.where(flux_array[:, t] == x1)[0]
                points_x2 = np.where(press_array[:, t + 1] == x2)[0]
                intersection = np.intersect1d(points_x1, points_x2)
                probs[i, j] = len(intersection) / len(points_x1)

                b_matrix_vals[i, j] = (x1 - x2) ** 2 * probs[i, j]
        np.save(files_path_prefix + f'Eigenvalues/tmp/{t + offset}_probs_flux-press.npy', probs)
        np.save(files_path_prefix + f'Eigenvalues/tmp/{t + offset}_b_vals_flux-press.npy', b_matrix_vals)
        eigenvalues, positions = get_save_eig(files_path_prefix, t + offset, 'flux-press', b_matrix_vals)

        matrix_list = list()
        lambdas_list = list()
        for i in range(n_bins):
            matrix = np.zeros(flux_array.shape[0])
            matrix[np.logical_not(mask)] = np.nan
            x1 = values_flux[positions[i]]
            x2 = values_press[positions[i]]
            lambda_value = eigenvalues[positions[i]]
            points_x1 = np.where(flux_array[:, t] == x1)[0]
            points_x2 = np.where(press_array[:, t + 1] == x2)[0]
            intersection = np.intersect1d(points_x1, points_x2)
            # print(len(intersection))
            if len(intersection):
                matrix[intersection] = lambda_value
                matrix_list.append(matrix)
                lambdas_list.append(lambda_value)
            if len(matrix_list) >= n_lambdas:
                break
        plot_eigenvalues(files_path_prefix, matrix_list, t + offset, lambdas_list, ('Flux', 'Pressure'), n_lambdas)

        # sst-press
        probs = np.zeros((n_bins, n_bins), dtype=float)
        b_matrix_vals = np.zeros((n_bins, n_bins), dtype=float)
        for i in range(len(values_sst)):
            x1 = values_sst[i]
            for j in range(len(values_press)):
                x2 = values_press[j]
                points_x1 = np.where(SST_array[:, t] == x1)[0]
                points_x2 = np.where(press_array[:, t + 1] == x2)[0]
                intersection = np.intersect1d(points_x1, points_x2)
                probs[i, j] = len(intersection) / len(points_x1)

                b_matrix_vals[i, j] = (x1 - x2) ** 2 * probs[i, j]
        np.save(files_path_prefix + f'Eigenvalues/tmp/{t + offset}_probs_sst-press.npy', probs)
        np.save(files_path_prefix + f'Eigenvalues/tmp/{t + offset}_b_vals_sst-press.npy', b_matrix_vals)
        eigenvalues, positions = get_save_eig(files_path_prefix, t + offset, 'sst-press', b_matrix_vals)

        matrix_list = list()
        lambdas_list = list()
        for i in range(n_bins):
            matrix = np.zeros(SST_array.shape[0])
            matrix[np.logical_not(mask)] = np.nan
            x1 = values_sst[positions[i]]
            x2 = values_press[positions[i]]
            lambda_value = eigenvalues[positions[i]]
            points_x1 = np.where(SST_array[:, t] == x1)[0]
            points_x2 = np.where(press_array[:, t + 1] == x2)[0]
            intersection = np.intersect1d(points_x1, points_x2)
            # print(len(intersection))
            if len(intersection):
                matrix[intersection] = lambda_value
                matrix_list.append(matrix)
                lambdas_list.append(lambda_value)
            if len(matrix_list) >= n_lambdas:
                break
        plot_eigenvalues(files_path_prefix, matrix_list, t + offset, lambdas_list, ('SST', 'Pressure'), n_lambdas)
    return
