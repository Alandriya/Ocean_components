import numpy as np
import tqdm
import os
from scipy.linalg import sqrtm
from data_processing import scale_to_bins
from Plotting.plot_eigenvalues import plot_eigenvalues


def get_eig(B: np.ndarray,):
    # A1 = np.dot(B, B.conjugate())
    # A1 = sqrtm(A1)
    # eigenvalues, eigenvectors = np.linalg.eig(A1)
    eigenvalues, eigenvectors = np.linalg.eig(B)
    # sort by absolute value of the eigenvalues
    positions = [x for x in range(len(eigenvalues))]
    positions = [x for _, x in reversed(sorted(zip(np.abs(eigenvalues), positions)))]
    return eigenvalues, eigenvectors, positions


def count_eigenvalues_pair(files_path_prefix,
                           n_bins,
                           array1,
                           array1_quantiles,
                           array2,
                           array2_quantiles,
                           t,
                           offset,
                           mask,
                           n_lambdas,
                           names):
    b_matrix_vals = np.zeros((n_bins, n_bins), dtype=float)
    matrix = np.zeros(161 * 181)
    matrix[np.logical_not(mask)] = np.nan
    # matrix_list = [np.zeros(161 * 181) for _ in range(n_lambdas)]
    lambda_list = []
    # for l in range(n_lambdas):
    #     matrix_list[l][np.logical_not(mask)] = np.nan
    probs = np.zeros((n_bins, n_bins), dtype=float)
    for i in range(len(array1_quantiles)-1):
        qi_1 = array1_quantiles[i]
        qi_2 = array1_quantiles[i+1]
        i_centre = (qi_1 + qi_2) / 2
        points_x1 = np.where((qi_1 <= array1[:, t]) & (array1[:, t] < qi_2))[0]
        points_x2 = np.intersect1d(points_x1, np.where((i_centre <= array1[:, t + 1]) & (array1[:, t + 1] < qi_2))[0])
        for j in range(len(array2_quantiles)-1):
            qj_1 = array2_quantiles[j]
            qj_2 = array2_quantiles[j + 1]
            j_centre = (qj_1 + qj_2) / 2
            points_y1 = np.where((qj_1 <= array2[:, t]) & (array2[:, t] < qj_2))[0]
            points_y2 = np.intersect1d(points_y1, np.where((j_centre <= array2[:, t+1]) & (array2[:, t+1] < qj_2))[0])
            if len(points_x1) and len(points_y1):
                probs[i, j] = len(points_x2) * len(points_y2) / (len(points_x1) * len(points_y1))
                for p1 in points_x2:
                    for p2 in points_y2:
                        b_matrix_vals[i, j] += (array1[p1, t+1] - array1[p1, t])*(array2[p2, t+1] - array2[p2, t])*probs[i, j]

                b_matrix_vals[i, j] *= probs[i, j]

    eigenvalues, eigenvectors, positions = get_eig(b_matrix_vals)
    n_lambdas = 100
    for l in range(n_lambdas):
        ind = positions[l]
        points_x = np.where((array1_quantiles[ind] <= array1[:, t]) & (array1[:, t] < array1_quantiles[ind+1]))[0]
        points_y = np.where((array2_quantiles[ind] <= array2[:, t]) & (array2[:, t] < array2_quantiles[ind+1]))[0]
        points = np.union1d(points_x, points_y)
        # matrix_list[0][points] = float(eigenvalues[ind])
        lambda_list.append(eigenvalues[ind])
        matrix[points] = float(eigenvalues[ind])

    plot_eigenvalues(files_path_prefix, matrix, t + offset, (names[0], names[1]))
    return


def count_eigenvalues_triplets(files_path_prefix: str,
                               mask: np.ndarray,
                               flux_array: np.ndarray,
                               SST_array: np.ndarray,
                               press_array: np.ndarray,
                               offset: int,
                               n_bins: int = 100,
                               n_lambdas: int = 3,
                               ):
    # TODO remove -----------------------------------------------------------------------------------------
    # flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
    # SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)
    # press_array_grouped, quantiles_press = scale_to_bins(press_array, n_bins)
    # np.save(files_path_prefix + f'Fluxes/FLUX_2019_quantiles.npy', quantiles_flux)
    # np.save(files_path_prefix + f'SST/SST_2019_quantiles.npy', quantiles_sst)
    # np.save(files_path_prefix + f'Pressure/PRESS_2019_quantiles.npy', quantiles_press)

    quantiles_flux = np.load(files_path_prefix + f'Fluxes/FLUX_2019_quantiles.npy')
    quantiles_sst = np.load(files_path_prefix + f'SST/SST_2019_quantiles.npy')
    quantiles_press = np.load(files_path_prefix + f'Pressure/PRESS_2019_quantiles.npy')
    # -----------------------------------------------------------------------------------------------------

    if not os.path.exists(files_path_prefix + f'Eigenvalues'):
        os.mkdir(files_path_prefix + f'Eigenvalues')
    if not os.path.exists(files_path_prefix + f'Eigenvalues/tmp'):
        os.mkdir(files_path_prefix + f'Eigenvalues/tmp')

    for t in tqdm.tqdm(range(flux_array.shape[1]-1)):
        # flux-sst
        count_eigenvalues_pair(files_path_prefix, n_bins, flux_array, quantiles_flux, SST_array, quantiles_sst,
                               t, offset, mask, n_lambdas, ('Flux', 'SST'))
        # flux-press
        count_eigenvalues_pair(files_path_prefix, n_bins, flux_array, quantiles_flux, press_array, quantiles_press,
                               t, offset, mask, n_lambdas, ('Flux', 'Pressure'))

        # sst-press
        count_eigenvalues_pair(files_path_prefix, n_bins, SST_array, quantiles_sst, press_array, quantiles_press,
                               t, offset, mask, n_lambdas, ('SST', 'Pressure'))

        # flux-flux
        count_eigenvalues_pair(files_path_prefix, n_bins, flux_array, quantiles_flux, flux_array, quantiles_flux,
                               t, offset, mask, n_lambdas, ('Flux', 'Flux'))

        # SST-SST
        count_eigenvalues_pair(files_path_prefix, n_bins, SST_array, quantiles_sst, SST_array, quantiles_sst,
                               t, offset, mask, n_lambdas, ('SST', 'SST'))

        # press-press
        count_eigenvalues_pair(files_path_prefix, n_bins, press_array, quantiles_press, press_array, quantiles_press,
                               t, offset, mask, n_lambdas, ('Pressure', 'Pressure'))
    return
