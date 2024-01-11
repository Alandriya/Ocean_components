import numpy as np
import tqdm
import os
from data_processing import scale_to_bins


def count_eigenvalues(files_path_prefix: str,
                      e_timelist: list,
                      start_idx: int = 0,
                      pair_name: str = '',
                      ):
    """
    :param files_path_prefix: path to the working directory
    :param e_timelist: list with e_matrix as elements, where e_matrix is np.array with shape (4, 161, 181)
        containing 4 matrices with elements of 2x2 matrix of coefficient E for every point of grid.
        0 is for E11 = sensible at t0 - sensible at t1,
        1 is for E12 = sensible at t0 - latent at t1,
        2 is for E21 = latent at t0 - sensible at t1,
        3 is for E22 = latent at t0 - latent at t1.
    :param start_idx: from which number to save arrays
    :param pair_name:
    :return:
    """
    if not os.path.exists(files_path_prefix + f'Eigenvalues'):
        os.mkdir(files_path_prefix + f'Eigenvalues')

    if not os.path.exists(files_path_prefix + f'Eigenvalues/{pair_name}'):
        os.mkdir(files_path_prefix + f'Eigenvalues/{pair_name}')

    print('Counting eigenvalues')
    for t in tqdm.tqdm(range(0, len(e_timelist))):
        e_matrix = e_timelist[t]
        shape = e_matrix.shape
        ev_matrix = np.zeros((2, shape[1], shape[2]))
        ev_matrix[:, np.isnan(e_matrix[0])] = np.nan

        for i in range(e_matrix.shape[1]):
            for j in range(e_matrix.shape[2]):
                if not np.isnan(e_matrix[:, i, j]).any():
                    eigenvalues, eigenvectors = np.linalg.eig(e_matrix[:, i, j].reshape((2,2)))
                    ev_matrix[:, i, j] = eigenvalues
                else:
                    ev_matrix[:, i, j] = np.nan

        np.save(files_path_prefix + f'Eigenvalues/{pair_name}/{start_idx + t}_lambda.npy', ev_matrix)
    return

