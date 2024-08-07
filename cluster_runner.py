import argparse
import numpy as np
import pandas as pd
from struct import unpack
import os
from scipy.linalg import sqrtm
from scipy.linalg.interpolative import estimate_spectral_norm
from numpy.linalg import norm
from multiprocessing import Pool
import datetime
# from eigenvalues import count_eigenvalues_parralel, scale_to_bins
from eigenvalues import count_eigenvalues_triplets, count_mean_year, get_trends
from Plotting.video import create_video
import copy

files_path_prefix = '/home/aosipova/EM_ocean/'

def count_abfe_coefficients(files_path_prefix: str,
                           mask: np.ndarray,
                           sensible_array: np.ndarray,
                           latent_array: np.ndarray,
                           time_start: int = 0,
                           time_end: int = 0,
                           offset: int = 0,
                           pair_name: str = ''):
    """
    Counts A B and F coefficients, saves them to files_path_prefix + Coeff_data dir.
    :param files_path_prefix: path to the working directory
    :param mask: boolean 1D mask with length 161*181. If true, it's ocean point, if false - land. Only ocean points are
        of interest
    :param sensible_array: np.array with expected shape (161*181, days), where days amount may differ
    :param latent_array: np.array with expected shape (161*181, days), where days amount may differ
    :param time_start: offset in days from the beginning of the flux arrays for the first counted element
    :param time_end: offset in days from the beginning of the flux arrays for the last counted element
        (that day included!)
    :param offset: offset in days from 01.01.1979, indicating the day of corresponding index 0 in flux arrays
    :param pair_name:
    :return:
    """
    # !!NOTE: t_absolut here is not an error in naming, it means not a global absolute index - offset from 01.01.1979,
    # but it is absolute in terms of fluxes array from the input indexing
    # for t_absolute in tqdm.tqdm(range(time_start + 1, time_end + 1)):     # comment tqdm if parallel counting
    for t_absolute in range(time_start + 1, time_end + 1):
        if t_absolute % 100 ==0:
            print(f'Iteration {t_absolute}', flush=True)
        if not os.path.exists(files_path_prefix + f'Coeff_data_3d/{pair_name}/{int(t_absolute + offset)}_A_sens.npy'):
        # if True:
            t_rel = t_absolute - time_start
            a_sens = np.zeros((161, 181), dtype=float)
            a_lat = np.zeros((161, 181), dtype=float)
            b_matrix = np.zeros((4, 161, 181), dtype=float)
            f = np.zeros((161, 181), dtype=float)

            # set nan where is not ocean in arrays
            for i in range(0, len(mask)):
                if not mask[i]:
                    a_sens[i // 181][i % 181] = np.nan
                    a_lat[i // 181][i % 181] = np.nan
                    b_matrix[:, i // 181, i % 181] = np.nan
                    f[i // 181, i % 181] = np.nan

            set_sens = np.unique(sensible_array[:, t_rel - 1])
            set_lat = np.unique(latent_array[:, t_rel - 1])

            for val_t0 in set_sens:
                if not np.isnan(val_t0):
                    points_sensible = np.where(sensible_array[:, t_rel - 1] == val_t0)[0]
                    amount_t0 = len(points_sensible)

                    # sensible t0 - sensible t1
                    set_t1 = np.unique(sensible_array[points_sensible, t_rel])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = len(np.where(sensible_array[points_sensible, t_rel] == val_t1)[0]) * 1.0 / amount_t0
                        probabilities.append(prob)

                    a = sum([(list(set_t1)[i] - val_t0) * probabilities[i] for i in range(len(probabilities))])
                    b_squared = sum(
                        [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

                    for idx in points_sensible:
                        a_sens[idx // 181][idx % 181] = a
                        b_matrix[0][idx // 181][idx % 181] = b_squared

                    # sensible t0 - latent t1
                    set_t1 = np.unique(latent_array[points_sensible, t_rel])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = len(np.where(latent_array[points_sensible, t_rel] == val_t1)[0]) * 1.0 / amount_t0
                        probabilities.append(prob)
                    b_squared = sum(
                        [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

                    for idx in points_sensible:
                        b_matrix[1][idx // 181][idx % 181] = b_squared

            for val_t0 in set_lat:
                if not np.isnan(val_t0):
                    points_latent = np.where(latent_array[:, t_rel - 1] == val_t0)[0]
                    amount_t0 = len(points_latent)

                    # latent - latent
                    set_t1 = np.unique(latent_array[points_latent, t_rel])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = len(np.where(latent_array[points_latent, t_rel] == val_t1)[0]) * 1.0 / amount_t0
                        probabilities.append(prob)

                    a = sum([(set_t1[i] - val_t0) * probabilities[i] for i in range(len(probabilities))])
                    b_squared = sum(
                        [(set_t1[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

                    for idx in points_latent:
                        a_lat[idx // 181][idx % 181] = a
                        b_matrix[3][idx // 181][idx % 181] = b_squared

                    # latent t0 - sensible t1
                    set_t1 = np.unique(sensible_array[points_latent, t_rel])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = len(np.where(sensible_array[points_latent, t_rel] == val_t1)[0]) * 1.0 / amount_t0
                        probabilities.append(prob)

                    b_squared = sum(
                        [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

                    for idx in points_latent:
                        b_matrix[2][idx // 181][idx % 181] = b_squared

            e_matrix = copy.deepcopy(b_matrix)
            # get matrix root from B and count F
            for i in range(161):
                for j in range(181):
                    if not np.isnan(b_matrix[:, i, j]).any():
                        b_matrix[:, i, j] = sqrtm(b_matrix[:, i, j].reshape(2, 2)).reshape(4)
                        a_vec = [a_sens[i, j], a_lat[i, j]]
                        b_part = b_matrix[:, i, j].reshape(2, 2)
                        f[i, j] = norm(a_vec, 2) / estimate_spectral_norm(b_part)
                    else:
                        f[i, j] = np.nan

            # save data
            if pair_name == 'sensible-latent':
                np.save(files_path_prefix + f'Coeff_data/{int(t_absolute + offset)}_A_sens.npy', a_sens)
                np.save(files_path_prefix + f'Coeff_data/{int(t_absolute + offset)}_A_lat.npy', a_lat)
                np.save(files_path_prefix + f'Coeff_data/{int(t_absolute + offset)}_B.npy', b_matrix)
                np.save(files_path_prefix + f'Coeff_data/{int(t_absolute + offset)}_F.npy', f)
            else:
                if not os.path.exists(files_path_prefix + f'Coeff_data_3d/{pair_name}'):
                    os.mkdir(files_path_prefix + f'Coeff_data_3d/{pair_name}')
                np.save(files_path_prefix + f'Coeff_data_3d/{pair_name}/{int(t_absolute + offset)}_A_sens.npy', a_sens)
                np.save(files_path_prefix + f'Coeff_data_3d/{pair_name}/{int(t_absolute + offset)}_A_lat.npy', a_lat)
                np.save(files_path_prefix + f'Coeff_data_3d/{pair_name}/{int(t_absolute + offset)}_B.npy', b_matrix)
                np.save(files_path_prefix + f'Coeff_data_3d/{pair_name}/{int(t_absolute + offset)}_F.npy', f)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("start_year", help="Amount of processes to parallel run", type=int)
    parser.add_argument("t_start", type=int)
    args_cmd = parser.parse_args()

    start_year = args_cmd.start_year
    t_start = args_cmd.t_start

    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    # ---------------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2023, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    # count ABF coefficients 3d
    if start_year == 2019:
        end_year = 2025
    else:
        end_year = start_year + 10

    if start_year == 1979:
        offset = 0
    elif start_year == 1989:
        offset = days_delta1
    elif start_year == 1999:
        offset = days_delta1 + days_delta2
    elif start_year == 2009:
        offset = days_delta1 + days_delta2 + days_delta3
    else:
        offset = days_delta1 + days_delta2 + days_delta3 + days_delta4

    print(f'start year = {start_year}, offset = {offset}', flush=True)
    flux = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_norm_scaled.npy')
    sst = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_norm_scaled.npy')
    press = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_norm_scaled.npy')
    count_abfe_coefficients(files_path_prefix,
                           mask,
                           sst,
                           press,
                           time_start=0,
                           time_end=sst.shape[1] - 1,
                           offset=offset,
                           pair_name='sst-press')

    count_abfe_coefficients(files_path_prefix,
                           mask,
                           flux,
                           sst,
                           time_start=0,
                           time_end=sst.shape[1] - 1,
                           offset=offset,
                           pair_name='flux-sst')

    count_abfe_coefficients(files_path_prefix,
                           mask,
                           flux,
                           press,
                           time_start=0,
                           time_end=flux.shape[1] - 1,
                           offset=offset,
                           pair_name='flux-press')