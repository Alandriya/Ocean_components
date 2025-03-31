import copy
import math
import os
from multiprocessing import Pool
from struct import unpack

import numpy as np
import tqdm
from data_processing import scale_to_bins
from numpy.linalg import norm
from scipy.linalg import sqrtm
from scipy.linalg.interpolative import estimate_spectral_norm
from scipy.stats import pearsonr
from skimage.measure import block_reduce

files_path_prefix = 'D://Data/OceanFull/'


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


def count_c_coeff(files_path_prefix: str,
                  a_timelist: list,
                  b_timelist: list,
                  start_idx: int = 1,
                  time_width: int = 14,
                  pair_name: str = '',
                  ):
    """
    Counts Pearson correlation between A and B coefficients on windows with length time_width on the range
    (0, len(a_timelist) - time_width) and saves them in files_path_prefix + Coeff_data dir.

    :param files_path_prefix: path to the working directory
    :param a_timelist: list with structure [a_sens, a_lat], where a_sens and a_lat are np.arrays with shape (161, 181)
        with values for A coefficient for sensible and latent fluxes, respectively
    :param b_timelist: list with b_matrix as elements, where b_matrix is np.array with shape (4, 161, 181)
        containing 4 matrices with elements of 2x2 matrix of coefficient B for every point of grid.
        0 is for B11 = sensible at t0 - sensible at t1,
        1 is for B12 = sensible at t0 - latent at t1,
        2 is for B21 = latent at t0 - sensible at t1,
        3 is for B22 = latent at t0 - latent at t1.
    :param time_width: time window width = width of vectors going into pearsonr function
    :param start_idx: from which number to save arrays
    :return:
    """
    a_sens_all = np.zeros((time_width, 161, 181), dtype=float)
    a_lat_all = np.zeros((time_width, 161, 181), dtype=float)
    b_all = np.zeros((time_width, 4, 161, 181), dtype=float)
    c_grid = np.zeros((2, 161, 181), dtype=float)

    print('Counting C')
    for t_start in tqdm.tqdm(range(0, len(a_timelist) - time_width)):
        # c_grid = np.zeros((4, 161, 181), dtype=float)
        window = range(t_start, t_start + time_width)
        # filling values
        for k in range(time_width):
            t = window[k]
            a_sens, a_lat = a_timelist[t]
            b_matrix = b_timelist[t]
            a_sens_all[k] = a_sens
            a_lat_all[k] = a_lat
            b_all[k] = b_matrix

        for i in range(161):
            for j in range(181):
                if np.isnan(a_sens_all[:, i, j]).any() or np.isnan(a_lat_all[:, i, j]).any() or \
                        np.isnan(b_all[:, 0, i, j]).any() or np.isnan(b_all[:, 3, i, j]).any():
                    c_grid[:, i, j] = np.nan
                else:
                    # 0 - (a_sens, a_lat), 1 - (b_sens, b_lat), 2 - (a_sens, b_sens), 3 - (a_lat, b_lat),
                    c_grid[0, i, j] = pearsonr(a_sens_all[:, i, j], a_lat_all[:, i, j])[0]
                    c_grid[1, i, j] = pearsonr(b_all[:, 0, i, j], b_all[:, 3, i, j])[0]
                    # c_grid[2, i, j] = pearsonr(a_sens_all[:, i, j], b_all[:, 0, i, j])[0]
                    # c_grid[3, i, j] = pearsonr(a_lat_all[:, i, j], b_all[:, 3, i, j])[0]
        np.save(files_path_prefix + f'Coeff_data_3d/{pair_name}/{int(t_start + start_idx)}_C.npy', c_grid)
    return


def _parallel_AB_func(arg):
    # Func for each process in parallel counting AB
    offset, borders, mask, sensible_array, latent_array = arg

    print('My process id:', os.getpid())
    start, end = borders
    for t in range(start, end):
        count_abfe_coefficients(files_path_prefix, mask, sensible_array, latent_array, start, end, offset)
    print(f'Process {os.getpid()} finished')
    return


def parallel_AB(cpu_count: int, filename_sensible: str, filename_latent: str, offset: int):
    """
    Launches and controlls parallel A, B and F counting

    :param cpu_count: amount of CPU to use
    :param filename_sensible: filename containing data of sensible flux
    :param filename_latent: filename containing data of latent flux
    :param offset: offset from (01.01.1979) in days for the first element to count
    :return:
    """
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)

    sensible_array = np.load(files_path_prefix + filename_sensible)
    latent_array = np.load(files_path_prefix + filename_latent)

    sensible_array = sensible_array.astype(float)
    latent_array = latent_array.astype(float)
    sensible_array[np.logical_not(mask), :] = np.nan
    latent_array[np.logical_not(mask)] = np.nan

    # mean by day = every 4 observations
    pack_len = 4
    sensible_array = block_reduce(sensible_array,
                                  block_size=(1, pack_len),
                                  func=np.mean, )
    latent_array = block_reduce(latent_array,
                                block_size=(1, pack_len),
                                func=np.mean, )

    start = 0
    end = sensible_array.shape[1]
    delta = (end - start + cpu_count // 2) // cpu_count

    sensible_array = scale_to_bins(sensible_array)
    latent_array = scale_to_bins(latent_array)

    borders = [[start + delta*i, start + delta*(i+1)] for i in range(cpu_count)]
    masks = [mask for b in borders]
    sensible_part = [sensible_array[:, b[0]:b[1]+1] for b in borders]
    latent_part = [latent_array[:, b[0]:b[1]+1] for b in borders]
    args = [[offset, borders[i], masks[i], sensible_part[i], latent_part[i]] for i in range(cpu_count)]

    with Pool(cpu_count) as p:
        p.map(_parallel_AB_func, args)
        p.close()
        p.join()
    return


def count_correlation_fluxes(files_path_prefix: str,
                             start: int = 0,
                             end: int = 0,
                             time_width: int = 14 * 4,
                             ):
    """
    Counts correlation in time_width days interval starting from start day, and until the right border of the window is
    less than end index.

    :param files_path_prefix: path to the working directory
    :param start: start index, and index of the 1st element of the first window position
    :param end: end index
    :param time_width: window width (in days)
    :return:
    """
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)

    sensible_array = np.load(files_path_prefix + f'5years_sensible.npy')
    latent_array = np.load(files_path_prefix + f'5years_latent.npy')

    sensible_array = sensible_array.astype(float)
    latent_array = latent_array.astype(float)
    sensible_array[np.logical_not(mask), :] = np.nan
    latent_array[np.logical_not(mask)] = np.nan

    # mean by day = every 4 observations
    pack_len = 4
    sensible_array = block_reduce(sensible_array,
                                  block_size=(1, pack_len),
                                  func=np.mean, )
    latent_array = block_reduce(latent_array,
                                block_size=(1, pack_len),
                                func=np.mean, )

    for t in tqdm.tqdm(range(start, end - time_width)):
        corr = np.zeros((161, 181), dtype=float)
        for i in range(161):
            for j in range(181):
                if not mask[i*181 + j]:
                    corr[i, j] = np.nan
                else:
                    sens_window = sensible_array[i * 181 + j, t:t + time_width]
                    lat_window = latent_array[i * 181 + j, t:t + time_width]
                    corr[i, j] = pearsonr(sens_window, lat_window)[0]

        np.save(files_path_prefix + f'Flux_correlations/FL_Corr_{t}', corr)
    return


def count_fraction(files_path_prefix: str,
                   a_timelist: list,
                   b_timelist: list,
                   mask: np.array,
                   mean_width: int = 7,
                   start_idx: int = 1):
    """
    Counts the F coefficient with the meaning of a fraction ||A|| / ||B|| in each point of (161,181) grid array,
    where A norm is standard Euclidean norm and B norm is an estimated norm of the B matrix

    :param files_path_prefix: path to the working directory
    :param a_timelist: list with structure [a_sens, a_lat], where a_sens and a_lat are np.arrays with shape (161, 181)
        with values for A coefficient for sensible and latent fluxes, respectively
    :param b_timelist: list with b_matrix as elements, where b_matrix is np.array with shape (4, 161, 181)
        containing 4 matrices with elements of 2x2 matrix of coefficient B for every point of grid.
        0 is for B11 = sensible at t0 - sensible at t1,
        1 is for B12 = sensible at t0 - latent at t1,
        2 is for B21 = latent at t0 - sensible at t1,
        3 is for B22 = latent at t0 - latent at t1.
    :param start: start index
    :param end: end index
    :param step: step (in days) in loop
    :return:
    """
    for t_start in tqdm.tqdm(range(0, len(a_timelist) - mean_width)):
        f = np.zeros((161, 181), dtype=float)
        a_sens = np.zeros((161, 181), dtype=float)
        a_lat = np.zeros((161, 181), dtype=float)
        b = np.zeros((4, 161, 181), dtype=float)

        for t in range(mean_width):
            a_sens += a_timelist[t_start + t][0]
            a_lat += a_timelist[t_start + t][1]
            b += b_timelist[t_start + t]

        a_sens /= mean_width
        a_lat /= mean_width
        b /= mean_width

        f[:, :] = np.nan
        for i in range(161):
            for j in range(181):
                if mask[i*181 + j]:
                    a_vec = [a_sens[i, j], a_lat[i, j]]
                    b_part = b[:, i, j].reshape(2, 2)

                    # f[i, j] = norm(a_vec, 2) / estimate_spectral_norm(b_part)
                    if b_part[0,0]**2 + b_part[1,1]**2 + b_part[0,1] + b_part[1,0] >= 0:
                        f[i, j] = (abs(a_vec[0] + a_vec[1]))/math.sqrt(b_part[0,0]**2 + b_part[1,1]**2 + b_part[0,1] + b_part[1,0])
                    else:
                        # print(f'point ({i}, {j}) time {t_start} value {b_part[0,0]**2 + b_part[1,1]**2 + b_part[0,1] + b_part[1,0]: .10f}')
                        f[i, j] = np.nan
        # np.save(files_path_prefix + f'Coeff_data/{t}_F.npy', f)
        # np.save(files_path_prefix + f'Coeff_data/{t_start + start_idx}_F_new.npy', f)
    return


def count_f_separate_coeff(files_path_prefix: str,
                           a_timelist: list,
                           b_timelist: list,
                           start_idx: int = 1,
                           mean_width: int = 7,
                           pair_name: str = '',
                           ):
    """
    Counts FS - fractions mean(abs(a_sens)) / mean(abs(B[0]))  and mean(abs(a_lat) / mean(abs(B[1]))
    coefficients' where mean is taken in [t, t + mean_width] days window for every point in grid.
    :param files_path_prefix: path to the working directory
    :param a_timelist: list with structure [a_sens, a_lat], where a_sens and a_lat are np.arrays with shape (161, 181)
        with values for A coefficient for sensible and latent fluxes, respectively
    :param b_timelist: list with b_matrix as elements, where b_matrix is np.array with shape (4, 161, 181)
        containing 4 matrices with elements of 2x2 matrix of coefficient B for every point of grid.
        0 is for B11 = sensible at t0 - sensible at t1,
        1 is for B12 = sensible at t0 - latent at t1,
        2 is for B21 = latent at t0 - sensible at t1,
        3 is for B22 = latent at t0 - latent at t1.
    :param start_idx: from which number to save arrays
    :param mean_width: width in days of window to count mean
    :return:
    """
    print('Counting F separate')
    for t_start in tqdm.tqdm(range(0, len(a_timelist) - mean_width)):
        f_grid = np.zeros((2, 161, 181), dtype=float)
        a_sens = np.zeros((161, 181), dtype=float)
        a_lat = np.zeros((161, 181), dtype=float)
        b_sens = np.zeros((161, 181), dtype=float)
        b_lat = np.zeros((161, 181), dtype=float)

        for t in range(mean_width):
            a_sens += a_timelist[t_start + t][0]
            a_lat += a_timelist[t_start + t][1]
            b_sens += b_timelist[t_start + t][0]
            b_lat += b_timelist[t_start + t][3]

        a_sens /= mean_width
        a_lat /= mean_width
        b_lat /= mean_width
        b_sens /= mean_width

        for i in range(161):
            for j in range(181):
                if np.isnan(a_sens[i, j]):
                    f_grid[:, i, j] = np.nan
                else:
                    f_grid[0, i, j] = abs(a_sens[i, j]) / abs(b_sens[i, j])
                    f_grid[1, i, j] = abs(a_lat[i, j]) / abs(b_lat[i, j])

        np.save(files_path_prefix + f'Coeff_data_3d/{pair_name}/{start_idx + t_start}_FS.npy', f_grid)
    return


def count_triplet(set_t0, variable1, variable2, a_matrix, e_matrix, var1_idx, pair_idx):
    for val_t0 in set_t0:
        if not np.isnan(val_t0):
            points_var1 = np.where(variable1 == val_t0)[0]
            amount_t0 = len(points_var1)

            # var1 t0 - var1 t1
            set_t1 = np.unique(variable1[points_var1])
            probabilities = list()
            for val_t1 in set_t1:
                prob = len(np.where(variable1[points_var1] == val_t1)[0]) * 1.0 / amount_t0
                probabilities.append(prob)

            a = sum([(list(set_t1)[i] - val_t0) * probabilities[i] for i in range(len(probabilities))])
            b_squared = sum(
                [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

            for idx in points_var1:
                a_matrix[var1_idx][idx // 181][idx % 181] = a
                e_matrix[var1_idx][idx // 181][idx % 181] = b_squared

            # var1 t0 - var2 t1
            set_t1 = np.unique(variable2[points_var1])
            probabilities = list()
            for val_t1 in set_t1:
                prob = len(np.where(variable2[points_var1 == val_t1])[0]) * 1.0 / amount_t0
                probabilities.append(prob)
            b_squared = sum(
                [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

            for idx in points_var1:
                e_matrix[pair_idx][idx // 181][idx % 181] = b_squared

    return  a_matrix, e_matrix


def count_AE(files_path_prefix: str,
                      mask: np.ndarray,
                      flux_array: np.ndarray,
                      SST_array: np.ndarray,
                      press_array: np.ndarray,
                      time_start: int = 0,
                      time_end: int = 0,
                      offset: int = 0,
                      ):
    a_flux = np.zeros((161, 181), dtype=float)
    a_sst = np.zeros((161, 181), dtype=float)
    a_press = np.zeros((161, 181), dtype=float)
    a_fake = np.zeros((161, 181), dtype=float)
    b_squared_matrix = np.zeros((9, 161, 181), dtype=complex)

    # set nan where is not ocean in arrays
    a_flux[np.logical_not(mask)] = np.nan
    a_sst[np.logical_not(mask)] = np.nan
    a_press[np.logical_not(mask)] = np.nan
    b_squared_matrix[:, :][np.logical_not(mask)] = np.nan

    print('Counting triplets A, B squared')
    for t_absolute in tqdm.tqdm(range(time_start + 1, time_end + 1)):
        t_rel = t_absolute - time_start
        set_flux = np.unique(flux_array[:, t_rel - 1])
        set_sst = np.unique(SST_array[:, t_rel - 1])
        set_press = np.unique(press_array[:, t_rel - 1])

        # E[0] = flux-flux, E[1] = SST-SST, E[2] = press-press,
        # E[3] = flux-sst, E[4] = flux-press, E[5] = sst-flux, E[6] = sst-press, E[7] = press-flux, E[8] = press-sst

        a_flux, b_squared_matrix = count_triplet(set_flux, flux_array[:, t_rel], SST_array[:, t_rel], a_flux, b_squared_matrix, 0, 3)
        _, b_squared_matrix = count_triplet(set_flux, flux_array[:, t_rel], press_array[:, t_rel], a_flux, b_squared_matrix, 0, 4)

        a_sst, b_squared_matrix = count_triplet(set_sst, SST_array[:, t_rel], flux_array[:, t_rel], a_sst, b_squared_matrix, 1, 5)
        _, b_squared_matrix = count_triplet(set_sst, SST_array[:, t_rel], press_array[:, t_rel], a_sst, b_squared_matrix, 1, 6)

        a_press, b_squared_matrix = count_triplet(set_press, press_array[:, t_rel], flux_array[:, t_rel], a_press, b_squared_matrix, 2, 7)
        _, b_squared_matrix = count_triplet(set_press, press_array[:, t_rel], SST_array[:, t_rel], a_press, b_squared_matrix, 2, 8)

        np.save(files_path_prefix + f'Coeff_data_3d/{int(t_absolute + offset)}_A_flux.npy', a_flux)
        np.save(files_path_prefix + f'Coeff_data_3d/{int(t_absolute + offset)}_A_sst.npy', a_sst)
        np.save(files_path_prefix + f'Coeff_data_3d/{int(t_absolute + offset)}_A_press.npy', a_press)
        np.save(files_path_prefix + f'Coeff_data_3d/{int(t_absolute + offset)}_B_squared.npy', b_squared_matrix)
    return

