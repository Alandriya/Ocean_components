from skimage.measure import block_reduce
from scipy.stats import pearsonr
import numpy as np
from math import sqrt
import tqdm
from struct import unpack
import multiprocessing
from multiprocessing import Pool
import os


files_path_prefix = 'D://Data/OceanFull/'


def scale_to_bins(arr):
    # set each value with the number of quantile from 0 to 100 in which it belongs
    quantiles = np.nanquantile(arr, np.linspace(0, 1, 100, endpoint=False))
    arr_digit = np.digitize(arr, quantiles)

    # return nan back :)
    arr_digit = arr_digit.astype(float)
    arr_digit[np.isnan(arr)] = np.nan

    return arr_digit


def count_A_B_coefficients(files_path_prefix, mask, sensible_array, latent_array, time_start=0, time_end=0):
    """
    Counts A and B coefficients, saves them to and adds them to a_timelist and b_timelist.

    a_timelist: list with structure [a_sens, a_lat], where a_sens and a_lat are np.arrays with shape (161, 181)
    with values for A coefficient for sensible and latent fluxes, respectively

    b_timelist: list with b_matrix as elements, where b_matrix is np.array with shape (4, 161, 181)
    containing 4 matrices with elements of 2x2 matrix of coefficient B for every point of grid.
    0 is for B11 = sensible at t0 - sensible at t1,
    1 is for B12 = sensible at t0 - latent at t1,
    2 is for B21 = latent at t0 - sensible at t1,
    3 is for B22 = latent at t0 - latent at t1.
    :param files_path_prefix:
    :param mask:
    :param time_start: first time step
    :param time_end: last time step
    :return:
    """

    # print('Counting A and B')
    # for t in tqdm.tqdm(range(time_start + 1, time_end + 1)):
    for t_absolute in range(time_start + 1, time_end + 1):
        if not os.path.exists(files_path_prefix + f'AB_coeff_data/{t_absolute}_A_sensible.npy'):
            t_rel = t_absolute - time_start
            a_sens = np.zeros((161, 181))
            a_lat = np.zeros((161, 181))
            b_matrix = np.zeros((4, 161, 181))

            # set nan where is not ocean in arrays
            for i in range(0, len(mask)):
                if not mask[i]:
                    a_sens[i // 181][i % 181] = np.nan
                    a_lat[i // 181][i % 181] = np.nan
                    b_matrix[:, i // 181, i % 181] = np.nan

            set_sens = set(sensible_array[:, t_rel - 1])
            set_lat = set(latent_array[:, t_rel - 1])

            for val_t0 in set_sens:
                if not np.isnan(val_t0):
                    points_sensible = np.nonzero(sensible_array[:, t_rel - 1] == val_t0)[0]
                    amount_t0 = len(points_sensible)

                    # sensible t0 - sensible t1
                    set_t1 = set(sensible_array[points_sensible][:, t_rel])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = sum(np.where(sensible_array[points_sensible][:, t_rel] == val_t1, 1, 0)) * 1.0 / amount_t0
                        probabilities.append(prob)

                    a = sum([(list(set_t1)[i] - val_t0) * probabilities[i] for i in range(len(probabilities))])
                    b_squared = sum(
                        [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2
                    b = sqrt(b_squared) if b_squared > 0 else None
                    for idx in points_sensible:
                        a_sens[idx // 181][idx % 181] = a
                        b_matrix[0][idx // 181][idx % 181] = b

                    # sensible t0 - latent t1
                    set_t1 = set(latent_array[points_sensible][:, t_rel])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = sum(np.where(latent_array[points_sensible][:, t_rel] == val_t1, 1, 0)) * 1.0 / amount_t0
                        probabilities.append(prob)

                    b_squared = sum(
                        [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2
                    b = sqrt(b_squared) if b_squared > 0 else None

                    for idx in points_sensible:
                        b_matrix[1][idx // 181][idx % 181] = b

            for val_t0 in set_lat:
                if not np.isnan(val_t0):
                    points_latent = np.nonzero(latent_array[:, t_rel - 1] == val_t0)[0]
                    amount_t0 = len(points_latent)

                    # latent - latent
                    set_t1 = set(latent_array[points_latent][:, t_rel])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = sum(np.where(latent_array[points_latent][:, t_rel] == val_t1, 1, 0)) * 1.0 / amount_t0
                        probabilities.append(prob)

                    a = sum([(list(set_t1)[i] - val_t0) * probabilities[i] for i in range(len(probabilities))])
                    b_squared = sum(
                        [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2
                    b = sqrt(b_squared) if b_squared > 0 else None
                    for idx in points_latent:
                        a_lat[idx // 181][idx % 181] = a
                        b_matrix[3][idx // 181][idx % 181] = b

                    # latent t0 - sensible t1
                    set_t1 = set(sensible_array[points_latent][:, t_rel])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = sum(np.where(sensible_array[points_latent][:, t_rel] == val_t1, 1, 0)) * 1.0 / amount_t0
                        probabilities.append(prob)

                    b_squared = sum(
                        [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2
                    b = sqrt(b_squared) if b_squared > 0 else None
                    for idx in points_latent:
                        b_matrix[2][idx // 181][idx % 181] = b

            # save data
            np.save(files_path_prefix + f'AB_coeff_data/{t_absolute}_A_sensible.npy', a_sens)
            np.save(files_path_prefix + f'AB_coeff_data/{t_absolute}_A_latent.npy', a_lat)
            np.save(files_path_prefix + f'AB_coeff_data/{t_absolute}_B.npy', b_matrix)
    return


def count_correlations(a_timelist, b_timelist, time_width=14):
    """
    Counts correlation between A and B coefficients on the range (0, len(a_timelist) - time_width) and collects them
    into list c_timelist. Elements of the list are np.arrays with shape (2, 161, 181) containing 2 matrices of
    correlation of A and B coefficients:
    0 is for (a_sens, B11) correlation,
    1 is for (a_lat, B22) correlation
    :param a_timelist: list with structure [a_sens, a_lat], where a_sens and a_lat are np.arrays with shape (161, 181)
    with values for A coefficient for sensible and latent fluxes, respectively
    :param b_timelist: list with b_matrix as elements, where b_matrix is np.array with shape (4, 161, 181)
    containing 4 matrices with elements of 2x2 matrix of coefficient B for every point of grid.
    0 is for B11 = sensible at t0 - sensible at t1,
    1 is for B12 = sensible at t0 - latent at t1,
    2 is for B21 = latent at t0 - sensible at t1,
    3 is for B22 = latent at t0 - latent at t1.
    :param time_width: time window width = width of vectors going into pearsonr function
    :return:
    """
    c_timelist = list()
    print('Counting C')
    for t_start in tqdm.tqdm(range(0, len(a_timelist) - time_width)):
        c_grid = np.zeros((2, 161, 181), dtype=float)
        window = range(t_start, t_start + time_width)
        a_sens_all = np.zeros((time_width, 161, 181), dtype=float)
        a_lat_all = np.zeros((time_width, 161, 181), dtype=float)
        b_all = np.zeros((time_width, 4, 161, 181), dtype=float)
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
                    # 1 - (a_sens, b_sens), 2 - (a_lat, b_lat)
                    c_grid[0, i, j] = pearsonr(a_sens_all[:, i, j], b_all[:, 0, i, j])[0]
                    c_grid[1, i, j] = pearsonr(a_lat_all[:, i, j], b_all[:, 3, i, j])[0]

        c_timelist.append(c_grid)
    return c_timelist


def _parallel_AB_func(arg):
    borders, mask, sensible_array, latent_array = arg

    print('My process id:', os.getpid())
    start, end = borders
    for t in range(start, end):
        count_A_B_coefficients(files_path_prefix, mask, sensible_array, latent_array, start, end)
    print(f'Process {os.getpid()} finished')
    return


def parallel_AB(cpu_count):
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

    start = 0
    end = sensible_array.shape[1]
    delta = (end - start + cpu_count // 2) // cpu_count

    sensible_array = scale_to_bins(sensible_array)
    latent_array = scale_to_bins(latent_array)

    borders = [[start + delta*i, start + delta*(i+1)] for i in range(cpu_count)]
    masks = [mask for b in borders]
    sensible_part = [sensible_array[:, b[0]:b[1]+1] for b in borders]
    latent_part = [latent_array[:, b[0]:b[1]+1] for b in borders]
    args = [[borders[i], masks[i], sensible_part[i], latent_part[i]] for i in range(cpu_count)]

    with Pool(cpu_count) as p:
        p.map(_parallel_AB_func, args)
        p.close()
        p.join()
    return
