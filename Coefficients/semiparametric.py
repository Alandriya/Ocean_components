import os.path

import numpy as np
from sklearn.mixture import GaussianMixture
from Coefficients.EM_hybrid import process_points
from Data_processing.data_processing import scale_to_bins
from Plotting.plot_compare import *
import os
from multiprocessing import Pool
from struct import unpack
import tqdm

width = 181
height = 161

def collect_point(files_path_prefix: str,
                      time_start: int,
                      time_end: int,
                      point: tuple,
                      path: str = 'Synthetic/',
                      method: str = 'Bel',
                      ):
    """
    Collects ar array for a specific point for Bel method from existing maps. Requires maps estimates at
    files_path_prefix + path + 'maps'.
    :param files_path_prefix: path to the working directory
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param point: tuple with point coordinates
    :param path: additional path to the folder from files_path_prefix, like 'Synthetic/', 'Components/sensible/',
    'Components/latent/'
    :return:
    """
    if not os.path.exists(files_path_prefix + path + f'{method}/points'):
        os.mkdir(files_path_prefix + path + f'{method}/points')


    a_Bel = np.zeros(time_end - time_start)
    for t in range(time_start, time_end):
        a_arr = np.load(files_path_prefix + path + f'{method}/daily/A_{t}.npy')
        a_Bel[t - time_start] = a_arr[point[0], point[1]]
        del a_arr

    np.save(files_path_prefix + path + f'{method}/points/point_({point[0]}, {point[1]})-A.npy', a_Bel)

    b_Bel = np.zeros(time_end - time_start)
    for t in range(time_start, time_end):
        b_arr = np.load(files_path_prefix + path + f'{method}/daily/B_{t}.npy')
        b_Bel[t - time_start] = b_arr[point[0], point[1]]
        del b_arr

    np.save(files_path_prefix + path + f'{method}/points/point_({point[0]}, {point[1]})-B.npy', b_Bel)
    return


def count_semi_1d_AB(files_path_prefix: str,
                     data_array: np.ndarray,
                     time_start: int,
                     time_end: int,
                     path: str = 'Synthetic/',
                     quantiles_amount: int = 50,
                     n_components: int = 2,
                     start_index: int = 0,
                     ):
    """
    Counts and saves to files_path_prefix + path + 'Semi_1d/daily' A and B estimates X array for each day
    t+start index for t in (time_start, time_end)
    :param files_path_prefix: path to the working directory
    :param data_array: np.array with shape [time_steps, height, width]
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param path: additional path to the folder from files_path_prefix, like 'Synthetic/', 'Components/sensible/',
    'Components/latent/'
    :param quantiles_amount: how many quantiles to use (for one step)
    :param n_components: amount of components for EM
    :param start_index: offset index when saving maps
    :return:
    """
    if not os.path.exists(files_path_prefix + path):
        os.mkdir(files_path_prefix + path)

    if not os.path.exists(files_path_prefix + path + f'Semi_1d'):
        os.mkdir(files_path_prefix + path + f'Semi_1d')

    if not os.path.exists(files_path_prefix + path + f'Semi_1d/daily'):
        os.mkdir(files_path_prefix + path + f'Semi_1d/daily')

    a_map = np.zeros((data_array.shape[1], data_array.shape[2]), dtype=float)
    b_map = np.zeros((data_array.shape[1], data_array.shape[2]), dtype=float)
    a_map[np.isnan(data_array[0])] = np.nan
    b_map[np.isnan(data_array[0])] = np.nan
    # start_time = time.time()
    for t in tqdm.tqdm(range(time_start + 1, time_end)):
    # for t in range(time_start + 1, time_end):
        # print(f't = {t}')
        if os.path.exists(files_path_prefix + path + f'Semi_1d/daily/A_{t+start_index}.npy'):
            continue
    
        data_array, quantiles = scale_to_bins(data_array[t - 1], quantiles_amount)
        data_set = np.unique(data_array[np.logical_not(np.isnan(data_array))].flat)
        data_set = data_set[np.logical_not(np.isnan(data_set))]
        for group in range(len(data_set)):
            value_t0 = data_set[group]
            # if np.isnan(value_t0):
            #     continue
            day_sample = (data_array[t][np.where(data_array == value_t0)] -
                          data_array[t - 1][np.where(data_array == value_t0)]).flatten()

            window = day_sample
            # gm = GaussianMixture(n_components=n_components,
            #                      tol=1e-6,
            #                      covariance_type='spherical',
            #                      max_iter=10000,
            #                      init_params='random',
            #                      n_init=30
            #                      ).fit(window.reshape(-1, 1))

            gm = GaussianMixture(n_components=n_components,
                                 tol=1e-4,
                                 covariance_type='spherical',
                                 max_iter=1000,
                                 init_params='random',
                                 n_init=10
                                 ).fit(window.reshape(-1, 1))
            means = gm.means_.flatten()
            sigmas_squared = gm.covariances_.flatten()
            weights = gm.weights_.flatten()
            weights /= sum(weights)

            a_sum = sum(means * weights)
            b_sum = math.sqrt(sum(weights * (means ** 2 + sigmas_squared)))

            a_map[np.where(data_array == value_t0)] = a_sum
            b_map[np.where(data_array == value_t0)] = b_sum
            # if t == 1:
            #     print(a_sum)

        # print('\n\n', flush=True)
        np.save(files_path_prefix + path + f'Semi_1d/daily/A_{t+start_index}.npy', a_map)
        np.save(files_path_prefix + path + f'Semi_1d/daily/B_{t+start_index}.npy', b_map)
        # print(f'Iteration {t}: {(time.time() - start_time):.1f} seconds')
        # start_time = time.time()
    return


def count_semi_2d_AB(files_path_prefix: str,
                     data_array: np.ndarray,
                     time_start: int,
                     time_end: int,
                     mask: np.array,
                     path: str = 'Synthetic/',
                     quantiles_amount: int = 10,
                     n_components: int = 2,
                     start_index: int = 0,
                     ):
    """
    Counts and saves to files_path_prefix + path + 'Semi_1d/daily' A and B estimates X array for each day
    t+start index for t in (time_start, time_end)
    :param files_path_prefix: path to the working directory
    :param data_array: np.array with shape [time_steps, height, width, 2] where 2 is for 2 components of X vector
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param path: additional path to the folder from files_path_prefix, like 'Synthetic/', 'Components/sensible/',
    'Components/latent/'
    :param quantiles_amount: how many quantiles to use (for one step)
    :param n_components: amount of components for EM
    :param start_index: offset index when saving maps
    :return:
    """
    if not os.path.exists(files_path_prefix + path):
        os.mkdir(files_path_prefix + path)

    if not os.path.exists(files_path_prefix + path + f'Semi_2d'):
        os.mkdir(files_path_prefix + path + f'Semi_2d')

    if not os.path.exists(files_path_prefix + path + f'Semi_2d/daily'):
        os.mkdir(files_path_prefix + path + f'Semi_2d/daily')

    a_map = np.zeros((data_array.shape[1], data_array.shape[2], 2), dtype=float)
    b_map = np.zeros((data_array.shape[1], data_array.shape[2], 4), dtype=float)
    a_map[np.logical_not(mask), :] = np.nan
    b_map[np.logical_not(mask), :] = np.nan
    data_array = data_array.reshape((data_array.shape[0], -1, 2))
    a_map = a_map.reshape((-1, 2))
    b_map = b_map.reshape((-1, 4))
    # start_time = time.time()
    data1 = data_array[:, :, 0]
    data2 = data_array[:, :, 1]
    for t in tqdm.tqdm(range(time_start + 1, time_end)):
        # for t in range(time_start + 1, time_end):
        # print(f't = {t}')
        if os.path.exists(files_path_prefix + path + f'Semi_1d/daily/A_{t + start_index}.npy'):
            continue

        data1_q, quantiles = scale_to_bins(data1[t - 1], quantiles_amount)
        data1_q = data1_q.flat
        data1_set = np.unique(data1_q)
        data1_set = data1_set[np.logical_not(np.isnan(data1_set))]

        data2_q, quantiles = scale_to_bins(data2[t - 1], quantiles_amount)
        data2_q = data2_q.flat
        data2_set = np.unique(data2_q)
        data2_set = data2_set[np.logical_not(np.isnan(data2_set))]

        for group1 in range(len(data1_set)):
            value1_t0 = data1_set[group1]
            for group2 in range(len(data2_set)):
                value2_t0 = data2_set[group2]
                place = np.intersect1d(np.where(data1_q == value1_t0), np.where(data2_q == value2_t0))
                window = (data_array[t, place, :] - data_array[t - 1, place, :])
                a_sum = np.zeros(2)
                b_sum = np.zeros(4)

                if window.shape[0] == 0:
                    continue

                if window.shape[0] == 1:
                    # print('Oops')
                    a_sum[0] = np.mean(window[:, 0])
                    a_sum[1] = np.mean(window[:, 1])
                    a_map[place] = a_sum
                    continue

                # gm = GaussianMixture(n_components=n_components,
                #                      tol=1e-6,
                #                      covariance_type='spherical',
                #                      max_iter=10000,
                #                      init_params='random',
                #                      n_init=30
                #                      ).fit(window.reshape(-1, 1))

                gm = GaussianMixture(n_components=n_components,
                                     tol=1e-4,
                                     covariance_type='spherical',
                                     max_iter=1000,
                                     init_params='random',
                                     n_init=10
                                     ).fit(window.reshape(-1, 2))
                means = gm.means_.flatten()
                sigmas_squared = gm.covariances_.flatten()
                weights = gm.weights_.flatten()
                weights /= sum(weights)

                a_sum[0] = sum([means[q * n_components] * weights[q] for q in range(n_components)])
                a_sum[1] = sum([means[1 + q * n_components] * weights[q] for q in range(n_components)])

                b_sum[0] = np.sqrt(
                    sum([(means[q * n_components]**2 + sigmas_squared[0]) * weights[q] for q in range(n_components)]))
                b_sum[1] = np.sqrt(
                    sum([(means[q * n_components]**2 + sigmas_squared[1]) * weights[q] for q in range(n_components)]))
                b_sum[2] = np.sqrt(
                    sum([(means[1 + q * n_components]**2 + sigmas_squared[0]) * weights[q] for q in range(n_components)]))
                b_sum[3] = np.sqrt(
                    sum([(means[1 + q * n_components]**2 + sigmas_squared[1]) * weights[q] for q in range(n_components)]))

                a_map[place] = a_sum
                b_map[place] = b_sum
                # if t == 1:
                #     print(a_sum)

        # print('\n\n', flush=True)
        np.save(files_path_prefix + path + f'Semi_2d/daily/A_{t + start_index}.npy', a_map)
        np.save(files_path_prefix + path + f'Semi_2d/daily/B_{t + start_index}.npy', b_map)
        # print(f'Iteration {t}: {(time.time() - start_time):.1f} seconds')
        # start_time = time.time()
    return

def parallel_semiparam(files_path_prefix: str,
                       data_array: np.ndarray,
                       cpu_count: int,
                       points_borders: list,
                       time_start: int,
                       time_end: int,
                       timedelta: int,
                       flux_type: str,
                       coeff_type: str,
                       window_width: int,
                       step_ticks: int,
                       ticks_by_day: int,
                       radius: int,
                       n_components: int,
                       draw: bool = False):
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)

    # select points to list
    all_points = list()
    for p in range(points_borders[0], points_borders[1]):
        if mask[p]:
            all_points.append(p)

    process_amount = len(all_points) // cpu_count + len(all_points) % cpu_count

    print(f'Processing {len(all_points)} points')
    # create args list
    all_args = list()
    for p in range(cpu_count):
        points = all_points[p * process_amount:(p + 1) * process_amount]
        points_info = list()
        samples = list()
        for point_idx in points:
            point_size = (radius * 2 + 1) ** 2
            point = (point_idx // width, point_idx % width)
            point_bigger = list()
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if 0 <= point[0] + i < height and 0 <= point[1] + j < width and \
                            mask[(point[0] + i) * width + point[1] + j]:
                        point_bigger.append((point[0] + i, point[1] + j))
                    else:
                        point_size -= 1

            sample = np.zeros((point_size, (time_end - time_start) * ticks_by_day - 1))
            for i in range(point_size):
                p = point_bigger[i]
                sample[i, :] = np.diff(
                    data_array[p[0] * width + p[1], time_start * ticks_by_day:time_end * ticks_by_day])

            # reshape
            sample = sample.transpose().flatten()

            points_info.append([point, point_bigger, point_size])
            samples.append(sample)

        time_info = [time_start, time_end, timedelta]
        args = [files_path_prefix, time_info, points_info, samples, flux_type, coeff_type, window_width, step_ticks,
                ticks_by_day, radius, n_components, draw]
        all_args.append(args)

    if cpu_count == 1:
        print("I'm the only process here")
        process_points(*all_args[0])
    else:
        print(f"I'm the parent with id = {os.getpid()}")
        with Pool(cpu_count) as p:
            p.starmap(process_points, all_args)
            p.close()
            p.join()
    return




