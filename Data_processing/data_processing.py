import datetime
import math
import os
import shutil
from copy import deepcopy
from struct import unpack

import numpy as np
import pandas as pd
import tqdm
from skimage.measure import block_reduce

# files_path_prefix = 'D://Data/OceanFull/'
width = 181
height = 161


def sort_by_means(files_path_prefix, flux_type):
    """
    Loads and sorts Dataframes with EM estimations
    :param files_path_prefix: path to the working directory
    :param flux_type: string of the flux type: 'sensible' or 'latent'
    :return:
    """
    filename = os.listdir(files_path_prefix + '5_years_weekly/')[0]
    data = pd.read_csv(files_path_prefix + '5_years_weekly/' + filename, delimiter=';')
    means_cols = data.filter(regex='mean_', axis=1).columns
    sigmas_cols = data.filter(regex='sigma_', axis=1).columns
    weights_cols = data.filter(regex='weight_', axis=1).columns

    for filename in tqdm.tqdm(os.listdir(files_path_prefix + '5_years_weekly/')):
        if flux_type in filename:
            df = pd.read_csv(files_path_prefix + '5_years_weekly/' + filename, delimiter=';')

            # sort all columns by means
            means = df[means_cols].values
            sigmas = df[sigmas_cols].values
            weights = df[weights_cols].values

            df.columns = list(means_cols) + list(sigmas_cols) + list(weights_cols) + ['ts']
            for i in range(len(df)):
                zipped = list(zip(means[i], sigmas[i], weights[i]))
                zipped.sort(key=lambda x: x[0])
                # the scary expression below is for flattening the sorted zip results
                df.iloc[i] = list(sum(list(zip(*zipped)), ())) + [df.loc[i, 'ts']]

            df.to_csv(files_path_prefix + '5_years_weekly/' + filename, sep=';', index=False)
    return


def binary_to_array(files_path_prefix, input_filename, output_filename, date_start, date_end):
    days_delta = (date_start - datetime.datetime(1979, 1, 1)).days
    # length_1 = 62396 - days_delta * 4
    length = (date_end - date_start).days * 4

    arr_10years = np.empty((length, 29141), dtype=float)
    file = open(files_path_prefix + input_filename, "rb")
    for i in tqdm.tqdm(range(length)):
        # offset_1 = 32 + (62396 - length) * 116564 + 116564 * i
        offset = 32 + (days_delta * 4) * 116564 + 116564 * i

        file.seek(offset, 0)
        binary_values = file.read(116564)  # reading one timepoint
        point = unpack('f' * 29141, binary_values)
        arr_10years[i] = point
    file.close()
    np.save(files_path_prefix + output_filename + '.npy', arr_10years.transpose())
    del arr_10years
    return


def EM_dataframes_to_grids(files_path_prefix, flux_type, mask, components_amount, timesteps):
    dataframes = list()
    indexes = list()
    print('Loading DataFrames\n')
    for filename in tqdm.tqdm(os.listdir(files_path_prefix + '5_years_weekly/')):
        if flux_type in filename:
            df = pd.read_csv(files_path_prefix + '5_years_weekly/' + filename, delimiter=';')
            dataframes.append(df)
            idx = int(filename[len(flux_type) + 1: -4])
            indexes.append(idx)

    missing_df = list()
    # print('Creating grids\n')
    # fill and save grids
    for t in tqdm.tqdm(range(timesteps)):
        if not os.path.exists(files_path_prefix + f'/tmp_arrays/{flux_type}/means_{t}.npy'):
            grid = np.full((components_amount, 161, 181), np.nan)
            means_grid = deepcopy(grid)
            sigmas_grid = deepcopy(grid)
            weights_grid = deepcopy(grid)

            for i in range(len(mask)):
                if mask[i] and i in indexes:
                    rel_i = indexes.index(i)
                    df = dataframes[rel_i]
                    for comp in range(components_amount):
                        means_grid[comp][i // 181][i % 181] = df.loc[t, f'mean_{comp + 1}']
                        sigmas_grid[comp][i // 181][i % 181] = df.loc[t, f'sigma_{comp + 1}']
                        weights_grid[comp][i // 181][i % 181] = df.loc[t, f'weight_{comp + 1}']

                elif mask[i]:
                    missing_df.append(i)

            np.save(files_path_prefix + f'tmp_arrays/{flux_type}/means_{t}.npy', means_grid)
            np.save(files_path_prefix + f'tmp_arrays/{flux_type}/sigmas_{t}.npy', sigmas_grid)
            np.save(files_path_prefix + f'tmp_arrays/{flux_type}/weights_{t}.npy', weights_grid)

    # print(f'Missing dataframes {flux_type}: ', missing_df)
    return dataframes, indexes


def load_ABCFE(files_path_prefix: str,
              time_start: int,
              time_end: int,
              load_a: bool = False,
              load_b: bool = False,
              load_c: bool = False,
              load_f: bool = False,
              load_fs: bool = False,
              load_e: bool = False,
              verbose: bool = False,
              path_local: str = 'Coeff_data'):
    """
    Loads data from files_path_prefix + path_local directory and counts borders
    :param files_path_prefix: path to the working directory
    :param time_start: first time step
    :param time_end: last time step
    :param load_a: if to load A data or not
    :param load_b:
    :param load_c:
    :param load_f:
    :param load_fs:
    :param verbose: if to print logs
    :param path_local: relative path from files_path_prefix to directory with coefficient's dirs: A, B, C, etc.
    :return:
    """
    a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, e_timelist = list(), list(), list(), list(), list(), list()

    a1_max, a2_max = 0, 0
    a1_min, a2_min = 0, 0

    b_max = [0, 0, 0, 0]
    b_min = [0, 0, 0, 0]
    f_min = 10
    f_max = -1
    e_max = [0, 0, 0, 0]
    e_min = [0, 0, 0, 0]

    maskfile = open(files_path_prefix + "DATA/mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)

    # sst_coeff = 37.95727539 / (0.8980015822683038 + 0.8980015822683038)
    # flux_coeff = 2558.356628 / (0.8544676970659135 + 0.8544676970659135)
    # press_coeff = 17950.53906 / (0.8447768941044158 + 0.8447768941044158)

    if verbose:
        print('Loading ABC data')
    for t in range(time_start, time_end):
        if load_a:
            try:
                # a_sens = np.load(files_path_prefix + f'{path_local}/{t}_A_sens.npy')
                # a_lat = np.load(files_path_prefix + f'{path_local}/{t}_A_lat.npy')
                a = np.load(files_path_prefix + f'{path_local}/A_{t}.npy')
                a = a.reshape((height, width, 2))
                a_sens = a[:, :, 0]
                a_lat = a[:, :, 1]
                a_timelist.append([a_sens, a_lat])

                a1_max = max(a1_max, np.nanmax(a_sens))
                a1_min = min(a1_min, np.nanmin(a_sens))
                a2_max = max(a2_max, np.nanmax(a_lat))
                a2_min = min(a2_min, np.nanmin(a_lat))
            except FileNotFoundError:
                pass
            # a_max = max(a_max, np.nanmax(a_sens), np.nanmax(a_lat))
            # a_min = min(a_min, np.nanmin(a_sens), np.nanmin(a_lat))

        if load_b:
            try:
                b_matrix = np.load(files_path_prefix + f'{path_local}/B_{t}.npy')
                b_matrix = b_matrix.transpose()
                b_matrix = b_matrix.reshape((4, height, width))
                for i in range(4):
                    np.nan_to_num(b_matrix[i], False, -10)
                    b_matrix[i][np.logical_not(mask.reshape((height, width)))] = np.nan

                    b_max[i] = max(b_max[i], np.nanmax(b_matrix[i]))
                    b_min[i] = min(b_min[i], np.nanmin(b_matrix[i]))
                b_timelist.append(b_matrix)
            except FileNotFoundError:
                pass
        if load_f:
            f = np.load(files_path_prefix + f'{path_local}/{t}_F.npy')
            f_timelist.append(f)
            if np.isfinite(np.nanmax(f)):
                f_max = max(f_max, np.nanmax(f))
            f_min = min(f_min, np.nanmin(f))
        if load_fs:
            fs = np.load(files_path_prefix + f'{path_local}/{t}_FS.npy')
            fs_timelist.append(fs)
            if np.isfinite(np.nanmax(fs)):
                f_max = max(f_max, np.nanmax(fs))
            f_min = min(f_min, np.nanmin(fs))

        if load_c:
            try:
                corr_matrix = np.load(files_path_prefix + f'{path_local}/{t}_C.npy')
                c_timelist.append(corr_matrix)
            except FileNotFoundError:
                pass
        if load_e:
            e_matrix = np.load(files_path_prefix + f'{path_local}/{t}_E.npy')
            for i in range(4):
                e_max[i] = max(e_max[i], np.nanmax(e_matrix[i]))
                e_min[i] = min(e_min[i], np.nanmin(e_matrix[i]))
            e_timelist.append(e_matrix)

    borders = [a1_min, a1_max, a2_min, a2_max, b_min, b_max, f_min, f_max, e_min, e_max]
    return a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, e_timelist, borders


def scale_to_bins(arr, bins=100):
    quantiles = list(np.nanquantile(arr, np.linspace(0, 1, bins, endpoint=False)))
    # quantiles = sorted(quantiles)

    arr_scaled = np.zeros_like(arr)
    arr_scaled[np.isnan(arr)] = np.nan
    quantiles += [np.nanmax(arr)]
    # for j in tqdm.tqdm(range(bins - 1)):
    for j in range(bins):
        if j < bins - 1:
            mask = (
                    np.logical_not(np.isnan(arr))
                    & (quantiles[j] <= arr)
                    & (arr < quantiles[j + 1])
            )
        else:
            mask = (
                    np.logical_not(np.isnan(arr))
                    & (quantiles[j] <= arr)
                    & (arr <= quantiles[j + 1])
            )

        arr_scaled[mask] = (
                                   quantiles[j] + quantiles[j + 1]
                           ) / 2
    return arr_scaled, quantiles


def load_prepare_fluxes(sensible_filename: str,
                        latent_filename: str,
                        files_path_prefix: str = 'D://Data/OceanFull/',
                        prepare=True):
    maskfile = open(files_path_prefix + "DATA/mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)

    sensible_array = np.load(files_path_prefix + sensible_filename)
    latent_array = np.load(files_path_prefix + latent_filename)

    sensible_array = sensible_array.astype(float)
    latent_array = latent_array.astype(float)
    sensible_array[np.logical_not(mask), :] = np.nan
    latent_array[np.logical_not(mask), :] = np.nan

    # mean by day = every 4 observations
    pack_len = 4
    sensible_array = block_reduce(sensible_array,
                                  block_size=(1, pack_len),
                                  func=np.mean, )
    latent_array = block_reduce(latent_array,
                                block_size=(1, pack_len),
                                func=np.mean, )
    if prepare:
        sensible_array = scale_to_bins(sensible_array)
        latent_array = scale_to_bins(latent_array)
    return sensible_array, latent_array


def find_lost_pictures(files_path_prefix, type_prefix):
    num_lost = []
    for i in range(15598):
        if not os.path.exists(files_path_prefix + f'videos/tmp-coeff/{type_prefix}_{i:05d}.png'):
            print(files_path_prefix + f'videos/tmp-coeff/{type_prefix}_{i:05d}.png')
            num_lost.append(i + 1)

    print(num_lost)
    print(len(num_lost))

    start = num_lost[0]
    borders = []
    sum_lost = 0
    for j in range(1, len(num_lost)):
        if start is None:
            start = num_lost[j]

        # if (num_lost[j-1] == num_lost[j] - 1) and j != len(num_lost) - 1 and mask[j]:
        #     pass
        if not start is None and (num_lost[j - 1] != num_lost[j] - 1):
            borders.append([start, num_lost[j - 1]])
            sum_lost += num_lost[j - 1] - start + 1
            start = num_lost[j]

    print(borders)
    print(sum_lost)


def collect_estimates(files_path_prefix: str,
                      path_local: str,
                      coeff_types: list,
                      time_start: int,
                      time_end: int,):
    for coeff_type in coeff_types:
        print(f'Collecting {coeff_type}')
        first = np.load(files_path_prefix + path_local + f'Daily/{coeff_type}_{time_start}.npy')
        print(first.shape)
        # arr = np.zeros((time_end - time_start + 1, height, width, first.shape[1]))
        arr = np.zeros((time_end - time_start + 1, first.shape[0], height, width))
        for t in tqdm.tqdm(range(time_end-time_start)):
            # arr[t] = np.load(files_path_prefix + path_local + f'daily/{coeff_type}_{time_start + t}.npy').reshape((height, width, -1))
            try:
                arr[t] = np.load(files_path_prefix + path_local + f'Daily/{coeff_type}_{time_start + t}.npy').reshape((-1, height, width))
            except FileNotFoundError:
                print('No file ' + files_path_prefix + path_local + f'Daily/{coeff_type}_{time_start + t}.npy!')
        np.save(files_path_prefix + path_local + f'{coeff_type}_{time_start}-{time_end}.npy', arr)
    return


def mean_blocks(data:np.ndarray,
                mask: np.ndarray,
                block_size: int = 5):
    if len(data.shape) == 3:
        data_small = np.zeros((data.shape[0], data.shape[1] // block_size, data.shape[2] // block_size))
    else:
        data_small = np.zeros((data.shape[0], data.shape[1] // block_size, data.shape[2] // block_size, data.shape[3]))
    for i_block in range(0, data.shape[1] // block_size):
        for j_block in range(data.shape[2] // block_size):
            amount = 0
            if len(data.shape) == 3:
                tmp = np.zeros(data.shape[0])
            else:
                tmp = np.zeros((data.shape[0], data.shape[3]))
            for i_shift in range(-(block_size - 1)//2, (block_size - 1) // 2):
                for j_shift in range(-(block_size - 1) // 2, (block_size - 1) // 2):
                    i = i_block * block_size + i_shift
                    j = j_block * block_size + j_shift
                    if i < 0 or i > data.shape[1]-1 or j < 0 or j > data.shape[2] - 1 or not mask[i, j]:
                        continue
                    amount += 1
                    tmp += data[:, i, j]
            if amount:
                data_small[:, i_block, j_block] = tmp / amount
            else:
                data_small[:, i_block, j_block] = np.nan

    return data_small


def count_mean_year(files_path_prefix: str,
                    start_year: int = 2009,
                    end_year: int = 2019,
                    coeff_type: str = 'A',
                    flux_type: str = 'sensible',
                    method: str = 'Kor',
                    mask: np.ndarray = None,
                    ):
    """
    Counts mean year as np.array with shape (365, height, width) for 365 days with mean values for each day
    :param files_path_prefix: path to the working directory
    :param start_year:
    :param end_year:
    :param coeff_type: A/B/C/F/FS
    :param flux_type: sensible/latent/flux/sst/press
    :param method: 'Bel' or 'Kor'
    :param mask: np.array with shape (height, width) with boolean values, where 0 is for land and 1 is for ocean
    :return:
    """

    mean_year = np.zeros((365, height, width))

    for year in tqdm.tqdm(range(start_year, end_year)):
        time_start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=1979, month=1, day=1)).days

        for day in range(364):
            if method == 'Kor':
                coeff = np.load(files_path_prefix + f'Components/{flux_type}/{method}/daily/' + f'{coeff_type}_{day + time_start + 1}.npy')
            else:
                if coeff_type == 'A' and flux_type == 'sensible':
                    postfix = '_sens'
                    coeff = np.load(files_path_prefix + f'Coeff_data/{day + time_start + 1}_{coeff_type}{postfix}.npy')
                elif coeff_type == 'A' and flux_type == 'latent':
                    postfix = '_lat'
                    coeff = np.load(files_path_prefix + f'Coeff_data/{day + time_start + 1}_{coeff_type}{postfix}.npy')
                elif coeff_type == 'B' and flux_type == 'sensible':
                    coeff = np.load(files_path_prefix + f'Coeff_data/{day + time_start + 1}_{coeff_type}.npy')[0]
                elif coeff_type == 'B' and flux_type == 'latent':
                    coeff = np.load(files_path_prefix + f'Coeff_data/{day + time_start + 1}_{coeff_type}.npy')[3]
                elif coeff_type == 'A' and flux_type == 'flux':
                    coeff = np.load(files_path_prefix + f'Coeff_data_3d/flux-press/{day + time_start + 1}_{coeff_type}_sens.npy')
                elif coeff_type == 'B' and flux_type == 'flux':
                    coeff = np.load(files_path_prefix + f'Coeff_data_3d/flux-press/{day + time_start + 1}_{coeff_type}.npy')[0]
                elif coeff_type == 'A' and flux_type == 'press':
                    coeff = np.load(files_path_prefix + f'Coeff_data_3d/flux-press/{day + time_start + 1}_{coeff_type}_lat.npy')
                elif coeff_type == 'B' and flux_type == 'press':
                    coeff = np.load(files_path_prefix + f'Coeff_data_3d/flux-press/{day + time_start + 1}_{coeff_type}.npy')[3]
                elif coeff_type == 'A' and flux_type == 'sst':
                    coeff = np.load(files_path_prefix + f'Coeff_data_3d/flux-sst/{day + time_start + 1}_{coeff_type}_lat.npy')
                elif coeff_type == 'B' and flux_type == 'sst':
                    coeff = np.load(files_path_prefix + f'Coeff_data_3d/flux-sst/{day + time_start + 1}_{coeff_type}.npy')[3]

            mean_year[day, :, :] += coeff
            mean_year[day][np.logical_not(mask)] = None

    mean_year /= (end_year - start_year)
    np.save(files_path_prefix + f'Mean_year/{method}/{flux_type}_{coeff_type}_{start_year}-{end_year}.npy', mean_year)
    return