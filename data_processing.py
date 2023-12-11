import math
import os
import pandas as pd
import tqdm
import shutil
import numpy as np
from struct import unpack
from copy import deepcopy
from skimage.measure import block_reduce
import datetime

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


def load_ABCF(files_path_prefix,
              time_start,
              time_end,
              load_a=False,
              load_b=False,
              load_c=False,
              load_f=False,
              load_fs=False,
              verbose=False,
              path_local: str = 'Coeff_data'):
    """
    Loads data from files_path_prefix + coeff_data directory and counts borders
    :param files_path_prefix: path to the working directory
    :param time_start: first time step
    :param time_end: last time step
    :param load_a: if to load A data or not
    :param load_b:
    :param load_c:
    :param load_f:
    :param verbose: if to print logs
    :return:
    """
    a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist = list(), list(), list(), list(), list()

    a1_max, a2_max = 0, 0
    a1_min, a2_min = 0, 0

    b_max = [0, 0, 0, 0]
    b_min = [0, 0, 0, 0]
    f_min = 10
    f_max = -1

    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)

    sst_coeff = 37.95727539 / (0.8980015822683038 + 0.8980015822683038)
    flux_coeff = 2558.356628 / (0.8544676970659135 + 0.8544676970659135)
    press_coeff = 17950.53906 / (0.8447768941044158 + 0.8447768941044158)

    coeff_1 = 1
    coeff_2 = 1

    if verbose:
        print('Loading ABC data')
    for t in range(time_start, time_end):
        if load_a:
            a_sens = np.load(files_path_prefix + f'{path_local}/{t}_A_sens.npy')
            a_lat = np.load(files_path_prefix + f'{path_local}/{t}_A_lat.npy')
            a_timelist.append([a_sens, a_lat])

            a_sens *= coeff_1
            a_lat *= coeff_2

            a1_max = max(a1_max, np.nanmax(a_sens))
            a1_min = min(a1_min, np.nanmin(a_sens))
            a2_max = max(a2_max, np.nanmax(a_lat))
            a2_min = min(a2_min, np.nanmin(a_lat))

            # a_max = max(a_max, np.nanmax(a_sens), np.nanmax(a_lat))
            # a_min = min(a_min, np.nanmin(a_sens), np.nanmin(a_lat))

        if load_b:
            b_matrix = np.load(files_path_prefix + f'{path_local}/{t}_B.npy')
            b_matrix[0] *= coeff_1
            b_matrix[3] *= coeff_2
            b_matrix[1] *= math.sqrt(coeff_1 * coeff_2)
            b_matrix[2] *= math.sqrt(coeff_1 * coeff_2)
            for i in range(4):
                np.nan_to_num(b_matrix[i], False, -10)
                b_matrix[i][np.logical_not(mask.reshape((height, width)))] = np.nan

                b_max[i] = max(b_max[i], np.nanmax(b_matrix[i]))
                b_min[i] = min(b_min[i], np.nanmin(b_matrix[i]))
            b_timelist.append(b_matrix)


        if load_f:
            f = np.load(files_path_prefix + f'{path_local}/{t}_F.npy')
            f_timelist.append(f)
            if np.isfinite(np.nanmax(f)):
                f_max = max(f_max, np.nanmax(f))
            f_min = min(f_min, np.nanmin(f))
        if load_fs:
            fs = np.load(files_path_prefix + f'{path_local}/{t}_FS.npy')
            # fs = np.load(files_path_prefix + f'{path_local}/{t}_F_separate.npy')
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

    borders = [a1_min, a1_max, a2_min, a2_max, b_min, b_max, f_min, f_max]
    return a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders


def scale_to_bins(arr, bins=100):
    quantiles = list(np.nanquantile(arr, np.linspace(0, 1, bins, endpoint=False)))

    arr_scaled = np.zeros_like(arr)
    arr_scaled[np.isnan(arr)] = np.nan
    for j in tqdm.tqdm(range(bins - 1)):
    # for j in range(bins - 1):
        arr_scaled[np.where((np.logical_not(np.isnan(arr))) & (quantiles[j] <= arr) & (arr < quantiles[j + 1]))] = \
            (quantiles[j] + quantiles[j + 1]) / 2

    quantiles += [np.nanmax(arr)]

    return arr_scaled, quantiles


def load_prepare_fluxes(sensible_filename: str,
                        latent_filename: str,
                        files_path_prefix: str = 'D://Data/OceanFull/',
                        prepare=True):
    maskfile = open(files_path_prefix + "mask", "rb")
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


def init_directory(files_path_prefix: str, flux_type: str):
    """
    Creates (or re-creates) subdirectories for saving pictures and video

    :param files_path_prefix: path to the working directory
    :param flux_type: string of the flux type: 'sensible' or 'latent'
    :return:
    """
    if not os.path.exists(files_path_prefix + f'videos/{flux_type}'):
        os.mkdir(files_path_prefix + f'videos/{flux_type}')

    if not os.path.exists(files_path_prefix + f'tmp_arrays/{flux_type}'):
        os.mkdir(files_path_prefix + f'tmp_arrays/{flux_type}')

    if os.path.exists(files_path_prefix + f'videos/{flux_type}/tmp'):
        shutil.rmtree(files_path_prefix + f'videos/{flux_type}/tmp')

    if not os.path.exists(files_path_prefix + f'videos/{flux_type}/tmp'):
        os.mkdir(files_path_prefix + f'videos/{flux_type}/tmp')
    return
