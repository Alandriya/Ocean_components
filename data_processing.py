import os
import pandas as pd
import tqdm
import numpy as np
from struct import unpack
from copy import deepcopy
from skimage.measure import block_reduce


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


def binary_to_array(files_path_prefix, input_filename, output_filename):
    """
    Creates np array from binary data
    :param files_path_prefix: path to the working directory
    :param flux_type: string of the flux type: 'sensible' or 'latent'
    :param filename:
    :return:
    """
    # length = 14640
    length = 62396 - 14640 * 4
    arr_10years = np.empty((length, 29141), dtype=float)
    file = open(files_path_prefix + input_filename, "rb")
    for i in tqdm.tqdm(range(length)):
        offset = 32 + 116564 * i + 116564 * (length * 4)
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


def load_ABCF(files_path_prefix, time_start, time_end, load_c=False, load_f=False):
    """
    Loads data from files_path_prefix + AB_coeff_data directory and counts borders
    :param files_path_prefix: path to the working directory
    :param time_start: first time step
    :param time_end: last time step
    :param load_c: load C coefficients flag
    :return:
    """
    a_timelist, b_timelist, c_timelist, f_timelist = list(), list(), list(), list()

    a_max = 0
    a_min = 0
    b_max = 0
    b_min = 0
    f_min = 10e9
    f_max = -1
    print('Loading ABC data')
    for t in tqdm.tqdm(range(time_start, time_end)):
        a_sens = np.load(files_path_prefix + f'Coeff_data/{t}_A_sens.npy')
        a_lat = np.load(files_path_prefix + f'Coeff_data/{t}_A_lat.npy')
        a_timelist.append([a_sens, a_lat])
        b_matrix = np.load(files_path_prefix + f'Coeff_data/{t}_B.npy')
        b_timelist.append(b_matrix)

        # update borders
        a_max = max(a_max, np.nanmax(a_sens), np.nanmax(a_lat))
        a_min = min(a_min, np.nanmin(a_sens), np.nanmin(a_lat))
        b_max = max(b_max, np.nanmax(b_matrix))
        b_min = min(b_min, np.nanmin(b_matrix))

        if load_f:
            f = np.load(files_path_prefix + f'Coeff_data/{t}_F.npy')
            f_timelist.append(f)
            f_max = max(f_max, np.nanmax(f))
            f_min = min(f_min, np.nanmin(f))
        if load_c:
            try:
                corr_matrix = np.load(files_path_prefix + f'Coeff_data/{t}_C.npy')
                c_timelist.append(corr_matrix)
            except FileNotFoundError:
                pass

    borders = [a_min, a_max, b_min, b_max, f_min, f_max]
    return a_timelist, b_timelist, c_timelist, f_timelist, borders


def scale_to_bins(arr):
    # set each value with the number of quantile from 0 to 100 in which it belongs
    quantiles = np.nanquantile(arr, np.linspace(0, 1, 100, endpoint=False))
    arr_digit = np.digitize(arr, quantiles)

    # return nan back :)
    arr_digit = arr_digit.astype(float)
    arr_digit[np.isnan(arr)] = np.nan
    return arr_digit


def load_prepare_fluxes(mask, sensible_filename, latent_filename):
    sensible_array = np.load(sensible_filename)
    latent_array = np.load(latent_filename)

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

    sensible_array = scale_to_bins(sensible_array)
    latent_array = scale_to_bins(latent_array)
    return sensible_array, latent_array