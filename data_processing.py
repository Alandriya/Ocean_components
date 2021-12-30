import os
import pandas as pd
import tqdm
import numpy as np
from struct import unpack
from copy import deepcopy


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


def binary_to_array(files_path_prefix, flux_type, filename):
    """
    Creates np array from binary data
    :param files_path_prefix: path to the working directory
    :param flux_type: string of the flux type: 'sensible' or 'latent'
    :param filename:
    :return:
    """
    length = 7320
    arr_5years = np.empty((length, 29141), dtype=float)
    for i in tqdm.tqdm(range(length)):
        file = open(files_path_prefix + filename, "rb")
        offset = 32 + (62396 - length) * 116564 + 116564 * i
        file.seek(offset, 0)
        binary_values = file.read(116564)
        file.close()
        point = unpack('f' * 29141, binary_values)
        arr_5years[i] = point
    np.save(files_path_prefix + f'5years_{flux_type}.npy', arr_5years.transpose())
    del arr_5years
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


def load_ABCF(files_path_prefix, time_start, time_end, load_c=False):
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
    f_min = 10e9
    f_max = -1
    print('Loading ABC data')
    for t in tqdm.tqdm(range(time_start, time_end)):
        a_sens = np.load(files_path_prefix + f'Coeff_data/{t}_A_sens.npy')
        a_lat = np.load(files_path_prefix + f'Coeff_data/{t}_A_lat.npy')
        a_timelist.append([a_sens, a_lat])
        b_matrix = np.load(files_path_prefix + f'Coeff_data/{t}_B.npy')
        b_timelist.append(b_matrix)
        f = np.load(files_path_prefix + f'Coeff_data/{t}_F.npy')
        f_timelist.append(f)

        if load_c:
            try:
                corr_matrix = np.load(files_path_prefix + f'Coeff_data/{t}_C.npy')
                c_timelist.append(corr_matrix)
            except FileNotFoundError:
                pass
        # update borders
        a_max = max(a_max, np.nanmax(a_sens), np.nanmax(a_lat))
        a_min = min(a_min, np.nanmin(a_sens), np.nanmin(a_lat))
        b_max = max(b_max, np.nanmax(b_matrix))
        f_max = max(f_max, np.nanmax(f))
        f_min = min(f_min, np.nanmin(f))

    borders = [a_min, a_max, b_max, f_min, f_max]
    return a_timelist, b_timelist, c_timelist, f_timelist, borders
