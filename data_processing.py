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


def dataframes_to_grids(files_path_prefix, flux_type, mask, components_amount, timesteps):
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
