import datetime
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from plotter import *
from scipy.stats import linregress
from scipy.stats import pearsonr

from loader import load_mask

# files_path_prefix = '/home/aosipova/EM_ocean/'
files_path_prefix = 'D:/Nastya/Data/OceanFull/'

def count_correlations(array1, array2, window_length, observations_per_day, names, offset):
    # corr_all = np.zeros((array1.shape[0]-window_length*observations_per_day, 81, 91), dtype=float)
    corr_all = np.zeros((330, 81, 91), dtype=float)
    corr = np.zeros((81, 91), dtype=float)
    for i in range(81):
        for j in range(91):
            if np.isnan(array1[:, i, j]).any():
                corr[i, j] = np.nan

    # for t in range(0, array1.shape[0] -window_length*observations_per_day, observations_per_day):
    for t in range(0, 330):
        if os.path.exists(files_path_prefix + f'Coeff_data_3d/{names[0]}-{names[1]}/C_{t + offset}.npy'):
            corr = np.load(files_path_prefix + f'Coeff_data_3d/{names[0]}-{names[1]}/C_{t + offset}.npy')
            print(f'Timestep {t}, exists', flush=True)
        else:
            print(f'Timestep {t}', flush=True)
            part1 = array1[(t+offset)*observations_per_day:(t+offset+window_length)*observations_per_day]
            part2 = array2[(t+offset)*observations_per_day:(t+offset+window_length)*observations_per_day]
            print(part1.shape)
            for i in range(81):
                for j in range(91):
                    if not np.isnan(part1[:, i, j]).any() and not np.isnan(part2[:, i, j]).any():
                        corr[i, j] = pearsonr(part1[:, i, j], part2[:, i, j])[0]
            np.save(files_path_prefix + f'Coeff_data_3d/{names[0]}-{names[1]}/C_{t + offset}.npy', corr)
            # corr = np.load(files_path_prefix + f'Coeff_data_3d/{names[0]}-{names[1]}/C_{t + offset}.npy')
        corr_all[t] = corr
    print('\n\n')
    np.save(files_path_prefix + f'Coeff_data_3d/{names[0]}-{names[1]}_correlations_{window_length}_days_freq_{observations_per_day}.npy', corr_all)
    return

# def collect_eigenvalues(files_path_prefix: str,
#                      n_lambdas: int,
#                      mask: np.ndarray,
#                      t_start: int,
#                      t_end: int,
#                      offset: int,
#                      array2,
#                      array2_quantiles,
#                      names: tuple = ('Sensible', 'Latent'),
#                      shape: tuple = (161, 181),
#                      ):
#     print(f'Collecting {names[0]}-{names[1]}', flush=True)
#     for t in range(t_start, t_end):
#         if t % 100 == 0:
#             print(t)
#         try:
#             eigenvalues = np.load(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}/eigenvalues_{t+offset}.npy')
#             eigenvalues = np.real(eigenvalues)
#             eigenvectors = np.load(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}/eigenvectors_{t+offset}.npy')
#             eigenvectors = np.real(eigenvectors)
#             print(f'Plot timestep {t + offset}', flush=True)
#         except FileNotFoundError:
#             print(f'No file step {t + offset}', flush=True)
#             continue
#
#         width, height = shape
#         matrix_list = [np.zeros(height * width) for _ in range(n_lambdas)]
#         lambda_list = []
#         max_list = []
#         min_list = []
#
#         n_bins = 100
#         for l in range(n_lambdas):
#             for j1 in range(0, n_bins):
#                 points_y1 = np.where((array2_quantiles[j1] <= array2[t+1]) & (array2[t+1] < array2_quantiles[j1 + 1]))[0]
#                 matrix_list[l][points_y1] = np.real(eigenvectors[j1, l])
#
#             matrix_list[l][np.logical_not(mask)] = np.nan
#             max_list.append(np.nanmax(matrix_list[l]))
#             min_list.append(np.nanmin(matrix_list[l]))
#             lambda_list.append(eigenvalues[l])
#
#         np.save(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}/eigen0_{t+offset}.npy', matrix_list[0])
#     return


# def get_eig(B: np.ndarray,
#             names: tuple):
#     """
#     Counts eigenvalues for the covariances matrix B for two cases: if both the variables in the data arrays are the
#     same, e.g. (Flux, Flux) and for different, e.g. (Flux, SST)
#     :param B: np.array with shape (n_bins, n_bins), two-dimensional
#     :param names: tuple with names of the data, e.g. ('Flux', 'SST'), ('Flux', 'Flux')
#     :return:
#     """
#     if names[0] == names[1]:
#         A = B
#     else:
#         # print('Performing A = B*B^T', flush=True)
#         A = np.dot(B, B.transpose())
#         # print('Getting sqrt(A)', flush=True)
#         A = sqrtm(A)
#
#     gc.collect()
#     # print('Counting eigenvalues', flush=True)
#     eigenvalues, eigenvectors = np.linalg.eig(A)
#     # sort by absolute value of the eigenvalues
#     eigenvalues = np.real(eigenvalues)
#     eigenvalues = [0 if np.isnan(e) else e for e in eigenvalues]
#     positions = [x for x in range(len(eigenvalues))]
#     positions = [x for _, x in reversed(sorted(zip(np.abs(eigenvalues), positions)))]
#     return np.take(eigenvalues, positions), np.take(eigenvectors, positions, axis=1), positions

# def count_eigenvalues_pair(files_path_prefix: str,
#                            array1: np.ndarray,
#                            array2: np.ndarray,
#                            array1_quantiles: list,
#                            array2_quantiles: list,
#                            t: int,
#                            n_bins: int,
#                            offset: int,
#                            names: tuple):
#     """
#
#     :param files_path_prefix: path to the working directory
#     :param array1: array with shape (height*width, n_days): e.g. (29141, 1410)
#     :param array2: array with shape (height*width, n_days): e.g. (29141, 1410)
#     :param array1_quantiles: list with length = n_bins + 1 of the quantiles built by scale_to_bins function
#     :param array2_quantiles: list with length = n_bins + 1 of the quantiles built by scale_to_bins function
#     :param t: relative time moment from the beginning of the array
#     :param n_bins: amount of bins to divide the values of each array
#     :param offset: shift of the beginning of the data arrays in days from 01.01.1979, for 01.01.2019 is 14610
#     :param names: tuple with names of the data arrays, e.g. ('Flux', 'SST')
#     :return:
#     """
#     if not os.path.exists(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}'):
#         os.mkdir(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}')
#
#     if os.path.exists(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}/eigenvalues_{t + offset}.npy'):
#         return
#
#     b_matrix = np.zeros((n_bins, n_bins))
#     for i1 in range(0, n_bins):
#         points_x1 = np.where((array1_quantiles[i1] <= array1[0]) & (array1[0] < array1_quantiles[i1 + 1]))[0]
#         for j1 in range(0, n_bins):
#             points_y1 = np.where((array2_quantiles[j1] <= array2[0]) & (array2[0] < array2_quantiles[j1 + 1]))[0]
#             if len(points_x1) and len(points_y1):
#                 mean1 = np.mean(array1[0, points_x1])
#                 mean2 = np.mean(array2[0, points_y1])
#                 vec1 = array1[1, points_x1] - mean1
#                 vec2 = array2[1, points_y1] - mean2
#                 b_matrix[i1, j1] = np.sum(np.multiply.outer(vec1, vec2).ravel())
#
#     b_matrix = np.nan_to_num(b_matrix)
#
#     # count eigenvalues
#     eigenvalues, eigenvectors, positions = get_eig(b_matrix, (names[0], names[1]))
#     # np.save(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}/eigenvalues_{t + offset}.npy', eigenvalues)
#     # np.save(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}/eigenvectors_{t + offset}.npy', eigenvectors)
#     return eigenvalues, eigenvectors, positions
#
# def count_eigenvalues_triplets(files_path_prefix: str,
#                                t_start: int,
#                                flux_array: np.ndarray,
#                                SST_array: np.ndarray,
#                                press_array: np.ndarray,
#                                mask: np.ndarray,
#                                offset: int = 14610,
#                                n_bins: int = 100,
#                                ):
#     """
#     Counts and plots eigenvalues and eigenvectors for pairs Flux-Flux, SST-SST, Flux-SST, Flux-Pressure for time range
#     offset + t_start, offset + len(flux_array)
#     :param files_path_prefix: path to the working directory
#     :param t_start: relative offset from the beginning of the array for time cycle
#     :param flux_array: array with shape (height*width, n_days): e.g. (29141, 1410) with flux values
#     :param SST_array: array with shape (height*width, n_days): e.g. (29141, 1410) with SST values
#     :param press_array: array with shape (height*width, n_days): e.g. (29141, 1410) with pressure values
#     :param mask:
#     :param offset: shift of the beginning of the data arrays in days from 01.01.1979, for 01.01.2019 is 14610
#     :param n_bins: amount of bins to divide the values of each array
#     :return:
#     """
#
#     # flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
#     # SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)
#     # press_array_grouped, quantiles_press = scale_to_bins(press_array, n_bins)
#
#     quantiles_flux = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_quantiles.npy')
#     quantiles_sst = np.load(cfg.root_path + f'DATA/SST_1979-2025_quantiles.npy')
#     quantiles_press = np.load(cfg.root_path + f'DATA/PRESS_1979-2025_quantiles.npy')
#
#     if not os.path.exists(files_path_prefix + f'Eigenvalues-mini'):
#         os.mkdir(files_path_prefix + f'Eigenvalues-mini')
#
#     def count_pair(pair_name, array1, array2, quantiles1, quantiles2):
#         print(f'Pair {pair_name}')
#         n_lambdas = 3
#         n_bins = 100
#         for t in range(t_start, flux_array.shape[0] - 1):
#             if t % 100 == 0:
#                 print(f'Timestep {t}', flush=True)
#             eigenvalues, eigenvectors, positions = count_eigenvalues_pair(files_path_prefix, array1[t:t + 2],
#                                                                           array2[t:t + 2], quantiles1,
#                                                                           quantiles2, t, n_bins,
#                                                                           offset, pair_name)
#             eigenvalues = np.real(eigenvalues)
#             eigenvectors = np.real(eigenvectors)
#             print(array1.shape)
#             width, height = array1.shape[2], array1.shape[1]
#             matrix_list = [np.zeros((height, width)) for _ in range(n_lambdas)]
#             lambda_list = []
#             max_list = []
#             min_list = []
#
#             n_bins = 100
#             for l in range(n_lambdas):
#                 for j1 in range(0, n_bins):
#                     points_y1 = \
#                     np.where((quantiles2[j1] <= array2[t + 1]) & (array2[t + 1] < quantiles2[j1 + 1]))[0]
#                     matrix_list[l][points_y1] = np.real(eigenvectors[j1, l])
#
#                 matrix_list[l][np.logical_not(mask)] = np.nan
#                 max_list.append(np.nanmax(matrix_list[l]))
#                 min_list.append(np.nanmin(matrix_list[l]))
#                 lambda_list.append(eigenvalues[l])
#
#             np.save(files_path_prefix + f'Eigenvalues-mini/{pair_name[0]}-{pair_name[1]}/eigen0_{t + offset}.npy',
#                     matrix_list[0])
#         return
#
#     count_pair(('Flux', 'Flux'), flux_array, flux_array, quantiles_flux, quantiles_flux)
#     count_pair(('SST', 'SST'), sst_array, sst_array, quantiles_sst, quantiles_sst)
#     count_pair(('Pressure', 'Pressure'), press_array, press_array, quantiles_press, quantiles_press)
#
#     count_pair(('Flux', 'SST'), flux_array, sst_array, quantiles_flux, quantiles_sst)
#     count_pair(('Flux', 'Pressure'), flux_array, press_array, quantiles_flux, quantiles_press)
#     count_pair(('SST', 'Pressure'), sst_array, press_array, quantiles_sst, quantiles_press)
#     return
# def count_1d_Korolev(files_path_prefix: str,
#                      flux: np.ndarray,
#                      time_start: int,
#                      time_end: int,
#                      path: str = 'Synthetic/',
#                      quantiles_amount: int = 50,
#                      n_components: int = 2,
#                      start_index: int = 0,
#                      ):
#     """
#     Counts and saves to files_path_prefix + path + 'Kor/daily' A and B estimates for flux array for each day
#     t+start index for t in (time_start, time_end)
#     :param files_path_prefix: path to the working directory
#     :param flux: np.array with shape [time_steps, height, width]
#     :param time_start: int counter of start day
#     :param time_end: int counter of end day
#     :param path: additional path to the folder from files_path_prefix, like 'Synthetic/', 'Components/sensible/',
#     'Components/latent/'
#     :param quantiles_amount: how many quantiles to use (for one step)
#     :param n_components: amount of components for EM
#     :param start_index: offset index when saving maps
#     :return:
#     """
#     if not os.path.exists(files_path_prefix + '3D_coeff_Kor'):
#         os.mkdir(files_path_prefix + '3D_coeff_Kor')
#
#     if not os.path.exists(files_path_prefix + '3D_coeff_Kor/' + path):
#         os.mkdir(files_path_prefix + '3D_coeff_Kor/' + path)
#
#     if not os.path.exists(files_path_prefix + f'3D_coeff_Kor/{path}/daily-mini'):
#         os.mkdir(files_path_prefix + f'3D_coeff_Kor/{path}/daily-mini')
#
#     a_map = np.zeros((flux.shape[1], flux.shape[2]), dtype=float)
#     b_map = np.zeros((flux.shape[1], flux.shape[2]), dtype=float)
#     a_map[np.isnan(flux[0])] = np.nan
#     b_map[np.isnan(flux[0])] = np.nan
#     # start_time = time.time()
#     for t in range(time_start + 1, time_end):
#         if t % 100 == 0:
#             print(f't = {t}', flush=True)
#         flux_array, quantiles = scale_to_bins(flux[t - 1], quantiles_amount)
#         flux_set = list(set(flux_array[np.logical_not(np.isnan(flux_array))].flat))
#         for group in range(len(flux_set)):
#             value_t0 = flux_set[group]
#             if np.isnan(value_t0):
#                 continue
#             day_sample = (flux[t][np.where(flux_array == value_t0)] -
#                           flux[t - 1][np.where(flux_array == value_t0)]).flatten()
#             # print(len(day_sample))
#             # plot_hist(day_sample, group)
#
#             window = day_sample
#             try:
#                 gm = GaussianMixture(n_components=n_components,
#                                      tol=1e-4,
#                                      covariance_type='spherical',
#                                      max_iter=1000,
#                                      init_params='random',
#                                      n_init=5
#                                      ).fit(window.reshape(-1, 1))
#                 means = gm.means_.flatten()
#                 sigmas_squared = gm.covariances_.flatten()
#                 weights = gm.weights_.flatten()
#                 weights /= sum(weights)
#             except ValueError:
#                 means = np.mean(window)
#                 weights = np.ones(2)
#                 sigmas_squared = np.array([1, 0])
#
#             a_sum = sum(means * weights)
#             b_sum = math.sqrt(sum(weights * (means ** 2 + sigmas_squared)))
#
#             a_map[np.where(flux_array == value_t0)] = a_sum
#             b_map[np.where(flux_array == value_t0)] = b_sum
#
#         np.save(files_path_prefix + f'3D_coeff_Kor/{path}/daily-mini/A_{t+start_index}.npy', a_map)
#         np.save(files_path_prefix + f'3D_coeff_Kor/{path}/daily-mini/B_{t+start_index}.npy', b_map)
#         # print(f'Iteration {t}: {(time.time() - start_time):.1f} seconds')
#         # start_time = time.time()
#     return


def plot_eigenvalues_trends(files_path_prefix: str,
                             eigenarray: np.ndarray,
                             time_start: int,
                             time_end: int,
                             mean_days: int,
                             names: list,
                             ):
    sns.set_style("whitegrid")
    font = {'size': 14}
    font_names = {'weight': 'bold', 'size': 20}
    matplotlib.rc('font', **font)
    days = [datetime.datetime(1979, 1, 1) + datetime.timedelta(days=t) for t in
            range(time_start, time_end, mean_days)]

    # fig, axs = plt.subplots(figsize=(20, 10))
    fig, axs = plt.subplots(6, 1, figsize=(20, 10))
    # Major ticks every half year, minor ticks every month,


    fig.suptitle(f'Eigenvalues trends, mean of every {mean_days} days')
    for i in range(len(names)):
        pair_name = names[i]
        eigens = eigenarray[:-300, i]

        # if mean_days == 365:
        #     axs[i].xaxis.set_minor_locator(mdates.MonthLocator())
        # axs[i].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs.xaxis.get_major_locator()))

        if len(eigens) % mean_days:
            eigens = eigens[:-(len(eigens) % mean_days)]


        eigens = np.mean(eigens.reshape(-1, mean_days), axis=1)
        days = days[:len(eigens)]
        x = np.array(range(time_start, time_end, mean_days))
        x = x[:len(eigens)]

        axs[i].plot(days, eigens, label=f'{pair_name[0]}-{pair_name[1]}')
        res = linregress(x, eigens)
        axs[i].plot(days, res.intercept + res.slope * x, '--', c='darkviolet', label='fitted trend')
        axs[i].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        print(f'{pair_name[0]}-{pair_name[1]}: {res.slope:.2e}')

    fig.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Eigenvalues/({time_start}-{time_end})_mean_{mean_days}_fit_regression_eigenvalues.png')
    return


if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------
    mask = load_mask(files_path_prefix)
    # ---------------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta7 = (datetime.datetime(2024, 11, 28, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    days_delta8 = (datetime.datetime(2024, 11, 28, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta9 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    variable = 'FLUX'

    # full_array = np.zeros((days_delta8, cfg.height, cfg.width), dtype=float)
    # for start_year in [1979, 1989, 1999, 2009, 2019]:
    #     end_year, offset = count_offset(start_year)
    #     if variable == 'FLUX':
    #         array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    #     elif variable == 'SST':
    #         array = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy')
    #     else:
    #         array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    #
    #     array = array.reshape((161, 181, -1))
    #     array = array[::2, ::2, :]
    #     array = np.swapaxes(array, 0, 2)
    #     array = np.swapaxes(array, 1, 2)
    #     print(array.shape, flush=True)
    #     if start_year == 2019:
    #         full_array[offset:] = array
    #     else:
    #         end_year2, offset2 = count_offset(start_year + 10)
    #         full_array[offset:offset2] = array
    #
    # if not os.path.exists(files_path_prefix + f'DATA'):
    #     os.mkdir(files_path_prefix + f'DATA')
    #
    # np.save(files_path_prefix + f'DATA/{variable}_1979-2025_grouped.npy', full_array)
    # print('Beginning scaling')
    # array_scaled, quantiles = scale_to_bins(full_array, bins=cfg.bins)
    # np.save(files_path_prefix + f'DATA/{variable}_1979-2025_grouped_scaled.npy', array_scaled)
    # np.save(files_path_prefix + f'DATA/{variable}_1979-2025_quantiles.npy', quantiles)
    # print('Ended')
    # array_scaled = np.load(files_path_prefix + f'DATA/{variable}_1979-2025_grouped_scaled.npy')
    # print(np.isnan(array_scaled).any())

    # for variable in ['FLUX', 'SST', 'PRESS']:
    #     print(variable, flush=True)
    #     # count A and B coeff mini
    #     array = np.load(files_path_prefix + f'DATA/{variable}_1979-2025_grouped.npy')
    #     np.nan_to_num(array, copy=False)
    #     # count_1d_Korolev(files_path_prefix, array, 0, array.shape[0], variable, 15)

        # print('Collecting a_coeff', flush=True)
        # a_coeff = np.zeros_like(array)
        # for day in range(0, a_coeff.shape[0]-1):
        #     a_day = np.load(files_path_prefix + f'3D_coeff_Kor/{variable}/daily-mini/A_{day+1}.npy')
        #     a_coeff[day] = a_day
        # np.save(files_path_prefix + f'DATA/{variable}_1979-2025_a_coeff.npy', a_coeff)

        # print('Collecting b_coeff', flush=True)
        # b_coeff = np.zeros_like(array)
        # for day in range(0, b_coeff.shape[0]-1):
        #     b_day = np.load(files_path_prefix + f'3D_coeff_Kor/{variable}/daily-mini/B_{day+1}.npy')
        #     b_coeff[day] = b_day
        # np.save(files_path_prefix + f'DATA/{variable}_1979-2025_b_coeff.npy', b_coeff)

    # flux_array = np.load(files_path_prefix + f'DATA/FLUX_1979-2025_grouped_diff.npy')
    # sst_array = np.load(files_path_prefix + f'DATA/SST_1979-2025_grouped.npy')
    # press_array = np.load(files_path_prefix + f'DATA/PRESS_1979-2025_grouped.npy')
    #
    # np.save(files_path_prefix + f'DATA/FLUX_1979-2025_grouped_diff.npy', np.diff(flux_array, axis=0))
    # np.save(files_path_prefix + f'DATA/SST_1979-2025_grouped_diff.npy', np.diff(sst_array, axis=0))
    # np.save(files_path_prefix + f'DATA/PRESS_1979-2025_grouped_diff.npy', np.diff(press_array, axis=0))
    #
    # np.nan_to_num(flux_array, copy=False)
    # np.nan_to_num(sst_array, copy=False)
    # np.nan_to_num(press_array, copy=False)

    # for variable in ['FLUX', 'SST', 'PRESS']:
    #     array = np.load(files_path_prefix + f'DATA/{variable}_1979-2025_grouped_diff.npy')
    #     array_scaled, quantiles = scale_to_bins(array, bins=cfg.bins)
    #     np.save(files_path_prefix + f'DATA/{variable}_1979-2025_grouped_diff_scaled.npy', array_scaled)
    #     np.save(files_path_prefix + f'DATA/{variable}_1979-2025_diff_quantiles.npy', np.array(quantiles))
    #     print(np.array(quantiles))
    # for variable in ['FLUX', 'SST', 'PRESS']:
    #     array = np.load(files_path_prefix + f'DATA/{variable}_1979-2025_grouped_diff.npy')[0:days_delta9]
    #     mean_year = np.zeros((365, 81, 91))
    #     for year in range(1979, 2024):
    #         print(year)
    #         for i in range(365):
    #             day = (datetime.datetime(year, 1, 1, 0, 0) -
    #                    datetime.datetime(1979, 1, 1, 0, 0)).days + i
    #             mean_year[i] += array[day]
    #
    #     mean_year /= (2024-1979)
    #     np.save(files_path_prefix + f'DATA/{variable}_mean_year_diff.npy', mean_year)


    # count_eigenvalues_triplets(files_path_prefix, 0, flux_array, sst_array, press_array, mask, 0, 100)

    # for var1 in ['Flux', 'SST', 'Pressure']:
    #     for var2 in ['Flux', 'SST', 'Pressure']:
    #         print(f'Collecting {var1}-{var2}', flush=True)
    #         eigenarray = np.zeros_like(flux_array)
    #         for t in range(flux_array.shape[0]):
    #             try:
    #                 eigenarray[t] = np.load(files_path_prefix + f'Eigenvalues/{var1}-{var2}/eigen0_{t}.npy').reshape((161, 181))[::2, ::2]
    #             except FileNotFoundError:
    #                 print(f'Not existing {files_path_prefix}/Eigenvalues/{var1}-{var2}/eigen0_{t}.npy', flush=True)
    #         np.save(files_path_prefix + f'DATA/{var1}-{var2}_eigen.npy', eigenarray)

    # pairs = [('Flux', 'Flux'), ('Flux', 'SST'), ('SST', 'SST'), ('SST', 'Pressure'), ('Pressure', 'Pressure'), ('Flux', 'Pressure')]
    # eigenvalues = np.zeros((flux_array.shape[0], 6))
    # for i in range(6):
    #     pair = pairs[i]
    #     print(f'Collecting pair {pair}')
    #     for t in range(eigenvalues.shape[0]):
    #         try:
    #             eigenvalues_1d = np.load(files_path_prefix + f'Eigenvalues/{pair[0]}-{pair[1]}/eigenvalues_{t}.npy')
    #             eigenvalues[t, i] = eigenvalues_1d[0]
    #         except FileNotFoundError:
    #             print(f'Not existing {files_path_prefix}/Eigenvalues/{pair[0]}-{pair[1]}/eigenvalues_{t}.npy', flush=True)
    # np.save(files_path_prefix + f'DATA/EIGENVALUES_1979-2025.npy', eigenvalues)

    # eigenvalues = np.load(files_path_prefix + f'DATA/EIGENVALUES_1979-2025.npy')
    # print(np.unique(eigenvalues[0]))
    # print(eigenvalues[0])

    # sensible = np.load(files_path_prefix + f'Fluxes/SENSIBLE_2024_hourly.npy')
    # latent =  np.load(files_path_prefix + f'Fluxes/LATENT_2024_hourly.npy')
    # flux_array = sensible + latent
    # np.save(files_path_prefix + f'DATA/FLUX_2024_hourly.npy', flux_array)

    # flux_array = np.load(files_path_prefix + f'DATA/FLUX_1979-2025_grouped.npy')
    # sst_array = np.load(files_path_prefix + f'DATA/SST_1979-2025_grouped.npy')
    # press_array = np.load(files_path_prefix + f'DATA/PRESS_1979-2025_grouped.npy')


    # flux_array = np.load(files_path_prefix + f'DATA/FLUX_2024_hourly.npy')
    # sst_array = np.load(files_path_prefix + f'DATA/SST_2024_hourly.npy')
    # press_array = np.load(files_path_prefix + f'DATA/PRESS_2024_hourly.npy')
    #
    # mask = load_mask(files_path_prefix)
    # mask_not = np.logical_not(mask)
    #
    # flux_array = flux_array[24:8784, ::2, ::2]
    # sst_array = sst_array[24:, ::2, ::2]
    # press_array = press_array[24:, ::2, ::2]
    #
    # flux_array[:, np.where(mask_not)[0], np.where(mask_not)[1]] = np.nan
    # sst_array[:, np.where(mask_not)[0], np.where(mask_not)[1]] = np.nan
    # press_array[:, np.where(mask_not)[0], np.where(mask_not)[1]] = np.nan
    #
    # # plot_flux_sst_press(files_path_prefix, flux_array, sst_array, press_array, 0, 24*30, frequency=24)
    # # raise ValueError
    # print(flux_array.shape)
    # # print(sst_array.shape)
    # # print(flux_array.shape[0] // 24)
    # frequency = 1
    # offset = 0 if frequency == 24 else days_delta9
    #
    # print(f'pair flux-sst', flush=True)
    # count_correlations(flux_array, sst_array, 7, frequency, ('flux', 'sst'), offset)
    # print(f'pair sst-press', flush=True)
    # count_correlations(sst_array, press_array,7, frequency, ('sst', 'press'), offset)
    # print(f'pair flux-press', flush=True)
    # count_correlations(flux_array, press_array, 7, frequency, ('flux', 'press'), offset)


    # flux_sst = np.load(files_path_prefix + f'Coeff_data_3d/flux-sst_correlations_7_days_freq_{frequency}.npy')
    # sst_press = np.load(files_path_prefix + f'Coeff_data_3d/sst-press_correlations_7_days_freq_{frequency}.npy')
    # flux_press = np.load(files_path_prefix + f'Coeff_data_3d/flux-press_correlations_7_days_freq_{frequency}.npy')

    # print(flux_sst.shape)
    # start = 0 if frequency == 24 else 1
    # start = 0
    # plot_flux_sst_press(files_path_prefix, flux_sst, sst_press, flux_press, start, 100, datetime.datetime(2024, 1, 2, 0, 0), 0, frequency)

    pairs = [('Flux', 'Flux'), ('Flux', 'SST'), ('SST', 'SST'), ('SST', 'Pressure'), ('Pressure', 'Pressure'),
             ('Flux', 'Pressure')]
    frequency = 365
    eigenvalues = np.load(files_path_prefix + f'DATA/EIGENVALUES_1979-2025.npy')
    print(eigenvalues.shape)
    plot_eigenvalues_trends(files_path_prefix, eigenvalues, 0, eigenvalues.shape[0], frequency, pairs)

