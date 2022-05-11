import datetime
import os.path
import time

import numpy as np
import pandas as pd
import tqdm

from video import *
from plot_fluxes import *
from plot_Bel_coefficients import *
from data_processing import *
from ABCF_coeff_counting import *
from func_estimation import *
from data_processing import load_prepare_fluxes
from func_estimation import estimate_a_flux_by_months
from extreme_evolution import *
import cycler
from EM_hybrid import *
from fluxes_distribution import *
from EM_count_coefficients import *
from copy import deepcopy

# Parameters
files_path_prefix = 'D://Data/OceanFull/'
flux_type = 'sensible'
# flux_type = 'latent'

# timesteps = 7320
timesteps = 1829

if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------
    borders = [[0, 1000]]
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    # # ---------------------------------------------------------------------------------------
    # sensible_array = np.load(files_path_prefix + 'SENSIBLE_2019-2021.npy')
    # sensible_array[np.logical_not(mask), :] = np.nan
    # latent_array = np.load(files_path_prefix + 'LATENT_2019-2021.npy')
    # latent_array[np.logical_not(mask), :] = np.nan
    # offset = (datetime.datetime(2021, 7, 1) - datetime.datetime(2019, 1, 1)).days * 4
    # plot_fluxes(files_path_prefix, sensible_array, latent_array, offset, offset + 10, 1, datetime.datetime(2021, 7, 1))
    # ---------------------------------------------------------------------------------------
    # raise ValueError
    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_1979-1989(scaled).npy')
    # latent_array = np.load(files_path_prefix + 'latent_grouped_1979-1989(scaled).npy')
    # sensible_array = sensible_array.astype(float)
    # sensible_array = np.diff(sensible_array)
    # latent_array = latent_array.astype(float)
    # latent_array = np.diff(latent_array)

    # lost = list()
    # for i in range(1, 15796):
    #     if not os.path.exists(files_path_prefix + f'Coeff_data/{i}_A_sens.npy'):
    #         lost.append(i)
    #
    # print(lost)
    # raise ValueError
    # [3653, 7305, 10958, 14610]
    # hole = 14610

    # time1 = '2009-2019'
    # time2 = '2019-2022'
    #
    # days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    # days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    # days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    # days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    # days_delta5 = (datetime.datetime(2022, 4, 2, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    #
    # sensible1 = np.load(files_path_prefix + f'sensible_grouped_{time1}(scaled).npy')
    # sensible2 = np.load(files_path_prefix + f'sensible_grouped_{time2}(scaled).npy')
    # sensible3 = np.zeros((161*181, sensible1.shape[1] + sensible2.shape[1]))
    # sensible3[:, :sensible1.shape[1]] = sensible1
    # sensible3[:, sensible1.shape[1]:] = sensible2
    # del sensible1
    # del sensible2
    #
    # latent1 = np.load(files_path_prefix + f'latent_grouped_{time1}(scaled).npy')
    # latent2 = np.load(files_path_prefix + f'latent_grouped_{time2}(scaled).npy')
    # latent3 = np.zeros((161*181, latent1.shape[1] + latent2.shape[1]))
    # latent3[:, :latent1.shape[1]] = latent1
    # latent3[:, latent1.shape[1]:] = latent2
    # # del latent1
    # del latent2
    #
    # count_abf_coefficients(files_path_prefix, mask, sensible3, latent3, time_start=latent1.shape[1] - 1, time_end=latent1.shape[1]+1,
    #                            offset=days_delta1 + days_delta2 + days_delta3)

    #
    # for border in borders:
    #     start = border[0]
    #     end = border[1]
    #
    #     count_abf_coefficients(files_path_prefix, mask, sensible_array[:, start:end+1], latent_array[:, start:end+1], time_start=0, time_end=end-start,
    #                            offset=start)

    # offset = 14640 / 4 * 1
    # parallel_AB(4, 'SENSIBLE_1989-1999.npy', 'LATENT_1989-1999.npy', offset)

    # ---------------------------------------------------------------------------------------
    # binary_to_array(files_path_prefix, "s79-21", 'SENSIBLE_1979-1989', datetime.datetime(2019, 1, 1), datetime.datetime(2021, 9, 16))
    # binary_to_array(files_path_prefix, "l79-21", 'LATENT_2009-2019', datetime.datetime(2009, 1, 1), datetime.datetime(2019, 1, 1))
    # ---------------------------------------------------------------------------------------
    # Components determination part
    # sort_by_means(files_path_prefix, flux_type)
    # init_directory(files_path_prefix, flux_type)

    # dataframes_to_grids(files_path_prefix, flux_type, mask, components_amount, 100)
    # draw_frames(files_path_prefix, flux_type, mask, components_amount, timesteps=timesteps)
    # create_video(files_path_prefix, files_path_prefix+'videos/{flux_type}/tmp/', '', f'{flux_type}_5years_weekly', speed=30)
    # ---------------------------------------------------------------------------------------
    # mask = np.array(mask, dtype=int)
    # points = plot_typical_points(files_path_prefix, mask)
    # point = points[12]
    # radius = 2
    # month = 1
    #
    # import shutil
    # shutil.rmtree(files_path_prefix + f'Func_repr/a-flux-monthly/{month}')
    # os.mkdir(files_path_prefix + f'Func_repr/a-flux-monthly/{month}')
    #
    # plot_current_bigpoint(files_path_prefix, mask, point, radius)
    # estimate_a_flux_by_months(files_path_prefix, month, point, radius)
    # raise ValueError
    # start_month = 1
    # points = plot_typical_points(files_path_prefix, np.array(mask, dtype=int))
    # # sensible_array, latent_array = load_prepare_fluxes('SENSIBLE_1979-1989.npy', 'LATENT_1979-1989.npy', False)
    # sensible_array, latent_array = None, None
    # for p in points:
    #     estimate_flux(files_path_prefix, sensible_array, latent_array, start_month, p)
    #     raise ValueError
    # ---------------------------------------------------------------------------------------

    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2022, 4, 2, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days

    # time_start = 1
    # time_end = 10
    # mean_width = 7
    #
    # plot_step = 1
    # delta = 0
    #
    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_1979-1989.npy')
    # latent_array = np.load(files_path_prefix + 'latent_grouped_1979-1989.npy')
    # plot_fluxes(files_path_prefix, sensible_array, latent_array, time_start, time_end)
    # raise ValueError

    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start, time_end, load_a=True, load_b=True)
    # count_c_coeff(files_path_prefix, a_timelist, b_timelist, time_start, 14)
    # count_f_separate_coeff(files_path_prefix, a_timelist, b_timelist, time_start, mean_width)
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start, time_end, load_fs=True)
    # plot_fs_coeff(files_path_prefix, fs_timelist, borders, 0, time_end-time_start - delta, start_pic_num=time_start + delta, mean_width=mean_width)

    # plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, borders, delta, time_end-time_start, plot_step, start_pic_num=time_start + delta)
    # plot_f_coeff(files_path_prefix, f_timelist, borders, 0, time_end-time_start - delta, plot_step, start_pic_num=time_start + delta)

    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                  time_end, load_c=True)
    # plot_c_coeff(files_path_prefix, c_timelist, delta, len(c_timelist), 1, time_start + delta)

    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                  time_end, load_fs=True)
    # plot_fs_coeff(files_path_prefix, fs_timelist, borders, delta, len(fs_timelist), 1, time_start + delta, mean_width)


    # year = 1979
    # time_start = (datetime.datetime(year, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days + 1
    # time_end = (datetime.datetime(year, 12, 31, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    #
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                  time_end, load_a=True, load_b=True)
    # plot_mean_year_AB(files_path_prefix, time_start, time_end, a_timelist, b_timelist, borders, year)

    # ---------------------------------------------------------------------------------------
    # time_start = 1
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                  time_end, load_a=True, load_b=True)


    # raise ValueError
    # ---------------------------------------------------------------------------------------
    # create_video(files_path_prefix, files_path_prefix+'videos/A/', 'A_', 'a_daily', 10)
    # create_video(files_path_prefix, files_path_prefix+'videos/B/', 'B_', 'b_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/C/', 'C_', 'c_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/F/', 'F_', 'f_daily', 10)
    # # create_video(files_path_prefix, files_path_prefix + 'videos/Flux-corr/', 'FL_corr_', 'flux_correlation_weekly', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/FS/', 'FS_', 'FS_daily_mean_7', 10)
    # ---------------------------------------------------------------------------------------

    # count_correlation_fluxes(files_path_prefix, 0, 1829)
    # plot_flux_correlations(files_path_prefix, 0, 1829, step=7)


    # sensible_array, latent_array = load_prepare_fluxes('SENSIBLE_2019-2022.npy',
    #                                                    'LATENT_2019-2022.npy',
    #                                                    prepare=False)
    #
    # np.save(files_path_prefix + 'sensible_grouped_2019-2022.npy', sensible_array)
    # np.save(files_path_prefix + 'latent_grouped_2019-2022.npy', latent_array)
    # ---------------------------------------------------------------------------------------
    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_2019-2022.npy')
    # print(sensible_array.shape)
    # sens_scaled, quantiles = scale_to_bins(sensible_array, 1000)
    # np.save(files_path_prefix + 'sensible_grouped_2019-2022(scaled).npy', sens_scaled)
    # np.save(files_path_prefix + 'Quantiles/sensible_2019-2022(quantiles).npy', np.array(quantiles))
    #
    # latent_array = np.load(files_path_prefix + 'latent_grouped_2019-2022.npy')
    # latent_scaled, quantiles = scale_to_bins(latent_array, 1000)
    # np.save(files_path_prefix + 'latent_grouped_2019-2022(scaled).npy', latent_scaled)
    # np.save(files_path_prefix + 'Quantiles/latent_2019-2022(quantiles).npy', np.array(quantiles))

    # # ----------------------------------------------------------------------------------------------

    # n_components = 3
    # window_width = 30
    #
    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_1979-1989(scaled).npy')
    # latent_array = np.load(files_path_prefix + 'latent_grouped_1979-1989(scaled).npy')
    # sensible_array = sensible_array.astype(float)
    # latent_array = latent_array.astype(float)
    # time_start, time_end = 1, 1500
    # point_start, point_end = 39, 41

    # points = range(point_start, point_end)
    # for point in tqdm.tqdm(points):
    #     if mask[point]:
    #         sample_sens = deepcopy(sensible_array[point, time_start:time_end])
    #         sample_sens = np.diff(sample_sens)
    #         point_df = hybrid(sample_sens, window_width, n_components, 1)
    #         point_df.to_excel(files_path_prefix + f'Components/{flux_type}/raw/point_{point}.xlsx', index=False)
    #
    #         df = pd.read_excel(files_path_prefix + f'Components/{flux_type}/raw/point_{point}.xlsx')
    #         new_df, new_n_components = cluster_components(df, n_components, point, files_path_prefix, flux_type, True)
    #         new_df.to_excel(files_path_prefix + f'Components/{flux_type}/point_{point}.xlsx', index=False)
    #         plot_components(new_df, new_n_components, point, files_path_prefix, flux_type)
    #         plot_a_sigma(df, n_components, point, files_path_prefix, flux_type)

    # ----------------------------------------------------------------------------------------------
    # sensible_array = sensible_array.astype(float)
    # latent_array = latent_array.astype(float)
    #
    # for month in range(1, 2):
    #     days_delta1 = (datetime.datetime(1979, month, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    #     time_start = days_delta1
    #     days_delta2 = (datetime.datetime(1979 + month // 12, month % 12 + 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    #     time_end = days_delta2
    #
    #     sample_x = sensible_array[mask == 1, time_start: time_end].ravel()
        #     sample_y = latent_array[mask == 1, time_start:time_end].ravel()
        #
    #     # months_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
    #     #                 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    #
    #     draw_3d_hist(files_path_prefix, sample_x, sample_y, time_start, time_end, month)
    # ----------------------------------------------------------------------------------------------

    # count_abf_Kor_from_points(files_path_prefix, time_start, time_end, point_start, point_end)
    # count_Bel_Kor_difference(files_path_prefix, time_start, time_end, point_start, point_end)
    # plot_a_Kor(files_path_prefix, time_start, time_end, 0)
    # plot_a_diff(files_path_prefix, time_start, time_end, 0)

    # time_start, time_end = 1, 530
    # plot_difference_1d(files_path_prefix, time_start, time_end, point_start, point_end, n_components, window_width)

    # print(sensible_array[39])
    # plt.hist(sensible_array[39, 0:300], bins=20)
    # plt.savefig(files_path_prefix + f'Components/tmp/hist.png')
    # ----------------------------------------------------------------------------------------------
    # time_start = 0
    # time_end = days_delta1 + days_delta2 + days_delta3 + days_delta4 + days_delta5
    #
    # sensible_all = np.zeros((161*181, time_end-time_start))
    # latent_all = np.zeros((161*181, time_end-time_start))
    # start = 0
    # for years in ['1979-1989', '1989-1999', '1999-2009', '2009-2019', '2019-2022']:
    #     sens_part = np.load(files_path_prefix + f'sensible_grouped_{years}.npy')
    #     sensible_all[:, start:start + sens_part.shape[1]] = sens_part
    #
    #     lat_part = np.load(files_path_prefix + f'latent_grouped_{years}.npy')
    #     latent_all[:, start:start + lat_part.shape[1]] = lat_part
    #     start += sens_part.shape[1]
    #
    # np.save(files_path_prefix + 'sensible_all.npy', sensible_all)
    # np.save(files_path_prefix + 'latent_all.npy', latent_all)
    # ----------------------------------------------------------------------------------------------
    # year = 2021
    # time_start = (datetime.datetime(year, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    # time_end = (datetime.datetime(year, 12, 31, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    #
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                  time_end, load_a=True, load_b=True)
    # mean_days = 1
    # coeff_type = 'a'
    # extract_extreme(files_path_prefix, a_timelist, coeff_type, time_start, time_end, mean_days)
    # plot_extreme(files_path_prefix, coeff_type, time_start, time_end, mean_days)
    #
    # coeff_type = 'b'
    # extract_extreme(files_path_prefix, b_timelist, coeff_type, time_start, time_end, mean_days)
    # plot_extreme(files_path_prefix, coeff_type, time_start, time_end, mean_days)

    # mean_days = 365
    # extract_extreme(files_path_prefix, b_timelist, coeff_type, time_start, time_end, mean_days)
    # plot_extreme(files_path_prefix, coeff_type, time_start, time_end, mean_days)

    # sensible_all = np.load(files_path_prefix + 'sensible_all.npy')
    # latent_all = np.load(files_path_prefix + 'latent_all.npy')
    # check_conditions(files_path_prefix, time_start, time_end, sensible_all, latent_all, mask)
    # ----------------------------------------------------------------------------------------------