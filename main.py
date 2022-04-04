import os.path
import time

import numpy as np
import pandas as pd
import tqdm

from video import *
from data_processing import *
from ABCF_coeff_counting import *
from func_estimation import *
from data_processing import load_prepare_fluxes
from func_estimation import estimate_a_flux_by_months
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

    # ---------------------------------------------------------------------------------------
    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_1979-1989.npy')
    # print(sensible_array.shape)
    # sens_scaled, quantiles = scale_to_bins(sensible_array, 1000)
    # np.save(files_path_prefix + 'sensible_grouped_1979-1989(scaled).npy', sens_scaled)
    # np.save(files_path_prefix + 'Quantiles/sensible_1979-1989(quantiles).npy', np.array(quantiles))
    #
    # latent_array = np.load(files_path_prefix + 'latent_grouped_1979-1989.npy')
    # latent_scaled, quantiles = scale_to_bins(latent_array, 1000)
    # np.save(files_path_prefix + 'latent_grouped_1979-1989(scaled).npy', latent_scaled)
    # np.save(files_path_prefix + 'Quantiles/latent_1979-1989(quantiles).npy', np.array(quantiles))
    # ---------------------------------------------------------------------------------------
    # raise ValueError
    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_1979-1989(scaled).npy')
    # latent_array = np.load(files_path_prefix + 'latent_grouped_1979-1989(scaled).npy')
    # sensible_array = sensible_array.astype(float)
    # sensible_array = np.diff(sensible_array)
    # latent_array = latent_array.astype(float)
    # latent_array = np.diff(latent_array)
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
    # binary_to_array(files_path_prefix, "s79-21", 'SENSIBLE_2019-2021')
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
    # days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    # days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    # # # days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days

    print(days_delta1)
    # time_start = 1
    # time_end = 914
    # mean_width = 7
    #
    # plot_step = 1
    # delta = 0

    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start, time_end, load_a=True, load_b=True)
    # count_f_separate_coeff(files_path_prefix, a_timelist, b_timelist, time_start, mean_width)
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start, time_end, load_fs=True)
    # plot_fs_coeff(files_path_prefix, fs_timelist, borders, 0, time_end-time_start - delta, start_pic_num=time_start + delta, mean_width=mean_width)

    # plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, borders, 0, time_end-time_start - delta, plot_step, start_pic_num=time_start + delta)
    # plot_f_coeff(files_path_prefix, f_timelist, borders, 0, time_end-time_start - delta, plot_step, start_pic_num=time_start + delta)
    # raise ValueError
    # ---------------------------------------------------------------------------------------
    # create_video(files_path_prefix, files_path_prefix+'videos/A/', 'A_', 'a_daily_FAST', 3)
    # create_video(files_path_prefix, files_path_prefix+'videos/B/', 'B_', 'b_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/C/', 'C_', 'c_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/Flux-corr/', 'FL_corr_', 'flux_correlation_weekly', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/FS/', 'FS_', 'FS_daily_mean_7', 4)
    # ---------------------------------------------------------------------------------------

    # count_correlation_fluxes(files_path_prefix, 0, 1829)
    # plot_flux_correlations(files_path_prefix, 0, 1829, step=7)

    # # ---------------------------------------------------------------------------------------
    # # create timeseries for students
    # points = plot_typical_points(files_path_prefix, mask)
    #
    # a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix, time_start,
    #                                                                                  time_end, load_a=True, load_b=True)
    # for p in points:
    #     ts_a = list()
    #     ts_b = list()
    #     for t in range(0, time_end - time_start):
    #         ts_a.append(np.array(a_timelist[t])[:, p[0], p[1]])
    #         ts_b.append(np.array(b_timelist[t])[:, p[0], p[1]])
    #
    #     ts_a = np.array(ts_a)
    #     ts_b = np.array(ts_b)
    #     print(ts_b.shape)
    #
    #     np.save(files_path_prefix + f'/TimeSeries/A_({p[0]}, {p[1]}).npy', ts_a)
    #     np.save(files_path_prefix + f'/TimeSeries/B_({p[0]}, {p[1]}).npy', ts_b)


    sensible_array, latent_array = load_prepare_fluxes('SENSIBLE_2019-2021.npy',
                                                       'LATENT_2019-2021.npy',
                                                       prepare=False)

    np.save(files_path_prefix + 'sensible_grouped_2019-2021.npy', sensible_array)
    np.save(files_path_prefix + 'latent_grouped_2019-2021.npy', latent_array)

    # # ----------------------------------------------------------------------------------------------
    # n_components = 3
    # window_width = 60
    #
    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_1979-1989(scaled).npy')
    # latent_array = np.load(files_path_prefix + 'latent_grouped_1979-1989(scaled).npy')
    # sensible_array = sensible_array.astype(float)
    # latent_array = latent_array.astype(float)
    # time_start, time_end = 1, 1500
    # point_start, point_end = 39, 41
    #
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