import os.path
import time

import numpy as np

from EM_staff import *
from video import *
from data_processing import *
from ABC_coeff_counting import *
from func_estimation import *
from data_processing import load_prepare_fluxes

# Parameters
files_path_prefix = 'D://Data/OceanFull/'
flux_type = 'sensible'
# flux_type = 'latent'

# timesteps = 7320
timesteps = 1829

if __name__ == '__main__':
    # num_lost = []
    # for i in range(15598):
    #     if not os.path.exists(files_path_prefix + f'Coeff_data/{i+1}_A_sens.npy'):
    #         num_lost.append(i+1)
    #
    # print(num_lost)
    # print(len(num_lost))
    # raise ValueError

    # start = num_lost[0]
    # borders = []
    # sum_lost=0
    # for j in range(1, len(num_lost)):
    #     if start is None:
    #         start = num_lost[j]
    #
    #     # if (num_lost[j-1] == num_lost[j] - 1) and j != len(num_lost) - 1 and mask[j]:
    #     #     pass
    #     if not start is None and (num_lost[j-1] != num_lost[j] - 1):
    #         borders.append([start, num_lost[j-1]])
    #         sum_lost += num_lost[j-1] - start + 1
    #         start = num_lost[j]
    #
    # print(borders)
    # print(sum_lost)
    # raise ValueError
    # borders = [[14640, 14640]]
    #
    # maskfile = open(files_path_prefix + "mask", "rb")
    # binary_values = maskfile.read(29141)
    # maskfile.close()
    # mask = unpack('?' * 29141, binary_values)
    #
    # sensible_array = np.load(files_path_prefix + 'sensible_all.npy')
    # latent_array = np.load(files_path_prefix + 'latent_all.npy')
    #
    # for border in borders:
    #     start = border[0]-1
    #     end = border[1]
    #     count_abf_coefficients(files_path_prefix, mask, sensible_array[start-1:end+1], latent_array[start-1:end+1], time_start=0, time_end=end-start,
    #                            offset=start)
    # # offset = 14640 / 4 * 1
    # parallel_AB(4, 'SENSIBLE_1989-1999.npy', 'LATENT_1989-1999.npy', offset)

    # ---------------------------------------------------------------------------------------
    # maskfile = open(files_path_prefix + "mask", "rb")
    # binary_values = maskfile.read(29141)
    # maskfile.close()
    # mask = unpack('?' * 29141, binary_values)
    #
    # sensible_array = np.load(files_path_prefix + f'5years_sensible.npy')
    # latent_array = np.load(files_path_prefix + f'5years_latent.npy')
    #
    # sensible_array = sensible_array.astype(float)
    # latent_array = latent_array.astype(float)
    # sensible_array[np.logical_not(mask), :] = np.nan
    # latent_array[np.logical_not(mask)] = np.nan
    #
    # # mean by day = every 4 observations
    # pack_len = 4
    # sensible_array = block_reduce(sensible_array,
    #                               block_size=(1, pack_len),
    #                               func=np.mean, )
    # latent_array = block_reduce(latent_array,
    #                             block_size=(1, pack_len),
    #                             func=np.mean, )
    #
    # sensible_array = scale_to_bins(sensible_array)
    # latent_array = scale_to_bins(latent_array)

    # binary_to_array(files_path_prefix, "s79-21", 'SENSIBLE_2019-2021')
    # ---------------------------------------------------------------------------------------
    # Components determination part
    # sort_by_means(files_path_prefix, flux_type)
    # init_directory(files_path_prefix, flux_type)

    # dataframes_to_grids(files_path_prefix, flux_type, mask, components_amount, 100)
    # draw_frames(files_path_prefix, flux_type, mask, components_amount, timesteps=timesteps)
    # create_video(files_path_prefix, files_path_prefix+'videos/{flux_type}/tmp/', '', f'{flux_type}_5years_weekly', speed=30)
    # ---------------------------------------------------------------------------------------
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days

    time_start = days_delta1
    time_end = days_delta2 + 1
    plot_step = 1

    a_timelist, b_timelist, _, f_timelist, borders = load_ABCF(files_path_prefix, time_start + 1, time_end)
    plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, borders, 0, len(a_timelist), plot_step, start_pic_num=time_start)
    # plot_f_coeff(files_path_prefix, f_timelist, borders, time_start, time_end-1, plot_step)

    # count_c_coeff(files_path_prefix, a_timelist, b_timelist, time_start + 1, 14)
    # a_timelist, b_timelist, c_timelist, f_timelist, borders = load_ABCF(files_path_prefix, time_start + 1, time_end, load_c=True)
    # print('lol')
    # plot_c_coeff(files_path_prefix, c_timelist, time_start, len(c_timelist), plot_step)

    # create_video(files_path_prefix, files_path_prefix+'videos/tmp-coeff/', 'a_', 'a_daily', 10)
    # create_video(files_path_prefix, files_path_prefix+'videos/tmp-coeff/', 'b_', 'b_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/tmp-coeff/', 'f_', 'f_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/tmp-coeff/', 'C_', 'c_weekly', 10)

    # count_correlation_fluxes(files_path_prefix, 0, 1829)
    # plot_flux_correlations(files_path_prefix, 0, 1829, step=7)
    # create_video(files_path_prefix, files_path_prefix + 'videos/Flux-corr/', 'FL_corr_', 'flux_correlation_weekly', 10)

    # plot_a_by_flux(files_path_prefix, a_timelist, borders, sensible_array, latent_array, 0, len(a_timelist))