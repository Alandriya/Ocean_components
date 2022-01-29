import os.path
import time

import numpy as np
import pandas as pd

from video import *
from data_processing import *
from ABCF_coeff_counting import *
from func_estimation import *
from data_processing import load_prepare_fluxes
from func_estimation import estimate_a_flux_by_months
import cycler


# Parameters
files_path_prefix = 'D://Data/OceanFull/'
flux_type = 'sensible'
# flux_type = 'latent'

# timesteps = 7320
timesteps = 1829

if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------
    # borders = [[7318, 7325]]
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    #
    # sensible_array = np.load(files_path_prefix + 'sensible_all.npy')
    # latent_array = np.load(files_path_prefix + 'latent_all.npy')
    #
    # for border in borders:
    #     start = border[0]-1
    #     end = border[1]
    #
    #     count_abf_coefficients(files_path_prefix, mask, sensible_array[:, start-1:end+1], latent_array[:, start-1:end+1], time_start=0, time_end=end-start,
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
    # estimate_flux_by_months(files_path_prefix, 1)
    start_month = 1
    points = plot_typical_points(files_path_prefix, np.array(mask, dtype=int))
    sensible_array, latent_array = load_prepare_fluxes('SENSIBLE_1979-1989.npy', 'LATENT_1979-1989.npy', False)
    for p in points:
        estimate_flux(files_path_prefix, sensible_array, latent_array, start_month, p)
    # ---------------------------------------------------------------------------------------

    # days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    # days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    # days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    # # days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    #
    # time_start = 10900
    # time_end = 11000

    # plot_step = 1
    # delta = 3474
    #
    # a_timelist, b_timelist, c_timelist, f_timelist, borders = load_ABCF(files_path_prefix, time_start, time_end, load_c=True)
    # plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, borders, 0, time_end-time_start - delta, plot_step, start_pic_num=time_start + delta)
    # plot_f_coeff(files_path_prefix, f_timelist, borders, 0, time_end-time_start - delta, plot_step, start_pic_num=time_start + delta)
    # plot_c_coeff(files_path_prefix, c_timelist, 0, len(c_timelist), step=1, start_pic_num=time_start)
    # ---------------------------------------------------------------------------------------

    # create_video(files_path_prefix, files_path_prefix+'videos/tmp-coeff/', 'A_', 'a_daily', 10)
    # create_video(files_path_prefix, files_path_prefix+'videos/tmp-coeff/', 'B_', 'b_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/tmp-coeff/', 'F_', 'f_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/tmp-coeff/', 'C_', 'c_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/Flux-corr/', 'FL_corr_', 'flux_correlation_weekly', 10)
    # ---------------------------------------------------------------------------------------

    # count_correlation_fluxes(files_path_prefix, 0, 1829)
    # plot_flux_correlations(files_path_prefix, 0, 1829, step=7)
