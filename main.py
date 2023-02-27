import datetime
import os.path
import time

import numpy as np
import pandas as pd
import scipy.stats
import tqdm

from video import *
from plot_fluxes import *
from plot_Bel_coefficients import *
from data_processing import *
from ABCF_coeff_counting import *
from Kor_Bel_compare import *
from func_estimation import *
from data_processing import load_prepare_fluxes
from func_estimation import estimate_a_flux_by_months
from extreme_evolution import *
import cycler
from EM_hybrid import *
from fluxes_distribution import *
from SRS_count_coefficients import *
from copy import deepcopy
import shutil
import pyswarms


# Parameters
files_path_prefix = 'D://Data/OceanFull/'

# timesteps = 7320
timesteps = 1829
width = 181
height = 161

if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------
    # Mask
    borders = [[0, 1000]]
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    # ---------------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2022, 4, 2, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2022, 9, 30, 0, 0) - datetime.datetime(2022, 4, 2, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    # # Components method and Belyaev-Korolev comparison
    # points = plot_typical_points(files_path_prefix, mask)
    # n_components = 3
    # radius = 0
    #
    # # ticks_by_day = 1
    # # step_ticks = 1
    # # ticks_by_day = 4
    # # step_ticks = 2
    # ticks_by_day = 24
    # step_ticks = 1
    # window_width = ticks_by_day * 5
    # draw = False
    # coeff_type = 'A'
    #
    # time_start = 1
    # time_end = 90
    # # point = (15, 160)
    # # point = (45, 90)
    # # point = (15, 60)
    # point = (130, 90)
    # point_flat = point[0] * width + point[1]

    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_2019-2022.npy')
    # sensible_array = np.load(files_path_prefix + 'SENSIBLE_2019-2022.npy')

    # sensible_array = np.load(files_path_prefix + 'SENSIBLE_2019_part_extended.npy')
    # sensible_array = sensible_array[:, :90*ticks_by_day]
    # flux_type = 'sensible'
    # data_array = sensible_array
    # data_grouped = np.load(files_path_prefix + 'sensible_grouped_2019-2022.npy')

    # latent_array = np.load(files_path_prefix + 'LATENT_2019_part_extended.npy')
    # latent_array = latent_array[:, :90*ticks_by_day]
    # flux_type = 'latent'
    # data_array = latent_array
    # data_grouped = np.load(files_path_prefix + 'latent_grouped_2019-2022.npy')
    #
    # sensible_grouped = np.load(files_path_prefix + 'sensible_grouped_2019-2022.npy')
    # plot_Kor_Bel_histograms(files_path_prefix, time_start, time_end, data_array[point_flat], data_grouped[point_flat, :90],
    #                         flux_type, point)
    # raise ValueError

    # latent_array = np.load(files_path_prefix + 'LATENT_2019_part_extended.npy')
    # latent_array = latent_array[:, :90*ticks_by_day]
    # flux_type = 'latent'
    # data_array = latent_array

    # latent_array = np.load(files_path_prefix + 'LATENT_2019-2022.npy')
    # latent_array = latent_array[:, :365*ticks_by_day]
    # flux_type = 'latent'
    # data_array = latent_array

    # time_start = 1
    # time_end = data_array.shape[1] // ticks_by_day
    # timedelta = days_delta1 + days_delta2 + days_delta3 + days_delta4
    #
    # cpu_count = 1
    # points_borders = [2000, 4000]
    #
    # shutil.rmtree(files_path_prefix + f'Components/{flux_type}/plots')
    # os.mkdir(files_path_prefix + f'Components/{flux_type}/plots')
    #
    # parallel_Korolev(files_path_prefix, data_array, cpu_count, points_borders, time_start, time_end, timedelta,
    #                  flux_type, coeff_type, window_width, step_ticks, ticks_by_day, radius, n_components, False)

    # for b in range(0000, 14000, 2000):
    #     points_borders = [b, min(b+2000, 29141)]
    #     time_start_process = time.time()
    #     parallel_Korolev(files_path_prefix, data_array, cpu_count, points_borders, time_start, time_end, timedelta,
    #                      flux_type, coeff_type, window_width, step_ticks, ticks_by_day, radius, n_components)
    #     print(f'Time passed in seconds: {(time.time() - time_start_process)}')
    #
    # collect_EM(files_path_prefix, time_start, time_end-2, flux_type, coeff_type, 'Sum')
    # plot_map_Kor(files_path_prefix, time_start, time_end-2, flux_type, coeff_type, radius)

    # plot_typical_points_difference(files_path_prefix, mask, time_start, time_end, flux_type, coeff_type)

    # ----------------------------------------------------------------------------------------------
    # create_synthetic_data(files_path_prefix, time_start=0, time_end=100)
    # sensible = np.load(f'{files_path_prefix}/Synthetic/sensible_full.npy')
    # latent = np.load(f'{files_path_prefix}/Synthetic/latent_full.npy')
    a_array = np.load(f'{files_path_prefix}/Synthetic/A_full.npy')

    # plot_synthetic_flux(files_path_prefix, sensible, latent, time_start=0, time_end=100)
    # count_synthetic_Bel(files_path_prefix, sensible, latent, time_start=0, time_end=100)
    # multiply_synthetic_Korolev(files_path_prefix)

    sensible_extended = np.load(f'{files_path_prefix}/Synthetic/sensible_full_extended.npy')
    # plt.hist(sensible_extended[:100, 0, 0].flatten(), bins=30)
    # plt.show()
    # count_synthetic_Korolev(files_path_prefix, 'sensible', sensible_extended, 0, 100, 100)
    plot_Kor_Bel_compare(files_path_prefix, 0, 99, a_array[:, 0, :, :], 'sensible', 'A')