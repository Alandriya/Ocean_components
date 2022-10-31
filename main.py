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


# Parameters
files_path_prefix = 'D://Data/OceanFull/'

# timesteps = 7320
timesteps = 1829

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
    # ---------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------
    # Components method and Belyaev-Korolev comparison
    points = plot_typical_points(files_path_prefix, mask)
    n_components = 5
    radius = 1

    # ticks_by_day = 1
    # step_ticks = 1
    # ticks_by_day = 4
    # step_ticks = 2
    ticks_by_day = 24
    step_ticks = 2
    window_width = ticks_by_day * 1
    draw = False
    coeff_type = 'B'

    # sensible_array = np.load(files_path_prefix + 'sensible_grouped_2019-2022.npy')
    # sensible_array = np.load(files_path_prefix + 'SENSIBLE_2019-2022.npy')

    sensible_array = np.load(files_path_prefix + 'SENSIBLE_2019_part_extended.npy')
    sensible_array = sensible_array[:, :90*ticks_by_day]
    flux_type = 'sensible'
    data_array = sensible_array

    # latent_array = np.load(files_path_prefix + 'LATENT_2019_part_extended.npy')
    # latent_array = latent_array[:, :90*ticks_by_day]
    # flux_type = 'latent'
    # data_array = latent_array

    # latent_array = np.load(files_path_prefix + 'LATENT_2019-2022.npy')
    # latent_array = latent_array[:, :365*ticks_by_day]
    # flux_type = 'latent'
    # data_array = latent_array

    time_start = 1
    time_end = data_array.shape[1] // ticks_by_day
    timedelta = days_delta1 + days_delta2 + days_delta3 + days_delta4

    cpu_count = 1
    points_borders = [2000, 4000]

    # shutil.rmtree(files_path_prefix + f'Components/{flux_type}/plots')
    # os.mkdir(files_path_prefix + f'Components/{flux_type}/plots')

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
