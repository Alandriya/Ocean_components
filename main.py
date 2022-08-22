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
from EM_count_coefficients import *
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
    # ---------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------
    # Components method and Belyaev-Korolev comparison
    points = plot_typical_points(files_path_prefix, mask)
    n_components = 3
    radius = 3

    ticks_by_day = 4
    step_ticks = 2
    window_width = ticks_by_day * 1

    # sensible_array = np.load(files_path_prefix + 'SENSIBLE_2019-2022.npy')
    # sensible_array = sensible_array[:, :365*ticks_by_day]
    # flux_type = 'sensible'
    # data_array = sensible_array

    latent_array = np.load(files_path_prefix + 'LATENT_2019-2022.npy')
    latent_array = latent_array[:, :365*ticks_by_day]
    flux_type = 'latent'
    data_array = latent_array

    time_start = 1
    time_end = data_array.shape[1]
    timedelta = days_delta1 + days_delta2 + days_delta3 + days_delta4

    for point in tqdm.tqdm([points[0]]):
        point_size = (radius * 2 + 1) ** 2
        point_bigger = [point]
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if mask[i*181 + j]:
                    point_bigger.append((point[0] + i, point[1]+j))
                else:
                    point_size -= 1

        sample = np.zeros((point_size, time_end - time_start - 1))
        for i in range(point_size):
            p = point_bigger[i]
            sample[i, :] = np.diff(data_array[p[0]*181 + p[1], time_start:time_end])

        # reshape
        sample = sample.transpose().flatten()

        # apply EM
        point_df = hybrid(sample, window_width * point_size, n_components, EM_steps=1, step=step_ticks*point_size)
        if not os.path.exists(files_path_prefix + f'Components/{flux_type}/raw'):
            os.mkdir(files_path_prefix + f'Components/{flux_type}/raw')
        point_df.to_excel(files_path_prefix + f'Components/{flux_type}/raw/point_({point[0]}, {point[1]}).xlsx', index=False)
        df = pd.read_excel(files_path_prefix + f'Components/{flux_type}/raw/point_({point[0]}, {point[1]}).xlsx')
        new_df, new_n_components = cluster_components(df, n_components, point, files_path_prefix, flux_type, True)
        if not os.path.exists(files_path_prefix + f'Components/{flux_type}/components-xlsx'):
            os.mkdir(files_path_prefix + f'Components/{flux_type}/components-xlsx')
        new_df.to_excel(files_path_prefix + f'Components/{flux_type}/components-xlsx/point_({point[0]}, {point[1]}).xlsx', index=False)
        plot_components(new_df, new_n_components, point, files_path_prefix, flux_type)
        plot_a_sigma(df, n_components, point, files_path_prefix, flux_type)


        count_Bel_Kor_difference(files_path_prefix,
                                 time_start,
                                 time_end//ticks_by_day,
                                 point_bigger,
                                 point_size,
                                 point,
                                 n_components,
                                 window_width,
                                 ticks_by_day,
                                 step_ticks,
                                 timedelta,
                                 flux_type)

        plot_difference_1d(files_path_prefix,
                           time_start,
                           time_end//ticks_by_day,
                           point,
                           window_width,
                           radius,
                           ticks_by_day,
                           step_ticks,
                           flux_type)
    # ----------------------------------------------------------------------------------------------
