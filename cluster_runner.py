import argparse
import numpy as np
import pandas as pd
from struct import unpack
import os
from scipy.linalg import sqrtm
from scipy.linalg.interpolative import estimate_spectral_norm
from numpy.linalg import norm
from multiprocessing import Pool
import datetime
# from eigenvalues import count_eigenvalues_parralel, scale_to_bins
from eigenvalues import count_eigenvalues_triplets, count_mean_year, get_trends
from Plotting.video import create_video
import copy

files_path_prefix = '/home/aosipova/EM_ocean/'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("start_year", help="Amount of processes to parallel run", type=int)
    parser.add_argument("t_start", type=int)
    args_cmd = parser.parse_args()

    start_year = args_cmd.start_year
    t_start = args_cmd.t_start

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
    days_delta5 = (datetime.datetime(2023, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    print(f'start year {start_year}')
    end_year = start_year + 10

    if start_year == 1979:
        offset = 0
    elif start_year == 1989:
        offset = days_delta1
    elif start_year == 1999:
        offset = days_delta1 + days_delta2
    elif start_year == 2009:
        offset = days_delta1 + days_delta2 + days_delta3
    else:
        offset = days_delta1 + days_delta2 + days_delta3 + days_delta4

    if start_year == 2019:
        end_year = 2025
    else:
        end_year = start_year + 10

    flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    SST_array = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy')
    press_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    t = t_start

    n_bins = 100
    offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    count_eigenvalues_triplets(files_path_prefix, 0, flux_array, SST_array, press_array, mask, offset, n_bins)