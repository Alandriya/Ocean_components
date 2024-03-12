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
from eigenvalues import count_eigenvalues_triplets


files_path_prefix = '/home/aosipova/EM_ocean/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("n_processes", help="Amount of processes to parallel run", type=int)
    args_cmd = parser.parse_args()

    cpu_count = args_cmd.n_processes

    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)

    # ---------------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2023, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    # count eigenvalues
    flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_2019-2023_grouped.npy')
    SST_array = np.load(files_path_prefix + f'SST/SST_2019-2023_grouped.npy')
    press_array = np.load(files_path_prefix + f'Pressure/PRESS_2019-2023_grouped.npy')
    t = 0
    offset = days_delta1 + days_delta2 + days_delta3 + days_delta4

    # flux_array = flux_array[:, t:t + 2]
    # SST_array = SST_array[:, t:t + 2]
    # press_array = press_array[:, t:t + 2]
    n_bins = 100

    # flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
    # SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)
    #
    # offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    # count_eigenvalues_parralel(files_path_prefix, cpu_count, flux_array, quantiles_flux, SST_array, quantiles_sst,
    #                        0, offset, ('Flux', 'SST'), n_bins)

    count_eigenvalues_triplets(files_path_prefix, flux_array, SST_array, press_array, 0, offset, n_bins, cpu_count)
