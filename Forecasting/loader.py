from torch.utils.data import Dataset
import os
import cv2
import sys
sys.path.append("..")
from config import cfg
import numpy as np
from datetime import datetime
import torch
from struct import unpack
from torch.utils.data import TensorDataset, DataLoader

# --------------------------------------------------------------------------------
# Days deltas
days_delta1 = (datetime(1989, 1, 1, 0, 0) - datetime(1979, 1, 1, 0, 0)).days
days_delta2 = (datetime(1999, 1, 1, 0, 0) - datetime(1989, 1, 1, 0, 0)).days
days_delta3 = (datetime(2009, 1, 1, 0, 0) - datetime(1999, 1, 1, 0, 0)).days
days_delta4 = (datetime(2019, 1, 1, 0, 0) - datetime(2009, 1, 1, 0, 0)).days
days_delta5 = (datetime(2024, 1, 1, 0, 0) - datetime(2019, 1, 1, 0, 0)).days
days_delta6 = (datetime(2024, 4, 28, 0, 0) - datetime(2019, 1, 1, 0, 0)).days
# ----------------------------------------------------------------------------------------------


def load_np_data(files_path_prefix, start_year, end_year):
    # load data
    flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    flux_array = np.diff(flux_array, axis=1)

    SST_array = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy')
    SST_array = np.diff(SST_array, axis=1)

    press_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    press_array = np.diff(press_array, axis=1)

    flux_array = flux_array.reshape((161, 181, -1))
    flux_array = flux_array[::2, ::2, :]
    SST_array = SST_array.reshape((161, 181, -1))
    SST_array = SST_array[::2, ::2, :]
    press_array = press_array.reshape((161, 181, -1))
    press_array = press_array[::2, ::2, :]

    return flux_array, SST_array, press_array


def load_mask(files_path_prefix):
    # Mask
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    mask = mask.reshape((161, 181))[::2, ::2]
    return mask


def count_offset(start_year):
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
    return end_year, offset

def create_dataloaders(files_path_prefix, start_year, end_year, cfg):
    x_train = torch.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{cfg.features_amount}.pt')
    y_train = torch.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_{cfg.features_amount}.pt')

    x_test = torch.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{cfg.features_amount}.pt')
    y_test = torch.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{cfg.features_amount}.pt')

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    return train_dataset, test_dataset, test_dataset

