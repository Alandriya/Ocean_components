# import cv2
import sys

from torch.utils.data import Dataset

sys.path.append("..")
import numpy as np
import torch
from struct import unpack


def scale_to_bins(arr, bins=100):
    quantiles = list(np.nanquantile(arr, np.linspace(0, 1, bins, endpoint=False)))

    arr_scaled = np.zeros_like(arr, dtype=float)
    # arr_scaled[np.isnan(arr)] = 0
    # for j in tqdm.tqdm(range(bins - 1)):
    for j in range(bins - 1):
        arr_scaled[np.where((np.logical_not(np.isnan(arr))) & (quantiles[j] <= arr) & (arr < quantiles[j + 1]))] = \
            (j + 1) / bins
            # (quantiles[j] + quantiles[j + 1]) / 2

    quantiles += [np.nanmax(arr)]
    return arr_scaled, quantiles


def load_np_data(files_path_prefix, start_year, end_year):
    # load data
    flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    # flux_array = np.diff(flux_array, axis=1)

    SST_array = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy')
    # SST_array = np.diff(SST_array, axis=1)

    press_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    # press_array = np.diff(press_array, axis=1)

    flux_array = flux_array.reshape((161, 181, -1))
    flux_array = flux_array[::2, ::2, :]
    SST_array = SST_array.reshape((161, 181, -1))
    SST_array = SST_array[::2, ::2, :]
    press_array = press_array.reshape((161, 181, -1))
    press_array = press_array[::2, ::2, :]

    return flux_array, SST_array, press_array


def load_mask(files_path_prefix):
    # Mask
    maskfile = open(files_path_prefix + "DATA/mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    mask = mask.reshape((161, 181))[::2, ::2]
    return mask


class Data(Dataset):
    def __init__(self, cfg, start_idx, end_idx, weights):
        super().__init__()
        self.features_amount = cfg.features_amount
        self.in_len = cfg.in_len
        self.out_len = cfg.out_len
        self.height = cfg.height
        self.width = cfg.width
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.weights = weights

        self.flux_array = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_grouped_diff.npy')[start_idx:end_idx]
        # self.sst_array = np.load(cfg.root_path + f'DATA/SST_1979-2025_grouped_diff.npy')[start_idx:end_idx]
        # self.press_array = np.load(cfg.root_path + f'DATA/PRESS_1979-2025_grouped_diff.npy')[start_idx:end_idx]
        np.nan_to_num(self.flux_array, copy=False)
        self.flux_array *= self.weights[0]
        # np.nan_to_num(self.sst_array, copy=False)
        # np.nan_to_num(self.press_array, copy=False)

        self.flux_quantiles = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_diff_quantiles.npy')
        # self.sst_quantiles = np.load(cfg.root_path + f'DATA/SST_1979-2025_diff_quantiles.npy')
        # self.press_quantiles = np.load(cfg.root_path + f'DATA/PRESS_1979-2025_diff_quantiles.npy')

        # self.flux_mean_year = np.load(cfg.root_path + f'DATA/FLUX_mean_year_diff.npy')
        # self.sst_mean_year = np.load(cfg.root_path + f'DATA/SST_mean_year_diff.npy')
        # self.press_mean_year = np.load(cfg.root_path + f'DATA/PRESS_mean_year_diff.npy')
        # np.nan_to_num(self.flux_mean_year, copy=False)
        # np.nan_to_num(self.sst_mean_year, copy=False)
        # np.nan_to_num(self.press_mean_year, copy=False)

        # self.flux_scaled = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_grouped_diff_scaled.npy')[start_idx:end_idx]
        # self.sst_scaled = np.load(cfg.root_path + f'DATA/SST_1979-2025_grouped_diff_scaled.npy')[start_idx:end_idx]
        # self.press_scaled = np.load(cfg.root_path + f'DATA/PRESS_1979-2025_grouped_diff_scaled.npy')[start_idx:end_idx]

        self.eigen_flux = None
        # self.eigen_sst = None
        # self.eigen_press = None

        # self.A_flux = None
        # self.A_sst = None
        # self.A_press =

        self.A_flux = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_a_coeff.npy')[start_idx:end_idx]
        np.nan_to_num(self.A_flux, copy=False)
        self.A_flux *= self.weights[1]
        self.B_flux = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_b_coeff.npy')[start_idx:end_idx]
        np.nan_to_num(self.B_flux, copy=False)
        self.B_flux *= self.weights[2]

    def __getitem__(self, index):
        sample = np.zeros((self.in_len + self.out_len, self.features_amount, self.height, self.width), dtype=float)
        for day in range(self.in_len + self.out_len):
            sample[day, 0] = self.flux_array[index + day]
            # sample[day, 1] = self.sst_array[index + day]
            # sample[day, 2] = self.press_array[index + day]


        # A and eigens are used from future to go into loss

        if self.features_amount == 3:
            for day in range(self.in_len + self.out_len):
                sample[day, 1] = self.A_flux[index + day]
                sample[day, 2] = self.B_flux[index + day]

        if self.features_amount >= 6:
            for day in range(self.in_len + self.out_len):
                sample[day, 3] = self.A_flux[index + day]
                sample[day, 4] = self.A_sst[index + day]
                sample[day, 5] = self.A_press[index + day]

        if self.features_amount == 9:
            for day in range(self.in_len + self.out_len):
                sample[day, 6] = self.B_flux[index + day]
                sample[day, 7] = self.B_sst[index + day]
                sample[day, 8] = self.B_press[index + day]

        if self.features_amount >= 12:
            for day in range(self.in_len + self.out_len):
                sample[day, 6] = self.eigen_flux[index + day]
                sample[day, 7] = self.eigen_sst[index + day]
                sample[day, 8] = self.eigen_press[index + day]

                sample[day, 9] = self.eigen_flux_sst[index + day]
                sample[day, 10] = self.eigen_flux_press[index + day]
                sample[day, 11] = self.eigen_sst_press[index + day]

        if self.features_amount >= 18:
            for day in range(self.in_len + self.out_len):
                sample[day, 12] = self.eigenvalues[index + day, 0]
                sample[day, 13] = self.eigenvalues[index + day, 1]
                sample[day, 14] = self.eigenvalues[index + day, 2]
                sample[day, 15] = self.eigenvalues[index + day, 3]
                sample[day, 16] = self.eigenvalues[index + day, 4]
                sample[day, 17] = self.eigenvalues[index + day, 5]

        # elif self.features_amount == 9:
        #     for day in range(self.in_len + self.out_len):
        #         sample[day, 3] = self.eigen_flux[index + day]
        #         sample[day, 4] = self.eigen_sst[index + day]
        #         sample[day, 5] = self.eigen_press[index + day]
        #
        #         sample[day, 6] = self.A_flux[index + day]
        #         sample[day, 7] = self.A_sst[index + day]
        #         sample[day, 8] = self.A_press[index + day]
        # if self.features_amount == 9:
            # for day in range(self.in_len):
            #     day_dt = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=index+cfg.in_len + day)
            #     day_shift = (day_dt - datetime.datetime(day_dt.year, 1, 1)).days
            #     sample[day, 3] = self.flux_mean_year[day_shift % 365]
            #     sample[day, 4] = self.sst_mean_year[day_shift % 365]
            #     sample[day, 5] = self.press_mean_year[day_shift % 365]
            #
            #     if not day_dt.year == 1979:
            #         try:
            #             offset = (datetime.datetime(day_dt.year-1, day_dt.month, day_dt.day) - datetime.datetime(1979, 1, 1)).days
            #         except ValueError:
            #             offset = (datetime.datetime(day_dt.year - 1, day_dt.month, day_dt.day-1) - datetime.datetime(1979,1, 1)).days
            #         sample[day, 6] = self.flux_array[offset]
            #         sample[day, 7] = self.sst_array[offset]
            #         sample[day, 8] = self.press_array[offset]


        return torch.from_numpy(sample).float()  # S*C*H*W

    def __len__(self):
        return self.flux_array.shape[0] - self.in_len - self.out_len

class Data_1d(Dataset):
    def __init__(self, cfg, start_idx, end_idx, point):
        super().__init__()
        self.point = point
        self.features_amount = cfg.features_amount
        self.in_len = cfg.in_len
        self.out_len = cfg.out_len
        self.height = cfg.height
        self.width = cfg.width
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.flux_array = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_grouped_diff.npy')[start_idx:end_idx, point[0], point[1]]
        np.nan_to_num(self.flux_array, copy=False)


        self.flux_quantiles = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_diff_quantiles.npy')

        # self.flux_mean_year = np.load(cfg.root_path + f'DATA/FLUX_mean_year_diff.npy')
        # np.nan_to_num(self.flux_mean_year, copy=False)
        # self.flux_scaled = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_grouped_diff_scaled.npy')[start_idx:end_idx]

        self.eigen_flux = None


        self.A_flux = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_a_coeff.npy')[start_idx:end_idx, point[0], point[1]]
        np.nan_to_num(self.A_flux, copy=False)
        self.B_flux = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_b_coeff.npy')[start_idx:end_idx, point[0], point[1]]
        np.nan_to_num(self.B_flux, copy=False)

    def __getitem__(self, index):
        sample = np.zeros((self.in_len + self.out_len, self.features_amount), dtype=float)
        for day in range(self.in_len + self.out_len):
            sample[day, 0] = self.flux_array[index + day]
        return torch.from_numpy(sample).float()  # S*C

    def __len__(self):
        return self.flux_array.shape[0] - self.in_len - self.out_len