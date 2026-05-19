import datetime
import math
import os.path
from statistics import quantiles
from struct import unpack
import json
import numpy as np
from matplotlib.pyplot import autumn
from scipy.stats import mannwhitneyu
from skimage.metrics import structural_similarity as ssim
from torchgen.executorch.api.et_cpp import return_names

from Data_processing.data_processing import *
from Data_processing.data_processing import mean_blocks
from Data_processing.func_estimation import *
# from Plotting.plot_eigenvalues import plot_eigenvalues, plot_mean_year
# from Plotting.plot_extreme import *
# from extreme_evolution import *
# from ABCF_coeff_counting import *
from Eigenvalues.eigenvalues import *
# from Forecasting.main import end_year
from Plotting.plot_func_estimations import plot_ab_functional
# from Plotting.plot_Bel_coefficients import *
# from SRS_count_coefficients import *
# from Plotting.mean_year import *
from Plotting.video import *
from Coefficients.semiparametric import *
from Forecasting.utils import fix_random
from Plotting.plot_func_estimations import *
from Plotting.plot_coefficients import *
from scipy.integrate import trapezoid
from statsmodels.stats.multitest import multipletests
from scipy.special import gammainc, gamma, erf, erfcx, gammaincc
# files_path_prefix = '/home/aosipova/EM_ocean/'
files_path_prefix = 'D:/Nastya/Data/OceanFull/'

width = 181
height = 161
fix_random(2025)

if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------
    # Mask
    maskfile = open(files_path_prefix + "DATA/mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    mask = mask.reshape((height, width))
    # ---------------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2025, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2025, 11, 1, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
    # days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # days_delta7 = (datetime.datetime(2024, 11, 28, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    # # # estimate the functional a(X) and b(X) from data
    # data1_name = 'sensible'
    # data2_name = 'latent'
    # str_types = 'sensible-latent'
    #
    # # data1_name = 'flux'
    # # data2_name = 'pressure'
    # # str_types = 'flux-pressure'
    # season = 'full_10_years'
    # start_year = 1979
    # end_year = start_year + 10
    # coef_start = 1
    # coef_end = 3653
    # block_size = 5
    #
    # if data1_name == 'sensible':
    #     data1_array = np.load(files_path_prefix + f'Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
    # else:
    #     data1_array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    # data1_array = data1_array.transpose()
    # data1_array = data1_array.reshape((-1, height, width))
    # # data1_array = data1_array[1:]
    # # data1_array = np.diff(data1_array, axis=0)
    # # plot_hist(files_path_prefix, data1_name + f'_{start_year}-{end_year}', data1_array)
    # print(f'min 1: {np.nanmin(data1_array)}')
    # print(f'max 1: {np.nanmax(data1_array)}')
    #
    # if data2_name == 'latent':
    #     data2_array = np.load(files_path_prefix + f'Fluxes/latent_grouped_{start_year}-{end_year}.npy')
    # else:
    #     data2_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    # data2_array = data2_array.transpose()
    # data2_array = data2_array.reshape((-1, height, width))
    # # data2_array = np.diff(data2_array, axis=0)
    # # data2_array = data2_array[1:]
    # # plot_hist(files_path_prefix, data2_name + f'_{start_year}-{end_year}', data2_array)
    # print(f'min 2: {np.nanmin(data2_array)}')
    # print(f'max 2: {np.nanmax(data2_array)}')

    # a_array = np.load(files_path_prefix + f'Components/{str_types}/Semi_2d/A_{coef_start}-{coef_end}.npy')
    # b_array = np.load(files_path_prefix + f'Components/{str_types}/Semi_2d/B_{coef_start}-{coef_end}.npy')
    # b_array = b_array**2
    #
    # mask = mask.reshape((height, width))
    # data1_array[:, np.logical_not(mask)] = np.nan
    # data2_array[:, np.logical_not(mask)] = np.nan
    # a_array[:, np.logical_not(mask)] = np.nan
    # b_array[:, np.logical_not(mask)] = np.nan
    #
    #
    # # for year in range(start_year, start_year + 10):
    # for year in [1979]:
    #     print(year)
    #     if season == 'year':
    #         start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=start_year, month=1,
    #                                                                                   day=1)).days
    #         end = (datetime.datetime(year=year, month=12, day=31) - datetime.datetime(year=start_year, month=1,
    #                                                                                   day=1)).days
    #     elif season == 'spring':
    #         start = (datetime.datetime(year=year, month=3, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
    #         end = (datetime.datetime(year=year, month=6, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
    #     elif season == 'summer':
    #         start = (datetime.datetime(year=year, month=6, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
    #         end = (datetime.datetime(year=year, month=9, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
    #     elif season == 'autumn':
    #         start = (datetime.datetime(year=year, month=9, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
    #         end = (datetime.datetime(year=year, month=12, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
    #     elif season == 'winter':
    #         start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
    #         end = (datetime.datetime(year=year, month=3, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
    #     else:
    #         start = 0
    #         end = 3650
    #     data1_small = mean_blocks(data1_array[start:end], mask, block_size)
    #     data2_small = mean_blocks(data2_array[start:end], mask, block_size)
    #     a_small = mean_blocks(a_array[start:end], mask, block_size)
    #     b_small = mean_blocks(b_array[start:end], mask, block_size)
    #
    #     quantiles1, quantiles2, a_grouped, b_grouped, x1_full, x2_full, a1_full, a2_full, b11_full, b22_full, = (
    #         estimate_A_B_2d(data1_small, data2_small, a_small, b_small, quantiles_amount=260))
    #
    #     data = [quantiles1, quantiles2, a_grouped, np.log(b_grouped), x1_full, x2_full, a1_full, a2_full, np.log(b11_full), np.log(b22_full)]
    #     # plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, season, year, scatter=True)
    #     plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, season, year, scatter=False)
    #
    # raise ValueError
    # ----------------------------------------------------------------------------------------------
    # plot stationary distribution 1d
    def kl_coeff(x, n, l):
        if n == 0:
            return 1 / l
        elif n == 1:
            return x / l - 1 / (l ** 2)
        elif n == 2:
            return x ** 2 / l - (2 * x) / (l ** 2) + 2 / (l ** 3)
        elif n == 3:
            return x ** 3 / l - (3 * x ** 2) / (l ** 2) + (6 * x) / (l ** 3) - 6 / (l ** 4)
        elif n == 4:
            return x ** 4 / l - (4 * x ** 3) / (l ** 2) + (12 * x ** 2) / (l ** 3) - (24 * x) / (l ** 4) + 24 / (l ** 5)
        else:
            raise ValueError("n must be 0..4")


    def Phi(x, l, k):
        k0, k1, k2, k3, k4 = k
        return np.exp(l * x) * (
                k4 * kl_coeff(x, 4, l)
                + k3 * kl_coeff(x, 3, l)
                + k2 * kl_coeff(x, 2, l)
                + k1 * kl_coeff(x, 1, l)
                + k0 * kl_coeff(x, 0, l)
        )


    def Psi_minus(x, c, k):
        tmp = 0.0
        for n in range(5):
            tmp += ((-1) ** n) * k[n] * c ** (-2 * n - 2) * gammaincc(2 * n + 2, c * np.sqrt(-x)) * gamma(2 * n + 2)
        return 2.0 * tmp


    def Psi_plus(x, c, k):
        tmp = 0.0
        for n in range(5):
            tmp += k[n] * c ** (-2 * n - 2) * gammaincc(2 * n + 2, c * np.sqrt(x)) * gamma(2 * n + 2)
        return -2.0 * tmp


    def I(x, k):
        dL = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
        dC = c3
        dR = c3 + (c2 - c4) * np.sqrt(abs(x0))

        prefL = 2 * np.exp(-dL)
        prefC = 2 * np.exp(-dC)
        prefR = 2 * np.exp(-dR)

        if x0 < 0:
            if x < x1:
                return prefL * Phi(x, c1, k) + const1
            elif x1 <= x < x0:
                return prefC * Psi_minus(x, c2, k) + const2
            elif x0 <= x < 0:
                return prefR * Psi_minus(x, c4, k) + const3
            else:
                return prefR * Psi_plus(x, c4, k) + const4

        else:
            if x < x1:
                return prefL * Phi(x, c1, k) + const1
            elif x1 <= x < 0:
                return prefC * Psi_minus(x, c2, k) + const2
            elif 0 <= x < x0:
                return prefC * Psi_plus(x, c2, k) + const3
            else:
                return prefR * Psi_plus(x, c4, k) + const4


    def log_b2(x):
        dL = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
        dC = c3
        dR = c3 + (c2 - c4) * np.sqrt(abs(x0))

        if x < x1:
            return c1 * abs(x) + dL
        elif x1 <= x < x0:
            return c2 * np.sqrt(abs(x)) + dC
        else:
            return c4 * np.sqrt(abs(x)) + dR

    def prob_stationary_sensible(x):
        if x > x_max:
            return np.exp(I(x_max, k) - log_b2(x))/ z
        return np.exp(I(x, k) - log_b2(x))/ z

    def prob_stationary_latent(x):
        if x < x_min:
            return np.exp(I(x_min, k) - log_b2(x))/ z
        return np.exp(I(x, k) - log_b2(x)) / z


    x0 = -3.647613525390625
    x1 = -40
    k4 = 1.026 * 10**(-7)
    k3 = 2.250 * 10**(-5)
    k2 = 5.919 * 10**(-4)
    k1 = -1.401 * 10 **(-1)
    k0 = 1.837 * 1
    k = [k0, k1, k2, k3, k4]
    x_min = -1118.5375366210938
    # x_max = 259.7684326171875
    x_max = 100

    c1 = 1.255 * 10**(-2)
    c2 = 3.367 * 10**(-1)
    c3 = 5.536
    c4 = 3.294 * 10**(-1)

    dL = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
    dC = c3
    dR = c3 + (c2 - c4) * np.sqrt(abs(x0))

    prefL = 2 * np.exp(-dL)
    prefC = 2 * np.exp(-dC)
    prefR = 2 * np.exp(-dR)

    const1 = -prefL * Phi(x1, c1, k)
    const2 = const1 + prefL * Phi(x1, c1, k) - prefC * Psi_minus(x1, c2, k)
    const3 = const2 + prefC * Psi_minus(x0, c2, k) - prefR * Psi_minus(x0, c4, k)
    const4 = const3 + prefR * Psi_minus(0, c4, k) - prefR * Psi_plus(0, c4, k)

    # print([const1, const2, const3, const4])
    z = 1
    z = trapezoid([prob_stationary_sensible(x) for x in np.linspace(-1500, 500, 2500)])
    print(f'Sensible z: {z:.3e}')
    plot_prob_1d(files_path_prefix, 'sensible', prob_stationary_sensible,  np.linspace(-300, 300, 2500))
    # plot_prob_and_hist(files_path_prefix, 'sensible', prob_stationary_sensible, np.linspace(-300, 300, 2500), data1_array[:365])

    # -----------------------------------------------------------------------------------
    x0 = -2.8423638983087227
    x1 = -40

    k4 = -1.230 * 10**(-9)
    k3 = 2.755 * 10**(-6)
    k2 = 1.230 * 10**(-3)
    k1 = 5.313 * 10 **(-2)
    k0 = -4.376 * 1
    k = [k0, k1, k2, k3, k4]

    c1 = 5.257 * 10**(-3)
    c2 = 1.951 * 10**(-1)
    c3 = 6.174
    c4 = 2.852 * 10**(-1)
    # x_min = -1225.4668579101562
    x_min = -300
    x_max = 225.44720458984375

    dL = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
    dC = c3
    dR = c3 + (c2 - c4) * np.sqrt(abs(x0))

    prefL = 2 * np.exp(-dL)
    prefC = 2 * np.exp(-dC)
    prefR = 2 * np.exp(-dR)

    const1 = -prefL * Phi(x1, c1, k)
    const2 = const1 + prefL * Phi(x1, c1, k) - prefC * Psi_minus(x1, c2, k)
    const3 = const2 + prefC * Psi_minus(x0, c2, k) - prefR * Psi_minus(x0, c4, k)
    const4 = const3 + prefR * Psi_minus(0, c4, k) - prefR * Psi_plus(0, c4, k)
    # print([const1, const2, const3, const4])
    z = 1
    z = trapezoid([prob_stationary_latent(x) for x in np.linspace(-1500, 500, 2500)])
    print(f'Latent z: {z:.3e}')
    plot_prob_1d(files_path_prefix, 'latent', prob_stationary_latent, np.linspace(-800, 500, 2500))
    # plot_prob_and_hist(files_path_prefix, 'latent', prob_stationary_latent, np.linspace(-800, 500, 2500), data2_array[:365])

