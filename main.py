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
    # # estimate the functional a(X) and b(X) from data
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
    #
    # if data1_name == 'sensible':
    #     data1_array = np.load(files_path_prefix + f'Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
    # else:
    #     data1_array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    # data1_array = data1_array.transpose()
    # data1_array = data1_array.reshape((-1, height, width))
    # # data1_array = data1_array[1:]
    # # data1_array = np.diff(data1_array, axis=0)
    # plot_hist(files_path_prefix, data1_name + f'_{start_year}-{end_year}', data1_array)
    #
    # if data2_name == 'latent':
    #     data2_array = np.load(files_path_prefix + f'Fluxes/latent_grouped_{start_year}-{end_year}.npy')
    # else:
    #     data2_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    # data2_array = data2_array.transpose()
    # data2_array = data2_array.reshape((-1, height, width))
    # # data2_array = np.diff(data2_array, axis=0)
    # # data2_array = data2_array[1:]
    # plot_hist(files_path_prefix, data2_name + f'_{start_year}-{end_year}', data2_array)
    #
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
    #     else:
    #         start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
    #         end = (datetime.datetime(year=year, month=3, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
    #
    #
    #     start = 0
    #     end = 3650
    #     data1_small = mean_blocks(data1_array[start:end], mask)
    #     data2_small = mean_blocks(data2_array[start:end], mask)
    #     a_small = mean_blocks(a_array[start:end], mask)
    #     b_small = mean_blocks(b_array[start:end], mask)
    #
    #     quantiles1, quantiles2, a_grouped, b_grouped, x1_full, x2_full, a1_full, a2_full, b11_full, b22_full, = (
    #         estimate_A_B_2d(data1_small, data2_small, a_small, b_small, quantiles_amount=390))
    #
    #     data = [quantiles1, quantiles2, a_grouped, np.log(b_grouped), x1_full, x2_full, a1_full, a2_full, np.log(b11_full), np.log(b22_full)]
    #     # plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, season, year, scatter=True)
    #     plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, season, year, scatter=False)
    # raise ValueError

    # b_hist = np.load(files_path_prefix + f'Mean_year/{data1_name}-{data2_name}_B_heatmap.npy')
    # # print(b_hist.shape)
    # ql1 = np.load(files_path_prefix + f'Mean_year/{data1_name}-{data2_name}_q1_little.npy')
    # ql2 = np.load(files_path_prefix + f'Mean_year/{data1_name}-{data2_name}_q2_little.npy')
    # plot_heatmap(files_path_prefix, data1_name, data2_name, ql1, ql2, b_hist,)
    # print(ql1)
    # print(len(ql1))
    # print(ql2)
    # print(len(ql2))
    # print(b_hist)
    # with open(files_path_prefix + f'Mean_year/{data1_name}-{data2_name}_B_heatmap.txt', "w") as f:
    #     for i in range(len(b_hist)):
    #         f.write(str(b_hist[i]))
    # ----------------------------------------------------------------------------------------------
    # # plot stationary distribution 1d
    def kl_coeff(x, n, l):
        if n == 0:
            return 1/l
        elif n == 1:
            return x/l - 1/(l**2)
        elif n == 2:
            return x**2 / l - (2*x)/(l**2) + 2/(l**3)
        elif n==3:
            return x**3 / l - (3 * x**2)/(l**2) + (6*x) / (l**3) - 6/(l**4)
        elif n == 4:
            return x**4 / l - (4*x**3)/(l**2) + (12 * x**2) / (l**3) - (24*x) / (l**4) + 24/(l**5)
        else:
            return 0

    def Phi(x, l):
        return np.exp(l*x) * (k4 * kl_coeff(x, 4, l) + k3 * kl_coeff(x, 3, l) + k2 * kl_coeff(x, 2, l) + k1 * kl_coeff(x, 1, l) +
                              k0 * kl_coeff(x, 0, l))

    def Psi_minus(x, c):
        tmp = 0
        for n in range(4):
            tmp += (-1)**n * k[n] * c**(-2*n -2) * gammaincc(2*n + 2, c*np.sqrt(-x)) * gamma(2*n + 2)
        return tmp * 2

    def Psi_plus(x, c):
        tmp = 0
        for n in range(4):
            tmp +=  k[n] * c**(-2*n -2) * gammaincc(2*n + 2, c*np.sqrt(x)) * gamma(2*n + 2)
        return tmp * (-2)

    def I(x):
        if x0 < 0:
            if x < x1:
                return 2 * np.exp(-c3) * Phi(x, c1) + const1
            elif x1 <= x < x0:
                return 2 * np.exp(-c3) * Psi_minus(x, c2) + const2
            elif x0 < x <= 0:
                return 2 * np.exp(-c3) * Psi_minus(x, c4) + const3
            else:
                return 2 * np.exp(-c3) * Psi_plus(x, c4) + const4
        else:
            if x < x1:
                return 2 * np.exp(-c3) * Phi(x, c1) + const1
            elif x1 <= x < 0:
                return 2 * np.exp(-c3) * Psi_minus(x, c2) + const2
            elif 0 < x <= x0:
                return 2 * np.exp(-c3) * Psi_plus(x, c4) + const3
            else:
                return 2 * np.exp(-c3) * Psi_plus(x, c4) + const4


    def d(x):
        """Computes the diffusion coefficient d(x) based on the left and right branches."""
        if x < x1:
            return c1 * abs(x) + c3
        elif x1 <= x < x0:
            return c2 * np.sqrt(abs(x)) + c3
        else:
            return c4 * np.sqrt(abs(x)) + c3

    def prob_stationary(x):
            return (1/z) * np.exp(I(x) - d(x))

    x0 = 0.10840501051683304
    x1 = -150
    k4 = -3.045 * 10**(-10)
    k3 = -2.736 * 10**(-7)
    k2 = -1.599 * 10**(-5)
    k1 = -1.109 * 10 **(-1)
    k0 = -2.082 * 10

    c1 = 2.829 * 10**(-3)
    c2 = 6.593 * 10**(-2)
    c3 = 8.309
    c4 = 8.706 * 10**(-2)
    z = 1
    k = [k0, k1, k2, k3, k4]

    if x0 < 0:
        const1 = 0
        const2 = const1 + 2 * np.exp(-c3)*(Phi(x1, c1) - Psi_minus(x1, c2))
        const3 = const2 + 2 * np.exp(-c3)*(Psi_minus(x0, c2) - Psi_minus(x0, c4))
        const4 = const3 + 2 * np.exp(-c3)*(Psi_minus(0, c4) - Psi_plus(0, c4))
    else:
        const1 = 0
        const2 = const1 + 2 * np.exp(-c3)*(Phi(x1, c1) - Psi_minus(x1, c2))
        const3 = const2 + 2 * np.exp(-c3)*(Psi_minus(0, c2) - Psi_plus(0, c2))
        const4 = const3 + 2 * np.exp(-c3)*(Psi_plus(x0, c2) - Psi_plus(x0, c4))

    print([const1, const2, const3, const4])

    plot_prob_1d(files_path_prefix, 'sensible', prob_stationary,  np.linspace(-1000, 500, 1500))

    # x = np.linspace(-50, 50, 1000)
    x0 = -1.9346534142127325
    x1 = -500
    k4 = -6.353 * 10**(-12)
    k3 = -1.865 * 10**(-8)
    k2 = 4.135 * 10**(-5)
    k1 = 6.311 * 10 **(-2)
    k0 = -3.860 * 10

    c1 = 3.155 * 10**(-4)
    c2 = 7.889 * 10**(-2)
    c3 = 8.222
    c4 = 2.177 * 10**(-1)
    z = 1
    k = [k0, k1, k2, k3, k4]
    if x0 < 0:
        const1 = 19000
        const2 = const1 + 2 * np.exp(-c3)*(Phi(x1, c1) - Psi_minus(x1, c2))
        const3 = const2 + 2 * np.exp(-c3)*(Psi_minus(x0, c2) - Psi_minus(x0, c4))
        const4 = const3 + 2 * np.exp(-c3)*(Psi_minus(0, c4) - Psi_plus(0, c4))
    else:
        const1 = 0
        const2 = const1 + 2 * np.exp(-c3)*(Phi(x1, c1) - Psi_minus(x1, c2))
        const3 = const2 + 2 * np.exp(-c3)*(Psi_minus(0, c2) - Psi_plus(0, c2))
        const4 = const3 + 2 * np.exp(-c3)*(Psi_plus(x0, c2) - Psi_plus(x0, c4))
    print([const1, const2, const3, const4])

    plot_prob_1d(files_path_prefix, 'latent', prob_stationary, np.linspace(-2500, 100, 1000))


    # x = np.linspace(-1000, 1000, 2000)
    # k2 = -4.867 * 10**(-6)
    # k1 = 4.549 * 10 **(-1)
    # k0 = -1.018 * 10**2
    # c1 = 4.319 * 10**(-6)
    # c2 = 1.453
    # c3 = 14.02
    # c4 = 1.015 * 10**(-5)
    # c5 = 1.359
    # c6 = 14.02
    # z = 1
    # plot_prob_1d(files_path_prefix, 'pressure', prob_stationary, x)

