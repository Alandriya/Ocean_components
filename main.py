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
from scipy.special import gammainc, gamma, erf, erfcx
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
    # estimate the functional a(X) and b(X) from data
    data1_name = 'sensible'
    data2_name = 'latent'
    str_types = 'sensible-latent'

    # data1_name = 'flux'
    # data2_name = 'pressure'
    # str_types = 'flux-pressure'
    season = 'full_10_years'
    start_year = 1979
    end_year = start_year + 10
    coef_start = 1
    coef_end = 3653

    if data1_name == 'sensible':
        data1_array = np.load(files_path_prefix + f'Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
    else:
        data1_array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    data1_array = data1_array.transpose()
    data1_array = data1_array.reshape((-1, height, width))
    # data1_array = data1_array[1:]
    # data1_array = np.diff(data1_array, axis=0)
    plot_hist(files_path_prefix, data1_name + f'_{start_year}-{end_year}', data1_array)

    if data2_name == 'latent':
        data2_array = np.load(files_path_prefix + f'Fluxes/latent_grouped_{start_year}-{end_year}.npy')
    else:
        data2_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    data2_array = data2_array.transpose()
    data2_array = data2_array.reshape((-1, height, width))
    # data2_array = np.diff(data2_array, axis=0)
    # data2_array = data2_array[1:]
    plot_hist(files_path_prefix, data2_name + f'_{start_year}-{end_year}', data2_array)

    a_array = np.load(files_path_prefix + f'Components/{str_types}/Semi_2d/A_{coef_start}-{coef_end}.npy')
    b_array = np.load(files_path_prefix + f'Components/{str_types}/Semi_2d/B_{coef_start}-{coef_end}.npy')
    b_array = b_array**2

    mask = mask.reshape((height, width))
    data1_array[:, np.logical_not(mask)] = np.nan
    data2_array[:, np.logical_not(mask)] = np.nan
    a_array[:, np.logical_not(mask)] = np.nan
    b_array[:, np.logical_not(mask)] = np.nan


    # for year in range(start_year, start_year + 10):
    for year in [1979]:
        print(year)
        if season == 'year':
            start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=start_year, month=1,
                                                                                      day=1)).days
            end = (datetime.datetime(year=year, month=12, day=31) - datetime.datetime(year=start_year, month=1,
                                                                                      day=1)).days
        elif season == 'spring':
            start = (datetime.datetime(year=year, month=3, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
            end = (datetime.datetime(year=year, month=6, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
        elif season == 'summer':
            start = (datetime.datetime(year=year, month=6, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
            end = (datetime.datetime(year=year, month=9, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
        elif season == 'autumn':
            start = (datetime.datetime(year=year, month=9, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
            end = (datetime.datetime(year=year, month=12, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
        else:
            start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
            end = (datetime.datetime(year=year, month=3, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days


        start = 0
        end = 3650
        data1_small = mean_blocks(data1_array[start:end], mask)
        data2_small = mean_blocks(data2_array[start:end], mask)
        a_small = mean_blocks(a_array[start:end], mask)
        b_small = mean_blocks(b_array[start:end], mask)

        quantiles1, quantiles2, a_grouped, b_grouped, x1_full, x2_full, a1_full, a2_full, b11_full, b22_full, = (
            estimate_A_B_2d(data1_small, data2_small, a_small, b_small, quantiles_amount=370))

        data = [quantiles1, quantiles2, a_grouped, np.log(b_grouped), x1_full, x2_full, a1_full, a2_full, np.log(b11_full), np.log(b22_full)]
        # plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, season, year, scatter=True)
        plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, season, year, scatter=False)
    raise ValueError

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
    def G(n, z, a):
        """Computes the nth generalized G function with given parameter a."""
        if a <= 0:
            raise ValueError(f"a must be positive, got {a}")
        return gammainc(n + 1, a * z) * gamma(n+1) * (a ** -(n + 1))


    # Define the F function with the new approach
    def F(n, z, a, b):
        """Computes the nth generalized F function with given parameters a, b."""
        if b <= 0:
            raise ValueError(f"b must be positive, got {b}")

        if n == 0:
            s = np.sqrt(b)
            alpha = a / (2 * s)
            beta = s * z
            return np.sqrt(np.pi) / (2 * s) * (
                    erfcx(alpha) - np.exp(-b * z ** 2 - a * z) * erfcx(alpha + beta)
            )
        elif n == 1:
            return (1 - np.exp(-b * z ** 2 - a * z)) / (2 * b) - (a / (2 * b)) * F(0, z, a, b)
        else:
            return ((n - 1) * F(n - 2, z, a, b) - a * F(n - 1, z, a, b) - z ** (n - 1) * np.exp(
                -b * z ** 2 - a * z)) / (2 * b)


    # Left branch with the full expression
    def Phi_plus_left(x, a, b, d):
        """Computes the right branch integral for x >= 0"""
        z = np.sqrt(x)
        return 4 * np.exp(-d) * (
                k4 * F(9, z, a, b) + k3 * F(7, z, a, b) + k2 * F(5, z, a, b) +
                k1 * F(3, z, a, b) + k0 * F(1, z, a, b)
        )


    # Left branch, for negative x
    def Phi_minus_left(x, a, b, d):
        """Computes the left branch integral for x <= 0"""
        z = np.sqrt(-x)
        return -4 * np.exp(-d) * (
                k4 * F(9, z, a, b) - k3 * F(7, z, a, b) + k2 * F(5, z, a, b) -
                k1 * F(3, z, a, b) + k0 * F(1, z, a, b)
        )


    def Phi_plus_right(x, c3, c4):
        """Computes the right branch integral for x >= x0, simplified formula"""
        z = np.sqrt(x)
        return 4 * np.exp(-c3) * (
                k4 * G(9, z, c4) + k3 * G(7, z, c4) + k2 * G(5, z, c4) +
                k1 * G(3, z, c4) + k0 * G(1, z, c4)
        )


    def Phi_minus_right(x, c3, c4):
        """Computes the right branch integral for x <= x0"""
        z = np.sqrt(-x)
        return -4 * np.exp(-c3) * (
                k4 * G(9, z, c4) - k3 * G(7, z, c4) + k2 * G(5, z, c4) -
                k1 * G(3, z, c4) + k0 * G(1, z, c4)
        )

    def I(x, x0):
        """Computes the integral I(x) with respect to the branching point x0."""
        if np.isclose(x0, 0.0):  # x0 == 0 case
            if x < 0:
                return Phi_minus_left(x, c1, c2, c3)
            else:
                return Phi_plus_right(x, c3, c4)
        elif x0 < 0:  # x0 < 0 case
            if x < x0:
                return Phi_minus_left(x, c1, c2, c3) - Phi_minus_left(x0, c1, c2, c3)
            elif x0 <= x < 0:
                return Phi_minus_right(x, c3, c4) - Phi_minus_right(x0, c3, c4)
            else:
                return -Phi_minus_right(x0, c3, c4) + Phi_plus_right(x, c3, c4)
        else:  # x0 > 0 case
            if x < 0:
                return Phi_minus_left(x, c1, c2, c3) - Phi_plus_left(x0, c1, c2, c3)
            elif 0 <= x < x0:
                return Phi_plus_left(x, c1, c2, c3) - Phi_plus_left(x0, c1, c2, c3)
            else:
                return Phi_plus_right(x, c3, c4) - Phi_plus_right(x0, c3, c4)


    def d(x):
        """Computes the diffusion coefficient d(x) based on the left and right branches."""
        if x < x0:
            return c1 * np.sqrt(abs(x)) + c2 * abs(x) + c3
        else:
            return c4 * np.sqrt(abs(x)) + c3

    def prob_stationary(x):
            return (1/z) * np.exp(I(x, x0) - d(x))


    # def compute_log_shift(x_min, x_max, num_points, I_func, d_func):
    #     """Computes the maximum of the log probability to normalize the distribution."""
    #     x_values = np.linspace(x_min, x_max, num_points)
    #     log_values = I_func(x_values) - d_func(x_values)
    #     return np.max(log_values)

    # def prob_stationary(x):
    #     """Computes the stationary probability P_s(x) given x."""
    #     # log_shift = compute_log_shift(-10, 10, 1000, I, d)  # Compute the log shift
    #     return np.exp(I(x, x0) - d(x) - log_shift) / z





    x = np.linspace(-500, 500, 1500)
    x0 = 0.10137566460502967
    k4 = -1.167 * 10**(-10)
    k3 = -7.945 * 10**(-8)
    k2 = 2.984 * 10**(-5)
    k1 = -1.123 * 10 **(-1)
    k0 = -2.132 * 10

    c1 = 6.240 * 10**(-2)
    c2 = 7.149 * 10**(-4)
    c3 = 8.287
    c4 = 8.713 * 10**(-2)
    z = 1

    # print(F(0, 1.0, c1, c2))
    # print(F(1, 1.0, c1, c2))
    # print(I(-10, x0), I(10, x0))

    plot_prob_1d(files_path_prefix, 'sensible', prob_stationary, x)

    x = np.linspace(-50, 50, 1000)
    x0 = -1.43658447265625
    k4 = -5.008 * 10**(-12)
    k3 = -1.391 * 10**(-8)
    k2 = 4.316 * 10**(-5)
    k1 = 5.817 * 10 **(-2)
    k0 = -4.077 * 10

    c1 = 6.240 * 10**(-2)
    c2 = 7.149 * 10**(-4)
    c3 = 8.287
    c4 = 1.742 * 10**(-1)
    z = 1

    plot_prob_1d(files_path_prefix, 'latent', prob_stationary, x)


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

