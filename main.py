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
    data1_name = 'sensible'
    data2_name = 'latent'
    str_types = 'sensible-latent'
    start_year = 1979
    end_year = start_year + 10

    if data1_name == 'sensible':
        data1_array = np.load(files_path_prefix + f'Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
    else:
        data1_array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    data1_array = data1_array.transpose()
    data1_array = data1_array.reshape((-1, height, width))

    # plot_hist(files_path_prefix, data1_name + f'_{start_year}-{end_year}', data1_array)
    print(f'min 1: {np.nanmin(data1_array)}')
    print(f'max 1: {np.nanmax(data1_array)}')

    if data2_name == 'latent':
        data2_array = np.load(files_path_prefix + f'Fluxes/latent_grouped_{start_year}-{end_year}.npy')
    else:
        data2_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    data2_array = data2_array.transpose()
    data2_array = data2_array.reshape((-1, height, width))
    # plot_hist(files_path_prefix, data2_name + f'_{start_year}-{end_year}', data2_array)
    print(f'min 2: {np.nanmin(data2_array)}')
    print(f'max 2: {np.nanmax(data2_array)}')

    mask = mask.reshape((height, width))
    create_quantiles(files_path_prefix, data1_array, data2_array, data1_name, data2_name, mask)
    # ----------------------------------------------------------------------------------------------
    # plot stationary distribution 1d
    def prob_stationary_sensible(x):
        if x > x_max:
            return np.exp(I(x_max, args) - log_b2(x, args))/ z
        return np.exp(I(x, args) - log_b2(x, args))/ z

    def prob_stationary_latent(x):
        if x < x_min:
            return np.exp(I(x_min, args) - log_b2(x, args))/ z
        return np.exp(I(x, args) - log_b2(x, args)) / z

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

    args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]

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

    args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]

    z = 1
    z = trapezoid([prob_stationary_latent(x) for x in np.linspace(-1500, 500, 2500)])
    print(f'Latent z: {z:.3e}')
    plot_prob_1d(files_path_prefix, 'latent', prob_stationary_latent, np.linspace(-800, 500, 2500))
    # plot_prob_and_hist(files_path_prefix, 'latent', prob_stationary_latent, np.linspace(-800, 500, 2500), data2_array[:365])
    # ----------------------------------------------------------------------------------------------
    # count and plot isolines

