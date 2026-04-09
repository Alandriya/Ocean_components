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
from Data_processing.data_processing import *
from Data_processing.data_processing import mean_blocks
from Data_processing.func_estimation import *
# from Plotting.plot_eigenvalues import plot_eigenvalues, plot_mean_year
# from Plotting.plot_extreme import *
# from extreme_evolution import *
# from ABCF_coeff_counting import *
from Eigenvalues.eigenvalues import *
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
    # print(np.sum(mask))
    # raise ValueError
    data1_name = 'sensible'
    data2_name = 'latent'
    str_types = 'sensible-latent'
    # data1_name = 'flux'
    # data2_name = 'pressure'
    season = 'year'
    start_year = 1979

    # spring_start = (datetime.datetime(year=2026, month=3, day=1) - datetime.datetime(year=2026, month=1, day=1)).days
    # spring_end = (datetime.datetime(year=2026, month=6, day=1) - datetime.datetime(year=2026, month=1, day=1)).days
    # start = spring_start
    # end = spring_end

    # start = 0
    # end = 365
    # start_index = 0

    data1_array = np.load(files_path_prefix + f'Fluxes/sensible_grouped_1979-1989.npy')
    data1_array = data1_array.transpose()
    data1_array = data1_array.reshape((-1, height, width))
    data1_array = np.diff(data1_array, axis=0)

    # np.save(files_path_prefix + 'Fluxes/tmp.npy', data1_array[0:100])
    # raise ValueError
    # tmp = np.load(files_path_prefix + 'Fluxes/tmp.npy')
    # tmp = np.reshape(tmp, (-1, tmp.shape[1], tmp.shape[2], 2))
    # tmp_little = mean_blocks(tmp, mask.reshape((height, width)))
    # print(tmp_little.shape)
    # print(len(tmp_little.shape))
    # raise ValueError


    data2_array = np.load(files_path_prefix + f'Fluxes/latent_grouped_1979-1989.npy')
    data2_array = data2_array.transpose()
    data2_array = data2_array.reshape((-1, height, width))
    data2_array = np.diff(data2_array, axis=0)

    # data2_array = np.load(files_path_prefix + f'Mean_year/mean_{data2_name}.npy')[start:end]
    # plot_hist(files_path_prefix, data2_name + '_mean', data2_array)

    a_array = np.load(files_path_prefix + f'Components/{str_types}/Semi_2d/A_1-3653.npy')
    b_array = np.load(files_path_prefix + f'Components/{str_types}/Semi_2d/B_1-3653.npy')
    # a_array = np.load(files_path_prefix + f'Mean_year/mean_{data1_name}-{data2_name}_A.npy')[start:end]
    # b_array = np.load(files_path_prefix + f'Mean_year/mean_{data1_name}-{data2_name}_B.npy')[start:end]
    b_array = b_array**2
    # print(a_array.shape)
    mask = mask.reshape((height, width))
    data1_array[:, np.logical_not(mask)] = np.nan
    data2_array[:, np.logical_not(mask)] = np.nan
    a_array[:, np.logical_not(mask)] = np.nan
    b_array[:, np.logical_not(mask)] = np.nan
    # print(a_array.shape)
    # print(b_array.shape)
    # print(data1_array.shape)

    for year in range(start_year, start_year + 10):
        print(year)
        # start = (datetime.datetime(year=year, month=3, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
        # end = (datetime.datetime(year=year, month=6, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days

        start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
        end = (datetime.datetime(year=year + 1, month=1, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days

        # start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
        # end = (datetime.datetime(year=year, month=2, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days

        data1_small = mean_blocks(data1_array[start:end], mask)
        data2_small = mean_blocks(data2_array[start:end], mask)
        a_small = mean_blocks(a_array[start:end], mask)
        b_small = mean_blocks(b_array[start:end], mask)

        quantiles1, quantiles2, a_grouped, b_grouped, x1_full, x2_full, a1_full, a2_full, b11_full, b22_full, = (
            estimate_A_B_2d(data1_small, data2_small, a_small, b_small, quantiles_amount=300))

        # idxes = np.argsort(x2_full)
        # idxes = idxes[::10]
        # print(len(idxes))

        # a1_full = np.array(a1_full)[idxes]
        # a2_full = np.array(a2_full)[idxes]
        # x1_full = np.array(x1_full)[idxes]
        # x2_full = np.array(x2_full)[idxes]
        # b11_full = np.array(b11_full)[idxes]
        # b22_full = np.array(b22_full)[idxes]

        data = [quantiles1, quantiles2, a_grouped, np.log(b_grouped), x1_full, x2_full, a1_full, a2_full, np.log(b11_full), np.log(b22_full)]
        plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, season, year, scatter=True)
        plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, season, year, scatter=False)

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
    # x = np.linspace(-5000, 2000, 1500)
    # k = 1.6 * 10**6
    # a = 20.425
    # b = 1351
    # r1 = -5845
    # def prob_sensible(x):
    #     return k * (x-r1)**(-0.8925) * (((x-a)**2 + b**2))**(-1.05373) * math.exp(-0.1493 * math.atan((x - a)/b))
    # plot_prob_1d(files_path_prefix, 'sensible', prob_sensible, x)

    # # plot stationary distribution 1d
    # x = np.linspace(-10000, 10000, 5000)
    # k = 9944.005636
    # a = 1.2703
    # b = -3851.6774
    # c = 1.7592 * 10**7
    # lambda_ = -0.112253
    # r1 = -5845
    # k2 =    -0.177185
    #
    # print(math.sqrt(4*a*c - b**2))
    #
    # def d(x):
    #     return a * (x**2) + b*x + c
    # def prob_sensible(x):
    #     return k * (d(x))**(lambda_ -1 ) * math.exp(k2 * math.atan((2*a*x + b)/math.sqrt(4*a*c - b**2)))
    # plot_prob_1d(files_path_prefix, 'sensible_2', prob_sensible, x)

