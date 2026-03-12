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
    days_delta5 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2025, 11, 1, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
    # days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # days_delta7 = (datetime.datetime(2024, 11, 28, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    # # Mean year
    # mean_year = np.zeros((365, 161, 181))
    # for start_year in [1979, 1989, 1999, 2009, 2019]:
    #     if start_year == 2019:
    #         end_year = 2026
    #     else:
    #         end_year = start_year + 10
    #
    #     print(f'Parsing {start_year}-{end_year}')
        # sensible = np.load(files_path_prefix + f'Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
        # sensible = sensible.transpose()
        # sensible = sensible.reshape((-1, height, width))
        # latent = np.load(files_path_prefix + f'Fluxes/latent_grouped_{start_year}-{end_year}.npy')
        # latent = latent.transpose()
        # latent = latent.reshape((-1, height, width))
        # press = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
        # press = press.transpose()
        # press = press.reshape((-1, height, width))
        # if end_year == 2026:
        #     end_year = 2025
        # for j in range(end_year-start_year):
        #     time_start = (datetime.datetime(year=start_year + j, month=1, day=1) - datetime.datetime(year=start_year, month=1,
        #                                                                                    day=1)).days
        #     time_end = time_start + 365
            # print(time_start)
            # print(time_end)
            # mean_year += sensible[time_start:time_end]
            # mean_year += latent[time_start:time_end]
            # mean_year += press[time_start:time_end]

    # mean_year /= (2025 - 1979)
    # np.save(files_path_prefix + f'Mean_year/mean_sensible.npy', mean_year)
    # np.save(files_path_prefix + f'Mean_year/mean_latent.npy', mean_year)
    # np.save(files_path_prefix + f'Mean_year/mean_flux.npy', mean_year)
    # np.save(files_path_prefix + f'Mean_year/mean_pressure.npy', mean_year)
    # raise ValueError

    # # mean season
    # mean_year = np.load(files_path_prefix + f'Mean_year/Bel/press_B_1979-2023.npy')
    # mean_season = np.zeros((4, 161, 181))
    # spring_start = (datetime.datetime(year=2026, month=3, day=1) - datetime.datetime(year=2026, month=1, day=1)).days
    # spring_end = (datetime.datetime(year=2026, month=6, day=1) - datetime.datetime(year=2026, month=1, day=1)).days
    # mean_season[1] = np.mean(mean_year[spring_start:spring_end], axis=0)
    # summer_start = (datetime.datetime(year=2026, month=6, day=1) - datetime.datetime(year=2026, month=1, day=1)).days
    # summer_end = (datetime.datetime(year=2026, month=9, day=1) - datetime.datetime(year=2026, month=1, day=1)).days
    # mean_season[2] = np.mean(mean_year[summer_start:summer_end], axis=0)
    # autumn_start = (datetime.datetime(year=2026, month=9, day=1) - datetime.datetime(year=2026, month=1, day=1)).days
    # autumn_end = (datetime.datetime(year=2026, month=12, day=1) - datetime.datetime(year=2026, month=1, day=1)).days
    # mean_season[3] = np.mean(mean_year[autumn_start:autumn_end], axis=0)
    #
    # winter1_start = (datetime.datetime(year=2026, month=1, day=1) - datetime.datetime(year=2026, month=1, day=1)).days
    # winter1_end = (datetime.datetime(year=2026, month=3, day=1) - datetime.datetime(year=2026, month=1, day=1)).days
    # winter2_start = (datetime.datetime(year=2027, month=1, day=1) - datetime.datetime(year=2026, month=12, day=1)).days
    # winter2_end = 365
    # mean_season[0] = (np.mean(mean_year[winter1_start:winter1_end], axis=0) + np.mean(mean_year[winter2_start:winter2_end], axis=0))/2.0
    # np.save(files_path_prefix + f'Mean_year/mean_season_press_B.npy', mean_season)
    # for data_name in ['sensible', 'latent', 'flux', 'pressure']:
    #     mean_year_a = np.zeros((364, 161, 181))
    #     mean_year_b = np.zeros((364, 161, 181))
    #     for year in range(1979, 2019):
    #         print(f'Year {year}', flush=True)
    #         time_start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=1979, month=1, day=1)).days
    #         time_end = time_start + 365
    #
    #         for day in tqdm.tqdm(range(1, 365)):
    #             a_map = np.load(files_path_prefix + f'Coeff_data_Korolev/{data_name}' + f'Kor/daily/A_{day + time_start}.npy')
    #             b_map = np.load(files_path_prefix + f'Coeff_data_Korolev/{data_name}' + f'Kor/daily/B_{day + time_start}.npy')
    #             mean_year_a[day-1, :, :] += a_map
    #             mean_year_b[day-1, :, :] += b_map
    #
    #     mean_year_a /= (2019 - 1979)
    #     mean_year_b /= (2019 - 1979)
    #     np.save(files_path_prefix + f'Mean_year/mean_{data_name}_A.npy', mean_year_a)
    #     np.save(files_path_prefix + f'Mean_year/mean_{data_name}_B.npy', mean_year_b)
    # raise ValueError
    # plot_mean_year(files_path_prefix, 'C_1')
    # ----------------------------------------------------------------------------------------------
    # Count Korolev coefficients for sensible, latent, press and flux
    # quantiles = 250
    # for data_name in ['pressure']:
    #     for start_year in [1979, 1989, 1999, 2009, 2019]:
    #         if start_year == 2019:
    #             end_year = 2026
    #         else:
    #             end_year = start_year + 10
    #
    #         if start_year == 1979:
    #             start_index = 0
    #         elif start_year == 1989:
    #             start_index = days_delta1
    #         elif start_year == 1999:
    #             start_index = days_delta1 + days_delta2
    #         elif start_year == 2009:
    #             start_index = days_delta1 + days_delta2 + days_delta3
    #         else:
    #             start_index = days_delta1 + days_delta2 + days_delta3 + 1870
    #
    #         print(f'Counting {start_year} - {end_year}: {data_name}')
    #         if data_name in ['sensible', 'latent']:
    #             data_array = np.load(files_path_prefix + f'Fluxes/{data_name}_grouped_{start_year}-{end_year}.npy')
    #         elif data_name == 'flux':
    #             sensible = np.load(files_path_prefix + f'Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
    #             latent = np.load(files_path_prefix + f'Fluxes/latent_grouped_{start_year}-{end_year}.npy')
    #             data_array = sensible + latent
    #             del sensible, latent
    #         elif data_name == 'pressure':
    #             data_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    #
    #         data_array = data_array.transpose()
    #         data_array = data_array.reshape((-1, height, width))
    #         # print(len(data_array))
    #         # print(data_array.shape)
    #         if not os.path.exists(files_path_prefix + f'Coeff_data_Korolev/{data_name}'):
    #             os.mkdir(files_path_prefix + f'Coeff_data_Korolev/{data_name}')
    #
    #
    #         count_1d_Korolev(files_path_prefix,
    #                          data_array,
    #                         0,
    #                          len(data_array),
    #                         f'Coeff_data_Korolev/{data_name}',
    #                          quantiles,
    #                          2,
    #                          start_index,
    #                          )

    # raise ValueError
    # ----------------------------------------------------------------------------------------------
    # # estimate the functional a(X) and b(X) from data
    # data_name = 'pressure'
    # start = 0
    # end = 364
    # start_index = 0
    # for data_name in ['sensible', 'latent', 'flux', 'pressure']:
    #     data_array = np.load(files_path_prefix + f'Mean_year/mean_{data_name}.npy')[start:end]
    #     a_array = np.load(files_path_prefix + f'Mean_year/mean_{data_name}_A.npy')[start:end]
    #     b_array = np.load(files_path_prefix + f'Mean_year/mean_{data_name}_B.npy')[start:end]
    #     # print(a_array.shape)
    #     mask = mask.reshape((height, width))
    #     data_array[:, np.logical_not(mask)] = np.nan
    #     a_array[:, np.logical_not(mask)] = np.nan
    #     b_array[:, np.logical_not(mask)] = np.nan
        # print(a_array.shape)
        # print(data_array.shape)
        # quantiles, a_grouped, b_grouped, x_full, a_full, b_full = estimate_A_B(files_path_prefix, data_array, a_array, b_array)

        # print(max(x_full))
        # print(min(x_full))
        # print(quantiles)
        # print(a_grouped)
        # print(b_grouped)
        # rng = np.random.default_rng()
        # random_indexes_unique = rng.choice(len(a_full), size=10000, replace=False)
        # idxes = np.argsort(x_full)
        # idxes = idxes[::1000]
        # print(len(idxes))

        # a_full = np.array(a_full)
        # a_full = a_full[idxes]
        # b_full = np.array(b_full)
        # b_full = b_full[idxes]
        # x_full = np.array(x_full)
        # x_full = x_full[idxes]
        # print(len(quantiles))
        # print(len(a_grouped))
        # print(len(b_grouped))
        # print(len(x_full))
        # plot_ab_functional(files_path_prefix, quantiles, a_grouped, b_grouped, data_name, x_full, a_full, b_full)
    # # ----------------------------------------------------------------------------------------------

    # plot stationary distribution 1d
    # x = np.linspace(-103, 50, 500)
    # k = 0.02845
    # a = -6.21
    # b = 11.59
    # r1 = -103
    # def prob_sensible(x):
    #     return k * (x-r1)**(3.485) * (((x-a)**2 + b**2))**(-3.242) * math.exp(-3.888 * math.atan((x - a)/b))
    # plot_prob_1d(files_path_prefix, 'sensible', prob_sensible, x)

    # x = np.linspace(1, 150000, 1500)
    # k = 1.67 * 10**8
    # a = 79286.5
    # b = 24980
    # r1 = 151600
    # def prob_pressure(x):
    #     return k * math.pow(r1-x, 1.486) * math.pow(((x-a)**2 + b**2), -2.243) * math.exp(2.342 * math.atan((x - a)/b))
    #
    # plot_prob_1d(files_path_prefix, 'pressure', prob_pressure, x)

    # x = np.linspace(-170, -20,  500)
    # k = 1.43* 10**(-5)
    # a = -136.85
    # b = 88.9
    # r1 = 23.92
    # def prob_latent(x):
    #     return k * abs(x - r1)**(33.20) * ((x - a)**2 + b**2)**(-18.1) * math.exp(28.39  *  math.atan((x - a)/b))
    #
    # plot_prob_1d(files_path_prefix, 'latent', prob_latent, x)
    #
    # raise ValueError
    # ----------------------------------------------------------------------------------------------
    # count 2d semiparametric
    # for start_year in [1979, 1989, 1999, 2009, 2019]:
    #     if start_year == 2019:
    #         end_year = 2026
    #     else:
    #         end_year = start_year + 10
    #
    #     if start_year == 1979:
    #         start_index = 0
    #     elif start_year == 1989:
    #         start_index = days_delta1
    #     elif start_year == 1999:
    #         start_index = days_delta1 + days_delta2
    #     elif start_year == 2009:
    #         start_index = days_delta1 + days_delta2 + days_delta3
    #     else:
    #         start_index = days_delta1 + days_delta2 + days_delta3 + 1870

    start_year = 1979
    end_year = 1989
    start_index = 0
    sensible = np.load(files_path_prefix + f'Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
    sensible = sensible.transpose()
    sensible = sensible.reshape((-1, height, width))
    latent = np.load(files_path_prefix + f'Fluxes/latent_grouped_{start_year}-{end_year}.npy')
    latent = latent.transpose()
    latent = latent.reshape((-1, height, width))

    arr = np.zeros((sensible.shape[0], height, width, 2))
    arr[:, :, :, 0]  = sensible
    arr[:, :, :, 1] = latent
    arr[:, np.logical_not(mask), :] = np.nan

    count_semi_2d_AB(files_path_prefix, arr, 0, 1000, mask, quantiles_amount=15,
                     start_index=start_index, path='Components/sensible-latent')
        # raise ValueError
    # ----------------------------------------------------------------------------------------------
    # plot 2d semiparametric
    # time_start = 0
    # time_end = 10
    #
    # if not os.path.exists(files_path_prefix + 'videos/Semi_2d'):
    #     os.mkdir(files_path_prefix + 'videos/Semi_2d')
    # if not os.path.exists(files_path_prefix + 'videos/Semi_2d/sensible-latent'):
    #     os.mkdir(files_path_prefix + 'videos/Semi_2d/sensible-latent')
    #
    # a_timelist, b_timelist, _, _, _, _, borders = load_ABCFE(files_path_prefix, time_start, time_end,
    #                                                          load_a=True, load_b=True, path_local='Synthetic/Semi_2d/daily')
    # plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, borders, time_start, time_end-2, 1,
    #                          1, path_local='Semi_2d/sensible-latent/')