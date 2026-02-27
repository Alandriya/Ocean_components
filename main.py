import datetime
import math
from struct import unpack
import json
import numpy as np
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
from Coefficients.Kor_Bel_compare import *
from Forecasting.utils import fix_random
from Plotting.plot_func_estimations import *
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
    # # Creating synthetic flux and counting Bel and Kor methods for it and plotting the difference
    # create_synthetic_data_1d(files_path_prefix, time_start=0, time_end=1000)
    # data_array = np.load(f'{files_path_prefix}Synthetic/flux_full.npy')
    # a_array = np.load(f'{files_path_prefix}Synthetic/A_full.npy')
    # b_array = np.load(f'{files_path_prefix}Synthetic/B_full.npy')
    # plot_synthetic_flux(files_path_prefix, data_array, 0, 5, a_array, b_array)
    # raise ValueError
    # path = 'Synthetic'
    #
    # # quantiles = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
    # quantiles = [50, 100, 150, 200, 250, 300, 350, 400, 500]
    # rmse_a_bel = np.zeros(len(quantiles), dtype=float)
    # rmse_b_bel = np.zeros(len(quantiles), dtype=float)
    # rmse_a_kor = np.zeros(len(quantiles), dtype=float)
    # rmse_b_kor = np.zeros(len(quantiles), dtype=float)
    # points_j_amount = 10
    # points_i_amount = 10
    # points_amount = points_i_amount * points_j_amount
    # for q in range(len(quantiles)):
    #     quantiles_amount = quantiles[q]
    #     print(quantiles_amount)
    #     count_1d_Bel(files_path_prefix, data_array, 0, 100,  path, 0, quantiles_amount)
    #     count_1d_Korolev(files_path_prefix, data_array, 0, 100, path, quantiles_amount, )

    #     rmse_Bel = [0.0, 0.0]
    #     rmse_Kor = [0.0, 0.0]
    #     a_results = 0
    #     b_results = 0
    #
    #     for point in [(i, j) for i in range(points_i_amount) for j in range(points_j_amount)]:
    #         collect_point(files_path_prefix, 1, 100, point, path, 'Bel')
    #         collect_point(files_path_prefix, 1, 100, point, path, 'Kor')
    #         # count_Bel_Kor_difference(files_path_prefix, 1, 100, point, '')
    #         # plot_difference_1d_synthetic(files_path_prefix, point, 3, 1, 99, 'A')
    #         # plot_difference_1d_synthetic(files_path_prefix, point, 3, 1, 99, 'B')
    #
    #         a_Bel = np.load(files_path_prefix + path + f'Bel/points/point_({point[0]}, {point[1]})-A.npy')
    #         b_Bel = np.load(files_path_prefix + path + f'Bel/points/point_({point[0]}, {point[1]})-B.npy')
    #         a_Kor = np.load(files_path_prefix + path + f'Kor/points/point_({point[0]}, {point[1]})-A.npy')
    #         b_Kor = np.load(files_path_prefix + path + f'Kor/points/point_({point[0]}, {point[1]})-B.npy')
    #         # count rmse
    #         rmse_Bel[0] += math.sqrt(sum((a_Bel - a_array[1:100, point[0], point[1]]) ** 2))
    #         rmse_Bel[1] += math.sqrt(sum((b_Bel - b_array[1:100, point[0], point[1]]) ** 2))
    #         rmse_Kor[0] += math.sqrt(sum((a_Kor - a_array[1:100, point[0], point[1]]) ** 2))
    #         rmse_Kor[1] += math.sqrt(sum((b_Kor - b_array[1:100, point[0], point[1]]) ** 2))
    #
    #     print(f'RMSE Bel: {rmse_Bel[0]/points_amount:.4f}, {rmse_Bel[1]/points_amount:.4f}, quantiles = {quantiles_amount}')
    #     print(f'RMSE Kor: {rmse_Kor[0] / points_amount:.4f}, {rmse_Kor[1] / points_amount:.4f}, quantiles = {quantiles_amount}')
    #     print('\n')
    #     rmse_a_bel[q] = rmse_Bel[0] / points_amount
    #     rmse_b_bel[q] = rmse_Bel[1] / points_amount
    #     rmse_a_kor[q] = rmse_Kor[0] / points_amount
    #     rmse_b_kor[q] = rmse_Kor[1] / points_amount
    #
    # np.save(files_path_prefix + 'Synthetic/' + 'rmse_A_bel.npy', rmse_a_bel)
    # np.save(files_path_prefix + 'Synthetic/' + 'rmse_B_bel.npy', rmse_b_bel)
    # np.save(files_path_prefix + 'Synthetic/' + 'rmse_A_kor.npy', rmse_a_kor)
    # np.save(files_path_prefix + 'Synthetic/' + 'rmse_B_kor.npy', rmse_b_kor)
    #
    # quantiles = np.array(quantiles)
    # plot_quantiles_amount_compare(files_path_prefix, 'A', quantiles, ['Kor'])
    # plot_quantiles_amount_compare(files_path_prefix, 'A', quantiles, ['Bel'])
    # plot_quantiles_amount_compare(files_path_prefix, 'A', quantiles, ['Kor', 'Bel'])
    #
    # plot_quantiles_amount_compare(files_path_prefix, 'B', quantiles, ['Kor'])
    # plot_quantiles_amount_compare(files_path_prefix, 'B', quantiles, ['Bel'])
    # plot_quantiles_amount_compare(files_path_prefix, 'B', quantiles, ['Kor', 'Bel'])
    # ----------------------------------------------------------------------------------------------
    # plot Bel and Kor on real data
    path = 'Components/pressure/'
    data_name = 'pressure'
    data_array = np.load(files_path_prefix + f'Pressure/PRESS_{1979}-{1989}_grouped.npy')
    # data_array = np.load(files_path_prefix + f'Fluxes/latent_grouped_{1979}-{1989}.npy')
    data_array = data_array.transpose()
    data_array = data_array.reshape((-1, height, width))
    quantiles_amount = 250


    t = 1
    for quantiles_amount in [50, 100, 150, 200, 250, 300, 350]:
        count_1d_Bel(files_path_prefix, data_array, 0, 3, path, 0, quantiles_amount)
        count_1d_Korolev(files_path_prefix, data_array, 0, 3, path, quantiles_amount, )
        print(quantiles_amount)
        for coeff_type in ['A', 'B']:
            bel = np.load(files_path_prefix + path + f'Bel/daily/{coeff_type}_{t}.npy')
            kor = np.load(files_path_prefix + path + f'Kor/daily/{coeff_type}_{t}.npy')
            # plot_methods_compare(files_path_prefix, bel, kor, data_name, coeff_type, t)
            data_min = min(np.nanmin(bel), np.nanmin(kor))
            data_max = max(np.nanmax(bel), np.nanmax(kor))
            bel = np.nan_to_num(bel)
            kor = np.nan_to_num(kor)
            print(f'Data type {data_name}, {coeff_type} SSIM: {ssim(bel, kor, data_range=data_max-data_min):.3f}')
    raise ValueError
    # ----------------------------------------------------------------------------------------------
    # # mann whitneyu test
    # # quantiles_amount = 50
    # # data_name = 'pressure'
    # # data_name = 'sensible'
    # data_name = 'latent'
    #
    # # data_array = np.load(files_path_prefix + f'Pressure/PRESS_{1979}-{1989}_grouped.npy')
    # # data_array = np.load(files_path_prefix + f'Fluxes/sensible_grouped_{1979}-{1989}.npy')
    # data_array = np.load(files_path_prefix + f'Fluxes/latent_grouped_{1979}-{1989}.npy')
    # data_array = data_array.transpose()
    # data_array = data_array.reshape((-1, height, width))
    # # path = 'Components/pressure/'
    # # path = 'Components/sensible/'
    # path = 'Components/latent/'
    #
    # # data_array = np.load(f'{files_path_prefix}Synthetic/flux_full.npy')
    # # path = 'Synthetic'
    # print(data_array.shape)
    #
    # points_j_amount = 10
    # points_i_amount = 10
    # amount = points_i_amount * points_j_amount
    # mask = mask.reshape((height, width))
    # # mask = np.ones((points_i_amount, points_j_amount))
    # for quantiles_amount in [50, 100, 150, 200, 250, 300, 350]:
    #     count_1d_Bel(files_path_prefix, data_array, 0, 100,  path, 0, quantiles_amount)
    #     count_1d_Korolev(files_path_prefix, data_array, 0, 100, path, quantiles_amount, )
    #     a_results = 0
    #     b_results = 0
    #     pvalues_a = list()
    #     pvalues_b = list()
    #     for point in [(i, j) for i in range(80, 80 + points_i_amount) for j in range(80, 80 + points_j_amount)]:
    #     # for point in [(i, j) for i in range(points_i_amount) for j in range(points_j_amount)]:
    #         i, j = point
    #         if not mask[i, j]:
    #             print('Minus 1 point')
    #             amount -= 1
    #             continue
    #         collect_point(files_path_prefix, 1, 100, point, path, 'Bel')
    #         collect_point(files_path_prefix, 1, 100, point, path, 'Kor')
    #
    #         a_Bel = np.load(files_path_prefix + path + f'Bel/points/point_({point[0]}, {point[1]})-A.npy')
    #         b_Bel = np.load(files_path_prefix + path + f'Bel/points/point_({point[0]}, {point[1]})-B.npy')
    #         a_Kor = np.load(files_path_prefix + path + f'Kor/points/point_({point[0]}, {point[1]})-A.npy')
    #         b_Kor = np.load(files_path_prefix + path + f'Kor/points/point_({point[0]}, {point[1]})-B.npy')
    #
    #         # print(f'Point {point}')
    #         # print(f'A test: {mannwhitneyu(a_Bel, a_Kor)}')
    #         # print(f'B test: {mannwhitneyu(b_Bel, b_Kor)}')
    #         i, j = point
    #         U1, pa = mannwhitneyu(a_Bel, a_Kor)
    #         pvalues_a.append(pa)
    #         if pa < 0.05:
    #             a_results += 1
    #             # print(pa)
    #         U1, pb = mannwhitneyu(b_Bel, b_Kor)
    #         pvalues_b.append(pb)
    #         if pb < 0.05:
    #             b_results += 1
    #
    #
    #     reject, pvals_corrected_a, _, _ = multipletests(pvalues_a, 0.05, 'holm')
    #     reject, pvals_corrected_b, _, _ = multipletests(pvalues_b, 0.05, 'holm')
    #
    #     a_results_corrected = sum([1 if p < 0.05 else 0 for p in pvals_corrected_a])
    #     b_results_corrected = sum([1 if p < 0.05 else 0 for p in pvals_corrected_b])
    #
    #     print(quantiles_amount)
    #     print(f'A result: {a_results * 1.0/amount}')
    #     print(f'B result: {b_results * 1.0/amount}')
    #     print(f'A FWER: {a_results_corrected * 1.0/amount}')
    #     print(f'B FWER: {b_results_corrected * 1.0 / amount}')
    #     print('\n')
    # ----------------------------------------------------------------------------------------------
    # checking different amount of components in EM
    # flux = np.load(f'{files_path_prefix}Synthetic/flux_full.npy')
    # a_array = np.load(f'{files_path_prefix}Synthetic/A_full.npy')
    # b_array = np.load(f'{files_path_prefix}Synthetic/B_full.npy')
    # rmse_a_kor = np.zeros(6, dtype=float)
    # rmse_b_kor = np.zeros(6, dtype=float)
    # points_j_amount = 1
    # points_i_amount = 1
    # points_amount = points_i_amount * points_j_amount
    # for n in range(2, 6):
    #     count_1d_Korolev(files_path_prefix, flux, 0, 10, 'Synthetic/', 100, n_components=n)
    #     rmse_Kor = [0.0, 0.0]
    #     for point in tqdm.tqdm([(i, j) for i in range(points_i_amount) for j in range(points_j_amount)]):
    #         collect_point(files_path_prefix, 1, 10, point, 'Synthetic/', 'Kor')
    #         a_Kor = np.load(files_path_prefix + 'Synthetic/' + f'Kor/points/point_({point[0]}, {point[1]})-A.npy')
    #         b_Kor = np.load(files_path_prefix + 'Synthetic/' + f'Kor/points/point_({point[0]}, {point[1]})-B.npy')
    #
    #         # count rmse
    #         rmse_Kor[0] += math.sqrt(sum((a_Kor - a_array[1:10, point[0], point[1]]) ** 2))
    #         rmse_Kor[1] += math.sqrt(sum((b_Kor - b_array[1:10, point[0], point[1]]) ** 2))
    #
    #     print(f'RMSE Kor: {rmse_Kor[0] / points_amount:.4f}, {rmse_Kor[1] / points_amount:.4f}, n = {n}')
    #     rmse_a_kor[n-1] = rmse_Kor[0] / points_amount
    #     rmse_b_kor[n-1] = rmse_Kor[1] / points_amount
    #
    # np.save(files_path_prefix + 'Synthetic/' + 'rmse_A_kor.npy', rmse_a_kor)
    # np.save(files_path_prefix + 'Synthetic/' + 'rmse_B_kor.npy', rmse_b_kor)
    # # ----------------------------------------------------------------------------------------------
    # # estimate the functional a(X) and b(X) from data
    start_year = 1979
    end_year = 1989
    start = 0
    end = 1000
    start_index = 0
    # # data_name = 'sensible'
    # # data_array = np.load(files_path_prefix + f'Fluxes/sensible_grouped_{start_year}-{end_year}.npy')[:, start: end+1]
    data_name = 'latent'
    data_array = np.load(files_path_prefix + f'Fluxes/latent_grouped_{start_year}-{end_year}.npy')[:, start: end+1]
    # # data_name = 'pressure'
    # # data_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')[:, start: end+1]
    #
    data_array = data_array.transpose()
    data_array = data_array.reshape((-1, height, width))
    # # print(data_array.shape)
    #
    # count_1d_Korolev(files_path_prefix,
    #                  data_array,
    #                  time_start=start,
    #                  time_end=end + 1,
    #                  path=f'Components/{data_name}/',
    #                  quantiles_amount=250,
    #                  n_components=2,
    #                  start_index=start_index)
    #
    # a_array = np.zeros_like(data_array)
    # b_array = np.zeros_like(data_array)
    # for t in tqdm.tqdm(range(data_array.shape[0]-1)):
    #     a_array[t] = np.load(files_path_prefix + f'Components/{data_name}/Kor/daily/A_{t+start_index+1}.npy')
    #     b_array[t] = np.load(files_path_prefix + f'Components/{data_name}/Kor/daily/B_{t+start_index+1}.npy')
    # np.save(files_path_prefix + f'Components/{data_name}/a_{start_index}-{start_index + a_array.shape[0]}.npy', a_array)
    # np.save(files_path_prefix + f'Components/{data_name}/b_{start_index}-{start_index + a_array.shape[0]}.npy', b_array)

    a_array = np.load(files_path_prefix + f'Components/{data_name}/a_{start_index}-{start_index + data_array.shape[0]}.npy')
    b_array = np.load(files_path_prefix + f'Components/{data_name}/b_{start_index}-{start_index + data_array.shape[0]}.npy')
    # print(a_array.shape)
    # print(data_array.shape)
    quantiles, a_grouped, b_grouped, x_full, a_full, b_full = estimate_A_B(files_path_prefix, data_array, a_array, b_array)

    # print(quantiles)
    # print(a_grouped)
    # print(b_grouped)
    a_full = a_full[::50]
    b_full = b_full[::50]
    x_full = x_full[::50]
    # print(len(a_grouped))
    # print(len(b_grouped))
    # print(len(x_full))
    plot_ab_functional(files_path_prefix, quantiles, a_grouped, b_grouped, data_name, x_full, a_full, b_full)
    # # ----------------------------------------------------------------------------------------------