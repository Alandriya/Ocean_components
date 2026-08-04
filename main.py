import datetime
import os
import numpy as np
import tqdm
from Plotting.plot_forecasts import *
from sklearn.metrics import root_mean_squared_error
from struct import unpack
files_path_prefix = 'D:/Nastya/Data/OceanFull/'
from Forecast_nn.utils import fix_random
from Data_processing.func_estimation import *
from Coefficients.semiparametric import count_BBT_from_B

width = 181
height = 161

block_size = 5
start_year = 1979
data1_name = 'sensible'
data2_name = 'latent'
str_types = f'{data1_name}-{data2_name}'

def get_mask():
    # Mask
    maskfile = open(files_path_prefix + "DATA/mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    mask = mask.reshape((height, width))
    return mask

if __name__ == '__main__':
    fix_random(2025)
    mask = get_mask()
    names = ('sensible', 'latent')
    n_bins = 100
    n_lambdas = 100
    height = 161
    width = 181
    b_type = 'eigen'

    start_year = 1979
    end_year = start_year + 10
    if start_year == 1979:
        coef_start = 1
        coef_end = 3653
    elif start_year == 1989:
        coef_start = 3654
        coef_end = 7305
    elif start_year == 1999:
        coef_start = 7306
        coef_end = 10958
    elif start_year == 2009:
        coef_start = 10959
        coef_end = 14610
    else:
        end_year = 2026
        coef_start = 14611
        coef_end = 17106
    offset = coef_start

    end_year = 2024
    coef_end = 17106
    missing_values_eigen = [3653, 7305, 10958, 14610]
    missing_days_coefs = [0, 7304, 10957]
    missing_days_eigen = [0, 3652, 7304, 10957, 14609]

    # data1_array = np.load(files_path_prefix + f'DATA/Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
    # data1_array = data1_array.transpose()
    # data1_array = data1_array.reshape((-1, height, width))
    # if end_year == 2024:
    #     data1_array = np.delete(data1_array, missing_days_eigen, axis=0)
    #
    # data2_array = np.load(files_path_prefix + f'DATA/Fluxes/latent_grouped_{start_year}-{end_year}.npy')
    # data2_array = data2_array.transpose()
    # data2_array = data2_array.reshape((-1, height, width))
    # if end_year == 2024:
    #     data2_array = np.delete(data2_array, missing_days_eigen, axis=0)

    start_year=19792024

    # np.save(files_path_prefix + f'DATA/Fluxes/sensible_mean_{start_year}-{end_year}.npy', np.mean(data1_array, axis=0))
    # np.save(files_path_prefix + f'DATA/Fluxes/latent_mean_{start_year}-{end_year}.npy', np.mean(data2_array, axis=0))

    # plot_hist(files_path_prefix, data1_name + f'_{start_year}-{end_year}', data1_array)
    # print(f'min 1: {np.nanmin(data1_array)}')
    # print(f'max 1: {np.nanmax(data1_array)}')
    # plot_hist(files_path_prefix, data2_name + f'_{start_year}-{end_year}', data2_array)
    # print(f'min 2: {np.nanmin(data2_array)}')
    # print(f'max 2: {np.nanmax(data2_array)}')

    # create_quantiles_2d(files_path_prefix, data1_array, data2_array, data1_name, data2_name, mask, 260,
    #                     coef_start, coef_end, start_year, block_size, b_type)
    # raise ValueError
    # quantiles1 = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_quantiles1.npy')
    # quantiles2 = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_quantiles2.npy')
    # a_grouped = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_a_grouped.npy')
    # b_grouped = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_b_grouped_log_{b_type}.npy')
    # if b_type == 'eigen' and start_year == 1999:
    #     b_grouped = np.load(files_path_prefix + f'Functional/{str_types}/3654-7305_b_grouped_log_{b_type}.npy')

    # data = [quantiles1, quantiles2, a_grouped, b_grouped]
    # coefs = plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, b_type, start_year)
    # raise ValueError
    # sensible_params, latent_params = coefs
    # print('\n\n\n')
    #
    # # plot stationary distribution 1d
    # def prob_stationary(x):
    #     if x > x_max:
    #         return np.exp(I(x_max, args) - log_b2(x, args))/ z
    #
    #     if x < x_min:
    #         return np.exp(I(x_min, args) - log_b2(x, args))/ z
    #     return np.exp(I(x, args) - log_b2(x, args))/ z
    #
    #
    # x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = sensible_params
    # dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4 = count_constants(sensible_params)
    # args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]
    # z = trapezoid([prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)])
    # print(f'Sensible z: {z:.3e}')
    # y1 = [prob_stationary(x) for x in np.linspace(-500, 500, 2500)] #simple

    # x = np.linspace(-2000, 1500, 2500)
    # mean1, var1 = moments(prob_stationary, x)
    # print(f'Mean sensible: {mean1:.1f}')
    # print(f'Var sensible: {var1:.1f}')
    # print(f'Sigma sensible: {np.sqrt(var1):.1f}')
    # plot_prob_1d(files_path_prefix, 'sensible', prob_stationary,  np.linspace(-300, 300, 2500), start_year, b_type)

    # b_grouped = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_b_grouped_log_eigen.npy')
    # data = [quantiles1, quantiles2, a_grouped, b_grouped]
    # coefs = plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, 'eigen', start_year)
    # sensible_params_new, _ = coefs
    #
    # x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = sensible_params_new
    # dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4 = count_constants(sensible_params_new)
    # args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]
    # z = trapezoid([prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)])
    # print(f'Sensible z 2: {z:.3e}')
    # y2 = [prob_stationary(x) for x in np.linspace(-500, 500, 2500)]
    # plot_prob_and_hist(files_path_prefix, 'sensible',y1, y2, np.linspace(-500, 500, 2500), start_year, data1_array[::10])
    #
    # x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = latent_params
    # if b_type == 'eigen' and start_year == 1999:
    #     x_max = -1
    # dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4 = count_constants(latent_params)
    # args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]
    # z = trapezoid([prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)])
    # print(f'Latent z: {z:.3e}')
    # y1 = [prob_stationary(x) for x in np.linspace(-800, 300, 2500)]
    #
    # b_grouped = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_b_grouped_log_eigen.npy')
    # data = [quantiles1, quantiles2, a_grouped, b_grouped]
    # coefs = plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, 'simple', start_year)
    # _, latent_params_new = coefs
    # x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = latent_params_new
    # dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4 = count_constants(latent_params_new)
    # args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]
    # z = trapezoid([prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)])
    # y2 = [prob_stationary(x) for x in np.linspace(-800, 300, 2500)]

    # mean2, var2 = moments(prob_stationary, np.linspace(-800, 300, 2500))
    # print(f'Mean latent: {mean2:.1f}')
    # print(f'Var latent: {var2:.1f}')
    # print(f'Sigma latent: {np.sqrt(var2):.1f}')
    # plot_prob_1d(files_path_prefix, 'latent', prob_stationary, np.linspace(-800, 300, 2500), start_year, b_type)
    # plot_prob_and_hist(files_path_prefix, 'latent', y1, y2, np.linspace(-800, 300, 2500), start_year, data2_array[::10])

    # -----------------------------------------------------------------------------------
    # # count and plot isolines

    # amount = 5
    # colors_list = ['yellow', 'orange', 'red', 'violet', 'blue']
    # isolines = get_isolines(prob_stationary, np.linspace(-300, 300, 2500), amount)
    # print(isolines)
    # isolines_list = [(isolines[i], isolines[i+1]) for i in range(amount)]
    # plot_areas(files_path_prefix, 'sensible', prob_stationary, np.linspace(-300, 300, 2500), isolines_list, colors_list)
    # plot_areas_map(files_path_prefix, data1_name, isolines_list, data1_array, 0, 10, mask,
    #                  0, [1, 2, 3, 4, 5], colors_list)
    #
    # # data1_mean = np.load(files_path_prefix + f'DATA/Fluxes/sensible_mean_{start_year}-{end_year}.npy')
    # data1_mean = np.load(files_path_prefix + f'DATA/Fluxes/sensible_mean_1979-2024.npy')
    # plot_isolines_map(files_path_prefix, 'sensible', data1_mean-mean1, mask)

    # isolines = get_isolines(prob_stationary, np.linspace(-800, 300, 2500), amount)
    # print(isolines)
    # isolines_list = [(isolines[i], isolines[i+1]) for i in range(amount)]
    # plot_areas(files_path_prefix, 'latent', prob_stationary, np.linspace(-800, 300, 2500), isolines_list, colors_list)
    # plot_areas_map(files_path_prefix, data2_name, isolines_list, data2_array, 0, 10, mask,
    #                  0, [1, 2, 3, 4, 5], colors_list)
    #
    # # data2_mean = np.load(files_path_prefix + f'DATA/Fluxes/latent_mean_{start_year}-{end_year}.npy')
    # data2_mean = np.load(files_path_prefix + f'DATA/Fluxes/latent_mean_1979-2024.npy')
    # plot_isolines_map(files_path_prefix, 'latent', data2_mean - mean2, mask)
    # -----------------------------------------------------------------------------------
    count_BBT_from_B(files_path_prefix, 0, 16100)


