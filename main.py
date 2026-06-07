import datetime

import numpy as np

from config import *

if __name__ == '__main__':
    fix_random(2025)
    mask = get_mask()

    start_year = 1979
    end_year = 2024
    # start_year = 19792024

    data1_array = np.load(files_path_prefix + f'DATA/Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
    data1_array = data1_array.transpose()
    data1_array = data1_array.reshape((-1, height, width))

    data2_array = np.load(files_path_prefix + f'DATA/Fluxes/latent_grouped_{start_year}-{end_year}.npy')
    data2_array = data2_array.transpose()
    data2_array = data2_array.reshape((-1, height, width))

    # if start_year == 2019:
    #     data1_array = data1_array[:days_delta8]
    #     data2_array = data2_array[:days_delta8]
    #
    np.save(files_path_prefix + f'DATA/Fluxes/sensible_mean_{start_year}-{end_year}.npy', np.mean(data1_array, axis=0))
    np.save(files_path_prefix + f'DATA/Fluxes/latent_mean_{start_year}-{end_year}.npy', np.mean(data2_array, axis=0))

    # plot_hist(files_path_prefix, data1_name + f'_{start_year}-{end_year}', data1_array)
    # print(f'min 1: {np.nanmin(data1_array)}')
    # print(f'max 1: {np.nanmax(data1_array)}')
    # plot_hist(files_path_prefix, data2_name + f'_{start_year}-{end_year}', data2_array)
    # print(f'min 2: {np.nanmin(data2_array)}')
    # print(f'max 2: {np.nanmax(data2_array)}')

    # create_quantiles(files_path_prefix, data1_array, data2_array, data1_name, data2_name, mask, coef_start, coef_end, start_year, block_size)
    # raise ValueError
    start_year = 19792024
    # plot stationary distribution 1d
    def prob_stationary(x):
        if x > x_max:
            return np.exp(I(x_max, args) - log_b2(x, args))/ z

        if x < x_min:
            return np.exp(I(x_min, args) - log_b2(x, args))/ z
        return np.exp(I(x, args) - log_b2(x, args))/ z


    x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = sensible_params
    dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4 = count_constants(sensible_params)
    args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]

    z = 1
    z = trapezoid([prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)])
    print(f'Sensible z: {z:.3e}')

    x = np.linspace(-2000, 1500, 2500)
    mean1, var1 = moments(prob_stationary, x)
    print(f'Mean sensible: {mean1:.1f}')
    print(f'Var sensible: {var1:.1f}')
    print(f'Sigma sensible: {np.sqrt(var1):.1f}')
    # plot_prob_1d(files_path_prefix, 'sensible', prob_stationary,  np.linspace(-300, 300, 2500), start_year)
    # plot_prob_and_hist(files_path_prefix, 'sensible', prob_stationary, np.linspace(-500, 500, 2500), start_year, data1_array[::10])

    # count and plot isolines
    amount = 5
    colors_list = ['yellow', 'orange', 'red', 'violet', 'blue']
    isolines = get_isolines(prob_stationary, x, amount)
    print(isolines)
    isolines_list = [(isolines[i], isolines[i+1]) for i in range(amount)]
    plot_areas(files_path_prefix, 'sensible', prob_stationary, x, isolines_list, colors_list)
    plot_areas_map(files_path_prefix, data1_name, isolines_list, data1_array, 0, 10, mask,
                     0, [1, 2, 3, 4, 5], colors_list)

    # data1_mean = np.load(files_path_prefix + f'DATA/Fluxes/sensible_mean_{start_year}-{end_year}.npy')
    data1_mean = np.load(files_path_prefix + f'DATA/Fluxes/sensible_mean_1979-2024.npy')
    plot_isolines_map(files_path_prefix, 'sensible', data1_mean-mean1, mask)
    # -----------------------------------------------------------------------------------

    x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = latent_params
    dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4 = count_constants(latent_params)
    args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]

    z = 1
    z = trapezoid([prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)])
    print(f'Latent z: {z:.3e}')

    x = np.linspace(-2000, 1500, 2500)

    mean2, var2 = moments(prob_stationary, x)
    print(f'Mean latent: {mean2:.1f}')
    print(f'Var latent: {var2:.1f}')
    print(f'Sigma latent: {np.sqrt(var2):.1f}')
    # plot_prob_1d(files_path_prefix, 'latent', prob_stationary, np.linspace(-800, 300, 2500), start_year)
    # plot_prob_and_hist(files_path_prefix, 'latent', prob_stationary, np.linspace(-1000, 1000, 2500), start_year, data2_array[::10])

    # count and plot isolines
    isolines = get_isolines(prob_stationary, x, amount)
    print(isolines)
    isolines_list = [(isolines[i], isolines[i+1]) for i in range(amount)]
    plot_areas(files_path_prefix, 'latent', prob_stationary, x, isolines_list, colors_list)
    plot_areas_map(files_path_prefix, data2_name, isolines_list, data2_array, 0, 10, mask,
                     0, [1, 2, 3, 4, 5], colors_list)

    # data2_mean = np.load(files_path_prefix + f'DATA/Fluxes/latent_mean_{start_year}-{end_year}.npy')
    data2_mean = np.load(files_path_prefix + f'DATA/Fluxes/latent_mean_1979-2024.npy')
    plot_isolines_map(files_path_prefix, 'latent', data2_mean - mean2, mask)
    # ----------------------------------------------------------------------------------------------

