# from imageio.config.plugins import summary
import os.path

import numpy as np

from Plotting.plot_coefficients import plot_velocity

files_path_prefix = 'D:/Nastya/Data/OceanFull/'
from Forecast_nn.utils import fix_random
from Functional.func_estimation import *
from Data_processing.data_processing import *
from Functional.check_solution_2d import check_zero_current_solution
from Coefficients.semiparametric import *
import scipy.sparse.linalg as spla
from Plotting.plot_coefficients import *

width = 181
height = 161

block_size = 5
start_year = 1989
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

    start_year = 2009
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
    elif start_year == 2019:
        end_year = 2026
        coef_start = 14611
        coef_end = 17106
    else:
        coef_start = 1
        end_year = 2024
        coef_end = 17106
    offset = coef_start

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

    # start_year=19792024

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
    # count C = B*B^T from 2d EN estimates
    # count_BBT_from_B(files_path_prefix, 'Components/sensible-latent/Daily/', 1, 17106)

    # collect_estimates(files_path_prefix, 'Components/sensible-latent/', ['BBT'],
    #                   1, 3653)

    # collect_estimates(files_path_prefix, 'Components/sensible-latent/', ['BBT'],
    #                   3654, 7305)

    # collect_estimates(files_path_prefix, 'Components/sensible-latent/', ['BBT'],
    #                   7306, 10958)

    # collect_estimates(files_path_prefix, 'Components/sensible-latent/', ['BBT'],
    #                   10959, 14610)
    #
    # collect_estimates(files_path_prefix, 'Components/sensible-latent/', ['BBT'],
    #                   14611, 17106)
    #
    # collect_estimates(files_path_prefix, 'Components/sensible-latent/', ['BBT'],
    #                   1, 17106)

    # c_array = np.load(files_path_prefix + f'Components/{str_types}/BBT_{coef_start}-{coef_end}.npy')
    # print(c_array.shape)
    # c_array = np.swapaxes(c_array, 1, 2)
    # c_array = np.swapaxes(c_array, 2, 3)
    # print(c_array.shape)
    # np.save(files_path_prefix + f'Components/{str_types}/BBT_{coef_start}-{coef_end}.npy', c_array)
    # -----------------------------------------------------------------------------------
    # count mesh and draw
    for q_amount_little in [15, 25, 35, 45]:
        for start_year in [1979, 1989, 1999, 2009, 2019]:
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
            elif start_year == 2019:
                end_year = 2026
                coef_start = 14611
                coef_end = 17106
            else:
                coef_start = 1
                start_year = 1979
                end_year = 2024
                coef_end = 17106
            offset = coef_start

            if not os.path.exists(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_a_mesh_{q_amount_little}.npy'):
                print(f'Counting year {start_year}, {q_amount_little} quantiles mesh')
                a_array = np.load(files_path_prefix + f'Components/{str_types}/A_{coef_start}-{coef_end}.npy')
                b_array = np.load(files_path_prefix + f'Components/{str_types}/B_{coef_start}-{coef_end}.npy')
                c_array = np.load(files_path_prefix + f'Components/{str_types}/BBT_{coef_start}-{coef_end}.npy')
                print(a_array.shape)
                print(b_array.shape)
                print(c_array.shape)

                data1_array = np.load(files_path_prefix + f'DATA/Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
                data1_array = data1_array.transpose().reshape((-1, height, width))

                data2_array = np.load(files_path_prefix + f'DATA/Fluxes/latent_grouped_{start_year}-{end_year}.npy')
                data2_array = data2_array.transpose().reshape((-1, height, width))

                if end_year == 2024:
                    data1_array = np.delete(data1_array, missing_days_eigen, axis=0)
                    data2_array = np.delete(data2_array, missing_days_eigen, axis=0)
                    a_array = np.delete(a_array, missing_days_eigen, axis=0)
                    b_array = np.delete(b_array, missing_days_eigen, axis=0)
                    c_array = np.delete(c_array, missing_days_eigen, axis=0)

                create_mesh(files_path_prefix,
                            data1_array,
                            data2_array,
                            q_amount_little,
                            coef_start,
                            coef_end,
                            str_types,
                            a_array,
                            b_array,
                            c_array)

                del data1_array, data2_array, a_array, b_array, c_array

    raise ValueError
    quantiles1 = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_quantiles1_{q_amount_little}.npy')
    quantiles2 = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_quantiles2_{q_amount_little}.npy')
    a_mesh = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_a_mesh_{q_amount_little}.npy')
    b_mesh = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_b_mesh_{q_amount_little}.npy')
    c_mesh = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_c_mesh_{q_amount_little}.npy')
    corr_mesh = count_correlation_BTT_mesh(c_mesh)
    plot_scalar_map(files_path_prefix, q_amount_little, a_mesh[0], 'A1', 'sensible', 'latent', coef_start, coef_end)
    plot_scalar_map(files_path_prefix, q_amount_little, a_mesh[1], 'A2', 'sensible', 'latent', coef_start, coef_end)
    plot_scalar_map(files_path_prefix, q_amount_little, b_mesh[0], 'B11', 'sensible', 'latent', coef_start, coef_end)
    plot_scalar_map(files_path_prefix, q_amount_little, b_mesh[1], 'B22', 'sensible', 'latent', coef_start, coef_end)
    plot_scalar_map(files_path_prefix, q_amount_little, b_mesh[2], 'B12', 'sensible', 'latent', coef_start, coef_end)
    plot_scalar_map(files_path_prefix, q_amount_little, b_mesh[3], 'B21', 'sensible', 'latent', coef_start, coef_end)
    plot_scalar_map(files_path_prefix, q_amount_little, c_mesh[0], 'C11', 'sensible', 'latent', coef_start, coef_end)
    plot_scalar_map(files_path_prefix, q_amount_little, c_mesh[1], 'C22', 'sensible', 'latent', coef_start, coef_end)
    plot_scalar_map(files_path_prefix, q_amount_little, c_mesh[2], 'C12', 'sensible', 'latent', coef_start, coef_end)
    plot_scalar_map(files_path_prefix, q_amount_little, corr_mesh, 'Correlation', 'sensible', 'latent', coef_start, coef_end)

    plot_drift_field(files_path_prefix,
                     np.linspace(1, q_amount_little, q_amount_little),
                     np.linspace(1, q_amount_little, q_amount_little),
                     a_mesh[0],
                     a_mesh[1],
                     'sensible',
                     'latent',
                     coef_start,
                     coef_end)

    plot_diffusion_ellipses(files_path_prefix,
                            np.linspace(1, q_amount_little, q_amount_little),
                            np.linspace(1, q_amount_little, q_amount_little),
                            c_mesh[0],
                            c_mesh[1],
                            c_mesh[2],
                            'sensible',
                            'latent',
                            coef_start, coef_end,
                            1,
                            1,
                            0.5,
                            )

    raise ValueError
    # -----------------------------------------------------------------------------------
    # check if there is analytical 2d solution
    # result = check_zero_current_solution(
    #     quantiles1=quantiles1,
    #     quantiles2=quantiles2,
    #     a1_mesh=a_mesh[0],
    #     a2_mesh=a_mesh[1],
    #     c11_mesh=c_mesh[0],
    #     c12_mesh=c_mesh[2],
    #     c22_mesh=c_mesh[1],
    #
    #     # Optional:
    #     counts_mesh=None,
    #     min_count=100,
    # )
    #
    # for name, value in result["summary"].items():
    #     print(f"{name}: {value}")
    # -----------------------------------------------------------------------------------
    # count 2d Fokker-Planck numerical solution
    from Functional.solve_2d_fokker_planck import (
        check_diffusion_tensor,
        build_fokker_planck_operator,
        solve_stationary_density,
        calculate_probability_current,
        calculate_stationary_moments,
        save_density_and_current_plots,
        solve_stationary_density_direct
    )

    # tensor_check = check_diffusion_tensor(
    #     c11_mesh=c_mesh[0],
    #     c12_mesh=c_mesh[2],
    #     c22_mesh=c_mesh[1],
    # )
    #
    # print(
    #     "Positive-definite fraction:",
    #     tensor_check["positive_definite_fraction"],
    # )
    #
    # print(
    #     "Minimum eigenvalue:",
    #     tensor_check["minimum_lambda"],
    # )
    #
    # print(
    #     "Maximum condition number:",
    #     tensor_check["maximum_condition_number"],
    # )

    fp_system = build_fokker_planck_operator(
        quantiles1=quantiles1,
        quantiles2=quantiles2,
        a1_mesh=a_mesh[0],
        a2_mesh=a_mesh[1],
        c11_mesh=c_mesh[0],
        c12_mesh=c_mesh[2],
        c22_mesh=c_mesh[1],

        # Recommended first choice.
        advection_scheme="upwind",
    )

    print(
        "Conservation error:",
        fp_system["conservation_error"],
    )

    stationary_density_direct, direct_info = (
        solve_stationary_density_direct(
            fp_system=fp_system,
            negative_mass_tolerance=1e-10,
            clip_tiny_negative_values=True,
            verbose=True,
        )
    )

    current_direct = calculate_probability_current(
        stationary_density=stationary_density_direct,
        fp_system=fp_system,
    )
    moments_direct = calculate_stationary_moments(
        stationary_density=stationary_density_direct,
        fp_system=fp_system,
    )

    for name in (
            "mean_x1",
            "mean_x2",
            "variance_x1",
            "variance_x2",
            "covariance_x1_x2",
            "correlation_x1_x2",
    ):
        print(
            name,
            "direct:",
            moments_direct[name],
        )

    save_density_and_current_plots(
        files_path_prefix=files_path_prefix,
        stationary_density=stationary_density_direct,
        current=current_direct,
        fp_system=fp_system,
        data1_name=data1_name,
        data2_name=data2_name,
        arrow_step=2,

        # False: arrow length shows current magnitude.
        # True: arrows show direction only.
        normalize_current_arrows=False,

        dpi=300,
    )

    p = stationary_density_direct
    J1 = current_direct["J1"]
    J2 = current_direct["J2"]

    # Exclude the very-low-density tails.
    density_threshold = 1e-3 * np.nanmax(p)
    supported = p >= density_threshold

    velocity_1 = np.full_like(p, np.nan)
    velocity_2 = np.full_like(p, np.nan)

    velocity_1[supported] = J1[supported] / p[supported]
    velocity_2[supported] = J2[supported] / p[supported]

    velocity_magnitude = np.hypot(
        velocity_1,
        velocity_2,
    )

    print(
        "Median current velocity:",
        np.nanmedian(velocity_magnitude),
    )

    print(
        "95% current-velocity quantile:",
        np.nanquantile(velocity_magnitude, 0.95),
    )
    plot_velocity(files_path_prefix, fp_system, velocity_magnitude, velocity_1, velocity_2, 'sensible', 'latent')

    # -----------------------------------------------------------------------------------
    A = fp_system["operator"].tocsc()

    rate_scale = max(
        np.max(np.abs(A.diagonal())),
        1.0,
    )

    eigenvalues = spla.eigs(
        A,
        k=8,
        sigma=-1e-10 * rate_scale,
        which="LM",
        return_eigenvectors=False,
    )

    # Order from closest to zero outward.
    eigenvalues = eigenvalues[
        np.argsort(np.abs(eigenvalues))
    ]

    print("Eigenvalues nearest zero:")

    for value in eigenvalues:
        print(
            f"{value.real:.12e} "
            f"{value.imag:+.12e}j"
        )
