# from imageio.config.plugins import summary
import os.path
import numpy as np
from Plotting.plot_coefficients import plot_velocity
files_path_prefix = 'D:/Nastya/Data/OceanFull/'
from Forecast_nn.utils import fix_random
from Functional.func_estimation import *
from Data_processing.data_processing import *
from Coefficients.semiparametric import *
import scipy.sparse.linalg as spla
from Plotting.plot_coefficients import *
from Plotting.plot_func_estimations import *
from Functional.copula_calculations import *
from Plotting.copula_plotting import *


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
    b_type = 'simple'

    start_year = 19792024
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
        start_year = 1979
        coef_start = 1
        end_year = 2024
        coef_end = 17106
    offset = coef_start

    missing_values_eigen = [3653, 7305, 10958, 14610]
    missing_days_coefs = [0, 7304, 10957]
    missing_days_eigen = [0, 3652, 7304, 10957, 14609]

    data1_array = np.load(files_path_prefix + f'DATA/Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
    data1_array = data1_array.transpose().reshape((-1, height, width))
    data2_array = np.load(files_path_prefix + f'DATA/Fluxes/latent_grouped_{start_year}-{end_year}.npy')
    data2_array = data2_array.transpose().reshape((-1, height, width))
    if end_year == 2024:
        data1_array = np.delete(data1_array, missing_days_eigen, axis=0)
        data2_array = np.delete(data2_array, missing_days_eigen, axis=0)

    # np.save(files_path_prefix + f'DATA/Fluxes/sensible_mean_{start_year}-{end_year}.npy', np.mean(data1_array, axis=0))
    # np.save(files_path_prefix + f'DATA/Fluxes/latent_mean_{start_year}-{end_year}.npy', np.mean(data2_array, axis=0))

    # plot_hist(files_path_prefix, data1_name + f'_{start_year}-{end_year}', data1_array)
    # print(f'min 1: {np.nanmin(data1_array)}')
    # print(f'max 1: {np.nanmax(data1_array)}')
    # plot_hist(files_path_prefix, data2_name + f'_{start_year}-{end_year}', data2_array)
    # print(f'min 2: {np.nanmin(data2_array)}')
    # print(f'max 2: {np.nanmax(data2_array)}')
    # -----------------------------------------------------------------------------------
    # create_quantiles_2d(files_path_prefix, data1_array, data2_array, data1_name, data2_name, mask, 260,
    #                     coef_start, coef_end, start_year, block_size, b_type)
    # raise ValueError
    quantiles1 = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_quantiles1.npy')
    quantiles2 = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_quantiles2.npy')
    a_grouped = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_a_grouped.npy')
    b_grouped = np.load(files_path_prefix + f'Functional/{str_types}/{coef_start}-{coef_end}_b_grouped_log_{b_type}.npy')
    # if b_type == 'eigen' and start_year == 1999:
    #     b_grouped = np.load(files_path_prefix + f'Functional/{str_types}/3654-7305_b_grouped_log_{b_type}.npy')

    data = [quantiles1, quantiles2, a_grouped, b_grouped]
    coefs = plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, b_type, start_year)
    # # raise ValueError
    sensible_params, latent_params = coefs
    # print('\n\n\n')

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
    z = trapezoid([prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)])
    # print(f'Sensible z: {z:.3e}')
    y1 = [prob_stationary(x) for x in np.linspace(-500, 500, 2500)] #simple
    p1_stationary = [prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)]

    x = np.linspace(-2000, 1500, 2500)
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
    plot_prob_and_hist(files_path_prefix, 'sensible',y1, None, np.linspace(-500, 500, 2500), start_year, data1_array[::10])
    #
    x0, x1, k, x_min, x_max, c1, c2, c3, c4, z = latent_params
    if b_type == 'eigen' and start_year == 1999:
        x_max = -1
    dL, dC, dR, prefL, prefC, prefR, const1, const2, const3, const4 = count_constants(latent_params)
    args = [c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4]
    z = trapezoid([prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)])
    # print(f'Latent z: {z:.3e}')
    y1 = [prob_stationary(x) for x in np.linspace(-800, 300, 2500)]
    p2_stationary = [prob_stationary(x) for x in np.linspace(-2000, 1500, 2500)]

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
    plot_prob_and_hist(files_path_prefix, 'latent', y1, None, np.linspace(-800, 300, 2500), start_year, data2_array[::10])
    raise ValueError
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

    data1_array = np.load(files_path_prefix + f'DATA/Fluxes/sensible_grouped_{start_year}-{end_year}.npy')
    data1_array = data1_array.transpose().reshape((-1, height, width))
    data2_array = np.load(files_path_prefix + f'DATA/Fluxes/latent_grouped_{start_year}-{end_year}.npy')
    data2_array = data2_array.transpose().reshape((-1, height, width))
    x1_stationary = np.linspace(-2000, 1500, 2500)
    x2_stationary = np.linspace(-2000, 1500, 2500)
    bernstein_results_by_m = {
        30: {'pearson_correlation': ..., 'covariance': ..., 'l1_x1': ..., 'l1_x2': ...},
        40: {'pearson_correlation': ..., 'covariance': ..., 'l1_x1': ..., 'l1_x2': ...},
        50: {'pearson_correlation': ..., 'covariance': ..., 'l1_x1': ..., 'l1_x2': ...},
        60: {'pearson_correlation': ..., 'covariance': ..., 'l1_x1': ..., 'l1_x2': ...},
        70: {'pearson_correlation': ..., 'covariance': ..., 'l1_x1': ..., 'l1_x2': ...}
    }

    for m in [50]:
        print(f'm = {m}')
        dependence = estimate_kendall_dependence(
            data1_array,
            data2_array,
            sample_size=300000,
            random_seed=12345,
        )

        rho = dependence["rho"]

        copula_result = build_gaussian_copula_stationary_density(
            x1_stationary,
            p1_stationary,
            x2_stationary,
            p2_stationary,
            rho,
        )

        joint_density = copula_result["joint_density"]

        marginal_x1, marginal_x2 = calculate_copula_marginals(
            joint_density,
            x1_stationary,
            x2_stationary,
        )

        moments = calculate_joint_density_moments(
            joint_density,
            x1_stationary,
            x2_stationary,
        )

        error_x1, error_x2 = calculate_marginal_errors(
            joint_density,
            x1_stationary,
            p1_stationary,
            x2_stationary,
            p2_stationary,
        )

        print()
        # print_copula_summary(dependence, moments, error_x1, error_x2)

        print()
        print(f"Mean X1: {moments['mean_x1']}")
        print(f"Mean X2: {moments['mean_x2']}")
        print(f"SD X1: {moments['sd_x1']}")
        print(f"SD X2: {moments['sd_x2']}")
        print(f"Covariance: {moments['covariance']}")
        print(f"Pearson correlation: {moments['correlation']}")

        plot_copula_marginals(files_path_prefix,
            x1_stationary,
            copula_result["p1"],
            marginal_x1,
            x2_stationary,
            copula_result["p2"],
            marginal_x2,
        )

        plot_joint_density(files_path_prefix,
            x1_stationary,
            x2_stationary,
            joint_density,
            title="Gaussian copula stationary density",
            xlim=(-60, 150),
            ylim=(-250, 50),
        )

        plot_joint_contours(files_path_prefix,
            x1_stationary,
            x2_stationary,
            joint_density,
            levels=15,
            title="Gaussian copula stationary density",
            xlim=(-60, 150),
            ylim=(-250, 50),
        )

        empirical_2d_density = build_empirical_2d_density(
            data1_array,
            data2_array,
            x1_stationary,
            x2_stationary,
        )

        plot_empirical_copula_comparison(files_path_prefix,
            x1_stationary,
            x2_stationary,
            empirical_2d_density,
            joint_density,
            xlim=(-60, 150),
            ylim=(-250, 50),
            same_scale=True,
        )

        plot_empirical_copula_contours(files_path_prefix,
            x1_stationary,
            x2_stationary,
            empirical_2d_density,
            joint_density,
            levels=12,
            xlim=(-60, 150),
            ylim=(-250, 50),
        )

        empirical_copula = build_empirical_copula_density(
            dependence["sampled_x1"],
            dependence["sampled_x2"],
            bins=60,
        )

        gaussian_copula = build_gaussian_copula_density_grid(
            dependence["rho"],
            empirical_copula["centers"],
        )

        copula_distances = compare_copula_densities(
            empirical_copula["density"],
            gaussian_copula,
            empirical_copula["centers"],
        )

        # print()
        # print(f"Gaussian copula TV: {copula_distances['total_variation']}")
        # print(f"Gaussian copula Hellinger: {copula_distances['hellinger']}")

        plot_copula_space_comparison(
            files_path_prefix,
            empirical_copula["centers"],
            empirical_copula["density"],
            gaussian_copula,
        )

        plot_copula_space_contours(
            files_path_prefix,
            empirical_copula["centers"],
            empirical_copula["density"],
            gaussian_copula,
            levels=12,
        )

        student_fit = fit_student_t_copula(
            empirical_copula["u"],
            empirical_copula["v"],
            dependence["rho"],
        )

        student_copula = build_student_t_copula_density_grid(
            student_fit["rho"],
            student_fit["df"],
            empirical_copula["centers"],
        )

        student_distances = compare_copula_densities(
            empirical_copula["density"],
            student_copula,
            empirical_copula["centers"],
        )

        tail_dependence = calculate_student_t_tail_dependence(
            student_fit["rho"],
            student_fit["df"],
        )

        # print(f"Student copula TV: {student_distances['total_variation']}")
        # print(f"Student copula Hellinger: {student_distances['hellinger']}")
        # print(f"Student copula tail dependence: {tail_dependence}")


        plot_copula_density(
            files_path_prefix,
            empirical_copula["centers"],
            empirical_copula["density"],
            "copula_empirical",
            "Empirical copula",
        )

        plot_copula_density(
            files_path_prefix,
            empirical_copula["centers"],
            gaussian_copula,
            "copula_gaussian",
            "Gaussian copula",
        )

        plot_copula_density(
            files_path_prefix,
            empirical_copula["centers"],
            student_copula,
            "copula_student_t",
            "Student t copula",
        )

        plot_copula_models_contours(
            files_path_prefix,
            empirical_copula["centers"],
            empirical_copula["density"],
            gaussian_copula,
            student_copula,
        )

        checkerboard = build_checkerboard_copula(
            empirical_copula["u"],
            empirical_copula["v"],
            bins=m,
        )

        checkerboard_result = build_checkerboard_stationary_density(
            x1_stationary,
            p1_stationary,
            x2_stationary,
            p2_stationary,
            checkerboard,
        )

        checkerboard_density = checkerboard_result["joint_density"]
        checkerboard_moments = checkerboard_result["moments"]

        print()
        print(f"Checkerboard mean X1: {checkerboard_moments['mean_x1']}")
        print(f"Checkerboard mean X2: {checkerboard_moments['mean_x2']}")
        print(f"Checkerboard SD X1: {checkerboard_moments['sd_x1']}")
        print(f"Checkerboard SD X2: {checkerboard_moments['sd_x2']}")
        print(f"Checkerboard covariance: {checkerboard_moments['covariance']}")
        print(f"Checkerboard Pearson correlation: {checkerboard_moments['correlation']}")
        print(f"Checkerboard L1 marginal error X1: {checkerboard_result['error_x1']}")
        print(f"Checkerboard L1 marginal error X2: {checkerboard_result['error_x2']}")
        bernstein_results_by_m[m]['pearson_correlation'] = checkerboard_moments['correlation']
        bernstein_results_by_m[m]['covariance'] = checkerboard_moments['covariance']
        bernstein_results_by_m[m]['l1_x1'] = checkerboard_result['error_x1']
        bernstein_results_by_m[m]['l1_x2'] = checkerboard_result['error_x2']

        plot_checkerboard_copula(
            files_path_prefix,
            checkerboard["centers"],
            checkerboard["density"],
        )

        plot_checkerboard_stationary_density(
            files_path_prefix,
            x1_stationary,
            x2_stationary,
            checkerboard_density,
            xlim=(-50, 150),
            ylim=(-250, 50),
        )

        plot_checkerboard_marginals(
            files_path_prefix,
            x1_stationary,
            checkerboard_result["p1"],
            checkerboard_result["marginal_x1"],
            x2_stationary,
            checkerboard_result["p2"],
            checkerboard_result["marginal_x2"],
        )

        empirical_2d_density = build_empirical_2d_density(
            data1_array,
            data2_array,
            x1_stationary,
            x2_stationary,
        )

        plot_empirical_checkerboard_contours(
            files_path_prefix,
            x1_stationary,
            x2_stationary,
            empirical_2d_density,
            checkerboard_density,
            levels=10,
            xlim=(-50, 150),
            ylim=(-250, 50),
        )

        bernstein = build_bernstein_copula_grid(
            checkerboard,
            grid_size=200,
        )

        bernstein_result = build_bernstein_stationary_density(
            x1_stationary,
            p1_stationary,
            x2_stationary,
            p2_stationary,
            checkerboard,
        )

        bernstein_density = bernstein_result["joint_density"]
        bernstein_moments = bernstein_result["moments"]

        print()
        print(f"Bernstein mean X1: {bernstein_moments['mean_x1']}")
        print(f"Bernstein mean X2: {bernstein_moments['mean_x2']}")
        print(f"Bernstein SD X1: {bernstein_moments['sd_x1']}")
        print(f"Bernstein SD X2: {bernstein_moments['sd_x2']}")
        print(f"Bernstein covariance: {bernstein_moments['covariance']}")
        print(f"Bernstein Pearson correlation: {bernstein_moments['correlation']}")
        print(f"Bernstein L1 marginal error X1: {bernstein_result['error_x1']}")
        print(f"Bernstein L1 marginal error X2: {bernstein_result['error_x2']}")


        plot_bernstein_copula(
            files_path_prefix,
            bernstein["grid"],
            bernstein["density"],
        )

        plot_bernstein_stationary_density(
            files_path_prefix,
            x1_stationary,
            x2_stationary,
            bernstein_density,
            xlim=(-50, 150),
            ylim=(-250, 50),
        )

        plot_bernstein_marginals(
            files_path_prefix,
            x1_stationary,
            bernstein_result["p1"],
            bernstein_result["marginal_x1"],
            x2_stationary,
            bernstein_result["p2"],
            bernstein_result["marginal_x2"],
        )

        plot_checkerboard_bernstein_comparison(
            files_path_prefix,
            checkerboard,
            bernstein,
        )
        print('--------------------------------------------------------------')

    m_values = sorted(bernstein_results_by_m.keys())
    corr_values = [bernstein_results_by_m[m]['pearson_correlation'] for m in m_values]
    cov_values = [bernstein_results_by_m[m]['covariance'] for m in m_values]
    l1_x1_values = [bernstein_results_by_m[m]['l1_x1'] for m in m_values]
    l1_x2_values = [bernstein_results_by_m[m]['l1_x2'] for m in m_values]

    print(f"m values: {m_values}")
    print(f"correlations: {corr_values}")
    print(f"covariances: {cov_values}")
    print(f"L1 X1: {l1_x1_values}")
    print(f"L1 X2: {l1_x2_values}")
    # plot_bernstein_m_sensitivity(files_path_prefix, m_values, corr_values, cov_values, l1_x1_values, l1_x2_values)