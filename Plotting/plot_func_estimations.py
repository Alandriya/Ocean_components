import os.path

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.cm
from numpy.polynomial import Polynomial
import datetime
import math
import scipy
import seaborn as sns
from scipy.optimize import curve_fit

months_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'}

font = {'size': 14}
matplotlib.rc('font', **font)


def plot_estimate_residuals(files_path_prefix, month, sens_res, lat_res, t, postfix='sin'):
    part = len(sens_res) // 3 * 2
    sens_res = sens_res[np.isfinite(sens_res)]
    lat_res = lat_res[np.isfinite(lat_res)]

    sens_norm = scipy.stats.norm.fit_loc_scale(sens_res)
    lat_norm = scipy.stats.norm.fit_loc_scale(lat_res)
    shapiro_sens = scipy.stats.shapiro(sens_res[:part])
    shapiro_lat = scipy.stats.shapiro(lat_res[:part])

    print(f'Shapiro-Wilk normality test for sensible: {shapiro_sens[1]:.5f}')
    print(f'Shapiro-Wilk normality test for latent: {shapiro_lat[1]:.5f}\n')

    # sens_vargamma = fit_ml(sens_res[:part])
    # lat_vargamma = fit_ml(lat_res[:part])

    sens_t = scipy.stats.t.fit(sens_res[:part])
    lat_t = scipy.stats.t.fit(lat_res[:part])

    fig, axs = plt.subplots(1, 2, figsize=(25, 10))

    mu, sigma = sens_norm
    x = np.linspace(min(sens_res), max(sens_res), 100)
    # print(f'Kolmogorov-Smirnov test for VarGamma for sensible: {scipy.stats.kstest(sens_res[part:], cdf, sens_vargamma)[1]}')
    axs[0].cla()
    axs[0].hist(sens_res, bins=20, density=True)
    axs[0].plot(x, scipy.stats.norm.pdf(x, mu, sigma), label='Fitted normal')
    axs[0].plot(x, scipy.stats.t.pdf(x, *sens_t), label='Fitted t')
    # axs[0].plot(x, pdf(x, *sens_vargamma), label='Fitted VarGamma')
    axs[0].set_title(f'Residials for sensible \n Shapiro-Wiik test p-value = {shapiro_sens[1]:.5f}')
    axs[0].legend()

    mu, sigma = lat_norm
    x = np.linspace(min(lat_res), max(lat_res), 100)
    # print(f'Kolmogorov-Smirnov test for VarGamma for latent: {scipy.stats.kstest(lat_res[part:], cdf, lat_vargamma)[1]}\n')
    axs[1].cla()
    axs[1].hist(lat_res, bins=20, density=True)
    axs[1].plot(x, scipy.stats.norm.pdf(x, mu, sigma), label='Fitted normal')
    axs[1].plot(x, scipy.stats.t.pdf(x, *lat_t), label='Fitted t')
    # axs[1].plot(x, pdf(x, *lat_vargamma), label='Fitted VarGamma')
    axs[1].set_title(f'Residials for latent \n Shapiro-Wiik test p-value = {shapiro_lat[1]:.5f}')
    axs[1].legend()

    fig.savefig(files_path_prefix + f'Func_repr/a-flux-monthly/{month}/residuals_{t:05d}_{postfix}.png')
    return


def plot_estimate_a_flux(files_path_prefix: str,
                         a_timelist: list,
                         borders: list,
                         sensible_array: np.ndarray,
                         latent_array: np.ndarray,
                         time_start: int,
                         time_end: int,
                         step: int = 1,
                         month: int = 1,
                         flat_points=None,
                         point_center=None):
    """
    Estimates with scipy.optimize.fit_curve the dependence of A coefficient from flux values in shape of func, the
    estimation is carried on fixed month

    :param files_path_prefix: path to the working directory
    :param a_timelist: list with length = timesteps with structure [a_sens, a_lat], where a_sens and a_lat are np.arrays
        with shape (161, 181) with values for A coefficient for sensible and latent fluxes, respectively
    :param borders: min and max values of A, B and F to display on plot: assumed structure is
        [a_min, a_max, b_min, b_max, f_min, f_max].
    :param sensible_array: np.array with expected shape (161*181, days), where days amount may differ
    :param latent_array: np.array with expected shape (161*181, days), where days amount may differ
    :param time_start: offset in days from the beginning of the flux arrays for the first counted element
    :param time_end: offset in days from the beginning of the flux arrays for the last counted element
    :param step: step in time for loop
    :param month: month number from 1 to 12
    :param flat_points: 1d array of all coordinates of grid in bigger point
    :param point_center: coordinate (in 1d format) of the center of point bigger
    :return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(25, 10))
    date_start = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
    date_end = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_end)

    # sens_set = np.unique(sensible_array[flat_points, time_start:time_end])
    # lat_set = np.unique(latent_array[flat_points, time_start:time_end])
    # sens_set = sorted(list(sens_set))
    # lat_set = sorted(list(lat_set))

    # Getting x and y coordinates for scatter points
    time_window = 14
    step = 7
    sens_x, lat_x, sens_y, lat_y = list(), list(), list(), list()
    for t in range(time_start, time_end - time_window, step):
        a_sens_mean = np.zeros_like(a_timelist[t - time_start][0])
        a_lat_mean = np.zeros_like(a_timelist[t - time_start][1])
        for i in range(time_window):
            a_sens_mean += a_timelist[t - time_start + i][0]
            a_lat_mean += a_timelist[t - time_start + i][1]

        a_sens_mean /= time_window
        a_lat_mean /= time_window

        sens_x.append(np.mean(sensible_array[flat_points, t]))
        a_sens_point = [a_sens_mean[p // 181, p % 181] for p in flat_points]
        sens_y.append(np.mean(a_sens_point))

        lat_x.append(np.mean(latent_array[flat_points, t]))
        a_lat_point = [a_lat_mean[p // 181, p % 181] for p in flat_points]
        lat_y.append(np.mean(a_lat_point))

        # for p in flat_points:
        #     sens_x.append(sensible_array[p, t])
        #     sens_y.append(a_sens[p // 181, p % 181])
        #
        #     lat_x.append(latent_array[p, t])
        #     lat_y.append(a_lat[p // 181, p % 181])

    sens_x, sens_y = zip(*sorted(zip(sens_x, sens_y)))
    lat_x, lat_y = zip(*sorted(zip(lat_x, lat_y)))
    sens_x = np.array(sens_x)
    lat_x = np.array(lat_x)
    sens_y = np.array(sens_y)
    lat_y = np.array(lat_y)

    # transform
    # sens_x = -sens_x
    # sens_x += abs(min(sens_x)) + 1
    # sens_x = np.log(sens_x)
    # sens_x = np.power(sens_x, 0.5)
    # lat_x = -lat_x
    # lat_x += min(lat_x) + 1
    # lat_x = np.power(np.abs(lat_x), 0.3)

    # # cut tails in quantiles
    # q_bigger_tr = 1000
    # q_less_tr = 0
    # sens_y = sens_y[sens_x < q_bigger_tr]
    # sens_x = sens_x[sens_x < q_bigger_tr]
    # lat_y = lat_y[lat_x < q_bigger_tr]
    # lat_x = lat_x[lat_x < q_bigger_tr]
    # sens_y = sens_y[sens_x > q_less_tr]
    # sens_x = sens_x[sens_x > q_less_tr]
    # lat_y = lat_y[lat_x > q_less_tr]
    # lat_x = lat_x[lat_x > q_less_tr]

    # polynomial
    sens_fit = Polynomial.fit(sens_x, sens_y, 5)
    lat_fit = Polynomial.fit(lat_x, lat_y, 5)
    sens_residuals_poly = sens_y - sens_fit(sens_x)
    sens_ss_poly = np.sum(sens_residuals_poly ** 2)
    lat_residuals_poly = lat_y - lat_fit(lat_x)
    lat_ss_poly = np.sum(lat_residuals_poly ** 2)

    # # sin
    # # a * np.sin(b * x + c) + d + f * x + g * x**2 + h * x**3
    # params_guess = [5, (math.pi / 2) / 5, math.pi] + [0, 0, 0, 0]
    # sens_popt, sens_pcov = curve_fit(func_sin, sens_x, sens_y, p0=params_guess, maxfev=10000)
    # sens_residuals = sens_y - func_sin(sens_x, *sens_popt)
    # sens_ss = np.sum(sens_residuals ** 2)
    #
    # params_guess = [5, (math.pi / 2) / 5, 0] + [0, 0, 0, 0]
    # lat_popt, lat_pcov = curve_fit(func_sin, lat_x, lat_y, p0=params_guess, maxfev=10000)
    # lat_residuals = lat_y - func_sin(lat_y, *lat_popt)
    # lat_ss = np.sum(lat_residuals ** 2)

    # # Fourier
    # x, y = variables('x, y')
    # w, = parameters('w')
    # model_dict = {y: fourier_series(x, f=w, n=10)}
    # fit = Fit(model_dict, x=sens_x, y=sens_y)
    # fit_sens = fit.execute()
    # sens_popt_fourier = fit_sens.params
    # print(sens_popt_fourier)
    #
    # sens_residuals_fourier = sens_y - np.array(fit.model(sens_x, **sens_popt_fourier)).flat
    # lat_residuals_fourier = lat_y - 0

    # estimate and plot residuals
    # plot_estimate_residuals(files_path_prefix, month, sens_residuals, lat_residuals, time_start)
    plot_estimate_residuals(files_path_prefix, month, sens_residuals_poly, lat_residuals_poly, time_start, 'polynomial')
    # plot_estimate_residuals(files_path_prefix, month, sens_residuals_fourier, lat_residuals_fourier, time_start, 'fourier')

    # plot everything
    fig.suptitle(f'A-flux value dependence \n {date_start.strftime("%Y-%m-%d")} - {date_end.strftime("%Y-%m-%d")}',
                 fontsize=25)

    axs[0].cla()
    axs[0].scatter(sens_x, sens_y, c='r')

    # label_sin = f'{a:.1f} * sin({b:.5f} * x + {c:.1f}) + {d:.1f}'
    # axs[0].plot(sens_x, func_sin(sens_x, *sens_popt), c='b')

    # axs[0].plot(sens_x, np.array(fit.model(sens_x, **sens_popt_fourier)).flat, c='violet')

    label_polynomial = sens_fit
    axs[0].plot(sens_x, sens_fit(sens_x), c='cyan', label=label_polynomial)

    # axs[0].plot(sens_mean_x, sens_mean, c='g', label='mean')
    axs[0].set_ylabel('A_sens', fontsize=20)
    axs[0].set_xlabel('Sensible flux', fontsize=20)
    axs[0].set_ylim([borders[0], borders[1]])
    axs[0].set_title(f'SS (polynomial) = {sens_ss_poly: .1f}')
    axs[0].legend()

    axs[1].cla()
    axs[1].scatter(lat_x, lat_y, c='orange')

    # a, b, c, d = lat_popt
    # label_sin = f'{a:.1f} * sin({b:.5f} * x + {c:.1f}) + {d:.1f}'
    # axs[1].plot(lat_x, func_sin(lat_x, *lat_popt), c='b', label=label_sin)

    label_polynomial = lat_fit
    axs[1].plot(lat_x, lat_fit(lat_x), c='cyan', label=label_polynomial)

    # axs[1].plot(lat_mean_x, lat_mean, c='g', label='mean')
    axs[1].set_ylabel('A_lat', fontsize=20)
    axs[1].set_xlabel('Latent flux', fontsize=20)
    axs[1].set_ylim([borders[0], borders[1]])
    axs[1].set_title(f'SS (polynomial) = {lat_ss_poly:.1f}')
    axs[1].legend()
    fig.savefig(files_path_prefix + f'Func_repr/a-flux-monthly/{month}/{time_start:05d}.png')
    return sens_fit, lat_fit, sens_ss_poly, lat_ss_poly


def func_sin(x, a, b, c, ):
    return a * np.sin(b * x + c)


def plot_ab_functional(files_path_prefix: str,
                       quantiles: np.ndarray,
                       a: list,
                       b: list,
                       data_name: str,
                       x_full: list,
                       a_full: list,
                       b_full: list,
                       ):
    sns.set_style('whitegrid')
    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    x = np.linspace(min(quantiles), max(quantiles), 500)

    a_coeff_fit = np.polyfit(x_full, a_full, 1)
    b_coeff_fit = np.polyfit(x_full, b_full, 3)

    a_poly_fit = np.poly1d(a_coeff_fit)
    b_poly_fit = np.poly1d(b_coeff_fit)

    a_string = f'{a_coeff_fit[0]:.3e} * x + {a_coeff_fit[1]:.3e}'
    b_string = f'{b_coeff_fit[0]:.3e} * x^3 + {b_coeff_fit[1]:.3e} * x^2 + {b_coeff_fit[2]:.3e} * x + {b_coeff_fit[3]:.3e}'

    # b_params_guess = [1, 0.005, 3*math.pi/2]
    # b_popt, b_pcov = scipy.optimize.curve_fit(func_sin, x_full, b, p0=b_params_guess, maxfev=10000)

    np.polynomial.set_default_printstyle('unicode')
    np.set_printoptions(precision=2, suppress=True)

    axs[0].scatter(x_full, a_full, c='blue')
    axs[0].plot(quantiles, a, c='cyan', label='mean')
    # axs[0].plot(x, a_lin_fit(x), c='red', label=a_lin_fit.convert())
    axs[0].plot(x, a_poly_fit(x), c='red', label=a_string)
    axs[0].set_xlabel(data_name + ' values', fontsize=20)
    axs[0].set_ylabel('A', fontsize=20)
    axs[0].legend()

    axs[1].scatter(x_full, b_full, c='blue')
    axs[1].plot(quantiles, b, c='cyan', label='mean')
    axs[1].plot(x, b_poly_fit(x), c='red', label=b_string)
    # axs[1].plot(x, func_sin(x, *b_popt), c='cyan', label='sin fit')
    axs[1].set_xlabel(data_name + ' values', fontsize=20)
    axs[1].set_ylabel('B', fontsize=20)
    axs[1].legend()
    fig.savefig(files_path_prefix + f'videos/Functional/{data_name}_scattered.png')
    # fig.savefig(files_path_prefix + f'videos/Functional/{data_name}.png')
    return


def get_rmse(fit_func, x, y):
    return np.sqrt(np.sum((fit_func(x) - y)**2) *1.0/ len(x))

def fit_exp_left(x, c1):
    return c1 * np.abs(x)

def fit_exp_center(x, c2, c3):
    return c2 * np.sqrt(np.abs(x)) + c3

def fit_exp_right(x, c4):
    return c4 * np.sqrt(np.abs(x))


def plot_ab_functional_2d(files_path_prefix: str,
                          data: list,
                          data1_name: str,
                          data2_name: str,
                          season: str,
                          year: int,
                          scatter: bool = False,
                          ):
    quantiles1, quantiles2, a_grouped, b_grouped, x1_full, x2_full, a1_full, a2_full, b11_full, b22_full = data
    sns.set_style('whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    x1_interval = np.linspace(min(quantiles1), max(quantiles1), 500)
    x2_interval = np.linspace(min(quantiles2), max(quantiles2), 500)

    # a1_coeff_fit = np.polyfit(x1_full, a1_full, 2)
    a1_coeff_fit = np.polyfit(quantiles1, a_grouped[0], 4)
    a1_poly_fit = np.poly1d(a1_coeff_fit)
    print(f'A1 rmse: {get_rmse(a1_poly_fit, quantiles1, a_grouped[0]): .3e}')
    a1_string = (f'{a1_coeff_fit[0]:.3e} * x^4 + {a1_coeff_fit[1]:.3e} * x^3 + {a1_coeff_fit[2]:.3e} * x^2'
                       f' +{a1_coeff_fit[3]:.3e} * x + {a1_coeff_fit[4]:.3e}')
    print(a1_string)

    # a1_sorted = a_grouped[0][u1.argsort()]
    # u1 = np.sort(u1)
    # a1_coeff_fit = np.polyfit(u1, a1_sorted, 5)
    # a1_poly_fit = np.poly1d(a1_coeff_fit)
    # print(f'A1 rmse: {get_rmse(a1_poly_fit, u1, a1_sorted): .3e}')

    b1_argmin = quantiles1[b_grouped[0].argsort()][0]
    b1_x1 = -150
    print(b1_argmin)
    quantiles1_left = quantiles1[quantiles1 < b1_x1]
    b1_left = b_grouped[0][quantiles1 < b1_x1]
    quantiles1_center = quantiles1[(b1_x1 <= quantiles1) & (quantiles1 < b1_argmin)]
    b1_center = b_grouped[0][(b1_x1 <= quantiles1) & (quantiles1 < b1_argmin)]
    quantiles1_right = quantiles1[quantiles1 >= b1_argmin]
    b1_right = b_grouped[0][quantiles1 >= b1_argmin]

    popt1_center, _ = curve_fit(fit_exp_center, quantiles1_center - b1_argmin, b1_center, maxfev=5000)
    c2_1, c3_1 = popt1_center
    popt1_left, _ = curve_fit(fit_exp_left, quantiles1_left - b1_x1, b1_left - fit_exp_center(b1_x1, c2_1, c3_1), maxfev=5000)
    c1_1 = popt1_left[0]
    popt1_right, _ = curve_fit(fit_exp_right, quantiles1_right - b1_argmin, b1_right - c3_1, maxfev=5000)
    c4_1 = popt1_right[0]

    # left_error = np.sum(fit_exp_left(quantiles1_left, *popt1_left) - (b1_left - fit_exp_center(b1_x1, c2_1, c3_1)))**2
    # center_error = np.sum(fit_exp_center(quantiles1_center, *popt1_center) - b1_center)**2
    # right_error = np.sum(fit_exp_right(quantiles1_right, *popt1_right) - (b1_right - c3_1))**2
    # print(f'B11 rmse: {np.sqrt(left_error + + center_error + right_error) *1.0/ len(quantiles1): .3e}')
    b11_left_string = f'{c1_1:.3e} * |x| + {c3_1:.3e}'
    b11_center_string = f'{c2_1:.3e} * sqrt(|x|)  + {c3_1:.3e}'
    b11_right_string = f'{c4_1:.3e} * sqrt(|x|) + {c3_1:.3e}'
    print(b11_left_string)
    print(b11_center_string)
    print(b11_right_string)

    a2_coeff_fit = np.polyfit(quantiles2, a_grouped[1], 4)
    a2_poly_fit = np.poly1d(a2_coeff_fit)
    print(f'A2 rmse: {get_rmse(a2_poly_fit, quantiles2, a_grouped[1]): .3e}')
    a2_string = (f'{a2_coeff_fit[0]:.3e} * x^4 + {a2_coeff_fit[1]:.3e} * x^3 + {a2_coeff_fit[2]:.3e} * x^2'
                       f' +{a2_coeff_fit[3]:.3e} * x + {a2_coeff_fit[4]:.3e}')
    print(a2_string)

    b2_x1 = -500
    b2_argmin = quantiles2[b_grouped[1].argsort()][0]
    print(b2_argmin)

    quantiles2_left = quantiles2[quantiles2 < b2_x1]
    b2_left = b_grouped[1][quantiles2 < b2_x1]
    quantiles2_center = quantiles2[(b2_x1 <= quantiles2) & (quantiles2 < b2_argmin)]
    b2_center = b_grouped[1][(b2_x1 <= quantiles2) & (quantiles2 < b2_argmin)]
    quantiles2_right = quantiles2[quantiles2 >= b2_argmin]
    b2_right = b_grouped[1][quantiles2 >= b2_argmin]

    popt2_center, _ = curve_fit(fit_exp_center, quantiles2_center - b2_argmin, b2_center, maxfev=5000)
    c2_2, c3_2 = popt2_center
    popt2_left, _ = curve_fit(fit_exp_left, quantiles2_left - b2_x1, b2_left - fit_exp_center(b2_x1, c2_2, c3_2), maxfev=5000)
    c1_2 = popt2_left[0]
    popt2_right, _ = curve_fit(fit_exp_right, quantiles2_right - b2_argmin, b2_right - c3_2, maxfev=5000)
    c4_2 = popt2_right[0]

    # left_error = np.sum(fit_exp_left(quantiles2_left, *popt2_left) - (b2_left - fit_exp_center(b2_x1, c2_2, c3_2))) ** 2
    # center_error = np.sum(fit_exp_center(quantiles2_center, *popt2_center) - b2_center) ** 2
    # right_error = np.sum(fit_exp_right(quantiles2_right, *popt2_right) - (b2_right - c3_2)) ** 2
    # print(f'B22 rmse: {np.sqrt(left_error + + center_error + right_error) * 1.0 / len(quantiles2): .3e}')
    b22_left_string = f'{c1_2:.3e} * |x| + {c3_2:.3e}'
    b22_center_string = f'{c2_2:.3e} * sqrt(|x|)  + {c3_2:.3e}'
    b22_right_string = f'{c4_2:.3e} * sqrt(|x|) + {c3_2:.3e}'
    print(b22_left_string)
    print(b22_center_string)
    print(b22_right_string)


    # b22_left_coeff_fit = np.polyfit(quantiles2_left, b2_left, 4)
    # b22_left_poly_fit = np.poly1d(b22_left_coeff_fit)
    # b22_right_coeff_fit = np.polyfit(quantiles2_right, b2_right, 1)
    # b22_right_poly_fit = np.poly1d(b22_right_coeff_fit)
    #
    # left_error = (np.sum(b22_left_poly_fit(quantiles2_left) - b2_left)**2)
    # right_error = (np.sum(b22_right_poly_fit(quantiles2_right) - b2_right)**2)
    # print(f'B22 rmse: {np.sqrt(left_error + right_error) *1.0/ len(quantiles2): .3e}')
    #
    # b22_left_string = (f'{b22_left_coeff_fit[0]:.3e} * x^4 + {b22_left_coeff_fit[1]:.3e} * x^3 + '
    #                    f'{b22_left_coeff_fit[2]:.3e} * x^2 + {b22_left_coeff_fit[3]:.3e} * x + {b22_left_coeff_fit[4]:.3e}')
    # b22_right_string = f'{b22_right_coeff_fit[0]:.3e} * x + {b22_right_coeff_fit[1]:.3e}'

    np.polynomial.set_default_printstyle('unicode')
    np.set_printoptions(precision=2, suppress=True)

    x1_left = x1_interval[x1_interval < b1_x1]
    x1_center = x1_interval[(x1_interval >= b1_x1) & (x1_interval < b1_argmin)]
    x1_right = x1_interval[x1_interval >= b1_argmin]

    x2_left = x2_interval[x2_interval < b2_x1]
    x2_center = x2_interval[(x2_interval >= b2_x1) & (x2_interval < b2_argmin)]
    x2_right = x2_interval[x2_interval >= b2_argmin]

    if scatter:
        axs[0, 0].scatter(x1_full[::1000], a1_full[::1000], c='blue')
        axs[0, 1].scatter(x2_full[::1000], a2_full[::1000], c='blue')
        axs[1, 0].scatter(x1_full[::1000], b11_full[::1000], c='blue')
        axs[1, 1].scatter(x2_full[::1000], b22_full[::1000], c='blue')

    axs[0, 0].plot(quantiles1, a_grouped[0], c='cyan', label='mean')
    axs[0, 0].plot(x1_interval, a1_poly_fit(x1_interval), c='red', label=a1_string)
    axs[0, 0].set_xlabel(data1_name + ' values', fontsize=20)
    axs[0, 0].set_ylabel('A', fontsize=20)
    axs[0, 0].legend()
    sns.move_legend(axs[0, 0], loc='upper center', bbox_to_anchor=(0.5, 1.1))

    axs[0, 1].plot(quantiles2, a_grouped[1], c='cyan', label='mean')
    axs[0, 1].plot(x2_interval, a2_poly_fit(x2_interval), c='red', label=a2_string)
    axs[0, 1].set_xlabel(data2_name + ' values', fontsize=20)
    axs[0, 1].set_ylabel('A', fontsize=20)
    axs[0, 1].legend()
    sns.move_legend(axs[0, 1], loc='upper center', bbox_to_anchor=(0.5, 1.1))

    axs[1, 0].plot(quantiles1, b_grouped[0], c='cyan', label='mean')
    axs[1, 0].plot(x1_left, fit_exp_left(x1_left - b1_x1, *popt1_left) + fit_exp_center(b1_x1, c2_1, c3_1), c='orange', label=b11_left_string)
    axs[1, 0].plot(x1_center, fit_exp_center(x1_center - b1_argmin, *popt1_center), c='purple', label=b11_center_string)
    axs[1, 0].plot(x1_right, fit_exp_right(x1_right - b1_argmin, *popt1_right) + c3_1, c='red', label=b11_right_string)
    axs[1, 0].set_xlabel(data1_name + ' values', fontsize=20)
    axs[1, 0].set_ylabel('log(B)', fontsize=20)
    axs[1, 0].legend()
    sns.move_legend(axs[1, 0], loc='upper center', bbox_to_anchor=(0.5, 1.2))

    axs[1, 1].plot(quantiles2, b_grouped[1], c='cyan', label='mean')
    axs[1, 1].plot(x2_left, fit_exp_left(x2_left - b2_x1, *popt2_left) + fit_exp_center(b2_x1, c2_2, c3_2), c='orange', label=b22_left_string)
    axs[1, 1].plot(x2_center, fit_exp_center(x2_center - b2_argmin, *popt2_center), c='purple', label=b22_center_string)
    axs[1, 1].plot(x2_right, fit_exp_right(x2_right - b2_argmin, *popt2_right) + c3_2, c='red', label=b22_right_string)
    axs[1, 1].set_xlabel(data2_name + ' values', fontsize=20)
    axs[1, 1].set_ylabel('log(B)', fontsize=20)
    axs[1, 1].legend()
    sns.move_legend(axs[1, 1], loc='upper center', bbox_to_anchor=(0.5, 1.2))

    plt.subplots_adjust(hspace=0.35)
    if not os.path.exists(files_path_prefix + f'videos/Functional/{season}'):
        os.mkdir(files_path_prefix + f'videos/Functional/{season}')
    if scatter:
        fig.savefig(files_path_prefix + f'videos/Functional/{season}/{data1_name}-{data2_name}_scattered_{year}.png')
    else:
        fig.savefig(files_path_prefix + f'videos/Functional/{season}/{data1_name}-{data2_name}_{year}.png')
    return

def plot_heatmap(files_path_prefix: str,
                          data1_name: str,
                          data2_name: str,
                 quantiles1: np.ndarray,
                 quantiles2: np.ndarray,
        b_hist: np.array,):
    fig, axs = plt.subplots(1, 1, figsize=(15, 15))
    sns.heatmap(b_hist, xticklabels=quantiles1, yticklabels=quantiles2)
    fig.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Functional/{data1_name}-{data2_name}_heatmap.png')


def plot_prob_1d(files_path_prefix: str,
                 data_name: str,
                 prob,
                 x):
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    plt.xlabel(f'Значения {data_name}', fontsize=14)
    plt.ylabel('Плотность стационарного распределения', fontsize=14)
    y = [prob(t) for t in x]
    axs.plot(x, y)
    # mode = 89300 # pressure
    # mode = -212 # sensible 1
    # mode = 1245.30 # sensible 2
    # axs.axvline(x=mode, color='gray', linestyle='--', linewidth=2, label=f'x={mode}')
    axs.legend()
    fig.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Functional/{data_name}_prob_1d.png')
    return


def plot_hist(files_path_prefix: str,
              data_name: str,
              x:np.ndarray):
    data = x[np.logical_not(np.isnan(x))].flatten()
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    # plt.xlabel(f'Значения {data_name}', fontsize=14)
    plt.xlabel(f'Differences of {data_name}', fontsize=14)
    axs.hist(data, bins=100)
    axs.legend()
    fig.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Functional/{data_name}_hist.png')
    return
