import matplotlib.pyplot as plt
import matplotlib
import tqdm
import datetime
import numpy as np
from scipy.optimize import curve_fit
import math
import matplotlib.ticker as mtick
import pandas as pd
import os
from data_processing import load_ABCF
import matplotlib.cm
import scipy.stats
from VarGamma import fit_ml, pdf, cdf, fit_moments
from data_processing import load_prepare_fluxes
import plotly.express as px
from symfit import parameters, variables, sin, cos, Fit
from numpy.polynomial import Polynomial


months_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'}

font = {'size': 14}
matplotlib.rc('font', **font)


def func_sin(x, a, b, c, d, f, g, h):
    return a * np.sin(b * x + c) + d + f * x + g * x**2 + h * x**3


def fourier_series(x, f, n=10):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series


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

    sens_vargamma = fit_ml(sens_res[:part])
    lat_vargamma = fit_ml(lat_res[:part])

    sens_t = scipy.stats.t.fit(sens_res[:part])
    lat_t = scipy.stats.t.fit(lat_res[:part])

    fig, axs = plt.subplots(1, 2, figsize=(25, 10))

    mu, sigma = sens_norm
    x = np.linspace(min(sens_res), max(sens_res), 100)
    print(f'Kolmogorov-Smirnov test for VarGamma for sensible: {scipy.stats.kstest(sens_res[part:], cdf, sens_vargamma)[1]}')
    axs[0].cla()
    axs[0].hist(sens_res, bins=20, density=True)
    axs[0].plot(x, scipy.stats.norm.pdf(x, mu, sigma), label='Fitted normal')
    axs[0].plot(x, scipy.stats.t.pdf(x, *sens_t),  label='Fitted t')
    axs[0].plot(x, pdf(x, *sens_vargamma), label='Fitted VarGamma')
    axs[0].set_title(f'Residials for sensible \n Shapiro-Wiik test p-value = {shapiro_sens[1]:.5f}')
    axs[0].legend()

    mu, sigma = lat_norm
    x = np.linspace(min(lat_res), max(lat_res), 100)
    print(f'Kolmogorov-Smirnov test for VarGamma for latent: {scipy.stats.kstest(lat_res[part:], cdf, lat_vargamma)[1]}\n')
    axs[1].cla()
    axs[1].hist(lat_res, bins=20, density=True)
    axs[1].plot(x, scipy.stats.norm.pdf(x, mu, sigma), label='Fitted normal')
    axs[1].plot(x, scipy.stats.t.pdf(x, *lat_t), label='Fitted t')
    axs[1].plot(x, pdf(x, *lat_vargamma), label='Fitted VarGamma')
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
        a_sens_mean = np.zeros_like(a_timelist[t-time_start][0])
        a_lat_mean = np.zeros_like(a_timelist[t-time_start][1])
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
    sens_ss_poly = np.sum(sens_residuals_poly**2)
    lat_residuals_poly = lat_y - lat_fit(lat_x)
    lat_ss_poly = np.sum(lat_residuals_poly**2)

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


def estimate_a_flux_by_months(files_path_prefix: str, month: int, point, radius):
    """
    Estimates the dependence of A coefficient from flux values in shape of func, the estimation is carried on all data
    of fixed month: e.g, all Januaries, all Februaries, ...

    :param files_path_prefix: path to the working directory
    :param month: month number from 1 to 12
    :return:
    """
    sensible_array = np.load(files_path_prefix + 'sensible_grouped_1979-1989(scaled).npy')
    latent_array = np.load(files_path_prefix + 'latent_grouped_1979-1989(scaled).npy')

    biases = [i for i in range(-radius, radius+1)]
    point_bigger = [(point[0] + i, point[1] + j) for i in biases for j in biases]
    flat_points = np.array([p[0] * 181 + p[1] for p in point_bigger])

    df_sens = pd.DataFrame(columns=['dates', 'a', 'b', 'c', 'd', 'ss'])
    df_lat = pd.DataFrame(columns=['dates', 'a', 'b', 'c', 'd', 'ss'])

    if not os.path.exists(files_path_prefix + f"Func_repr/a-flux-monthly/{month}"):
        os.mkdir(files_path_prefix + f"Func_repr/a-flux-monthly/{month}")

    years = 10
    max_year = 1988

    sens_fits, lat_fits = list(), list()
    for i in range(0, years):
        time_start = (datetime.datetime(1979 + i, month, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
        if month != 12:
            time_end = (datetime.datetime(1979 + i + 2, month + 0, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
        else:
            time_end = (datetime.datetime(1979 + i + 1, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days

        a_timelist, _, _, _, _, borders = load_ABCF(files_path_prefix, time_start + 1, time_end + 1, load_a=True)
        sens_fit, lat_fit, sens_err, lat_err = plot_estimate_a_flux(files_path_prefix, a_timelist, borders,
                                                                          sensible_array, latent_array, time_start,
                                                                          time_end, month=month, flat_points=flat_points,
                                                                          point_center=point[0] * 181 + point[1])
        del a_timelist
        date_start = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
        date_end = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
        sens_fits.append(sens_fit)
        lat_fits.append(lat_fit)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))
    x = np.linspace(np.nanmin(sensible_array), np.nanmax(sensible_array), 100)
    for i in range(0, 10):
        # sens_params = df_sens[['a', 'b', 'c', 'd']].loc[i].values
        # lat_params = df_lat[['a', 'b', 'c', 'd']].loc[i].values
        if i > 30:
            color = plt.cm.tab20(i % 20)
        else:
            color = 'gray'
        axes[0].plot(x, sens_fits[i](x), label=f'{1979 + i}', c=color)
        axes[1].plot(x, lat_fits[i](x), label=f'{1979 + i}', c=color)

    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1.0), ncol=2, fancybox=True, shadow=True)
    axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1.0), ncol=2, fancybox=True, shadow=True)
    axes[0].set_title('Sensible')
    axes[1].set_title('Latent')
    fig.suptitle(months_names[month])
    fig.tight_layout()
    fig.savefig(files_path_prefix + f"Func_repr/a-flux-monthly/{months_names[month]}.png")
    plt.close(fig)

    fig, axs = plt.subplots(figsize=(10, 5))
    axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.bar(range(1979, max_year), df_sens['ss'])
    fig.suptitle('Sensible - sum of squared residuals')
    fig.savefig(files_path_prefix + f"Func_repr/a-flux-monthly/{month}/{month}_error_sensible.png")
    plt.clf()

    axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.bar(range(1979, max_year), df_lat['ss'])
    fig.suptitle('Latent - sum of squared residuals')
    fig.savefig(files_path_prefix + f"Func_repr/a-flux-monthly/{month}/{month}_error_latent.png")
    plt.close(fig)
    return





