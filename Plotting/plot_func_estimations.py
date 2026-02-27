import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.cm
from numpy.polynomial import Polynomial
import datetime
import math
import scipy
import seaborn as sns

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
    axs[0].plot(x, scipy.stats.t.pdf(x, *sens_t),  label='Fitted t')
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


def func_sin(x, a, b, c,):
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
    fig, axs = plt.subplots(2, 1, figsize=(25, 10))
    x = np.linspace(min(x_full), max(x_full), 100)
    # sns.set_style('whitegrid')

    if data_name in ['sensible', 'pressure']:
        a_coeff_fit = np.polyfit(x_full, a, 1)
        b_coeff_fit = np.polyfit(x_full, b, 3)
    else: # latent
        a_coeff_fit = np.polyfit(x_full, a_full, 1)
        b_coeff_fit = np.polyfit(x_full, b_full, 2)

    a_poly_fit = np.poly1d(a_coeff_fit)
    b_poly_fit = np.poly1d(b_coeff_fit)

    # b_params_guess = [1, 0.005, 3*math.pi/2]
    # b_popt, b_pcov = scipy.optimize.curve_fit(func_sin, x_full, b, p0=b_params_guess, maxfev=10000)

    np.polynomial.set_default_printstyle('unicode')
    np.set_printoptions(precision=2, suppress=True)
    # axs[0].plot(quantiles, a, c='blue', label='dependence')
    axs[0].scatter(x_full, a_full, c='blue', label='dependence')
    # axs[0].plot(x, a_lin_fit(x), c='red', label=a_lin_fit.convert())
    axs[0].plot(x, a_poly_fit(x), c='cyan', label=a_poly_fit)
    axs[0].set_xlabel(data_name + ' values', fontsize=20)
    axs[0].set_ylabel('A', fontsize=20)
    axs[0].legend()


    # axs[1].plot(quantiles, b, c='blue', label='dependence')
    axs[1].scatter(x_full, b_full, c='blue', label='dependence')
    axs[1].plot(x, b_poly_fit(x), c='cyan', label=b_poly_fit)
    # axs[1].plot(x, func_sin(x, *b_popt), c='cyan', label='sin fit')
    axs[1].set_xlabel(data_name + ' values', fontsize=20)
    axs[1].set_ylabel('B', fontsize=20)
    axs[1].legend()
    fig.savefig(files_path_prefix + f'videos/Functional/{data_name}_scattered.png')
    # fig.savefig(files_path_prefix + f'videos/Functional/{data_name}.png')
    return

def plot_prob_1d(files_path_prefix: str,
                 data_name: str,
                 prob,
                 x):
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    # plt.xlabel('Значения явного потока тепла', fontsize=14)
    # plt.xlabel('Значения атмосферного давления', fontsize=14)
    plt.xlabel('Значения скрытого потока тепла', fontsize=14)
    plt.ylabel('Плотность стационарного распределения', fontsize=14)
    y = [prob(x0) for x0 in x]
    axs.plot(x, y)
    # mode = -12.2 # sensible
    # mode = 89300 # pressure
    mode = -120
    axs.axvline(x=mode, color='gray', linestyle='--', linewidth=2, label=f'x={mode}')
    axs.legend()
    fig.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Functional/{data_name}_prob_1d.png')
    return

