import datetime
import os
import pandas as pd

import tqdm
# from VarGamma import fit_ml, pdf, cdf
from numpy.polynomial import Polynomial
from symfit import parameters, sin, cos
from Plotting.plot_func_estimations import plot_estimate_residuals, plot_estimate_a_flux, plot_ab_functional_2d
from Data_processing.data_processing import load_ABCFE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.cm
from scipy.special import gamma, gammaincc
from Data_processing.data_processing import scale_to_bins, mean_blocks

months_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'}

font = {'size': 14}
matplotlib.rc('font', **font)

def func_sin(x, a, b, c, d, f, g, h):
    # return a * np.sin(b * x + c) + d + f * x + g * x**2 + h * x**3
    return a * np.sin(b * x + c)



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

        a_timelist, _, _, _, _, borders = load_ABCFE(files_path_prefix, time_start + 1, time_end + 1, load_a=True)
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

def estimate_A_B(files_path_prefix: str,
                 x: np.ndarray,
                 a:np.ndarray,
                 b:np.ndarray,
                 quantiles_amount: int = 250):
    x_grouped, _ = scale_to_bins(x, quantiles_amount)
    quantiles = np.unique(x_grouped)
    quantiles = quantiles[np.logical_not(np.isnan(quantiles))]
    quantiles = quantiles[quantiles != 0] #hotfix
    # quantiles = np.sort(quantiles)

    part = len(quantiles) // 20
    quantiles = quantiles[part:-part]

    a_grouped = np.zeros(len(quantiles))
    b_grouped = np.zeros(len(quantiles))
    a_full = list()
    b_full = list()
    x_full = list()

    print(len(quantiles))
    for g in tqdm.tqdm(range(len(quantiles))):
        quantile = quantiles[g]
        # data = x[x_grouped == quantile]
        if not np.isnan(quantile):
            # print(quantile)
            # print(np.sum((x_grouped==quantile)))
            a_grouped[g] = np.mean(a[x_grouped == quantile])
            b_grouped[g] = np.mean(b[x_grouped == quantile])
            a_full += list(a[x_grouped == quantile].flatten())
            b_full += list(b[x_grouped == quantile].flatten())
            x_full += list(x[x_grouped == quantile].flatten())

    return quantiles, a_grouped, b_grouped, x_full, a_full, b_full


def get_values(arr_grouped):
    quantiles = np.unique(arr_grouped)
    quantiles = quantiles[np.logical_not(np.isnan(quantiles))]
    quantiles = quantiles[quantiles != 0] #hotfix
    return quantiles

def estimate_A_B_2d(
                 x1: np.ndarray,
                x2: np.ndarray,
                 a:np.ndarray,
                 b:np.ndarray,
                 quantiles_amount: int = 500):
    little_q_amount = 40


    # x1_grouped, _ = scale_to_bins(x1, little_q_amount)
    # x2_grouped, _ = scale_to_bins(x2, little_q_amount)
    # quantiles_little1 = get_values(x1_grouped)
    # quantiles_little2 = get_values(x2_grouped)
    # part = len(quantiles_little1) // 20
    # quantiles_little1 = quantiles_little1[part:-part]
    # quantiles_little2 = quantiles_little2[part:-part]
    #
    # b_hist = np.zeros((len(quantiles_little1), len(quantiles_little1)))
    # # print(len(quantiles_little1))
    # for q1 in range(len(quantiles_little1)):
    #     quantile1 = quantiles_little1[q1]
    #     for q2 in range(len(quantiles_little2)):
    #         quantile2 = quantiles_little2[q2]
    #         b_hist[q1, q2] = np.mean(b[:, :, :, 1][np.logical_and(x1_grouped == quantile1, x2_grouped == quantile2)])


    x1_grouped, _ = scale_to_bins(x1, quantiles_amount)
    x2_grouped, _ = scale_to_bins(x2, quantiles_amount)
    quantiles1 = get_values(x1_grouped)
    quantiles2 = get_values(x2_grouped)

    # part = len(quantiles1) // 10
    part = 15
    # quantiles1 = quantiles1[part:-part]
    # quantiles2 = quantiles2[part:-part]

    quantiles1 = quantiles1[5:-5]
    quantiles2 = quantiles2[5:-5]

    a_grouped = np.zeros((2, len(quantiles1)))
    b_grouped = np.zeros((2, len(quantiles1)))

    a1_full = list()
    a2_full = list()

    b11_full = list()
    b22_full = list()

    x1_full = list()
    x2_full = list()
    # print(len(quantiles1))
    for q1 in range(len(quantiles1)):
        quantile1 = quantiles1[q1]
        a_grouped[0, q1] = np.mean(a[:, :, :, 0][x1_grouped == quantile1])
        b_grouped[0, q1] = np.mean(b[:, :, :, 0][x1_grouped == quantile1])

        # a_grouped[0, q1] = np.median(a[:, :, :, 0][x1_grouped == quantile1])
        # b_grouped[0, q1] = np.median(b[:, :, :, 0][x1_grouped == quantile1])

        a1_full += list(a[:, :, :, 0][x1_grouped == quantile1].flatten())
        b11_full += list(b[:, :, :, 0][x1_grouped == quantile1].flatten())
        x1_full += list(x1[x1_grouped == quantile1].flatten())

    for q2 in range(len(quantiles2)):
        quantile2 = quantiles2[q2]
        a_grouped[1, q2] = np.mean(a[:, :, :, 1][x2_grouped == quantile2])
        b_grouped[1, q2] = np.mean(b[:, :, :, 3][x2_grouped == quantile2])

        # a_grouped[1, q2] = np.median(a[:, :, :, 1][x2_grouped == quantile2])
        # b_grouped[1, q2] = np.median(b[:, :, :, 3][x2_grouped == quantile2])

        a2_full += list(a[:, :, :, 1][x2_grouped == quantile2].flatten())
        b22_full += list(b[:, :, :, 3][x2_grouped == quantile2].flatten())
        x2_full += list(x2[x2_grouped == quantile2].flatten())

    return (quantiles1, quantiles2, a_grouped, b_grouped, x1_full, x2_full, a1_full, a2_full, b11_full, b22_full,)
            # b_hist, quantiles_little1, quantiles_little2)

def kl_coeff(x, n, l):
    if n == 0:
        return 1 / l
    elif n == 1:
        return x / l - 1 / (l ** 2)
    elif n == 2:
        return x ** 2 / l - (2 * x) / (l ** 2) + 2 / (l ** 3)
    elif n == 3:
        return x ** 3 / l - (3 * x ** 2) / (l ** 2) + (6 * x) / (l ** 3) - 6 / (l ** 4)
    elif n == 4:
        return x ** 4 / l - (4 * x ** 3) / (l ** 2) + (12 * x ** 2) / (l ** 3) - (24 * x) / (l ** 4) + 24 / (l ** 5)
    else:
        raise ValueError("n must be 0..4")

def Phi(x, l, k):
    k0, k1, k2, k3, k4 = k
    return np.exp(l * x) * (
            k4 * kl_coeff(x, 4, l)
            + k3 * kl_coeff(x, 3, l)
            + k2 * kl_coeff(x, 2, l)
            + k1 * kl_coeff(x, 1, l)
            + k0 * kl_coeff(x, 0, l)
    )


def Psi_minus(x, c, k):
    tmp = 0.0
    for n in range(5):
        tmp += ((-1) ** n) * k[n] * c ** (-2 * n - 2) * gammaincc(2 * n + 2, c * np.sqrt(-x)) * gamma(2 * n + 2)
    return 2.0 * tmp


def Psi_plus(x, c, k):
    tmp = 0.0
    for n in range(5):
        tmp += k[n] * c ** (-2 * n - 2) * gammaincc(2 * n + 2, c * np.sqrt(x)) * gamma(2 * n + 2)
    return -2.0 * tmp


def I(x, args):
    c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4 = args
    dL = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
    dC = c3
    dR = c3 + (c2 - c4) * np.sqrt(abs(x0))

    prefL = 2 * np.exp(-dL)
    prefC = 2 * np.exp(-dC)
    prefR = 2 * np.exp(-dR)

    if x0 < 0:
        if x < x1:
            return prefL * Phi(x, c1, k) + const1
        elif x1 <= x < x0:
            return prefC * Psi_minus(x, c2, k) + const2
        elif x0 <= x < 0:
            return prefR * Psi_minus(x, c4, k) + const3
        else:
            return prefR * Psi_plus(x, c4, k) + const4

    else:
        if x < x1:
            return prefL * Phi(x, c1, k) + const1
        elif x1 <= x < 0:
            return prefC * Psi_minus(x, c2, k) + const2
        elif 0 <= x < x0:
            return prefC * Psi_plus(x, c2, k) + const3
        else:
            return prefR * Psi_plus(x, c4, k) + const4

def log_b2(x, args):
    c1, c2, c3, c4, x0, x1, k, const1, const2, const3, const4 = args
    dL = c3 + c2 * np.sqrt(abs(x1)) - c1 * abs(x1)
    dC = c3
    dR = c3 + (c2 - c4) * np.sqrt(abs(x0))

    if x < x1:
        return c1 * abs(x) + dL
    elif x1 <= x < x0:
        return c2 * np.sqrt(abs(x)) + dC
    else:
        return c4 * np.sqrt(abs(x)) + dR


def create_quantiles(files_path_prefix: str,
                     data1_array: np.ndarray,
                     data2_array: np.ndarray,
                     data1_name: str,
                     data2_name: str,
                     mask: np.ndarray):
    """

    :param files_path_prefix:
    :param data1_array:
    :param data2_array:
    :param data1_name: np.array with shape (t, height, width)
    :param data2_name:
    :param mask: np.array with shape (height, width)
    :return:
    """
    #estimate the functional a(X) and b(X) from data
    coef_start = 1
    coef_end = 3653
    block_size = 5
    season = 'full_10_years'
    start_year = 1979

    str_types = f'{data1_name}-{data2_name}'
    a_array = np.load(files_path_prefix + f'Components/{str_types}/Semi_2d/A_{coef_start}-{coef_end}.npy')
    b_array = np.load(files_path_prefix + f'Components/{str_types}/Semi_2d/B_{coef_start}-{coef_end}.npy')
    b_array = b_array**2


    data1_array[:, np.logical_not(mask)] = np.nan
    data2_array[:, np.logical_not(mask)] = np.nan
    a_array[:, np.logical_not(mask)] = np.nan
    b_array[:, np.logical_not(mask)] = np.nan


    # for year in range(start_year, start_year + 10):
    for year in [1979]:
        print(year)
        if season == 'year':
            start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=start_year, month=1,
                                                                                      day=1)).days
            end = (datetime.datetime(year=year, month=12, day=31) - datetime.datetime(year=start_year, month=1,
                                                                                      day=1)).days
        elif season == 'spring':
            start = (datetime.datetime(year=year, month=3, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
            end = (datetime.datetime(year=year, month=6, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
        elif season == 'summer':
            start = (datetime.datetime(year=year, month=6, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
            end = (datetime.datetime(year=year, month=9, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
        elif season == 'autumn':
            start = (datetime.datetime(year=year, month=9, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
            end = (datetime.datetime(year=year, month=12, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
        elif season == 'winter':
            start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
            end = (datetime.datetime(year=year, month=3, day=1) - datetime.datetime(year=start_year, month=1, day=1)).days
        else:
            start = 0
            end = 3650
        data1_small = mean_blocks(data1_array[start:end], mask, block_size)
        data2_small = mean_blocks(data2_array[start:end], mask, block_size)
        a_small = mean_blocks(a_array[start:end], mask, block_size)
        b_small = mean_blocks(b_array[start:end], mask, block_size)

        quantiles1, quantiles2, a_grouped, b_grouped, x1_full, x2_full, a1_full, a2_full, b11_full, b22_full, = (
            estimate_A_B_2d(data1_small, data2_small, a_small, b_small, quantiles_amount=260))

        data = [quantiles1, quantiles2, a_grouped, np.log(b_grouped), x1_full, x2_full, a1_full, a2_full, np.log(b11_full), np.log(b22_full)]
        # plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, season, year, scatter=True)
        plot_ab_functional_2d(files_path_prefix, data, data1_name, data2_name, season, year, scatter=False)
    return