import datetime
import os
import pandas as pd
import scipy.stats
import tqdm
# from VarGamma import fit_ml, pdf, cdf
from numpy.polynomial import Polynomial
from symfit import parameters, sin, cos
from Plotting.plot_func_estimations import plot_estimate_residuals, plot_estimate_a_flux
from Data_processing.data_processing import load_ABCFE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.cm
from Data_processing.data_processing import scale_to_bins

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
    a_grouped = np.zeros(len(quantiles))
    b_grouped = np.zeros(len(quantiles))
    for g in tqdm.tqdm(range(len(quantiles))):
        quantile = quantiles[g]
        # data = x[x_grouped == quantile]
        if not np.isnan(quantile):
            # print(quantile)
            # print(np.sum((x_grouped==quantile)))
            a_grouped[g] = np.mean(a[x_grouped == quantile])
            b_grouped[g] = np.mean(b[x_grouped == quantile])

    return quantiles, a_grouped, b_grouped
