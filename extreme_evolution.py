import os.path
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import matplotlib
import seaborn as sns
from scipy.optimize import curve_fit
import numpy, scipy.optimize
from data_processing import load_ABCF


def fit_linear(x, a, b):
    return a*x + b


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq",
    "period" and "fitfunc"'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2. ** 0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2. * numpy.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c): return A * numpy.sin(w * t + p) + c

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=10000)
    A, w, p, c = popt
    f = w / (2. * numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w * t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f, "fitfunc": fitfunc,
            "maxcov": numpy.max(pcov), "rawres": (guess, popt, pcov)}


def extract_extreme(files_path_prefix, timelist, coeff_type, time_start, time_end, mean_days=1):
    """
    Gets extreme characteristics from list with coefficient evolution and counts mean of this parameter, dividing the
    timeline in windows with length mean_days
    :param files_path_prefix: path to the working directory
    :param timelist: list with coefficient data
    :param coeff_type: 'a' or 'b'
    :param time_start: start day index, counting from 01.01.1979
    :param time_end: end day index
    :param mean_days: width of the window in days, in which mean is taken
    :return:
    """
    max_sens, min_sens, mean_sens, med_sens = list(), list(), list(), list()
    max_sens_points, min_sens_points = list(), list()
    max_lat, min_lat, mean_lat, med_lat = list(), list(), list(), list()
    max_lat_points, min_lat_points = list(), list()
    for t in tqdm.tqdm(range(0, time_end - time_start)):
        sens = timelist[t][0]
        max_sens.append(np.nanmax(sens))
        max_sens_points.append(np.nanargmax(sens))
        min_sens.append(np.nanmin(sens))
        min_sens_points.append(np.nanargmin(sens))
        mean_sens.append(np.nanmean(sens))
        med_sens.append(np.nanmedian(sens))

        lat = timelist[t][1]
        max_lat.append(np.nanmax(lat))
        max_lat_points.append(np.nanargmax(lat))
        min_lat.append(np.nanmin(lat))
        min_lat_points.append(np.nanargmin(lat))
        mean_lat.append(np.nanmean(lat))
        med_lat.append(np.nanmedian(lat))

    # take mean by the window with width = mean_days
    max_sens = [np.mean(max_sens[i:i + mean_days]) for i in range(0, len(max_sens) - mean_days, mean_days)]
    min_sens = [np.mean(min_sens[i:i + mean_days]) for i in range(0, len(min_sens) - mean_days, mean_days)]
    mean_sens = [np.mean(mean_sens[i:i + mean_days]) for i in range(0, len(mean_sens) - mean_days, mean_days)]
    med_sens = [np.mean(med_sens[i:i + mean_days]) for i in range(0, len(med_sens) - mean_days, mean_days)]

    max_lat = [np.mean(max_lat[i:i + mean_days]) for i in range(0, len(max_lat) - mean_days, mean_days)]
    min_lat = [np.mean(min_lat[i:i + mean_days]) for i in range(0, len(min_lat) - mean_days, mean_days)]
    mean_lat = [np.mean(mean_lat[i:i + mean_days]) for i in range(0, len(mean_lat) - mean_days, mean_days)]
    med_lat = [np.mean(med_lat[i:i + mean_days]) for i in range(0, len(med_lat) - mean_days, mean_days)]

    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy',
            np.array(max_sens))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy',
            np.array(min_sens))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_mean_sens({time_start}-{time_end})_{mean_days}.npy',
            np.array(mean_sens))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_med_sens({time_start}-{time_end})_{mean_days}.npy',
            np.array(med_sens))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_max_points_sens.npy', np.array(max_sens_points))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_min_points_sens.npy', np.array(min_sens_points))

    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_max_lat({time_start}-{time_end})_{mean_days}.npy',
            np.array(max_lat))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_min_lat({time_start}-{time_end})_{mean_days}.npy',
            np.array(min_lat))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_mean_lat({time_start}-{time_end})_{mean_days}.npy',
            np.array(mean_lat))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_med_lat({time_start}-{time_end})_{mean_days}.npy',
            np.array(med_lat))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_max_points_lat.npy', np.array(max_lat_points))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_min_points_lat.npy', np.array(min_lat_points))
    return


def plot_extreme(files_path_prefix: str, coeff_type: str, time_start: int, time_end: int, mean_days: int = 1):
    """
    Plots two pictures: first with evolution of max, min and mean of coefficient, second - the same, but adding
    approximation of max and min with sin function
    :param files_path_prefix: path to the working directory
    :param coeff_type: 'a' or 'b'
    :param time_start: start day index, counting from 01.01.1979
    :param time_end: end day index
    :param mean_days: width of the window in days, in which mean was taken
    :return:
    """
    font = {'size': 16}
    font_names = {'weight': 'bold', 'size': 20}
    matplotlib.rc('font', **font)
    sns.set_style("whitegrid")

    days = [datetime.datetime(1979, 1, 1) + datetime.timedelta(days=t) for t in
            range(time_start, time_end - mean_days, mean_days)]
    if os.path.exists(
            files_path_prefix + f'Extreme/data/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy'):
        max_sens = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy')
        min_sens = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy')
        mean_sens = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_mean_sens({time_start}-{time_end})_{mean_days}.npy')
        # med_sens = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_med_sens({time_start}-{time_end})_{mean_days}.npy')

        max_lat = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_max_lat({time_start}-{time_end})_{mean_days}.npy')
        min_lat = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_min_lat({time_start}-{time_end})_{mean_days}.npy')
        mean_lat = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_mean_lat({time_start}-{time_end})_{mean_days}.npy')
        # med_lat = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_med_lat({time_start}-{time_end})_{mean_days}.npy')

        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        # Major ticks every half year, minor ticks every month,
        # axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
        axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
        # axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
        axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
        axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
        axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))

        axs[0].set_title('Sensible', fontdict=font_names)
        axs[0].plot(days, max_sens, label='max', c='r', alpha=0.75)
        axs[0].plot(days, min_sens, label='min', c='b', alpha=0.75)
        axs[0].plot(days, mean_sens, label='mean', c='g')
        # axs[0].plot(days, med_sens, label='med', c='y')
        axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        axs[1].set_title('Latent', fontdict=font_names)
        axs[1].plot(days, max_lat, label='max', c='r', alpha=0.75)
        axs[1].plot(days, min_lat, label='min', c='b', alpha=0.75)
        axs[1].plot(days, mean_lat, label='mean', c='g')
        # axs[1].plot(days, med_lat, label='med', c='y')
        axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'Extreme/plots/{coeff_type}_({time_start}-{time_end})_{mean_days}.png')
        plt.close(fig)

        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
        axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
        axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
        axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))
        x = np.array(range(time_start, time_end - mean_days, mean_days))
        approx_string = f'A*sin({chr(969)}x + {chr(966)}) + c'
        fig.suptitle(f'{coeff_type} coefficient extreme', fontsize=20, fontweight='bold')

        axs[0].set_title('Sensible', fontdict=font_names)
        res = fit_sin(x, max_sens)
        rss = np.sqrt(np.sum((max_sens - res["fitfunc"](x)) ** 2)) / len(x)
        # string_fit = f"{res['amp']:.1f}*sin({res['omega']:.5f}*x + {res['phase']:.1f}) + {res['offset']:.1f}"
        axs[0].plot(days, res["fitfunc"](x), '--', c='darkviolet', label=f'{approx_string}\n MSE={rss:.2f}')
        axs[0].plot(days, max_sens, c='r', alpha=0.75, label='max')
        axs[0].plot(days, mean_sens, c='g', alpha=1, label='mean')

        res = fit_sin(x, min_sens)
        rss = np.sqrt(np.sum((min_sens - res["fitfunc"](x)) ** 2)) / len(x)
        # string_fit = f"{res['amp']:.1f}*sin({res['omega']:.5f}*x + {res['phase']:.1f}) + {res['offset']:.1f}"
        axs[0].plot(days, res["fitfunc"](x), '--', c='orange', label=f'{approx_string}\n MSE={rss:.2f}')
        axs[0].plot(days, min_sens, c='b', alpha=0.75, label='min')
        axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        axs[1].set_title('Latent', fontdict=font_names)
        res = fit_sin(x, max_lat)
        rss = np.sqrt(np.sum((max_lat - res["fitfunc"](x)) ** 2)) / len(x)
        # string_fit = f"{res['amp']:.1f}*sin({res['omega']:.5f}*x + {res['phase']:.1f}) + {res['offset']:.1f}"
        axs[1].plot(days, res["fitfunc"](x), '--', c='darkviolet', label=f'{approx_string}\n MSE={rss:.2f}')
        axs[1].plot(days, max_lat, c='r', alpha=0.75, label='max')

        axs[1].plot(days, mean_lat, c='g', alpha=1, label='mean')

        res = fit_sin(x, min_lat)
        rss = np.sqrt(np.sum((min_lat - res["fitfunc"](x)) ** 2)) / len(x)
        # string_fit = f"{res['amp']:.1f}*sin({res['omega']:.5f}*x + {res['phase']:.1f}) + {res['offset']:.1f}"
        axs[1].plot(days, res["fitfunc"](x), '--', c='orange', label=f'{approx_string}\n MSE={rss:.2f}')
        axs[1].plot(days, min_lat, c='b', alpha=0.75, label='min')
        axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'Extreme/plots/{coeff_type}_({time_start}-{time_end})_{mean_days}_fit.png')
    return


def check_conditions(files_path_prefix: str, time_start: int, time_end: int, sensible_all: np.ndarray,
                     latent_all: np.ndarray):
    """
    Estimates the constants K1 and K2 from the conditions on the fluxes from Skorohod book
    :param files_path_prefix: path to the working directory
    :param time_start: start day index, counting from 01.01.1979
    :param time_end: end day index
    :param sensible_all: np.array with sensible flux data
    :param latent_all: np.array with latent flux data
    :return:
    """
    mean_days = 1
    if os.path.exists(files_path_prefix + f'Extreme/data/a_max_sens({time_start}-{time_end})_{mean_days}.npy'):
        coeff_type = 'a'
        a_max_sens = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy')
        a_min_sens = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy')
        a_max_lat = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_max_lat({time_start}-{time_end})_{mean_days}.npy')
        a_min_lat = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_min_lat({time_start}-{time_end})_{mean_days}.npy')

        coeff_type = 'b'
        b_max_sens = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy')
        b_min_sens = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy')
        b_max_lat = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_max_lat({time_start}-{time_end})_{mean_days}.npy')
        b_min_lat = np.load(
            files_path_prefix + f'Extreme/data/{coeff_type}_min_lat({time_start}-{time_end})_{mean_days}.npy')

        sens_vals = list(np.unique(sensible_all))
        sens_vals.sort()
        sens_delim = min([abs(sens_vals[i + 1] - sens_vals[i]) if sens_vals[i + 1] != sens_vals[i] else 10
                          for i in range(0, len(sens_vals) - 1)])

        lat_vals = list(np.unique(latent_all))
        lat_vals.sort()
        lat_delim = min([abs(lat_vals[i + 1] - lat_vals[i]) if lat_vals[i + 1] != lat_vals[i] else 10
                         for i in range(0, len(lat_vals) - 1)])

        K1_sens, K1_lat = 0.0, 0.0
        K2_sens, K2_lat = 0.0, 0.0
        for t in tqdm.tqdm(range(0, time_end - time_start - 1)):
            K1_sens = max(K1_sens, (a_max_sens[t] - a_min_sens[t] + b_max_sens[t] - b_min_sens[t]) / sens_delim)
            K1_lat = max(K1_lat, (a_max_lat[t] - a_min_lat[t] + b_max_lat[t] - b_min_lat[t]) / lat_delim)

            K2_sens = max(K2_sens, (a_max_sens[t] ** 2 + b_max_sens[t] ** 2))
            K2_lat = max(K2_lat, (a_max_lat[t] ** 2 + b_max_lat[t] ** 2))

        K_array = np.array([K1_sens, K1_lat, K2_sens, K2_lat])
        np.save(files_path_prefix + f'Extreme/data/K_estimates.npy', K_array)
        print(K_array)
        print(max(K_array))
    return


def extract_extreme_coeff_flux(files_path_prefix: str,
                               coeff_type: str,
                               time_start: int,
                               time_end: int,
                               sensible_array: np.ndarray,
                               latent_array: np.ndarray,
                               window: int = 1):
    max_sens, min_sens, med_sens, max_lat, min_lat, med_lat = list(), list(), list(), list(), list(), list()
    for t in tqdm.tqdm(range(0, time_end - time_start - window, window)):
        a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCF(files_path_prefix,
                                                                                         time_start + t,
                                                                                         time_start + t + window,
                                                                                         load_a=True, load_b=True)
        if coeff_type == 'a':
            timelist = a_timelist
        else:
            timelist = b_timelist

        sens_coeff = np.zeros((sensible_array.shape[0], window))
        lat_coeff = np.zeros((latent_array.shape[0], window))
        for i in range(window):
            if coeff_type == 'a':
                sens_coeff[:, i] = timelist[i][0].flatten()
                lat_coeff[:, i] = timelist[i][1].flatten()
            else:
                sens_coeff[:, i] = timelist[i][0].flatten()
                lat_coeff[:, i] = timelist[i][3].flatten()

        sens_coeff = sens_coeff.flatten()
        lat_coeff = lat_coeff.flatten()

        max_points = np.nanargmax(sensible_array[:, t:t + window])
        min_points = np.nanargmin(sensible_array[:, t:t + window])
        med = np.nanmedian(sensible_array[:, t:t + window])
        med_points = np.flatnonzero(sensible_array[:, t:t + window] == med)

        max_sens.append(np.max(sens_coeff[max_points]))
        min_sens.append(np.min(sens_coeff[min_points]))
        med_sens.append(np.median(sens_coeff[med_points]))

        max_points = np.nanargmax(latent_array[:, t:t + window])
        min_points = np.nanargmin(latent_array[:, t:t + window])
        med = np.nanmedian(latent_array[:, t:t + window])
        med_points = np.flatnonzero(latent_array[:, t:t + window] == med)

        max_lat.append(np.max(lat_coeff[max_points]))
        min_lat.append(np.min(lat_coeff[min_points]))
        med_lat.append(np.median(lat_coeff[med_points]))

    np.save(files_path_prefix + f'Extreme/data/Flux_{coeff_type}_min_sens({time_start}-{time_end})_{window}.npy',
            min_sens)
    np.save(files_path_prefix + f'Extreme/data/Flux_{coeff_type}_max_sens({time_start}-{time_end})_{window}.npy',
            max_sens)
    np.save(files_path_prefix + f'Extreme/data/Flux_{coeff_type}_med_sens({time_start}-{time_end})_{window}.npy',
            med_sens)

    np.save(files_path_prefix + f'Extreme/data/Flux_{coeff_type}_min_lat({time_start}-{time_end})_{window}.npy',
            min_lat)
    np.save(files_path_prefix + f'Extreme/data/Flux_{coeff_type}_max_lat({time_start}-{time_end})_{window}.npy',
            max_lat)
    np.save(files_path_prefix + f'Extreme/data/Flux_{coeff_type}_med_lat({time_start}-{time_end})_{window}.npy',
            med_lat)
    return


def plot_extreme_coeff_flux(files_path_prefix: str,
         coeff_type: str,
         time_start: int,
         time_end: int,
         window: int = 1):
    min_sens = np.load(files_path_prefix + f'Extreme/data/Flux_{coeff_type}_min_sens({time_start}-{time_end})_{window}.npy')
    max_sens = np.load(files_path_prefix + f'Extreme/data/Flux_{coeff_type}_max_sens({time_start}-{time_end})_{window}.npy')
    med_sens = np.load(files_path_prefix + f'Extreme/data/Flux_{coeff_type}_med_sens({time_start}-{time_end})_{window}.npy')

    min_lat = np.load(files_path_prefix + f'Extreme/data/Flux_{coeff_type}_min_lat({time_start}-{time_end})_{window}.npy')
    max_lat = np.load(files_path_prefix + f'Extreme/data/Flux_{coeff_type}_max_lat({time_start}-{time_end})_{window}.npy')
    med_lat = np.load(files_path_prefix + f'Extreme/data/Flux_{coeff_type}_med_lat({time_start}-{time_end})_{window}.npy')

    font = {'size': 16}
    font_names = {'weight': 'bold', 'size': 20}
    matplotlib.rc('font', **font)
    sns.set_style("whitegrid")

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'{coeff_type}-flux dependence at extreme points \n window = {window} days', fontsize=20, fontweight='bold')
    axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
    axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
    axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
    axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))
    x = np.array(range(time_start, time_end - window, window))
    days = [datetime.datetime(1979, 1, 1) + datetime.timedelta(days=t) for t in range(time_start, time_end - window, window)]

    approx_string = f'A*sin({chr(969)}x + {chr(966)}) + c'
    axs[0].set_title('Sensible', fontdict=font_names)

    axs[0].plot(days, max_sens, c='r', label=f'at max points')
    # res = fit_sin(x, max_sens)
    # rss = np.sqrt(np.sum((max_sens - res["fitfunc"](x)) ** 2)) / len(x)
    # axs[0].plot(days, res["fitfunc"](x), '--', c='darkviolet', label=f'{approx_string}\n MSE={rss:.2f}')
    popt, pcov = curve_fit(fit_linear, x, max_sens)
    axs[0].plot(days, fit_linear(x, *popt), '--', c='darkviolet', label=f'Linear fit, k = {popt[0]:.1e}')


    axs[0].plot(days, min_sens, c='b', label=f'at min points')
    # res = fit_sin(x, min_sens)
    # rss = np.sqrt(np.sum((min_sens - res["fitfunc"](x)) ** 2)) / len(x)
    # axs[0].plot(days, res["fitfunc"](x), '--', c='orange', label=f'{approx_string}\n MSE={rss:.2f}')
    popt, pcov = curve_fit(fit_linear, x, min_sens)
    axs[0].plot(days, fit_linear(x, *popt), '--', c='orange', label=f'Linear fit, k = {popt[0]:.1e}')

    axs[0].plot(days, med_sens, c='y', label=f'at med points')
    axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    axs[1].set_title('Latent', fontdict=font_names)
    axs[1].plot(days, max_lat, c='r', label=f'at max points')
    # res = fit_sin(x, max_lat)
    # rss = np.sqrt(np.sum((max_lat - res["fitfunc"](x)) ** 2)) / len(x)
    # axs[1].plot(days, res["fitfunc"](x), '--', c='darkviolet', label=f'{approx_string}\n MSE={rss:.2f}')
    popt, pcov = curve_fit(fit_linear, x, max_lat)
    axs[1].plot(days, fit_linear(x, *popt), '--', c='darkviolet', label=f'Linear fit, k = {popt[0]:.1e}')

    axs[1].plot(days, min_lat, c='b', label=f'at min points')
    # res = fit_sin(x, min_lat)
    # rss = np.sqrt(np.sum((min_lat - res["fitfunc"](x)) ** 2)) / len(x)
    # axs[1].plot(days, res["fitfunc"](x), '--', c='orange', label=f'{approx_string}\n MSE={rss:.2f}')
    popt, pcov = curve_fit(fit_linear, x, min_lat)
    axs[1].plot(days, fit_linear(x, *popt), '--', c='orange', label=f'Linear fit, k = {popt[0]:.1e}')

    axs[1].plot(days, med_lat, c='y', label=f'at med points')
    axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(files_path_prefix + f'Extreme/plots/Flux/{coeff_type}_({time_start}-{time_end})_{window}_fit.png')
