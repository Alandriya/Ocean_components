import os
import os.path

import numpy
import numpy as np
import scipy.optimize
import tqdm
from scipy.optimize import curve_fit
from symfit import parameters, variables, sin, cos, Fit

from data_processing import load_ABCFE


def fit_linear(x, a, b):
    return a*x + b

def rmse(x, y):
    return np.sqrt(np.sum((x-y)**2))

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


def fit_fourier(sens_x, lat_x, sens_res, lat_res):
    x, y = variables('x, y')
    w, = parameters('w')
    model_dict = {y: fourier_series(x, f=w, n=10)}

    fit = Fit(model_dict, x=sens_x, y=sens_res)
    fit_sens = fit.execute()
    fit_sens_array = np.array(fit.model(sens_x, **fit_sens.params)).flat

    fit = Fit(model_dict, x=lat_x, y=lat_res)
    fit_lat = fit.execute()
    fit_lat_array = np.array(fit.model(lat_x, **fit_lat.params)).flat

    return fit_sens_array, fit_lat_array


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


def extract_extreme(files_path_prefix: str,
                    timelist: list,
                    coeff_type: str,
                    time_start: int,
                    time_end: int,
                    mean_days: int = 1,
                    local_path_prefix:str = ''):
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

    if not os.path.exists(files_path_prefix + 'Extreme'):
        os.mkdir(files_path_prefix + 'Extreme')
    if not os.path.exists(files_path_prefix + 'Extreme/data'):
        os.mkdir(files_path_prefix + 'Extreme/data')
    if not os.path.exists(files_path_prefix + f'Extreme/data/{local_path_prefix}'):
        os.mkdir(files_path_prefix + f'Extreme/data/{local_path_prefix}')

    if coeff_type == 'raw' or 'diff':
        for t in tqdm.tqdm(range(0, time_end - time_start)):
            if t >= timelist.shape[1]:
                break
            sens = timelist[:, t]
            max_sens.append(np.nanmax(sens))
            max_sens_points.append(np.nanargmax(sens))
            min_sens.append(np.nanmin(sens))
            min_sens_points.append(np.nanargmin(sens))
            mean_sens.append(np.nanmean(sens))
            med_sens.append(np.nanmedian(sens))

        max_sens = [np.mean(max_sens[i:i + mean_days]) for i in range(0, len(max_sens) - mean_days, mean_days)]
        min_sens = [np.mean(min_sens[i:i + mean_days]) for i in range(0, len(min_sens) - mean_days, mean_days)]
        mean_sens = [np.mean(mean_sens[i:i + mean_days]) for i in range(0, len(mean_sens) - mean_days, mean_days)]
        med_sens = [np.mean(med_sens[i:i + mean_days]) for i in range(0, len(med_sens) - mean_days, mean_days)]

        np.save(
            files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy',
            np.array(max_sens))
        np.save(
            files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy',
            np.array(min_sens))
        np.save(
            files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_mean_sens({time_start}-{time_end})_{mean_days}.npy',
            np.array(mean_sens))
        np.save(
            files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_med_sens({time_start}-{time_end})_{mean_days}.npy',
            np.array(med_sens))
        np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_max_points_sens.npy',
                np.array(max_sens_points))
        np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_min_points_sens.npy',
                np.array(min_sens_points))
        return

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

    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy',
            np.array(max_sens))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy',
            np.array(min_sens))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_mean_sens({time_start}-{time_end})_{mean_days}.npy',
            np.array(mean_sens))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_med_sens({time_start}-{time_end})_{mean_days}.npy',
            np.array(med_sens))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_max_points_sens.npy', np.array(max_sens_points))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_min_points_sens.npy', np.array(min_sens_points))

    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_max_lat({time_start}-{time_end})_{mean_days}.npy',
            np.array(max_lat))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_min_lat({time_start}-{time_end})_{mean_days}.npy',
            np.array(min_lat))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_mean_lat({time_start}-{time_end})_{mean_days}.npy',
            np.array(mean_lat))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_med_lat({time_start}-{time_end})_{mean_days}.npy',
            np.array(med_lat))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_max_points_lat.npy', np.array(max_lat_points))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_min_points_lat.npy', np.array(min_lat_points))
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
        a_timelist, b_timelist, c_timelist, f_timelist, fs_timelist, borders = load_ABCFE(files_path_prefix,
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


def collect_extreme(files_path_prefix: str,
                    coeff_type: str,
                    local_path_prefix: str = '',
                    mean_days: int = 365,
                    ):
    max_sens_list, min_sens_list, mean_sens_list = list(), list(), list()
    max_lat_list, min_lat_list, mean_lat_list = list(), list(), list()
    # for time_start, time_end in [(1, 3653), (3654, 7305), (7306, 10958), (10959, 14610), (14611, 16071)]:
    for time_start, time_end in [(0, 3653), (3653, 7305), (7305, 10958), (10958, 14610), (14610, 16554)]:
        if os.path.exists(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
                                              f'_max_sens({time_start}-{time_end})_{mean_days}.npy'):
            max_sens = np.load(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_'
                                                   f'max_sens({time_start}-{time_end})_{mean_days}.npy')
            min_sens = np.load(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_'
                                                   f'min_sens({time_start}-{time_end})_{mean_days}.npy')
            mean_sens = np.load(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_'
                                                    f'mean_sens({time_start}-{time_end})_{mean_days}.npy')
            max_sens_list += list(max_sens)
            min_sens_list += list(min_sens)
            mean_sens_list += list(mean_sens)

            if not coeff_type in ('raw', 'diff'):
                max_lat = np.load(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_'
                                                      f'max_lat({time_start}-{time_end})_{mean_days}.npy')
                min_lat = np.load(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_'
                                                      f'min_lat({time_start}-{time_end})_{mean_days}.npy')
                mean_lat = np.load(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_'
                                                       f'mean_lat({time_start}-{time_end})_{mean_days}.npy')

                max_lat_list += list(max_lat)
                min_lat_list += list(min_lat)
                mean_lat_list += list(mean_lat)
        else:
            print(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
                                              f'_max_sens({time_start}-{time_end})_{mean_days}.npy')
            print(f'Array {time_start}-{time_end} is not found!')

    time_start = 0
    time_end = 16554
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
            f'_max_sens({time_start}-{time_end})_{mean_days}.npy', np.array(max_sens_list))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
            f'_min_sens({time_start}-{time_end})_{mean_days}.npy', np.array(min_sens_list))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
            f'_mean_sens({time_start}-{time_end})_{mean_days}.npy', np.array(mean_sens_list))

    if coeff_type != 'raw':
        np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
                f'_max_lat({time_start}-{time_end})_{mean_days}.npy', np.array(max_lat_list))
        np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
                f'_min_lat({time_start}-{time_end})_{mean_days}.npy', np.array(min_lat_list))
        np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
                f'_mean_lat({time_start}-{time_end})_{mean_days}.npy', np.array(mean_lat_list))
    return
