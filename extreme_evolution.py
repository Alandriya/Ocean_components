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
from scipy.stats import linregress
import numpy, scipy.optimize
from data_processing import load_ABCFE
from symfit import parameters, variables, sin, cos, Fit
from sklearn.metrics import r2_score


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


def plot_extreme(files_path_prefix: str,
                 coeff_type: str,
                 time_start: int,
                 time_end: int,
                 mean_days: int = 1,
                 local_path_prefix:str = '',
                 names: tuple = ('Sensible', 'Latent'),
                 fit_sinus: bool = False,
                 fit_regression: bool = False,
                 fit_fourier_flag: bool = False):
    """
    Plots two pictures: first with evolution of max, min and mean of coefficient, second - the same, but adding
    approximation of max and min with sin function
    :param files_path_prefix: path to the working directory
    :param coeff_type: 'a' or 'b'
    :param time_start: start day index, counting from 01.01.1979
    :param time_end: end day index
    :param mean_days: width of the window in days, in which mean was taken
    :param local_path_prefix: local path from the files_path_prefix to data with extreme values
    :param names: a tuple with names for variables: e.g. ('Sensible', 'Latent'), ('Flux', 'SST')
    :param fit_sinus: if to fit sinus function
    :param fit_regression: if to fit linear regression like a*x + b
    :param fit_fourier_flag: it to approximate residuals without trend by fft
    :return:
    """
    font = {'size': 14}
    font_names = {'weight': 'bold', 'size': 20}
    matplotlib.rc('font', **font)
    sns.set_style("whitegrid")

    days = [datetime.datetime(1979, 1, 1) + datetime.timedelta(days=t) for t in
            range(time_start, time_end - mean_days, mean_days)]
    if os.path.exists(
            files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy'):
        max_sens = np.load(
            files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy')
        min_sens = np.load(
            files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy')
        mean_sens = np.load(
            files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_mean_sens({time_start}-{time_end})_{mean_days}.npy')
        # med_sens = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_med_sens({time_start}-{time_end})_{mean_days}.npy')

        max_lat = np.load(
            files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_max_lat({time_start}-{time_end})_{mean_days}.npy')
        min_lat = np.load(
            files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_min_lat({time_start}-{time_end})_{mean_days}.npy')
        mean_lat = np.load(
            files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}_mean_lat({time_start}-{time_end})_{mean_days}.npy')
        # med_lat = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_med_lat({time_start}-{time_end})_{mean_days}.npy')

        days = days[:len(max_sens)]
        fig, axs = plt.subplots(2, 1, figsize=(20, 10))
        # Major ticks every half year, minor ticks every month,
        if mean_days == 365:
            axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
            axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
        else:
            pass
            # axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
            # axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))

        axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
        axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))

        axs[0].set_title(names[0], fontdict=font_names)
        axs[0].plot(days, max_sens, label='max', c='r', alpha=0.75)
        axs[0].plot(days, min_sens, label='min', c='b', alpha=0.75)
        axs[0].plot(days, mean_sens, label='mean', c='g')
        # axs[0].plot(days, med_sens, label='med', c='y')
        axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        axs[1].set_title(names[1], fontdict=font_names)
        axs[1].plot(days, max_lat, label='max', c='r', alpha=0.75)
        axs[1].plot(days, min_lat, label='min', c='b', alpha=0.75)
        axs[1].plot(days, mean_lat, label='mean', c='g')
        # axs[1].plot(days, med_lat, label='med', c='y')
        axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        if not os.path.exists(files_path_prefix + f'Extreme/plots'):
            os.mkdir(files_path_prefix + f'Extreme/plots')
        if not os.path.exists(files_path_prefix + f'Extreme/plots/{local_path_prefix}'):
            os.mkdir(files_path_prefix + f'Extreme/plots/{local_path_prefix}')
        if not os.path.exists(files_path_prefix + f'Extreme/plots/{local_path_prefix}/{mean_days}'):
            os.mkdir(files_path_prefix + f'Extreme/plots/{local_path_prefix}/{mean_days}')

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'Extreme/plots/{local_path_prefix}/{mean_days}/{coeff_type}_({time_start}-{time_end})_{mean_days}.png')
        plt.close(fig)

        if fit_sinus:
            fig, axs = plt.subplots(2, 1, figsize=(20, 10))
            if mean_days == 365:
                axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
                axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
            else:
                axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
                axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
            axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
            axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))
            x = np.array(range(time_start, time_end - mean_days, mean_days))
            x = x[:len(max_sens)]


            # deleting trends
            res = linregress(x, max_sens)  # res.intercept + res.slope * x
            max_sens -= res.slope * x
            res = linregress(x, min_sens)
            min_sens -= res.slope * x
            res = linregress(x, mean_sens)
            mean_sens -= res.slope * x

            res = linregress(x, max_lat)
            max_lat -= res.slope * x
            res = linregress(x, min_lat)
            min_lat -= res.slope * x
            res = linregress(x, mean_lat)
            mean_lat -= res.slope * x

            fig.suptitle(f'{coeff_type} coefficient extreme residuals', fontsize=20, fontweight='bold')
            approx_string = f'A*sin({chr(969)}x + {chr(966)}) + c'
            axs[0].set_title(names[0], fontdict=font_names)
            res = fit_sin(x, max_sens)
            rss = np.sqrt(np.sum((max_sens - res["fitfunc"](x)) ** 2)) / len(x)
            # string_fit = f"{res['amp']:.1f}*sin({res['omega']:.5f}*x + {res['phase']:.1f}) + {res['offset']:.1f}"
            axs[0].plot(days, res["fitfunc"](x), '--', c='darkviolet', label=f'{approx_string}\n RMSE={rss:.1e}')
            axs[0].plot(days, max_sens, c='r', alpha=0.75, label='max')
            axs[0].plot(days, mean_sens, c='g', alpha=1, label='mean')

            res = fit_sin(x, min_sens)
            rss = np.sqrt(np.sum((min_sens - res["fitfunc"](x)) ** 2)) / len(x)
            # string_fit = f"{res['amp']:.1f}*sin({res['omega']:.5f}*x + {res['phase']:.1f}) + {res['offset']:.1f}"
            axs[0].plot(days, res["fitfunc"](x), '--', c='orange', label=f'{approx_string}\n RMSE = {rss:.1e}')
            axs[0].plot(days, min_sens, c='b', alpha=0.75, label='min')
            axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            axs[1].set_title(names[1], fontdict=font_names)
            res = fit_sin(x, max_lat)
            rss = np.sqrt(np.sum((max_lat - res["fitfunc"](x)) ** 2)) / len(x)
            # string_fit = f"{res['amp']:.1f}*sin({res['omega']:.5f}*x + {res['phase']:.1f}) + {res['offset']:.1f}"
            axs[1].plot(days, res["fitfunc"](x), '--', c='darkviolet', label=f'{approx_string}\n RMSE = {rss:.1e}')
            axs[1].plot(days, max_lat, c='r', alpha=0.75, label='max')
            axs[1].plot(days, mean_lat, c='g', alpha=1, label='mean')

            res = fit_sin(x, min_lat)
            rss = np.sqrt(np.sum((min_lat - res["fitfunc"](x)) ** 2)) / len(x)
            # string_fit = f"{res['amp']:.1f}*sin({res['omega']:.5f}*x + {res['phase']:.1f}) + {res['offset']:.1f}"
            axs[1].plot(days, res["fitfunc"](x), '--', c='orange', label=f'{approx_string}\n RMSE={rss:.1e}')
            axs[1].plot(days, min_lat, c='b', alpha=0.75, label='min')
            axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            fig.tight_layout()
            fig.savefig(files_path_prefix + f'Extreme/plots/{local_path_prefix}/{mean_days}/{coeff_type}_({time_start}-{time_end})_{mean_days}_fit_sinus.png')
        if fit_regression:
            fig, axs = plt.subplots(2, 1, figsize=(20, 10))
            if mean_days == 365:
                axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
                axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
            else:
                axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
                axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
            axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
            axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))
            x = np.array(range(time_start, time_end - mean_days, mean_days))
            x = x[:len(max_sens)]
            fig.suptitle(f'{coeff_type} coefficient extreme', fontsize=20, fontweight='bold')

            axs[0].set_title(names[0], fontdict=font_names)
            axs[0].plot(days, max_sens, c='r', alpha=0.75, label='max')
            res = linregress(x, max_sens)
            axs[0].plot(days, res.intercept + res.slope*x, '--', c='darkviolet', label=f'{res.slope:.2e} * x + {res.intercept: .5f}')
            axs[0].plot(days, mean_sens, c='g', alpha=1, label='mean')
            res = linregress(x, min_sens)
            axs[0].plot(days, res.intercept + res.slope*x, '--', c='orange', label=f'{res.slope:.2e} * x + {res.intercept: .5f}')
            axs[0].plot(days, min_sens, c='b', alpha=0.75, label='min')
            axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            axs[1].set_title(names[1], fontdict=font_names)
            axs[1].plot(days, max_lat, c='r', alpha=0.75, label='max')
            res = linregress(x, max_lat)
            axs[1].plot(days, res.intercept + res.slope*x, '--', c='darkviolet', label=f'{res.slope:.2e} * x + {res.intercept: .5f}')
            axs[1].plot(days, mean_lat, c='g', alpha=1, label='mean')
            res = linregress(x, min_lat)
            axs[1].plot(days, res.intercept + res.slope*x, '--', c='orange', label=f'{res.slope:.2e} * x + {res.intercept: .5f}')
            axs[1].plot(days, min_lat, c='b', alpha=0.75, label='min')
            axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            fig.tight_layout()
            fig.savefig(files_path_prefix + f'Extreme/plots/{local_path_prefix}/{mean_days}/{coeff_type}_'
                                            f'({time_start}-{time_end})_{mean_days}_fit_regression.png')

        if fit_fourier_flag:
            fig, axs = plt.subplots(2, 1, figsize=(20, 10))
            if mean_days == 365:
                axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
                axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
            else:
                axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
                axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
            axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
            axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))
            x = np.array(range(time_start, time_end - mean_days, mean_days))
            x = x[:len(max_sens)]

            # deleting trends
            res = linregress(x, max_sens)  # res.intercept + res.slope * x
            max_sens -= res.slope * x
            res = linregress(x, min_sens)
            min_sens -= res.slope * x
            res = linregress(x, mean_sens)
            mean_sens -= res.slope * x

            res = linregress(x, max_lat)
            max_lat -= res.slope * x
            res = linregress(x, min_lat)
            min_lat -= res.slope * x
            res = linregress(x, mean_lat)
            mean_lat -= res.slope * x

            max_sens_fourier, max_lat_fourier = fit_fourier(x, x, max_sens, max_lat)
            min_sens_fourier, min_lat_fourier = fit_fourier(x, x, min_sens, min_lat)
            mean_sens_fourier, mean_lat_fourier = fit_fourier(x, x, mean_sens, mean_lat)

            fig.suptitle(f'{coeff_type} coefficient extreme residuals', fontsize=20, fontweight='bold')
            axs[0].set_title(names[0], fontdict=font_names)
            axs[0].plot(days, max_sens, c='r', alpha=0.75, label='max')
            axs[0].plot(days, max_sens_fourier, '--', c='darkviolet', label=f'Fourier max, RMSE = {rmse(max_sens, max_sens_fourier):.1e}')
            axs[0].plot(days, mean_sens, c='g', alpha=1, label='mean')
            # axs[0].plot(days, mean_sens_fourier, '--', c='orange', label='Fourier mean')
            axs[0].plot(days, min_sens, c='b', alpha=0.75, label='min')
            axs[0].plot(days, min_sens_fourier, '--', c='orange', label=f'Fourier min, RMSE = {rmse(min_sens, min_sens_fourier):.1e}')
            axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            axs[1].set_title(names[1], fontdict=font_names)
            axs[1].plot(days, max_lat, c='r', alpha=0.75, label='max')
            axs[1].plot(days, max_lat_fourier, '--', c='darkviolet', label=f'Fourier max, RMSE = {rmse(max_lat, max_lat_fourier):.1e}')
            axs[1].plot(days, mean_lat, c='g', alpha=1, label='mean')
            axs[1].plot(days, min_lat_fourier, '--', c='orange', label=f'Fourier min, RMSE = {rmse(min_lat, min_lat_fourier):.1e}')
            axs[1].plot(days, min_lat, c='b', alpha=0.75, label='min')
            axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            fig.tight_layout()
            fig.savefig(files_path_prefix + f'Extreme/plots/{local_path_prefix}/{mean_days}/{coeff_type}_'
                                            f'({time_start}-{time_end})_{mean_days}_fit_residuals_fourier.png')

    return


def plot_extreme_3d(files_path_prefix: str,
                 coeff_type: str,
                 time_start: int,
                 time_end: int,
                 mean_days: int = 1,
                 fit_sinus: bool = False,
                 fit_regression: bool = False,
                 fit_fourier_flag: bool = False):
    """
    Plots two pictures: first with evolution of max, min and mean of coefficient, second - the same, but adding
    approximation of max and min with sin function
    :param files_path_prefix: path to the working directory
    :param coeff_type: 'a' or 'b'
    :param time_start: start day index, counting from 01.01.1979
    :param time_end: end day index
    :param mean_days: width of the window in days, in which mean was taken
    :param local_path_prefix: local path from the files_path_prefix to data with extreme values
    :param names: a tuple with names for variables: e.g. ('Sensible', 'Latent'), ('Flux', 'SST')
    :param fit_sinus: if to fit sinus function
    :param fit_regression: if to fit linear regression like a*x + b
    :param fit_fourier_flag: it to approximate residuals without trend by fft
    :return:
    """
    font = {'size': 14}
    font_names = {'weight': 'bold', 'size': 20}
    matplotlib.rc('font', **font)
    sns.set_style("whitegrid")

    days = [datetime.datetime(1979, 1, 1) + datetime.timedelta(days=t) for t in
            range(time_start, time_end - mean_days, mean_days)]
    if os.path.exists(
            files_path_prefix + f'Extreme/data/flux-sst/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy'):
        max_flux = np.load(
            files_path_prefix + f'Extreme/data/flux-sst/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy')
        min_flux = np.load(
            files_path_prefix + f'Extreme/data/flux-sst/{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy')
        mean_flux = np.load(
            files_path_prefix + f'Extreme/data/flux-sst/{coeff_type}_mean_sens({time_start}-{time_end})_{mean_days}.npy')

        max_sst = np.load(
            files_path_prefix + f'Extreme/data/flux-sst/{coeff_type}_max_lat({time_start}-{time_end})_{mean_days}.npy')
        min_sst = np.load(
            files_path_prefix + f'Extreme/data/flux-sst/{coeff_type}_min_lat({time_start}-{time_end})_{mean_days}.npy')
        mean_sst = np.load(
            files_path_prefix + f'Extreme/data/flux-sst/{coeff_type}_mean_lat({time_start}-{time_end})_{mean_days}.npy')

        max_press = np.load(
            files_path_prefix + f'Extreme/data/sst-press/{coeff_type}_max_lat({time_start}-{time_end})_{mean_days}.npy')
        min_press = np.load(
            files_path_prefix + f'Extreme/data/sst-press/{coeff_type}_min_lat({time_start}-{time_end})_{mean_days}.npy')
        mean_press = np.load(
            files_path_prefix + f'Extreme/data/sst-press/{coeff_type}_mean_lat({time_start}-{time_end})_{mean_days}.npy')

        days = days[:len(max_flux)]
        fig, axs = plt.subplots(3, 1, figsize=(20, 10))
        # Major ticks every half year, minor ticks every month,
        if mean_days == 365:
            axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
            axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
            axs[2].xaxis.set_minor_locator(mdates.MonthLocator())
        else:
            pass
            # axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
            # axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))

        axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
        axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))
        axs[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[2].xaxis.get_major_locator()))
        fig.suptitle(f'{coeff_type} coefficient extreme', fontsize=20, fontweight='bold')

        axs[0].set_title('Flux', fontdict=font_names)
        axs[0].plot(days, max_flux, label='max', c='r', alpha=0.75)
        axs[0].plot(days, min_flux, label='min', c='b', alpha=0.75)
        axs[0].plot(days, mean_flux, label='mean', c='g')
        axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        axs[1].set_title('SST', fontdict=font_names)
        axs[1].plot(days, max_sst, label='max', c='r', alpha=0.75)
        axs[1].plot(days, min_sst, label='min', c='b', alpha=0.75)
        axs[1].plot(days, mean_sst, label='mean', c='g')
        axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        axs[2].set_title('Pressure', fontdict=font_names)
        axs[2].plot(days, max_press, label='max', c='r', alpha=0.75)
        axs[2].plot(days, min_press, label='min', c='b', alpha=0.75)
        axs[2].plot(days, mean_press, label='mean', c='g')
        axs[2].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        if not os.path.exists(files_path_prefix + f'Extreme/plots'):
            os.mkdir(files_path_prefix + f'Extreme/plots')
        if not os.path.exists(files_path_prefix + f'Extreme/plots/3D'):
            os.mkdir(files_path_prefix + f'Extreme/plots/3D')
        if not os.path.exists(files_path_prefix + f'Extreme/plots/3D/{mean_days}'):
            os.mkdir(files_path_prefix + f'Extreme/plots/3D/{mean_days}')

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'Extreme/plots/3D/{mean_days}/{coeff_type}_({time_start}-{time_end})_{mean_days}.png')
        plt.close(fig)

        if fit_sinus:
            fig, axs = plt.subplots(3, 1, figsize=(20, 10))
            if mean_days == 365:
                axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
                axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
                axs[2].xaxis.set_minor_locator(mdates.MonthLocator())
            else:
                pass
                # axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
                # axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
            axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
            axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))
            axs[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))
            x = np.array(range(time_start, time_end - mean_days, mean_days))
            x = x[:len(max_flux)]

            # deleting trends
            res = linregress(x, max_flux)  # res.intercept + res.slope * x
            max_flux -= res.slope * x
            res = linregress(x, min_flux)
            min_flux -= res.slope * x
            res = linregress(x, mean_flux)
            mean_flux -= res.slope * x

            res = linregress(x, max_sst)
            max_sst -= res.slope * x
            res = linregress(x, min_sst)
            min_sst -= res.slope * x
            res = linregress(x, mean_sst)
            mean_sst -= res.slope * x

            res = linregress(x, max_press)
            max_press -= res.slope * x
            res = linregress(x, min_press)
            min_press -= res.slope * x
            res = linregress(x, mean_press)
            mean_press -= res.slope * x

            fig.suptitle(f'{coeff_type} coefficient extreme residuals', fontsize=20, fontweight='bold')
            approx_string = f'A*sin({chr(969)}x + {chr(966)}) + c'
            # -------------------------------------------------------
            axs[0].set_title('Flux', fontdict=font_names)

            res = fit_sin(x, max_flux)
            rss = np.sqrt(np.sum((max_flux - res["fitfunc"](x)) ** 2)) / len(x)
            axs[0].plot(days, res["fitfunc"](x), '--', c='darkviolet', label=f'{approx_string}\n RMSE={rss:.1e}')
            axs[0].plot(days, max_flux, c='r', alpha=0.75, label='max')
            axs[0].plot(days, mean_flux, c='g', alpha=1, label='mean')

            res = fit_sin(x, min_flux)
            rss = np.sqrt(np.sum((min_flux - res["fitfunc"](x)) ** 2)) / len(x)
            axs[0].plot(days, res["fitfunc"](x), '--', c='orange', label=f'{approx_string}\n RMSE = {rss:.1e}')
            axs[0].plot(days, min_flux, c='b', alpha=0.75, label='min')
            axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            # -------------------------------------------------------
            axs[1].set_title('SST', fontdict=font_names)
            res = fit_sin(x, max_sst)
            rss = np.sqrt(np.sum((max_sst - res["fitfunc"](x)) ** 2)) / len(x)
            axs[1].plot(days, res["fitfunc"](x), '--', c='darkviolet', label=f'{approx_string}\n RMSE = {rss:.1e}')
            axs[1].plot(days, max_sst, c='r', alpha=0.75, label='max')
            axs[1].plot(days, mean_sst, c='g', alpha=1, label='mean')

            res = fit_sin(x, min_sst)
            rss = np.sqrt(np.sum((min_sst - res["fitfunc"](x)) ** 2)) / len(x)
            axs[1].plot(days, res["fitfunc"](x), '--', c='orange', label=f'{approx_string}\n RMSE={rss:.1e}')
            axs[1].plot(days, min_sst, c='b', alpha=0.75, label='min')
            axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            # -------------------------------------------------------
            axs[2].set_title('Pressure', fontdict=font_names)
            res = fit_sin(x, max_press)
            rss = np.sqrt(np.sum((max_press - res["fitfunc"](x)) ** 2)) / len(x)
            axs[2].plot(days, res["fitfunc"](x), '--', c='darkviolet', label=f'{approx_string}\n RMSE = {rss:.1e}')
            axs[2].plot(days, max_press, c='r', alpha=0.75, label='max')
            axs[2].plot(days, mean_press, c='g', alpha=1, label='mean')

            res = fit_sin(x, min_press)
            rss = np.sqrt(np.sum((min_press - res["fitfunc"](x)) ** 2)) / len(x)
            axs[2].plot(days, res["fitfunc"](x), '--', c='orange', label=f'{approx_string}\n RMSE={rss:.1e}')
            axs[2].plot(days, min_press, c='b', alpha=0.75, label='min')
            axs[2].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            # -------------------------------------------------------

            fig.tight_layout()
            fig.savefig(files_path_prefix + f'Extreme/plots/3D/{mean_days}/{coeff_type}_({time_start}-{time_end})_'
                                            f'{mean_days}_fit_sinus_residuals.png')
        if fit_regression:
            fig, axs = plt.subplots(3, 1, figsize=(20, 10))
            if mean_days == 365:
                axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
                axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
                axs[2].xaxis.set_minor_locator(mdates.MonthLocator())
            else:
                pass
                # axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
                # axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
            axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
            axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))
            axs[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[2].xaxis.get_major_locator()))
            x = np.array(range(time_start, time_end - mean_days, mean_days))
            x = x[:len(max_flux)]
            fig.suptitle(f'{coeff_type} coefficient extreme', fontsize=20, fontweight='bold')

            axs[0].set_title('Flux', fontdict=font_names)
            axs[0].plot(days, max_flux, c='r')
            res = linregress(x, max_flux)  # res.intercept + res.slope * x
            r2 = r2_score(max_flux, res.intercept + res.slope*x)
            axs[0].plot(days, res.intercept + res.slope*x, '--', c='darkviolet',
                        label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
            axs[0].plot(days, mean_flux, c='g', alpha=1, label='mean')
            res = linregress(x, min_flux)
            r2 = r2_score(min_flux, res.intercept + res.slope * x)
            axs[0].plot(days, res.intercept + res.slope*x, '--', c='orange',
                        label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
            axs[0].plot(days, min_flux, c='b', alpha=0.75, label='min')
            axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            axs[1].set_title('SST', fontdict=font_names)
            axs[1].plot(days, max_sst, c='r')
            res = linregress(x, max_sst)  # res.intercept + res.slope * x
            r2 = r2_score(max_sst, res.intercept + res.slope*x)
            axs[1].plot(days, res.intercept + res.slope*x, '--', c='darkviolet',
                        label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
            axs[1].plot(days, mean_sst, c='g', alpha=1, label='mean')
            res = linregress(x, min_sst)
            r2 = r2_score(min_sst, res.intercept + res.slope * x)
            axs[1].plot(days, res.intercept + res.slope*x, '--', c='orange',
                        label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
            axs[1].plot(days, min_sst, c='b', alpha=0.75, label='min')
            axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            axs[2].set_title('Pressure', fontdict=font_names)
            axs[2].plot(days, max_press, c='r')
            res = linregress(x, max_press)  # res.intercept + res.slope * x
            r2 = r2_score(max_press, res.intercept + res.slope*x)
            axs[2].plot(days, res.intercept + res.slope*x, '--', c='darkviolet',
                        label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
            axs[2].plot(days, mean_press, c='g', alpha=1, label='mean')
            res = linregress(x, min_press)
            r2 = r2_score(min_press, res.intercept + res.slope * x)
            axs[2].plot(days, res.intercept + res.slope*x, '--', c='orange',
                        label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
            axs[2].plot(days, min_press, c='b', alpha=0.75, label='min')
            axs[2].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            fig.tight_layout()
            fig.savefig(files_path_prefix + f'Extreme/plots/3D/{mean_days}/{coeff_type}_'
                                            f'({time_start}-{time_end})_{mean_days}_fit_regression.png')
        if fit_fourier_flag:
            fig, axs = plt.subplots(3, 1, figsize=(20, 10))
            if mean_days == 365:
                axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
                axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
                axs[2].xaxis.set_minor_locator(mdates.MonthLocator())
            else:
                pass
                # axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
                # axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
            axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
            axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))
            axs[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[2].xaxis.get_major_locator()))

            x = np.array(range(time_start, time_end - mean_days, mean_days))
            x = x[:len(max_flux)]

            # deleting trends
            res = linregress(x, max_flux)  # res.intercept + res.slope * x
            max_flux -= res.slope * x
            res = linregress(x, min_flux)
            min_flux -= res.slope * x
            res = linregress(x, mean_flux)
            mean_flux -= res.slope * x

            res = linregress(x, max_sst)
            max_sst -= res.slope * x
            res = linregress(x, min_sst)
            min_sst -= res.slope * x
            res = linregress(x, mean_sst)
            mean_sst -= res.slope * x

            res = linregress(x, max_press)
            max_press -= res.slope * x
            res = linregress(x, min_press)
            min_press -= res.slope * x
            res = linregress(x, mean_press)
            mean_press -= res.slope * x

            max_flux_fourier, min_flux_fourier = fit_fourier(x, x, max_flux, min_flux)
            max_sst_fourier, min_sst_fourier = fit_fourier(x, x, max_sst, min_sst)
            max_press_fourier, min_press_fourier = fit_fourier(x, x, max_press, min_press)

            fig.suptitle(f'{coeff_type} coefficient extreme residuals', fontsize=20, fontweight='bold')
            axs[0].set_title('Flux', fontdict=font_names)
            axs[0].plot(days, max_flux, c='r', alpha=0.75, label='max')
            axs[0].plot(days, max_flux_fourier, '--', c='darkviolet',
                        label=f'Fourier max, RMSE = {rmse(max_flux, max_flux_fourier):.1e}')
            axs[0].plot(days, mean_flux, c='g', alpha=1, label='mean')
            # axs[0].plot(days, mean_sens_fourier, '--', c='orange', label='Fourier mean')
            axs[0].plot(days, min_flux, c='b', alpha=0.75, label='min')
            axs[0].plot(days, min_flux_fourier, '--', c='orange',
                        label=f'Fourier min, RMSE = {rmse(min_flux, min_flux_fourier):.1e}')
            axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            axs[1].set_title('SST', fontdict=font_names)
            axs[1].plot(days, max_sst, c='r', alpha=0.75, label='max')
            axs[1].plot(days, max_sst_fourier, '--', c='darkviolet',
                        label=f'Fourier max, RMSE = {rmse(max_sst, max_sst_fourier):.1e}')
            axs[1].plot(days, mean_sst, c='g', alpha=1, label='mean')
            axs[1].plot(days, min_sst, c='b', alpha=0.75, label='min')
            axs[1].plot(days, min_sst_fourier, '--', c='orange',
                        label=f'Fourier min, RMSE = {rmse(min_sst, min_sst_fourier):.1e}')
            axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            axs[2].set_title('Pressure', fontdict=font_names)
            axs[2].plot(days, max_press, c='r', alpha=0.75, label='max')
            axs[2].plot(days, max_press_fourier, '--', c='darkviolet',
                        label=f'Fourier max, RMSE = {rmse(max_press, max_press_fourier):.1e}')
            axs[2].plot(days, mean_press, c='g', alpha=1, label='mean')
            axs[2].plot(days, min_press, c='b', alpha=0.75, label='min')
            axs[2].plot(days, min_press_fourier, '--', c='orange',
                        label=f'Fourier min, RMSE = {rmse(min_press, min_press_fourier):.1e}')
            axs[2].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            fig.tight_layout()
            fig.savefig(files_path_prefix + f'Extreme/plots/3D/{mean_days}/{coeff_type}_'
                                            f'({time_start}-{time_end})_{mean_days}_fit_residuals_fourier.png')

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

def collect_extreme(files_path_prefix: str,
                    coeff_type: str,
                    local_path_prefix: str = '',
                    mean_days: int = 365,
                    ):
    max_sens_list, min_sens_list, mean_sens_list = list(), list(), list()
    max_lat_list, min_lat_list, mean_lat_list = list(), list(), list()
    for time_start, time_end in [(1, 3653), (3654, 7305), (7306, 10958), (10959, 14610), (14611, 16071)]:
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
            print(f'Array {time_start}-{time_end} is not found!')

    time_start = 1
    time_end = 16071
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
            f'_max_sens({time_start}-{time_end})_{mean_days}.npy', np.array(max_sens_list))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
            f'_min_sens({time_start}-{time_end})_{mean_days}.npy', np.array(min_sens_list))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
            f'_mean_sens({time_start}-{time_end})_{mean_days}.npy', np.array(mean_sens_list))

    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
            f'_max_lat({time_start}-{time_end})_{mean_days}.npy', np.array(max_lat_list))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
            f'_min_lat({time_start}-{time_end})_{mean_days}.npy', np.array(min_lat_list))
    np.save(files_path_prefix + f'Extreme/data/{local_path_prefix}{coeff_type}'
            f'_mean_lat({time_start}-{time_end})_{mean_days}.npy', np.array(mean_lat_list))
    return


def plot_eigenvalues_extreme(files_path_prefix: str,
                             time_start: int,
                             time_end: int,
                             mean_days: int,
                   names: tuple = ('Sensible', 'Latent'),
                ):
    sns.set_style("whitegrid")
    font = {'size': 14}
    font_names = {'weight': 'bold', 'size': 20}
    matplotlib.rc('font', **font)

    max_trend = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}_trends_max.npy')
    min_trend = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}_trends_min.npy')
    mean_trend = np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}_trends_mean.npy')

    days = [datetime.datetime(1979, 1, 1) + datetime.timedelta(days=t) for t in
            range(time_start, time_end, mean_days)]

    if len(max_trend) % mean_days:
        max_trend = max_trend[:-(len(max_trend)%mean_days)]
        min_trend = min_trend[:-(len(min_trend) % mean_days)]
        mean_trend = mean_trend[:-(len(mean_trend) % mean_days)]

    max_trend = np.mean(max_trend.reshape(-1, mean_days), axis=1)
    min_trend = np.min(min_trend.reshape(-1, mean_days), axis=1)
    mean_trend = np.mean(mean_trend.reshape(-1, mean_days), axis=1)

    days = days[:len(max_trend)]
    fig, axs = plt.subplots(figsize=(20, 10))
    # fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    # Major ticks every half year, minor ticks every month,
    if mean_days == 365:
        axs.xaxis.set_minor_locator(mdates.MonthLocator())
    axs.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs.xaxis.get_major_locator()))

    # axs.set_title(f'{names[0]}-{names[1]} trends, mean of every {mean_days} days', fontdict=font_names)
    axs.plot(days, max_trend, label='max', c='r', alpha=0.75)
    axs.plot(days, min_trend, label='min', c='b', alpha=0.75)
    axs.plot(days, mean_trend, label='mean', c='g')
    axs.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    if not os.path.exists(files_path_prefix + f'Extreme/plots'):
        os.mkdir(files_path_prefix + f'Extreme/plots')
    if not os.path.exists(files_path_prefix + f'Extreme/plots/Eigenvalues'):
        os.mkdir(files_path_prefix + f'Extreme/plots/Eigenvalues')

    fig.tight_layout()
    fig.savefig(
        files_path_prefix + f'Extreme/plots/Eigenvalues/{names[0]}-{names[1]}_({time_start}-{time_end})_mean_{mean_days}.png')

    if mean_days == 365:
        axs.xaxis.set_minor_locator(mdates.MonthLocator())
    axs.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs.xaxis.get_major_locator()))

    x = np.array(range(time_start, time_end, mean_days))
    x = x[:len(max_trend)]
    # fig.suptitle(f'Regression for {names[0]}-{names[1]} trends, mean of every {mean_days} days', fontsize=20, fontweight='bold')

    # axs.set_title(f'{names[0]}-{names[1]} trends, mean of every {mean_days} days', fontdict=font_names)
    axs.plot(days, max_trend, label='max', c='r', alpha=0.75)
    res = linregress(x, max_trend)
    axs.plot(days, res.intercept + res.slope * x, '--', c='darkviolet',
                label=f'{res.slope:.2e} * x + {res.intercept: .5f}')
    print(f'maximum R^2 {mean_days} mean days = {r2_score(max_trend, res.intercept + res.slope * x): .2e}')
    axs.plot(days, min_trend, label='min', c='b', alpha=0.75)
    res = linregress(x, min_trend)
    axs.plot(days, res.intercept + res.slope * x, '--', c='orange',
                label=f'{res.slope:.2e} * x + {res.intercept: .5f}')
    axs.plot(days, mean_trend, label='mean', c='g')
    axs.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(files_path_prefix + f'Extreme/plots/Eigenvalues/{names[0]}-{names[1]}_({time_start}-{time_end})_mean_{mean_days}_fit_regression.png')

    return
