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
from extreme_evolution import fit_sin, fit_fourier, fit_linear, rmse


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
    elif os.path.exists(files_path_prefix + f'Extreme/data/Flux/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy'):
        max_flux = np.load(
            files_path_prefix + f'Extreme/data/Flux/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy')
        min_flux = np.load(
            files_path_prefix + f'Extreme/data/Flux/{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy')
        mean_flux = np.load(
            files_path_prefix + f'Extreme/data/Flux/{coeff_type}_mean_sens({time_start}-{time_end})_{mean_days}.npy')

        max_sst = np.load(
            files_path_prefix + f'Extreme/data/SST/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy')
        min_sst = np.load(
            files_path_prefix + f'Extreme/data/SST/{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy')
        mean_sst = np.load(
            files_path_prefix + f'Extreme/data/SST/{coeff_type}_mean_sens({time_start}-{time_end})_{mean_days}.npy')

        max_press = np.load(
            files_path_prefix + f'Extreme/data/Pressure/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy')
        min_press = np.load(
            files_path_prefix + f'Extreme/data/Pressure/{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy')
        mean_press = np.load(
            files_path_prefix + f'Extreme/data/Pressure/{coeff_type}_mean_sens({time_start}-{time_end})_{mean_days}.npy')

    else:
        print('No files!')
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
    # fig.suptitle(f'{coeff_type} data extreme', fontsize=20, fontweight='bold')

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
    # fig.suptitle(f'{coeff_type} data extreme', fontsize=20, fontweight='bold')

    x = np.array(range(time_start, time_end - mean_days, mean_days))
    x = x[:len(max_flux)]
    axs[0].set_title('Flux', fontdict=font_names)
    axs[0].plot(days, max_flux - min_flux, label='max - min', c='b', alpha=0.75)
    res = linregress(x, max_flux - min_flux)  # res.intercept + res.slope * x
    axs[0].plot(days, res.intercept + res.slope * x, '--', c='darkviolet',
                label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
    axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    axs[1].set_title('SST', fontdict=font_names)
    axs[1].plot(days, max_sst - min_sst, label='max - min', c='r', alpha=0.75)
    res = linregress(x, max_sst - min_sst)  # res.intercept + res.slope * x
    axs[1].plot(days, res.intercept + res.slope * x, '--', c='darkviolet',
                label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
    axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    axs[2].set_title('Pressure', fontdict=font_names)
    axs[2].plot(days, max_press - min_press, label='max - min', c='orange', alpha=0.75)
    res = linregress(x, max_press - min_press)  # res.intercept + res.slope * x
    axs[2].plot(days, res.intercept + res.slope * x, '--', c='darkviolet',
                label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
    axs[2].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    if not os.path.exists(files_path_prefix + f'Extreme/plots'):
        os.mkdir(files_path_prefix + f'Extreme/plots')
    if not os.path.exists(files_path_prefix + f'Extreme/plots/3D'):
        os.mkdir(files_path_prefix + f'Extreme/plots/3D')
    if not os.path.exists(files_path_prefix + f'Extreme/plots/3D/{mean_days}'):
        os.mkdir(files_path_prefix + f'Extreme/plots/3D/{mean_days}')

    fig.tight_layout()
    fig.savefig(files_path_prefix + f'Extreme/plots/3D/{mean_days}/difference_max-min_{coeff_type}_({time_start}-{time_end})_{mean_days}.png')
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
        days_delta = (datetime.datetime(2023, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
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
        # fig.suptitle(f'{coeff_type} data extreme', fontsize=20, fontweight='bold')

        axs[0].set_title('Flux', fontdict=font_names)
        axs[0].plot(days, max_flux, c='r')
        res = linregress(x, max_flux)  # res.intercept + res.slope * x
        r2 = r2_score(max_flux, res.intercept + res.slope*x)
        axs[0].plot(days, res.intercept + res.slope*x, '--', c='darkviolet',
                    label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
        print(f'Flux amount of trend for max = {res.slope * (days_delta): .2e}')
        axs[0].plot(days, mean_flux, c='g', alpha=1, label='mean')
        res = linregress(x, min_flux)
        r2 = r2_score(min_flux, res.intercept + res.slope * x)
        axs[0].plot(days, res.intercept + res.slope*x, '--', c='orange',
                    label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
        print(f'Flux amount of trend for min = {res.slope * (days_delta): .2e}')
        axs[0].plot(days, min_flux, c='b', alpha=0.75, label='min')
        axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        axs[1].set_title('SST', fontdict=font_names)
        axs[1].plot(days, max_sst, c='r')
        res = linregress(x, max_sst)  # res.intercept + res.slope * x
        r2 = r2_score(max_sst, res.intercept + res.slope*x)
        axs[1].plot(days, res.intercept + res.slope*x, '--', c='darkviolet',
                    label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
        print(f'SST amount of trend for max = {res.slope * (days_delta): .2e}')
        axs[1].plot(days, mean_sst, c='g', alpha=1, label='mean')
        res = linregress(x, min_sst)
        r2 = r2_score(min_sst, res.intercept + res.slope * x)
        axs[1].plot(days, res.intercept + res.slope*x, '--', c='orange',
                    label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
        print(f'SST amount of trend for min = {res.slope * (days_delta): .2e}')
        axs[1].plot(days, min_sst, c='b', alpha=0.75, label='min')
        axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        axs[2].set_title('Pressure', fontdict=font_names)
        axs[2].plot(days, max_press, c='r')
        res = linregress(x, max_press)  # res.intercept + res.slope * x
        r2 = r2_score(max_press, res.intercept + res.slope*x)
        axs[2].plot(days, res.intercept + res.slope*x, '--', c='darkviolet',
                    label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
        print(f'press amount of trend for max = {res.slope * (days_delta): .2e}')
        axs[2].plot(days, mean_press, c='g', alpha=1, label='mean')
        res = linregress(x, min_press)
        r2 = r2_score(min_press, res.intercept + res.slope * x)
        axs[2].plot(days, res.intercept + res.slope*x, '--', c='orange',
                    label=f'{res.slope:.2e} * x + {res.intercept: .1f}')
        print(f'press amount of trend for min = {res.slope * (days_delta): .2e}')
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

    # axs.set_title(f'{names[0]}-{names[1]} eigenvalues trends, mean of every {mean_days} days', fontdict=font_names)
    # axs.plot(days, max_trend, label='max', c='r', alpha=0.75)
    # axs.plot(days, min_trend, label='min', c='b', alpha=0.75)
    # axs.plot(days, mean_trend, label='mean', c='g')
    # axs.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
                label=f'max regression fit')
    # axs.plot(days, res.intercept + res.slope * x, '--', c='darkviolet',
    #             label=f'{res.slope:.2e} * x + {res.intercept: .5f}')
    # print(f'maximum R^2 {mean_days} mean days = {r2_score(max_trend, res.intercept + res.slope * x): .2e}')

    axs.plot(days, min_trend, label='min', c='b', alpha=0.75)
    res = linregress(x, min_trend)
    # axs.plot(days, res.intercept + res.slope * x, '--', c='orange',
    #             label=f'{res.slope:.2e} * x + {res.intercept: .5f}')
    axs.plot(days, res.intercept + res.slope * x, '--', c='orange',
                label='min regression fit')
    axs.plot(days, mean_trend, label='mean', c='g')
    axs.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.tight_layout()
    if not os.path.exists(files_path_prefix + f'videos/Eigenvalues/{names[0]}-{names[1]}'):
        os.mkdir(files_path_prefix + f'videos/Eigenvalues/{names[0]}-{names[1]}')
    fig.savefig(files_path_prefix + f'videos/Eigenvalues/{names[0]}-{names[1]}/{names[0]}-{names[1]}_({time_start}-{time_end})_mean_{mean_days}_fit_regression.png')
    return
