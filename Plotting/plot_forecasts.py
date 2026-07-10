import datetime
import math
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Plotting.video import get_continuous_cmap

def plot_probabilistic_forecast_compare(files_path_prefix: str,
                                actual_array: np.ndarray,
                                prediction: np.ndarray,
                                time_start: int,
                                time_end: int,
                                flux_type: str,
                                offset: int = 0,
                                ):
    if not os.path.exists(files_path_prefix + f'videos/Forecast/Probabilistic'):
        os.mkdir(files_path_prefix + f'videos/Forecast/Probabilistic')
    amount = time_end-time_start

    fig = plt.figure(figsize=(6*(time_end-time_start), 15), constrained_layout=True)
    # fig = plt.figure(figsize=(6 * (time_end - time_start), 15))
    day_start = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=offset + time_start)
    day_end = day_start + datetime.timedelta(days=amount)
    # fig.suptitle(f'Predictions {flux_type}\n days {day_start.strftime("%d.%m.%Y")} - {day_end.strftime("%d.%m.%Y")}\n',
    #              fontsize=28, fontweight='bold')

    if flux_type == 'sensible':
        flux_type_rus = 'явного'
    else:
        flux_type_rus = 'скрытого'
    fig.suptitle(f'Прогноз {flux_type_rus} потока\n в интервале {day_start.strftime("%d.%m.%Y")} - {day_end.strftime("%d.%m.%Y")}\n',
                 fontsize=28, fontweight='bold')

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=3, ncols=1)
    subfigs[0].suptitle(f'Реальные значения', fontsize=18, fontweight='bold')
    subfigs[1].suptitle(f'Прогноз', fontsize=18, fontweight='bold')
    subfigs[2].suptitle(f'Абсолютная разность', fontsize=18, fontweight='bold')

    coeff_max = np.max([np.nanmax(actual_array), np.nanmax(prediction)])
    coeff_min = np.min([np.nanmin(actual_array), np.nanmin(prediction)])
    cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                               [0, (1.0 - coeff_min) / (coeff_max - coeff_min), 1])
    cmap2 = get_continuous_cmap(['#ffffff', '#ff0000'],
                               [0, 1])
    cmap.set_bad('lightgreen', 1.0)
    cmap2.set_bad('lightgreen', 1.0)

    axes_list = []
    for row in range(3):
        axs = subfigs[row].subplots(nrows=1, ncols=amount)
        axes_list.append(axs)

    img = [[None for _ in range(amount)] for _ in range(3)]
    cax  = [[None for _ in range(amount)] for _ in range(3)]
    for j in range(3):
        for i in range(5):
            divider = make_axes_locatable(axes_list[j][i])
            cax[j][i] = divider.append_axes('right', size='5%', pad=0.3)

    for i in range(amount):
        img[0][i] = axes_list[0][i].imshow(actual_array[i],
                       interpolation='none',
                       cmap=cmap,
                       vmin=coeff_min,
                       vmax=coeff_max)
        img[1][i] = axes_list[1][i].imshow(prediction[i],
                       interpolation='none',
                       cmap=cmap,
                       vmin=coeff_min,
                       vmax=coeff_max)
        img[2][i] = axes_list[2][i].imshow(np.abs(actual_array[i] - prediction[i]),
                       interpolation='none',
                       cmap=cmap2,
                       vmin=0,
                       vmax=abs(coeff_max) + abs(coeff_min))

    for j in range(3):
        for i in range(5):
            fig.colorbar(img[j][i], cax=cax[j][i], orientation='vertical')
    # fig.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Forecast/Probabilistic/{flux_type}_{time_start}-{time_end}.png')
    return

def plot_probabilistic_forecast(files_path_prefix: str,
                                prediction: np.ndarray,
                                pred_q05: np.ndarray,
                                pred_q95: np.ndarray,
                                time_start: int,
                                time_end: int,
                                flux_type: str,
                                offset: int = 0,
                                ):
    if not os.path.exists(files_path_prefix + f'videos/Forecast/Probabilistic'):
        os.mkdir(files_path_prefix + f'videos/Forecast/Probabilistic')
    amount = time_end-time_start

    fig = plt.figure(figsize=(6*(time_end-time_start), 15), constrained_layout=True)
    # fig = plt.figure(figsize=(6 * (time_end - time_start), 15))
    day_start = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=offset + time_start)
    day_end = day_start + datetime.timedelta(days=amount)
    # fig.suptitle(f'Predictions {flux_type}\n days {day_start.strftime("%d.%m.%Y")} - {day_end.strftime("%d.%m.%Y")}\n',
    #              fontsize=28, fontweight='bold')
    if flux_type == 'sensible':
        flux_type_rus = 'явного'
    else:
        flux_type_rus = 'скрытого'
    fig.suptitle(f'Прогноз {flux_type_rus} потока\n в интервале {day_start.strftime("%d.%m.%Y")} - {day_end.strftime("%d.%m.%Y")}\n',
                 fontsize=28, fontweight='bold')

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=3, ncols=1)
    subfigs[0].suptitle(f'Граница дов. интервала уровня 0.05', fontsize=18, fontweight='bold')
    subfigs[1].suptitle(f'Прогноз', fontsize=18, fontweight='bold')
    subfigs[2].suptitle(f'Граница дов. интервала уровня 0.95', fontsize=18, fontweight='bold')

    coeff_max = np.max([np.nanmax(pred_q95), np.nanmax(prediction)])
    coeff_min = np.min([np.nanmin(pred_q05), np.nanmin(prediction)])
    cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                               [0, (1.0 - coeff_min) / (coeff_max - coeff_min), 1])
    cmap.set_bad('lightgreen', 1.0)

    axes_list = []
    for row in range(3):
        axs = subfigs[row].subplots(nrows=1, ncols=amount)
        axes_list.append(axs)

    img = [[None for _ in range(amount)] for _ in range(3)]
    cax  = [[None for _ in range(amount)] for _ in range(3)]
    for j in range(3):
        for i in range(5):
            divider = make_axes_locatable(axes_list[j][i])
            cax[j][i] = divider.append_axes('right', size='5%', pad=0.3)

    for i in range(amount):
        img[0][i] = axes_list[0][i].imshow(pred_q05[i],
                       interpolation='none',
                       cmap=cmap,
                       vmin=coeff_min,
                       vmax=coeff_max)
        img[1][i] = axes_list[1][i].imshow(prediction[i],
                       interpolation='none',
                       cmap=cmap,
                       vmin=coeff_min,
                       vmax=coeff_max)
        img[2][i] = axes_list[2][i].imshow(pred_q95[i],
                       interpolation='none',
                       cmap=cmap,
                       vmin=coeff_min,
                       vmax=coeff_max)

    for j in range(3):
        for i in range(5):
            fig.colorbar(img[j][i], cax=cax[j][i], orientation='vertical')

    # fig.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Forecast/Probabilistic/{flux_type}_{time_start}-{time_end}_interval.png')
    return