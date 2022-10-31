import pandas as pd
import numpy as np
import datetime
import os
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from video import get_continuous_cmap

width = 181
height = 161


def plot_typical_points_difference(files_path_prefix: str,
                                   mask: np.ndarray,
                                   time_start: int,
                                   time_end: int,
                                   flux_type: str,
                                   coeff_type: str, ):
    mask_map = np.array(mask, dtype=float).reshape((height, width))
    mask_map[mask_map == 0] = np.nan
    mask_map[mask_map == 1] = 0
    fig, axs = plt.subplots(figsize=(15, 15))

    # cmap = matplotlib.colors.ListedColormap(['green', 'white', 'red'])
    # norm = matplotlib.colors.BoundaryNorm([0, 1, 2, 3], cmap.N)

    points = [(15, 160), (15, 60), (40, 10), (40, 60), (45, 90), (40, 120), (40, 150), (60, 90), (60, 120), (60, 150),
              (90, 40), (90, 60), (90, 90), (90, 120), (90, 150), (110, 40), (110, 60), (110, 90), (110, 120),
              (110, 10), (130, 20), (150, 10), (130, 40), (130, 60), (130, 90), (130, 120), (150, 90), (150, 120)]


    for point in points:
        difference = np.load(
            files_path_prefix + f'Components/{flux_type}/difference/point_({point[0]}, {point[1]})-{coeff_type}.npy')
        mask_map[point[0] - 3:point[0] + 3, point[1] - 3: point[1] + 3] = np.mean(np.abs(difference))

    date_start = datetime.datetime(2019, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
    date_end = datetime.datetime(2019, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
    fig.suptitle(f"{coeff_type} coeff - {flux_type} difference\n {date_start.strftime('%d.%m.%Y')}-{date_end.strftime('%d.%m.%Y')}",
                 fontsize=26, fontweight='bold')

    cmap = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
    cmap.set_bad('darkgreen', 1.0)
    axs.imshow(mask_map, interpolation='none', cmap=cmap)
    fig.savefig(files_path_prefix + f'Components/{flux_type}/{coeff_type}-difference_points.png')
    plt.close(fig)
    return


def plot_difference_1d(files_path_prefix: str,
                       time_start: int,
                       time_end: int,
                       point: tuple,
                       window_width: int,
                       radius: int,
                       ticks_by_day: int = 1,
                       step_ticks: int = 1,
                       n_components: int = 3,
                       flux_type: str = 'sensible',
                       coeff_type: str = 'A'):
    diff = np.load(
        files_path_prefix + f'Components/{flux_type}/difference/point_({point[0]}, {point[1]})-{coeff_type}.npy')
    Bel = np.load(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-{coeff_type}.npy')
    Kor_sum = np.load(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-{coeff_type}.npy')
    rmse = math.sqrt(sum(diff ** 2))
    # print(rmse)

    date_start = datetime.datetime(2019, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
    date_end = datetime.datetime(2019, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
    fig, axs = plt.subplots(1, 1, figsize=(20, 10))
    fig.suptitle(f'{coeff_type} coeff {flux_type} at point ({point[0]}, {point[1]}) \n radius = {radius}, '
                 f'window = {window_width // ticks_by_day} days, step={step_ticks} ticks, n_components = {n_components}'
                 f'\n {date_start.strftime("%Y-%m-%d")} - {date_end.strftime("%Y-%m-%d")}'
                 f'\n RMSE = {int(rmse)}', fontsize=20, fontweight='bold')
    axs.xaxis.set_minor_locator(mdates.MonthLocator())
    axs.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs.xaxis.get_major_locator()))
    x = range(0, min(len(Bel), len(Kor_sum)))
    days = [datetime.datetime(2019, 1, 1) + datetime.timedelta(days=t) for t in range(len(x))]
    axs.plot(days, Bel[:len(x)], c='b', label='Bel')
    axs.plot(days, Kor_sum[:len(x)], c='r', label='Kor')
    # axs.plot(x, diff[:len(x)], c='y', label='difference')
    axs.legend()
    fig.tight_layout()
    if not os.path.exists(files_path_prefix + f'Components/{flux_type}/plots'):
        os.mkdir(files_path_prefix + f'Components/{flux_type}/plots')
    fig.savefig(
        files_path_prefix + f'Components/{flux_type}/plots/difference_point_({point[0]}, {point[1]})-{coeff_type}.png')
    plt.close(fig)
    return


def count_Bel_Kor_difference(files_path_prefix: str,
                             time_start: int,
                             time_end: int,
                             point_bigger: list,
                             point_size: int,
                             point: tuple,
                             n_components: int,
                             window_width: int,
                             ticks_by_day: int = 1,
                             step_ticks: int = 1,
                             timedelta: int = 0,
                             flux_type: str = 'sensible'):
    if not os.path.exists(files_path_prefix + f'Components/{flux_type}/Bel'):
        os.mkdir(files_path_prefix + f'Components/{flux_type}/Bel')

    if True or not os.path.exists(
            files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-A.npy'):
        a_Bel = np.zeros(time_end - time_start)
        for t in range(timedelta + time_start, timedelta + time_end - window_width // ticks_by_day):
            if flux_type == 'sensible':
                a_arr = np.load(files_path_prefix + f'Coeff_data/{t}_A_sens.npy')
            else:
                a_arr = np.load(files_path_prefix + f'Coeff_data/{t}_A_lat.npy')
            a_Bel[t - time_start - timedelta] = sum([a_arr[p[0], p[1]] for p in point_bigger]) / point_size
            # a_Bel[t - time_start - timedelta] = a_arr[point[0], point[1]]
            del a_arr
        # a_Bel = np.diff(a_Bel)
        np.save(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-A.npy', a_Bel)
    else:
        a_Bel = np.load(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-A.npy')

    if True or not os.path.exists(
            files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-B.npy'):
        b_Bel = np.zeros(time_end - time_start)
        for t in range(timedelta + time_start, timedelta + time_end - window_width // ticks_by_day):
            if flux_type == 'sensible':
                b_arr = np.load(files_path_prefix + f'Coeff_data/{t}_B.npy')[0]
            else:
                b_arr = np.load(files_path_prefix + f'Coeff_data/{t}_B.npy')[3]
            b_Bel[t - time_start - timedelta] = sum([b_arr[p[0], p[1]] for p in point_bigger]) / point_size
            # b_Bel[t - time_start - timedelta] = b_arr[point[0], point[1]]
            del b_arr
        # b_Bel = np.diff(b_Bel)
        np.save(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-B.npy', b_Bel)
    else:
        b_Bel = np.load(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-B.npy')

    if not os.path.exists(files_path_prefix + f'Components/{flux_type}/Sum'):
        os.mkdir(files_path_prefix + f'Components/{flux_type}/Sum')

    if True or not os.path.exists(
            files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-A.npy'):
        Kor_df = pd.read_excel(
            files_path_prefix + f'Components/{flux_type}/components-xlsx/point_({point[0]}, {point[1]}).xlsx')
        Kor_df.fillna(0, inplace=True)
        a_sum = np.zeros(len(Kor_df))
        for i in range(1, n_components + 1):
            a_sum += Kor_df[f'mean_{i}'] * Kor_df[f'weight_{i}']
    else:
        a_sum = np.load(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-A.npy')

    a_sum = np.array(a_sum[:len(a_sum) - len(a_sum) % (ticks_by_day // step_ticks)])
    a_sum = np.mean(a_sum.reshape(-1, (ticks_by_day // step_ticks)), axis=1)
    # a_sum /= window_width/ticks_by_day
    np.save(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-A.npy', a_sum)
    a_diff = a_Bel[:len(a_sum)] - a_sum
    if not os.path.exists(files_path_prefix + f'Components/{flux_type}/difference'):
        os.mkdir(files_path_prefix + f'Components/{flux_type}/difference')
    np.save(files_path_prefix + f'Components/{flux_type}/difference/point_({point[0]}, {point[1]})-A.npy', a_diff)

    if True or not os.path.exists(
            files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-B.npy'):
        Kor_df = pd.read_excel(
            files_path_prefix + f'Components/{flux_type}/components-xlsx/point_({point[0]}, {point[1]}).xlsx')
        Kor_df.fillna(0, inplace=True)
        b_sum = np.zeros(len(Kor_df))

        a_sum = np.zeros(len(Kor_df))
        for i in range(1, n_components + 1):
            a_sum += Kor_df[f'mean_{i}'] * Kor_df[f'weight_{i}']

        for i in range(1, n_components + 1):
            b_sum += Kor_df[f'weight_{i}'] * (np.square(Kor_df[f'mean_{i}']) + np.square(Kor_df[f'sigma_{i}']))

        b_sum = np.sqrt(b_sum - np.square(a_sum))
    else:
        b_sum = np.load(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-B.npy')

    b_sum = np.array(b_sum[:len(b_sum) - len(b_sum) % (ticks_by_day // step_ticks)])
    b_sum = np.mean(b_sum.reshape(-1, (ticks_by_day // step_ticks)), axis=1)
    # b_sum /= window_width/ticks_by_day
    np.save(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-B.npy', b_sum)
    b_diff = b_Bel[:len(b_sum)] - b_sum
    if not os.path.exists(files_path_prefix + f'Components/{flux_type}/difference'):
        os.mkdir(files_path_prefix + f'Components/{flux_type}/difference')
    np.save(files_path_prefix + f'Components/{flux_type}/difference/point_({point[0]}, {point[1]})-B.npy', b_diff)
    return
