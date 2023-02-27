import pandas as pd
import numpy as np
import datetime
import os
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from video import get_continuous_cmap
# from PSO_coefficients import norm_sum
from copy import deepcopy
import shutil
from data_processing import scale_to_bins
from EM_hybrid import *
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.linalg import sqrtm
import matplotlib.colors as colors

# width = 181
# height = 161

mu = 0.01
sigma = 0.015
x0 = 1


def plot_Kor_Bel_histograms(files_path_prefix: str,
                            time_start: int,
                            time_end: int,
                            data_raw: np.ndarray,
                            data_quantiles: np.ndarray,
                            flux_type: str,
                            point: tuple,
                            ):
    date_start = datetime.datetime(2019, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
    date_end = datetime.datetime(2019, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
    fig, axs = plt.subplots(figsize=(15, 15))
    fig.suptitle(f"{flux_type} data at ({point[0]}, {point[1]}): {date_start.strftime('%d.%m.%Y')}-{date_end.strftime('%d.%m.%Y')}",
                 fontsize=26, fontweight='bold')
    plt.hist(data_raw, bins=50, alpha=0.5, label="raw", density=True, color='b')
    plt.hist(data_quantiles, bins=10, alpha=0.5, label="quantiles", density=True, color='r')
    plt.legend(loc='upper right')
    fig.savefig(files_path_prefix + f'Components/{flux_type}/data_compare_hist_point({point[0]}, {point[1]}).png')
    plt.close(fig)
    return


def plot_typical_points_difference(files_path_prefix: str,
                                   mask: np.ndarray,
                                   time_start: int,
                                   time_end: int,
                                   flux_type: str,
                                   coeff_type: str,
                                   height: int = 100,
                                   width: int = 100,):
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
    # diff = np.load(files_path_prefix + f'Components/{flux_type}/difference/point_({point[0]}, {point[1]})-{coeff_type}.npy')
    # Bel = np.load(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-{coeff_type}.npy')
    # Kor_sum = np.load(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-{coeff_type}.npy')
    diff = np.load(files_path_prefix + f'Synthetic/difference/point_({point[0]}, {point[1]})-{coeff_type}.npy')
    Bel = np.load(files_path_prefix + f'Synthetic/Bel/point_({point[0]}, {point[1]})-{coeff_type}.npy')
    Kor_sum = np.load(files_path_prefix + f'Synthetic/coeff_Kor/Components/Sum/point_({point[0]}, {point[1]})-{coeff_type}.npy')
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
    # if not os.path.exists(files_path_prefix + f'Components/{flux_type}/plots'):
    #     os.mkdir(files_path_prefix + f'Components/{flux_type}/plots')
    # fig.savefig(files_path_prefix + f'Components/{flux_type}/plots/difference_point_({point[0]}, {point[1]})-{coeff_type}.png')

    fig.savefig(files_path_prefix + f'Synthetic/Plots/difference_point_({point[0]}, {point[1]})-{coeff_type}.png')
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
                             flux_type: str = 'sensible',
                             Bel_halfwindow: int = 0,
                             ):
    # if not os.path.exists(files_path_prefix + f'Components/{flux_type}/Bel'):
    #     os.mkdir(files_path_prefix + f'Components/{flux_type}/Bel')
    if not os.path.exists(files_path_prefix + f'Synthetic/Bel'):
        os.mkdir(files_path_prefix + f'Synthetic/Bel')

    # if True or not os.path.exists(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-A.npy'):
    if True or not os.path.exists(files_path_prefix + f'Synthetic/Bel/point_({point[0]}, {point[1]})-A.npy'):
        a_Bel = np.zeros(time_end - time_start + 2*Bel_halfwindow)
        for t in range(timedelta + time_start-Bel_halfwindow, timedelta + time_end - window_width // ticks_by_day +
                       Bel_halfwindow):
            if flux_type == 'sensible':
                a_arr = np.load(files_path_prefix + f'Synthetic/coeff_Bel/A_sens_{t}.npy')
            else:
                a_arr = np.load(files_path_prefix + f'Synthetic/coeff_Bel/{t}_A_lat.npy')
            # a_Bel[t - time_start - timedelta + Bel_halfwindow] = sum([a_arr[p[0], p[1]] for p in point_bigger]) / point_size
            a_Bel[t - time_start - timedelta] = a_arr[point[0], point[1]]
            del a_arr


        a_Bel_mean = np.zeros(time_end - time_start)
        for i in range(len(a_Bel_mean)):
            a_Bel_mean[i] = np.mean(a_Bel[i:i+2*Bel_halfwindow + 1])
        del a_Bel

        a_Bel = a_Bel_mean
        # a_Bel = np.diff(a_Bel)

        # np.save(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-A.npy', a_Bel)
        np.save(files_path_prefix + f'Synthetic/Bel/point_({point[0]}, {point[1]})-A.npy', a_Bel)
    else:
        raise ValueError
        a_Bel = np.load(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-A.npy')

    if True or not os.path.exists(
            files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-B.npy'):
        b_Bel = np.zeros(time_end - time_start)
        for t in range(timedelta + time_start, timedelta + time_end - window_width // ticks_by_day):
            if flux_type == 'sensible':
                # b_arr = np.load(files_path_prefix + f'Coeff_data/{t}_B.npy')[0]
                b_arr = np.load(files_path_prefix + f'Synthetic/coeff_Bel/B_{t}.npy')[0]
            else:
                b_arr = np.load(files_path_prefix + f'Coeff_data/{t}_B.npy')[3]
            b_Bel[t - time_start - timedelta] = sum([b_arr[p[0], p[1]] for p in point_bigger]) / point_size
            # b_Bel[t - time_start - timedelta] = b_arr[point[0], point[1]]
            del b_arr
        # b_Bel = np.diff(b_Bel)
        # np.save(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-B.npy', b_Bel)
        np.save(files_path_prefix + f'Synthetic/Bel/point_({point[0]}, {point[1]})-B.npy', b_Bel)
    else:
        b_Bel = np.load(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-B.npy')

    # if not os.path.exists(files_path_prefix + f'Components/{flux_type}/Sum'):
    #     os.mkdir(files_path_prefix + f'Components/{flux_type}/Sum')

    if not os.path.exists(files_path_prefix + f'Synthetic/Sum'):
        os.mkdir(files_path_prefix + f'Synthetic/Sum')

    if True or not os.path.exists(
            files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-A.npy'):
        # Kor_df = pd.read_excel(files_path_prefix + f'Components/{flux_type}/components-xlsx/point_({point[0]}, {point[1]}).xlsx')
        Kor_df = pd.read_excel(files_path_prefix + f'Synthetic/coeff_Kor/Components/components-xlsx/point_({point[0]}, {point[1]}).xlsx')
        Kor_df.fillna(0, inplace=True)
        a_sum = np.zeros(len(Kor_df))
        for i in range(1, n_components + 1):
            a_sum += Kor_df[f'mean_{i}'] * Kor_df[f'weight_{i}']
    else:
        a_sum = np.load(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-A.npy')

    a_sum = np.array(a_sum[:len(a_sum) - len(a_sum) % (ticks_by_day // step_ticks)])
    a_sum = np.mean(a_sum.reshape(-1, (ticks_by_day // step_ticks)), axis=1)
    # a_sum /= window_width/ticks_by_day
    # np.save(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-A.npy', a_sum)
    np.save(files_path_prefix + f'Synthetic/coeff_Kor/Components/Sum/point_({point[0]}, {point[1]})-A.npy', a_sum)

    a_diff = a_Bel[:len(a_sum)] - a_sum
    # if not os.path.exists(files_path_prefix + f'Components/{flux_type}/difference'):
    #     os.mkdir(files_path_prefix + f'Components/{flux_type}/difference')
    # np.save(files_path_prefix + f'Components/{flux_type}/difference/point_({point[0]}, {point[1]})-A.npy', a_diff)

    if not os.path.exists(files_path_prefix + f'Synthetic/difference'):
        os.mkdir(files_path_prefix + f'Synthetic/difference')
    np.save(files_path_prefix + f'Synthetic/difference/point_({point[0]}, {point[1]})-A.npy', a_diff)


    if True or not os.path.exists(
            files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-B.npy'):
        # Kor_df = pd.read_excel(files_path_prefix + f'Components/{flux_type}/components-xlsx/point_({point[0]}, {point[1]}).xlsx')
        Kor_df = pd.read_excel(files_path_prefix + f'Synthetic/coeff_Kor/Components/components-xlsx/point_({point[0]}, {point[1]}).xlsx')
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
    # np.save(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-B.npy', b_sum)
    np.save(files_path_prefix + f'Synthetic/Sum/point_({point[0]}, {point[1]})-B.npy', b_sum)
    b_diff = b_Bel[:len(b_sum)] - b_sum
    # if not os.path.exists(files_path_prefix + f'Components/{flux_type}/difference'):
    #     os.mkdir(files_path_prefix + f'Components/{flux_type}/difference')
    # np.save(files_path_prefix + f'Components/{flux_type}/difference/point_({point[0]}, {point[1]})-B.npy', b_diff)
    if not os.path.exists(files_path_prefix + f'Synthetic/difference'):
        os.mkdir(files_path_prefix + f'Synthetic/difference')
    np.save(files_path_prefix + f'Synthetic/difference/point_({point[0]}, {point[1]})-B.npy', b_diff)
    return


def create_synthetic_data(files_path_prefix: str,
                          width: int = 100,
                          height: int = 100,
                          time_start: int = 0,
                          time_end: int = 1):
    """
    Creates 2-dimensional array of a and b coefficients and the corresponding fluxes as solution of the Langevin equation
    :param files_path_prefix: path to the working directory
    :param width: width of the map
    :param height: height of the map
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :return:
    """
    wiener = np.zeros((height, width), dtype=float)
    wiener_2 = np.zeros((height, width), dtype=float)

    sensible_full = np.zeros((time_end - time_start, height, width), dtype=float)
    latent_full = np.zeros((time_end - time_start, height, width), dtype=float)
    a_full = np.zeros((time_end - time_start, 2, height, width), dtype=float)
    b_full = np.zeros((time_end - time_start, 4, height, width), dtype=float)
    for t in range(time_start, time_end):
        normal = np.random.normal(0, 1, size=(height, width))
        wiener += normal

        normal = np.random.normal(0, 1, size=(height, width))
        wiener_2 += normal

        sensible_full[t] = x0 * np.exp((mu - sigma*sigma/2)*t + sigma*wiener)
        latent_full[t] = x0 * np.exp((mu - sigma*sigma/2)*t + sigma*wiener_2)
        a_full[t, 0] = mu * sensible_full[t]
        a_full[t, 1] = mu * latent_full[t]
        b_full[t, 0] = sigma * sensible_full[t]
        b_full[t, 1] = sigma * latent_full[t]

        # plt.hist(sensible_extended[1].flatten(), bins=30)
        # plt.show()

        # np.save(f'{files_path_prefix}/Synthetic/X_{t}.npy', X)
        # np.save(f'{files_path_prefix}/Synthetic/B_{t}.npy', b)
        # np.save(f'{files_path_prefix}/Synthetic/A_{t}.npy', a)

    plt.hist(sensible_full[:, 0, 0].flatten(), bins=30)
    plt.show()

    np.save(f'{files_path_prefix}/Synthetic/sensible_full.npy', sensible_full)
    np.save(f'{files_path_prefix}/Synthetic/latent_full.npy', latent_full)
    np.save(f'{files_path_prefix}/Synthetic/B_full.npy', b_full)
    np.save(f'{files_path_prefix}/Synthetic/A_full.npy', a_full)
    return


def multiply_synthetic_Korolev(files_path_prefix: str):
    sensible = np.load(f'{files_path_prefix}/Synthetic/sensible_full.npy')
    multiply_amount = 100
    sensible_extended = np.zeros((sensible.shape[0]*multiply_amount, sensible.shape[1], sensible.shape[2]))

    for t in range(sensible.shape[0]):
        current = np.copy(sensible[t])
        # plt.hist(current.flatten())
        # plt.show()
        sensible_extended[t * multiply_amount] = current
        for t1 in range(1, multiply_amount):
            noised = current + np.random.normal(0, 1, size=current.shape)
            sensible_extended[t * multiply_amount + t1] = noised
            # plt.hist(noised.flatten())
            # plt.show()

    # plt.hist(sensible_extended[1].flatten(), bins=30)
    # plt.show()

    np.save(f'{files_path_prefix}/Synthetic/sensible_full_extended.npy', sensible_extended)
    return


def count_synthetic_Bel(files_path_prefix: str,
                        sensible: np.array,
                        latent: np.array,
                        time_start: int,
                        time_end: int,
                        width: int = 100,
                        height: int = 100):
    """
    Counts estimation of a and b coefficients by Belyaev method from time_start to time_end index and saves them to
    files_path_prefix + 'Synthetic/coeff_Bel'
    :param files_path_prefix: path to the working directory
    :param sensible: np.array with shape (time_end - time_start, width, height) with imitated flux data
    :param latent: np.array with shape (time_end - time_start, width, height) with imitated flux data
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param width: width of the map
    :param height: height of the map
    :return:
    """
    if os.path.exists(files_path_prefix + 'Synthetic/coeff_Bel'):
        shutil.rmtree(files_path_prefix + 'Synthetic/coeff_Bel')
    os.mkdir(files_path_prefix + 'Synthetic/coeff_Bel')

    # quantification
    sensible_array, sens_quantiles = scale_to_bins(sensible, 100)
    latent_array, lat_quantiles = scale_to_bins(latent, 100)

    a_sens = np.zeros((height, width), dtype=float)
    a_lat = np.zeros((height, width), dtype=float)
    b_matrix = np.zeros((4, height, width), dtype=float)

    for t in range(time_start+1, time_end):
        set_sens = np.unique(sensible_array[t - 1])
        set_lat = np.unique(latent_array[t - 1])

        for val_t0 in set_sens:
            if not np.isnan(val_t0):
                points_sensible = np.where(sensible_array[t - 1] == val_t0)
                amount_t0 = len(points_sensible[0])

                # sensible t0 - sensible t1
                set_t1 = np.unique(sensible_array[t][points_sensible])
                probabilities = list()
                for val_t1 in set_t1:
                    prob = len(np.where(sensible_array[t][points_sensible] == val_t1)[0]) * 1.0 / amount_t0
                    probabilities.append(prob)

                a = sum([(list(set_t1)[i] - val_t0) * probabilities[i] for i in range(len(probabilities))])
                b_squared = sum(
                    [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

                a_sens[points_sensible] = a
                b_matrix[0][points_sensible] = np.sqrt(b_squared)

                # sensible t0 - latent t1
                set_t1 = np.unique(latent_array[t][points_sensible])
                probabilities = list()
                for val_t1 in set_t1:
                    prob = len(np.where(latent_array[t][points_sensible] == val_t1)) * 1.0 / amount_t0
                    probabilities.append(prob)
                b_squared = sum(
                    [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

                b_matrix[1][points_sensible] = np.sqrt(b_squared)

        for val_t0 in set_lat:
            if not np.isnan(val_t0):
                points_latent = np.where(latent_array[t - 1] == val_t0)
                amount_t0 = len(points_latent[0])

                # latent - latent
                set_t1 = np.unique(latent_array[t][points_latent])
                probabilities = list()
                for val_t1 in set_t1:
                    prob = len(np.where(latent_array[t][points_latent] == val_t1)[0]) * 1.0 / amount_t0
                    probabilities.append(prob)

                a = sum([(set_t1[i] - val_t0) * probabilities[i] for i in range(len(probabilities))])
                b_squared = sum(
                    [(set_t1[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

                a_lat[points_latent] = a
                b_matrix[3][points_latent] = np.sqrt(b_squared)

                # latent t0 - sensible t1
                set_t1 = np.unique(sensible_array[t][points_latent])
                probabilities = list()
                for val_t1 in set_t1:
                    prob = len(np.where(sensible_array[t][points_latent] == val_t1)[0]) * 1.0 / amount_t0
                    probabilities.append(prob)

                b_squared = sum(
                    [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

                b_matrix[2][points_latent] = np.sqrt(b_squared)

        # # get matrix root from B and count F
        # for i in range(161):
        #     for j in range(181):
        #         if not np.isnan(b_matrix[:, i, j]).any():
        #             b_matrix[:, i, j] = sqrtm(b_matrix[:, i, j].reshape(2, 2)).reshape(4)

        np.save(files_path_prefix + f'Synthetic/coeff_Bel/A_sens_{t}', a_sens)
        np.save(files_path_prefix + f'Synthetic/coeff_Bel/A_lat_{t}', a_lat)
        np.save(files_path_prefix + f'Synthetic/coeff_Bel/B_{t}', b_matrix)
    return


def count_synthetic_Korolev(files_path_prefix: str,
                            flux_type: str,
                            flux_data: np.ndarray,
                        time_start: int,
                        time_end: int,
                        width: int = 100,
                        height: int = 100,
                        window_width: int = 100,
                        n_components: int = 2,
                        ):
    """
    Counts estimation of a and b coefficients by Korolev method from time_start to time_end index and saves them to
    files_path_prefix + 'Synthetic/coeff_Kor'
    :param files_path_prefix: path to the working directory
    :param sensible: np.array with shape (time_end - time_start, width, height) with imitated flux data
    :param latent: np.array with shape (time_end - time_start, width, height) with imitated flux data
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param width: width of the map
    :param height: height of the map
    :param window_width: window width
    :param n_components: amount of components in the approximating mixture
    :return:
    """
    # if os.path.exists(files_path_prefix + 'Synthetic/coeff_Kor'):
    #     shutil.rmtree(files_path_prefix + 'Synthetic/coeff_Kor')
    # os.mkdir(files_path_prefix + 'Synthetic/coeff_Kor')
    # os.mkdir(files_path_prefix + 'Synthetic/coeff_Kor/Components')
    # os.mkdir(files_path_prefix + 'Synthetic/coeff_Kor/Components/raw')
    # os.mkdir(files_path_prefix + 'Synthetic/coeff_Kor/Components/components-xlsx')
    # os.mkdir(files_path_prefix + 'Synthetic/coeff_Kor/Components/plots')
    # os.mkdir(files_path_prefix + 'Synthetic/coeff_Kor/Components/Sum')

    point_size = 1
    step_ticks = 10
    ticks_by_day = 30
    # draw = True
    draw = False
    for i in tqdm.tqdm(range(0, 20)):
        for j in range(0, 100):
            # apply EM
            sample = flux_data[time_start*ticks_by_day:time_end*ticks_by_day, i, j]
            # plt.hist(sample, bins=30)
            # plt.show()
            point_df = hybrid(sample, window_width * point_size, n_components, EM_steps=1, step=step_ticks * point_size)
            point_df.to_excel(files_path_prefix + f'Synthetic/coeff_Kor/Components/raw/point_({i}, {j}).xlsx',
                              index=False)

            df = pd.read_excel(files_path_prefix + f'Synthetic/coeff_Kor/Components/raw/point_({i}, {j}).xlsx')
            new_df, new_n_components = cluster_components(df,
                                                          n_components,
                                                          files_path_prefix,
                                                          draw,
                                                          path=f'Synthetic/coeff_Kor/Components/plots/',
                                                          postfix=f'_point_({i}, {j})')

            new_df.to_excel(
                files_path_prefix + f'Synthetic/coeff_Kor/Components/components-xlsx/point_({i}, {j}).xlsx',
                index=False)

            Kor_df = new_df
            Kor_df.fillna(0, inplace=True)
            a_sum = np.zeros(len(Kor_df))
            b_sum = np.zeros(len(Kor_df))

            for k in range(1, n_components + 1):
                a_sum += Kor_df[f'mean_{k}'] * Kor_df[f'weight_{k}']

            for k in range(1, n_components + 1):
                b_sum += Kor_df[f'weight_{k}'] * (np.square(Kor_df[f'mean_{k}']) + np.square(Kor_df[f'sigma_{k}']))

            b_sum = np.sqrt(b_sum - np.square(a_sum))

            a_sum = np.array(a_sum[:len(a_sum) - len(a_sum) % (ticks_by_day // step_ticks)])
            a_sum = np.mean(a_sum.reshape(-1, (ticks_by_day // step_ticks)), axis=1)
            b_sum = np.array(b_sum[:len(b_sum) - len(b_sum) % (ticks_by_day // step_ticks)])
            b_sum = np.mean(b_sum.reshape(-1, (ticks_by_day // step_ticks)), axis=1)

            np.save(files_path_prefix + f'Synthetic/coeff_Kor/Components/Sum/point_({i}, {j})-A.npy', a_sum)
            np.save(files_path_prefix + f'Synthetic/coeff_Kor/Components/Sum/point_({i}, {j})-B.npy', b_sum)
            if draw:
                plot_components(new_df,
                                new_n_components,
                                (i, j),
                                files_path_prefix,
                                path=f'Synthetic/coeff_Kor/Components/plots/',
                                postfix=f'_point_({i}, {j})')

                plot_a_sigma(df,
                             n_components,
                             (i, j),
                             files_path_prefix,
                             path=f'Synthetic/coeff_Kor/Components/plots/',
                             postfix=f'_point_({i}, {j})')

                count_Bel_Kor_difference(files_path_prefix,
                                         time_start,
                                         time_end,
                                         [(i, j)],
                                         point_size,
                                         (i, j),
                                         n_components,
                                         window_width,
                                         ticks_by_day,
                                         step_ticks,
                                         1,
                                         flux_type,
                                         0)
                plot_difference_1d(files_path_prefix,
                                   time_start,
                                   time_end,
                                   (i, j),
                                   window_width,
                                   0,
                                   ticks_by_day,
                                   step_ticks,
                                   n_components,
                                   flux_type,)
    return


def plot_synthetic_flux(files_path_prefix: str,
                        sensible: np.array,
                        latent: np.array,
                        time_start: int,
                        time_end: int,
                        width: int = 100,
                        height: int = 100):
    if not os.path.exists(files_path_prefix + 'Synthetic/Plots/Flux'):
        os.mkdir(files_path_prefix + 'Synthetic/Plots/Flux')

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    img_sens, img_lat = None, None

    flux_max = min(max(np.nanmax(sensible), np.nanmax(latent)), 1000)
    flux_min = min(np.nanmin(sensible), np.nanmin(latent))

    # cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - flux_min) / (flux_max - flux_min), 1])
    cmap = plt.get_cmap('Reds')
    cmap.set_bad('darkgreen', 1.0)

    axs[0].set_title(f'Sensible', fontsize=20)
    divider = make_axes_locatable(axs[0])
    cax_sens = divider.append_axes('right', size='5%', pad=0.3)

    axs[1].set_title(f'Latent', fontsize=20)
    divider = make_axes_locatable(axs[1])
    cax_lat = divider.append_axes('right', size='5%', pad=0.3)

    for t in tqdm.tqdm(range(time_start, time_end)):
        fig.suptitle(f'dX = {mu:.2f} * X + {sigma:.2f} * X \n t = {t}', fontsize=30)
        if img_sens is None:
            img_sens = axs[0].imshow(sensible[t],
                                     interpolation='none',
                                     cmap=cmap,
                                     vmin=flux_min,
                                     vmax=flux_max)
        else:
            img_sens.set_data(sensible[t])

        fig.colorbar(img_sens, cax=cax_sens, orientation='vertical')

        if img_lat is None:
            img_lat = axs[1].imshow(latent[t],
                                     interpolation='none',
                                     cmap=cmap,
                                     vmin=flux_min,
                                     vmax=flux_max)
        else:
            img_lat.set_data(latent[t])

        fig.colorbar(img_lat, cax=cax_lat, orientation='vertical')
        fig.savefig(files_path_prefix + f'Synthetic/Plots/Flux/Flux_{t:05d}.png')
    return


def plot_Kor_Bel_compare(files_path_prefix: str,
                         time_start: int,
                         time_end: int,
                         coeff_array: np.array,
                         flux_type: str,
                         coeff_type: str,
                         height: int = 100,
                         width: int = 100,):
    if not os.path.exists(files_path_prefix + f'Synthetic/Plots/{coeff_type}_{flux_type}'):
        os.mkdir(files_path_prefix + f'Synthetic/Plots/{coeff_type}_{flux_type}')

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    img = [None for _ in range(4)]

    # coeff_max = np.nanmax(coeff_array)
    # coeff_min = np.nanmin(coeff_array)

    coeff_max = 0
    coeff_min = 1

    # cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - coeff_min) / (coeff_max - coeff_min), 1])
    cmap = plt.get_cmap('Reds')
    cmap.set_bad('darkgreen', 1.0)

    cax = list()
    for i in range(4):
        divider = make_axes_locatable(axs[i // 2][i % 2])
        cax.append(divider.append_axes('right', size='5%', pad=0.3))
        # if i == 0:
        #     axs[i // 2][i % 2].set_title(f'Bel', fontsize=20)
        # elif i == 1 or i == 3:
        #     axs[i // 2][i % 2].set_title(f'Real', fontsize=20)
        # elif i == 2:
        #     axs[i // 2][i % 2].set_title(f'Kor', fontsize=20)

    for t in tqdm.tqdm(range(time_start, time_end)):
        coeff_string = f'{coeff_type}_sens_{t}.npy'
        if not os.path.exists(files_path_prefix + f'Synthetic/coeff_Bel/{coeff_string}'):
            coeff_Bel = np.zeros((height, width))
        else:
            coeff_Bel = np.load(files_path_prefix + f'Synthetic/coeff_Bel/{coeff_string}')

        if not os.path.exists(files_path_prefix + f'Synthetic/coeff_Kor/{coeff_string}'):
            coeff_Kor = np.zeros((height, width))
        else:
            coeff_Kor = np.load(files_path_prefix + f'Synthetic/coeff_Kor/Components/{coeff_string}')

        fig.suptitle(f'{coeff_type} {flux_type}\n day {t}', fontsize=30)

        diff_Bel = coeff_Bel - coeff_array[t]
        rmse_Bel = math.sqrt(sum(diff_Bel.flatten() ** 2))

        diff_Kor = coeff_Kor - coeff_array[t]
        rmse_Kor = math.sqrt(sum(diff_Kor.flatten() ** 2))

        for i in range(4):
            if i == 0:
                axs[i // 2][i % 2].set_title(f'Bel, error = {rmse_Bel:.1f}', fontsize=20)
            elif i == 1 or i == 3:
                axs[i // 2][i % 2].set_title(f'Real', fontsize=20)
            elif i == 2:
                axs[i // 2][i % 2].set_title(f'Kor, error = {rmse_Kor:.1f}', fontsize=20)

        for i in [1, 3]:
            if img[i] is None:
                img[i] = axs[i // 2][i % 2].imshow(coeff_array[t],
                                                    interpolation='none',
                                                    cmap=cmap,
                                                    vmin=coeff_min,
                                                    vmax=coeff_max)
            else:
                img[i].set_data(coeff_array[t])
            if img[0] is None:
                img[0] = axs[0][0].imshow(coeff_Bel,
                                        interpolation='none',
                                        cmap=cmap,
                                        vmin=coeff_min,
                                        vmax=coeff_max)
            else:
                img[0].set_data(coeff_Bel)

            if img[2] is None:
                img[2] = axs[1][0].imshow(coeff_Kor,
                                        interpolation='none',
                                        cmap=cmap,
                                        vmin=coeff_min,
                                        vmax=coeff_max)
            else:
                img[2].set_data(coeff_Kor)

        for i in range(4):
            fig.colorbar(img[i], cax=cax[i], orientation='vertical')

        fig.savefig(files_path_prefix + f'Synthetic/Plots/{coeff_type}_{flux_type}/{t}.png')
    return


def collect_Kor(files_path_prefix: str,
                width: int = 100,
                height: int = 100,
                time_start: int = 0,
                time_end: int = 1,
                ):
    map_array = np.zeros((time_end - time_start, height, width), dtype=float)
    for i in tqdm.tqdm(range(height)):
        for j in range(width):
            try:
                point_arr = np.load(files_path_prefix + f'Synthetic/coeff_Kor/Components/Sum/point_({i}, {j})-A.npy')
                map_array[:len(point_arr), i, j] = point_arr
            except FileNotFoundError:
                map_array[:, i, j] = np.nan
            except ValueError:
                print(f'Value error point ({i}, {j})')
        if i % 10 == 0:
            np.save(files_path_prefix + f'Synthetic/coeff_Kor/Components/full_array_A.npy', map_array)
    np.save(files_path_prefix + f'Synthetic/coeff_Kor/Components/full_array_A.npy', map_array)
    return
