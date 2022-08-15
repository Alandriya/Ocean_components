import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import datetime
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from video import get_continuous_cmap
import matplotlib.dates as mdates


def count_abf_Kor_from_points(files_path_prefix: str, time_start: int, time_end: int, point_start: int, point_end: int):
    components = 3
    sens_df_list = list()
    sens_idxes = list()

    n_components = 3
    means_cols = [f'mean_{i}' for i in range(1, n_components + 1)]
    sigmas_cols = [f'sigma_{i}' for i in range(1, n_components + 1)]
    weights_cols = [f'weight_{i}' for i in range(1, n_components + 1)]

    print('Loading data')
    for p in tqdm.tqdm(range(point_start, point_end)):
        if os.path.exists(files_path_prefix + f'Components/sensible/raw/point_{p}.xlsx'):
            raw_sens = pd.read_excel(files_path_prefix + f'Components/sensible/raw/point_{p}.xlsx')
            raw_sens.columns = ['time', 'ts'] + means_cols + sigmas_cols + weights_cols
            sens_df_list.append(raw_sens)
            sens_idxes.append(p)

    for t in tqdm.tqdm(range(time_start, time_end)):
        a_sens = np.zeros((161, 181))
        # a_lat = np.zeros((161, 181))
        # b_matrix = np.zeros((4, 161, 181))
        # f = np.zeros((161, 181), dtype=float)

        for i in range(len(sens_idxes)):
            p = sens_idxes[i]
            df = sens_df_list[i]
            a_sens[p // 181, p % 181] = sum([df.loc[t, f'mean_{comp}'] * df.loc[t, f'weight_{comp}']
                                             for comp in range(1, components + 1)])
            # b_matrix[0, p // 181, p % 181] = None

        np.save(files_path_prefix + f'Components/sensible/{t}_A_sens.npy', a_sens)
    return


def count_Bel_Kor_difference(files_path_prefix, time_start: int, time_end: int, point: tuple, radius: int,
                             n_components:int, window_width:int):
    point_bigger = [(point[0] + i, point[1] + j) for i in range(-radius, radius + 1) for j in
                    range(-radius, radius + 1)]

    a_sens_Bel = np.zeros(time_end-time_start)
    point_size = (radius * 2 + 1) ** 2

    if not os.path.exists(files_path_prefix + f'Components/sensible/Bel/point_({point[0]}, {point[1]}).npy'):
        for t in tqdm.tqdm(range(time_start, time_end)):
            a_arr = np.load(files_path_prefix + f'Coeff_data/{t}_A_sens.npy')
            a_sens_Bel[t-time_start] = sum([a_arr[p[0], p[1]] for p in point_bigger]) / point_size
            del a_arr

        a_sens_Bel = np.diff(a_sens_Bel)
        np.save(files_path_prefix + f'Components/sensible/Bel/point_({point[0]}, {point[1]}).npy', a_sens_Bel)
    else:
        a_sens_Bel = np.load(files_path_prefix + f'Components/sensible/Bel/point_({point[0]}, {point[1]}).npy')

    if not os.path.exists(files_path_prefix + f'Components/sensible/Sum/point_({point[0]}, {point[1]}).npy'):
        a_sens_Kor = pd.read_excel(files_path_prefix + f'Components/sensible/point_({point[0]}, {point[1]}).xlsx')
        a_sens_Kor.fillna(0, inplace=True)
        a_sum = np.zeros(len(a_sens_Kor))
        for i in range(1, n_components+1):
            a_sum += a_sens_Kor[f'mean_{i}'] * a_sens_Kor[f'weight_{i}']

        np.save(files_path_prefix + f'Components/sensible/Sum/point_({point[0]}, {point[1]}).npy', a_sum)
    else:
        a_sum = np.load(files_path_prefix + f'Components/sensible/Sum/point_({point[0]}, {point[1]}).npy')

    a_diff = a_sens_Bel[:len(a_sum)] - a_sum
    np.save(files_path_prefix + f'Components/sensible/difference/point_({point[0]}, {point[1]}).npy', a_diff)
    return


def plot_a_Kor(files_path_prefix, time_start, time_end, start_pic_num):
    figa, axsa = plt.subplots(1, 2, figsize=(20, 10))
    img_a_sens, img_a_lat = None, None
    axsa[1].set_title(f'Latent', fontsize=20)
    divider = make_axes_locatable(axsa[1])
    cax_a_lat = divider.append_axes('right', size='5%', pad=0.3)

    axsa[0].set_title(f'Sensible', fontsize=20)
    divider = make_axes_locatable(axsa[0])
    cax_a_sens = divider.append_axes('right', size='5%', pad=0.3)

    pic_num = start_pic_num
    for t in tqdm.tqdm(range(time_start, time_end)):
        a_sens_Kor = np.load(files_path_prefix + f'Components/sensible/{t}_A_sens.npy')
        a_lat_Kor = np.zeros((161, 181))

        a_max = np.nanmax(a_sens_Kor)
        a_min = np.nanmin(a_sens_Kor)
        print([a_min, a_max])

        # cmap_a = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - a_min) / (a_max - a_min), 1])
        cmap_a = matplotlib.cm.get_cmap("YlOrRd").copy()
        cmap_a.set_bad('darkgreen', 1.0)
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=start_pic_num + (t - time_start))
        figa.suptitle(f'A coeff\n {date.strftime("%Y-%m-%d")}', fontsize=30)
        if img_a_sens is None:
            img_a_sens = axsa[0].imshow(a_sens_Kor,
                                        interpolation='none',
                                        cmap=cmap_a,
                                        vmin=a_min,
                                        vmax=a_max)
        else:
            img_a_sens.set_data(a_sens_Kor)

        figa.colorbar(img_a_sens, cax=cax_a_sens, orientation='vertical')

        if img_a_lat is None:
            img_a_lat = axsa[1].imshow(a_lat_Kor,
                                       interpolation='none',
                                       cmap=cmap_a,
                                       vmin=a_min,
                                       vmax=a_max)
        else:
            img_a_lat.set_data(a_lat_Kor)

        figa.colorbar(img_a_lat, cax=cax_a_lat, orientation='vertical')
        figa.savefig(files_path_prefix + f'videos/A_Kor/A_{pic_num:05d}.png')
        pic_num += 1
    return


def plot_a_diff(files_path_prefix, time_start, time_end, start_pic_num):
    figa, axsa = plt.subplots(1, 2, figsize=(20, 10))
    img_a_sens, img_a_lat = None, None
    axsa[1].set_title(f'Latent', fontsize=20)
    divider = make_axes_locatable(axsa[1])
    cax_a_lat = divider.append_axes('right', size='5%', pad=0.3)

    axsa[0].set_title(f'Sensible', fontsize=20)
    divider = make_axes_locatable(axsa[0])
    cax_a_sens = divider.append_axes('right', size='5%', pad=0.3)

    pic_num = start_pic_num
    for t in tqdm.tqdm(range(time_start, time_end)):
        a_sens_diff = np.load(files_path_prefix + f'Components/difference/{t}_A_sens.npy')
        a_lat_diff = np.zeros((161, 181))

        a_max = np.nanmax(a_sens_diff)
        a_min = np.nanmin(a_sens_diff)
        # print([a_min, a_max])

        cmap_a = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - a_min) / (a_max - a_min), 1])
        cmap_a.set_bad('darkgreen', 1.0)
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=start_pic_num + (t - time_start))
        figa.suptitle(f'A coeff\n {date.strftime("%Y-%m-%d")}', fontsize=30)
        if img_a_sens is None:
            img_a_sens = axsa[0].imshow(a_sens_diff,
                                        interpolation='none',
                                        cmap=cmap_a,
                                        vmin=a_min,
                                        vmax=a_max)
        else:
            img_a_sens.set_data(a_sens_diff)

        figa.colorbar(img_a_sens, cax=cax_a_sens, orientation='vertical')

        if img_a_lat is None:
            img_a_lat = axsa[1].imshow(a_lat_diff,
                                       interpolation='none',
                                       cmap=cmap_a,
                                       vmin=a_min,
                                       vmax=a_max)
        else:
            img_a_lat.set_data(a_lat_diff)

        figa.colorbar(img_a_lat, cax=cax_a_lat, orientation='vertical')
        figa.savefig(files_path_prefix + f'videos/A_difference/A_{pic_num:05d}.png')
        pic_num += 1
    return


def plot_difference_1d(files_path_prefix, time_start: int, time_end: int, point: tuple, window_width:int, radius:int):
    a_diff = np.load(files_path_prefix + f'Components/sensible/difference/point_({point[0]}, {point[1]}).npy')
    a_sens_Bel = np.load(files_path_prefix + f'Components/sensible/Bel/point_({point[0]}, {point[1]}).npy')
    a_sum = np.load(files_path_prefix + f'Components/sensible/Sum/point_({point[0]}, {point[1]}).npy')
    a_sum /= window_width

    date_start = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
    date_end = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
    fig, axs = plt.subplots(1, 1, figsize=(20, 10))
    fig.suptitle(f'A coeff sensible at point ({point[0]}, {point[1]}) \n radius = {radius}, window = {window_width} days'
                 f'\n {date_start.strftime("%Y-%m-%d")} - {date_end.strftime("%Y-%m-%d")}', fontsize=20, fontweight='bold')
    axs.xaxis.set_minor_locator(mdates.MonthLocator())
    axs.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs.xaxis.get_major_locator()))
    x = range(0, time_end - time_start)
    days = [datetime.datetime(1979, 1, 1) + datetime.timedelta(days=t) for t in range(time_start, time_end)]
    # dates = None
    axs.plot(days, a_sens_Bel[:len(x)], c='b', label='Bel')
    axs.plot(days, a_sum[:len(x)], c='r', label='Kor')
    # axs.plot(x, a_diff[:len(x)], c='y', label='difference')
    axs.legend()
    fig.tight_layout()
    fig.savefig(files_path_prefix + f'Components/plots/sensible/difference_point_({point[0]}, {point[1]}).png')
    return
