import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import datetime
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from video import get_continuous_cmap


def count_abf_Kor_from_points(files_path_prefix: str, time_start: int, time_end: int, point_start: int, point_end: int):
    components = 3
    sens_df_list = list()
    sens_idxes = list()

    print('Loading data')
    for p in tqdm.tqdm(range(point_start, point_end)):
        if os.path.exists(files_path_prefix + f'Components/raw_sensible/point_{p}.xlsx'):
            raw_sens = pd.read_excel(files_path_prefix + f'Components/raw_sensible/point_{p}.xlsx')
            sens_df_list.append(raw_sens)
            sens_idxes.append(p)

    print('Counting difference')
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


def count_Bel_Kor_difference(files_path_prefix, time_start: int, time_end: int, point_start: int, point_end: int):
    for t in tqdm.tqdm(range(time_start, time_end)):
        for p in range(point_start, point_end):
            if os.path.exists(files_path_prefix + f'Components/sensible/{t}_A_sens.npy'):
                a_sens_Bel = np.load(files_path_prefix + f'Coeff_data/{t}_A_sens.npy')
                a_sens_Kor = np.load(files_path_prefix + f'Components/sensible/{t}_A_sens.npy')
                a_diff = a_sens_Bel - a_sens_Kor
                a_diff[a_sens_Kor == 0] = 0
                a_diff[np.isnan(a_sens_Bel)] = np.nan
                np.save(files_path_prefix + f'Components/difference/{t}_A_sens.npy', a_diff)
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


def plot_difference_1d(files_path_prefix, time_start: int, time_end: int, point_start: int, point_end: int):
    components = 3

    a_sens_list = list()
    for t in range(time_start, time_end):
        a_sens = np.load(files_path_prefix + f'Coeff_data/{t}_A_sens.npy')
        a_sens_list.append(a_sens)

    for p in tqdm.tqdm(range(point_start, point_end)):
        if os.path.exists(files_path_prefix + f'Components/raw_sensible/point_{p}.xlsx'):
            df = pd.read_excel(files_path_prefix + f'Components/raw_sensible/point_{p}.xlsx')
            df.fillna(0, inplace=True)
            sens_Kor_ts = 0
            for comp in range(1, components + 1):
                sens_Kor_ts += df[f'mean_{comp}'] * df[f'weight_{comp}']
            sens_Bel_ts = list()
            for i in range(len(a_sens_list)):
                sens_Bel_ts.append(a_sens_list[i][p // 181, p % 181])

            diff = np.abs(np.array(sens_Kor_ts[time_start:time_end] - sens_Bel_ts))
            np.save(files_path_prefix + f'Components/difference_1d/point_{p}.npy', diff)

            fig = plt.figure(figsize=(30, 10))
            fig.suptitle(f'Point {p}')
            plt.plot(range(time_start, time_end), sens_Bel_ts, c='b', label='Bel')

            colors = [0, 'Lime', 'Orchid', 'Purple']
            for comp in range(1, components + 1):
                plt.plot(range(time_start, time_end),
                         np.array(df[f'mean_{comp}'] * df[f'weight_{comp}'])[time_start:time_end],
                         c=colors[comp], label=f'Comp_{comp}')
            plt.plot(range(time_start, time_end), sens_Kor_ts[time_start:time_end], c='y', label='Kor')
            plt.plot(range(time_start, time_end), diff, c='r', label='difference')
            plt.legend()
            plt.savefig(files_path_prefix + f'Components/tmp/point_{p}.png')
    return