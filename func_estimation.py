import matplotlib.pyplot as plt
import matplotlib
import tqdm
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from EM_staff import moving_average
import math
import matplotlib.ticker as mtick
import pandas as pd
import os
from data_processing import load_ABCF
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm


def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, ListedColormap):
            use_index=True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(0,1,N))
        return cycler("color",colors)


def plot_a_by_flux(files_path_prefix, a_timelist, borders, sensible_array, latent_array, time_start, time_end, step=1, start_pic_num=0):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    for t in tqdm.tqdm(range(time_start, time_end, step)):
        sens_set = set(sensible_array[:, t])
        lat_set = set(latent_array[:, t])

        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=start_pic_num + (t - time_start))
        a_sens = a_timelist[t][0]
        a_lat = a_timelist[t][1]
        sens_list, lat_list = list(), list()
        sens_x, lat_x = list(), list()
        for val in sorted(list(sens_set)):
            if not np.isnan(val):
                points_sensible = np.nonzero(sensible_array[:, t] == val)[0]
                sens_list.append(a_sens[points_sensible[0]//181, points_sensible[0] % 181])
                sens_x.append(val)

        for val in sorted(list(lat_set)):
            if not np.isnan(val):
                points_latent = np.nonzero(latent_array[:, t] == val)[0]
                lat_list.append(a_lat[points_latent[0]//181, points_latent[0] % 181])
                lat_x.append(val)

        sens_list = [x for _, x in sorted(zip(sens_x, sens_list))]
        sens_x = list(sorted(sens_x))
        lat_list = [x for _, x in sorted(zip(lat_x, lat_list))]
        lat_x = list(sorted(lat_x))

        fig.suptitle(f'A-flux value dependence \n {date.strftime("%Y-%m-%d")}', fontsize=25)
        axs[0].cla()
        axs[0].plot(sens_x, sens_list, c='r')
        axs[0].set_ylabel('A_sens', fontsize=20)
        axs[0].set_xlabel('Sensible flux', fontsize=20)
        axs[0].set_ylim([borders[0], borders[1]])

        axs[1].cla()
        axs[1].plot(lat_x, lat_list, c='orange')
        axs[1].set_ylabel('A_lat', fontsize=20)
        axs[1].set_xlabel('Latent flux', fontsize=20)
        axs[1].set_ylim([borders[0], borders[1]])
        fig.savefig(files_path_prefix + f'Func_repr/a-flux/{t:05d}.png')
    return


def func(x, a, b, c, d):
    # return a * pow(x, 2) + b * x + c
    return a * np.sin(b*x + c) + d


def plot_estimate_a_flux(files_path_prefix, a_timelist, borders, sensible_array, latent_array, time_start, time_end, step=1, month=1):
    fig, axs = plt.subplots(1, 2, figsize=(25, 10))
    date_start = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
    date_end = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_end)

    # print('Getting sets')
    sens_set = np.unique(sensible_array[:, time_start:time_end])
    lat_set = np.unique(latent_array[:, time_start:time_end])

    # print('Sorting sets')
    sens_set = sorted(list(sens_set))
    lat_set = sorted(list(lat_set))

    # print('Counting y')
    sens_x, lat_x, sens_y, lat_y = list(), list(), list(), list()
    for t in range(time_start, time_end, step):
        a_sens = a_timelist[t - time_start][0]
        a_lat = a_timelist[t-time_start][1]

        for val in sens_set:
            if not np.isnan(val):
                points_sensible = np.nonzero(sensible_array[:, t] == val)[0]
                if len(points_sensible):
                    sens_y.append(a_sens[points_sensible[0]//181, points_sensible[0] % 181])
                    sens_x.append(val)

        for val in lat_set:
            if not np.isnan(val):
                points_latent = np.nonzero(latent_array[:, t] == val)[0]
                if len(points_latent):
                    lat_y.append(a_lat[points_latent[0]//181, points_latent[0] % 181])
                    lat_x.append(val)

    # print('Estimating')
    # t_sens, sens_fit, params_sens = estimate_with_sin(sens_x, sens_y)
    # t_lat, lat_fit, params_lat = estimate_with_sin(lat_x, lat_y)
    # t_sens = np.linspace(min(sens_x), max(sens_x), len(sens_x))
    # t_lat = np.linspace(min(lat_x), max(lat_x), len(lat_x))
    sens_x, sens_y = zip(*sorted(zip(sens_x, sens_y)))
    lat_x, lat_y = zip(*sorted(zip(lat_x, lat_y)))
    sens_x = np.array(sens_x)
    lat_x = np.array(lat_x)
    sens_y = np.array(sens_y)
    lat_y = np.array(lat_y)
    sens_popt, sens_pcov = curve_fit(func, sens_x, sens_y, p0=[20, (math.pi / 2)/ 50, math.pi/6, 5], maxfev=10000)
    sens_residuals = sens_y - func(sens_x, *sens_popt)
    sens_ss = np.sum(sens_residuals ** 2)

    lat_popt, lat_pcov = curve_fit(func, lat_x, lat_y, p0=[10, (math.pi / 2)/ 40, math.pi/6, 5], maxfev=10000)
    lat_residuals = lat_y - func(lat_y, *lat_popt)
    lat_ss = np.sum(lat_residuals ** 2)

    # sens_mean = list()
    # sens_mean_x = list()
    # lat_mean = list()
    # lat_mean_x = list()
    # for v in sens_set:
    #     if not np.isnan(v):
    #         points = np.nonzero(sens_x == v)
    #         sens_mean.append(np.mean(sens_y[points]))
    #         sens_mean_x.append(v)
    #
    # for v in lat_set:
    #     if not np.isnan(v):
    #         points = np.nonzero(lat_x == v)
    #         lat_mean.append(np.mean(lat_y[points]))
    #         lat_mean_x.append(v)

    fig.suptitle(f'A-flux value dependence \n {date_start.strftime("%Y-%m-%d")} - {date_end.strftime("%Y-%m-%d")}', fontsize=25)
    axs[0].cla()
    axs[0].scatter(sens_x, sens_y, c='r')
    a, b, c, d = sens_popt
    # label = f'{a:.3f} * x^2 + {b:.3f} * x + {c:.3f}'
    label = f'{a:.1f} * sin({b:.5f} * x + {c:.1f}) + {d:.1f}'
    axs[0].plot(sens_x, func(sens_x, *sens_popt), c='b', label=label)
    # axs[0].plot(sens_mean_x, sens_mean, c='g', label='mean')
    axs[0].set_ylabel('A_sens', fontsize=20)
    axs[0].set_xlabel('Sensible flux', fontsize=20)
    axs[0].set_ylim([borders[0], borders[1]])
    axs[0].set_title(f'Sum of squared residuals = {sens_ss:.2f}')
    axs[0].legend()

    axs[1].cla()
    axs[1].scatter(lat_x, lat_y, c='orange')
    a, b, c, d = lat_popt
    # label = f'{a:.3f} * x^2 + {b:.3f} * x + {c:.3f}'
    label = f'{a:.1f} * sin({b:.5f} * x + {c:.1f}) + {d:.1f}'
    axs[1].plot(lat_x, func(lat_x, *lat_popt), c='b', label=label)
    # axs[1].plot(lat_mean_x, lat_mean, c='g', label='mean')
    axs[1].set_ylabel('A_lat', fontsize=20)
    axs[1].set_xlabel('Latent flux', fontsize=20)
    axs[1].set_ylim([borders[0], borders[1]])
    axs[1].set_title(f'Sum of squared residuals = {lat_ss:.2f}')
    axs[1].legend()
    fig.savefig(files_path_prefix + f'Func_repr/a-flux-monthly/{month}/{time_start:05d}.png')
    return sens_popt, lat_popt, sens_ss, lat_ss


def estimate_by_months(files_path_prefix, month):
    sensible_array = np.load(files_path_prefix + 'sensible_all.npy')
    latent_array = np.load(files_path_prefix + 'latent_all.npy')

    df_sens = pd.DataFrame(columns=['dates', 'a', 'b', 'c', 'd', 'ss'])
    df_lat = pd.DataFrame(columns=['dates', 'a', 'b', 'c', 'd', 'ss'])

    months_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                    9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    font = {'size': 14}
    matplotlib.rc('font', **font)

    if not os.path.exists(files_path_prefix + f"Func_repr/a-flux-monthly/{month}"):
        os.mkdir(files_path_prefix + f"Func_repr/a-flux-monthly/{month}")

    # estimate and save
    for i in range(0, 43):
        print(i)
        time_start = (datetime.datetime(1979 + i, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
        time_end = (datetime.datetime(1979 + i, 2, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days

        a_timelist, _, _, _, borders = load_ABCF(files_path_prefix, time_start + 1, time_end + 1, load_a=True)
        sens_params, lat_params, sens_err, lat_err = plot_estimate_a_flux(files_path_prefix, a_timelist, borders, sensible_array, latent_array, time_start, time_end, month=month)
        del a_timelist
        date_start = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
        date_end = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
        df_sens.loc[len(df_sens)] = [f"{date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')}"] + list(sens_params) + [sens_err]
        df_lat.loc[len(df_lat)] = [f"{date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')}"] + list(lat_params) + [lat_err]
    df_sens.to_csv(files_path_prefix + f"Func_repr/a-flux-monthly/{month}/sens_params_{months_names[month]}.csv", index=False, sep=';')
    df_lat.to_csv(files_path_prefix + f"Func_repr/a-flux-monthly/{month}/lat_params_{months_names[month]}.csv", index=False, sep=';')

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))
    x = np.linspace(np.nanmin(sensible_array), np.nanmax(sensible_array), 100)
    for i in range(0, 43):
            sens_params = df_sens[['a', 'b', 'c', 'd']].loc[i].values
            lat_params = df_lat[['a', 'b', 'c', 'd']].loc[i].values
            if i > 30:
                color = plt.cm.tab20(i % 20)
            else:
                color = 'gray'
            axes[0].plot(x, func(x, *sens_params), label=f'{1979 + i}', c=color)
            axes[1].plot(x, func(x, *lat_params), label=f'{1979 + i}', c=color)

    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1.0), ncol=2, fancybox=True, shadow=True)
    axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1.0), ncol=2, fancybox=True, shadow=True)
    axes[0].set_title('Sensible')
    axes[1].set_title('Latent')
    fig.suptitle(months_names[month])
    fig.tight_layout()
    fig.savefig(files_path_prefix + f"Func_repr/a-flux-monthly/{months_names[month]}.png")

    fig, axs = plt.subplots(figsize=(10, 5))
    axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    # axs.set_xticks(np.arange(41))
    # axs.set_xticklabels(range(1979, 2020))
    plt.bar(range(1979, 2022), df_sens['ss'])
    fig.suptitle('Sensible - sum of squared residuals')
    fig.savefig(files_path_prefix + f"Func_repr/a-flux-monthly/{month}/{month}_error_sensible.png")

    fig, axs = plt.subplots(figsize=(10, 5))
    axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    # axs.set_xticks(np.arange(41))
    # axs.set_xticklabels(range(1979, 2020))
    plt.bar(range(1979, 2022), df_lat['ss'])
    fig.suptitle('Latent - sum of squared residuals')
    fig.savefig(files_path_prefix + f"Func_repr/a-flux-monthly/{month}/{month}_error_latent.png")
    return