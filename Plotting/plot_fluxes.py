import datetime
import os.path
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Plotting.video import get_continuous_cmap


def plot_flux_correlations(files_path_prefix: str,
                           time_start: int,
                           time_end: int,
                           step: int = 1,
                           start_pic_num: int = 0):
    """
    Plots fluxes correlation and saves them into files_path_prefix + videos/Flux-corr directory

    :param files_path_prefix: path to the working directory
    :param time_start: start point for time
    :param time_end: end point for time
    :param step: step in time for loop
    :param start_pic_num: number of first picture
    :return:
    """
    fig, axs = plt.subplots(figsize=(15, 15))
    pic_num = start_pic_num
    for t in tqdm.tqdm(range(time_start, time_end, step)):
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=start_pic_num + (t - time_start))
        corr = np.load(files_path_prefix + f'Flux_correlations/FL_Corr_{t}.npy')
        fig.suptitle(f'Flux correlation\n {date.strftime("%Y-%m-%d")}', fontsize=30)

        cmap = get_continuous_cmap(['#4073ff', '#ffffff', '#ffffff', '#db4035'], [0, 0.4, 0.6, 1])
        cmap.set_bad('darkgreen', 1.0)
        im = axs.imshow(corr,
                        interpolation='none',
                        cmap=cmap,
                        vmin=-1,
                        vmax=1)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes('right', size='5%', pad=0.3)
        fig.colorbar(im, cax=cax, orientation='vertical')

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'videos/Flux-corr/FL_corr_{pic_num:05d}.png')
        pic_num += 1
    return


def plot_typical_points(files_path_prefix: str, mask: np.ndarray):
    """
    Creates and plots in points.png points where the fluxes distribution is observed

    :param files_path_prefix: path to the working directory
    :param mask: boolean 1D mask with length 161*181. If true, it's ocean point, if false - land. Only ocean points are
        of interest
    :return: list of points coordinates in tuple
    """
    mask_map = deepcopy(mask).reshape((161, 181))
    fig, axs = plt.subplots(figsize=(15, 15))

    cmap = matplotlib.colors.ListedColormap(['green', 'white', 'red'])
    norm = matplotlib.colors.BoundaryNorm([0, 1, 2, 3], cmap.N)

    points = [(15, 160), (15, 60), (40, 10), (40, 60), (45, 90), (40, 120), (40, 150), (60, 90), (60, 120), (60, 150),
              (90, 40), (90, 60), (90, 90), (90, 120), (90, 150), (110, 40), (110, 60), (110, 90), (110, 120),
              (110, 10), (130, 20), (150, 10), (130, 40), (130, 60), (130, 90), (130, 120), (150, 90), (150, 120)]

    points_bigger = [(p[0] + i, p[1] + j) for p in points for i in [-1, 0, 1] for j in [-1, 0, 1]]
    for point in points_bigger:
        mask_map[point] = 2

    axs.imshow(mask_map, interpolation='none', cmap=cmap, norm=norm)
    # fig.savefig(files_path_prefix + f'Func_repr/fluxes_distribution/points.png')
    fig.savefig(files_path_prefix + f'Components/points.png')
    return points


def plot_current_bigpoint(files_path_prefix: str, mask: np.ndarray, point: tuple, radius: int):
    """
    Plots the "big point" which is a square area with the center in point and length of side 2*radius
    :param files_path_prefix: path to the working directory
    :param mask: boolean 1D mask with length 161*181. If true, it's ocean point, if false - land. Only ocean points are
        of interest
    :param point: tuple with point coordinates in a grid
    :param radius: half of side length of a square
    :return:
    """
    mask_map = deepcopy(mask).reshape((161, 181))
    fig, axs = plt.subplots(figsize=(15, 15))

    cmap = matplotlib.colors.ListedColormap(['green', 'white', 'red'])
    norm = matplotlib.colors.BoundaryNorm([0, 1, 2, 3], cmap.N)

    biases = [i for i in range(-radius, radius + 1)]
    point_bigger = [(point[0] + i, point[1] + j) for i in biases for j in biases]
    for p in point_bigger:
        mask_map[p] = 2

    axs.imshow(mask_map, interpolation='none', cmap=cmap, norm=norm)
    fig.savefig(files_path_prefix + f'Func_repr/fluxes_distribution/POINT_({point[0]},{point[1]})/point_({point[0]},{point[1]}).png')
    return


def plot_fluxes(files_path_prefix: str,
                sensible: np.ndarray,
                latent: np.ndarray,
                start: int,
                end: int,
                group: int = 4,
                start_date: datetime.datetime = datetime.datetime(1979, 1, 1, 0, 0),
                start_pic_num: int = 1):
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    img_sens, img_lat = None, None

    flux_max = max(np.nanmax(sensible), np.nanmax(latent))
    flux_min = min(np.nanmin(sensible), np.nanmin(latent))

    cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - flux_min) / (flux_max - flux_min), 1])
    cmap.set_bad('lightgreen', 1.0)
    # cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - flux_min) / (flux_max - flux_min), 1])
    # cmap.set_bad('darkgreen', 1.0)

    # axs[0].set_title(f'Sensible', fontsize=20)
    divider = make_axes_locatable(axs[0])
    cax_sens = divider.append_axes('right', size='5%', pad=0.3)

    # axs[1].set_title(f'Latent', fontsize=20)
    divider = make_axes_locatable(axs[1])
    cax_lat = divider.append_axes('right', size='5%', pad=0.3)

    x_label_list = ['90W', '60W', '30W', '0']
    y_label_list = ['EQ', '30N', '60N', '80N']
    xticks = [0, 60, 120, 180]
    yticks = [160, 100, 40, 0]

    pic_num = start_pic_num
    for t in tqdm.tqdm(range(start, end)):
        date = start_date + datetime.timedelta(hours=6 * group * (t - start))

        # if group == 1:
        #     fig.suptitle(f'Fluxes\n {date.strftime("%Y-%m-%d %H:00")}', fontsize=30)
        # else:
        #     fig.suptitle(f'Fluxes\n {date.strftime("%Y-%m-%d")}', fontsize=30)
        if img_sens is None:
            img_sens = axs[0].imshow(sensible[:, t].reshape(161, 181),
                                     interpolation='none',
                                     cmap=cmap,
                                     vmin=flux_min,
                                     vmax=flux_max)
            axs[0].set_xticks(xticks)
            axs[0].set_yticks(yticks)
            axs[0].set_xticklabels(x_label_list)
            axs[0].set_yticklabels(y_label_list)
        else:
            img_sens.set_data(sensible[:, t].reshape(161, 181))

        fig.colorbar(img_sens, cax=cax_sens, orientation='vertical')

        if img_lat is None:
            img_lat = axs[1].imshow(latent[:, t].reshape(161, 181),
                                     interpolation='none',
                                     cmap=cmap,
                                     vmin=flux_min,
                                     vmax=flux_max)
            axs[1].set_xticks(xticks)
            axs[1].set_yticks(yticks)
            axs[1].set_xticklabels(x_label_list)
            axs[1].set_yticklabels(y_label_list)
        else:
            img_lat.set_data(latent[:, t].reshape(161, 181))

        fig.colorbar(img_lat, cax=cax_lat, orientation='vertical')
        fig.tight_layout()
        fig.savefig(files_path_prefix + f'videos/Fluxes/Flux_{pic_num:05d}.png')
        pic_num += 1
    return


def plot_flux_sst_press(files_path_prefix: str,
                        flux: np.ndarray,
                        sst: np.ndarray,
                        press: np.ndarray,
                        start: int,
                        end: int,
                        start_date: datetime.datetime = datetime.datetime(1979, 1, 1, 0, 0),
                        start_pic_num: int = 1,
                        frequency: int = 1,
                        ):
    sns.set_style("whitegrid")

    if not os.path.exists(files_path_prefix + f'videos/3D/flux-sst-press/hourly'):
        os.mkdir(files_path_prefix + f'videos/3D/flux-sst-press/hourly')

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    img_flux, img_sst, img_press = None, None, None

    flux_max = np.nanmax(flux)
    flux_min = np.nanmin(flux)
    sst_max = np.nanmax(sst)
    sst_min = np.nanmin(sst)
    press_max = np.nanmax(press)
    press_min = np.nanmin(press)

    cmap_flux = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - flux_min) / (flux_max - flux_min), 1])
    cmap_flux.set_bad('lightgreen', 1.0)
    # cmap_sst = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
    cmap_sst = plt.get_cmap('Reds').copy()
    cmap_sst.set_bad('lightgreen', 1.0)
    # cmap_press = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
    cmap_press = plt.get_cmap('Purples').copy()
    cmap_press.set_bad('lightgreen', 1.0)

    # axs[0].set_title(f'Sum flux', fontsize=20)
    divider = make_axes_locatable(axs[0])
    cax_flux = divider.append_axes('right', size='5%', pad=0.3)

    # axs[1].set_title(f'SST', fontsize=20)
    divider = make_axes_locatable(axs[1])
    cax_sst = divider.append_axes('right', size='5%', pad=0.3)

    # axs[2].set_title(f'Pressure', fontsize=20)
    divider = make_axes_locatable(axs[2])
    cax_press = divider.append_axes('right', size='5%', pad=0.3)

    x_label_list = ['90W', '60W', '30W', '0']
    y_label_list = ['EQ', '30N', '60N', '80N']
    xticks = [0, 60, 120, 180]
    yticks = [160, 100, 40, 0]

    pic_num = start_pic_num
    for t in tqdm.tqdm(range(start, end)):
        if frequency == 1:
            date = start_date + datetime.timedelta(days=(t - start))
            fig.suptitle(f'{date.strftime("%Y-%m-%d")}', fontsize=30)

        elif frequency == 24:
            date = start_date + datetime.timedelta(hours=(t - start))
            fig.suptitle(f'{date.strftime("%Y-%m-%d %H:00")}', fontsize=30)
        # fig.suptitle(f'{date.strftime("%Y-%m-%d")}', fontsize=30)

        if img_flux is None:
            img_flux = axs[0].imshow(flux[:, t].reshape(161, 181)[::2, ::2],
                                     interpolation='none',
                                     cmap=cmap_flux,
                                     vmin=flux_min,
                                     vmax=flux_max)
            axs[0].set_xticks(xticks)
            axs[0].set_yticks(yticks)
            axs[0].set_xticklabels(x_label_list)
            axs[0].set_yticklabels(y_label_list)

            img_sst = axs[1].imshow(sst[:, t].reshape(161, 181)[::2, ::2],
                                     interpolation='none',
                                     cmap=cmap_sst,
                                     vmin=sst_min,
                                     vmax=sst_max)
            axs[1].set_xticks(xticks)
            axs[1].set_yticks(yticks)
            axs[1].set_xticklabels(x_label_list)
            axs[1].set_yticklabels(y_label_list)

            img_press = axs[2].imshow(press[:, t].reshape(161, 181)[::2, ::2],
                                     interpolation='none',
                                     cmap=cmap_press,
                                     vmin=press_min,
                                     vmax=press_max)
            axs[2].set_xticks(xticks)
            axs[2].set_yticks(yticks)
            axs[2].set_xticklabels(x_label_list)
            axs[2].set_yticklabels(y_label_list)
        else:
            img_flux.set_data(flux[:, t].reshape(161, 181)[::2, ::2])
            img_sst.set_data(sst[:, t].reshape(161, 181)[::2, ::2])
            img_press.set_data(press[:, t].reshape(161, 181)[::2, ::2])

        fig.colorbar(img_flux, cax=cax_flux, orientation='vertical')
        fig.colorbar(img_sst, cax=cax_sst, orientation='vertical')
        fig.colorbar(img_press, cax=cax_press, orientation='vertical')

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'videos/3D/flux-sst-press/hourly/{pic_num:05d}.png')
        pic_num += 1
    return
