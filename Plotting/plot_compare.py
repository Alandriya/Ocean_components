import datetime
import math
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Plotting.video import get_continuous_cmap


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
    fig.suptitle(
        f"{flux_type} data at ({point[0]}, {point[1]}): {date_start.strftime('%d.%m.%Y')}-{date_end.strftime('%d.%m.%Y')}",
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
                                   width: int = 100, ):
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
    fig.suptitle(
        f"{coeff_type} coeff - {flux_type} difference\n {date_start.strftime('%d.%m.%Y')}-{date_end.strftime('%d.%m.%Y')}",
        fontsize=26, fontweight='bold')

    cmap = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
    cmap.set_bad('darkgreen', 1.0)
    axs.imshow(mask_map, interpolation='none', cmap=cmap)
    fig.savefig(files_path_prefix + f'Components/{flux_type}/{coeff_type}-difference_points.png')
    plt.close(fig)
    return


def plot_difference_1d_synthetic(files_path_prefix: str,
                                 point: tuple,
                                 n_components: int = 3,
                                 time_start: int = 0,
                                 time_end: int = 98,
                                 coeff_type: str = 'A'):
    """
    Plots 1d difference in point for synthetic data coefficients estimates and real coefficient for time steps in
    (time_start, time_end) for both Korolev and Belyaev methods for coeff_type
    :param files_path_prefix: path to the working directory
    :param point: tuple with point coordinates
    :param n_components: amount of components for EM
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param coeff_type: 'A' or 'B' or 'F'
    :return:
    """
    if not os.path.exists(files_path_prefix + f'Synthetic/Plots/Difference'):
        os.mkdir(files_path_prefix + f'Synthetic/Plots/Difference')

    Bel = np.load(files_path_prefix + f'Synthetic/Bel/points/point_({point[0]}, {point[1]})-{coeff_type}.npy')[
          time_start:time_end]
    Kor = np.load(files_path_prefix + f'Synthetic/Kor/points/point_({point[0]}, {point[1]})-{coeff_type}.npy')[
          time_start:time_end]
    Kor = np.nan_to_num(Kor)

    if coeff_type == 'A':
        a = np.load(f'{files_path_prefix}/Synthetic/A_full.npy')
        real = a[time_start:time_end, point[0], point[1]]
    else:
        b = np.load(f'{files_path_prefix}/Synthetic/B_full.npy')
        real = b[time_start:time_end, point[0], point[1]]
        real = np.sqrt(np.abs(real))

    rmse_Bel = math.sqrt(sum((Bel - real) ** 2))
    rmse_Kor = math.sqrt(sum((Kor - real) ** 2))

    fig, axs = plt.subplots(1, 1, figsize=(20, 10))
    # fig.suptitle(f'{coeff_type} coeff at point ({point[0]}, {point[1]}), n_components = {n_components}'
    #              f'\n RMSE Pointwise = {rmse_Bel:.2f}'
    #              f'\n RMSE ML = {rmse_Kor: .2f}',
    #              fontsize=20, fontweight='bold')
    print(f' {coeff_type} point ({point[0]}, {point[1]}) \n RMSE Pointwise = {rmse_Bel:.2f} \n RMSE ML = {rmse_Kor: .2f} \n')
    fig.suptitle(f'{str.lower(coeff_type)}[{point[0]}, {point[1]}]', fontsize=30, fontweight='bold')

    plt.xlabel('Time', fontsize=26)
    plt.ylabel('Value', fontsize=26)
    axs.plot(Bel, c='b', label='NP')
    axs.plot(Kor, c='g', label='P')
    axs.plot(real, linestyle=(0, (5, 5)), c='r', label='Real')
    axs.legend(fontsize="26")
    fig.tight_layout()

    fig.savefig(
        files_path_prefix + f'Synthetic/Plots/Difference/{coeff_type}_difference_point_({point[0]}, {point[1]})-{coeff_type}.png')
    plt.close(fig)
    return


def plot_difference_1d(files_path_prefix: str,
                       point: tuple,
                       n_components: int = 3,
                       time_start: int = 0,
                       time_end: int = 98,
                       coeff_type: str = 'A',
                       flux_type: str = None,
                       start_index: int = 0):
    """
    Plots 1d difference in point for real data coefficients estimates for time steps in (time_start, time_end) with
    offset start_index for both Korolev and Belyaev methods for coeff_type
    :param files_path_prefix: path to the working directory
    :param point: tuple with point coordinates
    :param n_components: amount of components for EM
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param coeff_type: 'A' or 'B' or 'F'
    :param flux_type: 'sensible' or 'latent'
    :param start_index: offset index when saving plots
    :return:
    """
    Bel = np.load(files_path_prefix + f'Synthetic/Bel/point_({point[0]}, {point[1]})-{coeff_type}.npy')[
          time_start:time_end]
    Kor = np.load(files_path_prefix + f'Components/{flux_type}/{coeff_type}_map.npy')[time_start:time_end, point[0],
          point[1]]
    Kor = np.nan_to_num(Kor)

    date_start = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_start + start_index)
    date_end = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_end + start_index)
    fig, axs = plt.subplots(1, 1, figsize=(20, 10))
    fig.suptitle(f'{coeff_type} coeff at point ({point[0]}, {point[1]}), n_components = {n_components}'
                 f'\n {date_start.strftime("%Y-%m-%d")} - {date_end.strftime("%Y-%m-%d")}')
    axs.xaxis.set_minor_locator(mdates.MonthLocator())
    axs.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs.xaxis.get_major_locator()))

    days = [date_start + datetime.timedelta(days=t) for t in range(time_end - time_start)]
    axs.plot(days, Bel, c='b', label='Pointwise')
    axs.plot(days, Kor, c='g', label='EM')
    axs.legend()
    fig.tight_layout()
    if not os.path.exists(files_path_prefix + f'Components/{flux_type}/plots'):
        os.mkdir(files_path_prefix + f'Components/{flux_type}/plots')
    fig.savefig(
        files_path_prefix + f'Components/{flux_type}/plots/difference_point_({point[0]}, {point[1]})-{coeff_type}.png')
    plt.close(fig)
    return


def plot_synthetic_flux(files_path_prefix: str,
                        flux: np.array,
                        time_start: int,
                        time_end: int,
                        a_array: np.ndarray = None,
                        b_array: np.ndarray = None):
    """
    Plots 3 maps for synthetic flux X, A and B coefficients for erery time step in (time_start, time_end)
    :param files_path_prefix: path to the working directory
    :param flux: np.array with generated flux data with shape [time_steps, height, width]
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param a_array: np.array with generated A coefficient data with shape [time_steps, height, width]
    :param b_array: np.array with generated B coefficient data with shape [time_steps, height, width]
    :return:
    """
    if not os.path.exists(files_path_prefix + 'Synthetic/Plots'):
        os.mkdir(files_path_prefix + 'Synthetic/Plots')
    if not os.path.exists(files_path_prefix + 'Synthetic/Plots/Flux'):
        os.mkdir(files_path_prefix + 'Synthetic/Plots/Flux')

    if a_array is None or b_array is None:
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        img = None
        flux_max = np.nanmax(flux)
        flux_min = np.nanmin(flux)
        # cmap = plt.get_cmap('Reds')
        cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                                   [0, (1.0 - flux_min) / (flux_max - flux_min), 1])
        cmap.set_bad('darkgreen', 1.0)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes('right', size='5%', pad=0.3)

        for t in tqdm.tqdm(range(time_start, time_end)):
            fig.suptitle(f'X \n t = {t}', fontsize=30)
            if img is None:
                img = axs.imshow(flux[t],
                                 interpolation='none',
                                 cmap=cmap,
                                 vmin=flux_min,
                                 vmax=flux_max)
            else:
                img.set_data(flux[t])

            fig.colorbar(img, cax=cax, orientation='vertical')
            fig.savefig(files_path_prefix + f'Synthetic/Plots/Flux/Flux_{t:05d}.png')
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        img_flux, img_a, img_b = None, None, None
        axs[0].set_title(f'X', fontsize=20)
        axs[1].set_title(f'a', fontsize=20)
        axs[2].set_title(f'b', fontsize=20)

        flux_max = np.nanmax(flux)
        flux_min = np.nanmin(flux)
        cmap_flux = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                                        [0, (1.0 - flux_min) / (flux_max - flux_min), 1])
        cmap_flux.set_bad('darkgreen', 1.0)
        divider = make_axes_locatable(axs[0])
        cax_flux = divider.append_axes('right', size='5%', pad=0.3)

        a_max = np.nanmax(a_array)
        a_min = np.nanmin(a_array)
        cmap_a = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - a_min) / (a_max - a_min), 1])
        cmap_a.set_bad('darkgreen', 1.0)
        divider = make_axes_locatable(axs[1])
        cax_a = divider.append_axes('right', size='5%', pad=0.3)

        b_max = np.nanmax(b_array)
        b_min = np.nanmin(b_array)
        cmap_b = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - b_min) / (b_max - b_min), 1])
        cmap_b.set_bad('darkgreen', 1.0)
        divider = make_axes_locatable(axs[2])
        cax_b = divider.append_axes('right', size='5%', pad=0.3)

        for t in tqdm.tqdm(range(time_start, time_end)):
            # fig.suptitle(f'Synthetic data, t = {t}', fontsize=30)
            fig.suptitle(f't = {t}', fontsize=30)
            if img_flux is None:
                img_flux = axs[0].imshow(flux[t],
                                         interpolation='none',
                                         cmap=cmap_flux)
                img_a = axs[1].imshow(a_array[t],
                                      interpolation='none',
                                      cmap=cmap_flux)
                img_b = axs[2].imshow(b_array[t],
                                      interpolation='none',
                                      cmap=cmap_flux)
            else:
                img_flux.set_data(flux[t])
                img_a.set_data(a_array[t])
                img_b.set_data(b_array[t])

            fig.colorbar(img_flux, cax=cax_flux, orientation='vertical')
            fig.colorbar(img_a, cax=cax_a, orientation='vertical')
            fig.colorbar(img_b, cax=cax_b, orientation='vertical')
            fig.tight_layout()
            fig.savefig(files_path_prefix + f'Synthetic/Plots/Flux/Flux_composite_{t:05d}.png')
    return


# def plot_methods_compare_synthetic(files_path_prefix: str,
#                          time_start: int,
#                          time_end: int,
#                          coeff_array: np.array,
#                          coeff_type: str,):
#     if not os.path.exists(files_path_prefix + f'Synthetic/Plots/{coeff_type}'):
#         os.mkdir(files_path_prefix + f'Synthetic/Plots/{coeff_type}')
#
#     fig, axs = plt.subplots(2, 2, figsize=(20, 20))
#     img = [None for _ in range(4)]
#
#     coeff_max = np.nanmax(coeff_array)
#     coeff_min = np.nanmin(coeff_array)
#
#
#     cax = list()
#     for i in range(4):
#         divider = make_axes_locatable(axs[i // 2][i % 2])
#         cax.append(divider.append_axes('right', size='5%', pad=0.3))
#
#     if not os.path.exists(files_path_prefix + f'Synthetic/coeff_Kor/Components/{coeff_type}_map.npy'):
#         raise FileNotFoundError('No Korolev map!')
#     else:
#
#         coeff_Kor = np.load(files_path_prefix + f'Synthetic/coeff_Kor/Components/{coeff_type}_map.npy')
#         height, width = coeff_Kor.shape[1], coeff_Kor.shape[2]
#
#     coeff_max = np.nanmax((coeff_max, np.nanmax(coeff_Kor)))
#     coeff_min = np.nanmin((coeff_min, np.nanmin(coeff_Kor)))
#
#     for t in tqdm.tqdm(range(time_start + 1, time_end)):
#         coeff_string = f'{coeff_type}_{t}.npy'
#         if not os.path.exists(files_path_prefix + f'Synthetic/coeff_Bel/{coeff_string}'):
#             raise FileNotFoundError(f'No Belyaev map step {t}!')
#         else:
#             coeff_Bel = np.load(files_path_prefix + f'Synthetic/coeff_Bel/{coeff_string}')
#
#         fig.suptitle(f'{coeff_type}\n day {t}', fontsize=30)
#
#         diff_Bel = coeff_Bel - coeff_array[t]
#         rmse_Bel = math.sqrt(sum(diff_Bel.flatten() ** 2))
#
#         coeff_Kor[t] = np.nan_to_num(coeff_Kor[t])
#         diff_Kor = coeff_Kor[t] - coeff_array[t]
#         rmse_Kor = math.sqrt(sum(diff_Kor.flatten() ** 2))
#
#         coeff_max = np.nanmax((coeff_max, np.nanmax(coeff_Bel)))
#         coeff_min = np.nanmin((coeff_min, np.nanmin(coeff_Bel)))
#
#         cmap = plt.get_cmap('Reds')
#
#         for i in range(4):
#             if i == 0:
#                 axs[i // 2][i % 2].set_title(f'Pointwise, rmse = {rmse_Bel:.1f}', fontsize=20)
#             elif i == 1 or i == 3:
#                 axs[i // 2][i % 2].set_title(f'Real', fontsize=20)
#             elif i == 2:
#                 axs[i // 2][i % 2].set_title(f'EM, rmse = {rmse_Kor:.1f}', fontsize=20)
#
#         for i in [1, 3]:
#             if img[i] is None:
#                 img[i] = axs[i // 2][i % 2].imshow(coeff_array[t],
#                                                    interpolation='none',
#                                                    cmap=cmap)
#             else:
#                 img[i].set_data(coeff_array[t])
#
#         if img[0] is None:
#             img[0] = axs[0][0].imshow(coeff_Bel,
#                                       interpolation='none',
#                                       cmap=cmap,
#                                       vmin=coeff_min,
#                                       vmax=coeff_max)
#         else:
#             img[0].set_data(coeff_Bel)
#
#         if img[2] is None:
#             img[2] = axs[1][0].imshow(coeff_Kor[t],
#                                       interpolation='none',
#                                       cmap=cmap,
#                                       vmin=coeff_min,
#                                       vmax=coeff_max)
#         else:
#             img[2].set_data(coeff_Kor[t])
#
#         for i in range(4):
#             fig.colorbar(img[i], cax=cax[i], orientation='vertical')
#
#         fig.savefig(files_path_prefix + f'Synthetic/Plots/{coeff_type}/{t}.png')


def plot_methods_compare(files_path_prefix: str,
                         coeff_Bel: np.ndarray,
                         coeff_Kor: np.ndarray,
                         flux_type: str,
                         coeff_type: str,
                         start_index: int = 0):
    """
    Plots maps for real data coefficients estimates with offset start_index
    for both Korolev and Belyaev methods for coeff_type
    :param files_path_prefix: path to the working directory
    :param coeff_Bel: calculated map-type coefficient
    :param coeff_Kor: calculated map-type coefficient
    :param flux_type: 'sensible' or 'latent' or 'pressure'
    :param coeff_type: 'A' or 'B' or 'F'
    :param start_index: offset index when saving plots
    :return:
    """

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    img = [None for _ in range(2)]

    cax = list()
    for i in range(2):
        divider = make_axes_locatable(axs[i])
        cax.append(divider.append_axes('right', size='5%', pad=0.3))

    coeff_max = max(np.nanmax(coeff_Kor), np.nanmax(coeff_Kor))
    coeff_min = min(np.nanmin(coeff_Kor), np.nanmin(coeff_Bel))

    # fig.suptitle(f'{coeff_type} {flux_type}\n day {start_index}', fontsize=30)

    # coeff_Bel[coeff_Bel == 0] = np.nan
    if coeff_type == 'B':
        cmap = get_continuous_cmap(['#ffffff', '#ff0000'],
                                   [0, 1])
    else:
        cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                                   [0, (1.0 - coeff_min) / (coeff_max - coeff_min), 1])

    cmap.set_bad('darkgreen', 1.0)

    img[0] = axs[0].imshow(coeff_Bel,
                           interpolation='none',
                           cmap=cmap,
                           vmin=coeff_min,
                           vmax=coeff_max)


    img[1] = axs[1].imshow(coeff_Kor,
                           interpolation='none',
                           cmap=cmap,
                           vmin=coeff_min,
                           vmax=coeff_max)

    axs[0].set_title(f'Непараметрический метод', fontsize=16)
    axs[1].set_title(f'Полупараметрический метод', fontsize=16)
    for i in range(2):
        fig.colorbar(img[i], cax=cax[i], orientation='vertical')
    fig.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Compare/{coeff_type}_{flux_type}_{start_index}.png')
    return


def plot_synthetic_difference_compare(files_path_prefix: str,
                                      time_start: int,
                                      time_end: int,
                                      coeff_array: np.array,
                                      coeff_type: str):
    """
    Plots 3 maps: 2 for absolute difference with real synthetic data for coefficients estimates for time steps in
    (time_start, time_end) for both Korolev and Belyaev methods for coeff_type and 1 for the absolute difference between
    the two methods
    :param files_path_prefix: path to the working directory
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param coeff_array: array with real coeff data
    :param coeff_type: 'A' or 'B' or 'F'
    :return:
    """
    if not os.path.exists(files_path_prefix + f'Synthetic/Plots/{coeff_type}'):
        os.mkdir(files_path_prefix + f'Synthetic/Plots/{coeff_type}')

    height, width = coeff_array.shape[1], coeff_array.shape[2]

    coeff_max = np.nanmax(coeff_array)
    coeff_min = np.nanmin(coeff_array)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    img = [None for _ in range(3)]

    cax = list()
    for i in range(3):
        divider = make_axes_locatable(axs[i])
        cax.append(divider.append_axes('right', size='5%', pad=0.3))

    if not os.path.exists(files_path_prefix + f'Synthetic/coeff_Kor/Components/{coeff_type}_map.npy'):
        coeff_Kor = np.zeros((time_end - time_start - 1, height, width))
    else:
        coeff_Kor = np.load(files_path_prefix + f'Synthetic/coeff_Kor/Components/{coeff_type}_map.npy')

    coeff_max = np.nanmax((coeff_max, np.nanmax(coeff_Kor)))
    coeff_min = np.nanmin((coeff_min, np.nanmin(coeff_Kor)))
    for t in tqdm.tqdm(range(time_start + 1, time_end)):
        coeff_string = f'{coeff_type}_{t}.npy'
        if not os.path.exists(files_path_prefix + f'Synthetic/coeff_Bel/{coeff_string}'):
            coeff_Bel = np.zeros((height, width))
        else:
            coeff_Bel = np.load(files_path_prefix + f'Synthetic/coeff_Bel/{coeff_string}')

        fig.suptitle(f'{coeff_type} day {t}', fontsize=30)

        diff_Bel = coeff_Bel - coeff_array[t]
        rmse_Bel = math.sqrt(sum(diff_Bel.flatten() ** 2))

        coeff_Kor[t] = np.nan_to_num(coeff_Kor[t])
        diff_Kor = coeff_Kor[t] - coeff_array[t]
        rmse_Kor = math.sqrt(sum(diff_Kor.flatten() ** 2))

        difference = coeff_Kor[t] - coeff_Bel
        rmse_diff = math.sqrt(sum(difference.flatten() ** 2))

        coeff_max = np.nanmax((coeff_max, np.nanmax(coeff_Bel)))
        coeff_min = np.nanmin((coeff_min, np.nanmin(coeff_Bel)))

        cmap = plt.get_cmap('Reds')
        axs[0].set_title(f'|Real - Pointwise|, rmse_pointwise = {rmse_Bel:.1f}', fontsize=15)
        axs[1].set_title(f'|Real - EM|, rmse_EM = {rmse_Kor:.1f}', fontsize=15)
        axs[2].set_title(f'|Pointwise - EM|, rmse_difference= {rmse_diff:.1f}', fontsize=15)

        if img[0] is None:
            img[0] = axs[0].imshow(np.abs(coeff_array[t] - coeff_Bel),
                                   interpolation='none',
                                   cmap=cmap)
        else:
            img[0].set_data(np.abs(coeff_array[t] - coeff_Bel))

        if img[1] is None:
            img[1] = axs[1].imshow(np.abs(coeff_array[t] - coeff_Kor[t]),
                                   interpolation='none',
                                   cmap=cmap)
        else:
            img[1].set_data(np.abs(coeff_array[t] - coeff_Kor[t]))

        if img[2] is None:
            img[2] = axs[2].imshow(np.abs(coeff_Bel - coeff_Kor[t]),
                                   interpolation='none',
                                   cmap=cmap)
        else:
            img[2].set_data(np.abs(coeff_Bel - coeff_Kor[t]))

        for i in range(3):
            fig.colorbar(img[i], cax=cax[i], orientation='vertical')

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'Synthetic/Plots/{coeff_type}/difference_{t}.png')
    return

def plot_quantiles_amount_compare(files_path_prefix:str,
                                  coeff_type: str,
                                  quantiles: np.array,
                                  list_types: list):
    if not os.path.exists(files_path_prefix + f'Synthetic/Plots'):
        os.mkdir(files_path_prefix + f'Synthetic/Plots')
    if not os.path.exists(files_path_prefix + f'Synthetic/Plots/Quantiles'):
        os.mkdir(files_path_prefix + f'Synthetic/Plots/Quantiles')
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    # print(f' {coeff_type} RMSE\n')
    # fig.suptitle(f'{coeff_type} RMSE', fontsize=20, fontweight='bold')

    # plt.xlabel('Quantiles amount', fontsize=14)
    plt.xlabel('Количество квантилей', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    Bel = np.load(files_path_prefix + 'Synthetic/' + f'rmse_{coeff_type}_bel.npy')
    Kor = np.load(files_path_prefix + 'Synthetic/' + f'rmse_{coeff_type}_kor.npy')
    if 'Bel' in list_types:
        axs.plot(quantiles, Bel, '-o', c='b', label='NP')
    if 'Kor' in list_types:
        axs.plot(quantiles, Kor, '-o', c='r', label='SP')
    x_label_list = quantiles
    xticks = quantiles
    axs.set_xticks(xticks)
    axs.set_xticklabels(x_label_list)
    axs.legend(fontsize="16")
    fig.tight_layout()

    fig.savefig(
        files_path_prefix + f'Synthetic/Plots/Quantiles/{coeff_type}_{list_types}.png')
    plt.close(fig)
    return
