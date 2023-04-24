import matplotlib.pyplot as plt
import datetime
import numpy as np
from video import get_continuous_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import matplotlib.dates as mdates
import os
import tqdm
import matplotlib


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
                       group: int = None,
                       flux_type: str = 'sensible',
                       coeff_type: str = 'A'):
    # diff = np.load(files_path_prefix + f'Components/{flux_type}/difference/point_({point[0]}, {point[1]})-{coeff_type}.npy')
    # Bel = np.load(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-{coeff_type}.npy')
    # Kor_sum = np.load(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-{coeff_type}.npy')

    if not group is None:
        Bel = np.load(files_path_prefix + f'Synthetic/Bel/point_({point[0]}, {point[1]})-{coeff_type}.npy')
        Kor_sum = np.load(files_path_prefix + f'Synthetic/coeff_Kor/Components/Sum/group_({point[0]}, {point[1]})-{coeff_type}.npy')
    else:
        diff = np.load(files_path_prefix + f'Synthetic/difference/point_({point[0]}, {point[1]})-{coeff_type}.npy')
        Bel = np.load(files_path_prefix + f'Synthetic/Bel/point_({point[0]}, {point[1]})-{coeff_type}.npy')
        Kor_sum = np.load(files_path_prefix + f'Synthetic/coeff_Kor/Components/Sum/point_({point[0]}, {point[1]})-{coeff_type}.npy')
    # rmse = math.sqrt(sum(diff ** 2))
    # print(rmse)

    x = range(0, min(len(Bel), len(Kor_sum)))
    Bel = Bel[0:len(x)]
    Kor_sum = Kor_sum[0:len(x)]
    a = np.load(f'{files_path_prefix}/Synthetic/A_full.npy')
    real = a[:, point[0], point[1]]
    real = real[:len(x)]
    rmse_Bel = math.sqrt(sum((Bel - real)**2))
    rmse_Kor = math.sqrt(sum((Kor_sum - real) ** 2))

    # date_start = datetime.datetime(2019, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
    # date_end = datetime.datetime(2019, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
    fig, axs = plt.subplots(1, 1, figsize=(20, 10))
    fig.suptitle(f'{coeff_type} coeff {flux_type} at point ({point[0]}, {point[1]}) \n radius = {radius}, '
                 f'window = {window_width // ticks_by_day} days, step={step_ticks} ticks, n_components = {n_components}'
                 # f'\n {date_start.strftime("%Y-%m-%d")} - {date_end.strftime("%Y-%m-%d")}'
                 f'\n RMSE_bel = {rmse_Bel:.2f}'
                 f'\n RMSE_Kor = {rmse_Kor: .2f}', fontsize=20, fontweight='bold')
    axs.xaxis.set_minor_locator(mdates.MonthLocator())
    axs.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs.xaxis.get_major_locator()))

    days = [datetime.datetime(2019, 1, 1) + datetime.timedelta(days=t) for t in range(len(x))]
    axs.plot(days, Bel, c='b', label='Bel')
    axs.plot(days, Kor_sum, c='g', label='Kor')
    axs.plot(days, real, c='r', label='Real')
    # axs.plot(x, diff[:len(x)], c='y', label='difference')
    axs.legend()
    fig.tight_layout()
    # if not os.path.exists(files_path_prefix + f'Components/{flux_type}/plots'):
    #     os.mkdir(files_path_prefix + f'Components/{flux_type}/plots')
    # fig.savefig(files_path_prefix + f'Components/{flux_type}/plots/difference_point_({point[0]}, {point[1]})-{coeff_type}.png')

    fig.savefig(files_path_prefix + f'Synthetic/Plots/Difference/difference_point_({point[0]}, {point[1]})-{coeff_type}.png')
    plt.close(fig)
    return

def plot_synthetic_flux(files_path_prefix: str,
                        dimensions: int,
                        flux: np.array,
                        sensible: np.array,
                        latent: np.array,
                        time_start: int,
                        time_end: int,
                        a_array: np.ndarray = None,
                        b_array: np.ndarray = None,
                        mu: float = None,
                        sigma: float = None):
    if not os.path.exists(files_path_prefix + 'Synthetic/Plots/Flux'):
        os.mkdir(files_path_prefix + 'Synthetic/Plots/Flux')

    if dimensions == 2:
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
    elif dimensions == 1:
        if a_array is None or b_array is None:
            fig, axs = plt.subplots(1, 1, figsize=(10, 10))
            img = None
            flux_max = np.nanmax(flux)
            flux_min = np.nanmin(flux)
            # cmap = plt.get_cmap('Reds')
            cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - flux_min) / (flux_max - flux_min), 1])
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
            fig, axs = plt.subplots(1, 3, figsize=(30, 10))
            img_flux, img_a, img_b = None, None, None
            axs[0].set_title(f'Flux', fontsize=20)
            axs[1].set_title(f'A', fontsize=20)
            axs[2].set_title(f'B', fontsize=20)

            flux_max = np.nanmax(flux)
            flux_min = np.nanmin(flux)
            cmap_flux = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - flux_min) / (flux_max - flux_min), 1])
            cmap_flux.set_bad('darkgreen', 1.0)
            divider = make_axes_locatable(axs[0])
            cax_flux = divider.append_axes('right', size='5%', pad=0.3)

            a_max = np.nanmax(a_array)
            a_min = np.nanmin(a_array)
            # cmap_a = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - a_min) / (a_max - a_min), 1])
            # cmap_a.set_bad('darkgreen', 1.0)
            divider = make_axes_locatable(axs[1])
            cax_a = divider.append_axes('right', size='5%', pad=0.3)

            b_max = np.nanmax(b_array)
            b_min = np.nanmin(b_array)
            # cmap_b = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - b_min) / (b_max - b_min), 1])
            # cmap_b.set_bad('darkgreen', 1.0)
            divider = make_axes_locatable(axs[2])
            cax_b = divider.append_axes('right', size='5%', pad=0.3)

            for t in tqdm.tqdm(range(time_start, time_end)):
                fig.suptitle(f'X \n t = {t}', fontsize=30)
                if img_flux is None:
                    img_flux = axs[0].imshow(flux[t],
                                     interpolation='none',
                                     cmap=cmap_flux,
                                     vmin=flux_min,
                                     vmax=flux_max)
                    img_a = axs[1].imshow(a_array[t],
                                     interpolation='none',
                                     cmap=cmap_flux,
                                     vmin=flux_min,
                                     vmax=flux_max)
                    img_b = axs[2].imshow(b_array[t],
                                     interpolation='none',
                                     cmap=cmap_flux,
                                     vmin=flux_min,
                                     vmax=flux_max)
                else:
                    img_flux.set_data(flux[t])
                    img_a.set_data(a_array[t])
                    img_b.set_data(b_array[t])

                fig.colorbar(img_flux, cax=cax_flux, orientation='vertical')
                fig.colorbar(img_a, cax=cax_a, orientation='vertical')
                fig.colorbar(img_b, cax=cax_b, orientation='vertical')
                fig.savefig(files_path_prefix + f'Synthetic/Plots/Flux/Flux_composite_{t:05d}.png')
    return


def plot_Kor_Bel_compare(files_path_prefix: str,
                         time_start: int,
                         time_end: int,
                         coeff_array: np.array,
                         flux_type: str,
                         coeff_type: str,
                         height: int = 100,
                         width: int = 100,):

    # if not os.path.exists(files_path_prefix + f'Synthetic/Plots/{coeff_type}_{flux_type}'):
    #     os.mkdir(files_path_prefix + f'Synthetic/Plots/{coeff_type}_{flux_type}')

    if not os.path.exists(files_path_prefix + f'Synthetic/Plots/{coeff_type}'):
        os.mkdir(files_path_prefix + f'Synthetic/Plots/{coeff_type}')

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    img = [None for _ in range(4)]

    coeff_max = np.nanmax(coeff_array)
    coeff_min = np.nanmin(coeff_array)

    # coeff_max = 0
    # coeff_min = 1

    cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - coeff_min) / (coeff_max - coeff_min), 1])
    # cmap = plt.get_cmap('Reds')
    cmap.set_bad('darkgreen', 1.0)

    cax = list()
    for i in range(4):
        divider = make_axes_locatable(axs[i // 2][i % 2])
        cax.append(divider.append_axes('right', size='5%', pad=0.3))

    for t in tqdm.tqdm(range(time_start, time_end)):
        # coeff_string = f'{coeff_type}_sens_{t}.npy'
        coeff_string = f'{coeff_type}_{t}.npy'
        if not os.path.exists(files_path_prefix + f'Synthetic/coeff_Bel/{coeff_string}'):
            coeff_Bel = np.zeros((height, width))
        else:
            coeff_Bel = np.load(files_path_prefix + f'Synthetic/coeff_Bel/{coeff_string}')

        if not os.path.exists(files_path_prefix + f'Synthetic/coeff_Kor/{coeff_string}'):
            coeff_Kor = np.zeros((height, width))
        else:
            coeff_Kor = np.load(files_path_prefix + f'Synthetic/coeff_Kor/Components/{coeff_string}')

        # fig.suptitle(f'{coeff_type} {flux_type}\n day {t}', fontsize=30)
        fig.suptitle(f'{coeff_type}\n day {t}', fontsize=30)

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

        # fig.savefig(files_path_prefix + f'Synthetic/Plots/{coeff_type}_{flux_type}/{t}.png')
        fig.savefig(files_path_prefix + f'Synthetic/Plots/{coeff_type}/{t}.png')
    return


def plot_group(files_path_prefix: str,
               group: int,
               a_sum: np.ndarray,
               b_sum: np.ndarray):
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'A coeff evolution in group {group}')

    # plt.xticks(list(range(a_sum.shape[0])))
    plt.xlim(0, 100)
    plt.plot(list(range(a_sum.shape[0])), a_sum, '-o', c='r')
    plt.tight_layout()
    fig.savefig(files_path_prefix + f'Synthetic/coeff_Kor/Components/plots/A_group_{group}.png')
    plt.close(fig)

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'B coeff evolution in group {group}')
    plt.xlim(0, 100)
    plt.plot(list(range(b_sum.shape[0])), b_sum, '-o', c='b')
    plt.tight_layout()
    fig.savefig(files_path_prefix + f'Synthetic/coeff_Kor/Components/plots/B_group_{group}.png')
    plt.close(fig)
    return
