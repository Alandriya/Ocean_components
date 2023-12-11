import numpy as np
import datetime
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Plotting.video import get_continuous_cmap
import tqdm

width = 181
height = 161


def count_mean_year(files_path_prefix: str,
                    start_year: int = 2009,
                    end_year: int = 2019,
                    coeff_type: str = 'A',
                    flux_type: str = 'sensible',
                    method: str = 'Kor',
                    mask: np.ndarray = None,
                    ):
    """
    Counts mean year as np.array with shape (365, height, width) for 365 days with mean values for each day
    :param files_path_prefix: path to the working directory
    :param start_year:
    :param end_year:
    :param coeff_type: A/B/C/F/FS
    :param flux_type: sensible/latent/flux/sst/press
    :param method: 'Bel' or 'Kor'
    :param mask: np.array with shape (height, width) with boolean values, where 0 is for land and 1 is for ocean
    :return:
    """

    mean_year = np.zeros((365, height, width))

    for year in tqdm.tqdm(range(start_year, end_year)):
        time_start = (datetime.datetime(year=year, month=1, day=1) - datetime.datetime(year=1979, month=1, day=1)).days

        for day in range(364):
            if method == 'Kor':
                coeff = np.load(files_path_prefix + f'Components/{flux_type}/{method}/daily/' + f'{coeff_type}_{day + time_start + 1}.npy')
            else:
                if coeff_type == 'A' and flux_type == 'sensible':
                    postfix = '_sens'
                    coeff = np.load(files_path_prefix + f'Coeff_data/{day + time_start + 1}_{coeff_type}{postfix}.npy')
                elif coeff_type == 'A' and flux_type == 'latent':
                    postfix = '_lat'
                    coeff = np.load(files_path_prefix + f'Coeff_data/{day + time_start + 1}_{coeff_type}{postfix}.npy')
                elif coeff_type == 'B' and flux_type == 'sensible':
                    coeff = np.load(files_path_prefix + f'Coeff_data/{day + time_start + 1}_{coeff_type}.npy')[0]
                elif coeff_type == 'B' and flux_type == 'latent':
                    coeff = np.load(files_path_prefix + f'Coeff_data/{day + time_start + 1}_{coeff_type}.npy')[3]
                elif coeff_type == 'A' and flux_type == 'flux':
                    coeff = np.load(files_path_prefix + f'Coeff_data_3d/flux-press/{day + time_start + 1}_{coeff_type}_sens.npy')
                elif coeff_type == 'B' and flux_type == 'flux':
                    coeff = np.load(files_path_prefix + f'Coeff_data_3d/flux-press/{day + time_start + 1}_{coeff_type}.npy')[0]
                elif coeff_type == 'A' and flux_type == 'press':
                    coeff = np.load(files_path_prefix + f'Coeff_data_3d/flux-press/{day + time_start + 1}_{coeff_type}_lat.npy')
                elif coeff_type == 'B' and flux_type == 'press':
                    coeff = np.load(files_path_prefix + f'Coeff_data_3d/flux-press/{day + time_start + 1}_{coeff_type}.npy')[3]
                elif coeff_type == 'A' and flux_type == 'sst':
                    coeff = np.load(files_path_prefix + f'Coeff_data_3d/flux-sst/{day + time_start + 1}_{coeff_type}_lat.npy')
                elif coeff_type == 'B' and flux_type == 'sst':
                    coeff = np.load(files_path_prefix + f'Coeff_data_3d/flux-sst/{day + time_start + 1}_{coeff_type}.npy')[3]

            mean_year[day, :, :] += coeff
            mean_year[day][np.logical_not(mask)] = None

    mean_year /= (end_year - start_year)
    np.save(files_path_prefix + f'Mean_year/{method}/{flux_type}_{coeff_type}_{start_year}-{end_year}.npy', mean_year)
    return


def plot_mean_year_2d(files_path_prefix: str, coeff_name: str):
    """
    Plots 2x3 graphics of "mean year" of coefficient coeff_name
    :param files_path_prefix: path to the working directory
    :param coeff_name: 'A_sens' or 'A_lat' or 'B11' or 'B22' or 'F'
    :return:
    """
    mean_year = np.load(files_path_prefix + f'Mean_year/{coeff_name}.npy')
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    # fig.suptitle(f'{coeff_name} mean year', fontsize=30)
    axs[0][0].title.set_text('February, 15')
    axs[0][1].title.set_text('April, 15')
    axs[0][2].title.set_text('June, 15')
    axs[1][0].title.set_text('August, 15')
    axs[1][1].title.set_text('October, 15')
    axs[1][2].title.set_text('December, 15')
    img = [None for _ in range(6)]
    cax = [None for _ in range(6)]
    days = [(datetime.datetime(1979, i*2, 15) - datetime.datetime(1979, 1, 2)).days for i in range(1, 7)]

    x_label_list = ['90W', '60W', '30W', '0']
    y_label_list = ['EQ', '30N', '60N', '80N']
    xticks = [0, 60, 120, 180]
    yticks = [160, 100, 40, 0]

    if coeff_name == 'A_sens' or coeff_name == 'A_lat':
        a_min = np.nanmin(mean_year)
        a_max = np.nanmax(mean_year)
        cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - a_min) / (a_max - a_min), 1])
        cmap.set_bad('darkgreen', 1.0)
        for i in range(6):
            divider = make_axes_locatable(axs[i // 3][i % 3])
            cax[i] = divider.append_axes('right', size='5%', pad=0.3)
            img[i] = axs[i // 3][i % 3].imshow(mean_year[days[i]],
                                               interpolation='none',
                                               cmap=cmap,
                                               vmin=a_min,
                                               vmax=a_max)

            axs[i // 3][i % 3].set_xticks(xticks)
            axs[i // 3][i % 3].set_yticks(yticks)
            axs[i // 3][i % 3].set_xticklabels(x_label_list)
            axs[i // 3][i % 3].set_yticklabels(y_label_list)
            fig.colorbar(img[i], cax=cax[i], orientation='vertical')

    elif coeff_name == 'B11' or coeff_name == 'B22':
        b_min = np.nanmin(mean_year)
        b_max = np.nanmax(mean_year)
        zero_percent = abs(0 - b_min) / (b_max - b_min)
        cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, zero_percent, 1])
        cmap.set_bad('darkgreen', 1.0)
        for i in range(6):
            divider = make_axes_locatable(axs[i // 3][i % 3])
            cax[i] = divider.append_axes('right', size='5%', pad=0.3)
            img[i] = axs[i // 3][i % 3].imshow(mean_year[days[i]],
                                               interpolation='none',
                                               cmap=cmap,
                                               vmin=b_min,
                                               vmax=b_max)
            axs[i // 3][i % 3].set_xticks(xticks)
            axs[i // 3][i % 3].set_yticks(yticks)
            axs[i // 3][i % 3].set_xticklabels(x_label_list)
            axs[i // 3][i % 3].set_yticklabels(y_label_list)
            fig.colorbar(img[i], cax=cax[i], orientation='vertical')
    elif coeff_name == 'F':
        f_max = np.nanmax(mean_year)
        cmap = colors.ListedColormap(['MintCream', 'Aquamarine', 'blue', 'red', 'DarkRed'])
        boundaries = [0, 0.25, 0.5, 1.0, 1.5, max(f_max, 2.0)]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        cmap.set_bad('darkgreen', 1.0)
        for i in range(6):
            divider = make_axes_locatable(axs[i // 3][i % 3])
            cax[i] = divider.append_axes('right', size='5%', pad=0.3)
            img[i] = axs[i // 3][i % 3].imshow(mean_year[days[i]],
                                               interpolation='none',
                                               cmap=cmap,
                                               norm=norm)
            axs[i // 3][i % 3].set_xticks(xticks)
            axs[i // 3][i % 3].set_yticks(yticks)
            axs[i // 3][i % 3].set_xticklabels(x_label_list)
            axs[i // 3][i % 3].set_yticklabels(y_label_list)
            fig.colorbar(img[i], cax=cax[i], orientation='vertical')
    elif coeff_name == 'C_0' or coeff_name == 'C_1':
        if coeff_name == 'C_0':
            fig.suptitle(f'A_sens - A_lat correlation mean year', fontsize=30)
        else:
            fig.suptitle(f'B11 - B22 correlation mean year', fontsize=30)
        cmap = get_continuous_cmap(['#4073ff', '#ffffff', '#ffffff', '#db4035'], [0, 0.4, 0.6, 1])
        cmap.set_bad('darkgreen', 1.0)
        for i in range(6):
            divider = make_axes_locatable(axs[i // 3][i % 3])
            cax[i] = divider.append_axes('right', size='5%', pad=0.3)
            img[i] = axs[i // 3][i % 3].imshow(mean_year[days[i]],
                                               interpolation='none',
                                               cmap=cmap,
                                               vmin=-1,
                                               vmax=1)
            axs[i // 3][i % 3].set_xticks(xticks)
            axs[i // 3][i % 3].set_yticks(yticks)
            axs[i // 3][i % 3].set_xticklabels(x_label_list)
            axs[i // 3][i % 3].set_yticklabels(y_label_list)
            fig.colorbar(img[i], cax=cax[i], orientation='vertical')

    plt.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Mean_year/{coeff_name}_mean_year.png')
    return


def plot_mean_year_1d(files_path_prefix: str,
                      mean_year: np.ndarray,
                      start_year: int = 2009,
                      end_year: int = 2019,
                      coeff_type: str = 'A',
                      flux_type: str = 'sensible',
                      method: str = 'Kor',
                      language: str = 'eng',
                      ):
    """
    Draws 2x3 plots with maps for 15.02, 15.04, 15.06, 15.10 and 15.12 from mean_year array (it is assumed that year is
    365 days long, the dates are shifts from the 1st day of the year, not accurate dates)
    :param files_path_prefix: path to the working directory
    :param mean_year: np.array with calculated mean year
    :param start_year:
    :param end_year:
    :param coeff_type: A/B/C/F/FS
    :param flux_type: sensible/latent/flux/sst/press
    :param method: 'Bel' or 'Kor'
    :param language: 'rus' for russian titles of the pictures
    :return:
    """

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    if language == 'rus':
        axs[0][0].title.set_text('15 февраля')
        axs[0][1].title.set_text('15 апреля')
        axs[0][2].title.set_text('15 июня')
        axs[1][0].title.set_text('15 августа')
        axs[1][1].title.set_text('15 октября')
        axs[1][2].title.set_text('15 декабря')
    else:
        fig.suptitle(f'{coeff_type} {flux_type} mean year', fontsize=30)
        axs[0][0].title.set_text('February, 15')
        axs[0][1].title.set_text('April, 15')
        axs[0][2].title.set_text('June, 15')
        axs[1][0].title.set_text('August, 15')
        axs[1][1].title.set_text('October, 15')
        axs[1][2].title.set_text('December, 15')

    img = [None for _ in range(6)]
    cax = [None for _ in range(6)]
    days = [(datetime.datetime(start_year, i * 2, 15) - datetime.datetime(start_year, 1, 2)).days for i in range(1, 7)]

    if flux_type == 'flux':
        mean_year /= (np.nanmax(mean_year) - np.nanmin(mean_year))
        mean_year *= 2460.551086
    elif flux_type == 'press':
        mean_year /= (np.nanmax(mean_year) - np.nanmin(mean_year))
        mean_year *= 17003.53672
    elif flux_type == 'sst':
        mean_year /= (np.nanmax(mean_year) - np.nanmin(mean_year))
        mean_year *= 38.6043457

    coeff_min = np.nanmin(mean_year)
    coeff_max = np.nanmax(mean_year)

    # for more contrast in maps
    if coeff_type == 'A' and flux_type in ['flux', 'press']:
        coeff_min /= 2
        coeff_max /= 2
    elif coeff_type == 'A' and flux_type == 'sst':
        coeff_min /= 5
        coeff_max /= 5
    elif coeff_type == 'B' and flux_type in ['flux', 'press', 'sst']:
        coeff_max /= 2

    cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - coeff_min) / (coeff_max - coeff_min), 1])
    cmap.set_bad('darkgreen', 1.0)

    for i in range(6):
        divider = make_axes_locatable(axs[i // 3][i % 3])
        cax[i] = divider.append_axes('right', size='5%', pad=0.3)
        img[i] = axs[i // 3][i % 3].imshow(mean_year[days[i]],
                                           interpolation='none',
                                           cmap=cmap,
                                           vmin=coeff_min,
                                           vmax=coeff_max)

        x_label_list = ['90W', '60W', '30W', '0']
        y_label_list = ['EQ', '30N', '60N', '80N']
        xticks = [0, 60, 120, 180]
        yticks = [160, 100, 40, 0]

        axs[i // 3][i % 3].set_xticks(xticks)
        axs[i // 3][i % 3].set_yticks(yticks)
        axs[i // 3][i % 3].set_xticklabels(x_label_list)
        axs[i // 3][i % 3].set_yticklabels(y_label_list)
        fig.colorbar(img[i], cax=cax[i], orientation='vertical')

    plt.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Mean_year/{method}/{flux_type}_{start_year}-{end_year}_{coeff_type}.png')
    return


def plot_mean_year_1d_difference(files_path_prefix: str,
                                 mean_year: np.ndarray,
                                 start_year: int = 2009,
                                 end_year: int = 2019,
                                 coeff_type: str = 'A',
                                 flux_type: str = 'sensible',
                                 ):
    """
    Draws 2x3 plots with maps for 15.02, 15.04, 15.06, 15.10 and 15.12 from mean_year array, which is assumed to be an
    absolute difference between two other mean years arrays
    :param files_path_prefix: path to the working directory
    :param mean_year: np.array with calculated mean year
    :param start_year:
    :param end_year:
    :param coeff_type: A/B/C/F/FS
    :param flux_type: sensible/latent/flux/sst/press
    :return:
    """

    coeff_min = np.nanmin(mean_year)
    coeff_max = np.nanmax(mean_year)

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f'{coeff_type} absolute differences {flux_type} mean year', fontsize=30)
    axs[0][0].title.set_text('February, 15')
    axs[0][1].title.set_text('April, 15')
    axs[0][2].title.set_text('June, 15')
    axs[1][0].title.set_text('August, 15')
    axs[1][1].title.set_text('October, 15')
    axs[1][2].title.set_text('December, 15')
    img = [None for _ in range(6)]
    cax = [None for _ in range(6)]
    days = [(datetime.datetime(start_year, i * 2, 15) - datetime.datetime(start_year, 1, 2)).days for i in range(1, 7)]

    cmap = plt.get_cmap('Reds')
    cmap.set_bad('darkgreen', 1.0)
    for i in range(6):
        divider = make_axes_locatable(axs[i // 3][i % 3])
        cax[i] = divider.append_axes('right', size='5%', pad=0.3)
        img[i] = axs[i // 3][i % 3].imshow(mean_year[days[i]],
                                           interpolation='none',
                                           cmap=cmap,
                                           vmin=coeff_min,
                                           vmax=coeff_max)

        x_label_list = ['90W', '60W', '30W', '0']
        y_label_list = ['EQ', '30N', '60N', '80N']
        xticks = [0, 60, 120, 180]
        yticks = [160, 100, 40, 0]

        axs[i // 3][i % 3].set_xticks(xticks)
        axs[i // 3][i % 3].set_yticks(yticks)
        axs[i // 3][i % 3].set_xticklabels(x_label_list)
        axs[i // 3][i % 3].set_yticklabels(y_label_list)
        fig.colorbar(img[i], cax=cax[i], orientation='vertical')

    plt.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Mean_year/difference/{coeff_type}_{flux_type}_{start_year}-{end_year}_.png')
