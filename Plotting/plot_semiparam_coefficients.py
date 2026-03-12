import datetime
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Plotting.video import get_continuous_cmap


def plot_map_semi(files_path_prefix: str,
                  time_start: int,
                  time_end: int,
                  flux_type: str,
                  coeff_type: str,
                  radius: int,
                  start_pic_num: int = 0):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    img_values, img_difference = None, None

    data_array = np.load(files_path_prefix + f'Components/{flux_type}/{coeff_type}_{time_start}-{time_end}_Sum.npy')
    # diff_array = np.load(files_path_prefix + f'Components/{flux_type}/{coeff_type}_{time_start}-{time_end}_difference.npy')
    # data_array *= 2

    axs[0].set_title(f'{coeff_type} coeff Kor - {flux_type}', fontsize=20)
    divider = make_axes_locatable(axs[0])
    cax_values = divider.append_axes('right', size='5%', pad=0.3)

    axs[1].set_title(f'Bel-Kor difference', fontsize=20)
    divider = make_axes_locatable(axs[1])
    cax_difference = divider.append_axes('right', size='5%', pad=0.3)

    coeff_min = np.nanmin(data_array)
    coeff_max = np.nanmax(data_array)
    # print(coeff_min)
    # print(coeff_max)

    # diff_min = np.nanmin(diff_array)
    # diff_max = np.nanmax(diff_array)

    if coeff_type == 'A':
        cmap = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                                   [0, (1.0 - coeff_min) / (coeff_max - coeff_min), 1])
        cmap.set_bad('darkgreen', 1.0)
        # cmap_diff = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - diff_min) / (diff_max - diff_min), 1])
        # cmap_diff.set_bad('darkgreen', 1.0)
    else:
        cmap = get_continuous_cmap(['#ffffff', '#ff0000'], [0, 1])
        cmap.set_bad('darkgreen', 1.0)
        # cmap_diff = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - diff_min) / (diff_max - diff_min), 1])
        # cmap_diff.set_bad('darkgreen', 1.0)

    pic_num = start_pic_num
    for t in tqdm.tqdm(range(time_start, time_end - 1)):
        date = datetime.datetime(2019, 1, 1, 0, 0) + datetime.timedelta(days=start_pic_num + (t - time_start))
        fig.suptitle(f'{coeff_type} coeff\n {date.strftime("%Y-%m-%d")}', fontsize=30)

        if flux_type == 'sensible':
            a_arr = np.load(files_path_prefix + f'Coeff_data/{t}_A_sens.npy')
        else:
            a_arr = np.load(files_path_prefix + f'Coeff_data/{t}_A_lat.npy')

        if flux_type == 'sensible':
            b_arr = np.load(files_path_prefix + f'Coeff_data/{t}_B.npy')[0]
        else:
            b_arr = np.load(files_path_prefix + f'Coeff_data/{t}_B.npy')[3]

        if coeff_type == 'A':
            Bel_arr = a_arr
        else:
            Bel_arr = b_arr

        if img_values is None:
            img_values = axs[0].imshow(data_array[t],
                                       interpolation='none',
                                       cmap=cmap,
                                       vmin=coeff_min,
                                       vmax=coeff_max)
        else:
            img_values.set_data(data_array[t])

        fig.colorbar(img_values, cax=cax_values, orientation='vertical')

        diff_array = Bel_arr - data_array[t]
        diff_min = np.nanmin(diff_array)
        diff_max = np.nanmax(diff_array)
        cmap_diff = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'],
                                        [0, (1.0 - diff_min) / (diff_max - diff_min), 1])
        cmap_diff.set_bad('darkgreen', 1.0)

        if img_difference is None:
            img_difference = axs[1].imshow(diff_array,
                                           interpolation='none',
                                           cmap=cmap_diff,
                                           vmin=diff_min,
                                           vmax=diff_max)
        else:
            img_difference.set_data(diff_array)

        fig.colorbar(img_difference, cax=cax_difference, orientation='vertical')
        fig.savefig(files_path_prefix + f'Components/plots/{coeff_type}_{flux_type}/{pic_num:05d}.png')
        pic_num += 1
    return
