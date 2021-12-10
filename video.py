import os
import shutil
import tqdm
from copy import deepcopy
import numpy as np
import subprocess
import matplotlib.pyplot as plt


def init_directory(files_path_prefix, flux_type):
    """

    :param files_path_prefix: path to the working directory
    :param flux_type:
    :return:
    """
    if not os.path.exists(files_path_prefix + f'videos/{flux_type}'):
        os.mkdir(files_path_prefix + f'videos/{flux_type}')

    if os.path.exists(files_path_prefix + f'videos/{flux_type}/tmp'):
        shutil.rmtree(files_path_prefix + f'videos/{flux_type}/tmp')

    if not os.path.exists(files_path_prefix + f'videos/{flux_type}/tmp'):
        os.mkdir(files_path_prefix + f'videos/{flux_type}/tmp')
    return


def draw_pictures(files_path_prefix, flux_type, mask, components_amount, dataframes, indexes):
    """

    :param files_path_prefix: path to the working directory
    :param flux_type:
    :param mask:
    :param components_amount:
    :param dataframes:
    :param indexes:
    :return:
    """
    start = 0
    end = len(mask)

    data = dataframes[0]
    means_cols = data.filter(regex='mean_', axis=1).columns
    sigmas_cols = data.filter(regex='sigma_', axis=1).columns
    weights_cols = data.filter(regex='weight_', axis=1).columns

    missing = list()
    for t in tqdm.tqdm(range(50)):
        grid = np.empty((161, 181))
        means_grid = deepcopy([grid] * components_amount)
        sigmas_grid = deepcopy([grid] * components_amount)
        weights_grid = deepcopy([grid] * components_amount)
        fig, axs = plt.subplots(3, components_amount, figsize=(components_amount * 5, 15))

        for i in range(start, end):
            if mask[i] and not i in indexes:
                missing.append(i)
            if mask[i] and i in indexes:
                rel_i = indexes.index(i)
                df = dataframes[rel_i]
                for comp in range(components_amount):
                    means_grid[comp][i // 181][i % 181] = df.loc[t, f'mean_{comp + 1}']
                    sigmas_grid[comp][i // 181][i % 181] = df.loc[t, f'sigma_{comp + 1}']
                    weights_grid[comp][i // 181][i % 181] = df.loc[t, f'weight_{comp + 1}']

        for comp in range(components_amount):
            im = axs[0, comp].imshow(means_grid[comp],
                                     extent=(0, 161, 181, 0),
                                     interpolation='nearest',
                                     vmin=min(df[means_cols].min()),
                                     vmax=max(df[means_cols].max()))
            axs[0, comp].title.set_text(f'Mean {comp + 1}')

            # # get the colors of the values, according to the colormap used by imshow
            # colors = [im.cmap(im.norm(value)) for value in values]
            # # create a patch (proxy artist) for every color
            # patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
            # # put those patched as legend-handles into the legend
            # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

            axs[1, comp].imshow(sigmas_grid[comp],
                                extent=(0, 161, 181, 0),
                                interpolation='nearest',
                                vmin=min(df[sigmas_cols].min()),
                                vmax=max(df[sigmas_cols].max()))
            axs[1, comp].title.set_text(f'Sigma {comp + 1}')

            axs[2, comp].imshow(weights_grid[comp],
                                extent=(0, 161, 181, 0),
                                interpolation='nearest',
                                vmin=min(df[weights_cols].min()),
                                vmax=max(df[weights_cols].max()))
            axs[2, comp].title.set_text(f'Weight {comp + 1}')

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'videos/{flux_type}/tmp/{t + 1:03d}.png')
        return


def create_video(files_path_prefix, flux_type, name):
    """

    :param files_path_prefix: path to the working directory
    :param flux_type:
    :param name:
    :return:
    """
    video_name = files_path_prefix + f'videos/{name}.mp4'
    if os.path.exists(video_name):
        os.remove(video_name)

    subprocess.call([
        'ffmpeg', '-i', files_path_prefix + f'videos/{flux_type}/tmp/%03d.png', '-r', '3', '-pix_fmt', 'yuv420p',
        video_name,
    ])
    return
