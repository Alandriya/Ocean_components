import os
import shutil
import tqdm
from copy import deepcopy
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


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


def draw_frames(files_path_prefix, flux_type, mask, components_amount, dataframes, indexes):
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

    font = {'weight': 'bold',
            'size': 16}

    matplotlib.rc('font', **font)

    data = dataframes[0]
    means_cols = data.filter(regex='mean_', axis=1).columns
    sigmas_cols = data.filter(regex='sigma_', axis=1).columns
    weights_cols = data.filter(regex='weight_', axis=1).columns

    for t in tqdm.tqdm(range(10)):
        grid = np.full((components_amount, 161, 181), np.nan)
        means_grid = deepcopy(grid)
        sigmas_grid = deepcopy(grid)
        weights_grid = deepcopy(grid)

        # fill grids
        for i in range(start, end):
            if mask[i] and i in indexes:
                rel_i = indexes.index(i)
                df = dataframes[rel_i]
                for comp in range(components_amount):
                    means_grid[comp][i // 181][i % 181] = df.loc[t, f'mean_{comp + 1}']
                    sigmas_grid[comp][i // 181][i % 181] = df.loc[t, f'sigma_{comp + 1}']
                    weights_grid[comp][i // 181][i % 181] = df.loc[t, f'weight_{comp + 1}']

        # 2d plots
        fig, axs = plt.subplots(3, components_amount, figsize=(components_amount * 5, 15))
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(hours=6*(62396 - 7320) + t*24*7)
        fig.suptitle(f'{flux_type}\n {date.strftime("%Y-%m-%d")}', fontsize=30)

        for comp in range(components_amount):
            masked_grid = np.ma.array(means_grid[comp], mask=means_grid[comp] is None)
            cmap = matplotlib.cm.get_cmap("Blues").copy()
            cmap = truncate_colormap(cmap, 0.2, 1.0)
            cmap.set_bad('white', 1.0)
            im = axs[0, comp].imshow(masked_grid,
                                     extent=(0, 161, 181, 0),
                                     interpolation='none',
                                     vmin=min(df[means_cols].min()),
                                     vmax=max(df[means_cols].max()),
                                     cmap=cmap)
            axs[0, comp].set_title(f'Mean {comp + 1}', fontsize=20)
            if comp == components_amount - 1:
                divider = make_axes_locatable(axs[0, comp])
                cax = divider.append_axes('right', size='10%', pad=0.7)
                fig.colorbar(im, cax=cax, orientation='vertical')

            masked_grid = np.ma.array(sigmas_grid[comp], mask=sigmas_grid[comp] is None)
            cmap = matplotlib.cm.get_cmap("YlOrRd").copy()
            cmap = truncate_colormap(cmap, 0.2, 1.0)
            cmap.set_bad('white', 1.0)
            im = axs[1, comp].imshow(masked_grid,
                                extent=(0, 161, 181, 0),
                                interpolation='nearest',
                                vmin=min(df[sigmas_cols].min()),
                                vmax=max(df[sigmas_cols].max()),
                                cmap=cmap)
            axs[1, comp].set_title(f'Sigma {comp + 1}', fontsize=20)
            if comp == components_amount - 1:
                divider = make_axes_locatable(axs[1, comp])
                cax = divider.append_axes('right', size='10%', pad=0.7)
                fig.colorbar(im, cax=cax, orientation='vertical')

            masked_grid = np.ma.array(weights_grid[comp], mask=weights_grid[comp] is None)
            cmap = matplotlib.cm.get_cmap("jet").copy()
            cmap.set_bad('white', 1.0)
            im = axs[2, comp].imshow(masked_grid,
                                extent=(0, 161, 181, 0),
                                interpolation='nearest',
                                vmin=min(df[weights_cols].min()),
                                vmax=max(df[weights_cols].max()),
                                cmap=cmap)
            axs[2, comp].set_title(f'Weight {comp + 1}', fontsize=20)
            if comp == components_amount - 1:
                divider = make_axes_locatable(axs[2, comp])
                cax = divider.append_axes('right', size='10%', pad=0.7)
                fig.colorbar(im, cax=cax, orientation='vertical')

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'videos/{flux_type}/tmp/{t + 1:03d}.png')
        plt.close(fig)

        # # 3d plots
        # date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(hours=6*(62396 - 7320) + t*24*7)
        # X = range(0, 161)
        # Y = range(0, 181)
        # X, Y = np.meshgrid(X, Y)
        # fig = plt.figure(figsize=(components_amount * 5, 15))
        # fig.suptitle(f'{flux_type}\n {date.strftime("%Y-%m-%d")}', fontsize=30)
        #
        # for comp in range(components_amount):
        #     ax = fig.add_subplot(3, components_amount, comp+1, projection='3d')
        #     masked_grid = np.ma.array(means_grid[comp], mask=np.isnan(means_grid[comp]))
        #     cmap = matplotlib.cm.get_cmap("Blues").copy()
        #     # cmap = truncate_colormap(cmap, 0.3, 1.0)
        #     cmap.set_bad('white', 1.0)
        #     surf = ax.plot_surface(X, Y, np.transpose(masked_grid), rstride=1, cstride=1, cmap=cmap, linewidth=0, antialiased=False)
        #
        #     ax.set_title(f'Mean {comp + 1}', fontsize=20)
        #     # if comp == components_amount - 1:
        #     #     divider = make_axes_locatable(ax)
        #     #     cax = fig.add_axes('bottom', size='10%', pad=0.7)
        #     #     fig.colorbar(surf, cax=cax, orientation='horizontal')
        #
        # # fig.tight_layout()
        # fig.savefig(files_path_prefix + f'videos/{flux_type}/3D/{t + 1:03d}.png')
        # plt.close(fig)
    return


def create_video(files_path_prefix, flux_type, name, speed=20):
    """

    :param speed:
    :param files_path_prefix: path to the working directory
    :param flux_type:
    :param name:
    :return:
    """
    video_name = files_path_prefix + f'videos/{name}.mp4'
    print(video_name)
    print(files_path_prefix + f'videos/{flux_type}/tmp/%03d.png')
    if os.path.exists(video_name):
        os.remove(video_name)

    subprocess.call([
        'ffmpeg', '-itsscale', str(speed), '-i', "D://Data/OceanFull/videos/sensible/tmp/%3d.png", '-r', '5', '-pix_fmt',
        'yuv420p', video_name,
    ])
    return
