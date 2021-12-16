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
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def init_directory(files_path_prefix, flux_type):
    """
    Creates (or re-creates) subdirectories for saving pictures and video
    :param files_path_prefix: path to the working directory
    :param flux_type: string of the flux type: 'sensible' or 'latent'
    :return:
    """
    if not os.path.exists(files_path_prefix + f'videos/{flux_type}'):
        os.mkdir(files_path_prefix + f'videos/{flux_type}')

    if os.path.exists(files_path_prefix + f'videos/{flux_type}/tmp'):
        shutil.rmtree(files_path_prefix + f'videos/{flux_type}/tmp')

    if not os.path.exists(files_path_prefix + f'videos/{flux_type}/tmp'):
        os.mkdir(files_path_prefix + f'videos/{flux_type}/tmp')
    return


def draw_4D():
    # # plot 4D into 3D: time as Z (and only means and sigmas)
    # means_array = np.array(means_timelist)
    # print(means_array.shape)
    #
    # fig = plt.figure(figsize=(components_amount * 10, 20))
    # fig.suptitle(f'{flux_type}\n {date.strftime("%Y-%m-%d")}', fontsize=30)
    #
    # for comp in range(components_amount):
    #     X = range(0, 161)
    #     Y = range(0, 181)
    #     Z = np.array(range(0, timesteps))
    #     X, Y = np.meshgrid(X, Y)
    #     Z = np.outer(Z.T, Z)
    #
    #     # means
    #     ax = fig.add_subplot(2, components_amount, comp + 1, projection='3d')
    #     ax.set_xlim(0, 161)
    #     ax.set_ylim(0, 181)
    #     ax.set_zlim(0, timesteps)
    #
    #     color_dimension = means_array
    #     minn, maxx = color_dimension.min(), color_dimension.max()
    #     norm = matplotlib.colors.Normalize(minn, maxx)
    #     m = plt.cm.ScalarMappable(norm=norm, cmap='Blues')
    #     m.set_array([])
    #     fcolors = m.to_rgba(color_dimension)
    #
    #     surf = ax.plot_surface(X, Y, Z,  facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
    #     ax.set_title(f'Mean {comp + 1}', fontsize=20)
    #     fig.colorbar(surf, shrink=0.5, aspect=10)
    #
    #     fig.tight_layout()
    #     fig.savefig(files_path_prefix + f'videos/{flux_type}/3D/TIME_{t + 1:03d}.png')
    #     plt.close(fig)
    return


def draw_3D(files_path_prefix,
            flux_type,
            components_amount,
            means_grid,
            sigmas_grid,
            weights_grid,
            timesteps):
    """
    Draws timesteps amount of .png pictures, showing 3 * components amount graphics with 3D surface plot of each
    parameter of each component
    :param files_path_prefix: path to the working directory
    :param flux_type: string of the flux type: 'sensible' or 'latent'
    :param components_amount: components amount in dataframes
    :param means_grid: array with shape (components_amount, 161,181) filled with estimated mean for each component,
    if ocean tile, and NaN, if land tile.
    :param sigmas_grid: same as means_grid, but for sigmas
    :param weights_grid: same as weights_grid, but for weights
    :param timesteps: amount of timesteps to draw, t goes from 0 to timesteps
    :return:
    """
    for t in range(timesteps):
        # plot surface
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(hours=6*(62396 - 7320) + t*24*7)
        X = range(0, 161)
        Y = range(0, 181)
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure(figsize=(components_amount * 10, 30))
        fig.suptitle(f'{flux_type}\n {date.strftime("%Y-%m-%d")}', fontsize=30)

        # means
        for comp in range(components_amount):
            ax = fig.add_subplot(3, components_amount, comp + 1, projection='3d')
            ax.set_xlim(0, 161)
            ax.set_ylim(0, 181)
            cmap = matplotlib.cm.get_cmap("Blues").copy()
            cmap = truncate_colormap(cmap, 0.2, 1.0)
            cmap.set_bad('white', 0.0)

            masked_grid = np.ma.array(means_grid[comp], mask=np.isnan(means_grid[comp]))
            surf = ax.plot_surface(X, Y, np.transpose(masked_grid), rstride=1, cstride=1, cmap=cmap,
                                   linewidth=0, antialiased=False, alpha=0.3)
            ax.set_zlim(np.min(masked_grid), np.max(masked_grid))
            ax.set_title(f'Mean {comp + 1}', fontsize=20)
            fig.colorbar(surf, shrink=0.5, aspect=10)

        # sigmas
        for comp in range(components_amount):
            ax = fig.add_subplot(3, components_amount, components_amount + comp + 1, projection='3d')
            ax.set_xlim(0, 161)
            ax.set_ylim(0, 181)
            cmap = matplotlib.cm.get_cmap("YlOrRd").copy()
            cmap = truncate_colormap(cmap, 0.2, 1.0)
            cmap.set_bad('white', 0.0)

            masked_grid = np.ma.array(sigmas_grid[comp], mask=np.isnan(sigmas_grid[comp]))
            surf = ax.plot_surface(X, Y, np.transpose(masked_grid), rstride=1, cstride=1, cmap=cmap,
                                   linewidth=0, antialiased=False, alpha=0.3)
            ax.set_zlim(np.min(masked_grid), np.max(masked_grid))
            ax.set_title(f'Sigma {comp + 1}', fontsize=20)
            fig.colorbar(surf, shrink=0.5, aspect=10)

        # weights
        for comp in range(components_amount):
            ax = fig.add_subplot(3, components_amount, components_amount * 2 + comp + 1, projection='3d')
            ax.set_xlim(0, 161)
            ax.set_ylim(0, 181)
            cmap = matplotlib.cm.get_cmap("jet").copy()
            cmap.set_bad('white', 0.0)

            masked_grid = np.ma.array(weights_grid[comp], mask=np.isnan(weights_grid[comp]))
            surf = ax.plot_surface(X, Y, np.transpose(masked_grid), rstride=1, cstride=1, cmap=cmap,
                                   linewidth=0, antialiased=False, alpha=0.3)
            ax.set_zlim(np.min(masked_grid), np.max(masked_grid))
            ax.set_title(f'Weight {comp + 1}', fontsize=20)
            fig.colorbar(surf, shrink=0.5, aspect=10)

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'videos/{flux_type}/3D/{t + 1:03d}.png')
        plt.close(fig)
    return


def draw_2D(files_path_prefix,
            flux_type,
            components_amount,
            means_timelist,
            sigmas_timelist,
            weights_timelist,
            timesteps,
            dataframes):
    """
    Draws timesteps amount of .png pictures, showing 3 * components amount graphics with 2D plot of each parameter of
    each component
    :param files_path_prefix: path to the working directory
    :param flux_type: string of the flux type: 'sensible' or 'latent'
    :param components_amount: components amount in dataframes
    :param means_timelist: list with length timesteps of arrays with shape (components_amount, 161,181) filled with
    estimated mean for each component, if ocean tile, and NaN, if land tile.
    :param sigmas_timelist: same as means_timelist, but for sigmas
    :param weights_timelist: same as means_timelist, but for weights
    :param timesteps: amount of timesteps to draw, t goes from 0 to timesteps
    :param dataframes: dataframes with estimated parameters of components with column names like 'mean_1', 'sigma_1',
    'weight_1', ...
    :return:
    """
    data = dataframes[0]

    means_cols = data.filter(regex='mean_', axis=1).columns
    sigmas_cols = data.filter(regex='sigma_', axis=1).columns
    weights_cols = data.filter(regex='weight_', axis=1).columns

    for t in range(timesteps):
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(hours=6 * (62396 - 7320) + t * 24 * 7)
        fig, axs = plt.subplots(3, components_amount, figsize=(components_amount * 5, 15))
        fig.suptitle(f'{flux_type}\n {date.strftime("%Y-%m-%d")}', fontsize=30)

        df = dataframes[t]
        means_grid = means_timelist[t]
        sigmas_grid = sigmas_timelist[t]
        weights_grid = weights_timelist[t]
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


def draw_frames(files_path_prefix, flux_type, mask, components_amount, dataframes, indexes):
    """

    :param files_path_prefix: path to the working directory
    :param flux_type: string of the flux type: 'sensible' or 'latent'
    :param mask: boolean 1D mask with length 161*181. If true, it's ocean point, if false - land. Only ocean points are
    of interest
    :param components_amount: components amount in dataframes
    :param dataframes: dataframes with estimated parameters of components with column names like 'mean_1', 'sigma_1',
    'weight_1', ...
    :param indexes: list of indexes of dataframes to relate to grid 161*181 point
    :return:
    """
    start = 0
    end = len(mask)
    timesteps = 10

    font = {'weight': 'bold',
            'size': 16}

    matplotlib.rc('font', **font)

    means_timelist = list()
    sigmas_timelist = list()
    weights_timelist = list()

    # fill grids
    for t in tqdm.tqdm(range(timesteps)):
        grid = np.full((components_amount, 161, 181), np.nan)
        means_grid = deepcopy(grid)
        sigmas_grid = deepcopy(grid)
        weights_grid = deepcopy(grid)

        for i in range(start, end):
            if mask[i] and i in indexes:
                rel_i = indexes.index(i)
                df = dataframes[rel_i]
                for comp in range(components_amount):
                    means_grid[comp][i // 181][i % 181] = df.loc[t, f'mean_{comp + 1}']
                    sigmas_grid[comp][i // 181][i % 181] = df.loc[t, f'sigma_{comp + 1}']
                    weights_grid[comp][i // 181][i % 181] = df.loc[t, f'weight_{comp + 1}']

        means_timelist.append(means_grid)
        sigmas_timelist.append(sigmas_grid)
        weights_timelist.append(weights_grid)

    draw_2D(files_path_prefix,
            flux_type,
            components_amount,
            means_timelist,
            sigmas_timelist,
            weights_timelist,
            timesteps,
            dataframes)
    return


def create_video(files_path_prefix, flux_type, name, speed=20):
    """
    Creates an .mp4 video from pictures in tmp subdirectory
    :param speed: coefficient of video speed, the more - the slower
    :param files_path_prefix: path to the working directory
    :param flux_type: string of the flux type: 'sensible' or 'latent'
    :param name: short name of videofile to create
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
