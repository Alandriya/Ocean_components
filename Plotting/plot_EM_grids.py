import datetime
import os
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import tqdm
from data_processing import EM_dataframes_to_grids
from mpl_toolkits.axes_grid1 import make_axes_locatable

from video import truncate_colormap


def draw_4D(files_path_prefix,
            flux_type,
            component,
            timesteps):
    """
    Draws mean of selected component in a 4D interactive way, where X and Y are for point coordinates, Z for time and
    color is for value

    :param files_path_prefix: path to the working directory
    :param flux_type: string of the flux type: 'sensible' or 'latent'
    :param component: number of component to draw
    :param timesteps: time steps, not working with timesteps > 15
    :return:
    """
    X_grid = np.arange(0, 161)
    Y_grid = np.arange(0, 181)
    Z_grid = np.arange(0, timesteps)
    X, Y, Z = np.meshgrid(X_grid, Y_grid, Z_grid)
    colors = np.zeros_like(X, dtype=float)
    for t in range(timesteps):
        means_grid = np.load(files_path_prefix + f'tmp_arrays/{flux_type}/means_{t}.npy')
        colors[:, :, t] = means_grid[component].T

    df = pd.DataFrame()
    df['x'] = X.flatten()
    df['y'] = Y.flatten()
    df['time'] = Z.flatten()
    df['color'] = colors.flatten()
    df.fillna(0, inplace=True)
    fig = px.scatter_3d(df, x='x', y='y', z='time', color='color', opacity=0.1, size_max=1,
                        color_continuous_scale=px.colors.sequential.Blues)

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1, z=0.01))
    fig.show()
    return


def draw_3D(files_path_prefix,
            flux_type,
            components_amount,
            means_grid,
            sigmas_grid,
            weights_grid,
            t,
            borders):
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
    :param t: time step counting from 0 - used in picture name
    :param borders: 3 lists of lists with borders for colorbar: global max and min for parameters for each component
    shape of each is n_component * 2
    :return:
    """
    # plot surface
    date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(hours=6 * (62396 - 7320) + t * 24 * 7)
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
        ax.set_zlim(borders[0][0], borders[0][1])
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
        ax.set_zlim(borders[1][0], borders[1][1])
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
        ax.set_zlim(borders[2][0], borders[2][1])
        ax.set_title(f'Weight {comp + 1}', fontsize=20)
        fig.colorbar(surf, shrink=0.5, aspect=10)

    fig.tight_layout()
    fig.savefig(files_path_prefix + f'videos/{flux_type}/3D/{t + 1:05d}.png')
    plt.close(fig)
    return


def draw_2D(files_path_prefix,
            flux_type,
            components_amount,
            means_timelist,
            sigmas_timelist,
            weights_timelist,
            timesteps,
            borders):
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
    :param borders: 3 lists of lists with borders for colorbar: global max and min for parameters for each component
    shape of each is n_component * 2
    :return:
    """
    for t in range(timesteps):
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(hours=6 * (62396 - 7320) + t * 24 * 7)
        fig, axs = plt.subplots(3, components_amount, figsize=(components_amount * 5, 15))
        fig.suptitle(f'{flux_type}\n {date.strftime("%Y-%m-%d")}', fontsize=30)

        means_grid = means_timelist[t]
        sigmas_grid = sigmas_timelist[t]
        weights_grid = weights_timelist[t]
        for comp in range(components_amount):
            masked_grid = np.ma.array(means_grid[comp], mask=means_grid[comp] is None)
            cmap = matplotlib.cm.get_cmap("Blues").copy()

            levels = np.percentile(means_grid[comp][np.isfinite(means_grid[comp])], np.linspace(0, 100, 101))
            norm = matplotlib.colors.BoundaryNorm(levels, 256)
            # cmap = truncate_colormap(cmap, 0.2, 1.0)
            cmap.set_bad('white', 1.0)
            im = axs[0, comp].imshow(masked_grid,
                                     interpolation='none',
                                     cmap=cmap,
                                     norm=norm)
            axs[0, comp].set_title(f'Mean {comp + 1}', fontsize=20)
            # if comp == components_amount - 1:
            divider = make_axes_locatable(axs[0, comp])
            cax = divider.append_axes('right', size='5%', pad=0.3)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            # cbar.ax.locator_params(nbins=5)

            masked_grid = np.ma.array(sigmas_grid[comp], mask=sigmas_grid[comp] is None)
            cmap = matplotlib.cm.get_cmap("YlOrRd").copy()
            levels = np.percentile(sigmas_grid[comp][np.isfinite(sigmas_grid[comp])], np.linspace(0, 100, 101))
            norm = matplotlib.colors.BoundaryNorm(levels, 256)
            cmap.set_bad('white', 1.0)
            im = axs[1, comp].imshow(masked_grid,
                                     interpolation='nearest',
                                     cmap=cmap,
                                     norm=norm)
            axs[1, comp].set_title(f'Sigma {comp + 1}', fontsize=20)
            # if comp == components_amount - 1:
            divider = make_axes_locatable(axs[1, comp])
            cax = divider.append_axes('right', size='5%', pad=0.3)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            # cbar.ax.locator_params(nbins=5)

            masked_grid = np.ma.array(weights_grid[comp], mask=weights_grid[comp] is None)
            # levels = np.percentile(weights_grid[comp][np.isfinite(weights_grid[comp])], np.linspace(0, 100, 101))
            # norm = matplotlib.colors.BoundaryNorm(levels, 256)
            cmap = matplotlib.cm.get_cmap("jet").copy()
            cmap.set_bad('white', 1.0)
            im = axs[2, comp].imshow(masked_grid,
                                     interpolation='nearest',
                                     cmap=cmap,
                                     vmin=0,
                                     vmax=1)
            axs[2, comp].set_title(f'Weight {comp + 1}', fontsize=20)
            # if comp == components_amount - 1:
            divider = make_axes_locatable(axs[2, comp])
            cax = divider.append_axes('right', size='5%', pad=0.3)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            # cbar.ax.locator_params(nbins=5)

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'videos/{flux_type}/tmp/{t + 1:05d}.png')
        plt.close(fig)


def draw_frames(files_path_prefix, flux_type, mask, components_amount, timesteps=1):
    """

    :param files_path_prefix: path to the working directory
    :param flux_type: string of the flux type: 'sensible' or 'latent'
    :param mask: boolean 1D mask with length 161*181. If true, it's ocean point, if false - land. Only ocean points are
    of interest
    :param components_amount: components amount in dataframes
    :param timesteps: amount of timesteps to draw, t goes from 0 to timesteps
    :return:
    """
    font = {'weight': 'bold',
            'size': 16}

    matplotlib.rc('font', **font)

    # # check if all grids exist
    # print('Check grids')
    # for t in tqdm.tqdm(range(timesteps)):
    #     if not os.path.exists(files_path_prefix + f'tmp_arrays/{flux_type}/means_{t}.npy'):
    #         dataframes_to_grids(files_path_prefix, flux_type, mask, components_amount, timesteps)
    #         break

    means_timelist = list()
    sigmas_timelist = list()
    weights_timelist = list()
    means_borders = [[0, 0] for _ in range(components_amount)]
    sigmas_borders = [[0, 0] for _ in range(components_amount)]
    weights_borders = [[0, 1] for _ in range(components_amount)]

    # just for reading dataframes set timesteps = 0
    dataframes, indexes = EM_dataframes_to_grids(files_path_prefix, flux_type, mask, components_amount, 0)

    tmp = list(zip(indexes, dataframes))
    tmp.sort()
    indexes = [y for y, _ in tmp]
    dataframes = [x for _, x in tmp]

    print('Creating grids')
    # fill grids
    for t in tqdm.tqdm(range(timesteps)):
        if os.path.exists(files_path_prefix + f'tmp_arrays/{flux_type}/means_{t}.npy'):
            means_grid = np.load(files_path_prefix + f'tmp_arrays/{flux_type}/means_{t}.npy')
            sigmas_grid = np.load(files_path_prefix + f'tmp_arrays/{flux_type}/sigmas_{t}.npy')
            weights_grid = np.load(files_path_prefix + f'tmp_arrays/{flux_type}/weights_{t}.npy')
        else:
            grid = np.full((components_amount, 161, 181), np.nan)
            means_grid = deepcopy(grid)
            sigmas_grid = deepcopy(grid)
            weights_grid = deepcopy(grid)

            for i in range(len(mask)):
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

        for comp in range(components_amount):
            means_borders[comp][0] = min(means_borders[comp][0], np.nanmin(means_grid[comp]))
            means_borders[comp][1] = max(means_borders[comp][1], np.nanmax(means_grid[comp]))
            sigmas_borders[comp][1] = max(sigmas_borders[comp][1], np.nanmax(sigmas_grid[comp]))
            sigmas_borders[comp][0] = min(sigmas_borders[comp][1], np.nanmin(sigmas_grid[comp]))

    borders = [means_borders, sigmas_borders, weights_borders]

    draw_2D(files_path_prefix,
            flux_type,
            components_amount,
            means_timelist,
            sigmas_timelist,
            weights_timelist,
            timesteps,
            borders)
    return
