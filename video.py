import os
import shutil
import tqdm
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import matplotlib
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from copy import deepcopy
from data_processing import EM_dataframes_to_grids, scale_to_bins
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
import plotly.express as px
import gc


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncates a matplotlib sequential colormap from [0, 1] segment to [minval, maxval]
    :param cmap: matplotlib colormap
    :param minval:
    :param maxval:
    :param n:
    :return:
    """
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

    if not os.path.exists(files_path_prefix + f'tmp_arrays/{flux_type}'):
        os.mkdir(files_path_prefix + f'tmp_arrays/{flux_type}')

    if os.path.exists(files_path_prefix + f'videos/{flux_type}/tmp'):
        shutil.rmtree(files_path_prefix + f'videos/{flux_type}/tmp')

    if not os.path.exists(files_path_prefix + f'videos/{flux_type}/tmp'):
        os.mkdir(files_path_prefix + f'videos/{flux_type}/tmp')
    return


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
    colors = np.zeros_like(X,dtype=float)
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
                                     extent=(0, 161, 181, 0),
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
                                extent=(0, 161, 181, 0),
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
                                extent=(0, 161, 181, 0),
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


def create_video(files_path_prefix, tmp_dir, pic_prefix, name, speed=20, start=0):
    """
    Creates an .mp4 video from pictures in tmp subdirectory of full_prefix path with pic_prefix in filenames
    :param speed: coefficient of video speed, the more - the slower
    :param files_path_prefix: path to the working directory
    :param name: short name of videofile to create
    :param start: start number of pictures
    :return:
    """
    video_name = files_path_prefix + f'videos/{name}.mp4'
    # print(video_name)
    # print(files_path_prefix + f'videos/{flux_type}/tmp/%05d.png')
    if os.path.exists(video_name):
        os.remove(video_name)

    subprocess.call([
        'ffmpeg', '-itsscale', str(speed), '-i', tmp_dir + f"{pic_prefix}%5d.png", '-start_number', str(start),
        '-r', '5', '-pix_fmt', 'yuv420p', video_name,
    ])
    return


def plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, borders, time_start, time_end, step=1, start_pic_num=0):
    """
    Plots A, B dynamics as frames and saves them into files_path_prefix + videos/tmp-coeff
    directory starting from start_pic_num, with step 1 (in numbers of pictures).
    :param files_path_prefix: path to the working directory
    :param a_timelist: list with length = timesteps with structure [a_sens, a_lat], where a_sens and a_lat are
    np.arrays with shape (161, 181) with values for A coefficient for sensible and latent fluxes, respectively
    :param b_timelist: list with length = timesteps with b_matrix as elements, where b_matrix is np.array with shape
    (4, 161, 181) containing 4 matrices with elements of 2x2 matrix of coefficient B for every point of grid.
    0 is for B11 = sensible at t0 - sensible at t1,
    1 is for B12 = sensible at t0 - latent at t1,
    2 is for B21 = latent at t0 - sensible at t1,
    3 is for B22 = latent at t0 - latent at t1.
    :param borders: min and max values of A and B to display on plot: assumed structure is
    [a_min, a_max, b_min, b_max, f_min, f_max].
    :param time_start: start point for time
    :param time_end: end point for time
    :param step: step in time for loop
    :param start_pic_num: number of first picture
    :return:
    """
    print('Saving A and B pictures')

    figa, axsa = plt.subplots(1, 2, figsize=(20, 15))
    figb, axsb = plt.subplots(2, 2, figsize=(20, 20))
    img_a_sens, img_a_lat = None, None
    img_b = [None for _ in range(4)]

    # TODO FIX THIS
    borders[3] = 100.0

    a_min = borders[0]
    a_max = borders[1]
    b_min = borders[2]
    b_max = borders[3]

    cmap_a = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, (1.0 - a_min) / (a_max - a_min), 1])
    cmap_a.set_bad('darkgreen', 1.0)

    axsa[1].set_title(f'Latent', fontsize=20)
    divider = make_axes_locatable(axsa[1])
    cax_a_lat = divider.append_axes('right', size='5%', pad=0.3)

    axsa[0].set_title(f'Sensible', fontsize=20)
    divider = make_axes_locatable(axsa[0])
    cax_a_sens = divider.append_axes('right', size='5%', pad=0.3)

    cax_b = list()
    for i in range(4):
        divider = make_axes_locatable(axsb[i // 2][i % 2])
        cax_b.append(divider.append_axes('right', size='5%', pad=0.3))
        if i == 0:
            axsb[i // 2][i % 2].set_title(f'Sensible - sensible', fontsize=20)
        elif i == 3:
            axsb[i // 2][i % 2].set_title(f'Latent - latent', fontsize=20)
        elif i == 1:
            axsb[i // 2][i % 2].set_title(f'Sensible - latent', fontsize=20)
        elif i == 2:
            axsb[i // 2][i % 2].set_title(f'Latent - sensible', fontsize=20)

    zero_percent = abs(0 - b_min) / (b_max - b_min)
    cmap_b = get_continuous_cmap(['#000080', '#ffffff', '#ff0000'], [0, zero_percent, 1])
    cmap_b.set_bad('darkgreen', 1.0)

    pic_num = start_pic_num
    for t in tqdm.tqdm(range(time_start, time_end, step)):
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=start_pic_num + t)
        a_sens = a_timelist[t][0]
        a_lat = a_timelist[t][1]
        b_matrix = b_timelist[t]

        figa.suptitle(f'A coeff\n {date.strftime("%Y-%m-%d")}', fontsize=30)
        if img_a_sens is None:
            img_a_sens = axsa[0].imshow(a_sens,
                                extent=(0, 161, 181, 0),
                                interpolation='none',
                                cmap=cmap_a,
                                vmin=a_min,
                                vmax=a_max)
        else:
            img_a_sens.set_data(a_sens)

        figa.colorbar(img_a_sens, cax=cax_a_sens, orientation='vertical')

        if img_a_lat is None:
            img_a_lat = axsa[1].imshow(a_lat,
                                extent=(0, 161, 181, 0),
                                interpolation='none',
                                cmap=cmap_a,
                                vmin=a_min,
                                vmax=a_max)
        else:
            img_a_lat.set_data(a_lat)

        figa.colorbar(img_a_lat, cax=cax_a_lat, orientation='vertical')
        figa.savefig(files_path_prefix + f'videos/tmp-coeff/A_{pic_num:05d}.png')

        figb.suptitle(f'B coeff\n {date.strftime("%Y-%m-%d")}', fontsize=30)
        for i in range(4):
            if img_b[i] is None:
                img_b[i] = axsb[i // 2][i % 2].imshow(b_matrix[i],
                                                extent=(0, 161, 181, 0),
                                                interpolation='none',
                                                cmap=cmap_b,
                                                vmin=borders[2],
                                                vmax=borders[3])
            else:
                img_b[i].set_data(b_matrix[i])

            figb.colorbar(img_b[i], cax=cax_b[i], orientation='vertical')

        figb.savefig(files_path_prefix + f'videos/tmp-coeff/B_{pic_num:05d}.png')
        pic_num += 1

        del a_sens, a_lat, b_matrix
        gc.collect()
    return


def plot_c_coeff(files_path_prefix, c_timelist, time_start, time_end, step=1, start_pic_num=0):
    """
    Plots C - correltion between A and B coefficients dynamics as frames and saves them into
    files_path_prefix + videos/tmp-coeff directory starting from start_pic_num, with step 1 (in numbers of pictures).

    :param files_path_prefix: path to the working directory
    :param c_timelist: list with not strictly defined length because of using window of some width to count its values,
    presumably its length = timesteps - time_window_width, where the second is defined in another function. Elements of
    the list are np.arrays with shape (2, 161, 181) containing 4 matrices of correlation of A and B coefficients:
    0 is for (a_sens, a_lat) correlation,
    1 is for (B11, B22) correlation.
    :param time_start: start point for time
    :param time_end: end point for time
    :param step: step in time for loop
    :param start_pic_num: number of first picture
    :return:
    """
    print('Saving C pictures')

    # prepare images
    figc, axsc = plt.subplots(1, 2, figsize=(20, 15))
    axsc[0].set_title(f'A_sens - A_lat correlation', fontsize=20)
    divider = make_axes_locatable(axsc[0])
    cax_1 = divider.append_axes('right', size='5%', pad=0.3)

    axsc[1].set_title(f'B11 - B22 correlation', fontsize=20)
    divider = make_axes_locatable(axsc[1])
    cax_2 = divider.append_axes('right', size='5%', pad=0.3)

    img_1, img_2 = None, None
    cmap = get_continuous_cmap(['#4073ff', '#ffffff', '#ffffff', '#db4035'], [0, 0.4, 0.6, 1])
    cmap.set_bad('darkgreen', 1.0)

    pic_num = start_pic_num
    for t in tqdm.tqdm(range(time_start, time_end, step)):
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(hours=6 * (62396 - 7320) + t * 24)
        figc.suptitle(f'Correlations\n {date.strftime("%Y-%m-%d")}', fontsize=30)
        if img_1 is None:
            img_1 = axsc[0].imshow(c_timelist[t][0],
                                extent=(0, 161, 181, 0),
                                interpolation='none',
                                cmap=cmap,
                                vmin=-1,
                                vmax=1)
        else:
            img_1.set_data(c_timelist[t][0])
        figc.colorbar(img_1, cax=cax_1, orientation='vertical')

        if img_2 is None:
            img_2 = axsc[1].imshow(c_timelist[t][1],
                                extent=(0, 161, 181, 0),
                                interpolation='none',
                                cmap=cmap,
                                vmin=-1,
                                vmax=1)
        else:
            img_2.set_data(c_timelist[t][1])

        figc.colorbar(img_2, cax=cax_2, orientation='vertical')

        figc.savefig(files_path_prefix + f'videos/tmp-coeff/C_{pic_num:05d}.png')
        pic_num += 1
    return


def plot_flux_correlations(files_path_prefix, time_start, time_end, step=1):
    fig, axs = plt.subplots(figsize=(15, 15))
    pic_num = 0
    for t in tqdm.tqdm(range(time_start, time_end, step)):
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(hours=6 * (62396 - 7320) + t * 24)
        corr = np.load(files_path_prefix + f'Flux_correlations/FL_Corr_{t}.npy')
        fig.suptitle(f'Flux correlation\n {date.strftime("%Y-%m-%d")}', fontsize=30)

        cmap = get_continuous_cmap(['#4073ff', '#ffffff', '#ffffff', '#db4035'], [0, 0.4, 0.6, 1])
        cmap.set_bad('darkgreen', 1.0)
        im = axs.imshow(corr,
                            extent=(0, 161, 181, 0),
                            interpolation='none',
                            cmap=cmap,
                            vmin=-1,
                            vmax=1)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes('right', size='5%', pad=0.3)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')

        # fig.tight_layout()
        fig.savefig(files_path_prefix + f'videos/Flux-corr/FL_corr_{pic_num:05d}.png')
        pic_num += 1

    return


def plot_f_coeff(files_path_prefix, f_timelist, borders, time_start, time_end, step=1, start_pic_num=0):
    """
    Plots F - fraction of A and B coefficients as frames and saves them into
    files_path_prefix + videos/tmp-coeff directory starting from start_pic_num, with step 1 (in numbers of pictures).
    :param files_path_prefix: path to the working directory
    :param f_timelist:
    :param borders: min and max values of A and B to display on plot: assumed structure is
    [a_min, a_max, b_min, b_max, f_min, f_max].
    :param time_start: start point for time
    :param time_end: end point for time
    :param step: step in time for loop
    :param start_pic_num: number of first picture
    :return:
    """
    print('Saving F pictures')
    fig, axs = plt.subplots(figsize=(15, 15))
    f_min = borders[4]
    f_max = borders[5]
    cmap = get_continuous_cmap(['#000080', '#ffffff', '#dc143c'], [0, (1.0 - f_min) / (f_max - f_min), 1])
    cmap.set_bad('darkgreen', 1.0)
    img_f = None
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.3)

    pic_num = start_pic_num
    for t in tqdm.tqdm(range(time_start, time_end, step)):
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(hours=t * 24)
        f = f_timelist[t]
        fig.suptitle(f'F coefficient\n {date.strftime("%Y-%m-%d")}', fontsize=30)

        if img_f is None:
            img_f = axs.imshow(f,
                                extent=(0, 161, 181, 0),
                                interpolation='none',
                                cmap=cmap,
                                vmin=f_min,
                                vmax=f_max)
        else:
            img_f.set_data(f)

        fig.colorbar(img_f, cax=cax, orientation='vertical')
        fig.savefig(files_path_prefix + f'videos/tmp-coeff/F_{pic_num:05d}.png')
        pic_num += 1
    return
