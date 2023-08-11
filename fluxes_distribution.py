import numpy as np
from struct import unpack
import matplotlib.pyplot as plt
import datetime
import matplotlib
import pylab
import os
import seaborn as sns
import tqdm

from VarGamma import fit_ml, pdf, cdf
import scipy
from scipy.stats import pearsonr


def draw_3d_hist(files_path_prefix: str,
                 sample_x: np.ndarray,
                 sample_y: np.ndarray,
                 time_start: int,
                 time_end: int,
                 postfix: str = ''):
    """
    Draws 3d histogram of joint fluxes distribution (without nan) in sample_x and sample_y from time+start to time_end period
    :param files_path_prefix: path to the working directory
    :param sample_x: sample of sensible flux without nan
    :param sample_y: sample of latent flux without nan
    :param time_start: start day index, counting from 01.01.1979
    :param time_end: end day index
    :param postfix: postfix to add to picture name
    :return:
    """
    fig = plt.figure(figsize=(15, 15))
    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=2, projection='3d')
    # ax1 = plt.subplot2grid((2, 2), (0, 1))
    # ax1 = plt.subplot2grid((2, 2), (1, 1))

    n_bins = 30
    borders = [-400, 200]
    hist, xedges, yedges = np.histogram2d(sample_x, sample_y, bins=n_bins, range=[borders, borders], density=True)
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = (borders[1] - borders[0]) / n_bins * np.ones_like(zpos)
    dz = hist.ravel()

    cmap = matplotlib.cm.get_cmap('jet').copy()
    max_height = np.max(dz)  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=rgba, alpha=0.6)

    date_start = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
    date_end = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
    fig.suptitle("Joint distribution of sensible and latent fluxes\n" +
                 f"{date_start.strftime('%Y-%m-%d')} - {date_end.strftime('%Y-%m-%d')}", fontsize=30)
    ax.set_xlabel("Sensible", fontsize=20, labelpad=20)
    ax.set_ylabel("Latent", fontsize=20, labelpad=20)

    # draw 2d projections
    fig.tight_layout()
    plt.savefig(files_path_prefix + f'Distributions/sens-lat_histogram_{postfix}.png')
    return


def plot_fluxes_2d(files_path_prefix: str,
                   sens_val: np.ndarray,
                   lat_val: np.ndarray,
                   time_start: int,
                   time_end: int):
    """
    Plots fluxes joint distrubition in 2d mode
    :param files_path_prefix: path to the working directory
    :param sens_val: sample of sensible flux without nan
    :param lat_val: sample of latent flux without nan
    :param time_start: start day index, counting from 01.01.1979
    :param time_end: end day index
    :return:
    """
    fig = plt.figure(figsize=(15, 15))
    # axs = fig.add_subplot(1, 2, 1, projection='3d')
    xedges = np.linspace(np.nanmin(sens_val), np.nanmax(sens_val), 100)
    yedges = np.linspace(np.nanmin(lat_val), np.nanmax(lat_val), 100)
    # xx, yy = np.meshgrid(x, y)

    H, xedges, yedges = np.histogram2d(sens_val, lat_val, bins=(xedges, yedges))

    # axs.cla()
    axs = fig.add_subplot(132, title=f'Sensible and latent fluxes distribution', aspect='equal')
    X, Y = np.meshgrid(xedges, yedges)
    # axs.pcolormesh(X, Y, H.T)
    axs.hist2d(sens_val, lat_val, bins=50, density=True, cmap='Reds')
    axs.set_title(f'Sensible and latent fluxes distribution')
    axs.legend()

    date_start = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
    date_end = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
    fig.suptitle(f"{date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')}")
    fig.savefig(files_path_prefix + f'Func_repr/fluxes_distribution/fluxes_2D_{time_start:05d}.png')
    return


def plot_estimate_fluxes_1d(files_path_prefix: str,
                            sens_val: np.ndarray,
                            lat_val: np.ndarray,
                            time_start: int,
                            time_end: int,
                            point: tuple,
                            start_year: int = 1979):
    """
    Plots histograms and estimates separate distributions of fluxes. Can use normal, t distribution and VarGamma
    :param files_path_prefix: path to the working directory
    :param sens_val: sample of sensible flux without nan
    :param lat_val: sample of latent flux without nan
    :param time_start: start day index, counting from 01.01.1979
    :param time_end: end day index
    :param point: point with coords (x, y) from which the sample is taken
    :param start_year: start year of data part (e.g. SENSIBLE_2009-2019.npy -> 2009)
    :return:
    """
    sens_val = sens_val[np.isfinite(sens_val)]
    lat_val = lat_val[np.isfinite(lat_val)]
    part = len(sens_val) // 4 * 3
    # sens_norm = scipy.stats.norm.fit_loc_scale(sens_val[:part])
    # lat_norm = scipy.stats.norm.fit_loc_scale(lat_val[:part])
    # print(f'Shapiro-Wilk normality test for sensible: {scipy.stats.shapiro(sens_val[part:part*2])[1]:.5f}')
    # print(f'Shapiro-Wilk normality test for latent: {scipy.stats.shapiro(lat_val[part:part*2])[1]:.5f}\n')

    # sens_t = scipy.stats.t.fit(sens_val[:part])
    # lat_t = scipy.stats.t.fit(lat_val[:part])

    sens_vargamma = fit_ml(sens_val[:part])
    lat_vargamma = fit_ml(lat_val[:part])
    print(f'Sensible parameters: {sens_vargamma}')
    print(f'Latent parameters: {lat_vargamma}')

    sns.set_style('whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    date_start = datetime.datetime(start_year, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
    date_end = datetime.datetime(start_year, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
    fig.suptitle(
        f"Sensible and latent fluxes distributions at ({point[0]}, {point[1]})\n {date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')}",
        fontsize=20)

    # mu, sigma = sens_norm
    x = np.linspace(-200, 100, 300)
    print(
        f'Kolmogorov-Smirnov test for VarGamma for sensible: {scipy.stats.kstest(sens_val[part:part * 2], cdf, sens_vargamma)[1]}')
    axs[0].cla()
    # axs[0].hist(sens_val[part:part*2], bins=15, density=True)
    sns.histplot(sens_val, bins=15, kde=False, ax=axs[0], stat='density')
    # axs[0].plot(x, scipy.stats.norm.pdf(x, mu, sigma), label='Fitted normal')
    axs[0].plot(x, pdf(x, *sens_vargamma), label=f'Fitted VarGamma,\n {chr(945)}='
                                                 f'{sens_vargamma[0]:.1f}; '
                                                 f'{chr(946)}={sens_vargamma[1]:.1f}; '
                                                 f'{chr(955)}={sens_vargamma[2]:.1f}; '
                                                 f'{chr(947)}={sens_vargamma[3]:.1f}', c='orange')
    # axs[0].plot(x, scipy.stats.t.pdf(x, *sens_t),  label='Fitted t')
    axs[0].set_title(f'Sensible', fontsize=16)
    axs[0].legend(bbox_to_anchor=(0.5, -0.5), loc="lower center")

    # mu, sigma = lat_norm
    x = np.linspace(-200, 100, 300)
    print(
        f'Kolmogorov-Smirnov test for VarGamma for latent: {scipy.stats.kstest(lat_val[part:part * 2], cdf, lat_vargamma)[1]}\n')
    axs[1].cla()
    # axs[1].hist(lat_val[part:part*2], bins=15, density=True)
    sns.histplot(lat_val, bins=15, kde=False, ax=axs[1], stat='density')
    # axs[1].plot(x, scipy.stats.norm.pdf(x, mu, sigma), label='Fitted normal')
    axs[1].plot(x, pdf(x, *lat_vargamma), label=f'Fitted VarGamma,\n {chr(945)}='
                                                f'{lat_vargamma[0]:.1f}; '
                                                f'{chr(946)}={lat_vargamma[1]:.1f}; '
                                                f'{chr(955)}={lat_vargamma[2]:.1f}; '
                                                f'{chr(947)}={lat_vargamma[3]:.1f}', c='orange')
    # axs[1].plot(x, scipy.stats.t.pdf(x, *lat_t), label='Fitted t')
    axs[1].set_title(f'Latent', fontsize=20)
    axs[1].legend(bbox_to_anchor=(0.5, -0.5), loc="lower center")

    plt.tight_layout()
    fig.savefig(files_path_prefix +
                f"Func_repr/fluxes_distribution/POINT_({point[0]},{point[1]})/({date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')}).png")
    plt.close(fig)
    return


def estimate_flux(files_path_prefix: str,
                  sensible_array: np.ndarray,
                  latent_array: np.ndarray,
                  month: int,
                  point: tuple,
                  radius: int):
    """
    Estimate separate distributions or joint distribution of the fluxes and plot results
    :param files_path_prefix: path to the working directory
    :param sensible_array: np.array with sensible flux data
    :param latent_array: np.array with latent flux data
    :param month: month start month of the window
    :param point: point with coords (x, y) from which the sample is taken
    :param radius: radius of the point
    :return:
    """
    point_bigger = [(point[0] + i, point[1] + j) for i in range(-radius, radius + 1) for j in
                    range(-radius, radius + 1)]
    flat_points = np.array([p[0] * 181 + p[1] for p in point_bigger])

    years = 1
    start_year = 2019
    for i in range(2, 3):
        time_start = (datetime.datetime(start_year + i, month, 1, 0, 0) - datetime.datetime(start_year, month, 1, 0,
                                                                                            0)).days
        time_end = (datetime.datetime(start_year + i+1, month + 0, 1, 0, 0) - datetime.datetime(start_year, month, 1, 0,
                                                                                              0)).days

        date_start = datetime.datetime(start_year + i, month, 1, 0, 0) + datetime.timedelta(days=time_start)
        date_end = datetime.datetime(start_year + i+1, month + 0, 1, 0, 0) + datetime.timedelta(days=time_end)

        if os.path.exists(files_path_prefix +
                          f"Func_repr/fluxes_distribution/data/POINT_({point[0]}, {point[1]})_({date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')})_sensible.npy"):
            sens_val = np.load(files_path_prefix +
                               f"Func_repr/fluxes_distribution/data/POINT_({point[0]}, {point[1]})_({date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')})_sensible.npy")
            lat_val = np.load(files_path_prefix +
                              f"Func_repr/fluxes_distribution/data/POINT_({point[0]}, {point[1]})_({date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')})_latent.npy")
        else:
            sens_val = sensible_array[flat_points, time_start:time_end].flatten()
            lat_val = latent_array[flat_points, time_start:time_end].flatten()
            np.save(files_path_prefix +
                    f"Func_repr/fluxes_distribution/data/POINT_({point[0]}, {point[1]})_({date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')})_sensible.npy",
                    sens_val)
            np.save(files_path_prefix +
                    f"Func_repr/fluxes_distribution/data/POINT_({point[0]}, {point[1]})_({date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')})_latent.npy",
                    lat_val)

        plot_estimate_fluxes_1d(files_path_prefix, sens_val, lat_val, time_start, time_end, point, start_year)
        # plot_fluxes_2d(files_path_prefix, sens_val, lat_val, time_start, time_end)
    return


def count_correlations(files_path_prefix: str,
                       sensible_array: np.ndarray,
                       latent_array: np.ndarray,
                       offset: int,
                       window_length: int=14,
                       observations_per_day: int=4,):
    """
    Counts Pearson correlations of the fluxes
    :param files_path_prefix: path to the working directory
    :param sensible_array: np.array with sensible flux data
    :param latent_array: np.array with latent flux data
    :param offset: offset in days from 01.01.1979, needed for output filename
    :param window_length: width of the moving window (in days) for correlation
    :param observations_per_day:
    :return:
    """
    corr = np.zeros((161, 181), dtype=float)
    for i in range(161):
        for j in range(181):
            if np.isnan(sensible_array[0, i, j]).any():
                corr[i, j] = np.nan

    for t in tqdm.tqdm(range(0, len(sensible_array)-window_length*observations_per_day, observations_per_day)):
        sensible = sensible_array[t:t+window_length*observations_per_day]
        latent = latent_array[t:t+window_length*observations_per_day]
        for i in range(161):
            for j in range(181):
                if not np.isnan(sensible[:, i, j]).any():
                    corr[i, j] = pearsonr(sensible[:, i, j], latent[:, i, j])[0]
        np.save(files_path_prefix + f'Flux_correlations/C_{t+offset}.npy', corr)
    return
