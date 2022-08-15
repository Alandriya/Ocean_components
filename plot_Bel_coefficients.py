import numpy as np
import datetime
import gc
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from video import get_continuous_cmap
import seaborn as sns
import scipy
from VarGamma import fit_ml, pdf, cdf


def plot_ab_coefficients(files_path_prefix: str,
                         a_timelist: list,
                         b_timelist: list,
                         borders: list,
                         time_start: int,
                         time_end: int,
                         step: int = 1,
                         start_pic_num: int = 0):
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
    :param borders: min and max values of A, B and F to display on plot: assumed structure is
        [a_min, a_max, b_min, b_max, f_min, f_max].
    :param time_start: start point for time
    :param time_end: end point for time
    :param step: step in time for loop
    :param start_pic_num: number of first picture
    :return:
    """
    print('Saving A and B pictures')

    figa, axsa = plt.subplots(1, 2, figsize=(20, 10))
    figb, axsb = plt.subplots(2, 2, figsize=(20, 15))
    img_a_sens, img_a_lat = None, None
    img_b = [None for _ in range(4)]

    # TODO change - it's not a very good way to cut the outliers
    borders[3] = 1000.0

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
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=start_pic_num + (t - time_start))
        a_sens = a_timelist[t][0]
        a_lat = a_timelist[t][1]
        b_matrix = b_timelist[t]

        figa.suptitle(f'A coeff\n {date.strftime("%Y-%m-%d")}', fontsize=30)
        if img_a_sens is None:
            img_a_sens = axsa[0].imshow(a_sens,
                                        interpolation='none',
                                        cmap=cmap_a,
                                        vmin=a_min,
                                        vmax=a_max)
        else:
            img_a_sens.set_data(a_sens)

        figa.colorbar(img_a_sens, cax=cax_a_sens, orientation='vertical')

        if img_a_lat is None:
            img_a_lat = axsa[1].imshow(a_lat,
                                       interpolation='none',
                                       cmap=cmap_a,
                                       vmin=a_min,
                                       vmax=a_max)
        else:
            img_a_lat.set_data(a_lat)

        figa.colorbar(img_a_lat, cax=cax_a_lat, orientation='vertical')
        figa.savefig(files_path_prefix + f'videos/A/A_{pic_num:05d}.png')

        figb.suptitle(f'B coeff\n {date.strftime("%Y-%m-%d")}', fontsize=30)
        for i in range(4):
            if img_b[i] is None:
                img_b[i] = axsb[i // 2][i % 2].imshow(b_matrix[i],
                                                      interpolation='none',
                                                      cmap=cmap_b,
                                                      vmin=borders[2],
                                                      vmax=borders[3])
            else:
                img_b[i].set_data(b_matrix[i])

            figb.colorbar(img_b[i], cax=cax_b[i], orientation='vertical')

        figb.savefig(files_path_prefix + f'videos/B/B_{pic_num:05d}.png')
        pic_num += 1

        del a_sens, a_lat, b_matrix
        gc.collect()
    return


def plot_c_coeff(files_path_prefix: str,
                 c_timelist: list,
                 time_start: int,
                 time_end: int,
                 step: int = 1,
                 start_pic_num: int = 0):
    """
    Plots C - correltion between A and B coefficients dynamics as frames and saves them into
    files_path_prefix + videos/tmp-coeff directory starting from start_pic_num, with step 1 (in numbers of pictures).

    :param files_path_prefix: path to the working directory
    :param c_timelist: list with not strictly defined length because of using window of some width to count its values,
        presumably its length = timesteps - time_window_width, where the second is defined in another function. Elements
        of the list are np.arrays with shape (2, 161, 181) containing 4 matrices of correlation of A and B coefficients:
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
    figc, axsc = plt.subplots(1, 2, figsize=(20, 10))
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
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=start_pic_num + (t - time_start))
        date_end = date + datetime.timedelta(days=14)
        figc.suptitle(f'Correlations\n {date.strftime("%Y-%m-%d")} - {date_end.strftime("%Y-%m-%d")}', fontsize=30)
        if img_1 is None:
            img_1 = axsc[0].imshow(c_timelist[t][0],
                                   interpolation='none',
                                   cmap=cmap,
                                   vmin=-1,
                                   vmax=1)
        else:
            img_1.set_data(c_timelist[t][0])
        figc.colorbar(img_1, cax=cax_1, orientation='vertical')

        if img_2 is None:
            img_2 = axsc[1].imshow(c_timelist[t][1],
                                   interpolation='none',
                                   cmap=cmap,
                                   vmin=-1,
                                   vmax=1)
        else:
            img_2.set_data(c_timelist[t][1])

        figc.colorbar(img_2, cax=cax_2, orientation='vertical')

        figc.savefig(files_path_prefix + f'videos/C/C_{pic_num:05d}.png')
        pic_num += 1
    return


def plot_f_coeff(files_path_prefix: str,
                 f_timelist: list,
                 borders: list,
                 time_start: int,
                 time_end: int,
                 step: int = 1,
                 start_pic_num: int = 0,
                 mean_width: int = 7):
    """
    Plots F - fraction of coefficients' norms and saves them into
    files_path_prefix + videos/tmp-coeff directory starting from start_pic_num, with step 1 (in numbers of pictures).

    :param files_path_prefix: path to the working directory
    :param f_timelist: list of np.arrays with shape (161, 181) containing fraction ||A||/||B|| in each point.
    :param borders: min and max values of A and B to display on plot: assumed structure is
        [a_min, a_max, b_min, b_max, f_min, f_max].
    :param time_start: start point for time
    :param time_end: end point for time
    :param step: step in time for loop
    :param start_pic_num: number of first picture
    :return:
    """
    print('Saving F pictures')
    f_max = borders[5]

    # prepare images
    fig, axs = plt.subplots(1, 1, figsize=(20, 10))
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.3)
    img_f = None

    cmap = colors.ListedColormap(['MintCream', 'Aquamarine', 'blue', 'red', 'DarkRed'])
    boundaries = [0, 0.25, 0.5, 1.0, 1.5, max(f_max, 2.0)]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    cmap.set_bad('darkgreen', 1.0)
    pic_num = start_pic_num
    for t in tqdm.tqdm(range(time_start, time_end, step)):
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=start_pic_num + (t - time_start))
        date_end = date + datetime.timedelta(days=mean_width)
        fig.suptitle(f'Fraction of coefficients\n {date.strftime("%Y-%m-%d")} - '
                     f'{date_end.strftime("%Y-%m-%d")}', fontsize=30)
        f = f_timelist[t]
        if img_f is None:
            img_f = axs.imshow(f,
                               interpolation='none',
                               cmap=cmap,
                               norm=norm)
        else:
            img_f.set_data(f)
        fig.colorbar(img_f, cax=cax, orientation='vertical')
        fig.savefig(files_path_prefix + f'videos/FN/FN_{pic_num:05d}.png')
        pic_num += 1
    return


def plot_fs_coeff(files_path_prefix: str,
                  f_timelist: list,
                  borders: list,
                  time_start: int,
                  time_end: int,
                  step: int = 1,
                  start_pic_num: int = 0,
                  mean_width: int = 7):
    """
    Plots FS - fractions mean(abs(a_sens)) / mean(abs(B[0]))  and mean(abs(a_lat) / mean(abs(B[1]))
    coefficients' where mean is taken in [t, t + mean_width] days window and saves them into
    files_path_prefix + videos/tmp-coeff directory starting from start_pic_num, with step 1 (in numbers of pictures).

    :param files_path_prefix: path to the working directory
    :param f_timelist: list of np.arrays with shape (2, 161, 181)
    :param borders: min and max values of A and B to display on plot: assumed structure is
        [a_min, a_max, b_min, b_max, f_min, f_max].
    :param time_start: start point for time
    :param time_end: end point for time
    :param step: step in time for loop
    :param start_pic_num: number of first picture
    :param mean_width: width in days of window to count mean
    :return:
    """
    print('Saving FS pictures')
    f_max = borders[5]

    # prepare images
    figf, axsf = plt.subplots(1, 2, figsize=(20, 10))
    axsf[0].set_title(f'Sensible', fontsize=20)
    divider = make_axes_locatable(axsf[0])
    cax_1 = divider.append_axes('right', size='5%', pad=0.3)

    axsf[1].set_title(f'Latent', fontsize=20)
    divider = make_axes_locatable(axsf[1])
    cax_2 = divider.append_axes('right', size='5%', pad=0.3)

    img_1, img_2 = None, None
    cmap = colors.ListedColormap(['MintCream', 'Aquamarine', 'blue', 'red', 'DarkRed'])
    boundaries = [0, 0.25, 0.5, 1.0, 1.5, max(f_max, 2.0)]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    cmap.set_bad('darkgreen', 1.0)

    pic_num = start_pic_num
    for t in tqdm.tqdm(range(time_start, time_end, step)):
        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=start_pic_num + (t - time_start))
        date_end = date + datetime.timedelta(days=mean_width)
        figf.suptitle(f'Fraction a/b for each flux type\n {date.strftime("%Y-%m-%d")} - '
                      f'{date_end.strftime("%Y-%m-%d")}', fontsize=30)
        if img_1 is None:
            img_1 = axsf[0].imshow(f_timelist[t][0],
                                   interpolation='none',
                                   cmap=cmap,
                                   norm=norm)
        else:
            img_1.set_data(f_timelist[t][0])
        figf.colorbar(img_1, cax=cax_1, orientation='vertical')

        if img_2 is None:
            img_2 = axsf[1].imshow(f_timelist[t][1],
                                   interpolation='none',
                                   cmap=cmap,
                                   norm=norm)
        else:
            img_2.set_data(f_timelist[t][1])

        figf.colorbar(img_2, cax=cax_2, orientation='vertical')
        figf.tight_layout()
        figf.savefig(files_path_prefix + f'videos/FS/FS_{pic_num:05d}.png')
        pic_num += 1
    return


def plot_mean_year(files_path_prefix: str, coeff_name: str):
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
            fig.colorbar(img[i], cax=cax[i], orientation='vertical')
    # elif coeff_name == 'C_0':
    #     fig.suptitle(f'A_sens - A_lat correlation mean year', fontsize=30)
    elif coeff_name == 'C_1':
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
            fig.colorbar(img[i], cax=cax[i], orientation='vertical')

    plt.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Mean_year/{coeff_name}.png')
    return


def plot_estimate_ab_distributions(files_path_prefix: str,
                          a_timelist:list,
                          b_timelist:list,
                          time_start:int,
                          time_end:int,
                          point: tuple):
    a_sens_sample = list()
    a_lat_sample = list()
    for t in tqdm.tqdm(range(0, time_end-time_start)):
        a_sens_sample.append(a_timelist[t][0][point])
        a_lat_sample.append(a_timelist[t][1][point])
    # a_sens_sample = np.load(files_path_prefix + f'Extreme/data/Flux_a_max_sens(1-15797)_30.npy')
    # a_lat_sample = np.load(files_path_prefix + f'Extreme/data/Flux_a_max_lat(1-15797)_30.npy')

    part = len(a_sens_sample) // 4 * 3
    sens_norm = scipy.stats.norm.fit_loc_scale(a_sens_sample[:part])
    lat_norm = scipy.stats.norm.fit_loc_scale(a_lat_sample[:part])
    print(f'Shapiro-Wilk normality test for sensible: {scipy.stats.shapiro(a_sens_sample[part:part*2])[1]:.5f}')
    print(f'Shapiro-Wilk normality test for latent: {scipy.stats.shapiro(a_lat_sample[part:part*2])[1]:.5f}\n')

    sens_t = scipy.stats.t.fit(a_sens_sample[:part])
    sens_t_pval = scipy.stats.kstest(a_sens_sample[part:], scipy.stats.t.cdf, sens_t)[1]
    lat_t = scipy.stats.t.fit(a_lat_sample[:part])
    lat_t_pval = scipy.stats.kstest(a_lat_sample[part:], scipy.stats.t.cdf, lat_t)[1]

    sens_vargamma = fit_ml(a_sens_sample[:part])
    sens_vg_pval = scipy.stats.kstest(a_sens_sample[part:], cdf, sens_vargamma)[1]
    lat_vargamma = fit_ml(a_lat_sample[:part])
    lat_vg_pval = scipy.stats.kstest(a_lat_sample[part:], cdf, lat_vargamma)[1]
    print(f'Sensible parameters: {sens_vargamma}')
    print(f'Latent parameters: {lat_vargamma}')

    sns.set_style('whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    date_start = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_start)
    date_end = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(days=time_end)
    fig.suptitle(
        f"a1 and a2 distributions at ({point[0]}, {point[1]})\n {date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')}",
        fontsize=20)

    # fig.suptitle(
    #     f"Max of a1 and a2 \n {date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')}",
    #     fontsize=20)

    x = np.linspace(-300, 300, 1000)
    data = a_sens_sample
    binwidth=5
    sns.histplot(a_sens_sample, bins=np.arange(min(data), max(data) + binwidth, binwidth), kde=False, ax=axs[0], stat='density')
    axs[0].set_title(f'a1', fontsize=16)
    mu, sigma = sens_norm
    axs[0].plot(x, scipy.stats.norm.pdf(x, mu, sigma), label='Fitted normal', c='y')
    axs[0].plot(x, scipy.stats.t.pdf(x, *sens_t), label=f'Fitted t, p_value = {sens_t_pval:.5f}', c='orange')
    axs[0].plot(x, pdf(x, *sens_vargamma), label=f'Fitted VarGamma,\n {chr(945)}='
                                                 f'{sens_vargamma[0]:.1f}; '
                                                 f'{chr(946)}={sens_vargamma[1]:.1f}; '
                                                 f'{chr(955)}={sens_vargamma[2]:.1f}; '
                                                 f'{chr(947)}={sens_vargamma[3]:.1f}\n'
                                                 f'p_value = {sens_vg_pval:.5f}', c='g')
    axs[0].legend(bbox_to_anchor=(0.5, -0.5), loc="lower center")

    data = a_lat_sample
    binwidth=5
    sns.histplot(a_lat_sample, bins=np.arange(min(data), max(data) + binwidth, binwidth), kde=False, ax=axs[1], stat='density')
    axs[1].set_title(f'a2', fontsize=16)
    mu, sigma = lat_norm
    axs[1].plot(x, scipy.stats.norm.pdf(x, mu, sigma), label='Fitted normal', c='y')
    axs[1].plot(x, pdf(x, *lat_vargamma), label=f'Fitted VarGamma,\n {chr(945)}='
                                                f'{lat_vargamma[0]:.1f}; '
                                                f'{chr(946)}={lat_vargamma[1]:.1f}; '
                                                f'{chr(955)}={lat_vargamma[2]:.1f}; '
                                                f'{chr(947)}={lat_vargamma[3]:.1f}\n'
                                                f'p_value = {lat_vg_pval:.5f}', c='g')
    axs[1].plot(x, scipy.stats.t.pdf(x, *lat_t), label=f'Fitted t, p_value = {lat_t_pval:.5f}', c='orange')
    axs[1].legend(bbox_to_anchor=(0.5, -0.5), loc="lower center")

    plt.tight_layout()
    fig.savefig(files_path_prefix +
                f"Distributions/AB_distr/A_HIST_POINT_({point[0]},{point[1]})_({date_start.strftime('%d.%m.%Y')} - {date_end.strftime('%d.%m.%Y')}).png")
    plt.close(fig)
    return