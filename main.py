import os.path
import time

import numpy as np
import pandas as pd

from video import *
from data_processing import *
from ABC_coeff_counting import *
from func_estimation import *
from data_processing import load_prepare_fluxes
from func_estimation import estimate_by_months
import cycler


# Parameters
files_path_prefix = 'D://Data/OceanFull/'
flux_type = 'sensible'
# flux_type = 'latent'

# timesteps = 7320
timesteps = 1829

if __name__ == '__main__':
    # borders = [[7318, 7325]]
    # maskfile = open(files_path_prefix + "mask", "rb")
    # binary_values = maskfile.read(29141)
    # maskfile.close()
    # mask = unpack('?' * 29141, binary_values)
    #
    # sensible_array = np.load(files_path_prefix + 'sensible_all.npy')
    # latent_array = np.load(files_path_prefix + 'latent_all.npy')
    #
    # for border in borders:
    #     start = border[0]-1
    #     end = border[1]
    #
    #     count_abf_coefficients(files_path_prefix, mask, sensible_array[:, start-1:end+1], latent_array[:, start-1:end+1], time_start=0, time_end=end-start,
    #                            offset=start)
    # offset = 14640 / 4 * 1
    # parallel_AB(4, 'SENSIBLE_1989-1999.npy', 'LATENT_1989-1999.npy', offset)

    # ---------------------------------------------------------------------------------------
    # binary_to_array(files_path_prefix, "s79-21", 'SENSIBLE_2019-2021')
    # ---------------------------------------------------------------------------------------
    # Components determination part
    # sort_by_means(files_path_prefix, flux_type)
    # init_directory(files_path_prefix, flux_type)

    # dataframes_to_grids(files_path_prefix, flux_type, mask, components_amount, 100)
    # draw_frames(files_path_prefix, flux_type, mask, components_amount, timesteps=timesteps)
    # create_video(files_path_prefix, files_path_prefix+'videos/{flux_type}/tmp/', '', f'{flux_type}_5years_weekly', speed=30)
    # ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------
    estimate_by_months(files_path_prefix, 5)
    raise ValueError

    sensible_array = np.load(files_path_prefix + 'sensible_all.npy')
    latent_array = np.load(files_path_prefix + 'latent_all.npy')

    months_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                    9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    font = {'size': 14}
    matplotlib.rc('font', **font)
    month = 2
    df_sens = pd.read_csv(files_path_prefix + f"Func_repr/a-flux-monthly/{month}/sens_params_{months_names[month]}.csv",
                          sep=';')

    df_lat = pd.read_csv(files_path_prefix + f"Func_repr/a-flux-monthly/{month}/lat_params_{months_names[month]}.csv",
                          sep=';')
    if not os.path.exists(files_path_prefix + f"Func_repr/a-flux-monthly/{month}"):
        os.mkdir(files_path_prefix + f"Func_repr/a-flux-monthly/{month}")

    fig, axes = plt.subplots(1, 2, figsize=(25, 10))
    x = np.linspace(np.nanmin(sensible_array), np.nanmax(sensible_array), 100)

    for i in range(0, 43):
        sens_params = df_sens[['a', 'b', 'c', 'd']].loc[i].values
        lat_params = df_lat[['a', 'b', 'c', 'd']].loc[i].values
        if i > 30:
            color = plt.cm.tab20(i % 20)
        else:
            color = 'gray'
        axes[0].plot(x, func(x, *sens_params), label=f'{1979 + i}', c=color)
        axes[1].plot(x, func(x, *lat_params), label=f'{1979 + i}', c=color)

    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1.0),
          ncol=2, fancybox=True, shadow=True)
    axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1.0),
          ncol=2, fancybox=True, shadow=True)
    axes[0].set_title('Sensible')
    axes[1].set_title('Latent')
    fig.suptitle(months_names[month])
    fig.tight_layout()
    fig.savefig(files_path_prefix + f"Func_repr/a-flux-monthly/{months_names[month]}.png")

    raise ValueError
    # ---------------------------------------------------------------------------------------

    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    # days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days

    time_start = days_delta1 + days_delta2 + days_delta3
    time_end = 15598

    plot_step = 1
    delta = 2879 + 806

    a_timelist, b_timelist, c_timelist, f_timelist, borders = load_ABCF(files_path_prefix, time_start, time_end, load_c=True)
    # plot_ab_coefficients(files_path_prefix, a_timelist, b_timelist, borders, 0, time_end-time_start - delta, plot_step, start_pic_num=time_start + delta)
    # plot_f_coeff(files_path_prefix, f_timelist, borders, 0, len(f_timelist), plot_step)

    # count_c_coeff(files_path_prefix, a_timelist, b_timelist, 0, 14)
    # a_timelist, b_timelist, c_timelist, f_timelist, borders = load_ABCF(files_path_prefix, time_start + 1, time_end, load_c=True)
    # print('lol')
    plot_c_coeff(files_path_prefix, c_timelist, 0, time_end-time_start - delta, plot_step, start_pic_num=time_start + delta)

    # create_video(files_path_prefix, files_path_prefix+'videos/tmp-coeff/', 'A_', 'a_daily', 10)
    # create_video(files_path_prefix, files_path_prefix+'videos/tmp-coeff/', 'B_', 'b_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/tmp-coeff/', 'F_', 'f_daily', 10)
    # create_video(files_path_prefix, files_path_prefix + 'videos/tmp-coeff/', 'C_', 'c_daily', 10)

    # count_correlation_fluxes(files_path_prefix, 0, 1829)
    # plot_flux_correlations(files_path_prefix, 0, 1829, step=7)
    # create_video(files_path_prefix, files_path_prefix + 'videos/Flux-corr/', 'FL_corr_', 'flux_correlation_weekly', 10)

    # plot_a_by_flux(files_path_prefix, a_timelist, borders, sensible_array, latent_array, 0, len(a_timelist))
