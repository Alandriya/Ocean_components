import matplotlib.pyplot as plt
import matplotlib
import tqdm
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_a_by_flux(files_path_prefix, a_timelist, borders, sensible_array, latent_array, time_start, time_end, step=1):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    pic_num = 0

    for t in tqdm.tqdm(range(time_start, time_end, step)):
        sens_set = set(sensible_array[:, t])
        lat_set = set(latent_array[:, t])

        date = datetime.datetime(1979, 1, 1, 0, 0) + datetime.timedelta(hours=6 * (62396 - 7320) + t * 24)
        a_sens = a_timelist[t][0]
        a_lat = a_timelist[t][1]
        sens_list, lat_list = list(), list()
        sens_x, lat_x = list(), list()
        for val in sorted(list(sens_set)):
            if not np.isnan(val):
                points_sensible = np.nonzero(sensible_array[:, t] == val)[0]
                sens_list.append(a_sens[points_sensible[0]//181, points_sensible[0] % 181])
                sens_x.append(val)

        for val in sorted(list(lat_set)):
            if not np.isnan(val):
                points_latent = np.nonzero(latent_array[:, t] == val)[0]
                lat_list.append(a_lat[points_latent[0]//181, points_latent[0] % 181])
                lat_x.append(val)

        sens_list = [x for _, x in sorted(zip(sens_x, sens_list))]
        sens_x = list(sorted(sens_x))
        lat_list = [x for _, x in sorted(zip(lat_x, lat_list))]
        lat_x = list(sorted(lat_x))

        fig.suptitle(f'A-flux value dependence \n {date.strftime("%Y-%m-%d")}', fontsize=25)
        axs[0].cla()
        axs[0].plot(sens_x, sens_list, c='r')
        axs[0].set_ylabel('A_sens', fontsize=20)
        axs[0].set_xlabel('Sensible flux', fontsize=20)
        axs[0].set_ylim([borders[0], borders[1]])

        axs[1].cla()
        axs[1].plot(lat_x, lat_list, c='orange')
        axs[1].set_ylabel('A_lat', fontsize=20)
        axs[1].set_xlabel('Latent flux', fontsize=20)
        axs[1].set_ylim([borders[0], borders[1]])
        fig.savefig(files_path_prefix + f'Func_repr/a-flux/{t:05d}.png')
    return
