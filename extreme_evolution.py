import os.path
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import matplotlib
import seaborn as sns
from sklearn.metrics import r2_score


def extract_extreme(files_path_prefix, timelist, coeff_type, time_start, time_end, mean_days=1):
    max_sens, min_sens, mean_sens, med_sens = list(), list(), list(), list()
    max_sens_points, min_sens_points = list(), list()
    max_lat, min_lat, mean_lat, med_lat = list(), list(), list(), list()
    max_lat_points, min_lat_points = list(), list()
    for t in tqdm.tqdm(range(0, time_end-time_start)):
        sens = timelist[t][0]
        max_sens.append(np.nanmax(sens))
        max_sens_points.append(np.nanargmax(sens))
        min_sens.append(np.nanmin(sens))
        min_sens_points.append(np.nanargmin(sens))
        mean_sens.append(np.nanmean(sens))
        med_sens.append(np.nanmedian(sens))

        lat = timelist[t][1]
        max_lat.append(np.nanmax(lat))
        max_lat_points.append(np.nanargmax(lat))
        min_lat.append(np.nanmin(lat))
        min_lat_points.append(np.nanargmin(lat))
        mean_lat.append(np.nanmean(lat))
        med_lat.append(np.nanmedian(lat))

    # take mean by the window with width = mean_days
    max_sens = [np.mean(max_sens[i:i+mean_days]) for i in range(0, len(max_sens)-mean_days, mean_days)]
    min_sens = [np.mean(min_sens[i:i + mean_days]) for i in range(0, len(min_sens) - mean_days, mean_days)]
    mean_sens = [np.mean(mean_sens[i:i + mean_days]) for i in range(0, len(mean_sens) - mean_days, mean_days)]
    med_sens = [np.mean(med_sens[i:i + mean_days]) for i in range(0, len(med_sens) - mean_days, mean_days)]

    max_lat = [np.mean(max_lat[i:i+mean_days]) for i in range(0, len(max_lat)-mean_days, mean_days)]
    min_lat = [np.mean(min_lat[i:i + mean_days]) for i in range(0, len(min_lat) - mean_days, mean_days)]
    mean_lat = [np.mean(mean_lat[i:i + mean_days]) for i in range(0, len(mean_lat) - mean_days, mean_days)]
    med_lat = [np.mean(med_lat[i:i + mean_days]) for i in range(0, len(med_lat) - mean_days, mean_days)]

    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy', np.array(max_sens))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy', np.array(min_sens))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_mean_sens({time_start}-{time_end})_{mean_days}.npy', np.array(mean_sens))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_med_sens({time_start}-{time_end})_{mean_days}.npy', np.array(med_sens))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_max_points_sens.npy', np.array(max_sens_points))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_min_points_sens.npy', np.array(min_sens_points))

    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_max_lat({time_start}-{time_end})_{mean_days}.npy', np.array(max_lat))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_min_lat({time_start}-{time_end})_{mean_days}.npy', np.array(min_lat))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_mean_lat({time_start}-{time_end})_{mean_days}.npy', np.array(mean_lat))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_med_lat({time_start}-{time_end})_{mean_days}.npy', np.array(med_lat))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_max_points_lat.npy', np.array(max_lat_points))
    np.save(files_path_prefix + f'Extreme/data/{coeff_type}_min_points_lat.npy', np.array(min_lat_points))
    return


def plot_extreme(files_path_prefix, coeff_type, time_start, time_end, mean_days=1):
    font = {'size': 20}
    font_names = {'weight': 'bold', 'size': 24}
    matplotlib.rc('font', **font)
    sns.set_style("whitegrid")

    days = [datetime.datetime(1979, 1, 1) + datetime.timedelta(days=t) for t in range(time_start, time_end - mean_days, mean_days)]
    if os.path.exists(files_path_prefix + f'Extreme/data/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy'):
        max_sens = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy')
        min_sens = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy')
        mean_sens = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_mean_sens({time_start}-{time_end})_{mean_days}.npy')
        # med_sens = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_med_sens({time_start}-{time_end})_{mean_days}.npy')

        max_lat = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_max_lat({time_start}-{time_end})_{mean_days}.npy')
        min_lat = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_min_lat({time_start}-{time_end})_{mean_days}.npy')
        mean_lat = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_mean_lat({time_start}-{time_end})_{mean_days}.npy')
        # med_lat = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_med_lat({time_start}-{time_end})_{mean_days}.npy')

        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        # Major ticks every half year, minor ticks every month,
        # axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
        axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
        # axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
        axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
        axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
        axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))

        axs[0].set_title('Sensible', fontdict=font_names)
        axs[0].plot(days, max_sens, label='max', c='r')
        axs[0].plot(days, min_sens, label='min', c='b')
        axs[0].plot(days, mean_sens, label='mean', c='g')
        # axs[0].plot(days, med_sens, label='med', c='y')
        axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        axs[1].set_title('Latent', fontdict=font_names)
        axs[1].plot(days, max_lat, label='max', c='r')
        axs[1].plot(days, min_lat, label='min', c='b')
        axs[1].plot(days, mean_lat, label='mean', c='g')
        # axs[1].plot(days, med_lat, label='med', c='y')
        axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'Extreme/plots/{coeff_type}_({time_start}-{time_end})_{mean_days}.png')
        plt.close(fig)

        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
        axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
        axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))
        axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))
        x = range(time_start, time_end - mean_days, mean_days)

        axs[0].set_title('Sensible', fontdict=font_names)
        poly = np.poly1d(np.polyfit(x, max_sens, 1))
        axs[0].plot(days, poly(x), label=f'max, k = {poly.coeffs[0]:.5f},\nR^2 = {r2_score(max_sens, poly(x)):.5f}', c='r')
        poly = np.poly1d(np.polyfit(x, min_sens, 1))
        axs[0].plot(days, poly(x), label=f'min, k = {poly.coeffs[0]:.5f},\nR^2 = {r2_score(min_sens, poly(x)):.5f}', c='b')
        poly = np.poly1d(np.polyfit(x, mean_sens, 1))
        axs[0].plot(days, poly(x), label=f'mean, k = {poly.coeffs[0]:.5f},\nR^2 = {r2_score(mean_sens, poly(x)):.5f}', c='g')
        # poly = np.poly1d(np.polyfit(x, med_sens, 1))
        # axs[0].plot(days, poly(x), label=f'med, {poly.coeffs[0]:.5f}', c='y')
        axs[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        axs[1].set_title('Latent', fontdict=font_names)
        poly = np.poly1d(np.polyfit(x, max_lat, 1))
        axs[1].plot(days, poly(x), label=f'max, k = {poly.coeffs[0]:.5f},\nR^2 = {r2_score(max_lat, poly(x)):.5f}', c='r')
        poly = np.poly1d(np.polyfit(x, min_lat, 1))
        axs[1].plot(days, poly(x), label=f'min, {poly.coeffs[0]:.5f},\nR^2 = {r2_score(min_lat, poly(x)):.5f}', c='b')
        poly = np.poly1d(np.polyfit(x, mean_lat, 1))
        axs[1].plot(days, poly(x), label=f'mean, {poly.coeffs[0]:.5f},\nR^2 = {r2_score(mean_lat, poly(x)):.5f}', c='g')
        # poly = np.poly1d(np.polyfit(x, med_lat, 1))
        # axs[1].plot(days, poly(x), label=f'med, {poly.coeffs[0]:.5f}', c='y')
        axs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        fig.tight_layout()
        fig.savefig(files_path_prefix + f'Extreme/plots/{coeff_type}_({time_start}-{time_end})_{mean_days}_regression.png')
    return


def check_conditions(files_path_prefix, time_start, time_end, sensible_all, latent_all, mask):
    mean_days = 1
    if os.path.exists(files_path_prefix + f'Extreme/data/a_max_sens({time_start}-{time_end})_{mean_days}.npy'):
        coeff_type = 'a'
        a_max_sens = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy')
        a_min_sens = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy')
        a_max_lat = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_max_lat({time_start}-{time_end})_{mean_days}.npy')
        a_min_lat = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_min_lat({time_start}-{time_end})_{mean_days}.npy')

        coeff_type = 'b'
        b_max_sens = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_max_sens({time_start}-{time_end})_{mean_days}.npy')
        b_min_sens = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_min_sens({time_start}-{time_end})_{mean_days}.npy')
        b_max_lat = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_max_lat({time_start}-{time_end})_{mean_days}.npy')
        b_min_lat = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_min_lat({time_start}-{time_end})_{mean_days}.npy')

        coeff_type = 'a'
        a_max_sens_points = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_max_points_sens.npy')
        a_min_sens_points = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_min_points_sens.npy')
        a_max_lat_points = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_max_points_lat.npy')
        a_min_lat_points = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_min_points_lat.npy')

        coeff_type = 'b'
        b_max_sens_points = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_max_points_sens.npy')
        b_min_sens_points = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_min_points_sens.npy')
        b_max_lat_points = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_max_points_lat.npy')
        b_min_lat_points = np.load(files_path_prefix + f'Extreme/data/{coeff_type}_min_points_lat.npy')

        sens_vals = list(np.unique(sensible_all))
        sens_vals.sort()
        sens_delim = min([abs(sens_vals[i + 1] - sens_vals[i]) if sens_vals[i+1] != sens_vals[i] else 10
                          for i in range(0, len(sens_vals) - 1)])

        lat_vals = list(np.unique(latent_all))
        lat_vals.sort()
        lat_delim = min([abs(lat_vals[i + 1] - lat_vals[i]) if lat_vals[i+1] != lat_vals[i] else 10
                         for i in range(0, len(lat_vals) - 1)])

        K1_sens, K1_lat = 0.0, 0.0
        K2_sens, K2_lat = 0.0, 0.0
        for t in tqdm.tqdm(range(0, time_end - time_start - 1)):
            K1_sens = max(K1_sens, (a_max_sens[t] - a_min_sens[t] + b_max_sens[t] - b_min_sens[t]) / sens_delim)
            K1_lat = max(K1_lat, (a_max_lat[t] - a_min_lat[t] + b_max_lat[t] - b_min_lat[t]) / lat_delim)

            K2_sens = max(K2_sens, (a_max_sens[t]**2 + b_max_sens[t]**2))
            K2_lat = max(K2_lat, (a_max_lat[t]**2 + b_max_lat[t]**2))

        K_array = np.array([K1_sens, K1_lat, K2_sens, K2_lat])
        np.save(files_path_prefix + f'Extreme/data/K_estimates.npy', K_array)
        print(K_array)
        print(max(K_array))
    return
