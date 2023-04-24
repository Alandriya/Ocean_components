import pandas as pd
import numpy as np
import os
import math
import shutil
from data_processing import scale_to_bins
from EM_hybrid import *
import tqdm
from plot_compare import *

# width = 181
# height = 161

mu = 0.01
sigma = 0.015
x0 = 1


def count_Bel_Kor_difference(files_path_prefix: str,
                             time_start: int,
                             time_end: int,
                             point: tuple,
                             n_components: int,
                             window_width: int,
                             ticks_by_day: int = 1,
                             step_ticks: int = 1,
                             timedelta: int = 0,
                             group: int = None,
                             flux_type: str = 'sensible',
                             Bel_halfwindow: int = 0,
                             ):
    # if not os.path.exists(files_path_prefix + f'Components/{flux_type}/Bel'):
    #     os.mkdir(files_path_prefix + f'Components/{flux_type}/Bel')
    if not os.path.exists(files_path_prefix + f'Synthetic/Bel'):
        os.mkdir(files_path_prefix + f'Synthetic/Bel')

    # if True or not os.path.exists(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-A.npy'):
    if True or not os.path.exists(files_path_prefix + f'Synthetic/Bel/point_({point[0]}, {point[1]})-A.npy'):
        a_Bel = np.zeros(time_end - time_start + 2*Bel_halfwindow)
        for t in range(timedelta + time_start-Bel_halfwindow, timedelta + time_end - window_width // ticks_by_day +
                       Bel_halfwindow):
            if flux_type == 'sensible':
                a_arr = np.load(files_path_prefix + f'Synthetic/coeff_Bel/A_sens_{t}.npy')
            elif flux_type == 'latent':
                a_arr = np.load(files_path_prefix + f'Synthetic/coeff_Bel/{t}_A_lat.npy')
            else:
                a_arr = np.load(files_path_prefix + f'Synthetic/coeff_Bel/A_{t}.npy')
            # a_Bel[t - time_start - timedelta + Bel_halfwindow] = sum([a_arr[p[0], p[1]] for p in point_bigger]) / point_size
            a_Bel[t - time_start - timedelta] = a_arr[point[0], point[1]]
            del a_arr


        a_Bel_mean = np.zeros(time_end - time_start)
        for i in range(len(a_Bel_mean)):
            a_Bel_mean[i] = np.mean(a_Bel[i:i+2*Bel_halfwindow + 1])
        del a_Bel

        a_Bel = a_Bel_mean
        # a_Bel = np.diff(a_Bel)

        # np.save(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-A.npy', a_Bel)
        np.save(files_path_prefix + f'Synthetic/Bel/point_({point[0]}, {point[1]})-A.npy', a_Bel)
    else:
        raise ValueError
        a_Bel = np.load(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-A.npy')

    # if True or not os.path.exists(
    #         files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-B.npy'):
    #     b_Bel = np.zeros(time_end - time_start)
    #     for t in range(timedelta + time_start, timedelta + time_end - window_width // ticks_by_day):
    #         if flux_type == 'sensible':
    #             # b_arr = np.load(files_path_prefix + f'Coeff_data/{t}_B.npy')[0]
    #             b_arr = np.load(files_path_prefix + f'Synthetic/coeff_Bel/B_{t}.npy')[0]
    #         else:
    #             b_arr = np.load(files_path_prefix + f'Coeff_data/{t}_B.npy')
    #         b_Bel[t - time_start - timedelta] = sum([b_arr[p[0], p[1]] for p in point_bigger]) / point_size
    #         # b_Bel[t - time_start - timedelta] = b_arr[point[0], point[1]]
    #         del b_arr
    #     # b_Bel = np.diff(b_Bel)
    #     # np.save(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-B.npy', b_Bel)
    #     np.save(files_path_prefix + f'Synthetic/Bel/point_({point[0]}, {point[1]})-B.npy', b_Bel)
    # else:
    #     b_Bel = np.load(files_path_prefix + f'Components/{flux_type}/Bel/point_({point[0]}, {point[1]})-B.npy')

    # if not os.path.exists(files_path_prefix + f'Components/{flux_type}/Sum'):
    #     os.mkdir(files_path_prefix + f'Components/{flux_type}/Sum')

    if not os.path.exists(files_path_prefix + f'Synthetic/Sum'):
        os.mkdir(files_path_prefix + f'Synthetic/Sum')

    if True or not os.path.exists(
            files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-A.npy'):
        # Kor_df = pd.read_excel(files_path_prefix + f'Components/{flux_type}/components-xlsx/point_({point[0]}, {point[1]}).xlsx')
        Kor_df = pd.read_excel(files_path_prefix + f'Synthetic/coeff_Kor/Components/components-xlsx/point_({point[0]}, {point[1]}).xlsx')
        Kor_df.fillna(0, inplace=True)
        a_sum = np.zeros(len(Kor_df))
        for i in range(1, n_components + 1):
            a_sum += Kor_df[f'mean_{i}'] * Kor_df[f'weight_{i}']
    else:
        a_sum = np.load(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-A.npy')

    a_sum = np.array(a_sum[:len(a_sum) - len(a_sum) % (ticks_by_day // step_ticks)])
    a_sum = np.mean(a_sum.reshape(-1, (ticks_by_day // step_ticks)), axis=1)
    # a_sum /= window_width/ticks_by_day
    # np.save(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-A.npy', a_sum)
    np.save(files_path_prefix + f'Synthetic/coeff_Kor/Components/Sum/point_({point[0]}, {point[1]})-A.npy', a_sum)

    a_diff = a_Bel[:len(a_sum)] - a_sum
    # if not os.path.exists(files_path_prefix + f'Components/{flux_type}/difference'):
    #     os.mkdir(files_path_prefix + f'Components/{flux_type}/difference')
    # np.save(files_path_prefix + f'Components/{flux_type}/difference/point_({point[0]}, {point[1]})-A.npy', a_diff)

    if not os.path.exists(files_path_prefix + f'Synthetic/difference'):
        os.mkdir(files_path_prefix + f'Synthetic/difference')
    np.save(files_path_prefix + f'Synthetic/difference/point_({point[0]}, {point[1]})-A.npy', a_diff)


    # if True or not os.path.exists(
    #         files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-B.npy'):
    #     # Kor_df = pd.read_excel(files_path_prefix + f'Components/{flux_type}/components-xlsx/point_({point[0]}, {point[1]}).xlsx')
    #     Kor_df = pd.read_excel(files_path_prefix + f'Synthetic/coeff_Kor/Components/components-xlsx/point_({point[0]}, {point[1]}).xlsx')
    #     Kor_df.fillna(0, inplace=True)
    #     b_sum = np.zeros(len(Kor_df))
    #
    #     a_sum = np.zeros(len(Kor_df))
    #     for i in range(1, n_components + 1):
    #         a_sum += Kor_df[f'mean_{i}'] * Kor_df[f'weight_{i}']
    #
    #     for i in range(1, n_components + 1):
    #         b_sum += Kor_df[f'weight_{i}'] * (np.square(Kor_df[f'mean_{i}']) + np.square(Kor_df[f'sigma_{i}']))
    #
    #     b_sum = np.sqrt(b_sum - np.square(a_sum))
    # else:
    #     b_sum = np.load(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-B.npy')
    #
    # b_sum = np.array(b_sum[:len(b_sum) - len(b_sum) % (ticks_by_day // step_ticks)])
    # b_sum = np.mean(b_sum.reshape(-1, (ticks_by_day // step_ticks)), axis=1)
    # # b_sum /= window_width/ticks_by_day
    # # np.save(files_path_prefix + f'Components/{flux_type}/Sum/point_({point[0]}, {point[1]})-B.npy', b_sum)
    # np.save(files_path_prefix + f'Synthetic/Sum/point_({point[0]}, {point[1]})-B.npy', b_sum)
    # b_diff = b_Bel[:len(b_sum)] - b_sum
    # # if not os.path.exists(files_path_prefix + f'Components/{flux_type}/difference'):
    # #     os.mkdir(files_path_prefix + f'Components/{flux_type}/difference')
    # # np.save(files_path_prefix + f'Components/{flux_type}/difference/point_({point[0]}, {point[1]})-B.npy', b_diff)
    # if not os.path.exists(files_path_prefix + f'Synthetic/difference'):
    #     os.mkdir(files_path_prefix + f'Synthetic/difference')
    # np.save(files_path_prefix + f'Synthetic/difference/point_({point[0]}, {point[1]})-B.npy', b_diff)
    return


def create_synthetic_data_1d(files_path_prefix: str,
                          width: int = 100,
                          height: int = 100,
                          time_start: int = 0,
                          time_end: int = 1,):
    """
    Creates 1-dimensional arrays of a and b coefficients and the corresponding fluxe as solution of the Langevin equation
    :param files_path_prefix: path to the working directory
    :param width: width of the map
    :param height: height of the map
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :return:
    """
    wiener = np.zeros((height, width), dtype=float)
    # X_start = np.ones((height, width), dtype=float)
    X_start = np.random.normal(1, 1, size=(height, width))
    alpha = -0.3
    beta = 0.1
    omega = [math.pi/2, math.pi, math.pi * 2]
    weights = [0.5, 0.3, 0.2]

    # omega = [math.pi/2]
    # weights = [1.0]

    X = np.zeros((time_end - time_start, height, width), dtype=float)
    X[0] = X_start

    a = np.zeros((time_end - time_start-1, height, width), dtype=float)
    b = np.zeros((time_end - time_start-1, height, width), dtype=float)
    for t in range(time_start + 1, time_end):
        normal = np.random.normal(0, 1, size=(height, width))
        wiener += normal

        dX = np.zeros((height, width), dtype=float)
        for k in range(len(omega)):
            a[t - 1] += math.cos(omega[k]*t) * weights[k] * alpha * X[t-1]
            b[t - 1] += math.cos(omega[k]*t) * weights[k] * beta * X[t-1]
            dX = a[t-1] + b[t-1]

        X[t] = X[t-1] + dX

    np.save(f'{files_path_prefix}/Synthetic/flux_full.npy', X)
    np.save(f'{files_path_prefix}/Synthetic/B_full.npy', b)
    np.save(f'{files_path_prefix}/Synthetic/A_full.npy', a)
    return



def create_synthetic_data_2d(files_path_prefix: str,
                          width: int = 100,
                          height: int = 100,
                          time_start: int = 0,
                          time_end: int = 1):
    """
    Creates 2-dimensional array of a and b coefficients and the corresponding fluxes as solution of the Langevin equation
    :param files_path_prefix: path to the working directory
    :param width: width of the map
    :param height: height of the map
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :return:
    """
    wiener = np.zeros((height, width), dtype=float)
    wiener_2 = np.zeros((height, width), dtype=float)

    sensible_full = np.zeros((time_end - time_start, height, width), dtype=float)
    latent_full = np.zeros((time_end - time_start, height, width), dtype=float)
    a_full = np.zeros((time_end - time_start, 2, height, width), dtype=float)
    b_full = np.zeros((time_end - time_start, 4, height, width), dtype=float)
    for t in range(time_start, time_end):
        normal = np.random.normal(0, 1, size=(height, width))
        wiener += normal

        normal = np.random.normal(0, 1, size=(height, width))
        wiener_2 += normal

        sensible_full[t] = x0 * np.exp((mu - sigma*sigma/2)*t + sigma*wiener)
        latent_full[t] = x0 * np.exp((mu - sigma*sigma/2)*t + sigma*wiener_2)
        a_full[t, 0] = mu * sensible_full[t]
        a_full[t, 1] = mu * latent_full[t]
        b_full[t, 0] = sigma * sensible_full[t]
        b_full[t, 1] = sigma * latent_full[t]

        # plt.hist(sensible_extended[1].flatten(), bins=30)
        # plt.show()

        # np.save(f'{files_path_prefix}/Synthetic/X_{t}.npy', X)
        # np.save(f'{files_path_prefix}/Synthetic/B_{t}.npy', b)
        # np.save(f'{files_path_prefix}/Synthetic/A_{t}.npy', a)

    plt.hist(sensible_full[:, 0, 0].flatten(), bins=30)
    plt.show()

    np.save(f'{files_path_prefix}/Synthetic/sensible_full.npy', sensible_full)
    np.save(f'{files_path_prefix}/Synthetic/latent_full.npy', latent_full)
    np.save(f'{files_path_prefix}/Synthetic/B_full.npy', b_full)
    np.save(f'{files_path_prefix}/Synthetic/A_full.npy', a_full)
    return


def multiply_synthetic_Korolev(files_path_prefix: str,
                               dimensions: int,):
    if dimensions == 2:
        sensible = np.load(f'{files_path_prefix}/Synthetic/sensible_full.npy')
        multiply_amount = 100
        sensible_extended = np.zeros((sensible.shape[0]*multiply_amount, sensible.shape[1], sensible.shape[2]))

        for t in range(sensible.shape[0]):
            current = np.copy(sensible[t])
            # plt.hist(current.flatten())
            # plt.show()
            sensible_extended[t * multiply_amount] = current
            for t1 in range(1, multiply_amount):
                noised = current + np.random.normal(0, 1, size=current.shape)
                sensible_extended[t * multiply_amount + t1] = noised
                # plt.hist(noised.flatten())
                # plt.show()

        # plt.hist(sensible_extended[1].flatten(), bins=30)
        # plt.show()

        np.save(f'{files_path_prefix}/Synthetic/sensible_full_extended.npy', sensible_extended)
    else:
        flux = np.load(f'{files_path_prefix}/Synthetic/flux_full.npy')
        multiply_amount = 30
        flux_extended = np.zeros((flux.shape[0] * multiply_amount, flux.shape[1], flux.shape[2]))

        for t in range(flux.shape[0]):
            current = np.copy(flux[t])
            # plt.hist(current.flatten())
            # plt.show()
            flux_extended[t * multiply_amount] = current
            for t1 in range(1, multiply_amount):
                noised = current + np.random.normal(0, 0.05, size=current.shape)
                flux_extended[t * multiply_amount + t1] = noised
                # plt.hist(noised.flatten())
                # plt.show()

        # plt.hist(flux_extended[1].flatten(), bins=30)
        # plt.show()
        np.save(f'{files_path_prefix}/Synthetic/flux_full_extended.npy', flux_extended)

    return


def count_synthetic_Bel(files_path_prefix: str,
                        dimensions: int,
                        flux: np.array,
                        sensible: np.array,
                        latent: np.array,
                        time_start: int,
                        time_end: int,
                        width: int = 100,
                        height: int = 100):
    """
    Counts estimation of a and b coefficients by Belyaev method from time_start to time_end index and saves them to
    files_path_prefix + 'Synthetic/coeff_Bel'
    :param files_path_prefix: path to the working directory
    :param sensible: np.array with shape (time_end - time_start, width, height) with imitated flux data
    :param latent: np.array with shape (time_end - time_start, width, height) with imitated flux data
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param width: width of the map
    :param height: height of the map
    :return:
    """
    if os.path.exists(files_path_prefix + 'Synthetic/coeff_Bel'):
        shutil.rmtree(files_path_prefix + 'Synthetic/coeff_Bel')
    os.mkdir(files_path_prefix + 'Synthetic/coeff_Bel')

    if dimensions == 2:
        # quantification
        sensible_array, sens_quantiles = scale_to_bins(sensible, 100)
        latent_array, lat_quantiles = scale_to_bins(latent, 100)

        a_sens = np.zeros((height, width), dtype=float)
        a_lat = np.zeros((height, width), dtype=float)
        b_matrix = np.zeros((4, height, width), dtype=float)

        for t in range(time_start+1, time_end):
            set_sens = np.unique(sensible_array[t - 1])
            set_lat = np.unique(latent_array[t - 1])

            for val_t0 in set_sens:
                if not np.isnan(val_t0):
                    points_sensible = np.where(sensible_array[t - 1] == val_t0)
                    amount_t0 = len(points_sensible[0])

                    # sensible t0 - sensible t1
                    set_t1 = np.unique(sensible_array[t][points_sensible])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = len(np.where(sensible_array[t][points_sensible] == val_t1)[0]) * 1.0 / amount_t0
                        probabilities.append(prob)

                    a = sum([(list(set_t1)[i] - val_t0) * probabilities[i] for i in range(len(probabilities))])
                    b_squared = sum(
                        [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

                    a_sens[points_sensible] = a
                    b_matrix[0][points_sensible] = np.sqrt(b_squared)

                    # sensible t0 - latent t1
                    set_t1 = np.unique(latent_array[t][points_sensible])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = len(np.where(latent_array[t][points_sensible] == val_t1)) * 1.0 / amount_t0
                        probabilities.append(prob)
                    b_squared = sum(
                        [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

                    b_matrix[1][points_sensible] = np.sqrt(b_squared)

            for val_t0 in set_lat:
                if not np.isnan(val_t0):
                    points_latent = np.where(latent_array[t - 1] == val_t0)
                    amount_t0 = len(points_latent[0])

                    # latent - latent
                    set_t1 = np.unique(latent_array[t][points_latent])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = len(np.where(latent_array[t][points_latent] == val_t1)[0]) * 1.0 / amount_t0
                        probabilities.append(prob)

                    a = sum([(set_t1[i] - val_t0) * probabilities[i] for i in range(len(probabilities))])
                    b_squared = sum(
                        [(set_t1[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

                    a_lat[points_latent] = a
                    b_matrix[3][points_latent] = np.sqrt(b_squared)

                    # latent t0 - sensible t1
                    set_t1 = np.unique(sensible_array[t][points_latent])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = len(np.where(sensible_array[t][points_latent] == val_t1)[0]) * 1.0 / amount_t0
                        probabilities.append(prob)

                    b_squared = sum(
                        [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in range(len(probabilities))]) - a ** 2

                    b_matrix[2][points_latent] = np.sqrt(b_squared)

            # # get matrix root from B and count F
            # for i in range(161):
            #     for j in range(181):
            #         if not np.isnan(b_matrix[:, i, j]).any():
            #             b_matrix[:, i, j] = sqrtm(b_matrix[:, i, j].reshape(2, 2)).reshape(4)

            np.save(files_path_prefix + f'Synthetic/coeff_Bel/A_sens_{t}', a_sens)
            np.save(files_path_prefix + f'Synthetic/coeff_Bel/A_lat_{t}', a_lat)
            np.save(files_path_prefix + f'Synthetic/coeff_Bel/B_{t}', b_matrix)
    else:
        # quantification
        flux, flux_quantiles = scale_to_bins(flux, 100)

        a = np.zeros((height, width), dtype=float)
        b = np.zeros((height, width), dtype=float)

        for t in range(time_start + 1, time_end):
            set_0 = np.unique(flux[t - 1])
            for val_t0 in set_0:
                if not np.isnan(val_t0):
                    points = np.where(flux[t - 1] == val_t0)
                    amount_t0 = len(points[0])

                    set_t1 = np.unique(flux[t][points])
                    probabilities = list()
                    for val_t1 in set_t1:
                        prob = len(np.where(flux[t][points] == val_t1)[0]) * 1.0 / amount_t0
                        probabilities.append(prob)

                    a_part = sum([(list(set_t1)[i] - val_t0) * probabilities[i] for i in range(len(probabilities))])
                    b_squared = sum(
                        [(list(set_t1)[i] - val_t0) ** 2 * probabilities[i] for i in
                         range(len(probabilities))]) - a_part ** 2
                    a[points] = a_part
                    b[points] = np.sqrt(b_squared)

            np.save(files_path_prefix + f'Synthetic/coeff_Bel/A_{t}', a)
            np.save(files_path_prefix + f'Synthetic/coeff_Bel/B_{t}', b)
    return


def count_synthetic_Korolev(files_path_prefix: str,
                            flux_type: str,
                            flux_data: np.ndarray,
                            time_start: int,
                            time_end: int,
                            window_width: int = 100,
                            n_components: int = 2,
                        ):
    """
    Counts estimation of a and b coefficients by Korolev method from time_start to time_end index and saves them to
    files_path_prefix + 'Synthetic/coeff_Kor'
    :param files_path_prefix: path to the working directory
    :param sensible: np.array with shape (time_end - time_start, width, height) with imitated flux data
    :param latent: np.array with shape (time_end - time_start, width, height) with imitated flux data
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param window_width: window width
    :param n_components: amount of components in the approximating mixture
    :return:
    """
    # if os.path.exists(files_path_prefix + 'Synthetic/coeff_Kor'):
    #     shutil.rmtree(files_path_prefix + 'Synthetic/coeff_Kor')
    # os.mkdir(files_path_prefix + 'Synthetic/coeff_Kor')
    # os.mkdir(files_path_prefix + 'Synthetic/coeff_Kor/Components')
    # os.mkdir(files_path_prefix + 'Synthetic/coeff_Kor/Components/raw')
    # os.mkdir(files_path_prefix + 'Synthetic/coeff_Kor/Components/components-xlsx')
    # os.mkdir(files_path_prefix + 'Synthetic/coeff_Kor/Components/plots')
    # os.mkdir(files_path_prefix + 'Synthetic/coeff_Kor/Components/Sum')

    quantiles_amount = 400

    draw = True
    if os.path.exists(files_path_prefix + 'Synthetic/Plots/Difference'):
        shutil.rmtree(files_path_prefix + 'Synthetic/Plots/Difference')
        os.mkdir(files_path_prefix + 'Synthetic/Plots/Difference')

    if not os.path.exists(files_path_prefix + 'Synthetic/flux_quantiles.npy'):
        print('Getting quantiles...')
        flux_array, quantiles = scale_to_bins(flux_data, quantiles_amount)
        np.save(files_path_prefix + 'Synthetic/flux_quantiles.npy', flux_array)
        np.save(files_path_prefix + 'Synthetic/quantiles.npy', np.array(quantiles))
        flux_set = list(set(flux_array.flat))
        np.save(files_path_prefix + 'Synthetic/values.npy', np.array(flux_set))
    else:
        flux_array = np.load(files_path_prefix + 'Synthetic/flux_quantiles.npy')
        flux_set = np.load(files_path_prefix + 'Synthetic/values.npy')

    if not os.path.exists(files_path_prefix + 'Synthetic/coeff_Kor/M.npy'):
        M = np.zeros((quantiles_amount, time_end-time_start), dtype=int)
        groups_array = np.zeros_like(flux_data, dtype=int)
        print('filling M matrix...')
        for t in range(time_end-time_start):
            for group in range(1, quantiles_amount+1):
                value = flux_set[group-1]
                groups_array[t][np.where(flux_array[t] == value)] = group
                amount_group = len(flux_data[np.where(flux_array[t] == value)].flat)
                M[group-1, t] = amount_group

        np.save(files_path_prefix + 'Synthetic/coeff_Kor/M.npy', M)
        np.save(files_path_prefix + f'Synthetic/coeff_Kor/Components/groups_array.npy', groups_array)

        a_map = np.zeros_like(flux_data, dtype=float)
        b_map = np.zeros_like(flux_data, dtype=float)
    else:
        M = np.load(files_path_prefix + 'Synthetic/coeff_Kor/M.npy')
        groups_array = np.load(files_path_prefix + f'Synthetic/coeff_Kor/Components/groups_array.npy')
        a_map = np.load(files_path_prefix + f'Synthetic/coeff_Kor/Components/A_map.npy')
        b_map = np.load(files_path_prefix + f'Synthetic/coeff_Kor/Components/B_map.npy')

    for group in tqdm.tqdm(range(1, quantiles_amount+1)):
        if os.path.exists(files_path_prefix + f'Synthetic/coeff_Kor/Components/Sum/group_{group}-A.npy'):
            continue
        sample = list()
        step_list = list()
        for t in range(time_start, time_end):
            if M[group, t]:
                day_sample = flux_data[np.where(groups_array[t] == group)].flat
                sample += list(day_sample)
                step_list.append(len(day_sample))

        point_df = hybrid(sample, window_width, n_components, EM_steps=1, step_list=step_list)
        point_df.to_excel(files_path_prefix + f'Synthetic/coeff_Kor/Components/raw/group_{group}.xlsx', index=False)
        del point_df

        df = pd.read_excel(files_path_prefix + f'Synthetic/coeff_Kor/Components/raw/group_{group}.xlsx')
        new_df, new_n_components = cluster_components(df,
                                                      n_components,
                                                      files_path_prefix,
                                                      draw=False,
                                                      path=f'Synthetic/coeff_Kor/Components/plots/',
                                                      postfix=f'_group_{group}')

        new_df.to_excel(files_path_prefix + f'Synthetic/coeff_Kor/Components/components-xlsx/group_{group}.xlsx', index=False)
        del df

        Kor_df = new_df
        Kor_df.fillna(0, inplace=True)
        a_sum = np.zeros(time_end - time_start, dtype=float)
        b_sum = np.zeros(time_end - time_start, dtype=float)

        row = 0
        for t in range(time_end - time_start):
            if M[group, t]:
                for k in range(1, n_components + 1):
                    a_sum[t] += Kor_df[f'mean_{k}'][row] * Kor_df[f'weight_{k}'][row]
                    b_sum[t] += math.sqrt(Kor_df[f'weight_{k}'][row] * (np.square(Kor_df[f'mean_{k}'][row]) +
                                                                      np.square(Kor_df[f'sigma_{k}'][row])))
                row += 1
            else:
                a_sum[t] = None
                b_sum[t] = None

        del Kor_df
        np.save(files_path_prefix + f'Synthetic/coeff_Kor/Components/Sum/group_{group}-A.npy', a_sum)
        np.save(files_path_prefix + f'Synthetic/coeff_Kor/Components/Sum/group_{group}-B.npy', b_sum)

        print(f'Filling maps for group {group}')
        for t in tqdm.tqdm(range(time_end-time_start)):
            a_map[t][np.where(groups_array[t] == group)] = a_sum[t]
            b_map[t][np.where(groups_array[t] == group)] = b_sum[t]

        np.save(files_path_prefix + f'Synthetic/coeff_Kor/Components/A_map.npy', a_map)
        np.save(files_path_prefix + f'Synthetic/coeff_Kor/Components/B_map.npy', b_map)

        if draw:
            plot_group(files_path_prefix,
                       group,
                       a_sum,
                       b_sum)

        del a_sum, b_sum
    return


def collect_Kor(files_path_prefix: str,
                width: int = 100,
                height: int = 100,
                time_start: int = 0,
                time_end: int = 1,
                ):
    flux_array = np.load(files_path_prefix + 'Synthetic/flux_quantiles.npy')
    flux_set = np.load(files_path_prefix + 'Synthetic/values.npy')
    map_array = np.zeros((time_end - time_start, height, width), dtype=float)
    for group in range(len(flux_set)):
        value = flux_set[group]
        group = np.load(files_path_prefix + f'Synthetic/coeff_Kor/Components/Sum/group_{group}-A.npy')
        for t in range(flux_array.shape[0]):
            points = np.where(flux_array[t] == value)
            map_array[t, points] = group[t]
    np.save(files_path_prefix + f'Synthetic/coeff_Kor/Components/full_array_A.npy', map_array)
    return
