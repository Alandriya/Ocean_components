import numpy as np

from data_processing import scale_to_bins
from EM_hybrid import *
from plot_compare import *


def collect_Bel_point(files_path_prefix: str,
                      time_start: int,
                      time_end: int,
                      point: tuple,
                      path: str = 'Synthetic/',
                      ):
    """
    Collects ar array for a specific point for Bel method from existing maps. Requires maps estimates at
    files_path_prefix + path + 'maps'.
    :param files_path_prefix: path to the working directory
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param point: tuple with point coordinates
    :param path: additional path to the folder from files_path_prefix, like 'Synthetic/', 'Components/sensible/',
    'Components/latent/'
    :return:
    """
    if not os.path.exists(files_path_prefix + path + 'Bel/points'):
        os.mkdir(files_path_prefix + path + 'Bel/points')

    if not os.path.exists(files_path_prefix + path + f'Bel/points/point_({point[0]}, {point[1]})-A.npy'):
        a_Bel = np.zeros(time_end - time_start)
        for t in range(time_start, time_end):
            a_arr = np.load(files_path_prefix + path + f'maps/A_{t}.npy')
            a_Bel[t - time_start] = a_arr[point[0], point[1]]
            del a_arr

        np.save(files_path_prefix + path + f'Bel/points/point_({point[0]}, {point[1]})-A.npy', a_Bel)

        b_Bel = np.zeros(time_end - time_start)
        for t in range(time_start, time_end):
            b_arr = np.load(files_path_prefix + path + f'maps/B_{t}.npy')
            b_Bel[t - time_start] = b_arr[point[0], point[1]]
            del b_arr

        np.save(files_path_prefix + path + f'Bel/points/point_({point[0]}, {point[1]})-B.npy', b_Bel)
    return


def create_synthetic_data_1d(files_path_prefix: str,
                             width: int = 100,
                             height: int = 100,
                             time_start: int = 0,
                             time_end: int = 1, ):
    """
    Creates 1-dimensional arrays of a and b coefficients and the corresponding fluxes as a solution of the
    Langevin equation
    :param files_path_prefix: path to the working directory
    :param width: width of the map
    :param height: height of the map
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :return:
    """
    wiener = np.zeros((height, width), dtype=float)
    X_start = np.random.normal(1, 1, size=(height, width))
    alpha = -0.3
    beta = 0.01
    omega = [math.pi / 2, math.pi * 2 / 3, math.pi * 4 / 3, math.pi * 2]
    weights = [0.4, 0.3, 0.2, 0.1]

    X = np.zeros((time_end - time_start, height, width), dtype=float)
    X[0] = X_start

    a = np.zeros((time_end - time_start - 1, height, width), dtype=float)
    b = np.zeros((time_end - time_start - 1, height, width), dtype=float)
    for t in range(time_start + 1, time_end):
        normal = np.random.normal(0, 1, size=(height, width))
        wiener += normal

        dX = np.zeros((height, width), dtype=float)
        for k in range(len(omega)):
            a[t - 1] += math.cos(omega[k] * t) * weights[k] * alpha * X[t - 1]
            b[t - 1] += math.cos(omega[k] * t) * weights[k] * beta * X[t - 1]
            dX = a[t - 1] + b[t - 1] * normal

        X[t] = X[t - 1] + dX

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

    mu = 0.01
    sigma = 0.015
    x0 = 1

    sensible_full = np.zeros((time_end - time_start, height, width), dtype=float)
    latent_full = np.zeros((time_end - time_start, height, width), dtype=float)
    a_full = np.zeros((time_end - time_start, 2, height, width), dtype=float)
    b_full = np.zeros((time_end - time_start, 4, height, width), dtype=float)
    for t in range(time_start, time_end):
        normal = np.random.normal(0, 1, size=(height, width))
        wiener += normal

        normal = np.random.normal(0, 1, size=(height, width))
        wiener_2 += normal

        sensible_full[t] = x0 * np.exp((mu - sigma * sigma / 2) * t + sigma * wiener)
        latent_full[t] = x0 * np.exp((mu - sigma * sigma / 2) * t + sigma * wiener_2)
        a_full[t, 0] = mu * sensible_full[t]
        a_full[t, 1] = mu * latent_full[t]
        b_full[t, 0] = sigma * sensible_full[t]
        b_full[t, 1] = sigma * latent_full[t]

    plt.hist(sensible_full[:, 0, 0].flatten(), bins=30)
    plt.show()

    np.save(f'{files_path_prefix}/Synthetic/sensible_full.npy', sensible_full)
    np.save(f'{files_path_prefix}/Synthetic/latent_full.npy', latent_full)
    np.save(f'{files_path_prefix}/Synthetic/B_full.npy', b_full)
    np.save(f'{files_path_prefix}/Synthetic/A_full.npy', a_full)
    return


def count_1d_Bel(files_path_prefix: str,
                 flux: np.ndarray,
                 time_start: int,
                 time_end: int,
                 path: str = 'Synthetic/',
                 start_index: int = 0,
                 quantiles_amount: int = 100,
                 mask: np.ndarray = None):
    """
    Counts A and B estimates for flux array by Belyaev method
    :param files_path_prefix: path to the working directory
    :param flux: np.array with shape [time_steps, height, width]
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param path: additional path to the folder from files_path_prefix, like 'Synthetic/', 'Components/sensible/',
    'Components/latent/'
    :param start_index: offset index when saving maps
    :param quantiles_amount: how many quantiles to use (for all data)
    :return:
    """
    if not os.path.exists(files_path_prefix + path):
        os.mkdir(files_path_prefix + path)

    if not os.path.exists(files_path_prefix + path + 'Bel'):
        os.mkdir(files_path_prefix + path + 'Bel')

    if not os.path.exists(files_path_prefix + path + 'Bel/daily'):
        os.mkdir(files_path_prefix + path + 'Bel/daily')

    flux, flux_quantiles = scale_to_bins(flux, quantiles_amount)
    height, width = flux.shape[1], flux.shape[2]

    a = np.zeros((height, width), dtype=float)
    b = np.zeros((height, width), dtype=float)
    a[mask] = np.NaN
    b[mask] = np.NaN

    for t in tqdm.tqdm(range(time_start + 1, time_end)):
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

        np.save(files_path_prefix + path + f'Bel/daily/A_{t + start_index}', a)
        np.save(files_path_prefix + path + f'Bel/daily/B_{t + start_index}', b)
    return


def count_1d_Korolev(files_path_prefix: str,
                     flux: np.ndarray,
                     time_start: int,
                     time_end: int,
                     path: str = 'Synthetic/',
                     quantiles_amount: int = 50,
                     n_components: int = 2,
                     start_index: int = 0,
                     ):
    """
    Counts and saves to files_path_prefix + path + 'Kor/daily' A and B estimates for flux array for each day
    t+start index for t in (time_start, time_end)
    :param files_path_prefix: path to the working directory
    :param flux: np.array with shape [time_steps, height, width]
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param path: additional path to the folder from files_path_prefix, like 'Synthetic/', 'Components/sensible/',
    'Components/latent/'
    :param quantiles_amount: how many quantiles to use (for one step)
    :param n_components: amount of components for EM
    :param start_index: offset index when saving maps
    :return:
    """
    if not os.path.exists(files_path_prefix + path):
        os.mkdir(files_path_prefix + path)

    if not os.path.exists(files_path_prefix + path + 'Kor'):
        os.mkdir(files_path_prefix + path + 'Kor')

    if not os.path.exists(files_path_prefix + path + 'Kor/daily'):
        os.mkdir(files_path_prefix + path + 'Kor/daily')

    a_map = np.zeros((flux.shape[1], flux.shape[2]), dtype=float)
    b_map = np.zeros((flux.shape[1], flux.shape[2]), dtype=float)
    a_map[np.isnan(flux[0])] = np.nan
    b_map[np.isnan(flux[0])] = np.nan
    for t in tqdm.tqdm(range(time_start + 1, time_end)):
        # print(f't = {t}')
        flux_array, quantiles = scale_to_bins(flux[t - 1], quantiles_amount)
        flux_set = list(set(flux_array[np.logical_not(np.isnan(flux_array))].flat))
        for group in range(len(flux_set)):
            value_t0 = flux_set[group]
            if np.isnan(value_t0):
                continue
            day_sample = (flux[t][np.where(flux_array == value_t0)] -
                          flux[t - 1][np.where(flux_array == value_t0)]).flatten()
            # print(len(day_sample))
            # plot_hist(day_sample, group)

            window = day_sample
            # gm = GaussianMixture(n_components=n_components,
            #                      tol=1e-6,
            #                      covariance_type='spherical',
            #                      max_iter=10000,
            #                      init_params='random',
            #                      n_init=30
            #                      ).fit(window.reshape(-1, 1))
            gm = GaussianMixture(n_components=n_components,
                                 tol=1e-4,
                                 covariance_type='spherical',
                                 max_iter=1000,
                                 init_params='random',
                                 n_init=5
                                 ).fit(window.reshape(-1, 1))
            means = gm.means_.flatten()
            sigmas_squared = gm.covariances_.flatten()
            weights = gm.weights_.flatten()
            weights /= sum(weights)

            a_sum = sum(means * weights)
            b_sum = math.sqrt(sum(weights * (means ** 2 + sigmas_squared)))

            a_map[np.where(flux_array == value_t0)] = a_sum
            b_map[np.where(flux_array == value_t0)] = b_sum

        np.save(files_path_prefix + path + f'Kor/daily/A_{t+start_index}.npy', a_map)
        np.save(files_path_prefix + path + f'Kor/daily/B_{t+start_index}.npy', b_map)
    return
