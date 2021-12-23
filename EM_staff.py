import math
import random
from copy import deepcopy
from itertools import permutations
import pandas as pd
import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from fcmeans import FCM
from plotting import *
from struct import unpack
from multiprocessing import Pool
import os


# Parameters
files_path_prefix = 'D://Data/OceanFull/'
flux_type = 'sensible'
shift = 4 * 7
components_amount = 4
cpu_count = 12
window_EM = 200
start = 0
end = 29141


def distance(vec1, vec2, p, weights=(1, 1, 0)):
    """
    Calculates lp distance (modificated with weights) between vec1 and vec2. It is supposed that the 0-th element
    of the vectors is mean, 1st -- variance, 2nd -- weight of the component
    :param vec1:
    :param vec2:
    :param p:
    :param weights: list with weights: 0-th for means, 1-st for sigmas ans 2-nd for the weights
    :return:
    """
    delta_mean = abs(vec1[0] - vec2[0]) ** p * weights[0]
    delta_sigma = abs(vec1[1] - vec2[1]) ** p * weights[1]
    delta_weight = abs(vec1[2] - vec2[2]) ** p * weights[2]
    return math.pow(delta_mean + delta_sigma + delta_weight, 1.0 / p)


def get_components_exhaustive(matrix, window, epsilon):
    """
    Finds components, corresponding to previous steps, for current window. Goes over every possible variant
    :param matrix: DataFrame with parameters for already known components
    :param window: current window
    :param epsilon: the precision threshold, above which a new component is created
    :return: best_idxes -- a list of indexes, showing which components in matrix corresponds to which in window,
    best_permutations -- corresponding permutation of window's elements
    """
    min_delta = 10e10
    best_idxes = list()
    best_permutation = None
    for permutation in permutations(window):
        idxes, delta = get_components_greedy(matrix, permutation, epsilon)
        if delta < min_delta:
            min_delta = delta
            best_idxes = idxes
            best_permutation = permutation

    return best_idxes, best_permutation


def get_components_greedy(matrix, window, epsilon):
    """
    Finds components, corresponding to previous steps, for current window. Finds them by greedy algorithm
    :param matrix: DataFrame with parameters for already known components
    :param window: current window
    :param epsilon: the precision threshold, above which a new component is created
    :return: idxes -- a list of indexes, showing which components in matrix corresponds to which in window,
    all_distance -- total conformity error
    """
    all_distance = 0.0
    idxes = list()
    available_idxes = deepcopy(list(matrix.index))
    max_idx = max(available_idxes) + 1
    for component in window:
        distances = [10e6 for _ in range(len(matrix))]
        for k in range(len(available_idxes)):
            distances[k] = distance(component, matrix.loc[available_idxes[k]], 2)
        if min(distances) < epsilon:
            idx = np.argmin(distances)
            idxes.append(available_idxes[idx])
            available_idxes.remove(available_idxes[idx])
            all_distance += min(distances)
        else:
            idxes.append(max_idx)
            max_idx += 1
    return idxes, all_distance


def local_components(date, type_string, series_name, start, end, params_df, eps, algo, matrix, comp_amount,
                     history=None,
                     clusters=4):
    """
    Extracts local components of a time series and collects history of their evolution in DataFrame history
    :param date: date of program running for results recording
    :param type_string: string with data type (ocean/physical) for results recording
    :param series_name: string with time series name for results recording
    :param start: index of the 0-th element in current run
    :param end: index of the last element in current run + 1
    :param params_df: DataFrame with pre-counted parameters for time series
    :param eps: threshold above which two components are considered different
    :param algo: which algorithm tgo use, string constant
    :param matrix: DataFrame with means, variances and weights for every component at it's last appearance moment.
    Has no memory, saves only the last state of components.
    :param comp_amount: int, the amount of extracted components by EM
    :param history: DataFrame containing history of components evolution from previous processed part of time series
    or None for the 0-th part.
    :param clusters: amount of potential components for this part of series. Does not decrease with time
    :return:
    """
    X = list()
    # initialisation of history DataFrame
    history = pd.DataFrame()
    labels_row_mask = [list() for _ in range(end - start)]
    for n in tqdm.tqdm(range(end - start)):
        history.loc[start + n, 'time'] = start + n
        for num_comp in range(1, comp_amount + 1):
            if params_df.loc[start + n, f'weight_{num_comp}'] > 0.0:
                point = np.array(
                    [params_df.loc[start + n, f'mean_{num_comp}'], params_df.loc[start + n, f'sigma_{num_comp}']])

                X.append(point)
                history.loc[start + n, f'mean_{num_comp}'] = params_df.loc[start + n, f'mean_{num_comp}']
                history.loc[start + n, f'sigma_{num_comp}'] = params_df.loc[start + n, f'sigma_{num_comp}']
                history.loc[start + n, f'weight_{num_comp}'] = params_df.loc[start + n, f'weight_{num_comp}']
                labels_row_mask[n].append(1)
            else:
                labels_row_mask[n].append(0)

    X = np.array(X)
    X = X.reshape((-1, 2))

    if algo == 'cluster-kmeans' or algo == 'cluster-fcm':
        if not matrix is None:
            centers = np.empty(shape=(clusters, 2))
            for i in range(min(len(matrix), clusters)):
                centers[i, :] = np.array([matrix.iloc[i]['mean'], matrix.iloc[i]['sigma']])
            for i in range(clusters - len(matrix)):
                centers[len(matrix) + i, :] = np.array([random.uniform(min(matrix['mean']), max(matrix['mean'])),
                                                        random.uniform(min(matrix['sigma']), max(matrix['sigma']))])

            if algo == 'cluster-kmeans':
                if not matrix is None:
                    model = KMeans(n_clusters=clusters, max_iter=1500, n_init=25, init=centers, tol=eps).fit(X)
                else:
                    model = KMeans(n_clusters=clusters, max_iter=1500, n_init=25, tol=eps).fit(X)
                labels = model.labels_
                print(model.cluster_centers_)
            else:
                fcm = FCM(n_clusters=clusters)
                fcm.fit(X)
                labels = fcm.u.argmax(axis=1)

            labels += 1
            plot_clusters(date, type_string, series_name, X, labels, f'{algo}({start} - {end})')

            cur_idx = 0
            for n in tqdm.tqdm(range(end - start)):
                for i in range(0, comp_amount):
                    if labels_row_mask[n][i]:
                        history.loc[start + n, f'label_{i + 1}'] = labels[cur_idx]
                        cur_idx += 1

            return history, set(list(labels))

    elif algo == 'greedy' or algo == 'exhaustive':
        labels = list()
        if matrix is None:
            matrix = pd.DataFrame(columns=['mean', 'sigma', 'weight'])
            for num_comp in range(1, comp_amount + 1):
                if abs(params_df.loc[start, f'mean_{num_comp}']) > 0.0:
                    matrix.loc[num_comp - 1] = [params_df.loc[start, f'mean_{num_comp}'],
                                                params_df.loc[start, f'sigma_{num_comp}'],
                                                params_df.loc[start, f'weight_{num_comp}']]

        # main cycle
        for n in tqdm.tqdm(range(end - start)):
            window = list()
            for num_comp in range(1, comp_amount + 1):
                weight = params_df.loc[start + n, f'weight_{num_comp}']
                if weight > 0.0:
                    mean = params_df.loc[start + n, f'mean_{num_comp}']
                    sigma = params_df.loc[start + n, f'sigma_{num_comp}']
                    window.append([mean, sigma, weight])

            if algo == 'greedy':
                idxes, _ = get_components_greedy(matrix, window, epsilon=eps)
            elif algo == 'exhaustive':
                idxes, window = get_components_exhaustive(matrix, window, epsilon=eps)
            else:
                raise ValueError

            # update components
            for k in range(len(idxes)):
                matrix.loc[idxes[k]] = window[k]
                num_comp = idxes[k] + 1
                history.loc[start + n, f'mean_{num_comp}'] = window[k][0]
                history.loc[start + n, f'sigma_{num_comp}'] = window[k][1]
                history.loc[start + n, f'weight_{num_comp}'] = window[k][2]
                history.loc[start + n, f'label_{k + 1}'] = num_comp
                labels.append(k)
        return matrix, history, labels


def moving_average(y, width):
    """
    Calculates moving average for time series
    :param y: time series
    :param width: the width of the window
    :return: list with counted moving average
    """
    ma = list()
    for i in tqdm(range(width // 2, len(y) - width // 2)):
        ma.append(sum(y[i - width:i]) / width)
    for i in range(width):
        ma.append(None)
    return ma


def integrate(history):
    """
    Counts "integrated" components and their sum
    :param history: DataFrame or None -- history of previous part
    :return: DataFrame with integrated components, list with existing labels, DataFrame with components with more
    detailed info
    """
    # getting set of labels == components
    labels_df = history.filter(like='label', axis=1)
    arr = labels_df.values.flatten()
    labels = list(set(arr[~np.isnan(arr)]))
    labels = [int(label) for label in labels]
    # df = pd.DataFrame(columns=[f'integrated_{x}' for x in labels])

    # # normalizing weights
    # weights_df = deepcopy(history.filter(like='weight', axis=1))
    # weights_df.fillna(0, inplace=True)
    # for t in range(len(weights_df)):
    #     sum_weights = sum(weights_df.loc[t, :])
    #     weights_df.loc[t, :] /= sum_weights
    #
    # history[weights_df.columns] = weights_df

    # DataFrame with full info about component
    columns_list = ['time']
    for label in labels:
        columns_list.append(f'mean_{label}')
        columns_list.append(f'sigma_{label}')
        columns_list.append(f'weight_{label}')
    # full_df = pd.DataFrame(columns=columns_list)

    full_df_list = list()
    df_list = list()
    for t in tqdm.tqdm(range(len(history))):
        row = list()
        row_full = [history.loc[t, 'time']]
        point_labels = history.loc[t, labels_df.columns]
        for i in range(len(labels)):
            point = point_labels[point_labels == labels[i]]
            if len(point):
                mean, weight, sigma = 0, 0, 0
                for j in range(len(point)):
                    number = int(point.index[j][6:])  # label_i
                    mean += history.loc[t, f'mean_{number}']
                    sigma += history.loc[t, f'sigma_{number}']
                    weight += history.loc[t, f'weight_{number}']
                mean /= len(point)
                sigma /= len(point)
                row.append(mean * weight)
                row_full += [mean, sigma, weight]
            else:
                row.append(None)
                row_full += [None, None, None]

        df_list.append(row)
        full_df_list.append(row_full)
    df = pd.DataFrame(df_list, columns=[f'integrated_{x}' for x in labels])
    full_df = pd.DataFrame(full_df_list, columns=columns_list)

    df['all_sum'] = df.sum(axis=1)
    return df, labels, full_df


def apply_EM(ts, shift, window_EM = 200):
    """
    Applies EM to data and collects parameters into np arrays with shape
    :param ts: time series
    :param shift: how many steps forward the window moves to it's next position
    :param window_EM: window width
    :return:
    """
    means = np.zeros((len(ts) // shift, 4))
    sigmas = np.zeros((len(ts) // shift, 4))
    weights = np.zeros((len(ts) // shift, 4))
    for i in range(0, (len(ts) - window_EM) // shift):
        window = np.array(ts[i*shift:i*shift+window_EM]).reshape(-1, 1)
        converged = False
        if i == 0:
          gm = GaussianMixture(n_components=4,
                              tol = 1e-7,
                              covariance_type='spherical',
                              max_iter=10000,
                              init_params = 'random',
                              weights_init=[0.5, 0.4, 0.05, 0.05],
                              n_init=25
                        ).fit(window)
          converged = gm.converged_
        else:
          gm = GaussianMixture(n_components=4,
                              tol = 1e-4,
                              covariance_type='spherical',
                              max_iter=10000,
                              means_init=means[i-1, :].reshape(-1, 1),
                              # precisions_init=sigmas[i-1, :],
                              init_params = 'random',
                              weights_init=weights[i-1, :],
                              n_init=10).fit(window)

        means[i, :] = gm.means_.reshape(1, -1)
        sigmas[i, :] = np.sqrt(gm.covariances_.reshape(1, -1))
        weights[i, :] = gm.weights_.reshape(1, -1)
    return means, sigmas, weights


def _parallel_EM_func(arg):
    print('My process id:', os.getpid())
    borders, mask, ts_array = arg
    start, end = borders
    for i in range(start, end):
        if mask[i-start]:
            # print(f'Processing {i}-th point')
            ts = np.diff(ts_array[i-start])
            # time_start = time.time()
            means, sigmas, weights = apply_EM(ts, shift)
            # print(f'Elapsed {(time.time() - time_start) // 60} minutes {(time.time() - time_start) % 60} seconds == {time.time() - time_start} seconds')

            column_list = [f'mean_{i}' for i in range(1, components_amount + 1)] + \
                          [f'sigma_{i}' for i in range(1, components_amount + 1)] + \
                          [f'weight_{i}' for i in range(1, components_amount + 1)]
            new_data = pd.DataFrame(data=np.concatenate((means, sigmas, weights), axis=1), columns=column_list)
            new_data['ts'] = ts[:-shift:shift]
            new_data.to_csv(files_path_prefix + f'5_years_weekly/{flux_type}_{i}.csv', sep=';', index=False)

    print(f'Process {os.getpid()} finished')
    return


def parallel_EM():
    # # print("Number of cpu : ", multiprocessing.cpu_count()) # 16 cpu
    # num_lost = []
    # for i in range(20335):
    #     if not os.path.exists(files_path_prefix + f'5_years_weekly/{flux_type}_{i}.csv'):
    #         num_lost.append(i)
    # print(len(num_lost))
    #
    # start = None
    # borders = []
    # sum_lost=0
    # for j in range(1, len(num_lost)):
    #     if start is None and mask[num_lost[j]]:
    #         start = num_lost[j]
    #
    #     # if (num_lost[j-1] == num_lost[j] - 1) and j != len(num_lost) - 1 and mask[j]:
    #     #     pass
    #     if not start is None and mask[num_lost[j]] and (num_lost[j-1] != num_lost[j] - 1):
    #         if j == len(num_lost) - 1:  # the end of the array
    #             borders.append([start, num_lost[j]])
    #         else:
    #             borders.append([start, num_lost[j-1]])
    #             sum_lost += num_lost[j-1] - start
    #             start = num_lost[j]
    #
    # # print(borders)
    # # print(sum_lost)
    # # raise ValueError

    delta = (end - start + cpu_count // 2) // cpu_count
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)

    ts_array = np.load(files_path_prefix + f'5years_{flux_type}.npy')

    borders = [[start + delta*i, start + delta*(i+1)] for i in range(cpu_count)]

    masks = [mask[b[0]:b[1]] for b in borders]
    ts_arrays = [ts_array[b[0]:b[1]] for b in borders]
    args = [[borders[i], masks[i], ts_arrays[i]] for i in range(cpu_count)]

    print(borders)
    print(len(borders))

    del ts_array, borders, masks, ts_arrays
    with Pool(cpu_count) as p:
        p.map(_parallel_EM_func, args)
        p.close()
        p.join()
    return
