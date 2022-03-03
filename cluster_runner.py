import argparse
import numpy as np
import pandas as pd
from struct import unpack
from math import exp, pi, sqrt
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from multiprocessing import Pool


def l2_to_optimizer(params, x, hist, n_components):
    means, sigmas, weights = params[0:n_components], params[n_components:2 * n_components], params[2 * n_components:]
    return sqrt(sum([(hist[i] - mixture_density(x[i], means, sigmas, weights)) ** 2 for i in range(len(x))]))


def phi(x: float):
    return exp(-x ** 2 / 2) / sqrt(2 * pi)


def mixture_density(x, means, sigmas, weights):
    return sum([weights[i] * phi((x - means[i]) / sigmas[i]) for i in range(len(means))])


def cluster_components(df: pd.DataFrame, n_components, point: int, files_path_prefix: str, flux_type: str,
                       draw: bool = False):
    colors = ['r', 'g', 'b', 'yellow', 'pink', 'black']
    X = np.zeros((len(df) * n_components, 3), dtype=float)
    eps = 0.05
    means_cols_hybrid = [f'mean_{i}_hybrid' for i in range(1, n_components + 1)]
    sigmas_cols_hybrid = [f'sigma_{i}_hybrid' for i in range(1, n_components + 1)]
    max_mean = max(np.max(df[means_cols_hybrid]))
    min_mean = min(np.min(df[means_cols_hybrid]))
    max_sigma = max(np.max(df[sigmas_cols_hybrid]))

    for comp in range(n_components):
        X[comp * len(df): (comp + 1) * len(df), 0] = df[f'sigma_{comp + 1}_hybrid'] / max_sigma
        X[comp * len(df): (comp + 1) * len(df), 1] = (df[f'mean_{comp + 1}_hybrid'] - min_mean) / (max_mean - min_mean)
        X[comp * len(df): (comp + 1) * len(df), 2] = df[f'weight_{comp + 1}_hybrid']

    new_n_components = 3
    kmeans = KMeans(n_clusters=new_n_components, random_state=0).fit(X)
    labels = kmeans.labels_

    if draw:
        fig_new, axs_new = plt.subplots(figsize=(15, 15))

    for new_comp in range(new_n_components):
        x_part = X[labels == new_comp, 0]
        y_part = X[labels == new_comp, 1]
        if draw:
            axs_new.scatter(x_part, y_part, color=colors[new_comp])

    means_cols = [f'mean_{i}' for i in range(1, new_n_components + 1)]
    sigmas_cols = [f'sigma_{i}' for i in range(1, new_n_components + 1)]
    weights_cols = [f'weight_{i}' for i in range(1, new_n_components + 1)]
    new_df = pd.DataFrame(columns=['time', 'ts'] + means_cols + sigmas_cols + weights_cols)
    new_df[means_cols + sigmas_cols + weights_cols] = np.zeros((len(df), 3*new_n_components))
    for i in range(0, len(df)):
        for j in range(n_components):
            label = labels[j * len(df) + i]
            new_df.loc[i, f'mean_{label + 1}'] += df.loc[i, f'mean_{j + 1}_hybrid']
            new_df.loc[i, f'sigma_{label + 1}'] += df.loc[i, f'sigma_{j + 1}_hybrid']**2
            new_df.loc[i, f'weight_{label + 1}'] += df.loc[i, f'weight_{j + 1}_hybrid']

        for k in range(new_n_components):
            if new_df.loc[i, f'weight_{k + 1}'] < eps:
                new_df.loc[i, f'mean_{k + 1}'] = None
                new_df.loc[i, f'sigma_{k + 1}'] = None
                new_df.loc[i, f'weight_{k + 1}'] = None

    if draw:
        fig_new.tight_layout()
        fig_new.savefig(files_path_prefix + f'Components/plots/{flux_type}/a-sigma_clustered_point_{point}.png')

    # get back to sigma from sigma**2
    for comp in range(new_n_components):
        new_df[f'sigma_{comp+1}'] = np.sqrt(new_df[f'sigma_{comp+1}'])

    return new_df, new_n_components


def hybrid(sample: np.ndarray, window_width: int, n_components: int, EM_steps: int):
    means_cols_hybrid = [f'mean_{i}_hybrid' for i in range(1, n_components + 1)]
    sigmas_cols_hybrid = [f'sigma_{i}_hybrid' for i in range(1, n_components + 1)]
    weights_cols_hybrid = [f'weight_{i}_hybrid' for i in range(1, n_components + 1)]

    sample_length = len(sample)

    means_hybrid = np.zeros((sample_length, n_components))
    sigmas_hybrid = np.zeros((sample_length, n_components))
    weights_hybrid = np.zeros((sample_length, n_components))

    columns = ['time', 'ts'] + means_cols_hybrid + sigmas_cols_hybrid + weights_cols_hybrid
    result_df = pd.DataFrame(columns=columns)

    eps = 0.3

    for i in range(0, sample_length - window_width):
        window = np.nan_to_num(sample[i:i + window_width])
        if i == 0:
            gm = GaussianMixture(n_components=n_components,
                                 tol=1e-6,
                                 covariance_type='spherical',
                                 max_iter=10000,
                                 init_params='random',
                                 n_init=30
                                 ).fit(window.reshape(-1, 1))
            means_hybrid[i, :] = gm.means_.reshape(1, -1)
            sigmas_hybrid[i, :] = np.sqrt(gm.covariances_.reshape(1, -1))
            weights_hybrid[i, :] = gm.weights_.reshape(1, -1)

        elif i % EM_steps == 0:
            gm = GaussianMixture(n_components=n_components,
                                 tol=1e-3,
                                 covariance_type='spherical',
                                 max_iter=10000,
                                 means_init=means_hybrid[i - 1, :].reshape(-1, 1),
                                 weights_init=weights_hybrid[i - 1, :],
                                 init_params='random',
                                 n_init=15).fit(window.reshape(-1, 1))

            means_hybrid[i, :] = gm.means_.reshape(1, -1)
            sigmas_hybrid[i, :] = np.sqrt(gm.covariances_.reshape(1, -1))
            weights_hybrid[i, :] = gm.weights_.reshape(1, -1)
        else:
            hist, bins = np.histogram(window, bins=30, density=True)
            points = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
            init_guess = [means_hybrid[i - 1], sigmas_hybrid[i - 1], weights_hybrid[i - 1]]
            bounds = [(None, None) for _ in range(n_components)] + [(1e-6, None) for _ in range(n_components)] + [
                (1e-6, 1) for _ in range(n_components)]
            results = minimize(l2_to_optimizer, init_guess, args=(points, hist, n_components), tol=1e-3, bounds=bounds)
            parameters = results.x

            means_hybrid[i, :] = parameters[:n_components]
            sigmas_hybrid[i, :] = parameters[n_components:2 * n_components]
            weights_hybrid[i, :] = parameters[2 * n_components:]
            weights_hybrid[i, :] = [w / sum(weights_hybrid[i, :]) for w in weights_hybrid[i, :]]

        # # check if 0
        # for comp in range(n_components):
        #     if weights_hybrid[i, comp] < eps:
        #         sigmas_hybrid[i, comp] = np.nan
        #         means_hybrid[i, comp] = np.nan
        #         weights_hybrid[i, comp] = np.nan

        # # sort by means
        # zipped = list(zip(means_hybrid[i], sigmas_hybrid[i], weights_hybrid[i]))
        # zipped.sort()
        # means_hybrid[i], sigmas_hybrid[i], weights_hybrid[i] = zip(*zipped)

        result_df.loc[i] = [i, window[0]] + list(means_hybrid[i]) + list(sigmas_hybrid[i]) + list(weights_hybrid[i])
    return result_df


def _parallel_worker(args):
    flux_array, mask, points = args
    print('My process id:', os.getpid())

    n_components = 3
    window_width = 300

    for i in range(len(points)):
        point = points[i]
        if mask[point]:
            sample = flux_array[i, :]
            df = hybrid(sample, window_width, n_components, 5)
            new_df, new_n_components = cluster_components(df, n_components, point, files_path_prefix, flux_type)
            new_df.to_excel(files_path_prefix + f'Components/{flux_type}/point_{point}.xlsx', index=False)

    print(f'Process {os.getpid()} finished')
    return


# files_path_prefix = 'D://Data/OceanFull/'
files_path_prefix = 'home/aosipova/EM_ocean'
flux_type = 'sensible'
# flux_type = 'latent'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("n_processes", help="Amount of processes to parallel run", type=int)
    args_cmd = parser.parse_args()

    cpu_count = args_cmd.n_processes

    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)

    flux_array = np.load(files_path_prefix + f'{flux_type}_all.npy')
    part_len = int(flux_array.shape[1] // cpu_count + 1)
    args = list()
    for n in range(cpu_count):
        start_idx = int(flux_array.shape[1] // cpu_count)
        points = list(range(part_len * n, part_len * (n + 1)))
        args.append([flux_array[points, :], mask, points])

    with Pool(cpu_count) as p:
        p.map(_parallel_worker, args)
        p.close()
        p.join()
