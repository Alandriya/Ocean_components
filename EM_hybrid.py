import pandas as pd
import numpy as np
import tqdm
from math import exp, pi, sqrt
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO


def l2_to_optimizer(params, x, hist, n_components):
    means, sigmas, weights = params[0:n_components], params[n_components:2 * n_components], params[2 * n_components:]
    return sqrt(sum([(hist[i] - mixture_density(x[i], means, sigmas, weights)) ** 2 for i in range(len(x))]))


def phi(x: float):
    return exp(-x ** 2 / 2) / sqrt(2 * pi)


def mixture_density(x, means, sigmas, weights):
    return sum([weights[i] * phi((x - means[i]) / sigmas[i]) for i in range(len(means))])


def plot_hist(window, step):
    files_path_prefix = 'D://Data/OceanFull/'
    fig, axes = plt.subplots(1,1)
    axes.hist(window, bins=20)
    fig.savefig(files_path_prefix + f'Components/tmp/hist_step_{step}.png')
    plt.close(fig)
    return


def hybrid(sample: np.ndarray,
           window_width: int,
           n_components: int,
           EM_steps: int,
           step: int = 1,
           step_list: list = None):
    means_cols_hybrid = [f'mean_{i}_hybrid' for i in range(1, n_components + 1)]
    sigmas_cols_hybrid = [f'sigma_{i}_hybrid' for i in range(1, n_components + 1)]
    weights_cols_hybrid = [f'weight_{i}_hybrid' for i in range(1, n_components + 1)]

    if step_list is None:
        sample_length = len(sample) - len(sample) % step
        step_list = [step for _ in range(sample_length)]
    else:
        sample_length = len(step_list)

    means_hybrid = np.zeros((sample_length, n_components))
    sigmas_hybrid = np.zeros((sample_length, n_components))
    weights_hybrid = np.zeros((sample_length, n_components))

    columns = ['time', 'ts'] + means_cols_hybrid + sigmas_cols_hybrid + weights_cols_hybrid
    result_df = pd.DataFrame(columns=columns)

    # eps = 0.3

    # for i in range(window_width // 2, sample_length - window_width // 2):
    #     window = np.nan_to_num(sample[i - window_width // 2 : i + window_width // 2])
    #         if i == window_width // 2:
    # for i in tqdm.tqdm(range(window_width, sample_length, step)):
    # for i in range(window_width, sample_length, step):
    #     window = np.nan_to_num(sample[i - window_width: i])

    for i in range(len(step_list)):
        first_ind = sum(step_list[:i])
        last_ind = sum(step_list[:i+1])
        # print(f'step {i}, window=[{first_ind}, {last_ind}]')

        window = np.nan_to_num(sample[first_ind:last_ind])
        if (i == 0 or i > 0 and sum(weights_hybrid[i - 1, :]) == 0) and last_ind - first_ind > 10:
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

        # elif i % EM_steps == 0:
        # elif False:
        elif last_ind - first_ind > 10 and sum(weights_hybrid[i - 1, :]) > 0:
            gm = GaussianMixture(n_components=n_components,
                                 tol=1e-4,
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
            means_hybrid[i, :] = [0 for _ in range(n_components)]
            sigmas_hybrid[i, :] = [0 for _ in range(n_components)]
            weights_hybrid[i, :] = [0 for _ in range(n_components)]
        # else:
        #     hist, bins = np.histogram(window, bins=30, density=True)
        #     points = [(bins[j] + bins[j + 1]) / 2 for j in range(len(bins) - 1)]
        #     init_guess = [means_hybrid[i//step - 1], sigmas_hybrid[i//step - 1], weights_hybrid[i//step - 1]]
        #     bounds = [(None, None) for _ in range(n_components)] + [(1e-6, None) for _ in range(n_components)] + [
        #         (1e-6, 1) for _ in range(n_components)]
        #     #TODO add PSO
        #     # results = minimize(l2_to_optimizer, init_guess, args=(points, hist, n_components), tol=1e-3, bounds=bounds)
        #     # parameters = results.x
        #
        #     # c1 - cognitive parameter, c2 - social parameter, w - inertia parameter
        #     options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        #     optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)
        #     kwargs = {}
        #     cost, parameters = optimizer.optimize(l2_to_optimizer, 1000, n_processes=1, **kwargs)
        #
        #     means_hybrid[i//step, :] = parameters[:n_components]
        #     sigmas_hybrid[i//step, :] = parameters[n_components:2 * n_components]
        #     weights_hybrid[i//step, :] = parameters[2 * n_components:]
        #     weights_hybrid[i//step, :] = [w / sum(weights_hybrid[i//step, :]) for w in weights_hybrid[i//step, :]]

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

        # result_df.loc[i//step] = [i//step, window[0]] + list(means_hybrid[i//step]) + \
        #                          list(sigmas_hybrid[i//step]) + list(weights_hybrid[i//step])
        result_df.loc[i] = [i, 0] + list(means_hybrid[i]) + list(sigmas_hybrid[i]) + list(weights_hybrid[i])
    return result_df


def plot_components(files_path_prefix: str,
                    df: pd.DataFrame,
                    n_components: int,
                    point: tuple,
                    path: str = 'Components/tmp/',
                    postfix: str = ''):
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    fig.suptitle(f'Components evolution in point ({point[0]}, {point[1]})')
    colors = ['r', 'g', 'b', 'yellow', 'pink', 'black']
    axs[0].set_title('Means')
    axs[1].set_title('Sigmas')
    axs[2].set_title('Weights')

    method_postfix = ''
    for comp in range(n_components):
        axs[0].plot(df[f'mean_{comp + 1}{method_postfix}'], color=colors[comp], label=f'mean_{comp + 1}')
        axs[1].plot(df[f'sigma_{comp + 1}{method_postfix}'], color=colors[comp], label=f'sigma_{comp + 1}')
        axs[2].plot(df[f'weight_{comp + 1}{method_postfix}'], color=colors[comp], label=f'weight_{comp + 1}')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    fig.tight_layout()
    fig.savefig(files_path_prefix + path + postfix + '.png')
    plt.close(fig)
    return


def cluster_components(df: pd.DataFrame,
                       n_components: int,
                       files_path_prefix: str,
                       draw: bool = False,
                       path: str = 'Components/tmp/', postfix: str = ''):
    colors = ['r', 'g', 'b', 'yellow', 'pink', 'black']
    X = np.zeros((len(df) * n_components, 2), dtype=float)
    eps = 0.05
    means_cols_hybrid = [f'mean_{i}_hybrid' for i in range(1, n_components + 1)]
    sigmas_cols_hybrid = [f'sigma_{i}_hybrid' for i in range(1, n_components + 1)]
    max_mean = max(np.max(df[means_cols_hybrid]))
    min_mean = min(np.min(df[means_cols_hybrid]))
    max_sigma = max(np.max(df[sigmas_cols_hybrid]))

    for comp in range(n_components):
        X[comp * len(df): (comp + 1) * len(df), 0] = df[f'sigma_{comp + 1}_hybrid'] / max_sigma
        X[comp * len(df): (comp + 1) * len(df), 1] = (df[f'mean_{comp + 1}_hybrid'] - min_mean) / (max_mean - min_mean)
        # X[comp * len(df): (comp + 1) * len(df), 0] = df[f'sigma_{comp + 1}_hybrid']
        # X[comp * len(df): (comp + 1) * len(df), 1] = df[f'mean_{comp + 1}_hybrid']
        # X[comp * len(df): (comp + 1) * len(df), 2] = df[f'weight_{comp + 1}_hybrid']

    new_n_components = n_components
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
        plt.xlabel('Sigma')
        plt.ylabel('Mean')
        fig_new.tight_layout()
        fig_new.savefig(files_path_prefix + path + '/a-sigma_clustered' + postfix + '.png')
        plt.close(fig_new)

    # get back to sigma from sigma**2
    for comp in range(new_n_components):
        new_df[f'sigma_{comp+1}'] = np.sqrt(new_df[f'sigma_{comp+1}'])

    return new_df, new_n_components


def plot_a_sigma(df: pd.DataFrame, n_components, point: tuple, files_path_prefix: str,
                 path: str = 'Components/tmp/', postfix: str = ''):
    fig, axs = plt.subplots(figsize=(15, 15))
    fig.suptitle(f'A-sigma in point ({point[0]}, {point[1]})')
    colors = ['r', 'g', 'b', 'yellow', 'pink', 'black']

    for comp in range(n_components):
        axs.scatter(df[f'sigma_{comp + 1}_hybrid'], df[f'mean_{comp + 1}_hybrid'], color=colors[comp])

    fig.tight_layout()
    fig.savefig(files_path_prefix + path + '/a-sigma' + postfix + '.png')
    plt.close(fig)
    return
