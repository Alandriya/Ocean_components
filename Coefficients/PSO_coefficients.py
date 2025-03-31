import random
import time
from math import exp, pi, sqrt

import numpy as np
import scipy
import scipy.stats as ss
import statsmodels.api as sm
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.mixture import GaussianMixture


class MixtureProblem(Problem):
    def __init__(self):
        super().__init__(n_var=9, n_obj=1, n_eq_constr=1,
                         xl=np.array((min_mean, min_mean, min_mean, 0, 0, 0, 0, 0, 0), dtype=float),
                         xu=np.array((max_mean, max_mean, max_mean, 10, 10, 10, 1, 1, 1), dtype=float),)

    def _evaluate(self, params, out, *args, **kwargs):
        errors = list()
        constraints = list()
        for i in range(params.shape[0]):
            means, sigmas, weights = params[i][0:n_components], params[i][n_components:2 * n_components], params[i][
                                                                                                          2 * n_components:]
            weights /= sum(weights)
            # first_prob = np.array([epdf(x[i]) for i in range(len(x))]).reshape(-1, 1)
            # second_prob = np.array([mixture_density(x[i], means, sigmas, weights) for i in range(len(x))]).reshape(-1, 1)
            # errors.append(sum(rel_entr(first_prob, second_prob)))
            # errors.append(total_variation_distance(first_prob, second_prob))
            errors.append(sqrt(sum([(epdf(x[i]) - mixture_density(x[i], means, sigmas, weights)) ** 2 for i in range(len(x))])))
            # errors.append(sum([abs(epdf(x[i]) - mixture_density(x[i], means, sigmas, weights)) for i in range(len(x))]))
            constraints.append(sum(params[i][2 * n_components:])-1)
        out["F"] = np.array(errors)
        out["H"] = np.ones(params.shape[0]) - np.array(constraints)


def norm_sum(n, norm_params, weights):
    """
    Generates mixture sample
    :param n: size of sample
    :param norm_params: (a, sigma) for each component
    :param weights: weights of each component in mixture
    :return:
    """
    # A stream of indices from which to choose the component
    mixture_idx = np.random.choice(len(weights), size=n, replace=True, p=weights)
    sample = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx), dtype=np.float64)
    return sample


def phi(x: float):
    return exp(-x ** 2 / 2) / sqrt(2 * pi)


def mixture_density(x, means, sigmas, weights):
    return sum([weights[i] * phi((x - means[i]) / sigmas[i]) for i in range(len(means))])


# # # def l2_to_optimizer(params, n_components, window):
def l2_to_optimizer(params, out):
    ecdf = sm.distributions.ECDF(window)
    hist, bins = np.histogram(window, bins=len(window), density=True)
    x = bins
    errors = list()
    for i in range(params.shape[0]):
        means, sigmas, weights = params[i][0:n_components], params[i][n_components:2 * n_components], params[i][2 * n_components:]
        weights /= sum(weights)
        # errors.append(sqrt(sum([(ecdf(x[i]) - mixture_density(x[i], means, sigmas, weights)) ** 2 for i in range(len(x))])))
        errors.append(sum([abs(ecdf(x[i]) - mixture_density(x[i], means, sigmas, weights)) for i in range(len(x))]))
    out = {'F': errors}
    return out

# def l2_to_optimizer(params, n_components, window):
# def l2_to_optimizer(params):
#     means, sigmas, weights = params[0:n_components], params[n_components:2 * n_components], params[2 * n_components:]
#     weights /= sum(weights)
#     error = sqrt(sum([(ecdf(x[i]) - mixture_density(x[i], means, sigmas, weights)) ** 2 for i in range(len(x))]))
#     return error


def optimize(func, window, init_params_EM=None):
    n_components = 3
    n_particles = 100
    delta = max(window) - min(window)
    min_bound = np.array([min(window) - delta/2] * n_components + [0] * n_components + [0] * n_components)
    max_bound = np.array([max(window) + delta/2] * n_components + [delta/2] * n_components + [1] * n_components)
    bounds = (min_bound, max_bound)

    init_params = np.zeros((n_particles, 3*n_components))
    for i in range(n_particles):
        init_params[i] = init_params_EM
        # init_params[i][0: n_components] += np.random.normal(0, 1, n_components)
        # init_params[i][n_components:2*n_components] += abs(np.random.normal(0, 0.1, n_components))

    # c1 - cognitive parameter, c2 - social parameter, w - inertia parameter
    # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    options = {'c1': 0.5, 'c2': 0.2, 'w': 0.7}
    optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=3*n_components, options=options, bounds=bounds,)
                              # init_pos=init_params)

    kwargs = {'n_components': n_components, 'window': window}
    cost, pos = optimizer.optimize(func, 100, **kwargs)
    print(f'Error: {cost}')
    print(f'Means:{pos[0:n_components]}')
    print(f'Sigmas: {pos[n_components:2*n_components]}')
    print(f'Weights: {pos[2*n_components:]}')
    return pos



random.seed(1)
n_components = 3
window = norm_sum(100, [(0, 1), (-5, 1), (10, 3)], [0.5, 0.3, 0.2])
# ecdf = sm.distributions.ECDF(window)
hist, bins = np.histogram(window, bins=len(window), density=True)
x = bins
epdf = scipy.stats.rv_histogram((hist, bins)).pdf

# plt.hist(window, bins=15)
# plt.show()

EM_start = time.time()
gm = GaussianMixture(n_components=n_components,
                     tol=1e-6,
                     covariance_type='spherical',
                     max_iter=10000,
                     init_params='random',
                     n_init=30
                     ).fit(window.reshape(-1, 1))
init_params = list(gm.means_.flatten()) + list(np.sqrt(gm.covariances_.flatten())) + list(gm.weights_.flatten())
print(f'Means EM:{list(gm.means_.flatten())}')
print(f'Sigmas EM: {list(np.sqrt(gm.covariances_.flatten()))}')
print(f'Weights EM: {list(gm.weights_.flatten())}')

print(f"EM time: {time.time() - EM_start} sec\n")

# optimize(l2_to_optimizer, window, np.array(init_params))
# optimize(l2_to_optimizer, window)
min_mean = min(window)
max_mean = max(window)
algorithm = PSO(pop_size=25, w=0.9, c1=2, c2=1,
                adaptive=True,
                initial_velocity='random',
                pertube_best=True)
# algorithm = NelderMead()

# algorithm = GA(
#     pop_size=100,
#     eliminate_duplicates=True)
opt_start = time.time()
problem = MixtureProblem()
res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

# print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
print(f'Error: {res.F}')
print(f'Means:{res.X[0:n_components]}')
print(f'Sigmas: {res.X[n_components:2 * n_components]}')
print(f'Weights: {res.X[2 * n_components:]}')
print(f"Optimisation time: {time.time() - opt_start} sec")