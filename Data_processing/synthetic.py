import math
import numpy as np
import matplotlib.pyplot as plt
import os

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
    # alpha = -0.3
    # beta = 0.01
    # omega = [math.pi / 2, math.pi * 2 / 3, math.pi * 4 / 3, math.pi * 2]
    # weights = [0.4, 0.3, 0.2, 0.1]
    alpha = -0.1
    beta = 0.001

    omega = [math.pi / 2, math.pi * 2 / 3, math.pi * 4 / 3, math.pi * 2, math.pi/6, math.pi/7]
    weights = [0.35, 0.2, 0.05, 0.1, 0.1, 0.1, 0.1]
    # print(sum(weights))

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

    if not os.path.exists(f'{files_path_prefix}Synthetic'):
        os.mkdir(f'{files_path_prefix}Synthetic')
    np.save(f'{files_path_prefix}Synthetic/flux_full.npy', X)
    np.save(f'{files_path_prefix}Synthetic/B_full.npy', b)
    np.save(f'{files_path_prefix}Synthetic/A_full.npy', a)
    return
