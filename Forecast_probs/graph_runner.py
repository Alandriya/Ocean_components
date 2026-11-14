import datetime

import numpy as np

from Forecast_probs.graph import Graph, HIST_LEN, n_components
import random
from struct import unpack
from Plotting.nn_plotter import plot_predictions
from project_paths import files_path_prefix
from Forecast_nn.config import cfg


def load_mask(files_path_prefix):
    # Mask is in the DATA root, not under the Git checkout.
    with open(files_path_prefix + "DATA/mask", "rb") as maskfile:
        binary_values = maskfile.read(29141)
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    return mask.reshape((161, 181))[::2, ::2]


forecast_steps = 3

if __name__ == '__main__':
    random.seed(2025)
    np.random.seed(2025)
    mask = load_mask(files_path_prefix)
    graph = Graph(mask.shape[0], mask.shape[1], mask)
    # This input is a separate legacy preprocessed array and is NOT automatically
    # generated from the modern 1979-2026 grouped files.
    flux_diff = np.load(files_path_prefix + 'DATA/FLUX_mini_diff.npy')
    prev_array = flux_diff[-HIST_LEN - forecast_steps-1:-forecast_steps-1]
    prev_array = np.moveaxis(prev_array, 0, 2)
    print(prev_array.shape)
    prev_array = np.nan_to_num(prev_array)
    graph.fill_prev(prev_array)
    forecast = np.zeros((forecast_steps, 1, mask.shape[0], mask.shape[1]))
    for t in range(forecast_steps):
        graph.count_weights()
        graph.weights = np.nan_to_num(graph.weights)
        graph.get_clusters()
        print(f'Clusters amount = {len(graph.clusters)}')
        graph.count_probs(n_components)
        forecast[t, 0] = -graph.get_forecast() * 10
        graph.update(forecast[t, 0])
    yreal = flux_diff[-forecast_steps-1:]
    yreal = yreal.reshape((yreal.shape[0], 1, yreal.shape[1], yreal.shape[2]))
    start_day = datetime.datetime(2019, 1, 1)
    plot_predictions(files_path_prefix, yreal, forecast, 'Graph', n_components, start_day, mask, cfg)
