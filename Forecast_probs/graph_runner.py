import numpy as np
from Forecast_probs.graph import Graph
import random

if __name__ == '__main__':
    random.seed(2025)
    np.random.seed(2025)

    mask = np.ones((9, 10), dtype=bool)
    mask[:3, 1:3] = 0
    graph = Graph(9, 10, mask)
    prev_array = np.random.random((10, 10, 30))
    graph.fill_prev(prev_array)
    print(graph.show_timestep(-1))

    graph.count_weights()
    print(graph.weights)

    graph.get_clusters()
    print(graph.color_clusters())

    graph.count_probs(n_components=3)
    print(graph.get_forecast())
