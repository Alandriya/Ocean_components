"""Legacy spatial correlation graph, Gaussian mixture forecasting and standalone graph experiment."""

# --- Graph ---

import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

HIST_LEN = 20
CORR_LENGTH = 10
WEIGHT_TRESHOLD = 0.9
NEIGH_COUNT = 24
n_components = 4

def count_weight(x_array, y_array):
    """Calculate the Pearson correlation weight using the most recent correlation window."""
    weight = stats.pearsonr(x_array[-CORR_LENGTH:], y_array[-CORR_LENGTH:])[0]
    return weight


def get_idx(shift_x, shift_y):
    """Map a spatial neighbor offset to the configured neighbor index."""
    if NEIGH_COUNT == 8:
        if shift_x == -1 and shift_y == -1:
            return 0
        if shift_x == -1 and shift_y == 0:
            return 1
        if shift_x == -1 and shift_y == 1:
            return 2
        if shift_x == 0 and shift_y == -1:
            return 3
        if shift_x == 0 and shift_y == 1:
            return 4
        if shift_x == 1 and shift_y == -1:
            return 5
        if shift_x == 1 and shift_y == 0:
            return 6
        if shift_x == 1 and shift_y == 1:
            return 7
    elif NEIGH_COUNT == 24:
        if shift_x == -2 and shift_y == -2:
            return 0
        if shift_x == -2 and shift_y == -1:
            return 1
        if shift_x == -2 and shift_y == 0:
            return 2
        if shift_x == -2 and shift_y == 1:
            return 3
        if shift_x == -2 and shift_y == 2:
            return 4
        if shift_x == -1 and shift_y == -2:
            return 5
        if shift_x == -1 and shift_y == -1:
            return 6
        if shift_x == -1 and shift_y == 0:
            return 7
        if shift_x == -1 and shift_y == 1:
            return 8
        if shift_x == -1 and shift_y == 2:
            return 9
        if shift_x == 0 and shift_y == -2:
            return 10
        if shift_x == 0 and shift_y == -1:
            return 11
        if shift_x == 0 and shift_y == 1:
            return 12
        if shift_x == 0 and shift_y == 2:
            return 13
        if shift_x == 1 and shift_y == -2:
            return 14
        if shift_x == 1 and shift_y == -1:
            return 15
        if shift_x == 1 and shift_y == 0:
            return 16
        if shift_x == 1 and shift_y == 1:
            return 17
        if shift_x == 1 and shift_y == 2:
            return 18
        if shift_x == 2 and shift_y == -2:
            return 19
        if shift_x == 2 and shift_y == -1:
            return 20
        if shift_x == 2 and shift_y == 0:
            return 21
        if shift_x == 2 and shift_y == 1:
            return 22
        if shift_x == 2 and shift_y == 2:
            return 23


class Vertice:
    """Store the history and spatial position of one graph vertex."""
    def __init__(self, x, y, water_flag):
        """Initialize Vertice with ``x``, ``y``, ``water_flag``."""
        self.x = x
        self.y = y
        self.water = water_flag
        self.prev = np.zeros(HIST_LEN, dtype=float)
        self.neighbours = [None for _ in range(NEIGH_COUNT)]
        self.forecast = 0.0

    def update_prev(self, value):
        """Compute update prev for Vertice from ``value``."""
        self.prev = np.roll(self.prev, -1)
        self.prev[-1] = value


class Cluster:
    """Store graph vertices grouped by strong spatial correlation."""
    def __init__(self, height, width, mask, vertices, label):
        """Initialize Cluster with ``height``, ``width``, ``mask``, ``vertices``, ``label``."""
        self.height = height
        self.width = width
        self.mask = mask
        self.x_list = [v.x for v in vertices]
        self.y_list = [v.y for v in vertices]
        self.idxes = [v.x * self.width + v.y for v in vertices]
        self.vertices = vertices
        self.label = label


class Graph:
    """Represent the spatial correlation graph and its mixture-based forecasting operations."""
    def __init__(self, height, width, mask):
        """Initialize Graph with ``height``, ``width``, ``mask``."""
        self.vertices = list()
        self.height = height
        self.width = width
        self.mask = mask

        for x in range(height):
            for y in range(width):
                self.vertices.append(Vertice(x, y, mask[x, y]))

        if NEIGH_COUNT == 8:
            shifts = [-1, 0, 1]
        elif NEIGH_COUNT == 24:
            shifts = [-2, -1, 0, 1, 2]
        for i in range(len(self.vertices)):
            for shift_x in shifts:
                for shift_y in shifts:
                    new_x = self.vertices[i].x + shift_x
                    new_y = self.vertices[i].y + shift_y
                    if not(shift_x == 0 and shift_y == 0) and (0 <= new_x < height) and (0 <= new_y < width):
                        if not mask[new_x, new_y]:
                            self.vertices[i].neighbours[get_idx(shift_x, shift_y)] = None
                        else:
                            self.vertices[i].neighbours[get_idx(shift_x, shift_y)] = self.vertices[new_x * width + new_y]

        self.weights = np.zeros((height * width, NEIGH_COUNT), dtype=float)
        self.clusters = list()

    def fill_prev(self, array):
        # array[x, y] = [0....0], length = HIST_LEN
        """Populate the graph with the initial history window."""
        for x in range(self.height):
            for y in range(self.width):
                self.vertices[x * self.width + y].prev = array[x, y]

    def update(self, array):
        # array[x, y] = 0
        """Update the internal state using the supplied observations or values."""
        for x in range(self.height):
            for y in range(self.width):
                self.vertices[x * self.width + y].update_prev(array[x, y])

    def count_weights(self):
        """Update the correlation weights of graph connections."""
        visited = list()
        for i in range(len(self.vertices)):
            if i in visited:
                continue
            visited.append(i)
            if NEIGH_COUNT == 8:
                shifts = [-1, 0, 1]
            elif NEIGH_COUNT == 24:
                shifts = [-2, -1, 0, 1, 2]
            for shift_x in shifts:
                for shift_y in shifts:
                    new_x = self.vertices[i].x + shift_x
                    new_y = self.vertices[i].y + shift_y
                    if not (shift_x == 0 and shift_y == 0) and (0 <= new_x < self.height) and (0 <= new_y < self.width):
                        neighbour = self.vertices[i].neighbours[get_idx(shift_x, shift_y)]
                        if not isinstance(neighbour, Vertice):
                            continue
                        self.weights[i, get_idx(shift_x, shift_y)] = count_weight(self.vertices[i].prev, neighbour.prev)

    def show_timestep(self, t):
        """Compute show timestep for Graph from ``t``."""
        array = np.zeros((self.height, self.width), dtype=float)
        array[np.logical_not(self.mask)] = None
        for x in range(self.height):
            for y in range(self.width):
                array[x, y] = self.vertices[x * self.width + y].prev[t]
        return array

    def extract_cluster(self, remaining_weights):
        """Extract a connected cluster from eligible graph neighbors."""
        label = len(self.clusters)
        new_weights = np.copy(remaining_weights)
        vertices = list()
        if (new_weights[new_weights > 0].any()):
            start = np.argmax(new_weights)
        else:
            start = np.argmax(np.abs(new_weights))

        i_idx = start // NEIGH_COUNT
        vertices.append(self.vertices[i_idx])
        while new_weights[i_idx].any() and max(new_weights[i_idx]) > WEIGHT_TRESHOLD: # Continue while neighbors remain and connections are strong enough.
            n_idx = np.argmax(new_weights[i_idx])

            # delete vertice
            new_weights[i_idx] = 0
            if NEIGH_COUNT == 8:
                shifts = [-1, 0, 1]
            elif NEIGH_COUNT == 24:
                shifts = [-2, -1, 0, 1, 2]
            for shift_x in shifts:
                for shift_y in shifts:
                    new_x = self.vertices[i_idx].x + shift_x
                    new_y = self.vertices[i_idx].y + shift_y
                    if not (shift_x == 0 and shift_y == 0) and (0 <= new_x < self.height) and (0 <= new_y < self.width):
                        new_weights[new_x * self.width + new_y, get_idx(-shift_x, -shift_y)] = 0

            neighbour = self.vertices[i_idx].neighbours[n_idx]
            vertices.append(neighbour)
            i_idx = neighbour.x * self.width + neighbour.y

        if len(vertices) == 1:
            # delete vertice
            new_weights[i_idx] = 0
            for shift_x in [-1, 0, 1]:
                for shift_y in [-1, 0, 1]:
                    new_x = self.vertices[i_idx].x + shift_x
                    new_y = self.vertices[i_idx].y + shift_y
                    if not (shift_x == 0 and shift_y == 0) and (0 <= new_x < self.height) and (0 <= new_y < self.width):
                        new_weights[new_x * self.width + new_y, get_idx(-shift_x, -shift_y)] = 0

        cluster = Cluster(self.height, self.width, self.mask, vertices, label)
        return cluster, new_weights

    def get_clusters(self):
        """Identify all clusters from the current graph weights."""
        remaining_weights = np.copy(self.weights)
        while remaining_weights.any():
            cluster, remaining_weights = self.extract_cluster(remaining_weights)
            # print(np.sum(remaining_weights))
            self.clusters.append(cluster)
            # print(cluster.label)
        return

    def color_clusters(self):
        """Assign a visual identifier to each graph cluster."""
        colored = np.zeros_like(self.mask, dtype=int)
        colored[np.logical_not(self.mask)] = -1
        for c in range(len(self.clusters)):
            label = self.clusters[c].label
            x_arr = np.array(self.clusters[c].x_list)
            y_arr = np.array(self.clusters[c].y_list)
            colored[x_arr, y_arr] = label
        return colored

    def count_probs(self, n_components):
        """Estimate cluster-level forecast probabilities using Gaussian mixtures."""
        for c in range(len(self.clusters)):
            cluster = self.clusters[c]
            all_prev = list()
            for v in cluster.vertices:
                all_prev += list(v.prev)

            # print(f'Cluster {c}, vertices = {len(cluster.vertices)}, len = {len(all_prev)}')
            if len(all_prev) < 3 * HIST_LEN + 1:
                for v in range(len(cluster.vertices)):
                    self.clusters[c].vertices[v].forecast = np.mean(all_prev)
            else:
                try:
                    gm = GaussianMixture(n_components=n_components,
                                         tol=1e-4,
                                         covariance_type='spherical',
                                         max_iter=1000,
                                         init_params='random',
                                         n_init=5,
                                         ).fit(np.array(all_prev).reshape(-1, 1))

                    means = gm.means_.flatten()
                    # sigmas_squared = gm.covariances_.flatten()
                    weights = gm.weights_.flatten()
                    weights /= sum(weights)
                    argmax = np.argmax(weights)

                    a_sum = sum(means * weights)
                    # b_sum = math.sqrt(sum(weights * (means ** 2 + sigmas_squared)))

                    for v in range(len(cluster.vertices)):
                        # self.clusters[c].vertices[v].forecast = means[argmax]
                        self.clusters[c].vertices[v].forecast = a_sum
                except ValueError:
                    for v in range(len(cluster.vertices)):
                        self.clusters[c].vertices[v].forecast = np.mean(all_prev)
        return

    def get_forecast(self):
        """Collect the predicted values from the graph vertices."""
        forecast = np.zeros((self.height, self.width))
        forecast[np.logical_not(self.mask)] = np.nan
        for i in range(len(self.vertices)):
            x = self.vertices[i].x
            y = self.vertices[i].y
            forecast[x, y] = self.vertices[i].forecast
        return forecast


# --- Graph Runner ---



from struct import unpack


def load_mask(files_path_prefix):
    # Mask is in the DATA root, not under the Git checkout.
    """Read and spatially subsample the legacy binary ocean mask."""
    with open(files_path_prefix + "DATA/mask", "rb") as maskfile:
        binary_values = maskfile.read(29141)
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    return mask.reshape((161, 181))[::2, ::2]


forecast_steps = 3

def main():
    """Run the legacy graph forecast and save the prediction comparison."""
    import datetime
    import random
    from Plotting.nn_plotter import plot_predictions
    from project_paths import files_path_prefix
    from Forecast_nn.config import cfg
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


if __name__ == "__main__":
    main()
