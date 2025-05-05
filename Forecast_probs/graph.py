import numpy as np
from scipy import stats

HIST_LEN = 30
PROBS_MAX = 5
CORR_LENGTH = 10
WEIGHT_TRESHOLD = 0.5
WEIGHT_COEFF = 0.3
import inspect


def count_weight(x_array, y_array):
    weight = stats.pearsonr(x_array[-CORR_LENGTH:], y_array[-CORR_LENGTH:])[0]
    return weight


def get_idx(shift_x, shift_y): #TODO another layer
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


class Vertice:
    def __init__(self, x, y, water_flag):
        self.x = x
        self.y = y
        self.water = water_flag
        self.prev = np.zeros(HIST_LEN, dtype=float)
        self.neighbours = [None for _ in range(8)]
        self.probs = np.zeros(PROBS_MAX, dtype=float)
        self.forecast = 0.0

    def update_prev(self, value):
        self.prev = np.roll(self.prev, -1)
        self.prev[-1] = value


class Cluster:
    def __init__(self, height, width, mask, vertices, label):
        self.height = height
        self.width = width
        self.mask = mask
        self.x_list = [v.x for v in vertices]
        self.y_list = [v.y for v in vertices]
        self.idxes = [v.x * self.width + v.y for v in vertices]
        self.vertices = vertices
        self.label = label


class Graph:
    def __init__(self, height, width, mask):
        self.vertices = list()
        self.height = height
        self.width = width
        self.mask = mask

        for x in range(height):
            for y in range(width):
                self.vertices.append(Vertice(x, y, mask[x, y]))

        for i in range(len(self.vertices)):
            for shift_x in [-1, 0, 1]:
                for shift_y in [-1, 0, 1]:
                    new_x = self.vertices[i].x + shift_x
                    new_y = self.vertices[i].y + shift_y
                    if not(shift_x == 0 and shift_y == 0) and (0 <= new_x < height) and (0 <= new_y < width):
                        if not mask[new_x, new_y]:
                            self.vertices[i].neighbours[get_idx(shift_x, shift_y)] = None
                        else:
                            self.vertices[i].neighbours[get_idx(shift_x, shift_y)] = self.vertices[new_x * width + new_y]

        self.weights = np.zeros((height * width, 8), dtype=float)
        self.clusters = list()

    def fill_prev(self, array):
        # array[x, y] = [0....0], length = HIST_LEN
        for x in range(self.height):
            for y in range(self.width):
                self.vertices[x * self.width + y].prev = array[x, y]

    def update(self, array):
        # array[x, y] = 0
        for x in range(self.height):
            for y in range(self.width):
                self.vertices[x * self.width + y].update_prev(array[x, y])

    def count_weights(self):
        visited = list()
        for i in range(len(self.vertices)):
            if i in visited:
                continue
            visited.append(i)
            for shift_x in [-1, 0, 1]:
                for shift_y in [-1, 0, 1]:
                    new_x = self.vertices[i].x + shift_x
                    new_y = self.vertices[i].y + shift_y
                    if not (shift_x == 0 and shift_y == 0) and (0 <= new_x < self.height) and (0 <= new_y < self.width):
                        neighbour = self.vertices[i].neighbours[get_idx(shift_x, shift_y)]
                        if not isinstance(neighbour, Vertice):
                            continue

    def show_timestep(self, t):
        array = np.zeros((self.height, self.width), dtype=float)
        array[np.logical_not(self.mask)] = None
        for x in range(self.height):
            for y in range(self.width):
                array[x, y] = self.vertices[x * self.width + y].prev[t]
        return array

    def update_forecast(self):
        forecast = np.zeros((self.height, self.width))
        for x in range(self.height):
            for y in range(self.width):
                i = x * self.width + y
                if self.vertices[i].water:
                    forecast[x, y] = self.vertices[i].forecast #fill
                    all_weights = 1.0
                    # update
                    for shift_x in [-1, 0, 1]:
                        for shift_y in [-1, 0, 1]:
                            new_x = self.vertices[i].x + shift_x
                            new_y = self.vertices[i].y + shift_y
                            if not (shift_x == 0 and shift_y == 0) and (0 <= new_x < self.height) and \
                                    (0 <= new_y < self.width) and self.mask[new_x, new_y]:
                                idx = get_idx(shift_x, shift_y)
                                weight = self.weights[i, idx]
                                if weight > WEIGHT_TRESHOLD:
                                    all_weights += weight * WEIGHT_COEFF
                                    forecast[x, y] += weight * WEIGHT_COEFF * self.vertices[i].neighbours[idx].forecast

                    forecast[x, y] /= all_weights
                else:
                    forecast[x, y] = np.nan

    def extract_cluster(self, remaining_weights):
        label = len(self.clusters)
        new_weights = np.copy(remaining_weights)
        vertices = list()
        #TODO
        cluster = Cluster(self.height, self.width, self.mask, vertices, label)
        return cluster, new_weights

    def get_clusters(self):
        remaining_weights = np.copy(self.weights)
        while remaining_weights.any():
            cluster, remaining_weights = self.extract_cluster(remaining_weights)
            self.clusters.append(cluster)
        #TODO check if lost not connected

    def color_clusters(self):
        colored = np.zeros_like(self.mask)
        colored[np.logical_not(self.mask)] = np.nan
        for c in range(len(self.clusters)):
            label = self.clusters[c].label
            x_arr = np.array(self.clusters[c].x_list)
            y_arr = np.array(self.clusters[c].y_list)
            colored[x_arr, y_arr] = label #TODO check
        return colored

    def count_probs(self):
        #TODO
        pass