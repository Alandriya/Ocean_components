import datetime
import gc
import os
import darts
import matplotlib.pyplot as plt
import numpy as np
from struct import unpack

import pandas as pd
import sklearn.metrics
import tqdm
from sklearn.cluster import KMeans
from darts.dataprocessing.transformers import Scaler
from sklearn.metrics import mean_squared_error
from darts import TimeSeries
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from darts.models import TransformerModel, ExponentialSmoothing
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from copy import copy
from plotter import plot_predictions
import joblib

import warnings
warnings.filterwarnings("ignore")

def rmse(y_predict, y_true):
    return np.sqrt(mean_squared_error(y_predict, y_true))

def plot_clusters(
        frequencies: np.ndarray,
        labels: np.ndarray,
        yhat: np.ndarray,
        filename: str):
    fig, axs = plt.subplots(3, 3, figsize=(30, 30))
    for label in labels:
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == label)
        # create scatter of these samples
        axs[0][0].scatter(frequencies[row_ix, 0], frequencies[row_ix, 1])
        axs[0][1].scatter(frequencies[row_ix, 0], frequencies[row_ix, 2])
        axs[0][2].scatter(frequencies[row_ix, 0], frequencies[row_ix, 3])
        axs[1][0].scatter(frequencies[row_ix, 1], frequencies[row_ix, 0])
        axs[1][1].scatter(frequencies[row_ix, 1], frequencies[row_ix, 2])
        axs[1][2].scatter(frequencies[row_ix, 1], frequencies[row_ix, 3])
        axs[2][0].scatter(frequencies[row_ix, 2], frequencies[row_ix, 0])
        axs[2][1].scatter(frequencies[row_ix, 2], frequencies[row_ix, 1])
        axs[2][2].scatter(frequencies[row_ix, 2], frequencies[row_ix, 3])

    # plt.tight_layout()
    fig.savefig(files_path_prefix + f'videos/Forecast/Clusters/{filename}.png')
    return


def clusterize(files_path_prefix: str,
               array: np.ndarray,
               mask: np.ndarray,
               n_clusters: int,
               filename: str,
               n_freq: int = 100):
    frequencies = np.zeros((np.sum(mask), n_freq))
    k = 0
    rows_idxes = np.zeros(np.sum(mask), dtype=int)
    for i in range(array.shape[0]):
        if mask[i]:
            frequencies[k, :] = np.fft.fft(array[i], n=n_freq)
            rows_idxes[k] = i
            k += 1

    np.save(files_path_prefix + f'Forecast/Models/Clusters/rows_idxes.npy', rows_idxes)
    np.save(files_path_prefix + f'Forecast/Models/Clusters/frequencies_{filename}.npy', frequencies)
    # define the model
    model = KMeans(n_clusters)
    # fit the model
    model.fit(frequencies)
    # assign a cluster to each example
    yhat = model.predict(frequencies)
    np.save(files_path_prefix + f'Forecast/Models/Clusters/clusters_{filename}.npy', yhat)
    # retrieve unique clusters
    labels = np.unique(yhat)
    print(f'N clusters = {len(labels)}')
    np.save(files_path_prefix + f'Forecast/Models/Clusters/clusters_labels_{filename}.npy', labels)
    plot_clusters(frequencies, labels, yhat, 'flux')
    return


def prepare_clusters(files_path_prefix: str,
                   array: np.ndarray,
                    masked_row_idxes: np.ndarray,
                   clusters: np.ndarray,
                     n_lags: int,
                     n_forecast: int,
                     start_day = datetime.datetime(2019, 1, 1)):
    labels = np.unique(clusters)
    for i in range(len(labels)):
    # for i in range(1):
        label = labels[i]
        row_idxes = np.where(clusters == label)
        amount = len(row_idxes[0])
        print(f'Creating dataset for cluster {i + 1}/{len(labels)} with {amount} time series')
        cluster_part = np.zeros((amount, array.shape[1]))
        train_len = int(array.shape[1] * 2 / 3.0)
        test_len = array.shape[1] - train_len - n_lags * 2 - n_forecast

        df_x_train = pd.DataFrame(columns=['day'] + [f'component_{i}-day_lag_{t}' for i in range(amount)
                                                         for t in range(1, n_lags + 1)])
        df_y_train = pd.DataFrame(columns=['day'] + [f'component_{i}-day_{t}' for i in range(amount)
                                                         for t in range(1, n_forecast + 1)])

        df_x_test = pd.DataFrame(columns=['day'] + [f'component_{i}-day_lag_{t}' for i in range(amount)
                                                     for t in range(1, n_lags + 1)])
        df_y_test = pd.DataFrame(columns=['day'] + [f'component_{i}-day_{t}' for i in range(amount)
                                                     for t in range(1, n_forecast + 1)])

        # x_train = np.zeros((train_len, amount, n_lags))
        # y_train = np.zeros((train_len, amount, n_forecast))
        # x_test = np.zeros((test_len, amount, n_lags))
        # y_test = np.zeros((test_len, amount, n_forecast))
        #
        # for k in range(amount):
        #     row = masked_row_idxes[row_idxes[0][k]]
        #     cluster_part[k] = array[row, :]
        #
        #     for t in range(0, train_len):
        #         x_train[t, k, :] = array[row, t:t + n_lags].reshape(n_lags)
        #         y_train[t, k, :] = array[row, t + n_lags + 1:t + n_lags + 1 + n_forecast].reshape(n_forecast)
        #
        #     for t in range(0, test_len):
        #         x_test[t, k, :] = array[row, train_len + n_lags + t:train_len + n_lags + t + n_lags].reshape(n_lags)
        #         y_test[t, k, :] = array[row, train_len + n_lags * 2 + t + 1:train_len + n_lags * 2 + t + 1 + n_forecast].\
        #             reshape(n_forecast)
        # np.save(files_path_prefix + f'Forecast/Train/x_train_{i}.npy', x_train)
        # np.save(files_path_prefix + f'Forecast/Train/y_train_{i}.npy', y_train)
        # np.save(files_path_prefix + f'Forecast/Test/x_test_{i}.npy', x_test)
        # np.save(files_path_prefix + f'Forecast/Test/y_test_{i}.npy', y_test)

        x_train = np.load(files_path_prefix + f'Forecast/Train/x_train_{i}.npy')
        y_train = np.load(files_path_prefix + f'Forecast/Train/y_train_{i}.npy')
        x_test = np.load(files_path_prefix + f'Forecast/Test/x_test_{i}.npy')
        y_test = np.load(files_path_prefix + f'Forecast/Test/y_test_{i}.npy')

        for t_first in range(0, train_len):
            day = start_day + datetime.timedelta(days=t_first)
            x_train_list, y_train_list = [day], [day]
            for comp_idx in range(x_train.shape[1]):
                for t in range(n_lags):
                    x_train_list.append(x_train[t_first, comp_idx, t])
                for t in range(n_forecast):
                    y_train_list.append(y_train[t_first, comp_idx, t])

            df_x_train.loc[len(df_x_train)] = x_train_list
            df_y_train.loc[len(df_y_train)] = y_train_list

        print(df_x_train.head(1))

        df_x_train.to_excel(files_path_prefix + f'Forecast/Train/df_x_train_{i}.xlsx', index=False)
        df_y_train.to_excel(files_path_prefix + f'Forecast/Train/df_y_train_{i}.xlsx', index=False)

        for t_first in range(train_len, train_len + x_test.shape[0]):
            day = start_day + datetime.timedelta(days=t_first)
            x_test_list, y_test_list = [day], [day]
            for comp_idx in range(x_test.shape[1]):
                for t in range(n_lags):
                    x_test_list.append(x_test[t_first - train_len, comp_idx, t])
                for t in range(n_forecast):
                    y_test_list.append(y_test[t_first - train_len, comp_idx, t])

            df_x_test.loc[len(df_x_test)] = x_test_list
            df_y_test.loc[len(df_y_test)] = y_test_list

        df_x_test.to_excel(files_path_prefix + f'Forecast/Test/df_x_test_{i}.xlsx', index=False)
        df_y_test.to_excel(files_path_prefix + f'Forecast/Test/df_y_test_{i}.xlsx', index=False)
    return


def train_cluster(files_path_prefix: str,
                  # x_train: TimeSeries,
                  targets: list,
                  cluster_idx: int,
                  n_lags: int,
                  n_forecast: int,
                  ):
    model = TransformerModel(
        input_chunk_length=n_lags,
        output_chunk_length=n_forecast,
        batch_size=32,
        n_epochs=25,
        model_name=f"cluster_{cluster_idx}",
        d_model=16,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        # d_model=8,
        # nhead=4,
        # num_encoder_layers=1,
        # num_decoder_layers=1,
        # dim_feedforward=32,
        dropout=0.1,
        activation="relu",
        random_state=42,
        save_checkpoints=True,
        force_reset=True,
        work_dir=files_path_prefix + f'Forecast/Models/Clusters/'
    )
    model.fit(targets, verbose=True)
    return model


def scale(array):
    arr_min = np.nanmin(array)
    arr_max = np.nanmax(array)
    return arr_min, arr_max, (array - arr_min)/(arr_max - arr_min)


if __name__ == '__main__':
    # parameters
    files_path_prefix = 'E:/Nastya/Data/OceanFull/'
    batch_size = 32
    days_known = 14
    days_prediction = 10
    n_clusters = 100
    # --------------------------------------------------------------------------------
    # Mask
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    # ---------------------------------------------------------------------------------------
    # load data
    array = np.load(files_path_prefix + f'Fluxes/FLUX_2019-2023_grouped.npy')
    # print(array.shape)
    train_len = int(array.shape[1] * 2 / 3 + 100)
    train_days = [datetime.datetime(2019, 1, 1) + datetime.timedelta(days=t) for t in range(train_len)]
    test_days = [datetime.datetime(2019, 1, 1) + datetime.timedelta(days=t) for t in range(train_len, array.shape[1])]

    filename = 'flux'
    # SST_array = np.load(files_path_prefix + f'SST/SST_2019-2023_grouped.npy')
    # press_array = np.load(files_path_prefix + f'Pressure/PRESS_2019-2023_grouped.npy')
    # print(flux_array.shape)

    clusterize(files_path_prefix, array, mask, n_clusters, filename, 100)
    clusters = np.load(files_path_prefix + f'Forecast/Models/Clusters/clusters_{filename}.npy')
    rows_idxes = np.load(files_path_prefix + f'Forecast/Models/Clusters/rows_idxes.npy')
    # print(clusters.shape)
    # print(clusters)
    # prepare_clusters(files_path_prefix, array, masked_rows, clusters, days_known, days_prediction)
    df_metrics = pd.DataFrame(columns=['cluster', 'RMSE'])
    df_scaling = pd.DataFrame(columns=['row', 'min', 'max'])

    real_values = np.zeros((161 * 181, days_prediction))
    real_values[np.logical_not(mask), :] = np.nan
    predictions = np.zeros((161 * 181, days_prediction))
    predictions[np.logical_not(mask), :] = np.nan

    error = 0
    model_str = 'TFT-complex-simple-clusters100'
    if not os.path.exists(files_path_prefix + f'Forecast/Models/{model_str}'):
        os.mkdir(files_path_prefix + f'Forecast/Models/{model_str}')
        os.mkdir(files_path_prefix + f'Forecast/Models/{model_str}/Clusters')
    for cluster_idx in range(n_clusters):
        # if True:
        #     continue
    # for cluster_idx in range(1):
        idxes = np.where(clusters == cluster_idx)[0]
        targets = list()
        scale_list = list()
        for i in range(len(idxes)):
            row = rows_idxes[idxes[i]]
            scaled = copy(array[row, :train_len])
            arr_min = np.min(scaled)
            arr_max = np.max(scaled)
            scale_list.append((arr_min, arr_max))
            df_scaling.loc[len(df_scaling)] = [row, arr_min, arr_max]
            scaled = (scaled - arr_min) / (arr_max - arr_min)

            target = pd.Series(scaled, train_days)
            targets.append(TimeSeries.from_series(target))

        model = train_cluster(files_path_prefix, targets, cluster_idx, days_known, days_prediction)
        model.save(files_path_prefix + f'Forecast/Models/{model_str}/Clusters/model_{cluster_idx}.pth')

        # test
        model = PastCovariatesTorchModel.load(files_path_prefix + f'Forecast/Models/{model_str}/Clusters/model_{cluster_idx}.pth')
        y_predict_list = model.predict(days_prediction, targets)

        cluster_error = 0
        for i in range(len(idxes)):
            row = rows_idxes[idxes[i]]
            y_test = array[row, 0:days_prediction]
            y_predict = y_predict_list[i].values()
            y_predict = np.array(y_predict).flatten()
            # y_predict = np.zeros_like(y_test)

            y_predict = y_predict * (scale_list[i][1] - scale_list[i][0]) + scale_list[i][0]
            real_values[row, :] = y_test
            predictions[row, :] = y_predict
            cluster_error += np.sum(np.square(y_predict - y_test))

        print(f'Cluster {cluster_idx} RMSE = {np.sqrt(cluster_error):.1f}')
        error += cluster_error

    # print(f'Sum RMSE = {np.sqrt(error): .1f}')
    # df_scaling.to_excel(files_path_prefix + f'Forecast/Models/{model_str}/df_scaling.xlsx', index=False)
    # np.save(files_path_prefix + f'Forecast/Models/{model_str}/predictions.npy', predictions)
    # np.save(files_path_prefix + f'Forecast/real.npy', real_values)

    # plot predictions
    real_values = np.load(files_path_prefix + f'Forecast/real.npy')
    predictions = np.load(files_path_prefix + f'Forecast/Models/{model_str}/predictions.npy')
    plot_predictions(files_path_prefix, real_values.reshape((161, 181, -1)), predictions.reshape((161, 181, -1)),
                     model_str, test_days[0])
