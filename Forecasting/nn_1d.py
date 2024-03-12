import matplotlib.pyplot as plt
import numpy as np
from struct import unpack

import pandas as pd
import sklearn.metrics
from sklearn.cluster import KMeans
from darts.dataprocessing.transformers import Scaler
from sklearn.metrics import mean_squared_error
from darts import TimeSeries
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from darts.models import TransformerModel, ExponentialSmoothing
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf

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
                     n_forecast: int,):
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

        x_train = np.zeros((train_len, amount, n_lags))
        y_train = np.zeros((train_len, amount, n_forecast))
        x_test = np.zeros((test_len, amount, n_lags))
        y_test = np.zeros((test_len, amount, n_forecast))

        for k in range(amount):
            row = masked_row_idxes[row_idxes[0][k]]
            cluster_part[k] = array[row, :]

            for t in range(0, train_len):
                x_train[t, k, :] = array[row, t:t + n_lags].reshape(n_lags)
                y_train[t, k, :] = array[row, t + n_lags + 1:t + n_lags + 1 + n_forecast].reshape(n_forecast)

            for t in range(0, test_len):
                x_test[t, k, :] = array[row, train_len + n_lags + t:train_len + n_lags + t + n_lags].reshape(n_lags)
                y_test[t, k, :] = array[row, train_len + n_lags * 2 + t + 1:train_len + n_lags * 2 + t + 1 + n_forecast].\
                    reshape(n_forecast)
        np.save(files_path_prefix + f'Forecast/Train/x_train_{i}.npy', x_train)
        np.save(files_path_prefix + f'Forecast/Train/y_train_{i}.npy', y_train)
        np.save(files_path_prefix + f'Forecast/Test/x_test_{i}.npy', x_test)
        np.save(files_path_prefix + f'Forecast/Test/y_test_{i}.npy', y_test)
    return


def train_cluster(files_path_prefix: str,
                  x_train: np.ndarray,
                  y_train: np.ndarray,
                  cluster_idx: int,
                  ):
    model = TransformerModel(
        input_chunk_length=12,
        output_chunk_length=1,
        batch_size=32,
        n_epochs=200,
        model_name=f"cluster_{cluster_idx}",
        nr_epochs_val_period=10,
        d_model=16,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        activation="relu",
        random_state=42,
        save_checkpoints=True,
        force_reset=True,
    )
    model.fit(series=(x_train, y_train), verbose=True)
    return model


if __name__ == '__main__':
    # parameters
    files_path_prefix = 'E:/Nastya/Data/OceanFull/'
    batch_size = 32
    days_known = 14
    days_prediction = 14
    n_clusters = 100
    # --------------------------------------------------------------------------------
    # Mask
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    masked_rows = np.load(files_path_prefix + f'Forecast/Models/Clusters/rows_idxes.npy')
    masked_rows.dtype = int
    # ---------------------------------------------------------------------------------------
    # load data
    # array = np.load(files_path_prefix + f'Fluxes/FLUX_2019-2023_grouped.npy')
    # filename = 'flux'
    # SST_array = np.load(files_path_prefix + f'SST/SST_2019-2023_grouped.npy')
    # press_array = np.load(files_path_prefix + f'Pressure/PRESS_2019-2023_grouped.npy')
    # print(flux_array.shape)

    # clusterize(files_path_prefix, array, mask, n_clusters, filename, 100)
    # clusters = np.load(files_path_prefix + f'Forecast/Models/Clusters/clusters_{filename}.npy')
    # prepare_clusters(files_path_prefix, array, masked_rows, clusters, days_known, days_prediction)
    df_metrics = pd.DataFrame(columns=['Cluster', 'RMSE'])

    for i in range(1):
        x_train = TimeSeries.from_values(np.load(files_path_prefix + f'Forecast/Train/x_train_{i}.npy'))
        y_train = TimeSeries.from_values(np.load(files_path_prefix + f'Forecast/Train/y_train_{i}.npy'))

        x_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
        x_train_scaled = x_scaler.fit_transform(x_train)
        y_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
        y_train_scaled = y_scaler.fit_transform(y_train)

        model = train_cluster(files_path_prefix, x_train_scaled, y_train_scaled, i)
        model.save(files_path_prefix + f'Forecast/Models/Clusters/model_{i}.pth')

        # test
        x_test = TimeSeries.from_values(np.load(files_path_prefix + f'Forecast/Test/x_test_{i}.npy'))
        y_test = TimeSeries.from_values(np.load(files_path_prefix + f'Forecast/Test/y_test_{i}.npy'))
        x_test_scaled = x_scaler.transform(x_test)
        y_test_scaled = y_scaler.transform(y_test)

        y_predict = model.predict(x_test_scaled)
        df_metrics.loc[len(df_metrics)] = [i, rmse(y_predict, y_test_scaled)]
        df_metrics.to_excel(files_path_prefix + f'Forecast/df_metrics.xlsx', index=False)
