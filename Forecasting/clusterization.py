import numpy as np
import pandas as pd
import datetime
from sklearn.cluster import KMeans
from Forecasting.plotter import plot_clusters


def clusterize(files_path_prefix: str,
               array: np.ndarray,
               mask: np.ndarray,
               n_clusters: int,
               filename: str,
               n_freq: int = 100):
    height, width = 161, 181
    frequencies = np.zeros((np.sum(mask), n_freq))
    coordinates = np.zeros((np.sum(mask), 2))
    k = 0
    rows_idxes = np.zeros(np.sum(mask), dtype=int)
    for i in range(array.shape[0]):
        if mask[i]:
            frequencies[k, :] = np.fft.fft(array[i], n=n_freq)
            rows_idxes[k] = i
            coordinates[k] = [i // width, i % width]
            k += 1

    np.save(files_path_prefix + f'Forecast/Models/Clusters/rows_idxes.npy', rows_idxes)
    np.save(files_path_prefix + f'Forecast/Models/Clusters/frequencies_{filename}.npy', frequencies)
    # define the model
    model = KMeans(n_clusters)

    features = np.hstack((coordinates, frequencies))

    # fit the model
    model.fit(features)
    # assign a cluster to each example
    yhat = model.predict(features)
    np.save(files_path_prefix + f'Forecast/Models/Clusters/clusters_{filename}.npy', yhat)
    # retrieve unique clusters
    labels = np.unique(yhat)
    print(f'N clusters = {len(labels)}')
    np.save(files_path_prefix + f'Forecast/Models/Clusters/clusters_labels_{filename}.npy', labels)
    plot_clusters(files_path_prefix, features, labels, yhat, 'flux')
    return


def prepare_clusters(files_path_prefix: str,
                   array: np.ndarray,
                   clusters: np.ndarray,
                     n_lags: int,
                     n_forecast: int,
                     start_day = datetime.datetime(2019, 1, 1)):
    labels = np.unique(clusters)
    # for i in range(len(labels)):
    for i in range(1):
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

        x_train = np.zeros((train_len, amount, n_lags))
        y_train = np.zeros((train_len, amount, n_forecast))
        x_test = np.zeros((test_len, amount, n_lags))
        y_test = np.zeros((test_len, amount, n_forecast))

        for k in range(amount):
            row = row_idxes[0][k]
            cluster_part[k] = array[row]

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
