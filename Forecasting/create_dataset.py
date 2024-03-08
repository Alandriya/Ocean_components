import numpy as np
from struct import unpack

# files_path_prefix = 'E:/Nastya/Data/OceanFull/'
files_path_prefix = 'D://Data/OceanFull/'
width = 181
height = 161

if __name__ == '__main__':
    n_lags = 14
    n_forecast = 7
    postfix = 'sensible'

    sensible = np.load(files_path_prefix + f'Fluxes/sensible_grouped_2019-2022.npy')
    # latent = np.load(files_path_prefix + f'Fluxes/latent_grouped_2019-2022.npy')
    array = sensible

    train_len = int(array.shape[1] * 2/3.0)
    test_len = array.shape[1] - train_len - n_lags*2 - n_forecast

    x_train = np.zeros((train_len, n_lags, height, width))
    y_train = np.zeros((train_len, n_forecast, height, width))
    x_test = np.zeros((test_len, n_lags, height, width))
    y_test = np.zeros((test_len, n_forecast, height, width))

    for t in range(0, train_len):
        x_train[t] = array[:, t:t+n_lags].transpose().reshape((n_lags, height, width))
        y_train[t] = array[:, t+n_lags+1:t+n_lags+1+n_forecast].transpose().reshape((n_forecast, height, width))

    for t in range(0, test_len):
        x_test[t] = array[:, train_len+n_lags+t:train_len+n_lags + t + n_lags].transpose().reshape((n_lags, height, width))
        y_test[t] = array[:, train_len+n_lags*2+t+1:train_len+n_lags*2+t+1 + n_forecast].transpose().reshape((n_forecast, height, width))

    np.save(files_path_prefix + f'Forecast/Train/X_train_{postfix}.npy', x_train)
    np.save(files_path_prefix + f'Forecast/Test/X_test_{postfix}.npy', x_test)
    np.save(files_path_prefix + f'Forecast/Train/Y_train_{postfix}.npy', y_train)
    np.save(files_path_prefix + f'Forecast/Test/Y_test_{postfix}.npy', y_test)
