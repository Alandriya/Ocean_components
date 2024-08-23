from struct import unpack
import datetime
import numpy as np
import tqdm
from sklearn.metrics import mean_absolute_percentage_error
from plotter import plot_predictions
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
# import tensorflow as tf
import sys

if __name__ == '__main__':
    # files_path_prefix = 'E:/Nastya/Data/OceanFull/'
    files_path_prefix = '/home/aosipova/EM_ocean/'
    np.set_printoptions(threshold=sys.maxsize)
    # --------------------------------------------------------------------------------
    # Mask
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    mask = mask.reshape((161, 181))[::2, ::2]
    # ---------------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    start_year = 2019
    if start_year == 2019:
        end_year = 2025
    else:
        end_year = start_year + 10

    offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    # ---------------------------------------------------------------------------------------
    # configs
    width = 91
    height = 81
    batch_size = 16
    days_known = 7
    days_prediction = 5
    features_amount = 3
    # ---------------------------------------------------------------------------------------
    x_mins = np.load(files_path_prefix + f'Forecast/Normalize/x_mins_{start_year}-{end_year}_6.npy')
    x_maxs = np.load(files_path_prefix + f'Forecast/Normalize/x_maxs_{start_year}-{end_year}_6.npy')
    y_mins = np.load(files_path_prefix + f'Forecast/Normalize/y_mins_{start_year}-{end_year}.npy')
    y_maxs = np.load(files_path_prefix + f'Forecast/Normalize/y_maxs_{start_year}-{end_year}.npy')
    print(x_mins)
    print(x_maxs)
    print(y_mins)
    print(y_maxs)
    # raise ValueError

    x_train = np.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_Unet_{features_amount}.npy')
    y_train = np.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_Unet_{features_amount}.npy')

    x_test = np.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_Unet_{features_amount}.npy')
    y_test = np.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_Unet_{features_amount}.npy')

    x_train[:, np.logical_not(mask)] = 0
    x_test[:, np.logical_not(mask)] = 0
    y_train[:, np.logical_not(mask)] = 0
    y_test[:, np.logical_not(mask)] = 0

    flux_error, sst_error, press_error = 0, 0, 0
    # get dumb prediction and plot it
    for t in range(10):
        y_pred = np.mean(y_train[-days_known:, :, :, :, :], axis=0)
        start_day = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=offset + x_train.shape[0] + t)
        # plot_predictions(files_path_prefix, y_test[t], y_pred, 'Dumb', features_amount, start_day, mask)
        y_pred[np.logical_not(mask)] = 0

        for t1 in range(days_prediction):
            # flux_error += np.sqrt(mean_squared_error(y_pred[t1, :, :, 0], y_test[t, t1, :, :, 0]))
            # sst_error += np.sqrt(mean_squared_error(y_pred[t1, :, :, 1], y_test[t, t1, :, :, 1]))
            # press_error += np.sqrt(mean_squared_error(y_pred[t1, :, :, 2], y_test[t, t1, :, :, 2]))
            flux_error += ssim(np.array(y_pred[:, :, t1, 0]), np.array(y_test[t, :, :, t1, 0]), data_range=1)
            sst_error +=ssim(y_pred[:, :, t1, 1], y_test[t, :, :, t1, 1], data_range=1)
            press_error += ssim(y_pred[:, :, t1, 2], y_test[t, :, :, t1, 2], data_range=1)

    flux_error /= days_prediction
    sst_error /= days_prediction
    press_error /= days_prediction
    flux_error /= 10
    sst_error /= 10
    press_error /= 10

    print(f'Mean Flux SSIM = {flux_error * 100: .2f}')
    print(f'Mean SST SSIM = {sst_error * 100: .2f}')
    print(f'Mean Press SSIM = {press_error * 100: .2f}')

    flux_error, sst_error, press_error = 0, 0, 0
    # get regression prediction and plot it
    for t in range(10):
        y_pred = np.zeros((height, width, days_prediction, 3), dtype=float)
        for k in range(3):
            for i in range(height):
                for j in range(width):
                    if mask[i, j]:
                        regr = linear_model.LinearRegression()
                        regr.fit(x_train[:, i, j, -days_prediction:].reshape((-1, features_amount)), y_train[:, i, j, :days_prediction, k].flatten())
                        y_pred[i, j, :, k] = regr.predict(x_test[0:1, i, j, -days_prediction:].reshape((-1, features_amount)))
        start_day = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=offset + x_train.shape[0] + t)
        plot_predictions(files_path_prefix, y_test[t], y_pred, 'Linear regression', features_amount, start_day, mask)

        y_pred = np.array(y_pred, dtype=float)
        for t1 in range(days_prediction):
            # flux_error += np.sqrt(mean_squared_error(y_pred[:, :, t1, 0], y_test[t, :, :, t1, 0]))
            # sst_error += np.sqrt(mean_squared_error(y_pred[:, :, t1, 1], y_test[t, :, :, t1, 1]))
            # press_error += np.sqrt(mean_squared_error(y_pred[:, :, t1, 2], y_test[t, :, :, t1, 2]))
            flux_error += ssim(y_pred[:, :, t1, 0], y_test[t, :, :, t1, 0], data_range=1)
            sst_error += ssim(y_pred[:, :, t1, 1], y_test[t, :, :, t1, 1], data_range=1)
            press_error += ssim(y_pred[:, :, t1, 2], y_test[t, :, :, t1, 2], data_range=1)

    flux_error /= days_prediction
    sst_error /= days_prediction
    press_error /= days_prediction
    flux_error /= 10
    sst_error /= 10
    press_error /= 10

    print(f'Regression Flux SSIM = {flux_error * 100: .2f}')
    print(f'Regression SST SSIM = {sst_error * 100: .2f}')
    print(f'Regression Press SSIM = {press_error * 100: .2f}')

