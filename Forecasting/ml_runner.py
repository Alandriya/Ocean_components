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
import tensorflow as tf


if __name__ == '__main__':
    files_path_prefix = 'E:/Nastya/Data/OceanFull/'
    # files_path_prefix = '/home/aosipova/EM_ocean/'
    # --------------------------------------------------------------------------------
    # Mask
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    mask = mask.reshape((161, 181))
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
    width = 181
    height = 161
    batch_size = 32
    days_known = 10
    days_prediction = 5
    features_amount = 9
    # 3 for flux, sst, press data, 3 * 2 for (a_flux, b_flux), (a_sst, b_sst), (a_press, b_press), 3 for eigen0_flux,
    # eigen0_sst, eigen0_press

    mask_batch = np.zeros((batch_size, 161, 181, days_prediction * 3))
    mask_batch[:, mask, :] = 0
    # ---------------------------------------------------------------------------------------
    # x_train = np.zeros((train_len, height, width, days_known, features_amount), dtype=float)
    # y_train = np.zeros((train_len, height, width, days_prediction, 3), dtype=float)

    # x_train = np.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train.npy')
    # y_train = np.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train.npy')

    # x_test = np.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test.npy')
    # y_test = np.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test.npy')

    # # get dumb prediction and plot it
    # for t in range(1):
    #     y_pred = np.mean(y_train[-days_known:, :, :, :, :], axis=0)
    #     start_day = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=offset + x_train.shape[0] + t)
    #     # plot_predictions(files_path_prefix, y_test[t], y_pred, 'Dumb', start_day, mask)
    #
    #     flux_error, sst_error, press_error = 0, 0, 0
    #     for t1 in range(days_prediction):
    #         flux_error += np.sqrt(mean_squared_error(y_pred[:, :, t1, 0], y_test[t, :, :, t1, 0]))
    #         sst_error += np.sqrt(mean_squared_error(y_pred[:, :, t1, 1], y_test[t, :, :, t1, 1]))
    #         press_error += np.sqrt(mean_squared_error(y_pred[:, :, t1, 2], y_test[t, :, :, t1, 2]))
    #         # flux_error += ssim(y_pred[:, :, t1, 0], y_test[t, :, :, t1, 0])
    #         # sst_error +=ssim(y_pred[:, :, t1, 1], y_test[t, :, :, t1, 1])
    #         # press_error += ssim(y_pred[:, :, t1, 2], y_test[t, :, :, t1, 2])
    #
    #     flux_error /= days_prediction
    #     sst_error /= days_prediction
    #     press_error /= days_prediction
    #
    #     print(f'Flux RMSE = {flux_error: .2e}')
    #     print(f'SST RMSE = {sst_error: .2e}')
    #     print(f'Press RMSE = {press_error: .2e}')

        # print(f'Flux SSIM = {flux_error: .2f}')
        # print(f'SST SSIM = {sst_error: .2f}')
        # print(f'Press SSIM = {press_error: .2f}')
    # raise ValueError


    # # get regression prediction and plot it
    # for t in range(1):
    #     y_pred = np.zeros((161, 181, days_prediction, 3))
    #     for k in range(3):
    #         for i in tqdm.tqdm(range(161)):
    #             for j in range(181):
    #                 if mask[i, j]:
    #                     regr = linear_model.LinearRegression()
    #                     regr.fit(x_train[-days_known:, i, j, :, :].reshape((-1, features_amount)), y_train[-days_known:, i, j, :, k].flatten())
    #                     y_pred[i, j, :, k] = regr.predict(x_test[0, i, j, :, :].reshape((-1, features_amount)))[:days_prediction]
    #     start_day = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=offset + x_train.shape[0] + t)
    #     # plot_predictions(files_path_prefix, y_test[t], y_pred, 'Linear regression', start_day, mask)
    #
    #     flux_error, sst_error, press_error = 0, 0, 0
    #     for t1 in range(days_prediction):
    #         # flux_error += np.sqrt(mean_squared_error(y_pred[:, :, t1, 0], y_test[t, :, :, t1, 0]))
    #         # sst_error += np.sqrt(mean_squared_error(y_pred[:, :, t1, 1], y_test[t, :, :, t1, 1]))
    #         # press_error += np.sqrt(mean_squared_error(y_pred[:, :, t1, 2], y_test[t, :, :, t1, 2]))
    #         flux_error += ssim(y_pred[:, :, t1, 0], y_test[t, :, :, t1, 0])
    #         sst_error +=ssim(y_pred[:, :, t1, 1], y_test[t, :, :, t1, 1])
    #         press_error += ssim(y_pred[:, :, t1, 2], y_test[t, :, :, t1, 2])
    #
    #     flux_error /= days_prediction
    #     sst_error /= days_prediction
    #     press_error /= days_prediction
    #
    #     # print(f'Flux RMSE = {flux_error: .2e}')
    #     # print(f'SST RMSE = {sst_error: .2e}')
    #     # print(f'Press RMSE = {press_error: .2e}')
    #
    #     print(f'Flux SSIM = {flux_error: .2f}')
    #     print(f'SST SSIM = {sst_error: .2f}')
    #     print(f'Press SSIM = {press_error: .2f}')

