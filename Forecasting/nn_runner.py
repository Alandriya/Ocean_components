import datetime
import os
import warnings
from struct import unpack
# from skimage.metrics import structural_similarity as ssim
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tqdm
import argparse
from model import MyUnetModel, MyLSTMModel
import sys
from skimage.metrics import structural_similarity as ssim
from plotter import plot_predictions
from copy import deepcopy
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # files_path_prefix = 'E:/Nastya/Data/OceanFull/'
    np.set_printoptions(threshold=sys.maxsize)

    files_path_prefix = '/home/aosipova/EM_ocean/'
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("features_amount", type=int)
    parser.add_argument("start_year", type=int)
    args_cmd = parser.parse_args()
    features_amount = args_cmd.features_amount
    model_type = args_cmd.model_name
    start_year = args_cmd.start_year
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
    # start_year = 2019
    if start_year == 1979:
        offset = 0
    elif start_year == 1989:
        offset = days_delta1
    elif start_year == 1999:
        offset = days_delta1 + days_delta2
    elif start_year == 2009:
        offset = days_delta1 + days_delta2 + days_delta3
    else:
        offset = days_delta1 + days_delta2 + days_delta3 + days_delta4

    if start_year == 2019:
        end_year = 2025
    else:
        end_year = start_year + 10
    # ---------------------------------------------------------------------------------------
    # configs
    # width = 181
    # height = 161
    width = 91
    height = 81
    batch_size = 16
    days_known = 7
    days_prediction = 5

    # features_amount = 6
    # model_type = 'Unet'

    # features_amount = 3 + 3
    # 3 for flux, sst, press data, 3 for eigen0_flux, eigen0_sst, eigen0_press
    # 3 * 2 for (a_flux, b_flux), (a_sst, b_sst), (a_press, b_press),
    if model_type == 'Unet':
        mask_batch = np.zeros((height, width, days_prediction, 3))
        mask_batch[mask, :, :] = 1
    else:
        mask_batch = np.zeros((days_prediction, height, width, 3))
        mask_batch[:, mask, :] = 1
    # # ---------------------------------------------------------------------------------------
    # # load data
    # flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    # flux_array = np.diff(flux_array, axis=1)
    #
    # SST_array = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy')
    # SST_array = np.diff(SST_array, axis=1)
    #
    # press_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    # press_array = np.diff(press_array, axis=1)
    #
    # flux_array = flux_array.reshape((161, 181, -1))
    # flux_array = flux_array[::2, ::2, :]
    # SST_array = SST_array.reshape((161, 181, -1))
    # SST_array = SST_array[::2, ::2, :]
    # press_array = press_array.reshape((161, 181, -1))
    # press_array = press_array[::2, ::2, :]
    #
    # # # TODO delete
    # # flux_array = flux_array[:, :, :25]
    # # SST_array = SST_array[:, :, :25]
    # # press_array = press_array[:, :, :25]
    #
    # if start_year == 2019:
    #     train_len = int(flux_array.shape[2] * 3 / 5)
    #     test_len = flux_array.shape[2] - train_len - days_prediction - days_known
    # else:
    #     train_len = int(flux_array.shape[2]) - days_prediction - days_known - 1
    #     test_len = 0
    # train_days = [datetime.datetime(start_year, 1, 1) + datetime.timedelta(days=t) for t in range(train_len)]
    # test_days = [datetime.datetime(start_year, 1, 1) + datetime.timedelta(days=t) for t in
    #              range(train_len, train_len + test_len)]
    #
    # # (batch_size, 161, 181, days_known, features_amount)
    #
    # if model_type == 'Unet':
    #     x_train = np.zeros((train_len, height, width, days_known, features_amount), dtype=float)
    #     y_train = np.zeros((train_len, height, width, days_prediction, 3), dtype=float)
    #     eigenvectors_flux_sst = np.zeros((height, width, days_known))
    #     eigenvectors_sst_press = np.zeros((height, width, days_known))
    #     eigenvectors_flux_press = np.zeros((height, width, days_known))
    # else:
    #     x_train = np.zeros((train_len, days_known, height, width, features_amount), dtype=float)
    #     y_train = np.zeros((train_len, days_prediction, height, width, 3), dtype=float)
    #     eigenvectors_flux_sst = np.zeros((days_known, height, width))
    #     eigenvectors_sst_press = np.zeros((days_known, height, width))
    #     eigenvectors_flux_press = np.zeros((days_known, height, width))
    #
    # print('Preparing train', flush=True)
    # for t in range(train_len):
    #     # flux
    #     if model_type == 'Unet':
    #         x_train[t, :, :, :, 0] = flux_array[:, :, t:t+days_known]
    #         y_train[t, :, :, :, 0] = flux_array[:, :, t + days_known:t + days_known + days_prediction]
    #     else:
    #         for t1 in range(days_known):
    #             x_train[t, t1, :, :, 0] = flux_array[:, :, t + t1]
    #         for t1 in range(days_prediction):
    #             y_train[t, t1, :, :, 0] = flux_array[:, :, t + days_known + t1]
    #
    #     # sst
    #     if model_type == 'Unet':
    #         x_train[t, :, :, :, 1] = SST_array[:, :, t:t+days_known]
    #         y_train[t, :, :, :, 1] = SST_array[:, :, t + days_known:t + days_known + days_prediction]
    #     else:
    #         for t1 in range(days_known):
    #             x_train[t, t1, :, :, 1] = SST_array[:, :, t + t1]
    #         for t1 in range(days_prediction):
    #             y_train[t, t1, :, :, 1] = SST_array[:, :, t + days_known + t1]
    #
    #     # press
    #     if model_type == 'Unet':
    #         x_train[t, :, :, :, 2] = press_array[:, :, t:t+days_known]
    #         y_train[t, :, :, :, 2] = press_array[:, :, t + days_known:t + days_known + days_prediction]
    #     else:
    #         for t1 in range(days_known):
    #             x_train[t, t1, :, :, 2] = press_array[:, :, t + t1]
    #         for t1 in range(days_prediction):
    #             y_train[t, t1, :, :, 2] = press_array[:, :, t + days_known + t1]
    #     # --------------------------------------------------------------------------------------------------
    #     # eigenvectors
    #     if features_amount > 3:
    #         for t_lag in range(days_known):
    #             if model_type == 'lstm':
    #                 # flux - sst
    #                 try:
    #                     eigenvectors_flux_sst[t_lag] = np.load(files_path_prefix + f'Eigenvalues/Flux-SST/eigen0_{t + offset + t_lag}.npy').reshape(
    #                         (161, 181))[::2, ::2]
    #                 except FileNotFoundError:
    #                     print(f'Not existing Eigenvalues/Flux-SST/eigen0_{t + offset + t_lag}.npy')
    #                     eigenvectors_flux_sst[t_lag] = np.zeros((height, width))
    #
    #                 # sst - press
    #                 try:
    #                     eigenvectors_sst_press[t_lag] = np.load(files_path_prefix + f'Eigenvalues/SST-Pressure/eigen0_{t + offset + t_lag}.npy').reshape(
    #                         (161, 181))[::2, ::2]
    #                 except FileNotFoundError:
    #                     print(f'Not existing Eigenvalues/SST-Pressure/eigen0_{t + offset + t_lag}.npy')
    #                     eigenvectors_sst_press[t_lag] = np.zeros((height, width))
    #
    #                 # flux - press
    #                 try:
    #                     eigenvectors_flux_press[t_lag] = np.load(files_path_prefix + f'Eigenvalues/Flux-Pressure/eigen0_{t + offset + t_lag}.npy').reshape(
    #                         (161, 181))[::2, ::2]
    #                 except FileNotFoundError:
    #                     print(f'Not existing Eigenvalues/Flux-Pressure/eigen0_{t + offset + t_lag}.npy')
    #                     eigenvectors_flux_press[t_lag] = np.zeros((height, width))
    #             else:
    #                 # flux - sst
    #                 try:
    #                     eigenvectors_flux_sst[:, :, t_lag] = np.load(
    #                         files_path_prefix + f'Eigenvalues/Flux-SST/eigen0_{t + offset + t_lag}.npy').reshape(
    #                         (161, 181))[::2, ::2]
    #                 except FileNotFoundError:
    #                     print(f'Not existing Eigenvalues/Flux-SST/eigen0_{t + offset + t_lag}.npy')
    #                     eigenvectors_flux_sst[:, :, t_lag] = np.zeros((height, width))
    #
    #                 # sst - press
    #                 try:
    #                     eigenvectors_sst_press[:, :, t_lag] = np.load(
    #                         files_path_prefix + f'Eigenvalues/SST-Pressure/eigen0_{t + offset + t_lag}.npy').reshape(
    #                         (161, 181))[::2, ::2]
    #                 except FileNotFoundError:
    #                     print(f'Not existing Eigenvalues/SST-Pressure/eigen0_{t + offset + t_lag}.npy')
    #                     eigenvectors_sst_press[:, :, t_lag] = np.zeros((height, width))
    #
    #                 # flux - press
    #                 try:
    #                     eigenvectors_flux_press[:, :, t_lag] = np.load(
    #                         files_path_prefix + f'Eigenvalues/Flux-Pressure/eigen0_{t + offset + t_lag}.npy').reshape(
    #                         (161, 181))[::2, ::2]
    #                 except FileNotFoundError:
    #                     print(f'Not existing Eigenvalues/Flux-Pressure/eigen0_{t + offset + t_lag}.npy')
    #                     eigenvectors_flux_press[:, :, t_lag] = np.zeros((height, width))
    #
    #         x_train[t, :, :, :, 3] = eigenvectors_flux_sst
    #         x_train[t, :, :, :, 4] = eigenvectors_sst_press
    #         x_train[t, :, :, :, 5] = eigenvectors_flux_press
    #     # -------------------------------------------------------------------------------------------------
    #
    # # print(y_train[0, 0, :, :, 0])
    # np.nan_to_num(x_train, copy=False)
    # np.nan_to_num(y_train, copy=False)
    # np.save(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{model_type}_{features_amount}.npy', x_train)
    # np.save(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_{model_type}_{features_amount}.npy', y_train)
    # del x_train, y_train
    #
    # if model_type == 'Unet':
    #     x_test = np.zeros((test_len, height, width, days_known, features_amount), dtype=float)
    #     y_test = np.zeros((test_len, height, width, days_prediction, 3), dtype=float)
    # else:
    #     x_test = np.zeros((test_len, days_known, height, width, features_amount), dtype=float)
    #     y_test = np.zeros((test_len, days_prediction, height, width, 3), dtype=float)
    #
    # print('Preparing test', flush=True)
    # for t in range(test_len):
    #     t_absolute = train_len + t
    #     # flux
    #     if model_type == 'Unet':
    #         x_test[t, :, :, :, 0] = flux_array[:, :, t_absolute:t_absolute+days_known]
    #         y_test[t, :, :, :, 0] = flux_array[:, :, t_absolute + days_known:t_absolute + days_known + days_prediction]
    #     else:
    #         for t1 in range(days_known):
    #             x_test[t, t1, :, :, 0] = flux_array[:, :, t_absolute + t1]
    #         for t1 in range(days_prediction):
    #             y_test[t, t1, :, :, 0] = flux_array[:, :, t_absolute + days_known + t1]
    #
    #     # sst
    #     if model_type == 'Unet':
    #         x_test[t, :, :, :, 1] = SST_array[:, :, t_absolute:t_absolute+days_known]
    #         y_test[t, :, :, :, 1] = SST_array[:, :, t_absolute + days_known:t_absolute + days_known + days_prediction]
    #     else:
    #         for t1 in range(days_known):
    #             x_test[t, t1, :, :, 1] = SST_array[:, :, t_absolute + t1]
    #         for t1 in range(days_prediction):
    #             y_test[t, t1, :, :, 1] = SST_array[:, :, t_absolute + days_known + t1]
    #
    #     # press
    #     if model_type == 'Unet':
    #         x_test[t, :, :, :, 2] = press_array[:, :, t_absolute:t_absolute+days_known]
    #         y_test[t, :, :, :, 2] = press_array[:, :, t_absolute + days_known:t_absolute + days_known + days_prediction]
    #     else:
    #         for t1 in range(days_known):
    #             x_test[t, t1, :, :, 2] = press_array[:, :, t_absolute + t1]
    #         for t1 in range(days_prediction):
    #             y_test[t, t1, :, :, 2] = press_array[:, :, t_absolute + days_known + t1]
    #
    #     # --------------------------------------------------------------------------------------------------
    #     # eigenvectors
    #     if features_amount > 3:
    #         for t_lag in range(days_known):
    #             if model_type == 'lstm':
    #                 # flux - sst
    #                 try:
    #                     eigenvectors_flux_sst[t_lag] = np.load(
    #                         files_path_prefix + f'Eigenvalues/Flux-SST/eigen0_{t_absolute + offset + t_lag}.npy').reshape(
    #                         (161, 181))[::2, ::2]
    #                 except FileNotFoundError:
    #                     print(f'Not existing Eigenvalues/Flux-SST/eigen0_{t_absolute + offset + t_lag}.npy')
    #                     eigenvectors_flux_sst[t_lag] = np.zeros((height, width))
    #
    #                 # sst - press
    #                 try:
    #                     eigenvectors_sst_press[t_lag] = np.load(
    #                         files_path_prefix + f'Eigenvalues/SST-Pressure/eigen0_{t_absolute + offset + t_lag}.npy').reshape(
    #                         (161, 181))[::2, ::2]
    #                 except FileNotFoundError:
    #                     print(f'Not existing Eigenvalues/SST-Pressure/eigen0_{t_absolute + offset + t_lag}.npy')
    #                     eigenvectors_sst_press[t_lag] = np.zeros((height, width))
    #
    #                 # flux - press
    #                 try:
    #                     eigenvectors_flux_press[t_lag] = np.load(
    #                         files_path_prefix + f'Eigenvalues/Flux-Pressure/eigen0_{t_absolute + offset + t_lag}.npy').reshape(
    #                         (161, 181))[::2, ::2]
    #                 except FileNotFoundError:
    #                     print(f'Not existing Eigenvalues/Flux-Pressure/eigen0_{t_absolute + offset + t_lag}.npy')
    #                     eigenvectors_flux_press[t_lag] = np.zeros((height, width))
    #             else:
    #                 # flux - sst
    #                 try:
    #                     eigenvectors_flux_sst[:, :, t_lag] = np.load(
    #                         files_path_prefix + f'Eigenvalues/Flux-SST/eigen0_{t_absolute + offset + t_lag}.npy').reshape(
    #                         (161, 181))[::2, ::2]
    #                 except FileNotFoundError:
    #                     print(f'Not existing Eigenvalues/Flux-SST/eigen0_{t_absolute + offset + t_lag}.npy')
    #                     eigenvectors_flux_sst[:, :, t_lag] = np.zeros((height, width))
    #
    #                 # sst - press
    #                 try:
    #                     eigenvectors_sst_press[:, :, t_lag] = np.load(
    #                         files_path_prefix + f'Eigenvalues/SST-Pressure/eigen0_{t_absolute + offset + t_lag}.npy').reshape(
    #                         (161, 181))[::2, ::2]
    #                 except FileNotFoundError:
    #                     print(f'Not existing Eigenvalues/SST-Pressure/eigen0_{t_absolute + offset + t_lag}.npy')
    #                     eigenvectors_sst_press[:, :, t_lag] = np.zeros((height, width))
    #
    #                 # flux - press
    #                 try:
    #                     eigenvectors_flux_press[:, :, t_lag] = np.load(
    #                         files_path_prefix + f'Eigenvalues/Flux-Pressure/eigen0_{t_absolute + offset + t_lag}.npy').reshape(
    #                         (161, 181))[::2, ::2]
    #                 except FileNotFoundError:
    #                     print(f'Not existing Eigenvalues/Flux-Pressure/eigen0_{t_absolute + offset + t_lag}.npy')
    #                     eigenvectors_flux_press[:, :, t_lag] = np.zeros((height, width))
    #
    #         x_test[t, :, :, :, 3] = eigenvectors_flux_sst
    #         x_test[t, :, :, :, 4] = eigenvectors_sst_press
    #         x_test[t, :, :, :, 5] = eigenvectors_flux_press
    #     # -------------------------------------------------------------------------------------------------
    #
    # # print(y_test[0, 0, :, :, 0])
    #
    # np.nan_to_num(x_test, copy=False)
    # np.nan_to_num(y_test, copy=False)
    # np.save(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{model_type}_{features_amount}.npy', x_test)
    # np.save(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{model_type}_{features_amount}.npy', y_test)
    # del x_test, y_test
    # # raise ValueError
    # # # ---------------------------------------------------------------------------------------
    if not os.path.exists(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_train_ds_{model_type}_{features_amount}'):
        # preparing datasets
        print('Creating train_ds', flush=True)

        x_train = np.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{model_type}_{features_amount}.npy')
        y_train = np.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_{model_type}_{features_amount}.npy')
        print(f'{model_type} days_prediction = {days_prediction} features = {features_amount}')

        # normalize and save min and max
        x_mins = np.amin(x_train, axis=(0, 1, 2, 3))
        x_maxs = np.amax(x_train, axis=(0, 1, 2, 3))
        # print(f'X_mins')
        # print(x_mins)
        # print(f'X_maxs')
        # print(x_maxs)

        y_mins = np.amin(y_train, axis=(0, 1, 2, 3))
        y_maxs = np.amax(y_train, axis=(0, 1, 2, 3))
        # print(f'y_mins')
        # print(y_mins)
        # print(f'y_maxs')
        # print(y_maxs)

        if not os.path.exists(files_path_prefix + f'Forecast/Normalize'):
            os.mkdir(files_path_prefix + f'Forecast/Normalize')
        np.save(files_path_prefix + f'Forecast/Normalize/x_mins_{start_year}-{end_year}_{features_amount}.npy', x_mins)
        np.save(files_path_prefix + f'Forecast/Normalize/x_maxs_{start_year}-{end_year}_{features_amount}.npy', x_maxs)
        np.save(files_path_prefix + f'Forecast/Normalize/y_mins_{start_year}-{end_year}.npy', y_mins)
        np.save(files_path_prefix + f'Forecast/Normalize/y_maxs_{start_year}-{end_year}.npy', y_maxs)

        # normalize train
        for i in range(features_amount):
            x_train[:, :, :, :, i] = (x_train[:, :, :, :, i] - x_mins[i])/(x_maxs[i]-x_mins[i])
        for i in range(3):
            y_train[:, :, :, :, i] = (y_train[:, :, :, :, i] - y_mins[i])/(y_maxs[i]-y_mins[i])

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size)
        del x_train, y_train
        tf.data.Dataset.save(train_ds, files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_train_ds_{model_type}_{features_amount}')
        del train_ds

        print('Creating test_ds', flush=True)
        x_test = np.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{model_type}_{features_amount}.npy')
        y_test = np.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{model_type}_{features_amount}.npy')

        # normalize test
        for i in range(features_amount):
            x_test[:, :, :, :, i] = (x_test[:, :, :, :, i] - x_mins[i]) / (x_maxs[i] - x_mins[i])
        for i in range(3):
            y_test[:, :, :, :, i] = (y_test[:, :, :, :, i] - y_mins[i]) / (y_maxs[i] - y_mins[i])

        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        del x_test, y_test
        tf.data.Dataset.save(test_ds, files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_test_ds_{model_type}_{features_amount}')
        del test_ds
    # # raise ValueError
    # ---------------------------------------------------------------------------------------
    # construct model
    if model_type == 'Unet':
        model = MyUnetModel(days_prediction, mask_batch)
    else:
        model = MyLSTMModel(days_prediction, mask_batch)
    model.compile()

    # if model_type == 'Unet':
    #     model.build(input_shape=(None, height, width, days_known, features_amount))
    #     model.make((height, width, days_known, features_amount))
    # else:
    #     model.build(input_shape=(None, days_known, height, width, features_amount))
    #     model.make((days_known, height, width, features_amount))

    loss_object = tf.keras.losses.MeanSquaredError()

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')
    # train_accuracy = tf.image.ssim

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')
    # test_accuracy = tf.image.ssim
    # ---------------------------------------------------------------------------------------
    @tf.function
    def train_step(x_train, y_train):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(x_train, training=True)
            # print(predictions.shape)
            # print(y_train.shape)
            loss = loss_object(y_train, predictions)
            # print('TRAIN', flush=True)
            # print(loss)
            # print('\n')
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(y_train, predictions)

    @tf.function
    def test_step(x_test, y_test):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(x_test, training=False)
        t_loss = loss_object(y_test, predictions)
        # print('TEST', flush=True)
        # tf.print(y_test[0, :, :, 0], flush=True)
        # tf.print(predictions[0, :, :, 0], flush=True)

        test_loss(t_loss)
        test_accuracy(y_test, predictions)
    # ---------------------------------------------------------------------------------------
    train_ds = tf.data.Dataset.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_train_ds_{model_type}_{features_amount}')
    test_ds = tf.data.Dataset.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_test_ds_{model_type}_{features_amount}')
    y_mins = np.load(files_path_prefix + f'Forecast/Normalize/y_mins_{start_year}-{end_year}.npy')
    y_maxs = np.load(files_path_prefix + f'Forecast/Normalize/y_maxs_{start_year}-{end_year}.npy')

    # train
    EPOCHS = 25

    # Loads the weights
    checkpoint_path = files_path_prefix + f'Forecast/Checkpoints/my_checkpoint_{model_type}_{features_amount}'
    # if start_year != 2019:
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    if start_year == 2019:
        for epoch in range(EPOCHS):
            # Reset the metrics at the start of the next epoch
            train_loss.reset_state()
            train_accuracy.reset_state()
            test_loss.reset_state()
            test_accuracy.reset_state()

            for x_train, y_train in train_ds:
                train_step(x_train, y_train)

            for x_test, y_test in test_ds:
                test_step(x_test, y_test)

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {train_loss.result():0.5f}, '
                f'Accuracy: {train_accuracy.result() * 100:0.5f}, '
                f'Test Loss: {test_loss.result():0.5f}, '
                f'Test Accuracy: {test_accuracy.result() * 100:0.5f}'
            )

            # Save the weights
            model.save_weights(files_path_prefix + f'Forecast/Checkpoints/my_checkpoint_{model_type}_{features_amount}')

        x_test = np.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{model_type}_{features_amount}.npy')
        y_test = np.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{model_type}_{features_amount}.npy')
        x_mins = np.load(files_path_prefix + f'Forecast/Normalize/x_mins_{start_year}-{end_year}_{features_amount}.npy')
        x_maxs = np.load(files_path_prefix + f'Forecast/Normalize/x_maxs_{start_year}-{end_year}_{features_amount}.npy')
        y_mins = np.load(files_path_prefix + f'Forecast/Normalize/y_mins_{start_year}-{end_year}.npy')
        y_maxs = np.load(files_path_prefix + f'Forecast/Normalize/y_maxs_{start_year}-{end_year}.npy')

        # normalize test
        for i in range(features_amount):
            x_test[:, :, :, :, i] = (x_test[:, :, :, :, i] - x_mins[i]) / (x_maxs[i] - x_mins[i])
        # for i in range(3):
        #     y_test[:, :, :, :, i] = (y_test[:, :, :, :, i] - y_mins[i]) / (y_maxs[i] - y_mins[i])

        # reverse normalization of prediction

        ssim_flux, ssim_sst, ssim_press = 0, 0, 0
        for i in range(10):
            y = y_test[i]
            pred = model(x_test[i:i+1], training=False)[0].numpy()
            for k in range(3):
                pred[:, :, :, k] *= (y_maxs[k] - y_mins[k])
                pred[:, :, :, k] += y_mins[k]

            start_day = datetime.datetime(start_year, 1, 1) + datetime.timedelta(days=len(train_ds) + i)
            print(f'max(pred) = {np.nanmax(pred)}')
            print(f'max(test) = {np.nanmax(y)}')
            print(f'min(pred) = {np.nanmin(pred)}')
            print(f'min(test) = {np.nanmin(y)}')
            plot_predictions(files_path_prefix, deepcopy(y), deepcopy(pred), model_type, features_amount, start_day, mask)

            pred = np.nan_to_num(pred)
            y = np.nan_to_num(y)
            for t1 in range(days_prediction):
                if model_type == 'Unet':
                    y_flux = y[:, :, t1, 0]
                    pred_flux = pred[:, :, t1, 0]
                    y_sst = y[:, :, t1, 1]
                    pred_sst = pred[:, :, t1, 1]
                    y_press = y[:, :, t1, 2]
                    pred_press = pred[:, :, t1, 2]
                else:
                    y_flux = y[t1, :, :, 0]
                    pred_flux = pred[t1, :, :, 0]
                    y_sst = y[t1, :, :, 1]
                    pred_sst = pred[t1, :, :, 1]
                    y_press = y[t1, :, :, 2]
                    pred_press = pred[t1, :, :, 2]

                flux_range = max([np.max(y_flux), np.max(pred_flux)]) - \
                             min([np.min(y_flux), np.min(pred_flux)])

                sst_range = max([np.max(y_sst), np.max(pred_sst)]) - \
                             min([np.min(y_sst), np.min(pred_sst)])

                press_range = max([np.max(y_press), np.max(pred_press)]) - \
                             min([np.min(y_press), np.min(pred_press)])

                ssim_flux += ssim(y_flux, pred_flux, data_range=y_maxs[0] - y_mins[0])
                ssim_sst += ssim(y_sst, pred_sst, data_range=y_maxs[1] - y_mins[1])
                ssim_press += ssim(y_press, pred_press, data_range=y_maxs[2] - y_mins[2])

        ssim_flux /= days_prediction
        ssim_sst /= days_prediction
        ssim_press /= days_prediction

        ssim_flux /= 10
        ssim_sst /= 10
        ssim_press /= 10
        print(f'{model_type} {features_amount} Flux SSIM = {ssim_flux * 100: .2f}')
        print(f'{model_type} {features_amount} SST SSIM = {ssim_sst * 100: .2f}')
        print(f'{model_type} {features_amount} Press SSIM = {ssim_press * 100: .2f}')
        # # Visualize the model
        # tf.keras.utils.plot_model(model, to_file=files_path_prefix + f'Forecast/model_{model_type}.png', expand_nested=True, dpi=60,
        #                           show_shapes=True, show_layer_names=True)
    else:
        for epoch in range(EPOCHS):
            # Reset the metrics at the start of the next epoch
            train_loss.reset_state()
            train_accuracy.reset_state()

            for x_train, y_train in train_ds:
                train_step(x_train, y_train)

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {train_loss.result():0.5f}, '
                f'Accuracy: {train_accuracy.result() * 100:0.5f}, '
            )
            # Save the weights
            model.save_weights(files_path_prefix + f'Forecast/Checkpoints/my_checkpoint_{model_type}_{features_amount}')