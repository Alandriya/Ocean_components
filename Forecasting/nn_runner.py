import datetime
import os
import warnings
from struct import unpack
from skimage.metrics import structural_similarity as ssim
import numpy as np
import tensorflow as tf
import tqdm

from model import MyUnetModel, MyLSTMModel

warnings.filterwarnings("ignore")

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

    # t_start = days_delta1 + days_delta2 + days_delta3 + days_delta4
    # t_end = t_start + days_delta6 - 1
    offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    # ---------------------------------------------------------------------------------------
    # configs
    width = 181
    height = 161
    batch_size = 32
    days_known = 10
    days_prediction = 5
    features_amount = 3
    model_type = 'lstm'

    # features_amount = 3 + 3*2
    # 3 for flux, sst, press data, 3 * 2 for (a_flux, b_flux), (a_sst, b_sst), (a_press, b_press), 3 for eigen0_flux,
    # eigen0_sst, eigen0_press
    if model_type == 'Unet':
        mask_batch = np.zeros((batch_size, 161, 181, days_known, 3))
        mask_batch[:, mask, :, :] = 1
    else:
        mask_batch = np.zeros((batch_size, 161, 181, days_prediction * 3))
        mask_batch[:, mask, :] = 1
    # ---------------------------------------------------------------------------------------
    # load data
    flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    flux_array = np.diff(flux_array, axis=1)

    SST_array = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy')
    SST_array = np.diff(SST_array, axis=1)

    press_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    press_array = np.diff(press_array, axis=1)

    flux_array = flux_array.reshape((height, width, -1))
    SST_array = SST_array.reshape((height, width, -1))
    press_array = press_array.reshape((height, width, -1))

    # TODO delete
    flux_array = flux_array[:, :, :500]
    SST_array = SST_array[:, :, :500]
    press_array = press_array[:, :, :500]

    train_len = int(flux_array.shape[2] * 3 / 5)
    train_days = [datetime.datetime(start_year, 1, 1) + datetime.timedelta(days=t) for t in range(train_len)]
    test_len = flux_array.shape[2] - train_len - days_prediction - days_known
    test_days = [datetime.datetime(start_year, 1, 1) + datetime.timedelta(days=t) for t in
                 range(train_len, train_len + test_len)]

    # (batch_size, 161, 181, days_known, features_amount)

    if model_type == 'Unet':
        x_train = np.zeros((train_len, height, width, days_known, features_amount), dtype=float)
    else:
        x_train = np.zeros((train_len, days_known, height, width, features_amount), dtype=float)
    y_train = np.zeros((train_len, height, width, days_prediction, 3), dtype=float)

    print('Preparing train', flush=True)
    for t in range(train_len):
        # flux
        if model_type == 'Unet':
            x_train[t, :, :, :, 0] = flux_array[:, :, t:t+days_known]
        else:
            x_train[t, :, :, :, 0] = flux_array[:, :, t:t + days_known].reshape((days_known, height, width))
        y_train[t, :, :, :, 0] = flux_array[:, :, t+days_known:t+days_known+days_prediction]

        # sst
        if model_type == 'Unet':
            x_train[t, :, :, :, 1] = SST_array[:, :, t:t+days_known]
        else:
            x_train[t, :, :, :, 1] = SST_array[:, :, t:t + days_known].reshape((days_known, height, width))
        y_train[t, :, :, :, 1] = SST_array[:, :, t+days_known:t+days_known+days_prediction]

        # press
        if model_type == 'Unet':
            x_train[t, :, :, :, 2] = press_array[:, :, t:t+days_known]
        else:
            x_train[t, :, :, :, 2] = press_array[:, :, t:t + days_known].reshape((days_known, height, width))
        y_train[t, :, :, :, 2] = press_array[:, :, t+days_known:t+days_known+days_prediction]

    np.nan_to_num(x_train, copy=False)
    np.nan_to_num(y_train, copy=False)
    np.save(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{model_type}.npy', x_train)
    np.save(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train.npy_{model_type}', y_train)
    del x_train, y_train

    if model_type == 'Unet':
        x_test = np.zeros((test_len, height, width, days_known, features_amount), dtype=float)
    else:
        x_test = np.zeros((test_len, days_known, height, width, features_amount), dtype=float)
    y_test = np.zeros((test_len, height, width, days_prediction, 3), dtype=float)

    print('Preparing test', flush=True)
    for t in range(test_len):
        t_absolute = train_len + t
        # flux
        if model_type == 'Unet':
            x_test[t, :, :, :, 0] = flux_array[:, :, t_absolute:t_absolute+days_known]
        else:
            x_test[t, :, :, :, 0] = flux_array[:, :, t_absolute:t_absolute + days_known].reshape(
                (days_known, height, width))
        y_test[t, :, :, :, 0] = flux_array[:, :, t_absolute+days_known:t_absolute+days_known+days_prediction]
        # sst
        if model_type == 'Unet':
            x_test[t, :, :, :, 1] = SST_array[:, :, t_absolute:t_absolute+days_known]
        else:
            x_test[t, :, :, :, 1] = SST_array[:, :, t_absolute:t_absolute + days_known].reshape(
                (days_known, height, width))
        y_test[t, :, :, :, 1] = SST_array[:, :, t_absolute+days_known:t_absolute+days_known+days_prediction]
        # press
        if model_type == 'Unet':
            x_test[t, :, :, :, 2] = press_array[:, :, t_absolute:t_absolute+days_known]
        else:
            x_test[t, :, :, :, 2] = press_array[:, :, t_absolute:t_absolute + days_known].reshape(
                (days_known, height, width))
        y_test[t, :, :, :, 2] = press_array[:, :, t_absolute+days_known:t_absolute+days_known+days_prediction]

    np.nan_to_num(x_test, copy=False)
    np.nan_to_num(y_test, copy=False)
    np.save(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{model_type}.npy', x_test)
    np.save(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{model_type}.npy', y_test)
    # raise ValueError
    # # ---------------------------------------------------------------------------------------
    x_train = np.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{model_type}.npy')
    y_train = np.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_{model_type}.npy')
    print('Check nans in train')
    print(np.isnan(x_train).any())
    print(np.isnan(y_train).any())
    # preparing datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(100).batch(batch_size)
    del x_train, y_train
    tf.data.Dataset.save(train_ds, files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_train_ds_{model_type}')
    del train_ds

    x_test = np.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{model_type}.npy')
    y_test = np.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{model_type}.npy')
    print(np.isnan(x_test).any())
    print(np.isnan(y_test).any())
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    del x_test, y_test
    tf.data.Dataset.save(test_ds, files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_test_ds_{model_type}')
    del test_ds
    # raise ValueError
    # ---------------------------------------------------------------------------------------
    # construct model
    model = MyUnetModel(days_prediction, mask_batch)
    # model.compile()

    model.build(input_shape=(batch_size, height, width, days_known, features_amount))
    model.make((height, width, days_known, features_amount))

    loss_object = tf.keras.losses.MeanSquaredError()

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')
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
            # tf.print(y_train)
            # tf.print(predictions)
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

        test_loss(t_loss)
        test_accuracy(y_test, predictions)
    # ---------------------------------------------------------------------------------------
    train_ds = tf.data.Dataset.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_train_ds_{model_type}')
    test_ds = tf.data.Dataset.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_test_ds_{model_type}')

    # train
    EPOCHS = 2

    # Loads the weights
    # model.load_weights(checkpoint_path)

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_state()
        train_accuracy.reset_state()
        test_loss.reset_state()
        test_accuracy.reset_state()

        for x_train, y_train in train_ds:
            train_step(x_train, y_train)

        for x_test, y_test in train_ds:
            train_step(x_test, y_test)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result():0.2f}, '
            f'Accuracy: {train_accuracy.result() * 100:0.2f}, '
            f'Test Loss: {test_loss.result():0.2f}, '
            f'Test Accuracy: {test_accuracy.result() * 100:0.2f}'
        )

        # Visualize the model
        tf.keras.utils.plot_model(model, to_file=files_path_prefix + f'Forecast/model_{model_type}.png', expand_nested=True, dpi=60,
                                  show_shapes=True, show_layer_names=True)

        # Save the weights
        model.save_weights(files_path_prefix + f'Forecast/Checkpoints/my_checkpoint_{model_type}')
