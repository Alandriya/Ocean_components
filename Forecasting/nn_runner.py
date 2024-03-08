from nn_model import VideoPrediction
import numpy as np


if __name__ == '__main__':
    # parameters
    # ----------------------------------------------------------------
    files_path_prefix = 'E:/Nastya/Data/OceanFull/'
    batch_size = 4
    n_days_lags = 14
    days_prediction = 7
    epochs = 10
    height = 161
    width = 181
    postfix = 'sensible'
    # ----------------------------------------------------------------
    #depth_model, depth_feedforward
    # Train part
    model = VideoPrediction(num_layers=10, d_model=10, num_heads=5, dff=2, filter_size=3, image_shape=(height, width),
                 pe_input=n_days_lags, pe_target=days_prediction, out_channel=1, loss_function='mse', optimizer='rmsprop')


    # checkpoint_path = files_path_prefix + f'Forecast/Models/Transfromer/Checkpoints'

    # model.load(checkpoint_path)
    X_train = np.load(files_path_prefix + f'Forecast/Train/X_train_{postfix}.npy')
    Y_train = np.load(files_path_prefix + f'Forecast/Train/Y_train_{postfix}.npy')
    model.train(X_train, Y_train, 100, 4, files_path_prefix + f'Forecast/Models/Transformer/Checkpoints/{postfix}')
    model.save(files_path_prefix + f'Forecast/Models/Transformer/{postfix}')
    raise ValueError

    # ----------------------------------------------------------------
    # Test part
    model = VideoPrediction(num_layers=10, d_model=2, num_heads=5, dff=2, filter_size=3, image_shape=(height, width),
                 pe_input=n_days_lags, pe_target=days_prediction, out_channel=1, loss_function='mse', optimizer='rmsprop')
    model.load(files_path_prefix + f'Forecast/Models/Transformer/{postfix}')
    X_test = np.load(files_path_prefix + f'Forecast/Test/X_test_{postfix}.npy')
    Y_test = np.load(files_path_prefix + f'Forecast/Test/Y_test_{postfix}.npy')
    model.predict(X_test, days_prediction)