from nn_model import Transformer
import numpy as np



if __name__ == '__main__':
    # parameters
    # ----------------------------------------------------------------
    files_path_prefix = 'E:/Nastya/Data/OceanFull/'
    batch_size = 4
    n_days_lags = 7
    days_prediction = 3
    epochs = 10
    height = 161
    width = 181
    # ----------------------------------------------------------------
    model = Transformer(batch_size=batch_size, n_days_lags=n_days_lags, height=height, width=width)
    checkpoint_path = files_path_prefix + f'Forecast/Models/Transfromer/Checkpoints'

    # model.load(checkpoint_path)
    X_train = np.load(files_path_prefix + 'Forecast/Train/X_train.npy')
    Y_train = np.load(files_path_prefix + 'Forecast/Train/Y_train.npy')
    X_test = np.load(files_path_prefix + 'Forecast/Test/X_test.npy')
    Y_test = np.load(files_path_prefix + 'Forecast/Test/Y_test.npy')

    # model.train(checkpoint_path, X_train, Y_train, batch_size, epochs, 'first')
