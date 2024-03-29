import os.path

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import numpy as np
#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR
from plotter import *
from struct import unpack
import time
import pandas as pd

# Parameters
files_path_prefix = 'D://Data/OceanFull/'

days_prediction = 7
# width = 181
# height = 161
# width = 20
# height = 20

if __name__ == '__main__':
    # load or create stats DataFrame
    if not os.path.exists(files_path_prefix + f'Forecast/stat_df.xlsx'):
        stat_df = pd.DataFrame(columns=['days prediction', 'width', 'height', 'model name', 'training time', 'RMSE'])
    else:
        stat_df = pd.read_excel(files_path_prefix + f'Forecast/stat_df.xlsx')

    model_name = 'RandomForest'
    # load data
    # training_data = np.load(files_path_prefix + 'Forecast/Train/train_simple.npy')
    # test_data = np.load(files_path_prefix + 'Forecast/Test/test_simple.npy')
    # maskfile = open(files_path_prefix + "mask", "rb")
    # binary_values = maskfile.read(29141)
    # maskfile.close()
    # mask = unpack('?' * 29141, binary_values)
    # mask = np.array(mask, dtype=int)

    training_data = np.load(files_path_prefix + 'Forecast/Train/train_sensible_cut.npy')
    test_data = np.load(files_path_prefix + 'Forecast/Test/test_sensible_cut.npy')

    width = training_data.shape[2]
    height = training_data.shape[1]
    mask = np.ones(height * width)

    X_train = training_data[:len(training_data) - days_prediction]
    Y_train = training_data[days_prediction:]
    X_test = test_data[:len(test_data) - days_prediction]
    Y_test = test_data[days_prediction:]

    X_train = X_train[np.where(np.logical_not(np.isnan(X_train)))]
    Y_train = Y_train[np.where(np.logical_not(np.isnan(Y_train)))]
    X_test = X_test[np.where(np.logical_not(np.isnan(X_test)))]
    Y_test = Y_test[np.where(np.logical_not(np.isnan(Y_test)))]

    # formatting pictures into 1d array
    X_train = np.reshape(X_train, (len(training_data) - days_prediction, -1))
    Y_train = np.reshape(Y_train, (len(training_data) - days_prediction, -1))
    X_test = np.reshape(X_test, (len(test_data) - days_prediction, -1))
    Y_test = np.reshape(Y_test, (len(test_data) - days_prediction, -1))

    # ---------------------------------------------------------------------------------------------
    # set up and fit model
    time_start = time.time()
    model = RandomForestRegressor()
    model_fit = model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    time_fitting = time.time() - time_start
    print(f'Model {model_name} was trained in {time_fitting: .1f} s')
    rmse = np.sqrt(np.sum(np.square(Y_test - Y_predict)))

    # save predictions
    np.save(files_path_prefix + f'Forecast/Predictions/{model_name}.npy', Y_predict)

    # save training info
    stat_df.loc[len(stat_df)] = [days_prediction, width, height, model_name, time_fitting, rmse]
    stat_df.to_excel(files_path_prefix + f'Forecast/stat_df.xlsx', index=False)
    # ---------------------------------------------------------------------------------------------

    # load predictions
    Y_predict = np.load(files_path_prefix + f'Forecast/Predictions/{model_name}.npy')

    rmse = np.sqrt(np.sum(np.square(Y_test - Y_predict)))
    print(f'Common RMSE {model_name}: {rmse}')

    # returning back to pictures format
    Y_test_pictures = np.zeros((Y_test.shape[0], height, width))
    Y_predict_pictures = np.zeros((Y_test.shape[0], height, width))

    mask_picture = np.reshape(mask, (height, width))
    Y_test_pictures[:, np.logical_not(mask_picture)] = np.nan
    Y_predict_pictures[:, np.logical_not(mask_picture)] = np.nan

    for t in range(days_prediction):
        Y_test_pictures[t][np.where(mask_picture)] = Y_test[t]
        Y_predict_pictures[t][np.where(mask_picture)] = Y_predict[t]

    plot_predictions(files_path_prefix, Y_test_pictures, Y_predict_pictures, model_name)



