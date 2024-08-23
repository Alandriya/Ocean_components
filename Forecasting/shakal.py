import os.path
import numpy as np
import sys
import argparse


def shakal_size(array, model_type):
    if model_type == 'Unet':
        return array[:, ::2, ::2]
    else:
        return array[:, :, ::2, ::2]


def shakal_time(array, model_type):
    if model_type == 'Unet':
        return array[:, :, :, :7]
    else:
        return array[:, :7]


if __name__ == '__main__':
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
    # ----------------------------------------------------------------------------------------------
    if start_year == 2019:
        end_year = 2025
    else:
        end_year = start_year + 10
    # ---------------------------------------------------------------------------------------
    # if model_type == 'Unet':
        # x_train = np.zeros((train_len, height, width, days_known, features_amount), dtype=float)
        # y_train = np.zeros((train_len, height, width, days_prediction, 3), dtype=float)
    # else:
    #     x_train = np.zeros((train_len, days_known, height, width, features_amount), dtype=float)
    #     y_train = np.zeros((train_len, days_prediction, height, width, 3), dtype=float)

    x_train = np.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{model_type}_{features_amount}.npy')
    x_train = shakal_size(x_train, model_type)
    x_train = shakal_time(x_train, model_type)
    np.save(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{model_type}_{features_amount}.npy', x_train)
    print(f'New x_train shape: {x_train.shape}', flush=True)
    del x_train

    y_train = np.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_{model_type}_{features_amount}.npy')
    y_train = shakal_size(y_train, model_type)
    np.save(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_{model_type}_{features_amount}.npy', y_train)
    print(f'New y_train shape: {y_train.shape}', flush=True)
    del y_train

    if os.path.exists(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{model_type}_{features_amount}.npy'):
        x_test = np.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{model_type}_{features_amount}.npy')
        x_test = shakal_size(x_test, model_type)
        x_test = shakal_time(x_test, model_type)
        np.save(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{model_type}_{features_amount}.npy', x_test)
        print(f'New x_test shape: {x_test.shape}', flush=True)
        del x_test

    if os.path.exists(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{model_type}_{features_amount}.npy'):
        y_test = np.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{model_type}_{features_amount}.npy')
        y_test = shakal_size(y_test, model_type)
        np.save(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{model_type}_{features_amount}.npy', y_test)
        print(f'New y_test shape: {y_test.shape}', flush=True)
        del y_test