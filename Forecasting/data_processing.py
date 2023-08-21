import numpy as np
from struct import unpack
import datetime

files_path_prefix = 'D://Data/OceanFull/'
width = 10
height = 10


if __name__ == '__main__':
    # train = np.zeros((100, height, width))
    # for t in range(100):
    #     train[t] = t * 10 + np.random.normal(0, 1, (height, width))
    #
    # np.save(files_path_prefix + 'Forecast/Train/train_simple.npy', train)
    #
    # test = np.zeros((30, 10, 10))
    # for t in range(30):
    #     test[t] = (t + 100) * 10 + np.random.normal(0, 1, (height, width))
    #
    # np.save(files_path_prefix + 'Forecast/Test/test_simple.npy', test)

    # get mask
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    # ---------------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2022, 4, 2, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2022, 9, 30, 0, 0) - datetime.datetime(2022, 4, 2, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    sensible_array = np.load(files_path_prefix + 'sensible_grouped_2019-2022.npy')
    sensible_array = sensible_array.transpose()
    sensible_array = sensible_array.reshape((sensible_array.shape[0], 161, 181))

    train = sensible_array[:1000, 100:120, 100:120]
    np.save(files_path_prefix + 'Forecast/Train/train_sensible_cut.npy', train)

    test = sensible_array[1000:1000 + 30, 100:120, 100:120]
    np.save(files_path_prefix + 'Forecast/Test/test_sensible_cut.npy', test)
