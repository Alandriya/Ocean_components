import numpy as np
files_path_prefix = 'D://Data/OceanFull/'
width = 10
height = 10

if __name__ == '__main__':
    train = np.zeros((100, height, width))
    for t in range(100):
        train[t] = t * 10 + np.random.normal(0, 1, (height, width))

    np.save(files_path_prefix + 'Forecast/Train/train_simple.npy', train)

    test = np.zeros((30, 10, 10))
    for t in range(30):
        test[t] = (t + 100) * 10 + np.random.normal(0, 1, (height, width))

    np.save(files_path_prefix + 'Forecast/Test/test_simple.npy', test)
