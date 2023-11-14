import numpy as np
from struct import unpack
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
from torchvision import datasets, models, transforms
import torch.backends.cudnn as cudnn
from nn_model import *
from plotter import *
from nn_runner import *
import os
from tempfile import TemporaryDirectory
from ConvLSTM import *
import torchsde

files_path_prefix = 'D://Data/OceanFull/'

if __name__ == '__main__':
    # parameters
    batch_size = 32
    days_known = 10
    days_prediction = 3
    # --------------------------------------------------------------------------------
    # load data
    train_part = np.load(files_path_prefix + 'Forecast/Train/train_sensible.npy')
    test_part = np.load(files_path_prefix + 'Forecast/Test/test_sensible.npy')
    height, width = train_part.shape[1], train_part.shape[2]
    # channels = train_part.shape[3] TODO
    channels = 1
    mask = np.ones(height * width) # TODO

    # create train and test X and Y
    train_amount = train_part.shape[0] - days_known - days_prediction
    X_train = np.zeros((train_amount, days_known, channels, height, width), dtype=float)
    Y_train = np.zeros((train_amount, days_prediction, channels, height, width), dtype=float)
    for day in range(train_part.shape[0] - days_known - days_prediction):
        X_train[day] = train_part[day:day+days_known].reshape((-1, days_known, channels, height, width))
        Y_train[day] = train_part[day+days_known:day+days_known+days_prediction].reshape((-1, days_prediction, channels, height, width))

    X_train = np.float32(X_train)
    Y_train = np.float32(Y_train)
    np.save(files_path_prefix + 'Forecast/Train/X_train.npy', X_train)
    np.save(files_path_prefix + 'Forecast/Train/Y_train.npy', Y_train)

    test_amount = test_part.shape[0] - days_known - days_prediction
    X_test = np.zeros((test_amount, days_known, channels, height, width))
    Y_test = np.zeros((test_amount, days_prediction, channels, height, width))
    for day in range(test_part.shape[0] - days_known - days_prediction):
        X_test[day] = test_part[day:day+days_known].reshape((-1, days_known, channels, height, width))
        Y_test[day] = test_part[day+days_known:day+days_known+days_prediction].reshape((-1, days_prediction, channels, height, width))

    X_test = np.float32(X_test)
    Y_test = np.float32(Y_test)
    np.save(files_path_prefix + 'Forecast/Test/X_test.npy', X_test)
    np.save(files_path_prefix + 'Forecast/Test/Y_test.npy', Y_test)
    # --------------------------------------------------------------------------------

    X_train = np.load(files_path_prefix + 'Forecast/Train/X_train.npy')
    Y_train = np.load(files_path_prefix + 'Forecast/Train/Y_train.npy')
    X_test = np.load(files_path_prefix + 'Forecast/Test/X_test.npy')
    Y_test = np.load(files_path_prefix + 'Forecast/Test/Y_test.npy')

    # create datasets
    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train)
    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test)

    training_data = SimpleDataset(X_train, Y_train)
    test_data = SimpleDataset(X_test, Y_test)

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # detecting device - GPU if cuda is availiable or CPU
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)

    model = ConvLSTM(input_dim=1,
                     hidden_dim=days_prediction,
                     kernel_size=(3, 3),
                     num_layers=5,
                     batch_first=True,
                     bias=False,
                     return_all_layers=False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # training part
    # --------------------------------------------------------------------------------
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        size = len(train_dataloader)
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            # Compute prediction and loss
            layer_output, last_state_list = model(X)
            # pred = last_state_list[0][1].reshape((-1, days_prediction, 1, height, width))
            pred = layer_output[0][0].reshape((-1, days_prediction, 1, height, width))
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print("Done!")

    # save model
    torch.save(model, files_path_prefix + 'Forecast/Models/model.pth')
    # --------------------------------------------------------------------------------

    # test part
    # --------------------------------------------------------------------------------
    # load model
    model = torch.load(files_path_prefix + 'Forecast/Models/model.pth')

    # set model to evaluation mode
    model.eval()
    num_batches = len(test_dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in test_dataloader:
            layer_output, last_state_list = model(X)
            # pred = last_state_list[0][1].reshape((-1, days_prediction, 1, height, width))
            pred = layer_output[0][0].reshape((-1, days_prediction, 1, height, width))
            test_loss += loss_fn(pred, y).item()

            # plot errors
            model_name = 'simple_nn'
            test_day = y[0][:][0]
            pred_day = pred[0][:][0]
            plot_predictions(files_path_prefix, test_day, pred_day, model_name)


    test_loss /= num_batches
    print(f"Test Error: {test_loss:>8f} \n")
    # --------------------------------------------------------------------------------

    # plot errors
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------