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
from nn_model import Model
from plotter import *
from nn_runner import *
import os
from tempfile import TemporaryDirectory

# Parameters
files_path_prefix = 'D://Data/OceanFull/'
# width = 181
# height = 161
width = 10
height = 10


if __name__ == '__main__':
    # parameters
    batch_size = 64

    # # load data
    training_data = torch.from_numpy(np.load(files_path_prefix + 'Forecast/Train/train_simple.npy'))
    test_data = torch.from_numpy(np.load(files_path_prefix + 'Forecast/Train/test_simple.npy'))

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

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

    # create model
    model = Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # train
    # trained_model = train(train_dataloader, model, loss_fn, optimizer, device)
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    # save model
    torch.save(model, files_path_prefix + 'Forecast/Models/model.pth')
    # torch.save(model.state_dict(), files_path_prefix + 'Forecast/Models/model_weights.pth')


    # load model
    # model = Model().to(device)  # creating untrained model
    # model.load_state_dict(torch.load(files_path_prefix + 'Forecast/Models/nn_model_weights.pth'))
    model = torch.load(files_path_prefix + 'Forecast/Models/model.pth')
    model.eval()

    # predict
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')