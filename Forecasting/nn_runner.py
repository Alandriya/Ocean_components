from struct import unpack
import numpy as np
import datetime
import pandas as pd
import os
from Forecasting.clusterization import clusterize
import warnings
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import torch
from torch import nn
from gan import Generator, Discriminator

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from darts.models import TransformerModel
import torchvision
import torchvision.transforms as transforms

def rmse(y_predict, y_true):
    return np.sqrt(mean_squared_error(y_predict, y_true))

def scale(array):
    arr_min = np.nanmin(array)
    arr_max = np.nanmax(array)
    return arr_min, arr_max, (array - arr_min) / (arr_max - arr_min)


def get_eigen_arrays(files_path_prefix: str, names:tuple, t_start: int, t_end: int):
    eigenvectors = np.zeros((t_end - t_start, 161*181))
    eigenvalues = np.zeros(t_end - t_start)
    for t in range(t_end - t_start):
        eigenvectors[t] = np.real(np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigen0_{t_start + t}.npy'))
        eigenvalues[t] = np.real(np.load(files_path_prefix + f'Eigenvalues/{names[0]}-{names[1]}/eigenvalues_{t_start + t}.npy')[0])
    return eigenvectors, eigenvalues

def get_coefficients(files_path_prefix: str, names:tuple, t_start: int, t_end: int):
    a = np.zeros((2, t_end - t_start, 161, 181))
    b = np.zeros((4, t_end - t_start, 4, 161, 181))
    for t in range(t_end - t_start):
        try:
            a[0, t] = np.load(files_path_prefix + f'Coeff_data_3d/{names[0]}-{names[1]}/{t_start + t}_A_sens.npy').reshape((161, 181))
            a[1, t] = np.load(files_path_prefix + f'Coeff_data_3d/{names[0]}-{names[1]}/{t_start + t}_A_lat.npy').reshape((161, 181))
            b[:, t] = np.load(files_path_prefix + f'Coeff_data_3d/{names[0]}-{names[1]}/{t_start + t}_B.npy').reshape((4, 161, 181))
        except FileNotFoundError:
            pass
    return a, b


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
    # ---------------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------

    # load data
    flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_2019-2025_grouped.npy')
    flux_array = np.diff(flux_array, axis=1)

    SST_array = np.load(files_path_prefix + f'SST/SST_2019-2025_grouped.npy')
    SST_array = np.diff(SST_array, axis=1)

    press_array = np.load(files_path_prefix + f'Pressure/PRESS_2019-2025_grouped.npy')
    press_array = np.diff(press_array, axis=1)

    t_start = days_delta1 + days_delta2 + days_delta3 + days_delta4
    t_end = t_start + days_delta6 - 1
    # ---------------------------------------------------------------------------------------
    # transform pictures to features
    # Load pre-trained VGG16 model
    model_image3d = VGG16(weights='imagenet', include_top=False, input_shape=(161, 181, 3))
    model_image1d = VGG16(weights='imagenet', include_top=False, input_shape=(161, 181, 3))
    # flux_array = flux_array.transpose().reshape((-1, 161, 181))
    # SST_array = SST_array.transpose().reshape((-1, 161, 181))
    # press_array = press_array.transpose().reshape((-1, 161, 181))
    #
    # input_array = np.zeros((t_end-t_start, 161, 181, 3))
    # input_array[:, :, :, 0] = flux_array
    # input_array[:, :, :, 1] = SST_array
    # input_array[:, :, :, 2] = press_array

    # base_vector = model_image3d.predict(input_array)
    # print(base_vector.shape)
    # np.save(files_path_prefix + f'Forecast/2019-2025_base_vector.npy', base_vector)

    # A and B coefficients
    for names in [('flux', 'sst'), ('flux', 'press'), ('sst', 'press')]:
        a, b = get_coefficients(files_path_prefix, names, t_start, t_start+10)
        if not os.path.exists(files_path_prefix + f'Forecast/{names[0]}-{names[1]}'):
            os.mkdir(files_path_prefix + f'Forecast/{names[0]}-{names[1]}')
        print(np.)
        a0 = model_image1d.predict(np.concatenate(a[:, 0], a[:, 0], a[:, 0]))
        print(a0.shape)
        np.save(files_path_prefix + f'Forecast/{names[0]}-{names[1]}/2019-2025_a0.npy', a0)
        a1 = model_image1d.predict(a[:, 1])
        np.save(files_path_prefix + f'Forecast/{names[0]}-{names[1]}/2019-2025_a1.npy', a1)

        b0 = model_image1d.predict(b[:, 0])
        np.save(files_path_prefix + f'Forecast/{names[0]}-{names[1]}/2019-2025_b0.npy', b0)
        b1 = model_image1d.predict(b[:, 1])
        np.save(files_path_prefix + f'Forecast/{names[0]}-{names[1]}/2019-2025_b1.npy', b1)
        b2 = model_image1d.predict(b[:, 2])
        np.save(files_path_prefix + f'Forecast/{names[0]}-{names[1]}/2019-2025_b2.npy', b2)
        b3 = model_image1d.predict(b[:, 3])
        np.save(files_path_prefix + f'Forecast/{names[0]}-{names[1]}/2019-2025_b3.npy', b3)


    # # adding eigenvectors and saving eigenvalues
    # i = 3
    # for names in [('Flux', 'Flux'), ('SST', 'SST'), ('Pressure', 'Pressure'), ('Flux', 'SST'), ('Flux', 'Pressure')]:
    #     eigenvectors, eigenvalues = get_eigen_arrays(files_path_prefix, names, t_start, t_end)
    #     input_array[:, :, :, i] = eigenvectors.reshape((-1, 161, 181))
    #     eigenvalues_all[:, i] = eigenvalues
    #     i += 1
    # np.save(files_path_prefix + 'Forecast/eigenvalues.npy', eigenvalues_all)
    # ---------------------------------------------------------------------------------------
    # configs
    raise ValueError
    base_vector = np.load(files_path_prefix + f'Forecast/2019-2025_base_vector.npy')
    print(base_vector.shape)
    # eigenvalues = np.load(files_path_prefix + 'Forecast/eigenvalues.npy')
    train_len = int(base_vector.shape[0] * 2 / 3 + 100)
    train_days = [datetime.datetime(2019, 1, 1) + datetime.timedelta(days=t) for t in range(train_len)]
    test_days = [datetime.datetime(2019, 1, 1) + datetime.timedelta(days=t) for t in range(train_len, base_vector.shape[0])]

    batch_size = 32
    days_known = 14
    days_prediction = 10

    lr = 0.0001
    num_epochs = 50
    loss_function = nn.BCELoss()

    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # ---------------------------------------------------------------------------------------
    # construct model
    discriminator = Discriminator().to(device=device)
    generator = Generator().to(device=device)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    # ---------------------------------------------------------------------------------------
    # train
    train_set = features
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    for epoch in range(num_epochs):
        for n, (real_samples, y) in enumerate(train_loader):
            # Data for training the discriminator
            real_samples = real_samples.to(device=device)
            real_samples_labels = torch.ones((batch_size, 1)).to(
                device=device
            )
            latent_space_samples = torch.randn((batch_size, 100)).to(
                device=device
            )
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1)).to(
                device=device
            )
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels)
            )

            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(
                output_discriminator, all_samples_labels
            )
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, 100)).to(
                device=device
            )

            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()

            # Show loss
            if n == batch_size - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")
