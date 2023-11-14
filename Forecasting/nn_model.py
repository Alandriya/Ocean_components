import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset
import time
import os


class SimpleDataset(Dataset):
    # defining values in the constructor
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        self.len = x.shape[0]

    # Getting the data samples
    def __getitem__(self, idx):
        sample = self.x[idx], self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Getting data size/length
    def __len__(self):
        return self.len


class Model(nn.Module):
    def __init__(self, days_known, days_prediction, channels):
        super(Model, self).__init__()
        self.days_known = days_known
        self.days_prediction = days_prediction
        self.channels = channels
        # channels = 1 for only lags or 3 for lags, A and B lags


    def forward(self, x):
        conv = nn.Conv3d(in_channels=self.channels,
                         out_channels=1,
                         kernel_size=(3, 3),

                         )
        return logits


    # def init_hidden(self, batch_size):
    #     # This method generates the first hidden state of zeros which we'll use in the forward pass
    #     # We'll send the tensor holding the hidden state to the device we specified earlier as well
    #     hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
    #     return hidden