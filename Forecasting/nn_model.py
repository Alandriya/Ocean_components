import torch
from torch import nn
import time
import os


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # # Defining the layers
        # # RNN Layer
        # self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # # Fully connected layer
        # self.fc = nn.Linear(hidden_dim, output_size)

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


    # def init_hidden(self, batch_size):
    #     # This method generates the first hidden state of zeros which we'll use in the forward pass
    #     # We'll send the tensor holding the hidden state to the device we specified earlier as well
    #     hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
    #     return hidden