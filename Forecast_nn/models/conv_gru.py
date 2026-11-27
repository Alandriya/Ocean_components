"""Convolutional gated recurrent units for spatiotemporal sequences."""


import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    """PyTorch module implementing the Conv GRUCell architecture or layer."""
    def __init__(self, channels):
        """Initialize Conv GRUCell with ``channels``."""
        super().__init__()
        self.gates = nn.Conv2d(channels*2, channels*2, 3, padding=1)
        self.candidate = nn.Conv2d(channels*2, channels, 3, padding=1)

    def forward(self, x, h):
        """Update the convolutional GRU hidden state using the input and previous hidden state."""
        if h is None:
            h = torch.zeros_like(x)
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        hc = torch.tanh(self.candidate(torch.cat([x, r*h], 1)))
        return (1-z)*h + z*hc


class ConvGRU(nn.Module):
    """PyTorch module implementing the Conv GRU architecture or layer."""
    def __init__(self, channels):
        """Initialize Conv GRU with ``channels``."""
        super().__init__()
        self.cell = ConvGRUCell(channels)

    def forward(self, x):
        """Apply the Conv GRU transformation to the input tensors."""
        h = None
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], h)
        return h
