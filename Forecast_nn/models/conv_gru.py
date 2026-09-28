
import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(channels*2, channels*2, 3, padding=1)
        self.candidate = nn.Conv2d(channels*2, channels, 3, padding=1)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros_like(x)
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        hc = torch.tanh(self.candidate(torch.cat([x, r*h], 1)))
        return (1-z)*h + z*hc


class ConvGRU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cell = ConvGRUCell(channels)

    def forward(self, x):
        h = None
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], h)
        return h
