from torch import nn
from config import cfg
import torch
import numpy as np
import math
from collections import OrderedDict


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4], dilation=v[5])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = cfg.CONV_conv(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4], dilation=v[5])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            elif 'prelu' in layer_name:
                layers.append(('prelu_' + layer_name, nn.PReLU()))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


class Model(nn.Module):
    def __init__(self, embed, rnn, fc):
        super().__init__()
        self.embed = make_layers(embed)
        self.rnns = rnn
        self.fc = make_layers(fc)

    def forward(self, inputs, mode=''):
        x, eta, epoch = inputs  # s b c h w
        out_len = cfg.out_len

        outputs = []
        decouple_losses = []
        layer_hiddens = None
        m = None

        for t in range(out_len):
            # in b c h w -> b c in h w
            input = torch.permute(x[t:t+cfg.in_len], dims=(1, 2, 0, 3, 4))
            # b c in h w -> b c*in h w
            input = torch.reshape(input, (x.shape[1], x.shape[2]*cfg.in_len, x.shape[3], x.shape[4]))
            # print(input.shape)
            output, m, layer_hiddens, decouple_loss = self.rnns(input, m, layer_hiddens, self.embed, self.fc)
            outputs.append(output)
            decouple_losses.append(decouple_loss)
        outputs = torch.stack(outputs)  # out b c h w
        decouple_losses = torch.stack(decouple_losses)  # out l b c
        return outputs, decouple_losses
