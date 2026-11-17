"""Generic-basis N-BEATS, adapted to joint two-channel point forecasts."""
import torch
from torch import nn


class NBEATSBlock(nn.Module):
    def __init__(self, length, horizon, channels, width, depth, dropout):
        super().__init__()
        layers=[]
        size=length*channels
        for _ in range(depth):
            layers += [nn.Linear(size,width),nn.ReLU(),nn.Dropout(dropout)]
            size=width
        self.body=nn.Sequential(*layers)
        self.backcast=nn.Linear(width,length*channels)
        self.forecast=nn.Linear(width,horizon*2)
        self.horizon=horizon
    def forward(self,x):
        z=self.body(x)
        return self.backcast(z),self.forecast(z)


class NBEATSForecaster(nn.Module):
    def __init__(self,length,horizon,channels,config):
        super().__init__()
        self.horizon=horizon
        self.blocks=nn.ModuleList([NBEATSBlock(length,horizon,channels,config['width'],config['depth'],config['dropout']) for _ in range(config['blocks'])])
    def forward(self,x):
        residual=x.flatten(1)
        forecast=x.new_zeros(x.shape[0],self.horizon*2)
        for block in self.blocks:
            back,fore=block(residual)
            residual=residual-back
            forecast=forecast+fore
        return forecast.reshape(-1,self.horizon,2)
