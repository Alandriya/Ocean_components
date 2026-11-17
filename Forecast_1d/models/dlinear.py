"""DLinear: moving-average decomposition and separate linear projections."""
import torch
from torch import nn
import torch.nn.functional as F


class DLinearForecaster(nn.Module):
    def __init__(self,length,horizon,channels,config):
        super().__init__()
        if channels!=2:
            raise ValueError('This DLinear variant expects two target channels')
        self.kernel=config['kernel_size']
        if self.kernel<1 or self.kernel%2==0:
            raise ValueError('Moving-average kernel must be positive and odd')
        self.trend=nn.ModuleList([nn.Linear(length,horizon) for _ in range(2)])
        self.residual=nn.ModuleList([nn.Linear(length,horizon) for _ in range(2)])
        for layer in (*self.trend,*self.residual):
            nn.init.constant_(layer.weight,1/length)
            nn.init.zeros_(layer.bias)
    def forward(self,x):
        z=x.transpose(1,2)
        k=self.kernel//2
        trend=F.avg_pool1d(F.pad(z,(k,k),mode='replicate'),self.kernel,stride=1)
        seasonal=z-trend
        return torch.stack([self.trend[c](trend[:,c])+self.residual[c](seasonal[:,c]) for c in range(2)],dim=-1)
