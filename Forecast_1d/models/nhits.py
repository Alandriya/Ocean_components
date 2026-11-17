"""N-HiTS-style multirate pooling and hierarchical linear interpolation."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class NHITSBlock(nn.Module):
    def __init__(self,length,horizon,channels,width,depth,dropout,pool,downsample):
        super().__init__()
        self.pool=pool
        self.horizon=horizon
        self.knots=max(1,math.ceil(horizon/downsample))
        layers=[]
        size=math.ceil(length/pool)*channels
        for _ in range(depth):
            layers += [nn.Linear(size,width),nn.ReLU(),nn.Dropout(dropout)]
            size=width
        self.body=nn.Sequential(*layers)
        self.backcast=nn.Linear(width,length*channels)
        self.coefficients=nn.Linear(width,self.knots*2)
    def forward(self,x):
        pooled=F.max_pool1d(x.transpose(1,2),self.pool,stride=self.pool,ceil_mode=True)
        z=self.body(pooled.flatten(1))
        back=self.backcast(z).reshape_as(x)
        knots=self.coefficients(z).reshape(-1,2,self.knots)
        forecast=F.interpolate(knots,size=self.horizon,mode='linear',align_corners=False).transpose(1,2)
        return back,forecast


class NHITSForecaster(nn.Module):
    def __init__(self,length,horizon,channels,config):
        super().__init__()
        self.horizon=horizon
        self.blocks=nn.ModuleList([NHITSBlock(length,horizon,channels,config['width'],config['depth'],config['dropout'],min(length,p),d) for p,d in zip(config['pool_sizes'],config['downsample']) for _ in range(config['blocks_per_scale'])])
    def forward(self,x):
        residual=x
        forecast=x.new_zeros(x.shape[0],self.horizon,2)
        for block in self.blocks:
            back,fore=block(residual)
            residual=residual-back
            forecast=forecast+fore
        return forecast
