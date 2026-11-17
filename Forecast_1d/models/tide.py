"""TiDE-style dense encoder/decoder with no exogenous covariates in this study."""
import torch
from torch import nn


class ResidualDense(nn.Module):
    def __init__(self,in_dim,out_dim,width,dropout):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,width),nn.ReLU(),nn.Linear(width,out_dim),nn.Dropout(dropout))
        self.skip=nn.Linear(in_dim,out_dim)
        self.norm=nn.LayerNorm(out_dim)
    def forward(self,x):
        return self.norm(self.net(x)+self.skip(x))


class TiDEForecaster(nn.Module):
    def __init__(self,length,horizon,channels,config):
        super().__init__()
        width=config['width']; decoder_dim=config['decoder_dim']; dropout=config['dropout']
        self.horizon=horizon; self.decoder_dim=decoder_dim
        self.encoder=nn.Sequential(ResidualDense(length*channels,width,width,dropout),ResidualDense(width,width,width,dropout))
        self.decoder=nn.Sequential(ResidualDense(width,width,width,dropout),ResidualDense(width,horizon*decoder_dim,width,dropout))
        # Lead-specific decoded features go through a shared temporal decoder.
        self.temporal=ResidualDense(decoder_dim,decoder_dim,width,dropout)
        self.head=nn.Linear(decoder_dim,2)
        self.skip=nn.Linear(length*channels,horizon*2)
    def forward(self,x):
        z=x.flatten(1)
        decoded=self.decoder(self.encoder(z)).reshape(-1,self.horizon,self.decoder_dim)
        return self.head(self.temporal(decoded))+self.skip(z).reshape(-1,self.horizon,2)
