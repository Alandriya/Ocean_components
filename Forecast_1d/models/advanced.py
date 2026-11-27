"""N-BEATS, N-HiTS, PatchTST, DLinear and TiDE point-forecasting architectures and their configuration factory."""

# --- Nbeats ---
# Generic-basis N-BEATS, adapted to joint two-channel point forecasts.

import torch
from torch import nn


class NBEATSBlock(nn.Module):
    """Produce backcast and forecast projections from a flattened history window."""
    def __init__(self, length, horizon, channels, width, depth, dropout):
        """Initialize NBEATSBlock with ``length``, ``horizon``, ``channels``, ``width``, ``depth``, ``dropout``."""
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
        """Return backcast and forecast vectors from the flattened history."""
        z=self.body(x)
        return self.backcast(z),self.forecast(z)


class NBEATSForecaster(nn.Module):
    """Sum successive N-BEATS block forecasts while subtracting backcasts from the history residual."""
    def __init__(self,length,horizon,channels,config):
        """Initialize NBEATSForecaster with ``length``, ``horizon``, ``channels``, ``config``."""
        super().__init__()
        self.horizon=horizon
        self.blocks=nn.ModuleList([NBEATSBlock(length,horizon,channels,config['width'],config['depth'],config['dropout']) for _ in range(config['blocks'])])
    def forward(self,x):
        """Accumulate block forecasts and return a tensor of shape (batch, horizon, 2)."""
        residual=x.flatten(1)
        forecast=x.new_zeros(x.shape[0],self.horizon*2)
        for block in self.blocks:
            back,fore=block(residual)
            residual=residual-back
            forecast=forecast+fore
        return forecast.reshape(-1,self.horizon,2)


# --- Nhits ---
# N-HiTS-style multirate pooling and hierarchical linear interpolation.

import math
import torch.nn.functional as F


class NHITSBlock(nn.Module):
    """Use pooled history and interpolated forecast knots to produce a backcast and forecast."""
    def __init__(self,length,horizon,channels,width,depth,dropout,pool,downsample):
        """Initialize NHITSBlock with ``length``, ``horizon``, ``channels``, ``width``, ``depth``, ``dropout``, ``pool`` and the remaining options."""
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
        """Pool the input history, predict backcast and interpolate forecast knots to all leads."""
        pooled=F.max_pool1d(x.transpose(1,2),self.pool,stride=self.pool,ceil_mode=True)
        z=self.body(pooled.flatten(1))
        back=self.backcast(z).reshape_as(x)
        knots=self.coefficients(z).reshape(-1,2,self.knots)
        forecast=F.interpolate(knots,size=self.horizon,mode='linear',align_corners=False).transpose(1,2)
        return back,forecast


class NHITSForecaster(nn.Module):
    """Combine multiscale N-HiTS blocks for joint two-channel forecasting."""
    def __init__(self,length,horizon,channels,config):
        """Initialize NHITSForecaster with ``length``, ``horizon``, ``channels``, ``config``."""
        super().__init__()
        self.horizon=horizon
        self.blocks=nn.ModuleList([NHITSBlock(length,horizon,channels,config['width'],config['depth'],config['dropout'],min(length,p),d) for p,d in zip(config['pool_sizes'],config['downsample']) for _ in range(config['blocks_per_scale'])])
    def forward(self,x):
        """Accumulate the multiscale residual forecasts with shape (batch, horizon, 2)."""
        residual=x
        forecast=x.new_zeros(x.shape[0],self.horizon,2)
        for block in self.blocks:
            back,fore=block(residual)
            residual=residual-back
            forecast=forecast+fore
        return forecast


# --- Patchtst ---
# Channel-independent PatchTST encoder; shared parameters within each point.



class PatchTSTForecaster(nn.Module):
    """Encode normalized overlapping history patches with a channel-independent Transformer."""
    def __init__(self,length,horizon,channels,config):
        """Initialize Patch TSTForecaster with ``length``, ``horizon``, ``channels``, ``config``."""
        super().__init__()
        if channels!=2:
            raise ValueError('This channel-independent variant expects two target channels')
        self.patch=min(length,config['patch_len'])
        self.stride=min(self.patch,config['stride'])
        self.pad=self.stride
        n_patches=(length+self.pad-self.patch)//self.stride+1
        dim=config['d_model']
        self.projection=nn.Linear(self.patch,dim)
        self.position=nn.Parameter(torch.zeros(1,n_patches,dim))
        nn.init.normal_(self.position,std=0.02)
        layer=nn.TransformerEncoderLayer(dim,config['heads'],dim*2,config['dropout'],activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,config['layers'],enable_nested_tensor=False)
        self.head=nn.Sequential(nn.Flatten(1),nn.Dropout(config['dropout']),nn.Linear(n_patches*dim,horizon))
        self.horizon=horizon
    def forward(self,x):
        """Normalize history, encode per-channel patches and restore forecast values to the original scale."""
        b,t,c=x.shape
        # RevIN-style per-window statistics; no future target statistics.
        mean=x.mean(dim=1,keepdim=True).detach()
        std=(x.var(dim=1,unbiased=False,keepdim=True)+1e-5).sqrt().detach()
        z=((x-mean)/std).transpose(1,2).reshape(b*c,1,t)
        z=F.pad(z,(0,self.pad),mode='replicate').squeeze(1).unfold(-1,self.patch,self.stride)
        z=self.encoder(self.projection(z)+self.position)
        y=self.head(z).reshape(b,c,self.horizon).transpose(1,2)
        return y*std+mean


# --- Dlinear ---
# DLinear: moving-average decomposition and separate linear projections.



class DLinearForecaster(nn.Module):
    """Project moving-average trend and residual components separately into the forecast horizon."""
    def __init__(self,length,horizon,channels,config):
        """Initialize DLinear Forecaster with ``length``, ``horizon``, ``channels``, ``config``."""
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
        """Decompose history into trend and seasonal residuals and sum their lead-wise projections."""
        z=x.transpose(1,2)
        k=self.kernel//2
        trend=F.avg_pool1d(F.pad(z,(k,k),mode='replicate'),self.kernel,stride=1)
        seasonal=z-trend
        return torch.stack([self.trend[c](trend[:,c])+self.residual[c](seasonal[:,c]) for c in range(2)],dim=-1)


# --- Tide ---
# TiDE-style dense encoder/decoder with no exogenous covariates in this study.



class ResidualDense(nn.Module):
    """Apply a residual dense transformation followed by layer normalization."""
    def __init__(self,in_dim,out_dim,width,dropout):
        """Initialize Residual Dense with ``in_dim``, ``out_dim``, ``width``, ``dropout``."""
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,width),nn.ReLU(),nn.Linear(width,out_dim),nn.Dropout(dropout))
        self.skip=nn.Linear(in_dim,out_dim)
        self.norm=nn.LayerNorm(out_dim)
    def forward(self,x):
        """Combine the dense transform and skip projection, then normalize the result."""
        return self.norm(self.net(x)+self.skip(x))


class TiDEForecaster(nn.Module):
    """Encode history with dense residual blocks and decode lead-specific two-channel predictions."""
    def __init__(self,length,horizon,channels,config):
        """Initialize Ti DEForecaster with ``length``, ``horizon``, ``channels``, ``config``."""
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
        """Decode flattened history into lead-specific predictions and add a linear skip forecast."""
        z=x.flatten(1)
        decoded=self.decoder(self.encoder(z)).reshape(-1,self.horizon,self.decoder_dim)
        return self.head(self.temporal(decoded))+self.skip(z).reshape(-1,self.horizon,2)


# --- Advanced ---
# Compact level-2 configurations; all models are trained from scratch.

def advanced_config(name,hidden_size=None,dropout=0.1):
    """Return model-specific hyperparameters for a level-2 forecasting architecture."""
    width=128 if hidden_size is None else int(hidden_size)
    if name=='nbeats':
        return dict(width=width,depth=3,blocks=4,dropout=dropout)
    if name=='nhits':
        return dict(width=width,depth=2,pool_sizes=[4,2,1],downsample=[4,2,1],blocks_per_scale=1,dropout=dropout)
    if name=='patchtst':
        dim=64 if hidden_size is None else max(4,4*(width//4))
        return dict(d_model=dim,heads=4,layers=2,patch_len=8,stride=4,dropout=dropout)
    if name=='dlinear':
        return dict(kernel_size=25)
    if name=='tide':
        return dict(width=width,decoder_dim=16,dropout=dropout)
    raise ValueError(f'Unknown level-2 model: {name}')


def advanced_model(name,length,horizon,channels,config):
    """Instantiate the requested level-2 model for the specified history, horizon and channel count."""
    cls={'nbeats':NBEATSForecaster,'nhits':NHITSForecaster,'patchtst':PatchTSTForecaster,'dlinear':DLinearForecaster,'tide':TiDEForecaster}[name]
    return cls(length,horizon,channels,config)
