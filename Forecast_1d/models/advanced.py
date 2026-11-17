"""Compact level-2 configurations; all models are trained from scratch."""
from .nbeats import NBEATSForecaster
from .nhits import NHITSForecaster
from .patchtst import PatchTSTForecaster
from .dlinear import DLinearForecaster
from .tide import TiDEForecaster


def advanced_config(name,hidden_size=None,dropout=0.1):
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
    cls={'nbeats':NBEATSForecaster,'nhits':NHITSForecaster,'patchtst':PatchTSTForecaster,'dlinear':DLinearForecaster,'tide':TiDEForecaster}[name]
    return cls(length,horizon,channels,config)
