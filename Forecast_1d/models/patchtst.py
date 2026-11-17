"""Channel-independent PatchTST encoder; shared parameters within each point."""
import torch
from torch import nn
import torch.nn.functional as F


class PatchTSTForecaster(nn.Module):
    def __init__(self,length,horizon,channels,config):
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
        b,t,c=x.shape
        # RevIN-style per-window statistics; no future target statistics.
        mean=x.mean(dim=1,keepdim=True).detach()
        std=(x.var(dim=1,unbiased=False,keepdim=True)+1e-5).sqrt().detach()
        z=((x-mean)/std).transpose(1,2).reshape(b*c,1,t)
        z=F.pad(z,(0,self.pad),mode='replicate').squeeze(1).unfold(-1,self.patch,self.stride)
        z=self.encoder(self.projection(z)+self.position)
        y=self.head(z).reshape(b,c,self.horizon).transpose(1,2)
        return y*std+mean
