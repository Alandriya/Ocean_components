"""Two explicit variants motivated by drift/diffusion matching.

This is NOT a reproduction of Nartallo-Kaluarachchi et al.'s low-rank RNN
construction. The variable-coefficient, gridded ocean setting requires new
architectural choices and introduces extra assumptions documented in DDM_README.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from Forecast_nn.models.simvp import SimVPForecaster
from Forecast_nn.joint_sde import covariance_from_b


def mask_mean(value, mask):
    """Mean over only ocean nodes, summing all non-spatial dimensions."""
    m = mask.to(value.device, value.dtype).reshape((1,) * (value.ndim - 2) + tuple(mask.shape))
    return (value * m).sum() / (m.sum() * (value.numel() // (value.shape[-1] * value.shape[-2]))).clamp_min(1)


class DDMSimVP(nn.Module):
    """Joint SimVP with shared causal backbone and two auxiliary moment heads.

    Heads fit *last observed* A_t and D_t=B_tB_t^T. Neither A nor B is an input
    to the forward() method, so evaluation never needs future coefficient maps.
    """
    def __init__(self, in_len, out_len, simvp_params):
        super().__init__()
        self.backbone = SimVPForecaster(in_len, out_len, in_channels=2, **simvp_params)
        width = int(simvp_params['base_channels']) * (2 ** int(simvp_params['depth']))
        hidden = max(16, width // 2)
        self.aux = nn.Sequential(nn.Conv2d(width, hidden, 3, padding=1),
                                 nn.GELU(), nn.Conv2d(hidden, 5, 1))

    def forward(self, x, mask):
        forecast, context = self.backbone(x, mask, return_bottleneck=True)
        params = F.interpolate(self.aux(context), size=x.shape[-2:], mode='bilinear', align_corners=False)
        a = params[:, 0:2].float()
        l11 = F.softplus(params[:, 2].float()) + 1e-4
        l21 = params[:, 3].float()
        l22 = F.softplus(params[:, 4].float()) + 1e-4
        d = torch.stack([l11.square(), l11*l21, l11*l21, l21.square()+l22.square()], dim=1)
        return forecast, a, d.reshape(-1, 2, 2, *x.shape[-2:])


def aux_moment_losses(a_pred, d_pred, a_ref, b_ref, mask, cov_scale=0.5):
    """Robust, scale-normalized past-time A/D auxiliary supervision."""
    if cov_scale <= 0:
        raise ValueError('cov_scale must be positive')
    d_ref = covariance_from_b(b_ref)
    drift = mask_mean(F.smooth_l1_loss(a_pred, a_ref.float(), reduction='none'), mask)
    # asinh is signed for the off-diagonal covariance and tames rare extremes.
    lhs = torch.asinh(d_pred / cov_scale)
    rhs = torch.asinh(d_ref / cov_scale)
    cov = mask_mean(F.smooth_l1_loss(lhs, rhs, reduction='none'), mask)
    return drift, cov


class EOFProjector:
    """Train-only PCA/EOF basis and frozen linear projection/reconstruction.

    Input and output fields are in the TRAIN-normalized 2-channel units.
    Retains full masked spatial mean, so rank truncation does not discard it.
    """
    def __init__(self, rank=16, snapshots=512, seed=2025):
        self.rank = int(rank)
        self.snapshots = int(snapshots)
        self.seed = int(seed)
        if self.rank < 2 or self.snapshots < self.rank + 2:
            raise ValueError('Require 2<=rank and snapshots>=rank+2')

    def fit(self, reader, mask, train_end, mean, std, chunk=16):
        mask = np.asarray(mask, bool)
        self.mask = mask
        self.shape = (2, *mask.shape)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        indices = np.unique(np.linspace(0, train_end - 1, min(train_end, self.snapshots)).astype(int))
        if len(indices) < self.rank + 2:
            raise ValueError('Insufficient training snapshots to fit EOF')
        chunks = []
        for start in range(0, len(indices), chunk):
            fields = reader.read_indices(indices[start:start+chunk])
            norm = np.nan_to_num((fields - self.mean[None,:,None,None]) / self.std[None,:,None,None],
                                 nan=0., posinf=0., neginf=0.)
            chunks.append(norm[:, :, mask].reshape(len(fields), -1))
        sample = np.concatenate(chunks, axis=0).astype(np.float32)
        self.spatial_mean = sample.mean(axis=0)
        sample -= self.spatial_mean
        torch.manual_seed(self.seed)
        with torch.no_grad():
            _, _, v = torch.pca_lowrank(torch.from_numpy(sample),
                                         q=min(len(sample)-1, sample.shape[1], self.rank+8),
                                         center=False, niter=2)
        self.components = v[:, :self.rank].numpy().T.copy()
        del sample, chunks
        # Only training observations determine the latent standardization and
        # innovation covariance. Do not inspect any val/test target here.
        count, s1, s2 = 0, np.zeros(self.rank), np.zeros(self.rank)
        prev = None
        inc1 = np.zeros(self.rank, dtype=np.float64)
        inc2 = np.zeros((self.rank, self.rank), dtype=np.float64)
        ninc = 0
        for start in range(0, train_end, chunk):
            frames = reader.read_range(start, min(start+chunk, train_end))
            norm = np.nan_to_num((frames - self.mean[None,:,None,None]) / self.std[None,:,None,None],
                                 nan=0., posinf=0., neginf=0.)
            z = (norm[:,:,mask].reshape(len(frames),-1) - self.spatial_mean) @ self.components.T
            for row in z:
                count += 1
                s1 += row
                s2 += row.astype(np.float64)**2
                if prev is not None:
                    inc = row.astype(np.float64) - prev
                    inc1 += inc
                    inc2 += np.outer(inc, inc)
                    ninc += 1
                prev = row.astype(np.float64)
        self.latent_mean = (s1/count).astype(np.float32)
        self.latent_std = np.maximum(np.sqrt(np.maximum(s2/count-(s1/count)**2, 0)), 1e-3).astype(np.float32)
        cov = inc2/max(ninc,1) - np.outer(inc1/max(ninc,1),inc1/max(ninc,1))
        self.innovation_cov = (cov / self.latent_std[:,None] / self.latent_std[None,:]
                               + 1e-3*np.eye(self.rank)).astype(np.float32)
        return self

    def state_dict(self):
        keys = ('rank','snapshots','seed','mask','shape','mean','std','spatial_mean',
                'components','latent_mean','latent_std','innovation_cov')
        return {k:getattr(self,k) for k in keys}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state['rank'],state['snapshots'],state['seed'])
        for k,v in state.items():
            setattr(obj,k,v)
        return obj

    def to_torch(self, device):
        self._device = torch.device(device)
        self._mask_t = torch.as_tensor(self.mask, dtype=torch.bool, device=device)
        self._spatial = torch.as_tensor(self.spatial_mean, dtype=torch.float32, device=device)
        self._basis = torch.as_tensor(self.components, dtype=torch.float32, device=device)
        self._center = torch.as_tensor(self.latent_mean, dtype=torch.float32, device=device)
        self._scale = torch.as_tensor(self.latent_std, dtype=torch.float32, device=device)
        self._innovation_cov = torch.as_tensor(self.innovation_cov, dtype=torch.float32, device=device)
        return self

    def encode(self, maps):
        # maps: [B,T,2,H,W] or [B,2,H,W], already normalized
        if maps.shape[-3] != 2:
            raise ValueError('EOF projector requires joint sensible+latent inputs')
        flat = maps[..., self._mask_t].reshape(*maps.shape[:-3], -1)
        z = (flat.float() - self._spatial) @ self._basis.T
        return (z - self._center) / self._scale

    def decode(self, scores):
        # z: [...,rank] -> [...,2,H,W], zero on land
        shp = tuple(scores.shape[:-1]); c,h,w = self.shape
        flat = ((scores * self._scale + self._center) @ self._basis + self._spatial)
        maps = flat.new_zeros(*shp, c, h, w)
        maps[...,self._mask_t] = flat.reshape(*shp,c,-1)
        return maps

    def project_drift(self, a):
        # Normalized A_t [B,2,H,W] -> standardized latent derivative [B,k]
        return (a.float()[...,self._mask_t].reshape(len(a),-1) @ self._basis.T) / self._scale

    def covariance_target(self, b):
        """Full latent covariance ONLY under independent site-noise assumption.

        B_t is [B,2,2,H,W] in normalized physical coordinates. This omits
        between-location covariance; do not misinterpret it as the true field
        covariance projected to EOF coordinates.
        """
        bb = covariance_from_b(b)  # B,2,2,H,W
        d = bb[...,self._mask_t]  # B,2,2,N
        components = self._basis.reshape(self.rank,2,-1) / self._scale[:,None,None]
        return torch.einsum('kis,bijs,ljs->bkl',components,d,components)


class LatentSDERNN(nn.Module):
    """Learned Gaussian stochastic transitions in the EOF latent state.

    Euler-Maruyama with dt=1 day. Cholesky factor is output by a GRU hidden
    state. Trajectories are updated with their OWN sampled next state.
    """
    def __init__(self, rank=16, hidden=128, dt=1.0):
        super().__init__()
        self.rank, self.hidden, self.dt = int(rank), int(hidden), float(dt)
        if self.rank < 2 or self.dt <= 0:
            raise ValueError('rank>=2 and dt>0 required')
        self.gru = nn.GRUCell(self.rank, self.hidden)
        self.start = nn.Linear(self.rank,self.hidden)
        self.body = nn.Sequential(nn.LayerNorm(self.hidden),nn.Linear(self.hidden,self.hidden),nn.Tanh())
        self.drift = nn.Linear(self.hidden,self.rank)
        self.cholesky = nn.Linear(self.hidden,self.rank*(self.rank+1)//2)
        # Start with moderate innovation spread; keep SPD via softplus diagonal.
        nn.init.constant_(self.cholesky.bias, 0.)
        with torch.no_grad():
            i,j=torch.tril_indices(self.rank,self.rank)
            self.cholesky.bias[i==j] = -1.0

    def encode_history(self, history):
        # [batch,time,rank]
        hidden = torch.tanh(self.start(history[:,0]))
        for t in range(history.shape[1]):
            hidden = self.gru(history[:,t],hidden)
        return hidden

    def params(self, hidden):
        z = self.body(hidden)
        drift = self.drift(z)
        tril = self.cholesky(z)
        lower = tril.new_zeros(len(tril),self.rank,self.rank)
        i,j=torch.tril_indices(self.rank,self.rank,device=tril.device)
        # Transform diagonals BEFORE the scatter. Mutating a view of lower
        # after the scatter breaks autograd's saved tensor version checks.
        values = torch.where((i==j)[None,:], F.softplus(tril)+1e-3, tril)
        lower[:,i,j] = values
        return drift,lower

    def fit_loss(self, history, future, mean_weight=1., nll_weight=0.1):
        """Teacher-forced Gaussian NLL plus autonomous mean-trajectory MSE."""
        ht = self.encode_history(history)
        ha = ht
        zteacher = history[:,-1]
        zauto = zteacher
        loss_nll = history.new_zeros(())
        loss_mse = history.new_zeros(())
        first_a, first_cov = None,None
        for step in range(future.shape[1]):
            a,l = self.params(ht)
            if step==0:
                first_a=a
                first_cov=l@l.transpose(-1,-2)
            # Linear algebra must run in float32, including under CUDA AMP.
            # solve_triangular on half precision may be unsupported/unstable.
            with torch.amp.autocast('cuda', enabled=False):
                lf=l.float()
                delta=(future[:,step].float()-zteacher.float())-a.float()*self.dt
                whitened=torch.linalg.solve_triangular(lf,delta.unsqueeze(-1),upper=False).squeeze(-1)
                nll=0.5*((whitened.square()).sum(-1)/self.dt
                          +2*torch.log(lf.diagonal(dim1=-2,dim2=-1)).sum(-1)
                          +self.rank*np.log(self.dt))
            loss_nll=loss_nll+nll.mean()/future.shape[1]/self.rank
            zteacher=future[:,step]
            ht=self.gru(zteacher,ht)
            a_auto,_=self.params(ha)
            zauto=zauto+a_auto*self.dt
            loss_mse=loss_mse+(zauto-future[:,step]).square().mean()/future.shape[1]
            ha=self.gru(zauto,ha)
        return mean_weight*loss_mse+nll_weight*loss_nll, {
            'latent_mse':loss_mse, 'latent_nll':loss_nll,
            'first_drift':first_a,'first_cov':first_cov}

    def mean_forecast(self, history, horizon):
        h=self.encode_history(history)
        z=history[:,-1]
        states=[]
        for _ in range(horizon):
            a,_=self.params(h)
            z=z+a*self.dt
            states.append(z)
            h=self.gru(z,h)
        return torch.stack(states,dim=1)

    @torch.no_grad()
    def sample(self, history, horizon=3, members=8):
        if members<1:raise ValueError('members>=1')
        batch, steps, r=history.shape
        h=self.encode_history(history).repeat_interleave(members,dim=0)
        z=history[:,-1].repeat_interleave(members,dim=0)
        states=[]
        for _ in range(horizon):
            a,l=self.params(h)
            eps=torch.randn_like(a)
            z=z+a*self.dt+(l@eps.unsqueeze(-1)).squeeze(-1)*np.sqrt(self.dt)
            states.append(z.reshape(batch,members,r))
            h=self.gru(z,h)
        return torch.stack(states,dim=2)
