"""Causally aligned, *joint* bivariate Ito-SDE coefficients for ocean fields.

Required layouts: A [T,2,H,W], B [T,4,H,W] in row-major B11,B12,B21,B22;
also supports [T,H,W,C], [space,T,C] and [T,space,C] with native space=29141.
File element 0 corresponds to observed flux t=1; no target-time maps are read.
The historical A/B maps are lagged estimates, not true future coefficients.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from Forecast_nn.data import NATIVE_HEIGHT, NATIVE_WIDTH


class _ComponentMap:
    def __init__(self, path, channels, downsample, expected_hw):
        self.path = str(Path(path))
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self.channels = channels
        self.step = downsample
        self.expected_hw = tuple(expected_hw)
        self.arr = None
        arr = np.load(self.path, mmap_mode='r')
        shape = arr.shape
        native_space = NATIVE_HEIGHT * NATIVE_WIDTH
        if arr.ndim == 4 and shape[1] == channels and shape[2:] in ((NATIVE_HEIGHT, NATIVE_WIDTH), self.expected_hw):
            self.layout, self.length = 'tchw', shape[0]
        elif arr.ndim == 4 and shape[-1] == channels and shape[1:3] in ((NATIVE_HEIGHT, NATIVE_WIDTH), self.expected_hw):
            self.layout, self.length = 'thwc', shape[0]
        elif arr.ndim == 3 and shape[0] == native_space and shape[2] == channels:
            self.layout, self.length = 'stc', shape[1]
        elif arr.ndim == 3 and shape[1] == native_space and shape[2] == channels:
            self.layout, self.length = 'tsc', shape[0]
        elif arr.ndim == 3 and shape[0] == self.expected_hw[0] * self.expected_hw[1] and shape[2] == channels:
            self.layout, self.length = 'stc_small', shape[1]
        elif arr.ndim == 3 and shape[1] == self.expected_hw[0] * self.expected_hw[1] and shape[2] == channels:
            self.layout, self.length = 'tsc_small', shape[0]
        else:
            raise ValueError(f'Unsupported A/B layout {shape} in {path}; expected time and {channels} components.')
        del arr

    def __getstate__(self):
        result = vars(self).copy()
        result['arr'] = None
        return result

    def get(self, indices):
        if self.arr is None:
            self.arr = np.load(self.path, mmap_mode='r')
        idx = np.asarray(indices, dtype=np.int64)
        if idx.min() < 0 or idx.max() >= self.length:
            raise IndexError('Coefficient time index outside available range')
        arr = self.arr
        if self.layout == 'tchw':
            x = np.asarray(arr[idx], dtype=np.float32)
        elif self.layout == 'thwc':
            x = np.moveaxis(np.asarray(arr[idx], dtype=np.float32), -1, 1)
        else:
            raw = np.asarray(arr[:, idx] if self.layout.startswith('stc') else arr[idx], dtype=np.float32)
            if self.layout.startswith('stc'):
                raw = raw.transpose(1, 2, 0)
            else:
                raw = raw.transpose(0, 2, 1)
            native = not self.layout.endswith('_small')
            h, w = (NATIVE_HEIGHT, NATIVE_WIDTH) if native else self.expected_hw
            x = raw.reshape(len(idx), self.channels, h, w)
        if x.shape[-2:] == (NATIVE_HEIGHT, NATIVE_WIDTH):
            x = x[..., ::self.step, ::self.step]
        if x.shape[-2:] != self.expected_hw:
            raise ValueError(f'SDE grid {x.shape[-2:]} differs from forecast grid {self.expected_hw}')
        return x


class JointSDEReader:
    def __init__(self, a_path, b_path, downsample, hw, state_std, time_offset=1, b_layout='row_major'):
        if b_layout not in ('row_major', 'column_major'):
            raise ValueError('Unknown b_layout; specify row_major or column_major explicitly')
        self.a = _ComponentMap(a_path, 2, downsample, hw)
        self.b = _ComponentMap(b_path, 4, downsample, hw)
        if self.a.length != self.b.length:
            raise ValueError('A and B have different time lengths')
        self.time_offset = int(time_offset)
        self.std = np.asarray(state_std, dtype=np.float32).reshape(2)
        self.b_layout = b_layout

    def get_normalized(self, observed_flux_indices):
        indices = np.atleast_1d(observed_flux_indices).astype(np.int64) - self.time_offset
        a = self.a.get(indices) / self.std[None, :, None, None]
        b = self.b.get(indices).reshape(len(indices), 2, 2, *self.a.expected_hw)
        if self.b_layout == 'column_major':
            b = np.swapaxes(b, 1, 2)
        b = b / self.std[None, :, None, None, None]
        return (torch.from_numpy(np.nan_to_num(a, copy=True)),
                torch.from_numpy(np.nan_to_num(b, copy=True)))


class JointSDEDataset(Dataset):
    def __init__(self, base_dataset, joint_reader):
        self.base = base_dataset
        self.reader = joint_reader
        self.channels = base_dataset.channels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        x, y = self.base[item]
        last_t = self.base.start_idx + int(item) + (self.base.in_len - 1) * self.base.stride
        a, b = self.reader.get_normalized([last_t])
        return x, y, a[0], b[0]


def covariance_from_b(b):
    """B [batch,2,2,H,W] => BB^T [batch,2,2,H,W]."""
    return torch.einsum('bikhw,bjkhw->bijhw', b.float(), b.float())


def joint_drift_loss(pred, last, a, b, mask, covariance_floor=0.1, dt=1.0):
    """Weak, causally lagged drift penalty for a *deterministic* forecast.

    Does NOT attempt to force an ensemble covariance onto one prediction.
    Uses the local diffusion covariance as a positive-definite weighting.
    """
    if pred.shape[2] != 2:
        raise ValueError('Joint drift loss requires two target channels.')
    if covariance_floor <= 0:
        raise ValueError('covariance_floor must be >0')
    delta = (pred[:, 0].float() - last.float() - a.float() * dt)
    d = covariance_from_b(b) * dt
    d00 = d[:, 0, 0] + covariance_floor
    d11 = d[:, 1, 1] + covariance_floor
    d01 = d[:, 0, 1]
    det = (d00*d11 - d01.square()).clamp_min(covariance_floor**2)
    mahal = (d11*delta[:, 0].square() + d00*delta[:, 1].square()
             - 2*d01*delta[:, 0]*delta[:, 1]) / det
    m = mask.to(device=pred.device, dtype=pred.dtype).reshape(1, *pred.shape[-2:])
    return ((mahal * m).sum() / (m.sum() * pred.shape[0]).clamp_min(1)) / 2


def joint_ensemble_moment_loss(ensemble, last, a, b, mask, dt=1.0, covariance_floor=0.01):
    """Ensemble first-increment mean and covariance vs lagged SDE moments.

    ensemble: [batch,members,horizon,2,H,W]; members >= 2.
    Use ONLY when trained ensemble has independent stochastic members.
    This comparison assumes locally frozen A_t/B_t over the first step.
    """
    if ensemble.shape[1] < 2:
        raise ValueError('At least two ensemble members required')
    inc = ensemble[:, :, 0].float() - last.float()[:, None]
    empirical_mean = inc.mean(1)
    drift_err = (empirical_mean - a.float()*dt).square().sum(1)
    centered = inc - empirical_mean[:, None]
    empirical_cov = torch.einsum('bmihw,bmjhw->bijhw', centered, centered) / (inc.shape[1]-1)
    target_cov = covariance_from_b(b) * dt
    cov_err = (empirical_cov - target_cov).square().sum((1, 2))
    m = mask.to(ensemble.device, ensemble.dtype).reshape(1, *ensemble.shape[-2:])
    den = (m.sum() * ensemble.shape[0]).clamp_min(1)
    return ((drift_err*m).sum()/den, (cov_err*m).sum()/den)
