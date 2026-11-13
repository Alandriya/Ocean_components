"""Causally aligned, *joint* bivariate Ito-SDE coefficients for ocean fields.

Expected channel order from Coefficients/common.py: A [T,2,H,W],
B [T,4,H,W] with components [B11,B22,B12,B21] (zero-based:
[B00,B11,B01,B10]). Also supports [T,H,W,C], [space,T,C], and
[T,space,C] with native space=29141. For genuine 2x2 matrix-axis
inputs the matrix axes, not the packed channel order, are authoritative.
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
        if channels == 4 and arr.ndim == 5 and shape[1:3] == (2, 2) and shape[-2:] in ((NATIVE_HEIGHT,NATIVE_WIDTH),self.expected_hw):
            self.layout, self.length = 'tmmhw', shape[0]
        elif channels == 4 and arr.ndim == 5 and shape[-2:] == (2, 2) and shape[1:3] in ((NATIVE_HEIGHT,NATIVE_WIDTH),self.expected_hw):
            self.layout, self.length = 'thwmm', shape[0]
        elif channels == 4 and arr.ndim == 4 and shape[-2:] == (2, 2) and shape[0] in (native_space, self.expected_hw[0] * self.expected_hw[1]):
            self.layout, self.length = 'stmm', shape[1]
        elif channels == 4 and arr.ndim == 4 and shape[-2:] == (2, 2) and shape[1] in (native_space, self.expected_hw[0] * self.expected_hw[1]):
            self.layout, self.length = 'tsmm', shape[0]
        elif arr.ndim == 3 and shape[1] == channels and shape[2] in (native_space,self.expected_hw[0]*self.expected_hw[1]):
            self.layout, self.length = 'tcs', shape[0]
        elif arr.ndim == 4 and shape[1] == channels and shape[2:] in ((NATIVE_HEIGHT, NATIVE_WIDTH), self.expected_hw):
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
        if self.layout == 'tmmhw':
            raw = np.asarray(arr[idx], dtype=np.float32)
            x = raw.reshape(len(idx),4,*raw.shape[-2:])
        elif self.layout == 'thwmm':
            raw = np.asarray(arr[idx],dtype=np.float32)
            x = np.moveaxis(raw.reshape(len(idx),*raw.shape[1:3],4),-1,1)
        elif self.layout in ('stmm','tsmm'):
            raw = np.asarray(arr[:,idx] if self.layout=='stmm' else arr[idx],dtype=np.float32)
            if self.layout=='stmm':raw=raw.transpose(1,2,3,0)
            else:raw=raw.transpose(0,2,3,1)
            nspace = raw.shape[-1]
            hh,ww = ((NATIVE_HEIGHT,NATIVE_WIDTH) if nspace==NATIVE_HEIGHT*NATIVE_WIDTH else self.expected_hw)
            x = raw.reshape(len(idx),4,hh,ww)
        elif self.layout == 'tcs':
            raw = np.asarray(arr[idx],dtype=np.float32)
            hh,ww = ((NATIVE_HEIGHT,NATIVE_WIDTH) if raw.shape[-1]==NATIVE_HEIGHT*NATIVE_WIDTH else self.expected_hw)
            x = raw.reshape(len(idx),self.channels,hh,ww)
        elif self.layout == 'tchw':
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
    def __init__(self, a_path, b_path, downsample, hw, state_std, time_offset=1, b_layout='diag_first'):
        if b_layout not in ('diag_first', 'row_major', 'column_major'):
            raise ValueError('Unknown b_layout; expected diag_first, row_major, or column_major')
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
        raw_b = self.b.get(indices)  # [T,4,H,W]
        # True matrix-axis inputs are already unambiguously [row,col].
        # In contrast, *flat* four-channel arrays from this repository use
        # [B11,B22,B12,B21], which is neither C nor F reshape order.
        matrix_axes = self.b.layout in ('tmmhw', 'thwmm', 'stmm', 'tsmm')
        if matrix_axes or self.b_layout == 'row_major':
            b = raw_b.reshape(len(indices), 2, 2, *self.a.expected_hw)
        elif self.b_layout == 'column_major':
            b = raw_b.reshape(len(indices), 2, 2, *self.a.expected_hw).swapaxes(1, 2)
        else:  # diag_first: [B11, B22, B12, B21]
            b = np.empty((len(indices), 2, 2, *self.a.expected_hw), dtype=raw_b.dtype)
            b[:, 0, 0] = raw_b[:, 0]
            b[:, 1, 1] = raw_b[:, 1]
            b[:, 0, 1] = raw_b[:, 2]
            b[:, 1, 0] = raw_b[:, 3]
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
