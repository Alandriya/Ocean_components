"""Train-only EOF/PCA + ridge VAR(p) over joint ocean-map coefficients.

Low-rank EOF space avoids fitting a separate VAR per grid cell. A stochastic
ensemble uses innovations estimated ONLY from training residuals. This is a
linear statistical baseline, NOT the image-generation model Visual-AR (VAR).
"""
import numpy as np
import torch


class EOFVAR:
    def __init__(self, rank=24, lags=5, ridge=1.0, snapshot_stride=7, seed=2025):
        self.rank = int(rank)
        self.lags = int(lags)
        self.ridge = float(ridge)
        self.snapshot_stride = int(snapshot_stride)
        self.seed = int(seed)
        if min(self.rank, self.lags, self.snapshot_stride) < 1 or self.ridge < 0:
            raise ValueError('rank, lags, snapshot_stride must be positive, ridge non-negative')

    def fit(self, reader, mask, train_end, mean, std, chunk=32):
        rng = np.random.default_rng(self.seed)
        self.mask = mask.astype(bool)
        self.shape = (int(reader.channels), *mask.shape)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        frames = np.arange(0, train_end, self.snapshot_stride)
        if frames.size < self.rank + 2 or train_end < self.lags+3:
            raise ValueError('Not enough train-only snapshots for requested EOF rank/lags')
        print(f'[EOF-VAR] PCA snapshots={len(frames)}; rank={self.rank}; train_end={train_end}', flush=True)
        chunks = []
        for i in range(0, len(frames), chunk):
            raw = reader.read_indices(frames[i:i+chunk])
            chunks.append(self._flatten(raw))
        X = np.concatenate(chunks, axis=0).astype(np.float32)
        self.spatial_mean = X.mean(0)
        X -= self.spatial_mean
        if X.shape[1] <= self.rank:
            raise ValueError('Rank exceeds number of ocean degrees of freedom')
        torch.manual_seed(self.seed)
        with torch.no_grad():
            _, _, V = torch.pca_lowrank(torch.from_numpy(X), q=min(X.shape[0]-1, X.shape[1], self.rank+8), center=False, niter=2)
        self.components = V[:, :self.rank].numpy().T.copy()  # [rank, features]
        del X, chunks
        seq = np.empty((train_end, self.rank), dtype=np.float32)
        for i in range(0, train_end, chunk):
            block = self._flatten(reader.read_range(i, min(i+chunk, train_end)))
            seq[i:i+len(block)] = (block - self.spatial_mean) @ self.components.T
        self.eof_center = seq.mean(axis=0)
        seq -= self.eof_center
        targets = seq[self.lags:]
        regressors = np.concatenate([seq[self.lags-j-1:train_end-j-1] for j in range(self.lags)], axis=1)
        design = np.concatenate([np.ones((len(regressors), 1), dtype=np.float64), regressors.astype(np.float64)], axis=1)
        targets = targets.astype(np.float64)
        gram = design.T @ design
        regularizer = np.eye(gram.shape[0]) * self.ridge
        regularizer[0, 0] = 0.0
        self.beta = np.linalg.solve(gram + regularizer, design.T @ targets).astype(np.float32)
        residuals = targets - design @ self.beta
        self.resid_chol = np.linalg.cholesky(np.cov(residuals, rowvar=False) + 1e-5*np.eye(self.rank)).astype(np.float32)
        print(f'[EOF-VAR] fitted {self.lags} lags, n_train={train_end}; innovation mean std={np.std(residuals):.4f}', flush=True)
        return self

    def _flatten(self, frames):
        # raw frames [N,C,H,W]; mask land, then ravel per channel.
        norm = (frames - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
        return norm[:, :, self.mask].reshape(len(frames), -1)

    def predict(self, history, horizon=3, members=1, seed=None):
        if isinstance(history, torch.Tensor):
            history = history.detach().cpu().numpy()
        history = np.asarray(history, dtype=np.float32)
        batch, timesteps, channels, h, w = history.shape
        if timesteps < self.lags or (channels, h, w) != self.shape:
            raise ValueError('Incompatible VAR history shape / lag order')
        rng = np.random.default_rng(seed)
        scores = (history[:, :, :, self.mask].reshape(batch, timesteps, -1) - self.spatial_mean) @ self.components.T - self.eof_center
        output = np.empty((batch, members, horizon, channels, h, w), dtype=np.float32)
        for member in range(members):
            lags = [scores[:, -i-1].copy() for i in range(self.lags)]
            for step in range(horizon):
                design = np.concatenate([np.ones((batch, 1), dtype=np.float32), *lags], axis=1)
                pred = design @ self.beta
                if members > 1:
                    pred += rng.standard_normal((batch, self.rank)).astype(np.float32) @ self.resid_chol.T
                reconstructed = (pred + self.eof_center) @ self.components + self.spatial_mean
                maps = np.zeros((batch, channels, h, w), dtype=np.float32)
                maps[:, :, self.mask] = reconstructed.reshape(batch, channels, -1)
                output[:, member, step] = maps
                lags = [pred] + lags[:-1]
        return output

    def state_dict(self):
        return {k: getattr(self, k) for k in ('rank','lags','ridge','snapshot_stride','seed','mask','shape','mean',
                      'std','spatial_mean','components','eof_center','beta','resid_chol')}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(*(state[k] for k in ('rank','lags','ridge','snapshot_stride','seed')))
        vars(obj).update(state)
        return obj
