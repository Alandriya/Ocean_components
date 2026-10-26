"""Shared, dependency-light data and scoring for probabilistic forecasting."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from Forecast_nn.config import cfg
from Forecast_nn.data import (resolve_flux_source, make_field_reader, load_mask,
    source_exists, load_or_create_normalization, load_or_create_psnr_ranges, OceanSequenceDataset)
from Forecast_nn.splits import resolve_splits
from Forecast_nn.runtime import make_loader_kwargs
from Forecast_nn.evaluation import to_jsonable


def prepare_data(args, *, split=None, batch_size=None, workers=None):
    src = resolve_flux_source(args.variable, cfg.data_files)
    if not source_exists(src):
        raise FileNotFoundError(f'No flux maps at {src}; edit Forecast_nn/config.py on the cluster')
    mask = load_mask(cfg.mask_file, args.downsample)
    reader = make_field_reader(src, args.downsample)
    bounds = resolve_splits(reader.length, cfg.base_date, args.end_date, args.val_steps, args.test_steps,
                            cfg.in_len, cfg.out_len, cfg.stride)
    mean, std, _ = load_or_create_normalization(src, mask, args.variable, args.downsample,
                              bounds.train_end, cfg.stats_root)
    ranges, _ = load_or_create_psnr_ranges(src, mask, args.variable, args.downsample,
                              bounds.train_end, cfg.stats_root, cfg.psnr_percentiles, cfg.psnr_max_sample_frames)
    train = OceanSequenceDataset(src, mask, *bounds.dataset_bounds('train', cfg.in_len, cfg.stride),
                                 cfg.in_len, cfg.out_len, mean, std, cfg.stride, args.downsample)
    name = split or 'val'
    evaluation = OceanSequenceDataset(src, mask, *bounds.dataset_bounds(name, cfg.in_len, cfg.stride),
                                      cfg.in_len, cfg.out_len, mean, std, cfg.stride, args.downsample)
    batch_size = batch_size or args.batch_size
    workers = args.num_workers if workers is None else workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    make = lambda ds, shuffle: DataLoader(ds, **make_loader_kwargs(batch_size, workers, device, shuffle=shuffle))
    return {'src':src, 'reader':reader, 'mask':mask, 'bounds':bounds, 'mean':mean, 'std':std,
            'ranges':ranges, 'train':train, 'evaluation':evaluation,
            'train_loader':make(train,True), 'eval_loader':make(evaluation,False), 'device':device}


def metadata(args, info):
    bounds=info['bounds']
    return {'model_name':args.model, 'variable':args.variable,'data_file':info['src'],
            'in_len':cfg.in_len,'out_len':cfg.out_len,'stride':cfg.stride,
            'downsample':args.downsample, 'mean':info['mean'],'std':info['std'],
            'psnr_ranges':info['ranges'], 'train_end_idx':bounds.train_end,
            'val_start_idx':bounds.val_start,'test_start_idx':bounds.test_start,
            'data_end_idx':bounds.data_end,'data_last_date':args.end_date,
            'width':args.width,'diffusion_steps':args.diffusion_steps,
            'var_rank':args.var_rank,'var_lags':args.var_lags,'var_ridge':args.var_ridge,
            'var_snapshot_stride':args.var_snapshot_stride,
            'sde_moment_weight':args.sde_moment_weight}


def run_tag(args):
    common = f'ds{args.downsample}_i{cfg.in_len}_o{cfg.out_len}'
    if args.model=='ddpm':
        extra=f'w{args.width}_n{args.diffusion_steps}'
    elif args.model=='gan':
        extra=f'w{args.width}'
    else:
        extra=f'r{args.var_rank}_p{args.var_lags}_a{args.var_ridge:g}'
    if args.model=='var' and args.var_snapshot_stride!=7:
        extra+=f'_ss{args.var_snapshot_stride}'
    if args.model=='gan' and args.sde_moment_weight>0:
        extra+=f'_sm{args.sde_moment_weight:g}'
    if args.run_id:
        extra+=f'_{args.run_id}'
    return f'{common}_{extra}'


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


class EnsembleScores:
    """Physical-unit, ocean-only point MSE/MAE/CRPS and ensemble spread.

    Sample shape [B,M,K,C,H,W]. The CRPS computation is exact for the empirical
    ensemble and is evaluated separately for every grid cell and lead time.
    """
    def __init__(self, horizon, std, ranges):
        self.std = np.asarray(std, np.float64)
        self.ranges = np.asarray(ranges, np.float64)
        self.errors = np.zeros((horizon, len(self.std)), np.float64)
        self.abs_errors = np.zeros_like(self.errors)
        self.count = np.zeros_like(self.errors)
        self.crps = np.zeros_like(self.errors)
        self.spread_var = np.zeros_like(self.errors)
        self.baseline = np.zeros_like(self.errors)
        self.energy_sum = np.zeros(horizon, np.float64)
        self.energy_count = np.zeros(horizon, np.float64)

    @torch.no_grad()
    def update(self, ensemble, truth, last, mask):
        # All values are normalized. Convert errors to physical units with train std.
        b, m, k, c, h, w = ensemble.shape
        scale = torch.as_tensor(self.std, device=truth.device, dtype=torch.float32).reshape(1,1,1,c,1,1)
        ys = truth.float() * scale[:,0]
        samples = ensemble.float() * scale
        pred = samples.mean(dim=1)
        error = pred - ys
        raw = (last[:,None].float()-truth.float()) * scale[:,0]
        mask_f = mask.to(device=truth.device, dtype=torch.float32).reshape(1, 1, 1, h, w)
        self.errors += (error.square() * mask_f).sum(dim=(0,3,4)).cpu().numpy()
        self.abs_errors += (error.abs() * mask_f).sum(dim=(0,3,4)).cpu().numpy()
        self.baseline += (raw.square() * mask_f).sum(dim=(0,3,4)).cpu().numpy()
        self.count += np.full((k,c), b * int(mask.sum()), dtype=np.float64)
        point = (samples - ys[:,None]).abs().mean(1)
        pair_sum = torch.zeros_like(point)
        for i in range(m):
            for j in range(i+1, m):
                pair_sum += (samples[:,i] - samples[:,j]).abs()
        point = point - pair_sum / (m*m)
        self.crps += (point * mask_f).sum(dim=(0,3,4)).cpu().numpy()
        self.spread_var += (samples.var(1, unbiased=False) * mask_f).sum(dim=(0,3,4)).cpu().numpy()
        if c == 2:
            # Two-component energy score retains information about A/B correlation.
            point_energy = torch.linalg.vector_norm(samples - ys[:, None], dim=3).mean(1)
            for i in range(m):
                for j in range(i+1,m):
                    point_energy -= torch.linalg.vector_norm(samples[:,i]-samples[:,j],dim=2)/(m*m)
            mw = mask_f[:, :, 0]
            self.energy_sum += (point_energy*mw).sum(dim=(0,2,3)).cpu().numpy()
            self.energy_count += b*int(mask.sum())

    def summary(self):
        mse = self.errors / np.maximum(self.count,1)
        baseline_mse = self.baseline / np.maximum(self.count,1)
        crps = self.crps / np.maximum(self.count,1)
        spread = np.sqrt(self.spread_var / np.maximum(self.count,1))
        psnr = 10*np.log10(self.ranges[None,:]**2 / np.maximum(mse,1e-30))
        results=[]
        for k in range(mse.shape[0]):
            results.append({'horizon':k+1,'rmse':float(np.sqrt(mse[k].mean())),
                'rmse_per_channel':np.sqrt(mse[k]).tolist(),
                'crps':float(crps[k].mean()),'crps_per_channel':crps[k].tolist(),
                'spread':float(spread[k].mean()),
                'mse_skill_vs_persistence':float(1-mse[k].sum()/max(baseline_mse[k].sum(),1e-15)),
                'psnr_db':float(psnr[k].mean())})
        return {'summary':{'overall_rmse':float(np.sqrt(mse.mean())),
                'overall_mse':float(mse.mean()),
                'overall_mae':float((self.abs_errors/np.maximum(self.count,1)).mean()),
                'overall_psnr':float(psnr.mean()), 'crps':float(crps.mean()),
                'spread':float(spread.mean()),
                'spread_skill_ratio':float(np.sqrt((self.spread_var/np.maximum(self.count,1)).mean()) / max(np.sqrt(mse.mean()),1e-12)),
                'bivariate_energy_score':(float((self.energy_sum/np.maximum(self.energy_count,1)).mean())
                                          if np.any(self.energy_count) else None),
                'mse_skill_vs_persistence':float(1-mse.sum()/max(baseline_mse.sum(),1e-15))},
                'per_horizon':results, 'per_channel':[{'channel':i, 'rmse':float(np.sqrt(mse[:,i].mean())),
                        'crps':float(crps[:,i].mean()),'psnr_db':float(psnr[:,i].mean())}
                        for i in range(len(self.std))]}
