"""Shared DDM experiments: paths, run names, coefficient preparation, losses."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Forecast_nn.config import cfg
from Forecast_nn.generative_utils import prepare_data
from Forecast_nn.joint_sde import JointSDEReader, JointSDEDataset, joint_drift_loss
from Forecast_nn.models.ddm import mask_mean
from Forecast_nn.runtime import make_loader_kwargs


def name(args):
    common=f'ds{args.downsample}_i{cfg.in_len}_o{cfg.out_len}'
    if args.model=='simvp_ddm':
        tag=f'{common}_b{args.base_channels}_d{args.depth}_tb{args.temporal_blocks}'
    else:
        tag=f'{common}_r{args.rank}_h{args.hidden}'
    if args.lambda_a:
        tag+=f'_a{args.lambda_a:g}'
    if args.lambda_d:
        tag+=f'_d{args.lambda_d:g}'
    if args.model=='simvp_ddm' and args.lambda_forecast_drift:
        tag+=f'_fd{args.lambda_forecast_drift:g}'
    if args.model=='latent_sde_rnn' and args.lambda_d:
        tag+=f'_{args.cov_prior}'
    if args.run_id:
        if not args.run_id.replace('-','').replace('_','').isalnum():
            raise ValueError('--run-id may contain letters, numbers, - and _ only')
        tag+=f'_{args.run_id}'
    return tag


def experiment_root(args, tag):
    return Path(cfg.results_root)/args.model/'joint'/tag


def prepare(args, *, split='val', training=False):
    if args.variable!='joint':
        raise ValueError('Both bivariate DDM experiments require --variable joint.')
    info=prepare_data(args, split=split, batch_size=args.batch_size, workers=args.num_workers)
    use_sde=bool(args.lambda_a or args.lambda_d or (args.model=='simvp_ddm' and args.lambda_forecast_drift))
    joint=None
    if use_sde and training:
        if not args.joint_a_file or not args.joint_b_file:
            raise ValueError('SDE losses need --joint-a-file and --joint-b-file. '
                             'For a non-informed control, set --lambda-a 0 --lambda-d 0.')
        joint=JointSDEReader(args.joint_a_file,args.joint_b_file,args.downsample,
                             info['mask'].shape,info['std'],time_offset=args.sde_time_offset,
                             b_layout=args.joint_b_layout)
        # Fail BEFORE expensive training when the last historical coefficient
        # needed by the validation input window is missing.
        val_ds=info['evaluation']
        first_seen=info['train'].start_idx + (cfg.in_len-1)*cfg.stride
        last_seen=val_ds.start_idx + len(val_ds) - 1 + (cfg.in_len-1)*cfg.stride
        if first_seen < joint.time_offset or last_seen-joint.time_offset >= joint.a.length:
            raise ValueError(f'Coefficient coverage {joint.time_offset}..{joint.time_offset+joint.a.length-1} '
                             f'does not include observed input indices {first_seen}..{last_seen}.')
        # Val uses its *observed history* coefficients, never future target coefficients.
        for which,shuffle in [('train',True),('evaluation',False)]:
            dataset=JointSDEDataset(info[which],joint)
            info[which+'_loader']=DataLoader(dataset,**make_loader_kwargs(
                args.batch_size,args.num_workers,info['device'],shuffle=shuffle))
    info['joint_reader']=joint
    return info


def info_metadata(args, info, tag):
    b=info['bounds']
    return dict(model_name=args.model,variable='joint',run_tag=tag,downsample=args.downsample,
                in_len=cfg.in_len,out_len=cfg.out_len,stride=cfg.stride,
                train_end_idx=b.train_end,val_start_idx=b.val_start,test_start_idx=b.test_start,
                data_end_idx=b.data_end,data_last_date=args.end_date,
                mean=info['mean'],std=info['std'],psnr_ranges=info['ranges'],
                joint_a_file=args.joint_a_file,joint_b_file=args.joint_b_file,
                joint_b_layout=args.joint_b_layout,sde_time_offset=args.sde_time_offset,
                lambda_a=args.lambda_a,lambda_d=args.lambda_d,
                seed=args.seed)


def assert_boundaries(meta, info):
    b=info['bounds']
    for key,val in [('train_end_idx',b.train_end),('val_start_idx',b.val_start),
                    ('test_start_idx',b.test_start),('data_end_idx',b.data_end)]:
        if int(meta[key])!=int(val):
            raise ValueError(f'Split mismatch: checkpoint {key}={meta[key]}, requested {val}')


def unpack(batch, device):
    return tuple(x.to(device,non_blocking=True) for x in batch)


def forecast_mse(pred, target, mask):
    return mask_mean((pred.float()-target.float()).square(),mask)


def moments_from_history(projector, model, details, a_ref, b_ref, mask, args, qref=1.):
    """SDE moments for *last-observed* t only. No future maps are accessed."""
    loss_a=details['latent_mse'].new_zeros(())
    loss_d=loss_a.clone()
    if a_ref is None:
        return loss_a,loss_d
    if args.lambda_a:
        tgt=projector.project_drift(a_ref)
        loss_a=F.smooth_l1_loss(details['first_drift'],tgt)
    if args.lambda_d:
        if args.cov_prior=='independent':
            # Strong independent-spatial-Brownian assumption; OFF by default.
            target=projector.covariance_target(b_ref)
        elif args.cov_prior=='trace_scaled':
            # Heuristic: use local B's trace to scale an EMPIRICAL latent
            # covariance measured from train-only EOF innovations. This is NOT
            # the mathematically exact projection of the full field covariance.
            from Forecast_nn.joint_sde import covariance_from_b
            d=covariance_from_b(b_ref)
            q=(d[:,0,0]+d[:,1,1])[:,mask].mean(dim=-1)
            target=projector._innovation_cov[None]* (q/qref).clamp(0.05,20)[:,None,None]
        elif args.cov_prior=='empirical':
            target=projector._innovation_cov[None].expand(len(a_ref),-1,-1)
        else:
            raise ValueError('Unknown covariance prior')
        loss_d=F.smooth_l1_loss(torch.asinh(details['first_cov']/args.cov_scale),
                                torch.asinh(target/args.cov_scale))
    return loss_a,loss_d


def training_qref(joint_reader, train_end, mask, samples=64):
    """Fixed scale for B trace: ONLY coefficient dates in the training period."""
    if joint_reader is None:return 1.
    from Forecast_nn.joint_sde import covariance_from_b
    inds=np.unique(np.linspace(max(joint_reader.time_offset,1),train_end-1,min(samples,train_end-1)).astype(int))
    vals=[]
    for i in inds:
        _,b=joint_reader.get_normalized([i])
        d=covariance_from_b(b)
        vals.append(float((d[0,0,0][mask]+d[0,1,1][mask]).mean()))
    return max(float(np.mean(vals)),1e-6)
