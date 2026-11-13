"""Train two research variants: simvp_ddm and latent_sde_rnn.

From repo root:
 python -m Forecast_nn.train_ddm --model simvp_ddm --variable joint --joint-a-file ... --joint-b-file ...
 python -m Forecast_nn.train_ddm --model latent_sde_rnn --variable joint --joint-a-file ... --joint-b-file ...

Both variants use TARGET-DISJOINT chronological train/val and never see test.
For no-SDE ablations set --lambda-a 0 --lambda-d 0.
"""
from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Forecast_nn.config import cfg
from Forecast_nn.ddm_utils import (prepare, name, experiment_root, info_metadata,
                                   assert_boundaries, unpack, forecast_mse,
                                   moments_from_history, training_qref)
from Forecast_nn.evaluation import OnlineErrorMetrics
from Forecast_nn.generative_utils import save_json
from Forecast_nn.joint_sde import joint_drift_loss
from Forecast_nn.logging_utils import tee_output
from Forecast_nn.models.ddm import DDMSimVP, EOFProjector, LatentSDERNN, aux_moment_losses
from Forecast_nn.runtime import configure_runtime


def arguments(argv=None):
    p=argparse.ArgumentParser(description='SDE-informed joint ocean forecasts (DDM-inspired)')
    p.add_argument('--model',choices=('simvp_ddm','latent_sde_rnn'),required=True)
    p.add_argument('--variable',choices=('joint',),default='joint')
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--patience',type=int,default=12)
    p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--grad-accum',type=int,default=1)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--seed',type=int,default=cfg.seed)
    p.add_argument('--run-id',default='')
    p.add_argument('--downsample',type=int,default=cfg.downsample)
    p.add_argument('--end-date',default=cfg.data_last_date)
    p.add_argument('--val-steps',type=int,default=cfg.val_steps)
    p.add_argument('--test-steps',type=int,default=cfg.test_steps)
    p.add_argument('--joint-a-file',default=None,help='Historical full joint A: [T,2,H,W] etc.')
    p.add_argument('--joint-b-file',default=None,help='Historical full joint B: [T,4,H,W] etc.')
    p.add_argument('--joint-b-layout',choices=('diag_first','row_major','column_major'),default='diag_first',
                   help='Four-channel B: diag_first=[B11,B22,B12,B21] (repository default); row_major=[B11,B12,B21,B22]')
    p.add_argument('--sde-time-offset',type=int,default=1,help='Flux index represented by coefficient element 0')
    p.add_argument('--lambda-a',type=float,default=0.05,help='Auxiliary historical drift matching')
    p.add_argument('--lambda-d',type=float,default=0.02,help='Auxiliary historical covariance matching')
    p.add_argument('--cov-scale',type=float,default=0.5,help='Scale in signed asinh covariance loss')
    p.add_argument('--lambda-forecast-drift',type=float,default=0.,
                   help='OPTIONAL regularize SimVP first-step increment towards last observed A_t')
    p.add_argument('--joint-cov-floor',type=float,default=0.1)
    p.add_argument('--base-channels',type=int,default=16)
    p.add_argument('--depth',type=int,default=3)
    p.add_argument('--temporal-blocks',type=int,default=4)
    p.add_argument('--init-from-simvp',default=None,help='Optional SimVP joint best.pth for backbone initialization')
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--pca-snapshots',type=int,default=512)
    p.add_argument('--latent-nll-weight',type=float,default=0.1)
    p.add_argument('--latent-mean-weight',type=float,default=1.)
    p.add_argument('--cov-prior',choices=('trace_scaled','empirical','independent'),
                   default='trace_scaled',help='Latent D reference; trace_scaled is a heuristic, independent assumes site-wise white noise')
    p.add_argument('--max-train-batches',type=int,default=0,help='DEBUG only: truncated training')
    p.add_argument('--max-val-batches',type=int,default=0,help='DEBUG only: truncated validation')
    args=p.parse_args(argv)
    if args.epochs<1 or args.patience<0 or args.grad_accum<1:
        p.error('epochs/grad-accum must be positive and patience non-negative')
    if min(args.lambda_a,args.lambda_d,args.lambda_forecast_drift,args.cov_scale,args.joint_cov_floor)<0:
        p.error('Loss weights and covariance scale must be non-negative')
    if args.cov_scale<=0 or args.joint_cov_floor<=0:
        p.error('Covariance scales must be positive')
    if args.model!='simvp_ddm' and args.init_from_simvp:
        p.error('--init-from-simvp applies only to simvp_ddm')
    return args


def simvp_params(args):
    params=dict(cfg.model_params['simvp'])
    params.update(base_channels=args.base_channels,depth=args.depth,temporal_blocks=args.temporal_blocks)
    return params


def init_model(args,projector,device):
    if args.model=='simvp_ddm':
        return DDMSimVP(cfg.in_len,cfg.out_len,simvp_params(args)).to(device)
    return LatentSDERNN(args.rank,args.hidden,dt=1.).to(device)


def run(args):
    configure_runtime(args.seed, cfg.fast_cuda)
    if args.lambda_a==0 and args.lambda_d==0 and args.lambda_forecast_drift==0:
        print('[ablation] no SDE coefficient supervision; only observations are used.',flush=True)
    info=prepare(args,training=True)
    dev=info['device']; mask=torch.as_tensor(info['mask'],device=dev,dtype=torch.bool)
    amp=bool(args.amp and dev.type=='cuda')
    tag=name(args); outdir=experiment_root(args,tag);outdir.mkdir(parents=True,exist_ok=True)
    best_path=outdir/'best.pth';last_path=outdir/'last.pth'
    checkpoint=torch.load(last_path,map_location='cpu',weights_only=False) if args.resume else None
    if args.resume and checkpoint is None:raise FileNotFoundError(last_path)
    meta=info_metadata(args,info,tag)
    meta.update({'simvp_params':simvp_params(args) if args.model=='simvp_ddm' else None,
                 'rank':args.rank,'hidden':args.hidden,'pca_snapshots':args.pca_snapshots,
                 'cov_prior':args.cov_prior,'cov_scale':args.cov_scale,
                 'lambda_forecast_drift':args.lambda_forecast_drift,
                 'latent_nll_weight':args.latent_nll_weight,
                 'latent_mean_weight':args.latent_mean_weight,
                 'joint_cov_floor':args.joint_cov_floor,
                 'init_from_simvp':args.init_from_simvp})
    if checkpoint is not None:
        assert_boundaries(checkpoint['metadata'],info)
        if checkpoint['metadata']['model_name']!=args.model or checkpoint['metadata']['run_tag']!=tag:
            raise ValueError('Resume checkpoint architecture/run tag differs')
        for k in ('lambda_a','lambda_d','lambda_forecast_drift',
                  'joint_a_file','joint_b_file','joint_b_layout','cov_prior','cov_scale'):
            if checkpoint['metadata'].get(k)!=meta.get(k):
                raise ValueError(f'Resume setting {k} differs from saved checkpoint')
    if args.model=='latent_sde_rnn':
        projector=(EOFProjector.from_state_dict(checkpoint['projector_state']) if checkpoint is not None
                   else EOFProjector(args.rank,args.pca_snapshots,args.seed).fit(
                       info['reader'],info['mask'],info['bounds'].train_end,info['mean'],info['std']))
        projector.to_torch(dev)
    else:
        projector=None
    model=init_model(args,projector,dev)
    if args.init_from_simvp and checkpoint is None:
        base=torch.load(args.init_from_simvp,map_location='cpu',weights_only=False)
        if base.get('model_name')!='simvp' or base.get('variable')!='joint':
            raise ValueError('Initialization checkpoint is not joint SimVP')
        model.backbone.load_state_dict(base['model_state'])
        print(f'[init] copied deterministic backbone from {args.init_from_simvp}',flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    history=[];start=1;best=float('inf');best_epoch=0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        start=int(checkpoint['epoch'])+1
        best=float(checkpoint['best_val_rmse']);best_epoch=int(checkpoint['best_epoch'])
        history=list(checkpoint['history'])
        print(f'[resume] from epoch {start-1}',flush=True)
    qref=training_qref(info['joint_reader'],info['bounds'].train_end,info['mask']) if (
        args.model=='latent_sde_rnn' and args.lambda_d and args.cov_prior=='trace_scaled') else 1.
    if checkpoint is not None and args.model=='latent_sde_rnn':
        qref=float(checkpoint.get('qref',qref))
    print(f'[setup] {args.model} {tag}; train={len(info["train"])} val={len(info["evaluation"])} '
          f'device={dev}, AMP={amp}, params={sum(p.numel() for p in model.parameters()):,}, qref={qref:.5g}',flush=True)
    if args.cov_prior=='independent' and args.lambda_d:
        print('[WARNING] local_independent B projection assumes INDEPENDENT random forcing at all ocean pixels.',flush=True)
    idle=max(0,start-best_epoch-1) if start>1 else 0
    for epoch in range(start,start+args.epochs):
        begun=time.time();model.train();total=0.;pieces={'forecast':0.,'drift':0.,'cov':0.,'nll':0.}
        opt_batches=0
        optimizer.zero_grad(set_to_none=True)
        n_batches=min(len(info['train_loader']),args.max_train_batches or len(info['train_loader']))
        for idx,batch in enumerate(info['train_loader'],1):
            if idx>n_batches:break
            x,y,*coefs=unpack(batch,dev)
            a,b=coefs if coefs else (None,None)
            with torch.amp.autocast('cuda',enabled=amp):
                if args.model=='simvp_ddm':
                    forecast,ahat,dhat=model(x,mask)
                    base=forecast_mse(forecast,y,mask)
                    if a is not None:
                        la,ld=aux_moment_losses(ahat,dhat,a,b,mask,args.cov_scale)
                    else:
                        la=ld=base.new_zeros(())
                    extra=base.new_zeros(())
                    if args.lambda_forecast_drift:
                        extra=joint_drift_loss(forecast,x[:,-1],a,b,mask,
                                               covariance_floor=args.joint_cov_floor)
                    loss=base+args.lambda_a*la+args.lambda_d*ld+args.lambda_forecast_drift*extra
                    nll=base.new_zeros(())
                else:
                    zin=projector.encode(x)
                    zout=projector.encode(y)
                    loss,details=model.fit_loss(zin,zout,args.latent_mean_weight,args.latent_nll_weight)
                    base=details['latent_mse'];nll=details['latent_nll']
                    la,ld=moments_from_history(projector,model,details,a,b,mask,args,qref)
                    loss=loss+args.lambda_a*la+args.lambda_d*ld
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at epoch={epoch} batch={idx}')
            group_first=((idx-1)//args.grad_accum)*args.grad_accum
            size=min(args.grad_accum,n_batches-group_first)
            scaler.scale(loss/size).backward()
            if idx % args.grad_accum==0 or idx==n_batches:
                scaler.unscale_(optimizer)
                if args.grad_clip>0:torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                opt_batches+=1
            total+=float(loss.detach());pieces['forecast']+=float(base.detach())
            pieces['drift']+=float(la.detach());pieces['cov']+=float(ld.detach());pieces['nll']+=float(nll.detach())
            if idx==1 or idx % 100==0 or idx==n_batches:
                print(f'[train] epoch {epoch} batch {idx}/{n_batches}: loss={total/idx:.5f} '
                      f'mean={pieces["forecast"]/idx:.5f} A={pieces["drift"]/idx:.5f} '
                      f'D={pieces["cov"]/idx:.5f} NLL={pieces["nll"]/idx:.5f}',flush=True)
        if not opt_batches:raise RuntimeError('Empty train loader')
        model.eval();vmetric=OnlineErrorMetrics(cfg.out_len,info['std'],info['ranges']);vn=0
        with torch.inference_mode():
            for idx,batch in enumerate(info['eval_loader'],1):
                if args.max_val_batches and idx>args.max_val_batches:break
                x,y,*_=unpack(batch,dev)
                with torch.amp.autocast('cuda',enabled=amp):
                    if args.model=='simvp_ddm':pred,_,_=model(x,mask)
                    else:pred=projector.decode(model.mean_forecast(projector.encode(x),cfg.out_len))
                vmetric.update(pred,y,mask)
                vn+=len(x)
        if not vn:raise RuntimeError('Empty validation split')
        score=vmetric.summary();rmse=score['overall_rmse']
        if not math.isfinite(rmse):raise FloatingPointError('Nonfinite validation RMSE')
        improved=rmse<best-1e-6
        if improved:best=rmse;best_epoch=epoch;idle=0
        else:idle+=1
        history.append({'epoch':epoch,'train_loss':total/n_batches,
                        'val_rmse':rmse,'val_psnr':score['overall_psnr'],
                        **{k:v/n_batches for k,v in pieces.items()},'seconds':time.time()-begun})
        payload={'metadata':meta,'model_state':model.state_dict(),
                 'optimizer_state':optimizer.state_dict(),'scaler_state':scaler.state_dict(),
                 'projector_state':projector.state_dict() if projector else None,
                 'qref':qref,'epoch':epoch,'best_epoch':best_epoch,
                 'best_val_rmse':best,'history':history}
        torch.save(payload,last_path)
        if improved:torch.save(payload,best_path)
        save_json(outdir/'history.json',history)
        print(f'[epoch {epoch}] train={total/n_batches:.5f} val_RMSE={rmse:.5f} '
              f'val_PSNR={score["overall_psnr"]:.3f} dB best={best:.5f}@{best_epoch} '
              f'duration={time.time()-begun:.1f}s',flush=True)
        if args.patience>0 and idle>=args.patience:
            print(f'[early stopping] {idle} epochs without improving validation RMSE',flush=True)
            break
    print(f'[saved] best={best_path} last={last_path}',flush=True)
    return best_path


def main(argv=None):
    args=arguments(argv)
    tag=name(args)
    logfile=Path(cfg.results_root).parent/'logs'/f'train_{args.model}_{tag}_joint.txt'
    with tee_output(logfile):run(args)


if __name__=='__main__':main()
