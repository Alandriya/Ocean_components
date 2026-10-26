"""Train conditional DDPM, conditional GAN or low-rank EOF-VAR.

Examples:
 python -m Forecast_nn.train_generative --model ddpm --variable joint --epochs 60 --amp
 python -m Forecast_nn.train_generative --model gan --variable joint --epochs 70 --amp
 python -m Forecast_nn.train_generative --model var --variable joint --var-rank 24 --var-lags 5

Run from the repository root so Forecast_nn is importable. Training uses only
train and validation data. The final year (test) is untouched.
"""
import argparse
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Forecast_nn.config import cfg
from Forecast_nn.runtime import configure_runtime, make_loader_kwargs
from Forecast_nn.logging_utils import tee_output
from Forecast_nn.generative_utils import prepare_data, metadata, run_tag, save_json
from Forecast_nn.models.generative import ConditionalDDPM, ConditionalGAN
from Forecast_nn.models.eof_var import EOFVAR
from Forecast_nn.joint_sde import JointSDEReader, JointSDEDataset, joint_ensemble_moment_loss


def arguments():
    p = argparse.ArgumentParser(description='Conditional generative ocean forecast models')
    p.add_argument('--model', choices=('ddpm','gan','var'), required=True)
    p.add_argument('--variable', choices=('sensible','latent','joint'), default='joint')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--diffusion-steps', type=int, default=50)
    p.add_argument('--gan-recon-weight', type=float, default=10.0,
                   help='MSE loss on mean of two independent generated trajectories.')
    p.add_argument('--gan-adv-weight', type=float, default=0.1)
    p.add_argument('--downsample', type=int, default=cfg.downsample)
    p.add_argument('--end-date', default=cfg.data_last_date)
    p.add_argument('--val-steps', type=int, default=cfg.val_steps)
    p.add_argument('--test-steps', type=int, default=cfg.test_steps)
    p.add_argument('--run-id', default='')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--max-train-batches', type=int, default=0, help='SMOKE TEST ONLY: not a complete epoch')
    p.add_argument('--max-val-batches', type=int, default=0, help='SMOKE TEST ONLY: not a complete validation')
    p.add_argument('--var-rank', type=int, default=24)
    p.add_argument('--var-lags', type=int, default=5)
    p.add_argument('--var-ridge', type=float, default=1.0)
    p.add_argument('--var-snapshot-stride', type=int, default=7)
    p.add_argument('--sde-joint-a-file', default=None)
    p.add_argument('--sde-joint-b-file', default=None)
    p.add_argument('--sde-b-layout', choices=('row_major','column_major'), default='row_major')
    p.add_argument('--sde-moment-weight', type=float, default=0.0, help='GAN ONLY: weak mean/cov moment constraint on 4 ensemble members')
    p.add_argument('--sde-cov-weight', type=float, default=0.01, help='Relative weight of covariance mismatch within SDE GAN moment loss')
    p.add_argument('--seed', type=int, default=cfg.seed)
    return p.parse_args()


def run(args):
    configure_runtime(args.seed, cfg.fast_cuda)
    if args.sde_moment_weight > 0 and (args.model != 'gan' or args.variable != 'joint'):
        raise ValueError('Joint SDE ensemble moment penalty currently implemented only for joint GAN')
    info = prepare_data(args)
    dev = info['device']
    tag = run_tag(args)
    outdir = Path(cfg.results_root)/args.model/args.variable/tag
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir/'best.pth'
    last_path = outdir/'last.pth'
    meta = metadata(args, info)
    meta.update({'run_tag':tag,'seed':args.seed,'sde_joint_a_file':args.sde_joint_a_file,
                 'sde_joint_b_file':args.sde_joint_b_file,'sde_b_layout':args.sde_b_layout})
    print(f'[model] {args.model}, variable={args.variable}; device={dev}, train={len(info["train"])} val={len(info["evaluation"])}')
    print(f'[split] {info["bounds"]}; output={outdir}')
    if args.model=='var':
        if args.resume:
            raise ValueError('VAR fit is deterministic; do not use --resume')
        model = EOFVAR(args.var_rank, args.var_lags, args.var_ridge,args.var_snapshot_stride,args.seed)
        model.fit(info['reader'],info['mask'],info['bounds'].train_end,info['mean'],info['std'])
        torch.save({'metadata':meta,'model_name':'var','var_state':model.state_dict(), 'epoch':1},ckpt_path)
        print(f'[saved] {ckpt_path}')
        return ckpt_path

    params = {'channels':info['reader'].channels,'in_len':cfg.in_len,'out_len':cfg.out_len,'width':args.width}
    if args.model=='ddpm':
        model=ConditionalDDPM(**params,diffusion_steps=args.diffusion_steps).to(dev)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        opt_d=None
    else:
        model=ConditionalGAN(**params).to(dev)
        optimizer=torch.optim.AdamW(list(model.generator.parameters())+list(model.condition.parameters()), lr=args.lr, betas=(.5,.999))
        opt_d=torch.optim.AdamW(model.critic.parameters(),lr=args.lr,betas=(.5,.999))

    sde=None
    if args.sde_moment_weight>0:
        if not args.sde_joint_a_file or not args.sde_joint_b_file:
            raise ValueError('SDE moment loss requires both full joint coefficient files')
        sde=JointSDEReader(args.sde_joint_a_file,args.sde_joint_b_file,args.downsample,
                    info['mask'].shape,info['std'],b_layout=args.sde_b_layout)
        ds=JointSDEDataset(info['train'],sde)
        info['train_loader']=DataLoader(ds,**make_loader_kwargs(args.batch_size,args.num_workers,dev,shuffle=True))

    mask=torch.as_tensor(info['mask'],dtype=torch.float32,device=dev)
    use_amp=args.amp and dev.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    best=float('inf')
    best_epoch=0
    first=1
    history=[]
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(last_path)
        old=torch.load(last_path,map_location=dev,weights_only=False)
        if old['metadata']['data_end_idx'] != meta['data_end_idx'] or old['metadata']['val_start_idx'] != meta['val_start_idx']:
            raise ValueError('Refusing to resume on different chronological split')
        model.load_state_dict(old['model_state'])
        optimizer.load_state_dict(old['optimizer_state'])
        if opt_d is not None:
            opt_d.load_state_dict(old['critic_optimizer_state'])
        scaler.load_state_dict(old['scaler_state'])
        best=float(old['best_val']);best_epoch=int(old['best_epoch'])
        first=int(old['epoch'])+1
        history=list(old['history'])
        print(f'[resume] {last_path} epoch={first-1}')
    epochs_no_best=max(0, first-best_epoch-1)
    print(f'[model] parameters={sum(p.numel() for p in model.parameters()):,} AMP={use_amp}')

    for epoch in range(first,first+args.epochs):
        started=time.time()
        model.train()
        total=0.0; n=0
        for i,batch in enumerate(info['train_loader'],1):
            if args.max_train_batches and i>args.max_train_batches: break
            if sde is not None:
                x,y,a,bcoef=batch
                a=a.to(dev); bcoef=bcoef.to(dev)
            else:
                x,y=batch
            x=x.to(dev);y=y.to(dev)
            if args.model=='ddpm':
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=use_amp):
                    loss=model.training_loss(x,y,mask)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
                scaler.step(optimizer);scaler.update()
            else:
                # Critic pass: generated samples detached; no gradients to generator.
                for p in model.critic.parameters(): p.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=use_amp):
                    fake=model.generate(x,mask).detach()
                    d_real=model.critic(x,y,mask)
                    d_fake=model.critic(x,fake,mask)
                    dloss=F.relu(1-d_real).mean()+F.relu(1+d_fake).mean()
                scaler.scale(dloss).backward()
                scaler.step(opt_d);scaler.update()
                # Generator pass: reconstruction of the ensemble mean, NOT single-member L2.
                for p in model.critic.parameters(): p.requires_grad_(False)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=use_amp):
                    member1=model.generate(x,mask)
                    member2=model.generate(x,mask)
                    average=(member1+member2)/2
                    valid=mask.reshape(1,1,1,*mask.shape)
                    recon=((average-y).square()*valid).sum()/(valid.sum()*y.shape[0]*y.shape[1]*y.shape[2]).clamp_min(1)
                    adv=-model.critic(x,member1,mask).mean()
                    loss=args.gan_recon_weight*recon+args.gan_adv_weight*adv
                    if sde is not None:
                        member3=model.generate(x,mask)
                        member4=model.generate(x,mask)
                        drift, covariance=joint_ensemble_moment_loss(
                            torch.stack([member1,member2,member3,member4],1),x[:,-1],a,bcoef,mask)
                        loss=loss+args.sde_moment_weight*(drift+args.sde_cov_weight*covariance)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip>0: torch.nn.utils.clip_grad_norm_(list(model.generator.parameters())+list(model.condition.parameters()),args.grad_clip)
                scaler.step(optimizer);scaler.update()
                for p in model.critic.parameters():p.requires_grad_(True)
            total+=float(loss.detach());n+=1
            if i==1 or i%100==0:
                print(f'  epoch {epoch} batch {i}/{len(info["train_loader"])} train_loss={total/n:.5f}',flush=True)

        # Fixed validation random stream, so noise differs little across checkpoints.
        model.eval();val=0.0;nv=0
        with torch.no_grad(), torch.random.fork_rng(devices=[dev.index or 0] if dev.type=='cuda' else []):
            torch.manual_seed(args.seed+123)
            if dev.type=='cuda':torch.cuda.manual_seed_all(args.seed+123)
            for j,batch in enumerate(info['eval_loader'],1):
                if args.max_val_batches and j>args.max_val_batches:break
                x,y=[item.to(dev) for item in batch[:2]]
                with torch.amp.autocast('cuda',enabled=use_amp):
                    if args.model=='ddpm':
                        v=model.training_loss(x,y,mask)
                    else:
                        out=(model.generate(x,mask)+model.generate(x,mask))/2
                        valid=mask.reshape(1,1,1,*mask.shape)
                        v=((out-y).square()*valid).sum()/(valid.sum()*y.shape[0]*y.shape[1]*y.shape[2]).clamp_min(1)
                val+=float(v)*len(x);nv+=len(x)
        val/=max(nv,1)
        rec={'epoch':epoch,'train_loss':total/max(n,1),'val_score':val,'epoch_seconds':time.time()-started}
        history.append(rec)
        improved=math.isfinite(val) and val<best-1e-6
        if improved: best=val;best_epoch=epoch;epochs_no_best=0
        else: epochs_no_best+=1
        record={'metadata':meta,'model_name':args.model,'model_state':model.state_dict(),
                'optimizer_state':optimizer.state_dict(),'critic_optimizer_state':opt_d.state_dict() if opt_d else None,
                'scaler_state':scaler.state_dict(),'epoch':epoch,'best_val':best,'best_epoch':best_epoch,'history':history}
        torch.save(record,last_path)
        if improved:torch.save(record,ckpt_path)
        print(f'[epoch {epoch}] train={total/max(n,1):.5f}; validation={val:.5f}; best={best:.5f}@{best_epoch}; {rec["epoch_seconds"]:.1f}s',flush=True)
        save_json(outdir/'history.json',history)
        if args.patience>0 and epochs_no_best>=args.patience:
            print(f'[early stop] {epochs_no_best} epochs without improvement',flush=True)
            break
    print(f'[saved] best={ckpt_path}; last={last_path}')
    if args.model=='ddpm':print('[note] DDPM validation score is *denoising noise MSE*, not forecast RMSE.')
    return ckpt_path


def main():
    args=arguments()
    tag=run_tag(args)
    logfile=Path(cfg.results_root).parent/'logs'/f'train_{args.model}_{tag}_{args.variable}.txt'
    with tee_output(logfile):run(args)

if __name__=='__main__':main()
