"""Evaluate true probabilistic ensembles (DDPM, GAN, EOF-VAR) vs persistence.

Run --split val for model selection; use --split test only after freezing model.
"""
import argparse
from pathlib import Path
import torch
from Forecast_nn.config import cfg
from Forecast_nn.logging_utils import tee_output
from Forecast_nn.generative_utils import prepare_data, run_tag, save_json, EnsembleScores
from Forecast_nn.models.generative import ConditionalDDPM, ConditionalGAN
from Forecast_nn.models.eof_var import EOFVAR


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=('ddpm','gan','var'),required=True)
    p.add_argument('--variable',choices=('sensible','latent','joint'),default='joint')
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--members',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--downsample',type=int,default=cfg.downsample)
    p.add_argument('--end-date',default=cfg.data_last_date)
    p.add_argument('--val-steps',type=int,default=cfg.val_steps)
    p.add_argument('--test-steps',type=int,default=cfg.test_steps)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--diffusion-steps',type=int,default=50)
    p.add_argument('--var-rank',type=int,default=24)
    p.add_argument('--var-lags',type=int,default=5)
    p.add_argument('--var-ridge',type=float,default=1.)
    p.add_argument('--var-snapshot-stride',type=int,default=7)
    p.add_argument('--sde-moment-weight',type=float,default=0.)
    p.add_argument('--run-id',default='')
    p.add_argument('--seed',type=int,default=2025)
    p.add_argument('--max-batches',type=int,default=0,help='SMOKE TEST ONLY: incomplete evaluation')
    return p.parse_args()


def run(args):
    if args.members<2:
        raise ValueError('Probabilistic evaluation requires at least 2 members for meaningful spread/CRPS')
    info=prepare_data(args,split=args.split)
    dev=info['device']
    tag=run_tag(args)
    default=Path(cfg.results_root)/args.model/args.variable/tag/'best.pth'
    ckpt=Path(args.checkpoint) if args.checkpoint else default
    saved=torch.load(ckpt,map_location='cpu',weights_only=False)
    meta=saved['metadata']
    if meta['model_name'] != args.model or meta['variable'] != args.variable:
        raise ValueError('Model or variable incompatible with checkpoint')
    if (meta['in_len'], meta['out_len'], meta['downsample']) != (cfg.in_len, cfg.out_len, args.downsample):
        raise ValueError('Checkpoint input/output length or downsample differs from current configuration')
    for key, actual in [('train_end_idx',info['bounds'].train_end),('val_start_idx',info['bounds'].val_start),
                        ('test_start_idx',info['bounds'].test_start),('data_end_idx',info['bounds'].data_end)]:
        if meta[key]!=actual:
            raise ValueError(f'Chronological boundary {key}: checkpoint={meta[key]}, evaluated={actual}')
    if args.model=='var':
        model=EOFVAR.from_state_dict(saved['var_state'])
    else:
        kw=dict(channels=info['reader'].channels,in_len=meta['in_len'],out_len=meta['out_len'],width=meta['width'])
        if args.model=='ddpm':
            model=ConditionalDDPM(**kw,diffusion_steps=meta['diffusion_steps'])
        else:model=ConditionalGAN(**kw)
        model.load_state_dict(saved['model_state'])
        model=model.to(dev).eval()
    scores=EnsembleScores(meta['out_len'],info['std'],info['ranges'])
    mask=torch.as_tensor(info['mask'],device=dev,dtype=torch.float32)
    torch.manual_seed(args.seed)
    if dev.type=='cuda':torch.cuda.manual_seed_all(args.seed)
    print(f'[test] {args.model} {args.variable}, {args.split}, M={args.members}, checkpoint={ckpt}',flush=True)
    processed=0
    for idx,batch in enumerate(info['eval_loader'],1):
        if args.max_batches and idx>args.max_batches:break
        x,y=[v.to(dev) for v in batch[:2]]
        with torch.inference_mode():
            if args.model=='var':
                samples=torch.from_numpy(model.predict(x,horizon=meta['out_len'],members=args.members,
                                                      seed=args.seed+idx)).to(dev)
            else:
                samples=model.sample(x,mask,members=args.members)
            scores.update(samples,y,x[:,-1],mask)
        processed+=len(x)
        if idx==1 or idx%20==0:
            print(f'[test] batch {idx}/{len(info["eval_loader"])} samples={processed}',flush=True)
    result={'schema_version':1,'method':args.model,'variable':args.variable,'split':args.split,
            'run_tag':meta['run_tag'],'members':args.members,'samples_evaluated':processed,
            'checkpoint':{k:meta[k] for k in ('model_name','variable','train_end_idx',
                          'val_start_idx','test_start_idx','data_end_idx','data_last_date','run_tag')},
            **scores.summary()}
    output=Path(cfg.results_root)/args.model/args.variable/meta['run_tag']/args.split
    output.mkdir(parents=True,exist_ok=True)
    target=output/f'metrics_{args.model}_{args.variable}.json'
    save_json(target,result)
    print(f'[scores] {result["summary"]}',flush=True)
    print(f'[saved] {target}',flush=True)
    return result


def main():
    args=arguments()
    name=f'test_{args.model}_{run_tag(args)}_{args.variable}_{args.split}.txt'
    with tee_output(Path(cfg.results_root).parent/'logs'/name):run(args)

if __name__=='__main__':main()
