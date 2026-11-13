"""Compare DDM-inspired joint SimVP and latent SDE-RNN on val/test.

The latent SDE-RNN is evaluated as an ensemble, the SimVP as a degenerate
(one-member) forecast. No SDE files are read at evaluation/inference time.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from Forecast_nn.config import cfg
from Forecast_nn.evaluation import to_jsonable
from Forecast_nn.generative_utils import prepare_data, EnsembleScores, save_json
from Forecast_nn.logging_utils import tee_output
from Forecast_nn.models.ddm import DDMSimVP, EOFProjector, LatentSDERNN
from Forecast_nn.runtime import configure_runtime


def arguments(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=('simvp_ddm','latent_sde_rnn'),required=True)
    p.add_argument('--variable',choices=('joint',),default='joint')
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--checkpoint',default=None,help='Full best.pth path (recommended)')
    p.add_argument('--members',type=int,default=8,help='Latent ensemble members; ignored for deterministic SimVP')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--downsample',type=int,default=cfg.downsample)
    p.add_argument('--end-date',default=cfg.data_last_date)
    p.add_argument('--val-steps',type=int,default=cfg.val_steps)
    p.add_argument('--test-steps',type=int,default=cfg.test_steps)
    p.add_argument('--seed',type=int,default=cfg.seed)
    p.add_argument('--max-batches',type=int,default=0,help='DEBUG only: incomplete evaluation')
    # When --checkpoint is omitted, auto-identify a unique saved experiment.
    p.add_argument('--run-id',default='')
    return p.parse_args(argv)


def resolve_checkpoint(args):
    if args.checkpoint:return Path(args.checkpoint)
    root=Path(cfg.results_root)/args.model/'joint'
    paths=sorted(root.glob('*/best.pth')) if root.is_dir() else []
    if args.run_id:
        paths=[p for p in paths if p.parent.name.endswith('_'+args.run_id)]
    if len(paths)!=1:
        raise ValueError(f'Expected exactly one checkpoint under {root}; found {len(paths)}. '
                         'Pass --checkpoint /path/to/best.pth (or --run-id to disambiguate).')
    return paths[0]


def run(args):
    if args.members<2 and args.model=='latent_sde_rnn':
        raise ValueError('Use at least 2 ensemble members for probabilistic evaluation.')
    configure_runtime(args.seed,cfg.fast_cuda)
    checkpoint=resolve_checkpoint(args)
    record=torch.load(checkpoint,map_location='cpu',weights_only=False)
    meta=record['metadata']
    if meta['model_name']!=args.model or meta['variable']!='joint':
        raise ValueError('Checkpoint method/variable does not match --model joint')
    info=prepare_data(args,split=args.split)
    for key,current in [('train_end_idx',info['bounds'].train_end),
                        ('val_start_idx',info['bounds'].val_start),
                        ('test_start_idx',info['bounds'].test_start),
                        ('data_end_idx',info['bounds'].data_end)]:
        if int(meta[key]) != current:
            raise ValueError(f'Chronological split differs for {key}: {meta[key]} != {current}')
    for key,current in [('in_len',cfg.in_len),('out_len',cfg.out_len),('downsample',args.downsample)]:
        if int(meta[key])!=int(current):
            raise ValueError(f'Checkpoint setting differs: {key}')
    for key,current in [('mean',info['mean']),('std',info['std']),('psnr_ranges',info['ranges'])]:
        if not np.allclose(meta[key],current,rtol=1e-4,atol=1e-5):
            raise ValueError(f'Train-derived normalization differs from saved checkpoint: {key}')
    dev=info['device'];mask=torch.as_tensor(info['mask'],device=dev,dtype=torch.bool)
    if args.model=='simvp_ddm':
        model=DDMSimVP(cfg.in_len,cfg.out_len,meta['simvp_params']).to(dev)
        model.load_state_dict(record['model_state']);projector=None;members=1
    else:
        projector=EOFProjector.from_state_dict(record['projector_state']).to_torch(dev)
        model=LatentSDERNN(meta['rank'],meta['hidden'],1.).to(dev)
        model.load_state_dict(record['model_state']);members=args.members
    model.eval()
    stats=EnsembleScores(cfg.out_len,info['std'],info['ranges'])
    count=0
    with torch.inference_mode():
        for batch_index,batch in enumerate(info['eval_loader'],1):
            if args.max_batches and batch_index>args.max_batches:break
            x,y=[tensor.to(dev,non_blocking=True) for tensor in batch[:2]]
            if args.model=='simvp_ddm':
                pred,_,_=model(x,mask)
                samples=pred[:,None]
            else:
                z=projector.encode(x)
                torch.manual_seed(args.seed+batch_index)
                if dev.type=='cuda':torch.cuda.manual_seed_all(args.seed+batch_index)
                samples=projector.decode(model.sample(z,horizon=cfg.out_len,members=members))
            stats.update(samples,y,x[:,-1],mask)
            count+=len(x)
            if batch_index==1 or batch_index % 25==0:
                print(f'[eval] batch {batch_index}/{len(info["eval_loader"])} samples={count}',flush=True)
    if not count:raise RuntimeError('No evaluation samples')
    scored=stats.summary()
    # Align aggregated PSNR with the existing deterministic evaluator: compute
    # dB AFTER aggregating squared errors over all horizons, per channel.
    mse_channel=stats.errors.sum(axis=0)/np.maximum(stats.count.sum(axis=0),1)
    psnr_channel=10*np.log10(np.asarray(info['ranges'])**2/np.maximum(mse_channel,1e-30))
    scored['summary']['overall_psnr']=float(psnr_channel.mean())
    for row in scored['per_channel']:
        row['psnr_db']=float(psnr_channel[row['channel']])
    result={'schema_version':1,'method':args.model,'variable':'joint',
            'split':args.split,'members':members,'samples_evaluated':count,
            'run_tag':meta['run_tag'],'checkpoint_path':str(checkpoint),
            'best_val_rmse':record.get('best_val_rmse'),
            'sde_inference_requires_future_coefficients':False,
            'latent_covariance_assumption':meta.get('cov_prior') if projector else None,
            'checkpoint':{key:meta[key] for key in ('model_name','variable','data_last_date',
                'train_end_idx','val_start_idx','test_start_idx','data_end_idx','run_tag')},
            **scored}
    if args.max_batches:
        result['warning']='INCOMPLETE debug evaluation: max-batches was specified'
    outdir=checkpoint.parent/args.split
    path=outdir/f'metrics_{args.model}_joint.json'
    save_json(path,result)
    print(f'[metrics] {result["summary"]}',flush=True)
    print(f'[saved] {path}',flush=True)
    return result


def main(argv=None):
    args=arguments(argv)
    path=Path(cfg.results_root).parent/'logs'/f'test_{args.model}_{args.split}_joint.txt'
    with tee_output(path):run(args)


if __name__=='__main__':main()
