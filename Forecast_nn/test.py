"""Common evaluation entry point for every registered Forecast_nn model.

Run from the repository root:
    python -m Forecast_nn.test

The script compares the selected neural network against:
- historical mean of the input sequence;
- persistence (copy the last observed map).
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Forecast_nn.config import cfg
from Forecast_nn.data import OceanFieldReader, OceanSequenceDataset, load_mask
from Forecast_nn.evaluation import MetricAccumulator, build_run_tag, save_metrics_xlsx, save_plot_samples
from Forecast_nn.models import build_model, available_models
from Forecast_nn.runtime import configure_runtime, make_loader_kwargs, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=available_models(), default=cfg.model_name)
    p.add_argument('--variable', choices=tuple(cfg.data_files), default=cfg.variable)
    p.add_argument('--downsample', type=int, default=cfg.downsample)
    p.add_argument('--batch-size', type=int, default=cfg.batch_size)
    p.add_argument('--num-workers', type=int, default=cfg.num_workers)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=cfg.amp)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--plot-samples', default=','.join(map(str, cfg.plot_samples)))
    return p.parse_args()


def parse_ids(text):
    return sorted({int(v.strip()) for v in text.split(',') if v.strip()}) if text else []


def main():
    args = parse_args()
    configure_runtime(cfg.seed, cfg.fast_cuda)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(args.amp and device.type == 'cuda')

    model_params = dict(cfg.model_params[args.model])
    default_run_tag = build_run_tag(args.model, args.variable, args.downsample, cfg.in_len, cfg.out_len, cfg.stride, model_params)
    default_run_dir = Path(cfg.results_root) / args.model / args.variable / default_run_tag
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_run_dir / 'checkpoint.pth'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
    model_name = checkpoint['model_name']
    variable = checkpoint['variable']
    downsample = int(checkpoint['downsample'])
    in_len = int(checkpoint['in_len'])
    out_len = int(checkpoint['out_len'])
    stride = int(checkpoint['stride'])
    model_kwargs = dict(checkpoint['model_kwargs'])
    mean = np.asarray(checkpoint['mean'], dtype=np.float32)
    std = np.asarray(checkpoint['std'], dtype=np.float32)
    train_end = int(checkpoint['train_end_idx'])
    run_tag = checkpoint.get('run_tag') or build_run_tag(model_name, variable, downsample, in_len, out_len, stride, model_kwargs)

    data_file = Path(checkpoint.get('data_file', cfg.data_files[variable]))
    mask = load_mask(cfg.mask_file, downsample)
    reader = OceanFieldReader(data_file, downsample)

    # Include the preceding input history so sample 0 predicts the first held-out test frame.
    test_input_start = max(0, train_end - in_len * stride)
    test_data = OceanSequenceDataset(
        data_file, mask, test_input_start, reader.length, in_len, out_len,
        mean, std, stride, downsample
    )
    loader = DataLoader(test_data, **make_loader_kwargs(args.batch_size, args.num_workers, device, shuffle=False))

    print('\n' + '=' * 92, flush=True)
    print('FORECAST_NN TEST + BASELINES', flush=True)
    print('=' * 92, flush=True)
    print(f'[runtime] device={device} AMP={use_amp} workers={args.num_workers} batch={args.batch_size}', flush=True)
    print(f'[model] {model_name}  variable={variable}  checkpoint={checkpoint_path}', flush=True)
    print(f'[data] test target starts at frame {train_end}; sequences={len(test_data):,}; grid={mask.shape[0]}x{mask.shape[1]}', flush=True)

    model = build_model(model_name, in_len, out_len, reader.channels, model_kwargs)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device).eval()
    mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)[None, None]
    params = sum(p.numel() for p in model.parameters())
    print(f'[model] parameters={params:,}', flush=True)

    methods = {
        model_name: MetricAccumulator(out_len),
        'historical_mean': MetricAccumulator(out_len),
        'persistence': MetricAccumulator(out_len),
    }
    sample_ids = parse_ids(args.plot_samples)
    saved = {k: [] for k in ('sample_id', 'forecast_start_index', 'input_last', 'truth', model_name, 'historical_mean', 'persistence')}
    scale = std.reshape(1, 1, -1, 1, 1)
    shift = mean.reshape(1, 1, -1, 1, 1)
    processed = 0
    started = time.time()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                nn_pred = model(x, mask_tensor)

            mean_pred = x.mean(dim=1, keepdim=True).repeat(1, out_len, 1, 1, 1)
            persistence_pred = x[:, -1:].repeat(1, out_len, 1, 1, 1)

            x_np = x.float().cpu().numpy()
            truth_n = y.float().cpu().numpy()
            arrays_n = {
                model_name: nn_pred.float().cpu().numpy(),
                'historical_mean': mean_pred.float().cpu().numpy(),
                'persistence': persistence_pred.float().cpu().numpy(),
            }
            truth = truth_n * scale + shift
            last_input = x_np[:, -1] * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
            arrays = {name: pred * scale + shift for name, pred in arrays_n.items()}

            for name, pred in arrays.items():
                methods[name].update(pred, truth, last_input, mask, processed)

            for local_idx in range(x.size(0)):
                sample_id = processed + local_idx
                if sample_id in sample_ids:
                    saved['sample_id'].append(sample_id)
                    saved['forecast_start_index'].append(train_end + sample_id)
                    saved['input_last'].append(last_input[local_idx])
                    saved['truth'].append(truth[local_idx])
                    for name in (model_name, 'historical_mean', 'persistence'):
                        saved[name].append(arrays[name][local_idx])

            processed += x.size(0)
            if batch_idx == 1 or batch_idx % cfg.log_every == 0 or batch_idx == len(loader):
                elapsed = max(time.time() - started, 1e-9)
                rate = processed / elapsed
                eta = (len(test_data) - processed) / max(rate, 1e-9)
                print(f'  batch {batch_idx:04d}/{len(loader)} | {processed}/{len(test_data)} samples | {rate:.2f} samples/s | ETA={eta/60:.1f} min', flush=True)

    print('\nMetrics:', flush=True)
    summaries = {name: acc.summary() for name, acc in methods.items()}
    for name, m in summaries.items():
        print(
            f'  {name:18s} RMSE={m["overall_rmse"]:.6g} MAE={m["overall_mae"]:.6g} '
            f'delta_RMSE={m["delta_rmse"]:.6g} delta_corr={m["delta_correlation"]:.4f} '
            f'|delta| pred/true={m["pred_mean_abs_change"]:.4g}/{m["true_mean_abs_change"]:.4g}',
            flush=True
        )

    output_dir = Path(cfg.results_root) / model_name / variable / run_tag / 'test'
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, acc in methods.items():
        path = output_dir / f'metrics_{name}_{run_tag}.xlsx'
        save_metrics_xlsx(path, name, variable, run_tag, summaries[name], checkpoint, acc.sample_rows)
        print(f'  saved: {path}', flush=True)

    if saved['sample_id']:
        plot_file = output_dir / f'forecast_samples_{run_tag}.npz'
        save_plot_samples(plot_file, saved, {
            'run_tag': run_tag,
            'model_name': model_name,
            'methods': [model_name, 'historical_mean', 'persistence'],
            'variable': variable,
            'stride': stride,
            'downsample': downsample,
            'base_date': cfg.base_date,
            'mask_file': str(cfg.mask_file),
        })
        print(f'  plotting data: {plot_file}', flush=True)
        print(f'  plot command: python -m Forecast_nn.plotting.forecast_comparison --input "{plot_file}" --method all', flush=True)
    print('=' * 92 + '\n', flush=True)


if __name__ == '__main__':
    main()
