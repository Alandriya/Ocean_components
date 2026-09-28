# Example:
# python -m Forecast_nn.nn_test_sde_energy --variable sensible --batch-size 2 --amp
import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from Forecast_nn.config import cfg
from Forecast_nn.evaluation_utils import MetricAccumulator, build_run_tag, save_metrics_xlsx, save_plot_samples
from Forecast_nn.models.sde_energy import SDEEnergyNet
from Forecast_nn.sde_energy_dataloader import OceanFieldReader, OceanSequenceDataset, load_full_mask


def default_data_file(variable):
    return os.path.join(cfg.root_path, 'DATA', 'Fluxes', f'{variable}_grouped_1979-2024.npy')


def default_model_path(variable, downsample=2):
    return os.path.join(cfg.root_path, cfg.work_path, 'save', cfg.dataset, 'SDE-Energy', variable, 'models', f'sde_energy_{variable}_ds{downsample}_days_{cfg.out_len}.pth')


def parse_sample_ids(text):
    return sorted({int(v.strip()) for v in text.split(',') if v.strip()}) if text else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', choices=('sensible', 'latent'), default='sensible')
    parser.add_argument('--data-file', default=None)
    parser.add_argument('--mask-file', default=os.path.join(cfg.root_path, 'DATA', 'mask'))
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--test-start-index', type=int, default=None)
    parser.add_argument('--downsample', type=int, default=2, help='Used to select the default checkpoint; the checkpoint value is authoritative during testing.')
    parser.add_argument('--test-end-index', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output-dir', default=None, help='Directory for XLSX metrics and plotting samples.')
    parser.add_argument('--plot-samples', default='0', help='Comma-separated zero-based test sample IDs to save for the plotting script, e.g. 0,20,100.')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(args.amp and device.type == 'cuda')
    data_file = args.data_file or default_data_file(args.variable)
    model_path = args.model_path or default_model_path(args.variable, args.downsample)
    print('\n' + '=' * 90, flush=True); print('SDE-ENERGY TEST + BASELINE COMPARISON', flush=True); print('=' * 90, flush=True)
    print(f'[runtime] device={device}, AMP={use_amp}, workers={args.num_workers}, batch_size={args.batch_size}', flush=True)
    if device.type == 'cuda': print(f'[runtime] GPU={torch.cuda.get_device_name(device)}', flush=True)

    print('\n[1/5] Loading checkpoint, mask and test data...', flush=True)
    checkpoint = torch.load(model_path, map_location='cpu')
    downsample = int(checkpoint.get('downsample', args.downsample))
    if downsample != args.downsample:
        print(f'  checkpoint uses downsample={downsample}; overriding CLI value {args.downsample}.', flush=True)
    mask = load_full_mask(mask_file=args.mask_file, downsample=downsample)
    model_kwargs = checkpoint['model_kwargs']
    mean, std = np.asarray(checkpoint['mean'], dtype=np.float32), np.asarray(checkpoint['std'], dtype=np.float32)
    stride = int(checkpoint.get('stride', 1)); in_len = int(checkpoint.get('in_len', cfg.in_len)); out_len = int(checkpoint.get('out_len', cfg.out_len))
    reader = OceanFieldReader(data_file, downsample=downsample)
    test_start = args.test_start_index if args.test_start_index is not None else int(checkpoint.get('train_end_idx', max(0, reader.length - 365)))
    test_end = args.test_end_index if args.test_end_index is not None else reader.length
    test_data = OceanSequenceDataset(data_file, mask, test_start, test_end, in_len, out_len, mean, std, stride, downsample=downsample)
    loader_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
        if os.name == 'nt':
            loader_kwargs['multiprocessing_context'] = 'spawn'
    loader = DataLoader(test_data, **loader_kwargs)
    print(f'  variable={args.variable}, data={data_file}', flush=True); print(f'  downsample={downsample}, grid={reader.height}x{reader.width}', flush=True); print(f'  test indices=[{test_start}, {test_end}), sequences={len(test_data)}, batches={len(loader)}', flush=True)
    if os.name == 'nt' and args.num_workers > 0: print('  Windows DataLoader: spawn workers + per-worker lazy mmap reopen enabled.', flush=True)

    print('\n[2/5] Loading model...', flush=True)
    model = SDEEnergyNet(mask=mask, **model_kwargs); model.load_state_dict(checkpoint['model_state']); model = model.to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    print(f'  checkpoint={model_path}', flush=True); print(f'  parameters={params:,}', flush=True)
    print('  baselines: historical mean = mean of input frames at each ocean grid cell; persistence = copy last input frame.', flush=True)

    print('\n[3/5] Running NN and baselines...', flush=True)
    methods = {'sde_energy': MetricAccumulator(out_len), 'historical_mean': MetricAccumulator(out_len), 'persistence': MetricAccumulator(out_len)}
    scale = std.reshape(1, 1, -1, 1, 1).astype(np.float32)
    sample_ids = parse_sample_ids(args.plot_samples); saved = {k: [] for k in ('sample_id', 'forecast_start_index', 'input_last', 'truth', 'sde_energy', 'historical_mean', 'persistence')}
    processed = 0; started = time.time()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                nn_pred, _ = model(x)
            mean_pred = x.mean(dim=1, keepdim=True).repeat(1, out_len, 1, 1, 1)
            persistence_pred = x[:, -1:].repeat(1, out_len, 1, 1, 1)
            arrays = {'sde_energy': nn_pred.float().cpu().numpy(), 'historical_mean': mean_pred.float().cpu().numpy(), 'persistence': persistence_pred.float().cpu().numpy()}
            truth_np = y.float().cpu().numpy()
            for name, pred_np in arrays.items():
                methods[name].update((pred_np - truth_np) * scale, mask, processed)
            for local_idx in range(x.size(0)):
                global_id = processed + local_idx
                if global_id in sample_ids:
                    x_np = x[local_idx].float().cpu().numpy() * std[:, None, None] + mean[:, None, None]
                    y_np = truth_np[local_idx] * std[:, None, None] + mean[:, None, None]
                    saved['sample_id'].append(global_id); saved['forecast_start_index'].append(test_start + global_id + in_len * stride); saved['input_last'].append(x_np[-1]); saved['truth'].append(y_np)
                    for name in ('sde_energy', 'historical_mean', 'persistence'):
                        saved[name].append(arrays[name][local_idx] * std[:, None, None] + mean[:, None, None])
            processed += x.size(0)
            if batch_idx == 1 or batch_idx % args.log_every == 0 or batch_idx == len(loader):
                elapsed = max(time.time() - started, 1e-9); rate = processed / elapsed; eta = (len(test_data) - processed) / max(rate, 1e-9)
                print(f'  batch {batch_idx:04d}/{len(loader)} | samples {processed}/{len(test_data)} ({100*processed/max(len(test_data),1):5.1f}%) | {rate:.2f} samples/s | ETA={eta/60:.1f} min', flush=True)

    print('\n[4/5] Error statistics...', flush=True)
    summaries = {name: acc.summary() for name, acc in methods.items()}
    for name, metrics in summaries.items():
        print(f'  {name:16s}: MSE={metrics["overall_mse"]:.8g}, MAE={metrics["overall_mae"]:.8g}, RMSE={metrics["overall_rmse"]:.8g}', flush=True)
    nn_rmse = summaries['sde_energy']['overall_rmse']
    for baseline in ('historical_mean', 'persistence'):
        b = summaries[baseline]['overall_rmse']
        print(f'  NN RMSE improvement vs {baseline}: {(b - nn_rmse) / b * 100.0:.2f}%', flush=True)

    print('\n[5/5] Saving XLSX files and selected forecasts...', flush=True)
    run_tag = build_run_tag(args.variable, checkpoint)
    output_dir = args.output_dir or os.path.join(cfg.root_path, 'Forecast/Results', 'SDE-Energy', args.variable, run_tag)
    os.makedirs(output_dir, exist_ok=True)
    for name, acc in methods.items():
        xlsx_path = os.path.join(output_dir, f'metrics_{name}_{run_tag}.xlsx')
        save_metrics_xlsx(xlsx_path, name, args.variable, run_tag, summaries[name], checkpoint, acc.sample_rows)
        print(f'  {xlsx_path}', flush=True)
    if saved['sample_id']:
        plot_path = os.path.join(output_dir, f'forecast_samples_{run_tag}.npz')
        save_plot_samples(plot_path, saved, {'run_tag': run_tag, 'variable': args.variable, 'mask_file': args.mask_file, 'stride': stride, 'downsample': downsample, 'base_date': '1979-01-01'})
        print(f'  plotting data: {plot_path}', flush=True)
        print(f'  draw all three forecast figures with: python -m Forecast_nn.plot_sde_energy_comparison --input "{plot_path}" --sample-index 0 --method all', flush=True)
    print('=' * 90 + '\n', flush=True)


if __name__ == '__main__':
    main()
