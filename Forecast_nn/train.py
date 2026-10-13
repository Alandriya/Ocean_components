"""Common training entry point for every registered Forecast_nn model.

Examples from repository root:
    python -m Forecast_nn.train --model simvp
    python -m Forecast_nn.train --model simvp_sde --sde-weight 1e-4
    python -m Forecast_nn.train --model fno_sde --sde-weight 5e-5
    python -m Forecast_nn.train --resume --epochs 15

For --resume, --epochs means additional epochs for this invocation.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Forecast_nn.config import cfg
from Forecast_nn.data import OceanFieldReader, OceanSequenceDataset, load_mask, load_or_create_normalization
from Forecast_nn.evaluation import build_run_tag, append_sde_run_tag
from Forecast_nn.losses import forecast_loss
from Forecast_nn.models import build_model, available_models, model_uses_sde_conditioning
from Forecast_nn.plotting.training_curve import plot_training_curve
from Forecast_nn.runtime import configure_runtime, make_loader_kwargs, gpu_memory_string, load_checkpoint
from Forecast_nn.sde import SDECoefficientLookup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=available_models(), default=cfg.model_name)
    p.add_argument('--variable', choices=tuple(cfg.data_files), default=cfg.variable)
    p.add_argument('--epochs', type=int, default=cfg.epochs, help='Epochs in this invocation; with --resume they are additional.')
    p.add_argument('--batch-size', type=int, default=cfg.batch_size)
    p.add_argument('--lr', type=float, default=cfg.learning_rate)
    p.add_argument('--downsample', type=int, default=cfg.downsample)
    p.add_argument('--num-workers', type=int, default=cfg.num_workers)
    p.add_argument('--dynamic-weight', type=float, default=cfg.dynamic_weight)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=cfg.amp)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--rebuild-stats', action='store_true')

    # None means: use cfg.sde_weight for *_sde models, otherwise 0.
    p.add_argument('--sde-weight', type=float, default=None)
    p.add_argument('--sde-file', default=None, help='Override cfg.sde_sources[variable]["file"].')
    p.add_argument('--sde-b-kind', choices=('variance', 'diffusion'), default=cfg.sde_b_kind)
    p.add_argument('--sde-dt', type=float, default=cfg.sde_dt)
    p.add_argument('--sde-eps', type=float, default=cfg.sde_eps)
    return p.parse_args()


def _sde_source(variable, override=None):
    source = dict(cfg.sde_sources[variable])
    if override:
        source = {'file': Path(override)}
    return source


def save_checkpoint(path, model, optimizer, scaler, epoch, history, metadata):
    payload = dict(metadata)
    payload.update({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'epoch': int(epoch),
        'epochs': int(epoch),
        'history': {k: list(v) for k, v in history.items()},
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main():
    args = parse_args()
    configure_runtime(cfg.seed, cfg.fast_cuda)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(args.amp and device.type == 'cuda')

    uses_sde_conditioning = model_uses_sde_conditioning(args.model)
    sde_weight = float(cfg.sde_weight if (args.sde_weight is None and uses_sde_conditioning) else (args.sde_weight or 0.0))
    sde_enabled = uses_sde_conditioning or sde_weight > 0.0

    data_file = Path(cfg.data_files[args.variable])
    if not data_file.exists():
        raise FileNotFoundError(f'Data file not found: {data_file}')
    if not Path(cfg.mask_file).exists():
        raise FileNotFoundError(f'Mask file not found: {cfg.mask_file}')

    print('\n' + '=' * 96, flush=True)
    print('FORECAST_NN TRAINING', flush=True)
    print('=' * 96, flush=True)
    print(f'[runtime] Python={sys.version.split()[0]}  PyTorch={torch.__version__}  CUDA build={torch.version.cuda}', flush=True)
    print(f'[runtime] device={device}  AMP={use_amp}  workers={args.num_workers}  batch={args.batch_size}', flush=True)
    if device.type == 'cuda':
        print(f'[runtime] GPU={torch.cuda.get_device_name(device)}', flush=True)

    print('\n[1/7] Data and split...', flush=True)
    mask = load_mask(cfg.mask_file, args.downsample)
    reader = OceanFieldReader(data_file, args.downsample)
    train_end = reader.length - int(cfg.test_steps)
    if train_end <= (cfg.in_len + cfg.out_len) * cfg.stride:
        raise ValueError('Training period is too short after reserving test_steps.')
    print(f'  model={args.model}  variable={args.variable}', flush=True)
    print(f'  source={data_file}', flush=True)
    print(f'  source shape={reader.data.shape}, layout={reader.layout}, time_steps={reader.length}', flush=True)
    print(f'  grid=161x181 -> {reader.height}x{reader.width} (downsample={args.downsample})', flush=True)
    print(f'  ocean points={int(mask.sum())}/{mask.size} ({100*mask.mean():.1f}%)', flush=True)
    print(f'  training frames=[0,{train_end}), held-out test frames=[{train_end},{reader.length})', flush=True)

    print('\n[2/7] Normalization...', flush=True)
    mean, std, stats_path = load_or_create_normalization(
        data_file, mask, args.variable, args.downsample, train_end, cfg.stats_root, args.rebuild_stats
    )
    print(f'  stats={stats_path}', flush=True)
    print(f'  mean={mean.tolist()}  std={std.tolist()}', flush=True)

    print('\n[3/7] Ito-SDE information...', flush=True)
    sde_lookup = None
    sde_source = None
    if sde_enabled:
        sde_source = _sde_source(args.variable, args.sde_file)
        sde_lookup = SDECoefficientLookup.from_source(
            sde_source,
            b_kind=args.sde_b_kind,
            feature_clip=cfg.sde_feature_clip,
        )
        print(f'  source={sde_source}', flush=True)
        print(f'  table={sde_lookup.describe()}', flush=True)
        print(f'  conditioning={uses_sde_conditioning}  sde_weight={sde_weight:g}  dt={args.sde_dt:g}', flush=True)
    else:
        print('  disabled for this run.', flush=True)

    print('\n[4/7] Dataset and DataLoader...', flush=True)
    dataset = OceanSequenceDataset(
        data_file, mask, 0, train_end, cfg.in_len, cfg.out_len,
        mean, std, cfg.stride, args.downsample
    )
    loader = DataLoader(dataset, **make_loader_kwargs(args.batch_size, args.num_workers, device, shuffle=True))
    print(f'  sequences={len(dataset):,}  batches/epoch={len(loader):,}', flush=True)
    print(f'  input={cfg.in_len} days  output={cfg.out_len} days  stride={cfg.stride}', flush=True)
    if os.name == 'nt' and args.num_workers > 0:
        print('  Windows-safe lazy mmap workers are enabled.', flush=True)

    print('\n[5/7] Model and optimizer...', flush=True)
    model_params = dict(cfg.model_params[args.model])
    model_kwargs = dict(model_params)
    model = build_model(args.model, cfg.in_len, cfg.out_len, dataset.channels, model_params).to(device)
    run_tag = build_run_tag(args.model, args.variable, args.downsample, cfg.in_len, cfg.out_len, cfg.stride, model_params)
    if sde_enabled:
        run_tag = append_sde_run_tag(run_tag, sde_weight, args.sde_b_kind)
    run_dir = Path(cfg.results_root) / args.model / args.variable / run_tag
    checkpoint_path = run_dir / 'checkpoint.pth'
    history_path = run_dir / 'training_history.npz'
    curve_path = run_dir / 'training_curve.png'

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history = {
        'epoch': [], 'total_loss': [], 'mse': [], 'dynamic': [],
        'sde': [], 'sde_drift': [], 'lr': [], 'epoch_time_s': [],
    }
    previous_epoch = 0

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Cannot resume: checkpoint not found: {checkpoint_path}')
        checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            for group in optimizer.param_groups:
                group['lr'] = args.lr
        if 'scaler_state' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state'])
        previous_epoch = int(checkpoint.get('epoch', 0))
        old_history = checkpoint.get('history', {})
        for key in history:
            history[key] = list(old_history.get(key, []))
        print(f'  resumed from {checkpoint_path} at epoch {previous_epoch}', flush=True)

    params = sum(p.numel() for p in model.parameters())
    print(f'  run_tag={run_tag}', flush=True)
    print(f'  parameters={params:,}', flush=True)
    print(f'  model_params={model_kwargs}', flush=True)
    print(
        f'  lr={args.lr:g} weight_decay={cfg.weight_decay:g} dynamic_weight={args.dynamic_weight:g} '
        f'sde_weight={sde_weight:g}', flush=True
    )
    print(f'  output={run_dir}', flush=True)

    mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)[None, None]
    metadata = {
        'model_name': args.model,
        'model_kwargs': model_kwargs,
        'variable': args.variable,
        'data_file': str(data_file),
        'mean': mean,
        'std': std,
        'train_end_idx': train_end,
        'total_time_steps': reader.length,
        'downsample': args.downsample,
        'grid_shape': tuple(mask.shape),
        'in_len': cfg.in_len,
        'out_len': cfg.out_len,
        'stride': cfg.stride,
        'learning_rate': args.lr,
        'weight_decay': cfg.weight_decay,
        'dynamic_weight': args.dynamic_weight,
        'run_tag': run_tag,
        'sde_enabled': bool(sde_enabled),
        'sde_conditioning': bool(uses_sde_conditioning),
        'sde_weight': float(sde_weight),
        'sde_b_kind': args.sde_b_kind,
        'sde_dt': float(args.sde_dt),
        'sde_eps': float(args.sde_eps),
        'sde_source': sde_source,
    }

    print('\n[6/7] Training...', flush=True)
    training_started = time.time()
    first_epoch = previous_epoch + 1
    final_epoch = previous_epoch + args.epochs

    for epoch in range(first_epoch, final_epoch + 1):
        model.train()
        epoch_started = time.time()
        total_sum = mse_sum = dyn_sum = sde_sum = drift_sum = 0.0
        batches = 0
        interval_started = time.time(); interval_samples = 0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        for batch_idx, (x, y) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            sde_features = None
            if uses_sde_conditioning:
                # Keep coefficient lookup in float32; the network itself can still use AMP.
                sde_features = sde_lookup.conditioning_features(x.float(), mask_tensor)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if uses_sde_conditioning:
                    pred = model(x, mask_tensor, sde_features=sde_features)
                else:
                    pred = model(x, mask_tensor)

            # Loss in float32, including the SDE term.
            loss, parts = forecast_loss(
                pred, y, x[:, -1], mask_tensor,
                dynamic_weight=args.dynamic_weight,
                sde_lookup=sde_lookup if sde_weight > 0 else None,
                sde_weight=sde_weight,
                sde_dt=args.sde_dt,
                sde_eps=args.sde_eps,
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_sum += loss.item()
            mse_sum += parts['mse'].item()
            dyn_sum += parts['dynamic'].item()
            sde_sum += parts['sde'].item()
            drift_sum += parts['sde_drift'].item()
            batches += 1
            interval_samples += x.size(0)

            if batch_idx == 1 or batch_idx % cfg.log_every == 0 or batch_idx == len(loader):
                now = time.time(); dt = max(now - interval_started, 1e-9)
                speed = interval_samples / dt
                ratio = batch_idx / max(len(loader), 1)
                eta = (now - epoch_started) * (1.0 - ratio) / max(ratio, 1e-9)
                print(
                    f'  epoch {epoch:03d}/{final_epoch} | batch {batch_idx:05d}/{len(loader)} ({100*ratio:5.1f}%) | '
                    f'loss={total_sum/batches:.6f} mse={mse_sum/batches:.6f} dyn={dyn_sum/batches:.6f} '
                    f'sde={sde_sum/batches:.6f} drift={drift_sum/batches:.6f} | '
                    f'{speed:.2f} samples/s | epoch ETA={eta/60:.1f} min | {gpu_memory_string(device)}',
                    flush=True
                )
                interval_started = now; interval_samples = 0

        epoch_seconds = time.time() - epoch_started
        history['epoch'].append(epoch)
        history['total_loss'].append(total_sum / batches)
        history['mse'].append(mse_sum / batches)
        history['dynamic'].append(dyn_sum / batches)
        history['sde'].append(sde_sum / batches)
        history['sde_drift'].append(drift_sum / batches)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time_s'].append(epoch_seconds)

        peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0.0
        elapsed = time.time() - training_started
        remaining = elapsed / max(epoch - first_epoch + 1, 1) * (final_epoch - epoch)
        print(
            f'  -> epoch {epoch:03d}: loss={history["total_loss"][-1]:.6f}, mse={history["mse"][-1]:.6f}, '
            f'dyn={history["dynamic"][-1]:.6f}, sde={history["sde"][-1]:.6f}, '
            f'time={epoch_seconds/60:.1f} min, peak_GPU={peak:.2f} GB, run ETA={remaining/60:.1f} min\n', flush=True
        )

        if epoch % cfg.checkpoint_every == 0 or epoch == final_epoch:
            save_checkpoint(checkpoint_path, model, optimizer, scaler, epoch, history, metadata)
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez(history_path, **{k: np.asarray(v) for k, v in history.items()})

    print('[7/7] Final outputs...', flush=True)
    save_checkpoint(checkpoint_path, model, optimizer, scaler, final_epoch, history, metadata)
    np.savez(history_path, **{k: np.asarray(v) for k, v in history.items()})
    plot_training_curve(history_path, curve_path, title=f'{args.model}: {args.variable}, ds={args.downsample}')
    print(f'  checkpoint: {checkpoint_path}', flush=True)
    print(f'  history:    {history_path}', flush=True)
    print(f'  curve:      {curve_path}', flush=True)
    print(f'  total time: {(time.time()-training_started)/60:.1f} min', flush=True)
    print('=' * 96 + '\n', flush=True)


if __name__ == '__main__':
    main()
