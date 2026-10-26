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

from Forecast_nn.joint_sde import JointSDEReader, JointSDEDataset, joint_drift_loss
from Forecast_nn.config import cfg
from Forecast_nn.data import (make_field_reader, resolve_flux_source, source_exists, OceanSequenceDataset, load_mask,
                              load_or_create_normalization, load_or_create_psnr_ranges)
from Forecast_nn.evaluation import build_run_tag, append_sde_run_tag, add_run_id, OnlineErrorMetrics
from Forecast_nn.losses import forecast_loss
from Forecast_nn.logging_utils import tee_output, log_file_path, run_tag_from_args
from Forecast_nn.splits import resolve_splits
from Forecast_nn.models import build_model, available_models, model_uses_sde_conditioning
from Forecast_nn.plotting.training_curve import plot_training_curve
from Forecast_nn.runtime import configure_runtime, make_loader_kwargs, gpu_memory_string, load_checkpoint
from Forecast_nn.sde import (
    SDEFieldReader, SDEOceanSequenceDataset,
    load_or_create_sde_field_normalization, source_to_serializable,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=available_models(), default=cfg.model_name)
    # Only used by the new heavy hybrid; explicit flags also identify run folders.
    p.add_argument('--base-channels', type=int, default=None)
    p.add_argument('--temporal-tokens', type=int, default=None)
    p.add_argument('--attention-depth', type=int, default=None)
    p.add_argument('--heads', type=int, default=None)
    p.add_argument('--window-size', type=int, default=None)
    p.add_argument('--variable', choices=(*tuple(cfg.data_files), 'joint'), default=cfg.variable)
    p.add_argument('--epochs', type=int, default=cfg.epochs, help='Epochs in this invocation; with --resume they are additional.')
    p.add_argument('--batch-size', type=int, default=cfg.batch_size)
    p.add_argument('--lr', type=float, default=cfg.learning_rate)
    p.add_argument('--downsample', type=int, default=cfg.downsample)
    p.add_argument('--num-workers', type=int, default=cfg.num_workers)
    p.add_argument('--dynamic-weight', type=float, default=cfg.dynamic_weight)
    p.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps (effective batch=batch-size * grad-accum).')
    p.add_argument('--grad-clip', type=float, default=1.0, help='Maximum gradient norm, 0 disables clipping.')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=cfg.amp)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--run-id', default='', help='Optional short suffix to separate runs with different training or split settings.')
    p.add_argument('--rebuild-stats', action='store_true')
    p.add_argument('--end-date', type=str, default=cfg.data_last_date,
                   help='Inclusive last day, e.g. 2023-12-31. Must exist in the input series.')
    p.add_argument('--val-steps', type=int, default=cfg.val_steps)
    p.add_argument('--test-steps', type=int, default=cfg.test_steps)
    p.add_argument('--patience', type=int, default=cfg.early_stopping_patience,
                   help='Consecutive validation epochs without improved RMSE. 0 disables stopping.')
    p.add_argument('--min-delta', type=float, default=cfg.early_stopping_min_delta,
                   help='Minimum validation RMSE decrease for a new best checkpoint.')

    # None means: use cfg.sde_weight for *_sde models, otherwise 0.
    p.add_argument('--sde-weight', type=float, default=None)
    p.add_argument('--joint-a-file', default=None, help='Full 2-channel historical drift A array for joint loss.')
    p.add_argument('--joint-b-file', default=None, help='Full 4-channel historical B matrix, row-major B11,B12,B21,B22.')
    p.add_argument('--joint-b-layout', choices=('row_major', 'column_major'), default='row_major')
    p.add_argument('--joint-cov-floor', type=float, default=0.1, help='Ridge added to normalized covariance for stable joint drift loss.')
    p.add_argument('--sde-a-file', default=None, help='Override the configured A coefficient NPY file.')
    p.add_argument('--sde-b-file', default=None, help='Override the configured B coefficient NPY file.')
    p.add_argument('--sde-time-offset', type=int, default=None, help='Flux index represented by coefficient element 0; default comes from config (1).')
    p.add_argument('--sde-b-kind', choices=('variance', 'diffusion'), default=cfg.sde_b_kind)
    p.add_argument('--sde-dt', type=float, default=cfg.sde_dt)
    p.add_argument('--sde-eps', type=float, default=cfg.sde_eps)
    p.add_argument('--sde-loss-mode', choices=('first_step', 'last_observed_all'), default=cfg.sde_loss_mode)
    return p.parse_args()


def _sde_source(variable, a_override=None, b_override=None, time_offset=None):
    source = dict(cfg.sde_sources[variable])
    if a_override:
        source['a_file'] = Path(a_override)
    if b_override:
        source['b_file'] = Path(b_override)
    if time_offset is not None:
        source['time_offset'] = int(time_offset)
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


def _run(args):
    if args.grad_accum < 1 or args.grad_clip < 0:
        raise ValueError('Require grad-accum >= 1 and grad-clip >= 0.')
    if args.epochs < 1 or args.patience < 0 or args.min_delta < 0:
        raise ValueError('Require epochs >= 1, patience >= 0 and min_delta >= 0.')
    configure_runtime(cfg.seed, cfg.fast_cuda)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(args.amp and device.type == 'cuda')

    uses_sde_conditioning = model_uses_sde_conditioning(args.model)
    sde_weight = float(cfg.sde_weight if (args.sde_weight is None and uses_sde_conditioning) else (args.sde_weight or 0.0))
    sde_enabled = uses_sde_conditioning or sde_weight > 0.0

    joint_loss_only = args.variable == 'joint' and sde_weight > 0 and not uses_sde_conditioning
    if args.variable == 'joint' and uses_sde_conditioning:
        raise ValueError('Joint feature-conditioned *_sde architecture is not implemented; use --model simvp with --joint-a-file/--joint-b-file and --sde-weight.')
    if joint_loss_only and (not args.joint_a_file or not args.joint_b_file):
        raise ValueError('Joint SDE loss requires BOTH --joint-a-file and --joint-b-file (full A2/B4 maps).')
    if args.joint_cov_floor <= 0:
        raise ValueError('--joint-cov-floor must be > 0')
    # The joint SDE loss has its own coefficient reader; the old SDE path remains unmodified.
    sde_enabled = sde_enabled and not joint_loss_only
    data_file = resolve_flux_source(args.variable, cfg.data_files)
    if not source_exists(data_file):
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

    print('\n[1/7] Data and chronological split...', flush=True)
    mask = load_mask(cfg.mask_file, args.downsample)
    reader = make_field_reader(data_file, args.downsample)
    split = resolve_splits(reader.length, cfg.base_date, args.end_date,
                           args.val_steps, args.test_steps, cfg.in_len, cfg.out_len, cfg.stride)
    train_end, val_start, test_start, data_end = (
        split.train_end, split.val_start, split.test_start, split.data_end
    )
    print(f'  model={args.model}  variable={args.variable}', flush=True)
    print(f'  source={data_file}', flush=True)
    print(f'  source shape={reader.data.shape}, time_steps={reader.length}', flush=True)
    print(f'  effective end date={args.end_date}, final target index={data_end-1}', flush=True)
    print(f'  grid=161x181 -> {reader.height}x{reader.width} (downsample={args.downsample})', flush=True)
    print(f'  ocean points={int(mask.sum())}/{mask.size} ({100*mask.mean():.1f}%)', flush=True)
    print(f'  TRAIN targets=[0,{train_end}) | VAL targets=[{val_start},{test_start}) | '
          f'TEST targets=[{test_start},{data_end})', flush=True)

    print('\n[2/7] Train-only normalization and PSNR data ranges...', flush=True)
    mean, std, stats_path = load_or_create_normalization(
        data_file, mask, args.variable, args.downsample, train_end, cfg.stats_root, args.rebuild_stats
    )
    psnr_ranges, psnr_path = load_or_create_psnr_ranges(
        data_file, mask, args.variable, args.downsample, train_end, cfg.stats_root,
        cfg.psnr_percentiles, cfg.psnr_max_sample_frames, args.rebuild_stats
    )
    print(f'  normalization={stats_path}; mean={mean.tolist()}; std={std.tolist()}', flush=True)
    print(f'  PSNR ranges={psnr_ranges.tolist()} from {psnr_path}', flush=True)

    print('\n[3/7] Ito-SDE information...', flush=True)
    sde_reader = None
    joint_reader = None
    sde_source = None
    sde_feature_stats = None
    sde_stats_path = None
    if joint_loss_only:
        joint_reader = JointSDEReader(args.joint_a_file, args.joint_b_file, args.downsample,
                                      mask.shape, std, b_layout=args.joint_b_layout)
        sde_source = {'a_file': args.joint_a_file, 'b_file': args.joint_b_file,
                      'time_offset': joint_reader.time_offset, 'b_layout': args.joint_b_layout}
        if joint_reader.a.length + joint_reader.time_offset < data_end:
            raise ValueError('Joint SDE coefficients do not cover the final observed field')
        print(f'  joint loss-only: {sde_source}; covariance_floor={args.joint_cov_floor}', flush=True)
    elif sde_enabled:
        sde_source = _sde_source(
            args.variable, args.sde_a_file, args.sde_b_file, args.sde_time_offset
        )
        sde_feature_stats, sde_stats_path = load_or_create_sde_field_normalization(
            sde_source,
            mask,
            args.variable,
            args.downsample,
            train_end,
            cfg.stats_root,
            b_kind=args.sde_b_kind,
            feature_clip=cfg.sde_feature_clip,
            rebuild=args.rebuild_stats,
        )
        sde_reader = SDEFieldReader(
            sde_source,
            downsample=args.downsample,
            b_kind=args.sde_b_kind,
            feature_clip=cfg.sde_feature_clip,
            feature_stats=sde_feature_stats,
            state_std=std,
        )
        print(f'  source={source_to_serializable(sde_source)}', flush=True)
        print(f'  maps={sde_reader.describe()}', flush=True)
        if sde_reader.last_flux_index < data_end - 1:
            raise ValueError(f'SDE coefficients end at flux index {sde_reader.last_flux_index}, '
                             f'but data end index is {data_end-1}. No future SDE maps may be used.')
        print(f'  feature_stats={sde_stats_path}', flush=True)
        print(
            f'  alignment: coefficient element 0 -> flux t={sde_reader.time_offset}; '
            f'A_t/B_t are attached to observed X_t and future maps are never read',
            flush=True,
        )
        print(
            f'  conditioning={uses_sde_conditioning}  sde_weight={sde_weight:g}  '
            f'b_kind={args.sde_b_kind} loss_mode={args.sde_loss_mode} dt={args.sde_dt:g}',
            flush=True,
        )
    else:
        print('  disabled for this run.', flush=True)

    print('\n[4/7] Train and validation DataLoaders...', flush=True)
    dataset_cls = SDEOceanSequenceDataset if sde_enabled else OceanSequenceDataset
    dataset_kwargs = {'sde_reader': sde_reader} if sde_enabled else {}
    dataset = dataset_cls(
        data_file, mask, *split.dataset_bounds('train', cfg.in_len, cfg.stride),
        cfg.in_len, cfg.out_len, mean, std, cfg.stride, args.downsample, **dataset_kwargs
    )
    val_data = dataset_cls(
        data_file, mask, *split.dataset_bounds('val', cfg.in_len, cfg.stride),
        cfg.in_len, cfg.out_len, mean, std, cfg.stride, args.downsample, **dataset_kwargs
    )
    if joint_loss_only:
        dataset = JointSDEDataset(dataset, joint_reader)
        val_data = JointSDEDataset(val_data, joint_reader)
    loader = DataLoader(dataset, **make_loader_kwargs(args.batch_size, args.num_workers, device, shuffle=True))
    val_loader = DataLoader(val_data, **make_loader_kwargs(args.batch_size, args.num_workers, device, shuffle=False))
    print(f'  train samples={len(dataset):,}, validation samples={len(val_data):,}', flush=True)
    print(f'  input={cfg.in_len}, output={cfg.out_len}, stride={cfg.stride}', flush=True)
    if os.name == 'nt' and args.num_workers > 0:
        print('  Windows-safe lazy mmap workers are enabled.', flush=True)

    print('\n[5/7] Model and optimizer...', flush=True)
    model_params = dict(cfg.model_params[args.model])
    overrides = {name: getattr(args, name) for name in ('base_channels', 'temporal_tokens', 'attention_depth', 'heads', 'window_size')
                 if getattr(args, name) is not None}
    if overrides and args.model != 'swin_gru':
        raise ValueError('Heavy-model size flags are only supported for --model swin_gru.')
    model_params.update(overrides)
    model_kwargs = dict(model_params)
    model = build_model(args.model, cfg.in_len, cfg.out_len, dataset.channels, model_params).to(device)
    run_tag = build_run_tag(args.model, args.variable, args.downsample, cfg.in_len, cfg.out_len, cfg.stride, model_params)
    if sde_enabled or joint_loss_only:
        run_tag = append_sde_run_tag(run_tag, sde_weight, args.sde_b_kind, args.sde_loss_mode)
    run_tag = add_run_id(run_tag, args.run_id)
    run_dir = Path(cfg.results_root) / args.model / args.variable / run_tag
    last_path = run_dir / 'last.pth'
    best_path = run_dir / 'best.pth'
    checkpoint_path = run_dir / 'checkpoint.pth'  # last epoch, backward-compatible name
    history_path = run_dir / 'training_history.npz'
    curve_path = run_dir / 'training_curve.png'

    # Chronological split is deliberately omitted from the short folder name.
    # Refuse to overwrite a different split with the same identifier.
    if not args.resume:
        previous_path = last_path if last_path.exists() else best_path
        if previous_path.exists():
            previous = load_checkpoint(previous_path, map_location='cpu')
            expected_bounds = (data_end, val_start, test_start)
            actual_bounds = tuple(previous.get(key) for key in (
                'data_end_idx', 'val_start_idx', 'test_start_idx'
            ))
            if actual_bounds != expected_bounds:
                raise ValueError(
                    f'Existing experiment {previous_path} uses split {actual_bounds}, '
                    f'but requested split is {expected_bounds}. '
                    'Use --run-id <short-name> to avoid overwriting that experiment.'
                )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history = {
        'epoch': [], 'total_loss': [], 'mse': [], 'dynamic': [],
        'sde': [], 'sde_drift': [], 'lr': [], 'epoch_time_s': [],
        'train_rmse': [], 'train_psnr': [], 'train_psnr_horizons': [], 'val_loss': [],
        'val_rmse': [], 'val_psnr': [], 'val_psnr_horizons': [],
    }
    previous_epoch = 0
    best_val_rmse = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    if args.resume:
        resume_path = last_path if last_path.exists() else checkpoint_path
        if not resume_path.exists():
            raise FileNotFoundError(f'Cannot resume: checkpoint not found: {resume_path}')
        checkpoint = load_checkpoint(resume_path, map_location='cpu')
        if checkpoint.get('data_end_idx') != data_end or checkpoint.get('val_start_idx') != val_start:
            raise ValueError('Resume checkpoint uses different chronological split; refusing to mix experiments.')
        model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            for group in optimizer.param_groups:
                group['lr'] = args.lr
        if 'scaler_state' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state'])
        previous_epoch = int(checkpoint.get('epoch', 0))
        best_val_rmse = float(checkpoint.get('best_val_rmse', float('inf')))
        best_epoch = int(checkpoint.get('best_epoch', 0))
        epochs_without_improvement = int(checkpoint.get('epochs_without_improvement', 0))
        old_history = checkpoint.get('history', {})
        for key in history:
            history[key] = list(old_history.get(key, []))
        print(f'  resumed from {resume_path} at epoch {previous_epoch}', flush=True)

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
        'data_file': data_file,
        'mean': mean,
        'std': std,
        'train_end_idx': train_end,
        'val_start_idx': val_start,
        'test_start_idx': test_start,
        'data_end_idx': data_end,
        'data_last_date': args.end_date,
        'total_time_steps': reader.length,
        'psnr_ranges': psnr_ranges,
        'psnr_percentiles': cfg.psnr_percentiles,
        'psnr_stats_path': str(psnr_path),

        'downsample': args.downsample,
        'grid_shape': tuple(mask.shape),
        'in_len': cfg.in_len,
        'out_len': cfg.out_len,
        'stride': cfg.stride,
        'learning_rate': args.lr,
        'weight_decay': cfg.weight_decay,
        'dynamic_weight': args.dynamic_weight,
        'grad_accum': args.grad_accum,
        'grad_clip': args.grad_clip,
        'run_tag': run_tag,
        'sde_enabled': bool(sde_enabled or joint_loss_only),
        'sde_joint_loss_only': bool(joint_loss_only),
        'joint_cov_floor': args.joint_cov_floor,
        'sde_conditioning': bool(uses_sde_conditioning),
        'sde_weight': float(sde_weight),
        'sde_b_kind': args.sde_b_kind,
        'sde_loss_mode': args.sde_loss_mode,
        'sde_time_offset': int(sde_source.get('time_offset', 1)) if sde_source else None,
        'sde_feature_stats': sde_feature_stats,
        'sde_dt': float(args.sde_dt),
        'sde_eps': float(args.sde_eps),
        'sde_source': source_to_serializable(sde_source) if sde_source else None,
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
        train_metrics = OnlineErrorMetrics(cfg.out_len, std, psnr_ranges)
        optimizer.zero_grad(set_to_none=True)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        for batch_idx, batch in enumerate(loader, 1):
            if joint_loss_only:
                x, y, joint_a, joint_b = batch
                joint_a = joint_a.to(device, non_blocking=True)
                joint_b = joint_b.to(device, non_blocking=True)
                sde_features = sde_last_coefficients = None
            elif sde_enabled:
                x, y, sde_features, sde_last_coefficients = batch
                sde_features = sde_features.to(device, non_blocking=True)
                sde_last_coefficients = sde_last_coefficients.to(device, non_blocking=True)
            else:
                x, y = batch
                sde_features = None
                sde_last_coefficients = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                if uses_sde_conditioning:
                    pred = model(x, mask_tensor, sde_features=sde_features)
                else:
                    pred = model(x, mask_tensor)

            # Loss in float32, including the SDE term.
            loss, parts = forecast_loss(
                pred, y, x[:, -1], mask_tensor,
                dynamic_weight=args.dynamic_weight,
                sde_last_coefficients=sde_last_coefficients if sde_weight > 0 else None,
                sde_weight=sde_weight,
                sde_b_kind=args.sde_b_kind,
                sde_loss_mode=args.sde_loss_mode,
                sde_dt=args.sde_dt,
                sde_eps=args.sde_eps,
            )

            if joint_loss_only:
                jloss = joint_drift_loss(pred, x[:, -1], joint_a, joint_b, mask_tensor,
                                         covariance_floor=args.joint_cov_floor, dt=args.sde_dt)
                loss = loss + sde_weight * jloss
                parts['sde'] = jloss.detach()
                parts['sde_drift'] = jloss.detach()

            # Divide by actual microbatches in this accumulation group, including
            # its possibly smaller last group.
            group_first = ((batch_idx - 1) // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(loader) - group_first)
            scaler.scale(loss / group_size).backward()
            if batch_idx % args.grad_accum == 0 or batch_idx == len(loader):
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_metrics.update(pred, y, mask_tensor)

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

        train_score = train_metrics.summary()
        model.eval()
        val_metrics = OnlineErrorMetrics(cfg.out_len, std, psnr_ranges)
        val_loss_sum = 0.0
        val_samples = 0
        with torch.inference_mode():
            for val_batch in val_loader:
                if joint_loss_only:
                    vx, vy, vjoint_a, vjoint_b = val_batch
                    vjoint_a = vjoint_a.to(device, non_blocking=True)
                    vjoint_b = vjoint_b.to(device, non_blocking=True)
                    vsde_features = vsde_last = None
                elif sde_enabled:
                    vx, vy, vsde_features, vsde_last = val_batch
                    vsde_features = vsde_features.to(device, non_blocking=True)
                    vsde_last = vsde_last.to(device, non_blocking=True)
                else:
                    vx, vy = val_batch
                    vsde_features = vsde_last = None
                vx = vx.to(device, non_blocking=True)
                vy = vy.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    vpred = (model(vx, mask_tensor, sde_features=vsde_features)
                             if uses_sde_conditioning else model(vx, mask_tensor))
                vloss, _ = forecast_loss(
                    vpred, vy, vx[:, -1], mask_tensor,
                    dynamic_weight=args.dynamic_weight,
                    sde_last_coefficients=vsde_last if sde_weight > 0 else None,
                    sde_weight=sde_weight, sde_b_kind=args.sde_b_kind,
                    sde_loss_mode=args.sde_loss_mode,
                    sde_dt=args.sde_dt, sde_eps=args.sde_eps,
                )
                if joint_loss_only:
                    vloss = vloss + sde_weight * joint_drift_loss(
                        vpred, vx[:, -1], vjoint_a, vjoint_b, mask_tensor,
                        covariance_floor=args.joint_cov_floor, dt=args.sde_dt)
                val_metrics.update(vpred, vy, mask_tensor)
                val_loss_sum += float(vloss) * vx.size(0)
                val_samples += vx.size(0)
        val_score = val_metrics.summary()
        if not np.isfinite(val_score['overall_rmse']):
            raise FloatingPointError('Validation RMSE is non-finite; refusing to save an invalid best checkpoint.')
        val_loss_value = val_loss_sum / max(val_samples, 1)

        epoch_seconds = time.time() - epoch_started
        history['epoch'].append(epoch)
        history['total_loss'].append(total_sum / batches)
        history['mse'].append(mse_sum / batches)
        history['dynamic'].append(dyn_sum / batches)
        history['sde'].append(sde_sum / batches)
        history['sde_drift'].append(drift_sum / batches)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time_s'].append(epoch_seconds)
        history['train_rmse'].append(train_score['overall_rmse'])
        history['train_psnr'].append(train_score['overall_psnr'])
        history['train_psnr_horizons'].append(train_score['psnr'])
        history['val_loss'].append(val_loss_value)
        history['val_rmse'].append(val_score['overall_rmse'])
        history['val_psnr'].append(val_score['overall_psnr'])
        history['val_psnr_horizons'].append(val_score['psnr'])

        peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0.0
        elapsed = time.time() - training_started
        remaining = elapsed / max(epoch - first_epoch + 1, 1) * (final_epoch - epoch)
        print(
            f'  -> epoch {epoch:03d}: loss={history["total_loss"][-1]:.6f}, mse={history["mse"][-1]:.6f}, '
            f'dyn={history["dynamic"][-1]:.6f}, sde={history["sde"][-1]:.6f}, '
            f'train_RMSE={train_score["overall_rmse"]:.5g} train_PSNR={train_score["overall_psnr"]:.3f} dB | '
            f'val_loss={val_loss_value:.5g} val_RMSE={val_score["overall_rmse"]:.5g} '
            f'val_PSNR={val_score["overall_psnr"]:.3f} dB | '
            f'time={epoch_seconds/60:.1f} min, peak_GPU={peak:.2f} GB, run ETA={remaining/60:.1f} min\n', flush=True
        )

        # Rank checkpoints on physical validation RMSE, not the training loss.
        improved = val_score['overall_rmse'] < best_val_rmse - args.min_delta
        if improved:
            best_val_rmse = val_score['overall_rmse']
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint_metadata = dict(metadata, best_val_rmse=best_val_rmse,
                                   best_epoch=best_epoch,
                                   epochs_without_improvement=epochs_without_improvement)
        if improved:
            save_checkpoint(best_path, model, optimizer, scaler, epoch, history, checkpoint_metadata)
            print(f'  -> NEW BEST: epoch={epoch}, validation RMSE={best_val_rmse:.6g}', flush=True)
        # Save last epoch (both names, for backwards compatibility).
        if epoch % cfg.checkpoint_every == 0 or epoch == final_epoch or (
            args.patience > 0 and epochs_without_improvement >= args.patience
        ):
            save_checkpoint(last_path, model, optimizer, scaler, epoch, history, checkpoint_metadata)
            save_checkpoint(checkpoint_path, model, optimizer, scaler, epoch, history, checkpoint_metadata)
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez(history_path, **{k: np.asarray(v) for k, v in history.items()})
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f'  -> Early stopping: {epochs_without_improvement} epochs without '
                  f'validation RMSE improvement; best epoch={best_epoch}', flush=True)
            break

    print('[7/7] Final outputs...', flush=True)
    final_epoch_reached = int(history['epoch'][-1])
    final_metadata = dict(metadata, best_val_rmse=best_val_rmse, best_epoch=best_epoch,
                          epochs_without_improvement=epochs_without_improvement)
    save_checkpoint(last_path, model, optimizer, scaler, final_epoch_reached, history, final_metadata)
    save_checkpoint(checkpoint_path, model, optimizer, scaler, final_epoch_reached, history, final_metadata)
    np.savez(history_path, **{k: np.asarray(v) for k, v in history.items()})
    plot_training_curve(history_path, curve_path, title=f'{args.model}: {args.variable}, ds={args.downsample}')
    print(f'  best checkpoint: {best_path} (epoch {best_epoch}, validation RMSE={best_val_rmse:.6g})', flush=True)
    print(f'  last checkpoint: {last_path}', flush=True)
    print(f'  compatibility checkpoint: {checkpoint_path}', flush=True)
    print(f'  history:    {history_path}', flush=True)
    print(f'  curve:      {curve_path} (plus *_psnr.png)', flush=True)
    print(f'  total time: {(time.time()-training_started)/60:.1f} min', flush=True)
    print('=' * 96 + '\n', flush=True)


def main():
    """Run with real-time, append-only stdout/stderr duplication to Forecast_nn/logs."""
    args = parse_args()
    log_path = log_file_path("train", args.model, run_tag_from_args(args),
                             variable=args.variable)
    with tee_output(log_path):
        _run(args)


if __name__ == '__main__':
    main()
