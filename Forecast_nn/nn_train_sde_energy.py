# Recommended single-GPU Windows launch:
# python -m Forecast_nn.nn_train_sde_energy --variable sensible --amp --num-workers 2
import argparse
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Forecast_nn.config import cfg
from Forecast_nn.models.sde_energy import SDEEnergyNet
from Forecast_nn.models.sde_energy_v2 import SDEEnergyV2
from Forecast_nn.sde_energy_dataloader import OceanFieldReader, OceanSequenceDataset, build_statistics, load_full_mask, load_statistics, save_statistics
from Forecast_nn.sde_energy_loss import SDEEnergyLoss
from Forecast_nn.losses_v2 import combined_loss


def configure_runtime(seed, fast_cuda=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        # Fixed-size image tensors benefit from cuDNN autotuning. This intentionally favors speed over exact bitwise reproducibility.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = bool(fast_cuda)
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def init_distributed():
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(local_rank)
    if distributed:
        backend = 'gloo' if os.name == 'nt' else ('nccl' if device.type == 'cuda' else 'gloo')
        torch.distributed.init_process_group(backend=backend)
    return distributed, torch.distributed.get_rank() if distributed else 0, device


def reduce_sum(value, distributed):
    if not distributed:
        return value
    t = torch.tensor(float(value), dtype=torch.float64)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t.item()


def default_data_file(variable):
    return os.path.join(cfg.root_path, 'DATA', 'Fluxes', f'{variable}_grouped_1979-2024.npy')


def gpu_memory_string(device):
    if device.type != 'cuda':
        return 'CPU'
    alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    return f'GPU mem {alloc:.2f}/{reserved:.2f} GB alloc/reserved'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', choices=('sensible', 'latent'), default='sensible')
    parser.add_argument('--data-file', default=None)
    parser.add_argument('--mask-file', default=os.path.join(cfg.root_path, 'DATA', 'mask'))
    parser.add_argument('--stats-file', default=None)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--downsample', type=int, default=2, help='Spatial thinning factor. 2 reproduces the old [::2, ::2] pipeline: 161x181 -> 81x91. Use 1 for full resolution.')
    parser.add_argument('--holdout-steps', type=int, default=365)
    parser.add_argument('--sde-bins', type=int, default=500)
    parser.add_argument('--sde-lambda', type=float, default=1e-4)
    parser.add_argument("--dyn-lambda", type=float, default=0.1)
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--latent-channels', type=int, default=64)
    parser.add_argument('--kan-dim', type=int, default=64)
    parser.add_argument('--rbf-centers', type=int, default=32)
    parser.add_argument('--rbf-gamma', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=cfg.epoch)
    parser.add_argument('--lr', type=float, default=cfg.LR)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=2, help='On Windows try 0, 2 and 4; choose the fastest stable value.')
    parser.add_argument('--log-every', type=int, default=50, help='Print a training progress line every N mini-batches.')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True, help='CUDA automatic mixed precision. Usually faster and uses less VRAM.')
    parser.add_argument('--fast-cuda', action=argparse.BooleanOptionalAction, default=True, help='Enable cuDNN benchmark and TF32 where supported.')
    parser.add_argument('--load-model', action='store_true', default=cfg.LOAD_MODEL)
    parser.add_argument('--rebuild-stats', action='store_true')
    args = parser.parse_args()

    configure_runtime(args.seed, args.fast_cuda)
    distributed, rank, device = init_distributed()
    use_amp = bool(args.amp and device.type == 'cuda')
    data_file = args.data_file or default_data_file(args.variable)

    if rank == 0:
        print('\n' + '=' * 90, flush=True)
        print('SDE-ENERGY TRAINING', flush=True)
        print('=' * 90, flush=True)
        print(f'[runtime] Python={sys.version.split()[0]}, PyTorch={torch.__version__}, CUDA build={torch.version.cuda}', flush=True)
        print(f'[runtime] device={device}, distributed={distributed}, AMP={use_amp}, cuDNN benchmark={torch.backends.cudnn.benchmark}', flush=True)
        if device.type == 'cuda':
            print(f'[runtime] GPU={torch.cuda.get_device_name(device)}, capability={torch.cuda.get_device_capability(device)}', flush=True)
        print(f'[runtime] DataLoader workers={args.num_workers}, batch_size={args.batch_size}', flush=True)

    if rank == 0: print('\n[1/6] Loading mask and inspecting data...', flush=True)
    mask = load_full_mask(mask_file=args.mask_file, downsample=args.downsample)
    reader = OceanFieldReader(data_file, downsample=args.downsample)
    train_end = reader.length - args.holdout_steps if args.holdout_steps > 0 else reader.length
    stats_file = args.stats_file or os.path.join(cfg.root_path, 'DATA', 'Fluxes', f'sde_energy_stats_{args.variable}_ds{args.downsample}_stride_{args.stride}_bins_{args.sde_bins}.npz')
    model_path = args.model_path or os.path.join(cfg.root_path, cfg.work_path, 'save', cfg.dataset, 'SDE-Energy', args.variable, 'models', f'sde_energy_{args.variable}_ds{args.downsample}_days_{cfg.out_len}.pth')
    if rank == 0:
        print(f'  variable={args.variable}', flush=True); print(f'  data={data_file}', flush=True)
        print(f'  source shape={reader.data.shape}, layout={reader.layout}, time steps={reader.length}', flush=True)
        print(f'  spatial downsample={args.downsample}: 161x181 -> {reader.height}x{reader.width}', flush=True)
        print(f'  ocean points={int(mask.sum())}/{mask.size} ({100.0 * mask.mean():.1f}%)', flush=True)
        print(f'  train steps=[0, {train_end}), holdout={reader.length - train_end}', flush=True)

    if rank == 0: print('\n[2/6] Preparing normalization and SDE a/b statistics...', flush=True)
    if rank == 0 and (args.rebuild_stats or not os.path.exists(stats_file)):
        started = time.time()
        save_statistics(stats_file, build_statistics(data_file, mask, 0, train_end, args.sde_bins, args.stride, downsample=args.downsample, verbose=True))
        print(f'  Saved statistics to {stats_file} ({(time.time() - started) / 60:.1f} min)', flush=True)
    elif rank == 0:
        print(f'  Reusing {stats_file}', flush=True)
    if distributed:
        torch.distributed.barrier()
    stats = load_statistics(stats_file)

    if rank == 0: print('\n[3/6] Building dataset and DataLoader...', flush=True)
    train_data = OceanSequenceDataset(data_file, mask, 0, train_end, cfg.in_len, cfg.out_len, stats['mean'], stats['std'], args.stride, downsample=args.downsample)
    sampler = DistributedSampler(train_data, shuffle=True) if distributed else None
    loader_kwargs = dict(batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
        if os.name == 'nt':
            loader_kwargs['multiprocessing_context'] = 'spawn'
    loader = DataLoader(train_data, **loader_kwargs)
    if rank == 0:
        print(f'  sequences={len(train_data)}, batches/epoch={len(loader)}, in_len={cfg.in_len}, out_len={cfg.out_len}, stride={args.stride}', flush=True)
        if os.name == 'nt' and args.num_workers > 0: print('  Windows DataLoader: spawn workers + per-worker lazy mmap reopen enabled.', flush=True)

    if rank == 0: print('\n[4/6] Building model, loss and optimizer...', flush=True)
    model_kwargs = dict(output_length=cfg.out_len, in_channels=train_data.channels, out_channels=train_data.channels, base_channels=args.base_channels,
                        latent_channels=args.latent_channels, kan_dim=args.kan_dim, rbf_centers=args.rbf_centers, rbf_gamma=args.rbf_gamma, dropout=args.dropout)
    model = SDEEnergyNet(mask=mask, **model_kwargs).to(device)
    model = SDEEnergyV2(mask=mask, **model_kwargs).to(device)
    if args.load_model and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint)
        if rank == 0: print(f'  Loaded checkpoint: {model_path}', flush=True)
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' and os.name != 'nt' else None)
    criterion = SDEEnergyLoss(mask, stats['bin_edges'], stats['a'], stats['b'], args.sde_lambda).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if rank == 0:
        raw_model = model.module if distributed else model
        params = sum(p.numel() for p in raw_model.parameters())
        trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f'  parameters={params:,}, trainable={trainable:,}', flush=True)
        print(f'  lr={args.lr:g}, weight_decay={args.weight_decay:g}, sde_lambda={args.sde_lambda:g}, dyn_lambda={args.dyn_lambda:g}, sde_bins={args.sde_bins}', flush=True)
        print(f'  KAN: dim={args.kan_dim}, centers={args.rbf_centers}, gamma={args.rbf_gamma:g}; dropout={args.dropout:g}', flush=True)
        print(f'  {gpu_memory_string(device)}', flush=True)

    if rank == 0: print('\n[5/6] Training...', flush=True)
    losses = []
    training_started = time.time()
    for epoch in range(1, args.epochs + 1):
        if sampler is not None: sampler.set_epoch(epoch)
        model.train()
        total_sum = mse_sum = sde_sum = 0.0; batches = 0
        total = 0
        mse_total = 0
        dyn_total = 0
        epoch_started = time.time(); interval_started = time.time(); interval_samples = 0
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        for batch_idx, (x, y) in enumerate(loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred, _ = model(x)
                # loss, mse, sde = criterion(y, pred, x[:, -1])
                loss, parts = combined_loss(
                    pred,
                    y,
                    x[:, -1],
                    mask[:, None],
                    args.dyn_lambda,
                )
                total += loss.item()
                mse_total += parts["mse"].item()
                dyn_total += parts["dynamic"].item()

            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            # sde_sum += sde.item(); mse_sum += mse.item();
            total_sum += loss.item();  batches += 1; interval_samples += x.size(0)
            if rank == 0 and (batch_idx == 1 or batch_idx % args.log_every == 0 or batch_idx == len(loader)):
                now = time.time(); dt = max(now - interval_started, 1e-9); speed = interval_samples / dt
                done_ratio = batch_idx / max(len(loader), 1); epoch_elapsed = now - epoch_started; eta = epoch_elapsed * (1.0 - done_ratio) / max(done_ratio, 1e-9)
                print(f'  epoch {epoch:03d}/{args.epochs} | batch {batch_idx:05d}/{len(loader)} ({100*done_ratio:5.1f}%) | '
                      f'loss={total_sum/batches:.6f} mse={mse_sum/batches:.6f} sde={sde_sum/batches:.4f} | '
                      f'{speed:.2f} samples/s | epoch ETA={eta/60:.1f} min | {gpu_memory_string(device)}', flush=True)
                interval_started = now; interval_samples = 0
        total_sum = reduce_sum(total_sum, distributed); mse_sum = reduce_sum(mse_sum, distributed); sde_sum = reduce_sum(sde_sum, distributed); batches = reduce_sum(batches, distributed)
        epoch_loss = total_sum / batches; losses.append(epoch_loss)
        if rank == 0:
            epoch_seconds = time.time() - epoch_started
            total_elapsed = time.time() - training_started
            remaining = (total_elapsed / epoch) * (args.epochs - epoch)
            peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else 0.0
            print(f'  -> epoch {epoch:03d} finished: loss={epoch_loss:.6f}, mse={mse_sum/batches:.6f}, sde={sde_sum/batches:.6f}, '
                  f'time={epoch_seconds/60:.1f} min, peak_GPU={peak:.2f} GB, training ETA={remaining/60:.1f} min\n', flush=True)

    if rank == 0:
        print('[6/6] Saving checkpoint and loss curve...', flush=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        raw_model = model.module if distributed else model
        torch.save({'model_state': raw_model.state_dict(), 'model_kwargs': model_kwargs, 'mean': stats['mean'], 'std': stats['std'], 'variable': args.variable,
                    'data_file': data_file, 'train_end_idx': train_end, 'total_time_steps': reader.length, 'stride': args.stride, 'in_len': cfg.in_len,
                    'out_len': cfg.out_len, 'sde_bins': args.sde_bins, 'sde_lambda': args.sde_lambda, 'dyn_lambda': args.dyn_lambda, 'lr': args.lr, 'weight_decay': args.weight_decay,
                    'batch_size': args.batch_size, 'epochs': args.epochs, 'amp': use_amp, 'downsample': args.downsample, 'grid_shape': tuple(mask.shape)}, model_path)
        loss_dir = os.path.join(cfg.root_path, 'Losses'); os.makedirs(loss_dir, exist_ok=True)
        loss_path = os.path.join(loss_dir, f'loss_SDE-Energy_{args.variable}_ds{args.downsample}.npy'); np.save(loss_path, np.asarray(losses, dtype=np.float32))
        print(f'  model: {model_path}', flush=True); print(f'  losses: {loss_path}', flush=True)
        print(f'Total training time: {(time.time() - training_started) / 60:.1f} min', flush=True)
        print('=' * 90 + '\n', flush=True)
    if distributed: torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
