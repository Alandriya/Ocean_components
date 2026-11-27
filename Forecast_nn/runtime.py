"""Shared neural forecasting runtime setup, data loaders, checkpoint loading and experiment console logging."""

# --- Runtime ---
# Small runtime helpers shared by train.py and test.py.

import os
import random
import numpy as np
import torch


def configure_runtime(seed=2025, fast_cuda=True):
    """Seed random generators and configure optional CUDA acceleration settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = bool(fast_cuda)
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def make_loader_kwargs(batch_size, num_workers, device, shuffle):
    """Build PyTorch DataLoader settings for the device and worker count."""
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=device.type == 'cuda')
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
        if os.name == 'nt':
            kwargs['multiprocessing_context'] = 'spawn'
    return kwargs


def gpu_memory_string(device):
    """Report allocated and reserved CUDA memory, or identify the CPU device."""
    if device.type != 'cuda':
        return 'CPU'
    alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    return f'GPU mem {alloc:.2f}/{reserved:.2f} GB alloc/reserved'


def load_checkpoint(path, map_location='cpu'):
    """Load our own training checkpoint across old and new PyTorch versions.

    PyTorch 2.6+ defaults torch.load to weights_only=True, while these checkpoints
    intentionally contain optimizer/history metadata in addition to tensors.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# --- Logging Utils ---
# Duplicate CLI stdout/stderr to readable experiment logs without muting the console.
#
# Logs live in Forecast/logs (next to Forecast_nn). Files are appended on
# resume/repeated runs; each invocation starts with an explicit timestamp marker.

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys
import traceback



class _TeeStream:
    """Write text to the user's original stream AND a shared UTF-8 log file."""

    def __init__(self, original_stream, log_file):
        """Initialize Tee Stream with ``original_stream``, ``log_file``."""
        self._original = original_stream
        self._log = log_file

    def write(self, message):
        """Write the supplied data to the configured destination."""
        n = self._original.write(message)
        self._log.write(message)
        # Make the log usable while the job is running, including on a cluster.
        self._log.flush()
        return n

    def flush(self):
        """Flush buffered output to the underlying streams."""
        self._original.flush()
        self._log.flush()

    def __getattr__(self, name):
        """Delegate an unresolved attribute to the wrapped object."""
        return getattr(self._original, name)


@contextmanager
def tee_output(log_path):
    """Capture Python-level stdout, stderr and uncaught exceptions in a file."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', buffering=1) as output:
        stdout = _TeeStream(sys.stdout, output)
        stderr = _TeeStream(sys.stderr, output)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f'\n{"=" * 72}', flush=True)
            print(f'[log] Started: {datetime.now().astimezone().isoformat(timespec="seconds")}', flush=True)
            print(f'[log] Command: {" ".join(sys.argv)}', flush=True)
            print(f'[log] File: {log_path}', flush=True)
            print(f'{"=" * 72}', flush=True)
            try:
                yield
            except BaseException:
                # Otherwise the interpreter prints the traceback after streams
                # have been restored, so it would be absent from the log.
                traceback.print_exc()
                raise
            finally:
                print(f'[log] Finished: {datetime.now().astimezone().isoformat(timespec="seconds")}', flush=True)


def run_tag_from_args(args):
    """Mirror train/test short experiment tags before either pipeline starts."""
    from Forecast_nn.config import cfg
    from Forecast_nn.evaluation import build_run_tag, append_sde_run_tag, add_run_id
    from Forecast_nn.models import model_uses_sde_conditioning

    params = dict(cfg.model_params[args.model])
    tunable = ('base_channels', 'temporal_tokens', 'attention_depth', 'heads', 'window_size')
    overrides = {key: getattr(args, key) for key in tunable if getattr(args, key, None) is not None}
    if overrides and args.model != 'swin_gru':
        raise ValueError('Heavy-model size flags are only supported for --model swin_gru.')
    params.update(overrides)
    conditioned = model_uses_sde_conditioning(args.model)
    sde_weight = float(cfg.sde_weight if (args.sde_weight is None and conditioned)
                       else (args.sde_weight or 0.0))
    run_tag = build_run_tag(args.model, args.variable, args.downsample,
                            cfg.in_len, cfg.out_len, cfg.stride, params)
    if conditioned or sde_weight > 0:
        run_tag = append_sde_run_tag(run_tag, sde_weight,
                                     args.sde_b_kind, args.sde_loss_mode)
    return add_run_id(run_tag, args.run_id)


def log_file_path(stage, model, run_tag, variable='sensible', split='test'):
    """Make short, collision-resistant names in Forecast/logs.

    Sensible (the default) follows the requested exact example; latent and joint
    get an explicit suffix so their logs never overwrite sensible runs.
    """
    if stage not in ('train', 'test'):
        raise ValueError("stage must be 'train' or 'test'")
    stem = f'{stage}_{model}_{run_tag}'
    if variable != 'sensible':
        stem += f'_{variable}'
    if stage == 'test' and split != 'test':
        stem += f'_{split}'
    return Path(__file__).resolve().parent.parent / 'Forecast' / 'logs' / f'{stem}.txt'
