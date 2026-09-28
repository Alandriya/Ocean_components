"""Small runtime helpers shared by train.py and test.py."""
import os
import random
import numpy as np
import torch


def configure_runtime(seed=2025, fast_cuda=True):
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
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=device.type == 'cuda')
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
        if os.name == 'nt':
            kwargs['multiprocessing_context'] = 'spawn'
    return kwargs


def gpu_memory_string(device):
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
