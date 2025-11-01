import os
import random

import numpy as np
import torch

# from Forecasting.SSIM import get_SSIM
from Forecasting.config import cfg
from Forecasting.loader import load_mask

IN_LEN = cfg.in_len
OUT_LEN = cfg.out_len
if 'kth' in cfg.dataset:
    EVAL_LEN = cfg.eval_len
gpu_nums = cfg.gpu_nums
decimals = cfg.metrics_decimals

# fix random seed
def fix_random(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def sum_batch(data):
    return data.sum(axis=0)


def as_type(data):
    return data.astype(cfg.data_type)


def normalize_data(batch, min_vals, max_vals):
    # print(max_vals)
    # print(min_vals)
    # print('\n\n')
    # for i in range(3, 9):
    #     print(i)
    #     print(torch.max(batch[:, :, i]))
    #     print(torch.min(batch[:, :, i]))

    for channel in range(cfg.channels):
        batch[:, :cfg.in_len + cfg.out_len, channel] = (batch[:, :cfg.in_len + cfg.out_len, channel] -
                                                        min_vals[channel]) / (max_vals[channel] - min_vals[channel])
        # # normalize A
        # batch[:, :cfg.in_len + cfg.out_len, channel + cfg.channels] = (batch[:, :cfg.in_len + cfg.out_len, channel + cfg.channels] -
        #                                                     min_vals[channel]) / (max_vals[channel] - min_vals[channel])
        #
        # # normalize B
        # batch[:, :cfg.in_len + cfg.out_len, channel + cfg.channels*2] = (batch[:, :cfg.in_len + cfg.out_len, channel + cfg.channels*2] -
        #                                                 (min_vals[channel])) / ((max_vals[channel] - min_vals[channel]))

        #
        # # normalize eigenvalues
        # batch[:, :cfg.in_len + cfg.out_len, channel + 9 + channel*2] = (batch[:, :cfg.in_len + cfg.out_len, channel*2 + 9] -
        #                                                 min_vals[channel]) / (max_vals[channel] - min_vals[channel])
        # batch[:, :cfg.in_len + cfg.out_len, channel + 10 + channel*2] = (batch[:, :cfg.in_len + cfg.out_len, channel*2 + 10] -
        #                                                 min_vals[channel]) / (max_vals[channel] - min_vals[channel])

    # return batch.cuda()
    # print('\n\n')
    # for i in range(3, 18):
    #     print(i)
    #     print(torch.max(batch[:, :, i]))
    #     print(torch.min(batch[:, :, i]))
    # raise ValueError
    return batch


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= gpu_nums
    return rt


# is main process ?
def is_master_proc(gpu_nums=gpu_nums):
    return torch.distributed.get_rank() % gpu_nums == 0


def nan_to_num(metrics):
    for i in range(len(metrics)):
        metrics[i] = np.nan_to_num(metrics[i])
    return metrics
