import numpy as np

import datetime
import os

import torch
from config import cfg
import time
import shutil
from tqdm import tqdm
# import cv2
from thop import profile
from SSIM import get_SSIM
from earlystopping import EarlyStopping
from tensorboardX import SummaryWriter


IN_LEN = cfg.in_len
OUT_LEN = cfg.out_len
if 'kth' in cfg.dataset:
    EVAL_LEN = cfg.eval_len
gpu_nums = cfg.gpu_nums
decimals = cfg.metrics_decimals



def sum_batch(data):
    return data.sum(axis=1)


def as_type(data):
    return data.astype(cfg.data_type)


class Evaluation(object):
    def __init__(self, seq_len, use_central=False, thresholds=None, max_val=(1.0, 1.0, 1.0)):
        self._total_batch_num = 0
        self._thresholds = cfg.HKO.THRESHOLDS if thresholds is None else thresholds
        self._ssim = np.zeros((OUT_LEN, 3), dtype=cfg.data_type)
        self._mae = np.zeros((OUT_LEN,), dtype=cfg.data_type)
        self._mse = np.zeros((OUT_LEN,), dtype=cfg.data_type)
        self._seq_len = seq_len
        self._use_central = use_central
        self._max_val = max_val

    def clear_all(self):
        self._total_batch_num = 0
        self._ssim[:] = 0
        self._mse[:] = 0
        self._mae[:] = 0

    def update(self, gt, pred):
        batch_size = gt.shape[1]
        assert gt.shape[0] == self._seq_len
        assert gt.shape == pred.shape

        if self._use_central:
            # Crop the central regions for evaluation
            central_region = cfg.HKO.CENTRAL_REGION
            pred = pred[:, :, :, central_region[1]:central_region[3], central_region[0]:central_region[2]]
            gt = gt[:, :, :, central_region[1]:central_region[3], central_region[0]:central_region[2]]

        self._total_batch_num += batch_size
        ssim = get_SSIM(prediction=pred, truth=gt)

        # # S*B*1*H*W
        mse = np.square(pred - gt).sum(axis=(2, 3, 4))
        mae = np.abs(pred - gt).sum(axis=(2, 3, 4))

        self._ssim += sum_batch(ssim)
        self._mse += sum_batch(mse)
        self._mae += sum_batch(mae)


    def get_metrics(self):
        ssim = self._ssim / self._total_batch_num
        mse = self._mse / self._total_batch_num
        mae = self._mae / self._total_batch_num
        l_all = [ssim, mse, mae]
        return l_all


def normalize_data_cuda(batch, min_vals, max_vals):
    #print(type(batch))
    # print(f'min_vals = {min_vals}')
    # print(f'max_vals = {max_vals}')
    batch = torch.permute(batch, (1, 0, 2, 3, 4))  # S x B x C x H x W
    for channel in range(batch.shape[2]):
        batch[:, :, channel] = (batch[:, :, channel] - min_vals[channel]) / max_vals[channel]
    return batch.cuda()


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
